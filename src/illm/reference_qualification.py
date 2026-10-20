"""Portable V0.1 evidence sealing, with no tensor runtime or promotion authority.

Sealing checks the live interpreter and source. Reading checks embedded facts,
not the author's now possibly absent environment paths. Checksums are integrity
evidence, not signatures or proof against an author who fabricates every input.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import hashlib
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEventKind as K
from illm.model.spec import ModelError, SupportedModelSpec
from illm.model.loader import build_weight_manifest, inspect_safetensors
from illm.qualification import parse_junit
from illm.sampling import SamplingSpec, SamplingState, commit_sample, philox4x32_10, prepare_sample


FORMAL_PROFILE_RUNTIMES = {
    "tiny-reference-cpu-v1": "py312-win-amd64-cpu",
    "clean-room-release-v1": "py312-linux-x86_64-cpu",
    "real-model-single-gpu-v1": "py312-linux-x86_64-cu126",
}
PROFILES = set(FORMAL_PROFILE_RUNTIMES)
SOURCE_FIELDS = {"project_root_digest", "bootstrap_digest", "base_lock_digest"}
RUNTIME_LOCKS = {
    "py312-win-amd64-cpu": "5149ad41ff2e0f261fbec8fe11dcefe2bed90ab8431b1c5339b8e08b59320a34",
    "py312-linux-x86_64-cpu": "e0e40645e6e54f053b08587c030331f665e5cf648143f846b284960e09f4d799",
    "py312-linux-x86_64-cu126": "f190bf575dd92c56a2526feabd458144ffab19b8d7e85b644a76277a9cfcc56c",
}
SAMPLER_SHA256 = "584ee565e2afa0d8fb3b7845102d1dab5933300f87e70f64acf525404b00b49a"
PROFILE_FIELDS = {"profile_id", "result", "required_for_functional", "environment_manifest_digest", "evidence_digest", "reason"}
INDEX_FIELDS = {"schema_version", "profile_id", "os", "arch", "python_version", "source_identity", "runtime_profile_id", "runtime_lock_digest", "files", "evidence_digest", "index_digest"}
CAPTURE_FILES = {"capture-manifest.json", "checkpoint.json", "events.jsonl", "state-digests.jsonl", "checksums.json", "COMPLETE"}
TINY_FILES = {"config.json", "tokenizer.json", "supported-model-spec.json", "weight-load-manifest.json", "fixture-manifest.json", "model.safetensors"}
BASE_LOCKS = ("requirements-bootstrap.lock", "requirements-dev.lock")
BOOTSTRAP_FILES = ("scripts/bootstrap.py", "scripts/_toolchain_common.py", "scripts/_runtime_artifacts.py")
PROJECT_FILES = (".gitattributes", ".gitignore", "AI_INFERENCE_ENGINE_PROJECT_PLAN.md", "LICENSE", "README.md", "pyproject.toml", *BASE_LOCKS)
PROJECT_DIRECTORIES = (".github", "docs", "evidence", "src", "scripts", "tests", "schemas", "registries", "runtime", "configs", "fixtures")
CRITICAL_TESTS = {
    "test_real_greedy_recomputes_complete_context_and_replays", "test_seeded_sampling_is_independent_of_telemetry_and_reuse",
    "test_canonical_per_op_matches_independent_transformers", "test_all_twenty_fixed_greedy_inputs_and_linked_reports",
    "test_bad_transpose_localizes_first_q_projection_not_just_logits", "test_unknown_semantic_fields_fail_closed",
    "test_missing_unexpected_wrong_shape_fail", "test_duplicate_across_shards_rejected",
    "test_random123_all_zero", "test_random123_all_ones", "test_random123_digits_of_pi",
    "test_frozen_fixture_includes_every_stage_and_commit_abort", "test_discarded_proposal_consumes_no_draw_and_retry_is_identical",
    "test_greedy_and_sample_positions_have_separate_draw_count", "test_commit_rejects_replaced_token_spec_seed_lanes_and_probabilities",
    "test_truly_blocked_exporter_is_bounded_and_does_not_block_outcome",
}


def _fail(message: str, code: ErrorCode = ErrorCode.INVALID_ARGUMENT) -> NoReturn:
    raise IllmError(code, message)


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _fs(path: Path) -> Path:
    """Windows extended paths preserve long sealed trees without changing IDs."""
    absolute = path.absolute()
    raw = str(absolute)
    if os.name != "nt" or raw.startswith("\\\\?\\"):
        return absolute
    return Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in qualification evidence")
            result[key] = value
        return result

    def constant(value: str) -> None:
        _fail(f"non-finite JSON constant in qualification evidence: {value}")

    try:
        result = json.loads(_fs(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=constant)
    except (OSError, ValueError, UnicodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification JSON artifact is unreadable", {"file": path.name}) from exc
    if not isinstance(result, dict):
        _fail("qualification JSON artifact must be an object")
    return cast(dict[str, Any], result)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        _fail("evidence link must be a portable relative path")
    path = PurePosixPath(str(value))
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{index}" for index in range(1, 10)], *[f"LPT{index}" for index in range(1, 10)]}
    if not path.parts or path.is_absolute() or any(part in {".", ".."} or part.rstrip(" .") != part or part.split(".")[0].upper() in reserved for part in path.parts) or path.as_posix() != value:
        _fail("evidence link escapes or aliases the sealed root")
    return str(value)


def _inventory(root: Path) -> dict[str, str]:
    root = _fs(root)
    if root.is_symlink() or not root.is_dir():
        _fail("evidence root must be an ordinary directory")
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("symbolic links are forbidden in qualification evidence")
        if path.is_file():
            name = _relative(path.relative_to(root).as_posix())
            if not path.resolve().is_relative_to(root.resolve()):
                _fail("qualification artifact escapes its root")
            result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path = _fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(dict(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _signed(value: Mapping[str, Any], field: str) -> None:
    if not _digest(value.get(field)) or value[field] != sha256_json({key: item for key, item in value.items() if key != field}):
        _fail(f"qualification {field} mismatch")


def _hash_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        _fail("qualification hash inventory is missing")
    result: dict[str, str] = {}
    for key, item in cast(dict[str, Any], value).items():
        name = _relative(key)
        if not _digest(item):
            _fail("qualification hash inventory contains an invalid checksum")
        result[name] = item
    return result


def _entries_digest(entries: Mapping[str, str]) -> str:
    return sha256_json(sorted(entries.items()))


def reference_source_identity(project_root: Path) -> dict[str, str]:
    """Recompute source identity shared across CPU/CUDA runtime profiles."""
    return cast(dict[str, str], _source_provenance(project_root)["source_identity"])


def _source_provenance(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    paths = [project_root / name for name in PROJECT_FILES]
    for name in PROJECT_DIRECTORIES:
        directory = project_root / name
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts and not any(part.endswith(".egg-info") for part in path.parts))
    project_files = {path.relative_to(project_root).as_posix(): sha256_file(path) for path in sorted(set(paths)) if path.is_file()}
    base_paths = [project_root / name for name in BASE_LOCKS] + sorted((project_root / "vendor" / "wheels").glob("*.whl"))
    if len(base_paths) <= len(BASE_LOCKS) or any(not path.is_file() for path in base_paths):
        _fail("base offline lock authority is missing", ErrorCode.TOOLCHAIN_MISMATCH)
    base_files = {path.relative_to(project_root).as_posix(): sha256_file(path) for path in base_paths}
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    identity = {"project_root_digest": _entries_digest(project_files),
                "bootstrap_digest": _entries_digest({name: project_files[name] for name in BOOTSTRAP_FILES}),
                "base_lock_digest": _entries_digest(base_files)}
    return {"schema_version": 1, "project_files": project_files, "base_lock_files": base_files,
            "project_name": project["name"], "project_version": project["version"], "source_identity": identity}


def _verify_source(root: Path, expected: Mapping[str, str] | None) -> tuple[dict[str, str], dict[str, str]]:
    provenance = _object(root / "source-provenance.json")
    if set(provenance) != {"schema_version", "project_files", "base_lock_files", "project_name", "project_version", "source_identity"} or type(provenance["schema_version"]) is not int or provenance["schema_version"] != 1:
        _fail("unsupported source provenance schema")
    files = _hash_map(provenance["project_files"])
    base = _hash_map(provenance["base_lock_files"])
    if not set(PROJECT_FILES) <= set(files) or not set(BOOTSTRAP_FILES) <= set(files) or not set(BASE_LOCKS) < set(base):
        _fail("source provenance omits required authority files")
    if any(name not in BASE_LOCKS and not name.startswith("vendor/wheels/") for name in base):
        _fail("base lock inventory includes non-authority artifacts")
    identity = {"project_root_digest": _entries_digest(files),
                "bootstrap_digest": _entries_digest({name: files[name] for name in BOOTSTRAP_FILES}),
                "base_lock_digest": _entries_digest(base)}
    if provenance["source_identity"] != identity or (expected is not None and dict(expected) != identity):
        _fail("qualification evidence belongs to different source", ErrorCode.TOOLCHAIN_MISMATCH)
    versions: dict[str, str] = {}
    for name in BASE_LOCKS:
        path = root / "locks" / name
        if sha256_file(path) != base[name] or files[name] != base[name]:
            _fail("embedded base lock differs from source provenance")
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line.strip())
            if match:
                package = re.sub(r"[-_.]+", "-", match[1]).lower()
                if package in versions and versions[package] != match[2]:
                    _fail("embedded base lock package pins conflict")
                versions[package] = match[2]
    if provenance["project_name"] != "illm" or not isinstance(provenance["project_version"], str) or re.fullmatch(r"0\.1\.0(?:\.dev[0-9]+)?", provenance["project_version"]) is None:
        _fail("qualification project package identity is not V0.1")
    project_path = root / "locks" / "pyproject.toml"
    if sha256_file(project_path) != files["pyproject.toml"]:
        _fail("embedded project metadata differs from source provenance")
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
    if project.get("name") != provenance["project_name"] or project.get("version") != provenance["project_version"]:
        _fail("qualified project version/name differs from actual package metadata")
    versions["illm"] = provenance["project_version"]
    return identity, versions


def _verify_toolchain(root: Path, identity: Mapping[str, str], versions: dict[str, str]) -> dict[str, Any]:
    toolchain = _object(root / "toolchain.json")
    _signed(toolchain, "manifest_digest")
    if type(toolchain.get("schema_version")) is not int or toolchain["schema_version"] != 1 or toolchain.get("finalized") is not True:
        _fail("reference qualification requires a finalized schema1 runtime", ErrorCode.TOOLCHAIN_MISMATCH)
    runtime_id = toolchain.get("runtime_profile_id")
    if runtime_id not in RUNTIME_LOCKS or toolchain.get("runtime_lock_digest") != RUNTIME_LOCKS[runtime_id]:
        _fail("runtime profile/lock is not pinned by V0.1", ErrorCode.TOOLCHAIN_MISMATCH)
    lock = _object(root / "locks" / "runtime-lock.json")
    _signed(lock, "lock_digest")
    if lock.get("schema_version") != 1 or lock.get("kind") != "illm-runtime-artifact-lock" or lock.get("profile_id") != runtime_id or lock["lock_digest"] != toolchain["runtime_lock_digest"]:
        _fail("embedded runtime lock differs from toolchain")
    provenance = _object(root / "source-provenance.json")
    if sha256_file(root / "locks" / "runtime-lock.json") != provenance["project_files"].get(f"runtime/artifacts/{runtime_id}.json"):
        _fail("embedded runtime lock differs from source file authority")
    if any(toolchain.get(name) != identity[name] for name in ("project_root_digest", "bootstrap_digest")) or toolchain.get("lock_digest") != sha256_json({"base_lock_digest": identity["base_lock_digest"], "runtime_lock_digest": toolchain["runtime_lock_digest"]}):
        _fail("toolchain source/base/runtime lock identity mismatch", ErrorCode.TOOLCHAIN_MISMATCH)
    target = lock.get("target", {})
    if target.get("os") != toolchain.get("os") or toolchain.get("python_version") != target.get("python_validation_patch") or not str(toolchain.get("python_version", "")).startswith("3.12.") or toolchain.get("arch") not in {"AMD64", "x86_64"}:
        _fail("toolchain platform differs from pinned CPython3.12 x86_64 runtime")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        _fail("runtime lock package closure is absent")
    projected: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not _digest(artifact.get("sha256")):
            _fail("runtime lock artifact is invalid")
        name, version = artifact["name"], artifact["version"]
        if name in versions and versions[name] != version:
            _fail("base/runtime package version conflict")
        versions[name] = version
        projected.append({key: artifact[key] for key in ("name", "version", "filename", "sha256", "size_bytes")})
    if toolchain.get("runtime_artifacts") != projected or toolchain.get("installed_packages") != dict(sorted(versions.items())):
        _fail("installed packages/artifacts differ from exact pinned closure", ErrorCode.TOOLCHAIN_MISMATCH)
    probe = toolchain.get("runtime_probe")
    if not isinstance(probe, dict) or probe.get("smoke_sum") != 3.0:
        _fail("runtime execution probe is missing")
    if any(probe.get(name) != versions.get(name) for name in ("torch", "transformers", "safetensors", "tokenizers")):
        _fail("runtime probe package identity mismatch")
    cuda = runtime_id.endswith("cu126")
    if cuda:
        capability = probe.get("device_capability")
        if probe.get("device") != "cuda" or probe.get("cuda_build") != "12.6" or not isinstance(probe.get("gpu_name"), str) or not probe["gpu_name"] or not isinstance(capability, list) or len(capability) != 2 or any(type(item) is not int or item < 0 for item in capability) or capability[0] < 8 or f"sm_{capability[0]}{capability[1]}" not in probe.get("compiled_arches", []):
            _fail("CUDA qualification requires an executed compatible GPU probe")
    elif probe.get("device") != "cpu" or probe.get("cuda_build") is not None or probe.get("gpu_name") is not None or probe.get("device_capability") is not None:
        _fail("CPU runtime probe cannot claim GPU execution")
    return toolchain


def _verify_tests(root: Path, toolchain: Mapping[str, Any]) -> None:
    parse_junit(root / "junit.xml")
    suites = ET.parse(root / "junit.xml").getroot()
    passed = {case.attrib.get("name", "").split("[")[0] for case in suites.iter("testcase") if case.find("skipped") is None and case.find("failure") is None and case.find("error") is None}
    if not CRITICAL_TESTS <= passed:
        _fail("JUnit omits required executed model/engine/sampler regression tests")
    quality = _object(root / "quality.json")
    checking = quality.get("type_check", {})
    provenance = _object(root / "source-provenance.json")
    count = sum(name.startswith("src/illm/") and name.endswith(".py") for name in provenance["project_files"])
    if type(quality.get("schema_version")) is not int or quality.get("schema_version") != 0 or quality.get("status") != "pass" or quality.get("issues") != [] or type(quality.get("checked_python_files")) is not int or quality["checked_python_files"] != count:
        _fail("static quality evidence omits current source or is not passing")
    if not isinstance(checking, dict) or checking.get("tool") != "mypy" or checking.get("version") != toolchain["installed_packages"]["mypy"] or checking.get("returncode") != 0 or type(checking.get("returncode")) is not int or checking.get("status") != "pass" or not isinstance(checking.get("output"), list) or not any(re.fullmatch(r"Success: no issues found in [1-9][0-9]* source files?", line) for line in checking["output"] if isinstance(line, str)):
        _fail("strict pinned mypy evidence is absent or not passing")


def _op_order(layers: int) -> list[str]:
    names = ["embedding"]
    for layer in range(layers):
        names.extend(f"layer.{layer}.{name}" for name in ("input_norm", "q_projection", "k_projection", "v_projection", "rope_q", "rope_k", "attention_scores", "attention_probabilities", "attention_context", "attention_output", "attention_residual", "post_attention_norm", "mlp_gate", "mlp_up", "mlp_silu", "mlp_product", "mlp_output", "mlp_residual"))
    return [*names, "final_norm", "logits"]


def _canonical_shape(name: str, spec: SupportedModelSpec, length: int) -> list[int]:
    if name == "logits":
        return [1, length, spec.vocab_size]
    operation = name.split(".")[-1]
    if operation in {"q_projection", "rope_q", "attention_context"}:
        return [1, spec.num_attention_heads, length, spec.head_dim]
    if operation in {"k_projection", "v_projection", "rope_k"}:
        return [1, spec.num_kv_heads, length, spec.head_dim]
    if operation in {"attention_scores", "attention_probabilities"}:
        return [1, spec.num_attention_heads, length, length]
    if operation in {"mlp_gate", "mlp_up", "mlp_silu", "mlp_product"}:
        return [1, length, spec.intermediate_size]
    return [1, length, spec.hidden_size]


def _oracle_contract(dtype: str) -> dict[str, Any]:
    return {"factory": "Qwen2ForCausalLM._from_config(dtype=runtime_dtype)",
            "weight_source": "independent-local-verified-safetensors", "attention_implementation": "eager",
            "parameter_dtype": {"fp32": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16"}[dtype],
            "rope_inv_freq_dtype": "torch.float32", "device_transfer": "device-only-no-global-buffer-dtype-cast",
            "trace_matches_ordinary_forward": True}


def _tolerances(dtype: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if dtype == "bf16":
        return ({"atol": 0.0625, "rtol": 0.02, "profile_id": "real-bf16-op-v1"}, {"atol": 0.005, "rtol": 0.02, "profile_id": "real-bf16-attention-probability-v1"})
    if dtype == "fp16":
        return ({"atol": 0.0078125, "rtol": 0.01, "profile_id": "real-fp16-op-v1"}, {"atol": 0.001, "rtol": 0.01, "profile_id": "real-fp16-attention-probability-v1"})
    tiny = {"atol": 1e-4, "rtol": 1e-3, "profile_id": "tiny-fp32-op-v1"}
    return tiny, tiny


def _finite(value: object, *, positive: bool = False) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _tiny_fault(root: Path) -> dict[str, Any]:
    directory = root / "tiny-fixture"
    provenance = _object(root / "source-provenance.json")["project_files"]
    if set(_inventory(directory)) != TINY_FILES:
        _fail("tiny fault fixture coverage is incomplete")
    for name in TINY_FILES:
        if sha256_file(directory / name) != provenance.get(f"fixtures/tiny-decoder/{name}"):
            _fail("tiny fault fixture differs from source provenance")
    raw = _object(directory / "supported-model-spec.json")
    spec_id = raw.pop("spec_id", None)
    spec = SupportedModelSpec(**raw)
    if spec.spec_id != spec_id or spec_id != "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873":
        _fail("tiny fault fixture spec differs from immutable V0.1 fixture")
    path = directory / "model.safetensors"
    weights = build_weight_manifest((path,), spec)
    if weights != _object(directory / "weight-load-manifest.json") or sha256_file(path) != "8eeaeeb38565ca8ac40177f239a8405396a2b6642aef5223b5bf09205ffb056a":
        _fail("tiny fault fixture weights differ from immutable V0.1 fixture")
    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = inspect_safetensors(path)[name]
    data = path.read_bytes()[tensor.source_byte_start:tensor.source_byte_end]
    rows, columns = tensor.shape
    transposed = b"".join(data[(row * columns + column) * 4:(row * columns + column + 1) * 4] for column in range(columns) for row in range(rows))
    return {"spec_id": spec_id, "weights_digest": weights["weights_digest"], "load_manifest_digest": weights["manifest_digest"],
            "weights_identity_role": "unmodified-source-before-explicit-fault", "fault_fixture_id": "transpose-layer0-q-weight-v1",
            "fault_transform": {"transform_id": "transpose-axes-0-1-contiguous-v1", "tensor_name": name,
                                "shape": [columns, rows], "source_tensor_checksum": hashlib.sha256(data).hexdigest(),
                                "mutated_tensor_checksum": hashlib.sha256(transposed).hexdigest()},
            "capture_identity": "public-transpose-diagnostic-v1"}


def _compound_fault(root: Path, simple: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "tiny-fixture" / "model.safetensors"
    storage = inspect_safetensors(path)["model.norm.weight"]
    result = {name: simple[name] for name in ("spec_id", "weights_digest", "load_manifest_digest", "weights_identity_role")}
    result.update(fault_fixture_id="q-transpose-plus-zero-final-norm-v1",
                  fault_transforms=[simple["fault_transform"], {"transform_id": "zero-tensor-v1", "tensor_name": "model.norm.weight",
                                                              "shape": list(storage.shape), "source_tensor_checksum": storage.checksum,
                                                              "mutated_tensor_checksum": hashlib.sha256(bytes(storage.storage_bytes)).hexdigest()}],
                  capture_identity="public-two-fault-token-diagnostic-v1")
    return result


def _verify_diagnostic(diagnostic: Mapping[str, Any], directory: Path, expected_identity: Mapping[str, Any], *, token_mismatch: bool) -> str:
    fault = diagnostic.get("report", {})
    first = fault.get("first_divergence", {})
    link = _relative(diagnostic.get("report_link"))
    ops = fault.get("ops", [])
    if diagnostic.get("status") != "pass" or fault.get("status") != "fail" or fault.get("report_link") != link or first.get("op") != "layer.0.q_projection" or first.get("reason") != "outside_tolerance" or first.get("report_link") != link or first.get("tolerance") != _tolerances("fp32")[0] or _object(directory / link) != fault or len(ops) != 39 or [item.get("op") for item in ops] != _op_order(2) or ops[2] != first or any(item.get("status") != "pass" for item in ops[:2]):
        _fail("first-divergence diagnostic does not locate the injected transpose fault")
    expected_value, actual_value = first.get("expected_value"), first.get("actual_value")
    if first.get("expected_shape") != [1, 4, 6, 8] or first.get("actual_shape") != [1, 4, 6, 8] or type(first.get("mismatch_count")) is not int or not 1 <= first["mismatch_count"] <= 192 or type(first.get("first_flat_index")) is not int or not 0 <= first["first_flat_index"] < 192 or isinstance(expected_value, bool) or not isinstance(expected_value, (int, float)) or not math.isfinite(expected_value) or isinstance(actual_value, bool) or not isinstance(actual_value, (int, float)) or not math.isfinite(actual_value) or abs(expected_value - actual_value) <= 1e-4 + 1e-3 * abs(expected_value):
        _fail("fault first-divergence values do not demonstrate an actual tolerance violation")
    expected = {**expected_identity, "fault_execution_identity": sha256_json(dict(expected_identity))}
    if fault.get("identity") != expected:
        _fail("fault diagnostic lacks immutable model/capture provenance")
    comparison = fault.get("output_comparison", {})
    if comparison.get("evidence_kind") != "measured" or type(comparison.get("final_logits_changed_count")) is not int or comparison["final_logits_changed_count"] < 1 or not _finite(comparison.get("final_logits_max_absolute_error"), positive=True) or comparison.get("expected_next_token") != 114 or comparison.get("actual_next_token") != (0 if token_mismatch else 114) or comparison.get("token_mismatch") is not token_mismatch:
        _fail("fault diagnostic misrepresents measured final-token/logit divergence")
    return link


def _verify_validation(directory: Path, toolchain: Mapping[str, Any], tiny_fault: Mapping[str, Any], compound_fault: Mapping[str, Any]) -> dict[str, Any]:
    report = _object(directory / "validation-report.json")
    _signed(report, "report_digest")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("status") != "pass" or report.get("fixed_input_count") != 20 or type(report.get("fixed_input_count")) is not int or report.get("tf32_enabled") is not False or report.get("torch_version") != toolchain["installed_packages"]["torch"] or report.get("transformers_version") != "4.57.1":
        _fail("model validation report is not a complete pinned twenty-input pass")
    raw_spec = _object(directory / "model-spec.json")
    spec_id = raw_spec.pop("spec_id", None)
    try:
        spec = SupportedModelSpec(**raw_spec)
    except (TypeError, ModelError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualified model spec is invalid") from exc
    if spec_id != spec.spec_id or spec.runtime_dtype != report.get("runtime_dtype"):
        _fail("qualified model spec identity mismatch")
    if canonical_json_bytes(report.get("oracle_contract")) != canonical_json_bytes(_oracle_contract(spec.runtime_dtype)):
        _fail("model oracle did not preserve native dtype initialization and FP32 RoPE")
    if report.get("nonfinite_policy") != "only-paired-strict-causal-future-negative-infinity-in-attention-scores-v1":
        _fail("model report lacks the strict paired causal-mask sentinel contract")
    weights = _object(directory / "weight-load-manifest.json")
    _signed(weights, "manifest_digest")
    storages = weights.get("storages")
    if type(weights.get("schema_version")) is not int or weights.get("schema_version") != 1 or weights.get("spec_id") != spec_id or weights.get("finalized") is not True or weights.get("runtime_dtype") != spec.runtime_dtype or weights.get("parameter_count") != spec.parameter_count or weights.get("missing_keys") != [] or weights.get("unexpected_keys") != [] or weights.get("duplicate_keys") != [] or not isinstance(storages, list) or len(storages) != len(spec.tensor_shapes()) or weights.get("canonical_storage_count") != len(storages):
        _fail("qualified weight manifest is incomplete or disagrees with spec")
    shapes = spec.tensor_shapes()
    if [item.get("canonical_name") for item in storages] != sorted(shapes):
        _fail("qualified canonical storage coverage/order mismatch")
    for item in storages:
        name = item["canonical_name"]
        aliases = [name, "lm_head.weight"] if name == "model.embed_tokens.weight" and spec.tie_word_embeddings else [name]
        transforms = [] if spec.weight_dtype == spec.runtime_dtype else [{"op": "cast", "from": spec.weight_dtype, "to": spec.runtime_dtype}]
        source_range = item.get("source_byte_range")
        if item.get("shape") != list(shapes[name]) or item.get("source_shard") != "model.safetensors" or item.get("source_dtype") != {"fp32": "F32", "fp16": "F16", "bf16": "BF16"}[spec.weight_dtype] or item.get("runtime_dtype") != spec.runtime_dtype or item.get("parameter_aliases") != aliases or item.get("tied_weight_group") != ("embed-lm-head-v1" if len(aliases) == 2 else None) or item.get("transforms") != transforms or not _digest(item.get("checksum")) or not _digest(item.get("final_storage_checksum")) or (not transforms and item["final_storage_checksum"] != item["checksum"]) or item.get("final_storage_bytes") != math.prod(shapes[name]) * {"fp32": 4, "fp16": 2, "bf16": 2}[spec.runtime_dtype] or not isinstance(source_range, list) or len(source_range) != 2 or any(type(value) is not int or value < 8 for value in source_range) or source_range[1] - source_range[0] != math.prod(shapes[name]) * {"fp32": 4, "fp16": 2, "bf16": 2}[spec.weight_dtype]:
            _fail("qualified storage metadata is invalid")
    ranges = sorted(item["source_byte_range"] for item in storages)
    if any(left[1] != right[0] for left, right in zip(ranges, ranges[1:])):
        _fail("qualified weight storage ranges overlap or leave a hole")
    weights_digest = sha256_json([{"name": item["canonical_name"], "shape": item["shape"], "source_dtype": item["source_dtype"], "checksum": item["checksum"]} for item in storages])
    tokenizer = _object(directory / "tokenizer-spec.json")
    artifacts = report.get("model_artifacts")
    metadata_names = {"model-spec.json", "weight-load-manifest.json", "tokenizer-spec.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != metadata_names or any(artifacts[name] != {"path": name, "sha256": sha256_file(directory / name)} for name in metadata_names):
        _fail("model validation metadata links/checksums are incomplete")
    identity = report.get("identity")
    if not isinstance(identity, dict) or identity.get("spec_id") != spec_id or identity.get("weights_digest") != weights_digest or weights.get("weights_digest") != weights_digest or identity.get("load_manifest_digest") != weights["manifest_digest"] or identity.get("tokenizer_revision") != spec.tokenizer_revision or identity.get("capture_identity") != "public-fixed-model-validation-v1":
        _fail("model validation identity differs from immutable model assets")
    tiny = identity.get("fixture_id") == "tiny-decoder"
    if tiny:
        if spec_id != tiny_fault["spec_id"] or identity["weights_digest"] != tiny_fault["weights_digest"] or identity["load_manifest_digest"] != tiny_fault["load_manifest_digest"] or spec.runtime_dtype != "fp32" or "sha256:" + sha256_json(tokenizer) != spec.tokenizer_revision:
            _fail("tiny model/tokenizer does not match pinned fixture")
    elif identity.get("fixture_id") is not None or spec.model_revision != "060db6499f32faf8b98477b0a26969ef7d8b9987" or spec.tokenizer_revision != spec.model_revision or spec.num_layers != 24 or spec.hidden_size != 896 or spec.vocab_size != 151936 or tokenizer.get("revision") != spec.tokenizer_revision or spec.intermediate_size != 4864 or spec.num_attention_heads != 14 or spec.num_kv_heads != 2 or spec.head_dim != 64 or spec.rms_norm_eps != 1e-6 or spec.rope_theta != 1e6 or spec.max_position_embeddings != 32768 or spec.tie_word_embeddings is not True or spec.weight_dtype != "bf16" or spec.bos_token_id != 151643 or spec.eos_token_id != 151643 or weights.get("source_shards") != [{"shard": "model.safetensors", "sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"}]:
        _fail("real model differs from pinned Qwen2.5-0.5B revision/architecture")
    tolerance, probability = _tolerances(spec.runtime_dtype)
    if report.get("tolerance_profile") != tolerance or report.get("attention_probability_tolerance") != probability:
        _fail("model validation used unregistered or loosened tolerances")
    expected_inputs = [f"fixed-greedy-{index:02d}" for index in range(20)]
    greedy, op_reports = report.get("greedy_reports"), report.get("per_op_reports")
    if not isinstance(greedy, list) or not isinstance(op_reports, list) or len(greedy) != 20 or len(op_reports) != 20:
        _fail("model validation omits fixed greedy/per-op cases")
    if [item.get("public_input_fixture_id") for item in greedy] != expected_inputs:
        _fail("fixed greedy inputs are duplicated, reordered or missing")
    links: set[str] = {"validation-report.json", "model-spec.json", "weight-load-manifest.json", "tokenizer-spec.json"}
    order = _op_order(spec.num_layers)
    for token_case, op_report in zip(greedy, op_reports):
        tokens = token_case.get("reference_tokens")
        if token_case.get("status") != "pass" or not isinstance(tokens, list) or not 1 <= len(tokens) <= 8 or any(type(token) is not int or not 0 <= token < spec.vocab_size for token in tokens) or (len(tokens) < 8 and tokens[-1] != spec.eos_token_id) or spec.eos_token_id in tokens[:-1] or tokens != token_case.get("oracle_tokens") or type(token_case.get("generated_count")) is not int or token_case["generated_count"] != len(tokens) or not _digest(token_case.get("input_identity")):
            _fail("greedy oracle token equality is missing")
        op_identity = op_report.get("identity", {})
        if any(op_identity.get(key) != value for key, value in identity.items()) or op_identity.get("input_identity") != token_case["input_identity"] or op_identity.get("public_input_fixture_id") != token_case["public_input_fixture_id"]:
            _fail("per-op report is bound to another model/input")
        link = _relative(op_report.get("report_link"))
        if link in links or _object(directory / link) != op_report:
            _fail("per-op report link is duplicate or does not match embedded report")
        links.add(link)
        ops = op_report.get("ops")
        if op_report.get("status") != "pass" or op_report.get("first_divergence") is not None or op_report.get("unexpected_ops") != [] or op_report.get("op_count") != len(order) or not isinstance(ops, list) or [item.get("op") for item in ops] != order:
            _fail("per-op golden coverage/status is incomplete")
        embedding_shape = ops[0].get("expected_shape")
        if not isinstance(embedding_shape, list) or len(embedding_shape) != 3 or type(embedding_shape[1]) is not int or not 1 <= embedding_shape[1] <= spec.max_position_embeddings:
            _fail("per-op golden lacks a valid input sequence shape")
        length = embedding_shape[1]
        dtype = {"fp32": "torch.float32", "bf16": "torch.bfloat16", "fp16": "torch.float16"}[spec.runtime_dtype]
        for op in ops:
            active = probability if op["op"].endswith("attention_probabilities") else tolerance
            expected_shape = _canonical_shape(op["op"], spec, length)
            if op.get("status") != "pass" or op.get("evidence_kind") != "measured" or op.get("tolerance") != active or op.get("report_link") != link or op.get("expected_shape") != expected_shape or op.get("actual_shape") != expected_shape or op.get("expected_dtype") != dtype or op.get("actual_dtype") != dtype or type(op.get("mismatch_count")) is not int or op["mismatch_count"] != 0 or not _finite(op.get("max_absolute_error")) or not _finite(op.get("max_relative_error")):
                _fail("per-op golden is not a measured numerical pass")
            masked_count = op.get("masked_negative_infinity_count")
            limit = spec.num_attention_heads * length * (length - 1) // 2 if op["op"].endswith(".attention_scores") else 0
            if type(masked_count) is not int or not 0 <= masked_count <= limit:
                _fail("per-op report admits non-finite values outside paired future-mask sentinels")
    simple_link = _verify_diagnostic(report.get("first_divergence_diagnostic", {}), directory, tiny_fault, token_mismatch=False)
    compound_link = _verify_diagnostic(report.get("token_mismatch_diagnostic", {}), directory, compound_fault, token_mismatch=True)
    if simple_link in links or compound_link in links or simple_link == compound_link:
        _fail("fault diagnostic report links alias other evidence")
    links.update((simple_link, compound_link))
    # Model metadata may carry a hashes manifest generated alongside the report.
    actual_files = set(_inventory(directory))
    if actual_files != links:
        _fail("validation directory contains missing or unreferenced evidence files")
    return {"report": report, "spec": spec, "weights": weights, "tokenizer": tokenizer}


def _capture(root: Path, validation: Mapping[str, Any]) -> ReferenceCaptureBundle:
    if set(_inventory(root)) != CAPTURE_FILES:
        _fail("reference capture artifact coverage is not exact")
    bundle = ReferenceCaptureBundle.open(root)
    identity = bundle.manifest["identity"]
    report_identity = validation["report"]["identity"]
    expected = {"model_spec_id": report_identity["spec_id"], "weights_digest": report_identity["weights_digest"],
                "load_manifest_digest": report_identity["load_manifest_digest"], "tokenizer_digest": sha256_json(validation["tokenizer"])}
    if bundle.final_state["status"] != "FINISHED" or bundle.manifest["replay_capabilities"] != ["R0", "R1_REFERENCE"] or any(identity.get(name) != value for name, value in expected.items()) or identity.get("runtime_dtype") != validation["spec"].runtime_dtype or identity.get("device") != validation["report"]["device"] or identity.get("backend") != "illm_eager":
        _fail("reference capture is not a finished replayable run of the validated model")
    return bundle


_TELEMETRY_RESULT_FIELDS = ("input_token_ids", "output_token_ids", "finish_reason", "actual_seed", "sampling_spec_digest",
                            "next_output_position", "rng_draws_consumed", "committed_draw_addresses")


def _verify_telemetry_invariance(directory: Path, value: Any, validation: Mapping[str, Any], lengths: set[int]) -> set[str]:
    expected_spec = SamplingSpec(mode="sample", temperature=0.8, top_k=50, top_p=0.9, actual_seed=7)
    fields = {"schema_version", "status", "evidence_kind", "scope", "token_id_disclosure", "modes", "comparison_fields",
              "sampling_spec", "sampling_spec_digest", "context_buckets"}
    if not isinstance(value, dict) or set(value) != fields or type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("status") != "pass" or value.get("evidence_kind") != "measured" or value.get("scope") != "fixed-public-synthetic-context-fixtures-only" or value.get("token_id_disclosure") != "public-synthetic-fixture-results-only; captures-remain-metadata-only" or value.get("modes") != ["off", "request", "ops"] or value.get("comparison_fields") != list(_TELEMETRY_RESULT_FIELDS) or value.get("sampling_spec") != expected_spec.to_dict() or value.get("sampling_spec_digest") != expected_spec.digest:
        _fail("benchmark omits the executed sampled telemetry-invariance contract")
    buckets = value.get("context_buckets")
    if not isinstance(buckets, list) or len(buckets) != len(lengths):
        _fail("telemetry invariance omits benchmark context fixtures")
    seen: set[int] = set()
    referenced: set[str] = set()
    for bucket in buckets:
        length = bucket.get("context_tokens")
        outcomes = bucket.get("outcomes")
        if set(bucket) != {"context_tokens", "public_input_fixture_id", "status", "outcomes"} or type(length) is not int or length not in lengths or length in seen or bucket.get("status") != "pass" or not isinstance(outcomes, list) or len(outcomes) != 3:
            _fail("telemetry-invariance context coverage is invalid")
        seen.add(length)
        ids = [3 + index % (validation["spec"].vocab_size - 3) for index in range(length)]
        if bucket.get("public_input_fixture_id") != sha256_json({"schema_version": 1, "kind": "ascending-vocab-v1", "input_ids": ids}):
            _fail("telemetry-invariance public input identity is invalid")
        common: bytes | None = None
        for mode, outcome in zip(("off", "request", "ops"), outcomes, strict=True):
            expected_fields = {"telemetry", "request_id", "generation_result", "generation_result_digest", "capture", "capture_manifest_digest"}
            if not isinstance(outcome, dict) or set(outcome) != expected_fields or outcome.get("telemetry") != mode or outcome.get("request_id") != f"telemetry-{length}":
                _fail("telemetry-invariance mode invocation is missing or duplicated")
            result = outcome.get("generation_result")
            if not isinstance(result, dict) or set(result) != set(_TELEMETRY_RESULT_FIELDS) or outcome.get("generation_result_digest") != sha256_json(result):
                _fail("telemetry-invariance public result fields/checksum are invalid")
            tokens = result.get("output_token_ids")
            if result.get("input_token_ids") != ids or not isinstance(tokens, list) or len(tokens) != 1 or any(type(token) is not int or not 0 <= token < validation["spec"].vocab_size for token in tokens) or type(result.get("actual_seed")) is not int or result["actual_seed"] != 7 or result.get("sampling_spec_digest") != expected_spec.digest or type(result.get("next_output_position")) is not int or result["next_output_position"] != 1 or type(result.get("rng_draws_consumed")) is not int or result["rng_draws_consumed"] != 1 or canonical_json_bytes(result.get("committed_draw_addresses")) != b"[[0,0,0]]":
                _fail("telemetry-invariance result changes input/seed/spec/RNG accounting")
            expected_finish = "EOS" if tokens[0] == validation["spec"].eos_token_id else "MAX_NEW_TOKENS"
            if result.get("finish_reason") != expected_finish:
                _fail("telemetry-invariance result finish reason disagrees with its public token")
            encoded = canonical_json_bytes(result)
            if common is not None and encoded != common:
                _fail("telemetry changed a public fixture generation result")
            common = encoded
            link = _relative(outcome.get("capture"))
            if link != f"telemetry-invariance/context-{length}/{mode}":
                _fail("telemetry-invariance capture mode/fixture link is invalid")
            capture = _capture(directory / link, validation)
            state = capture.final_state
            if capture.manifest["manifest_digest"] != outcome.get("capture_manifest_digest") or capture.manifest["sampling_spec"] != expected_spec.to_dict() or state["request_id"] != outcome["request_id"] or state["input_tokens"] != length or state["max_new_tokens"] != 1 or state["output_tokens"] != 1 or state["finish_reason"] != result["finish_reason"] or any(state[name] != result[name] for name in ("sampling_spec_digest", "next_output_position", "rng_draws_consumed", "committed_draw_addresses")):
                _fail("telemetry-invariance result does not bind replayable capture controls")
            operations = [event for event in capture.events if event.kind is K.MODEL_OP]
            expected_ops = _op_order(validation["spec"].num_layers) if mode == "ops" else []
            if [event.payload["op_name"] for event in operations] != expected_ops or any(event.payload["output_position"] != 0 or list(event.payload["shape"]) != _canonical_shape(event.payload["op_name"], validation["spec"], length) for event in operations):
                _fail("telemetry-invariance capture does not execute its declared op detail mode")
            referenced.update(f"{link}/{name}" for name in CAPTURE_FILES)
    return referenced


def _verify_benchmark(directory: Path, toolchain: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    files = _inventory(directory)
    checksums = _hash_map(_object(directory / "checksums.json"))
    if "INCOMPLETE" in files or "COMPLETE" not in files or set(checksums) != set(files) - {"COMPLETE", "checksums.json"} or any(files.get(name) != digest for name, digest in checksums.items()):
        _fail("benchmark is incomplete or checksum coverage is not exact")
    report = _object(directory / "benchmark-report.json")
    _signed(report, "report_digest")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("status") != "pass" or report.get("experiment_id") != "reference-full-context-v1" or report.get("performance_outcome") != "baseline_only" or report.get("evidence_kind") != "measured" or report.get("speedup_claim") is not None:
        _fail("benchmark is not a stable measured baseline")
    identity = report.get("identity", {})
    if identity != {"model_spec_id": validation["report"]["identity"]["spec_id"], "weights_digest": validation["report"]["identity"]["weights_digest"], "toolchain_manifest_digest": toolchain["manifest_digest"], "project_root_digest": toolchain["project_root_digest"]}:
        _fail("benchmark source/model/toolchain identity mismatch")
    gpu = str(validation["report"]["device"]).startswith("cuda")
    hardware = report.get("hardware", {})
    if hardware.get("device") != validation["report"]["device"] or hardware.get("torch_version") != toolchain["installed_packages"]["torch"] or type(hardware.get("cpu_threads")) is not int or hardware["cpu_threads"] < 1:
        _fail("benchmark hardware/runtime metadata is invalid")
    if gpu and (hardware.get("name") != toolchain["runtime_probe"]["gpu_name"] or hardware.get("compute_capability") != toolchain["runtime_probe"]["device_capability"] or hardware.get("cuda_runtime") != toolchain["runtime_probe"]["cuda_build"] or type(hardware.get("total_device_memory_bytes")) is not int or hardware["total_device_memory_bytes"] < 1):
        _fail("benchmark GPU hardware differs from executed runtime probe")
    contract = report.get("measurement_contract", {})
    warmups, repetitions = contract.get("warmups_per_bucket"), contract.get("repetitions_per_bucket")
    if type(warmups) is not int or warmups < 1 or type(repetitions) is not int or repetitions < 3 or contract.get("output_tokens_per_repetition") != 1 or contract.get("clock") != "perf_counter_ns" or contract.get("tf32_enabled") is not False or contract.get("telemetry") != "request" or contract.get("synchronization") != ("cuda_device_boundary" if gpu else "synchronous_cpu") or contract.get("cpu_peak_memory_measured") is not False or contract.get("load_time_excluded") is not True or contract.get("stability_rule") != "relative_median_absolute_deviation<=0.20; no observations discarded":
        _fail("benchmark measurement contract is incomplete")
    if contract.get("partial_timing_spans_are_total_overhead") is not False or contract.get("telemetry_invariance_is_timing_comparison") is not False:
        _fail("benchmark must distinguish partial timing spans and output invariance from overhead claims")
    buckets = report.get("context_buckets")
    if not isinstance(buckets, list) or len(buckets) < 3:
        _fail("benchmark requires three distinct context buckets")
    lengths: set[int] = set()
    referenced = {"benchmark-report.json", "checksums.json", "COMPLETE"}
    for bucket in buckets:
        length = bucket.get("context_tokens")
        observations = bucket.get("observations")
        if type(length) is not int or length < 1 or length in lengths or not isinstance(observations, list) or len(observations) != repetitions:
            _fail("benchmark context/repetition coverage is invalid")
        lengths.add(length)
        ids = tuple(3 + index % (validation["spec"].vocab_size - 3) for index in range(length))
        if bucket.get("public_input_fixture_id") != sha256_json({"schema_version": 1, "kind": "ascending-vocab-v1", "input_ids": ids}):
            _fail("benchmark synthetic input identity mismatch")
        times: list[int] = []
        for index, observation in enumerate(observations):
            duration, total = observation.get("server_full_context_forward_wall_time_ns"), observation.get("total_request_wall_time_ns")
            if observation.get("repetition") != index or type(observation.get("repetition")) is not int or observation.get("context_tokens") != length or observation.get("output_tokens") != 1 or type(duration) is not int or duration < 1 or type(total) is not int or total < duration:
                _fail("benchmark observation timing/count is invalid")
            if any(type(observation.get(name)) is not int or not 0 <= observation[name] <= total for name in ("logits_host_export_wall_time_ns", "sampling_wall_time_ns")):
                _fail("benchmark must retain nonnegative integer host-export and sampling timing spans")
            if duration + observation["logits_host_export_wall_time_ns"] + observation["sampling_wall_time_ns"] > total:
                _fail("benchmark nonoverlapping partial timing spans exceed the total request duration")
            times.append(duration)
            link = _relative(observation.get("capture"))
            if any(f"{link}/{name}" in referenced for name in CAPTURE_FILES):
                _fail("benchmark reuses a capture for multiple observations")
            capture = _capture(directory / link, validation)
            if capture.manifest["manifest_digest"] != observation.get("capture_manifest_digest") or capture.final_state["input_tokens"] != length or capture.final_state["output_tokens"] != 1 or capture.final_state["max_new_tokens"] != 1 or capture.final_state["request_id"] != f"measured-{length}-{index}":
                _fail("benchmark observation does not bind its replayable capture")
            for name in CAPTURE_FILES:
                referenced.add(f"{link}/{name}")
            allocated, reserved = observation.get("peak_device_allocated_bytes"), observation.get("peak_device_reserved_bytes")
            if gpu:
                if type(allocated) is not int or type(reserved) is not int or not 0 < allocated <= reserved <= hardware["total_device_memory_bytes"]:
                    _fail("benchmark GPU peak allocation evidence is invalid")
            elif allocated is not None or reserved is not None:
                _fail("CPU benchmark cannot claim measured GPU memory")
        median = float(statistics.median(times))
        relative_mad = float(statistics.median([abs(value - median) for value in times])) / median
        if bucket.get("median_forward_wall_time_ns") != median or bucket.get("min_forward_wall_time_ns") != min(times) or bucket.get("max_forward_wall_time_ns") != max(times) or bucket.get("relative_median_absolute_deviation") != relative_mad or relative_mad > 0.20 or bucket.get("stability") != "pass":
            _fail("benchmark aggregates/stability do not recompute from all observations")
    referenced.update(_verify_telemetry_invariance(directory, report.get("telemetry_invariance"), validation, lengths))
    if set(files) != referenced:
        _fail("benchmark contains unreferenced or missing capture artifacts")


def _sampler_report(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SAMPLER_SHA256:
        _fail("canonical sampler fixture differs from frozen v1 known answers")
    fixture = _object(path)
    spec = SamplingSpec.from_dict(fixture["sampling_spec"])
    initial = SamplingState.for_request(fixture["initial_state"]["request_id"], spec)
    proposal = prepare_sample(fixture["logits"], spec, initial)
    committed = commit_sample(initial, proposal, spec=spec, logits=fixture["logits"])
    decision = json.loads(canonical_json_bytes(proposal.to_dict()))
    if initial.to_dict() != fixture["initial_state"] or initial.to_dict() != fixture["abort_state"] or decision != fixture["decision"] or committed.to_dict() != fixture["committed_state"]:
        _fail("canonical sampler stages/commit/abort differ from known answers")
    vectors = [([0] * 4, [0] * 2, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
               ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
               ([0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344], [0xA4093822, 0x299F31D0], (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1))]
    if any(philox4x32_10(counter, key) != expected for counter, key, expected in vectors):
        _fail("Philox differs from independent Random123 known answers")
    return {"schema_version": 1, "status": "pass", "fixture_sha256": SAMPLER_SHA256,
            "random123_vectors_passed": 3, "all_sampler_stages_compared": True,
            "commit_abort_controls_compared": True, "sampling_spec_digest": spec.digest}


def _verify_gpu_failure(directory: Path, toolchain: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    if set(_inventory(directory)) != {"oom-report.json", *(f"capture/{name}" for name in CAPTURE_FILES)}:
        _fail("GPU failure evidence has incomplete or unreferenced artifacts")
    report = _object(directory / "oom-report.json")
    _signed(report, "report_digest")
    if not str(toolchain["runtime_profile_id"]).endswith("cu126") or not str(validation["report"]["device"]).startswith("cuda"):
        _fail("CPU/injected failures cannot qualify actual GPU allocator OOM")
    fixed = {"schema_version": 1, "profile_id": "real-cuda-allocator-oom-v1", "status": "pass", "evidence_kind": "measured",
             "isolation": "separate-process-pytorch-allocator-budget", "allocation_budget_scope": "this_process_only",
             "injection": "allocator_budget_only_no_mock_no_tensor_corruption", "retry_count": 0,
             "caught_error_code": "RESOURCE_EXHAUSTED", "genuine_torch_oom": True, "probe_error": None,
             "device": validation["report"]["device"], "device_name": toolchain["runtime_probe"]["gpu_name"],
             "compute_capability": toolchain["runtime_probe"]["device_capability"],
             "torch_version": toolchain["installed_packages"]["torch"], "cuda_runtime_version": "12.6"}
    if any(report.get(name) != value or type(report.get(name)) is not type(value) for name, value in fixed.items()) or type(report.get("process_id")) is not int or report["process_id"] < 1:
        _fail("GPU failure report does not describe an actual isolated allocator-budget OOM")
    expected_identity = {"model_spec_id": validation["report"]["identity"]["spec_id"],
                         "weights_digest": validation["report"]["identity"]["weights_digest"],
                         "load_manifest_digest": validation["report"]["identity"]["load_manifest_digest"],
                         "runtime_dtype": validation["spec"].runtime_dtype, "tokenizer_revision": validation["spec"].tokenizer_revision}
    if report.get("identity") != expected_identity:
        _fail("GPU OOM test ran a different model/dtype than qualified generation")
    fraction, memory = report.get("allocator_budget_fraction"), report.get("total_device_memory_bytes")
    if not _finite(fraction, positive=True):
        _fail("GPU OOM allocator fraction is invalid")
    budget_fraction = float(cast(float, fraction))
    # CUDA reports the effective integer-byte allowance divided by device
    # capacity, not necessarily the originally requested floating fraction.
    if budget_fraction > 1e-4 or type(memory) is not int or memory < 1 or type(report.get("allocator_budget_bytes")) is not int or report["allocator_budget_bytes"] != int(memory * budget_fraction) or report.get("allocator_applied_fraction") != report["allocator_budget_bytes"] / memory:
        _fail("GPU OOM allocator budget was not the declared process-local bound")
    before, after, restore = report.get("allocator_before", {}), report.get("allocator_after_request", {}), report.get("allocator_restore", {})
    if not _finite(before.get("fraction"), positive=True) or before["fraction"] > 1 or restore.get("status") != "pass" or restore.get("original_fraction") != before["fraction"] or restore.get("restored_fraction") != before["fraction"] or report.get("engine_close") != {"status": "pass"} or type(after.get("oom_count_delta")) is not int or after["oom_count_delta"] < 1:
        _fail("GPU OOM evidence lacks actual allocator failure/restoration")
    for allocation in (before, after):
        allocated, reserved = allocation.get("allocated_bytes"), allocation.get("reserved_bytes")
        if type(allocated) is not int or type(reserved) is not int or not 0 < allocated <= reserved <= memory:
            _fail("GPU OOM allocator measurements are invalid")
    chain = report.get("exception_chain")
    if not isinstance(chain, list) or not 1 <= len(chain) <= 8 or not any(item.get("is_torch_oom") is True and item.get("exception_type") == "torch.OutOfMemoryError" for item in chain):
        _fail("GPU failure has no genuine torch.OutOfMemoryError cause")
    for item in chain:
        if set(item) != {"exception_type", "is_torch_oom", "message_sha256", "message_excerpt"} or type(item["is_torch_oom"]) is not bool or not _digest(item["message_sha256"]) or not isinstance(item["message_excerpt"], str) or len(item["message_excerpt"]) > 512:
            _fail("GPU exception provenance is malformed")
    bundle = ReferenceCaptureBundle.open(directory / "capture")
    accepted = sum(event.kind is K.REQUEST_ACCEPTED for event in bundle.events)
    failed = [event for event in bundle.events if event.kind is K.REQUEST_FAILED]
    finished = sum(event.kind is K.REQUEST_FINISHED for event in bundle.events)
    identity = bundle.manifest["identity"]
    if accepted != 1 or len(failed) != 1 or finished != 0 or failed[0].payload["error_code"] != "RESOURCE_EXHAUSTED" or bundle.final_state["status"] != "FAILED" or bundle.final_state["error_code"] != "RESOURCE_EXHAUSTED" or identity.get("device") != report["device"] or any(identity.get(name) != expected_identity[name] for name in ("model_spec_id", "weights_digest", "load_manifest_digest", "runtime_dtype")) or report.get("context_tokens") != bundle.final_state["input_tokens"] or report.get("max_new_tokens") != 1 or bundle.final_state["max_new_tokens"] != 1:
        _fail("GPU OOM capture does not prove exactly one accepted FAILED request")
    expected_capture = {"path": "capture", "manifest_digest": bundle.manifest["manifest_digest"],
                        "manifest_sha256": sha256_file(directory / "capture" / "capture-manifest.json"),
                        "event_count": len(bundle.events), "accepted_count": 1, "failed_count": 1, "finished_count": 0,
                        "run_status": "FAILED", "r1_status": "pass", "final_state_digest": sha256_json(bundle.final_state),
                        "privacy_mode": "metadata_only", "critical_event_drop_count": 0}
    if report.get("capture") != expected_capture or bundle.manifest["replay_capabilities"] != ["R0", "R1_REFERENCE"]:
        _fail("GPU OOM capture/replay identity differs from report")


def _verify_gate(directory: Path, toolchain: Mapping[str, Any], identity: Mapping[str, str]) -> None:
    report = _object(directory / "gate-report.json")
    _signed(report, "report_digest")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("status") != "pass" or report.get("toolchain_manifest_digest") != toolchain["manifest_digest"] or report.get("source_identity") != dict(identity) or report.get("fresh_environment") is not True or report.get("offline_bootstrap") is not True or report.get("network_used") != "not_observed" or report.get("network_blocking") != "not_enforced" or report.get("network_policy") != "offline_resolution_and_local_model_only":
        _fail("clean-room report is not honest complete offline-resolution evidence")
    commands = report.get("commands")
    phases = {"bootstrap", "quality", "pytest", "inspect", "generate", "validate", "replay-trace", "replay-steps", "benchmark"}
    if not isinstance(commands, list) or not phases <= {item.get("phase") for item in commands} <= phases | {"validate-fp16", "gpu-oom"} or len(commands) != len({item.get("phase") for item in commands}):
        _fail("clean-room gate omits required executed phases")
    files = {"gate-report.json"}
    for command in commands:
        if set(command) != {"phase", "argv", "returncode", "stdout_file", "stderr_file", "stdout_sha256", "stderr_sha256"} or type(command["returncode"]) is not int or command["returncode"] != 0 or not isinstance(command["argv"], list) or not command["argv"] or any(not isinstance(arg, str) or not arg for arg in command["argv"]):
            _fail("clean-room phase has invalid invocation/exit facts")
        argv = command["argv"]
        joined = " ".join(argv)
        phase = command["phase"]
        required = {"bootstrap": ("bootstrap.py", "--locked", "--offline", "--runtime-profile"), "quality": ("in_env.py", "check_quality.py"), "pytest": ("in_env.py", "pytest", "--junitxml"), "inspect": ("in_env.py", "illm", "model", "inspect"), "generate": ("in_env.py", "illm", "generate", "--capture"), "validate": ("in_env.py", "illm", "model", "validate"), "validate-fp16": ("in_env.py", "illm", "model", "validate"), "replay-trace": ("in_env.py", "illm", "replay", "trace"), "replay-steps": ("in_env.py", "illm", "replay", "state", "--assert-state-digest"), "benchmark": ("in_env.py", "benchmark_reference.py"), "gpu-oom": ("in_env.py", "probe_gpu_oom.py")}[phase]
        if any(token not in joined for token in required):
            _fail("clean-room phase command does not execute its declared gate")
        for stream in ("stdout", "stderr"):
            link = _relative(command[f"{stream}_file"])
            if link in files or not _digest(command[f"{stream}_sha256"]) or sha256_file(directory / link) != command[f"{stream}_sha256"]:
                _fail("clean-room transcript checksum/link is invalid")
            files.add(link)
        _verify_phase_output(phase, directory / command["stdout_file"], directory.parent, toolchain)
    if set(_inventory(directory)) != files:
        _fail("clean-room transcript inventory is not exact")


def _verify_phase_output(phase: str, stdout: Path, root: Path, toolchain: Mapping[str, Any]) -> None:
    text = stdout.read_text(encoding="utf-8")
    if phase == "pytest":
        counts = parse_junit(root / "junit.xml")
        passed = int(counts["tests"] - counts["skipped"])
        if str(passed) not in re.findall(r"(?<![0-9])([0-9]+) passed", text):
            _fail("pytest transcript does not match actual JUnit accounting")
        return
    objects: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    if not objects:
        _fail("clean-room transcript lacks its actual structured command result")
    result = objects[-1]
    capture = ReferenceCaptureBundle.open(root / "capture")
    validation_reports = [_object(path / "validation-report.json") for path in (root / "validation").iterdir()]
    primary = next(item for item in validation_reports if item["identity"]["spec_id"] == capture.manifest["identity"]["model_spec_id"])
    valid = False
    if phase == "bootstrap":
        valid = result == toolchain
    elif phase == "quality":
        valid = result == _object(root / "quality.json")
    elif phase == "inspect":
        valid = result.get("spec", {}).get("spec_id") == primary["identity"]["spec_id"] and result.get("weight_manifest", {}).get("manifest_digest") == primary["identity"]["load_manifest_digest"]
    elif phase == "generate":
        inputs, outputs = result.get("input_token_ids"), result.get("output_token_ids")
        valid = (result.get("capture_manifest_digest") == capture.manifest["manifest_digest"]
                 and result.get("model_spec_id") == primary["identity"]["spec_id"]
                 and result.get("sampling_spec_digest") == capture.final_state["sampling_spec_digest"]
                 and result.get("actual_seed") == capture.manifest["sampling_spec"]["actual_seed"]
                 and isinstance(inputs, list) and len(inputs) == capture.final_state["input_tokens"]
                 and isinstance(outputs, list) and len(outputs) == capture.final_state["output_tokens"]
                 and result.get("next_output_position") == capture.final_state["next_output_position"]
                 and result.get("rng_draws_consumed") == capture.final_state["rng_draws_consumed"])
    elif phase in {"validate", "validate-fp16"}:
        expected = next((item for item in validation_reports if item["runtime_dtype"] == "fp16"), None) if phase == "validate-fp16" else primary
        valid = result == expected
    elif phase == "replay-trace":
        displayed = result.get("displayed_events")
        valid = result.get("mode") == "R0" and result.get("total_events") == len(capture.events) and type(displayed) is int and 0 < displayed <= len(capture.events) and result.get("capture_schema_version") == 1 and result.get("privacy_mode") == "metadata_only"
    elif phase == "replay-steps":
        valid = result == {"mode": "R1_REFERENCE", "asserted": True, "numerical_reexecution": False,
                           "final_state_digest": capture.manifest["final_state_digest"], "status": "FINISHED",
                           "output_tokens": capture.final_state["output_tokens"], "rng_draws_consumed": capture.final_state["rng_draws_consumed"]}
    elif phase == "benchmark":
        valid = result.get("status") == "pass" and result.get("report_digest") == _object(root / "benchmark" / "benchmark-report.json")["report_digest"]
    elif phase == "gpu-oom":
        valid = result.get("status") == "pass" and result.get("report_digest") == _object(root / "gpu-failure" / "oom-report.json")["report_digest"]
    if not valid:
        _fail("clean-room command transcript does not bind its verified output artifact")


def _verify_facts(root: Path, profile_id: str, expected: Mapping[str, str] | None) -> tuple[dict[str, Any], dict[str, str]]:
    allowed = {"source-provenance.json", "toolchain.json", "junit.xml", "quality.json", "locks", "tiny-fixture", "validation", "capture", "benchmark", "sampling",
               "gate", "gpu-failure", "profile-result.json", "evidence-index.json", "SHA256SUMS.json", "COMPLETE", "INCOMPLETE"}
    if any(path.name not in allowed for path in root.iterdir()) or set(_inventory(root / "locks")) != {*BASE_LOCKS, "runtime-lock.json", "pyproject.toml"} or set(_inventory(root / "sampling")) != {"known-answer.json", "verification.json"}:
        _fail("qualification contains missing or unreferenced evidence artifacts")
    identity, versions = _verify_source(root, expected)
    toolchain = _verify_toolchain(root, identity, versions)
    required_runtime = FORMAL_PROFILE_RUNTIMES[profile_id]
    if toolchain.get("runtime_profile_id") != required_runtime:
        _fail(
            f"{profile_id} requires exact formal runtime {required_runtime}",
            ErrorCode.TOOLCHAIN_MISMATCH,
        )
    _verify_tests(root, toolchain)
    directories = sorted((root / "validation").iterdir())
    if not directories or any(not directory.is_dir() or directory.is_symlink() for directory in directories):
        _fail("qualification model validations are absent")
    tiny_fault = _tiny_fault(root)
    compound_fault = _compound_fault(root, tiny_fault)
    validations = [_verify_validation(directory, toolchain, tiny_fault, compound_fault) for directory in directories]
    cuda = str(toolchain["runtime_profile_id"]).endswith("cu126")
    if profile_id == "real-model-single-gpu-v1" and (not cuda or {item["spec"].runtime_dtype for item in validations} != {"bf16", "fp16"} or any(item["report"]["identity"]["fixture_id"] is not None or not str(item["report"]["device"]).startswith("cuda") for item in validations) or len(validations) != 2):
        _fail("GPU profile requires actual CUDA BF16 and FP16 validation, never CPU substitution")
    if profile_id == "tiny-reference-cpu-v1" and (cuda or len(validations) != 1 or validations[0]["report"]["identity"]["fixture_id"] != "tiny-decoder" or validations[0]["report"]["device"] != "cpu"):
        _fail("tiny CPU profile requires an executed FP32 tiny CPU validation")
    bundle_identity = _object(root / "capture" / "capture-manifest.json").get("identity", {})
    primary = [item for item in validations if item["report"]["identity"]["spec_id"] == bundle_identity.get("model_spec_id")]
    if len(primary) != 1:
        _fail("generation capture is not bound to exactly one validated model")
    _capture(root / "capture", primary[0])
    _verify_benchmark(root / "benchmark", toolchain, primary[0])
    if profile_id == "real-model-single-gpu-v1" and not (root / "gpu-failure").is_dir():
        _fail("GPU profile requires actual isolated allocator OOM evidence")
    if (root / "gpu-failure").exists():
        _verify_gpu_failure(root / "gpu-failure", toolchain, primary[0])
    sampler = _sampler_report(root / "sampling" / "known-answer.json")
    if _object(root / "sampling" / "verification.json") != sampler:
        _fail("sampler qualification report does not recompute")
    if profile_id == "clean-room-release-v1":
        _verify_gate(root / "gate", toolchain, identity)
    elif (root / "gate").exists():
        _fail("non-clean-room profile contains unrelated gate evidence")
    return toolchain, identity


def _assert_live_environment(toolchain: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    if Path(str(toolchain.get("python_executable", ""))).resolve() != Path(sys.executable).resolve() or toolchain.get("os") != platform.system() or toolchain.get("arch") != platform.machine() or toolchain.get("python_version") != platform.python_version() or os.environ.get("ILLM_ENVIRONMENT_ID") != toolchain.get("environment_id") or os.environ.get("ILLM_LOCK_DIGEST") != toolchain.get("lock_digest"):
        _fail("qualification is not executing inside its finalized manifest environment", ErrorCode.TOOLCHAIN_MISMATCH)
    installed = {re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(): distribution.version for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    if installed != toolchain.get("installed_packages"):
        _fail("live installed packages differ from finalized environment", ErrorCode.TOOLCHAIN_MISMATCH)
    provenance = _source_provenance(project_root)
    if any(provenance["source_identity"][name] != toolchain.get(name) for name in ("project_root_digest", "bootstrap_digest")):
        _fail("toolchain source changed before qualification sealing", ErrorCode.TOOLCHAIN_MISMATCH)
    return provenance


def create_reference_profile_evidence(*, profile_id: str, toolchain_manifest: Path, junit: Path, quality: Path,
                                      capture: Path, validation_directories: Sequence[Path], benchmark: Path,
                                      sampler_fixture: Path, output: Path,
                                      expected_source_identity: Mapping[str, str] | None = None,
                                      clean_room_report: Path | None = None,
                                      gpu_failure_evidence: Path | None = None) -> dict[str, Any]:
    """Validate and atomically seal one immutable profile; never tag or promote."""
    if profile_id not in PROFILES or not validation_directories:
        _fail("unknown V0.1 profile or missing model validation")
    output = _fs(output).resolve()
    if output.exists():
        _fail("qualification output already exists and cannot be overwritten")
    toolchain = _object(toolchain_manifest)
    project_root = Path(str(toolchain.get("project_root", ""))).resolve()
    provenance = _assert_live_environment(toolchain, project_root)
    if (profile_id == "clean-room-release-v1") != (clean_room_report is not None):
        _fail("only clean-room profile requires a full gate report")
    if profile_id == "real-model-single-gpu-v1" and gpu_failure_evidence is None:
        _fail("GPU profile requires actual allocator OOM failure evidence")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        _fail("qualification staging output already exists")
    temporary.mkdir(parents=True)
    (temporary / "INCOMPLETE").write_text("reference_qualification_in_progress\n", encoding="ascii")
    try:
        for source, relative_destination in ((toolchain_manifest, "toolchain.json"), (junit, "junit.xml"), (quality, "quality.json"), (sampler_fixture, "sampling/known-answer.json")):
            target = temporary / relative_destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        _write(temporary / "source-provenance.json", provenance)
        locks = temporary / "locks"
        locks.mkdir()
        for name in BASE_LOCKS:
            shutil.copy2(project_root / name, locks / name)
        shutil.copy2(project_root / "pyproject.toml", locks / "pyproject.toml")
        (temporary / "tiny-fixture").mkdir()
        for name in TINY_FILES:
            shutil.copy2(project_root / "fixtures" / "tiny-decoder" / name, temporary / "tiny-fixture" / name)
        runtime_id = str(toolchain.get("runtime_profile_id", ""))
        if runtime_id not in RUNTIME_LOCKS:
            _fail("selected runtime profile is unsupported")
        shutil.copy2(project_root / "runtime" / "artifacts" / f"{runtime_id}.json", locks / "runtime-lock.json")
        for source, destination in ((capture, temporary / "capture"), (benchmark, temporary / "benchmark")):
            _inventory(source)
            shutil.copytree(source, destination)
        for index, source in enumerate(validation_directories):
            _inventory(source)
            shutil.copytree(source, temporary / "validation" / f"model-{index}")
        if gpu_failure_evidence is not None:
            _inventory(gpu_failure_evidence)
            shutil.copytree(gpu_failure_evidence, temporary / "gpu-failure")
        _write(temporary / "sampling" / "verification.json", _sampler_report(temporary / "sampling" / "known-answer.json"))
        if clean_room_report is not None:
            report = _object(clean_room_report)
            _write(temporary / "gate" / "gate-report.json", report)
            for command in report.get("commands", []):
                for stream in ("stdout", "stderr"):
                    link = _relative(command.get(f"{stream}_file"))
                    source = clean_room_report.parent / link
                    target = temporary / "gate" / link
                    if source.is_symlink() or not source.resolve().is_relative_to(clean_room_report.parent.resolve()):
                        _fail("clean-room transcript escapes gate root")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        _verify_facts(temporary, profile_id, expected_source_identity)
        (temporary / "INCOMPLETE").unlink()
        files = _inventory(temporary)
        identity = provenance["source_identity"]
        evidence_digest = sha256_json({"schema_version": 1, "profile_id": profile_id, "source_identity": identity, "files": files})
        profile: dict[str, Any] = {"profile_id": profile_id, "result": "pass", "required_for_functional": True,
                                   "environment_manifest_digest": toolchain["manifest_digest"], "evidence_digest": evidence_digest, "reason": None}
        index_record: dict[str, Any] = {"schema_version": 1, "profile_id": profile_id,
                                       "os": toolchain["os"], "arch": toolchain["arch"], "python_version": toolchain["python_version"],
                                       "source_identity": identity, "runtime_profile_id": toolchain["runtime_profile_id"],
                                       "runtime_lock_digest": toolchain["runtime_lock_digest"], "files": files, "evidence_digest": evidence_digest}
        index_record["index_digest"] = sha256_json(index_record)
        _write(temporary / "profile-result.json", profile)
        _write(temporary / "evidence-index.json", index_record)
        (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
        _write(temporary / "SHA256SUMS.json", _inventory(temporary))
        verify_reference_profile_evidence(temporary, expected_source_identity=expected_source_identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return profile
    except Exception:
        (temporary / "INCOMPLETE").write_text("reference_qualification_failed\n", encoding="ascii")
        raise


def _verify_reference_profile_evidence(root: Path, expected_source_identity: Mapping[str, str] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck portable embedded evidence; never access recorded author paths."""
    root = _fs(root)
    if root.is_symlink():
        _fail("qualification root cannot be a symbolic link")
    root = root.resolve()
    if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
        _fail("reference qualification evidence is incomplete", ErrorCode.CAPTURE_INCOMPLETE)
    files = _inventory(root)
    checksums = _hash_map(_object(root / "SHA256SUMS.json"))
    if checksums != {name: digest for name, digest in files.items() if name != "SHA256SUMS.json"}:
        _fail("reference qualification checksum coverage/match is invalid")
    profile, index = _object(root / "profile-result.json"), _object(root / "evidence-index.json")
    if set(profile) != PROFILE_FIELDS or set(index) != INDEX_FIELDS or type(index["schema_version"]) is not int or index["schema_version"] != 1:
        _fail("unsupported reference qualification schema", ErrorCode.UNSUPPORTED_SCHEMA)
    _signed(index, "index_digest")
    profile_id = profile["profile_id"]
    if profile_id not in PROFILES or index["profile_id"] != profile_id or profile["result"] != "pass" or profile["required_for_functional"] is not True or profile["reason"] is not None:
        _fail("reference qualification profile is not a required complete pass")
    facts = {name: digest for name, digest in files.items() if name not in {"SHA256SUMS.json", "COMPLETE", "profile-result.json", "evidence-index.json"}}
    if _hash_map(index["files"]) != facts:
        _fail("qualification index does not cover exact evidence facts")
    toolchain, identity = _verify_facts(root, profile_id, expected_source_identity)
    evidence_digest = sha256_json({"schema_version": 1, "profile_id": profile_id, "source_identity": identity, "files": facts})
    if index["source_identity"] != identity or index["evidence_digest"] != evidence_digest or profile["evidence_digest"] != evidence_digest or profile["environment_manifest_digest"] != toolchain["manifest_digest"] or any(index[name] != toolchain[name] for name in ("os", "arch", "python_version", "runtime_profile_id", "runtime_lock_digest")):
        _fail("qualification index/profile does not bind validated provenance")
    return profile, index


def verify_reference_profile_evidence(root: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Portable fail-closed reader, including malformed/pathological JSON types."""
    try:
        return _verify_reference_profile_evidence(root, expected_source_identity)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration, OverflowError, RecursionError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "reference qualification artifacts are malformed", {"exception_type": type(exc).__name__}) from exc

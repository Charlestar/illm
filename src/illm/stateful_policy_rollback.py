"""Portable policy-aware V0.2 -> immutable V0.1 rollback evidence.

The public reader only verifies an actual, source-bound CPU rollback component.
It never executes an author path and never grants global or GPU qualification.
Checksums are integrity receipts, not signatures against a malicious author.
Historical rollback-v1 evidence keeps its original reader and interpretation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
from typing import Any, Mapping, NoReturn, Sequence, cast
import uuid

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.bundle import CaptureBundle
from illm.capture.dispatch import open_capture
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.capture.reference import ReferenceCaptureBundle
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.policy_capture_contract import PROTOCOL_SHA256, validate_execution_binding
from illm.reference_qualification import _fs, _object, _signed, _verify_source, _verify_toolchain
from illm.reference_release import verify_reference_release_bundle
from illm.stateful_qualification import _verify_v02_source


REPORT_KIND = "illm-stateful-policy-rollback-v2"
REPORT_SCHEMA_VERSION = 2
OLD_TAG = "v0.1.0"
OLD_COMMIT = "8df86f2ab0b0f3e0c67bde68f65a9aa7f4adfa02"
OLD_TAG_OBJECT = "0cebfd3ea1730db6f89ee3bc68579064a0534f80"
OLD_BUNDLE = "1e87e9cdf10d66a55a4a3231e87b96e97d0dfe0d7ed6a138fc0fc265ec9cdbac"
OLD_SOURCE = {
    "project_root_digest": "3b0de700bfa4e7e43e9aaed9765dc1ad1274d1f0c33f86015105d00629488c09",
    "bootstrap_digest": "1a090333ad37390fbd9dce5691d0176996eec8b53b61a5b3b116d6ce34fdfdd9",
    "base_lock_digest": "19489d7dd52ec0f54bbd45b784add2453453523008ab46550dea8bf989cef274",
}

CANDIDATE_REQUEST = "rollback-v2-candidate-c"
CANDIDATE_REJECTED_REQUEST = "rollback-v2-candidate-after-close"
LEGACY_REQUEST = "rollback-v2-legacy-a"
LEGACY_REJECTED_REQUEST = "rollback-v2-legacy-after-close"
SCHEMA2_REQUEST = "rollback-v2-historical-schema2"
OLD_REQUEST = "rollback-v2-old-v01"
EXPECTED_INPUT_TOKEN_IDS = [1, 107, 104, 111, 111, 114]
EXPECTED_OUTPUT_TOKEN_IDS = [114] * 8
EXPECTED_TEXT = "oooooooo"

RESOURCE_NAMES = (
    "active_requests",
    "reservations",
    "quarantined_requests",
    "active_tickets",
    "input_leases",
    "output_leases",
    "workspace_leases",
    "kv_write_leases",
    "kv_reserved_bytes",
    "kv_valid_tokens",
    "kv_staged_tokens",
)
RESOURCE_FIELDS = frozenset(RESOURCE_NAMES)

CANDIDATE_CONFIG: dict[str, Any] = {
    "model": "fixture:tiny-decoder",
    "device": "cpu",
    "prompt": "hello",
    "max_new_tokens": 8,
    "sampling_mode": "greedy",
    "seed": 7,
    "request_id": CANDIDATE_REQUEST,
    "kv_mode": "contiguous",
    "numerical_policy": "candidate-fp32",
    "telemetry": "request",
}
LEGACY_CONFIG: dict[str, Any] = {
    **CANDIDATE_CONFIG,
    "request_id": LEGACY_REQUEST,
    "kv_mode": "none",
    "numerical_policy": "legacy-fp32",
}
SCHEMA2_CONFIG: dict[str, Any] = {
    **CANDIDATE_CONFIG,
    "request_id": SCHEMA2_REQUEST,
    "numerical_policy": None,
}
TOY_CONFIG: dict[str, Any] = {"seed": 7, "steps": 4}
OLD_CONFIG: dict[str, Any] = {
    "model": "fixture:tiny-decoder",
    "device": "cpu",
    "prompt": "hello",
    "max_new_tokens": 8,
    "sampling_mode": "greedy",
    "seed": 7,
    "request_id": OLD_REQUEST,
    "telemetry": "request",
}
CONFIGS: dict[str, Any] = {
    "candidate_c": CANDIDATE_CONFIG,
    "legacy_a": LEGACY_CONFIG,
    "historical_schema2": SCHEMA2_CONFIG,
    "toy_schema0": TOY_CONFIG,
    "old_v01": OLD_CONFIG,
}

CAPTURE_PATHS = {
    "candidate_c": "captures/candidate-c",
    "legacy_a": "captures/legacy-a",
    "historical_schema2": "captures/historical-schema2",
    "toy_schema0": "captures/toy-schema0",
    "old_schema1": "captures/old-schema1",
}

PHASES = (
    "resolve-tag",
    "archive",
    "current-transition",
    "new-create-schema0",
    "new-read-schema0",
    "new-run-schema2",
    "new-read-schema2",
    "new-read-schema3-candidate",
    "new-read-schema3-legacy",
    "bootstrap-old",
    "old-environment",
    "old-generate-schema1",
    "old-replay-schema1-trace",
    "old-replay-schema1-state",
    "old-bundle-verify",
    "old-reject-schema2",
    "old-reject-schema3",
    "new-read-schema1-state",
)

COMPATIBILITY = {
    "new_reader": {"schema0": "R0", "schema1": "R1_REFERENCE", "schema2": "R0", "schema3": "R0"},
    "old_reader": {"schema1": "R0+R1_REFERENCE", "schema2": "UNSUPPORTED_SCHEMA", "schema3": "UNSUPPORTED_SCHEMA"},
}

REPORT_FIELDS = frozenset({
    "schema_version", "report_kind", "status", "evidence_mode", "rollback_kind",
    "qualification_status", "global_qualification_granted", "gpu_qualification_granted",
    "source_identity", "protocol_sha256", "current_toolchain_manifest_digest",
    "current_environment_id", "execution_binding", "execution_binding_digest",
    "trigger_reason", "last_safe_request_id", "from_policy", "from_policy_digest",
    "target_policy", "target_policy_digest", "configs", "configs_digest", "captures",
    "safe_boundary", "compatibility", "old_runtime_identity", "original_sources_unchanged",
    "unrelated_processes_interrupted", "files", "report_digest",
})

MANIFEST_FIELDS = frozenset({
    "schema_version", "report_kind", "source_identity", "protocol_sha256",
    "current_toolchain_manifest_digest", "current_environment_id", "from_policy_digest",
    "target_policy_digest", "configs_digest", "primary_capture_manifest_digest",
    "old_release_tag", "old_release_commit", "old_release_tag_object",
    "old_bundle_manifest_digest",
})

COMMAND_FIELDS = frozenset({
    "phase", "argv", "returncode", "stdout_file", "stderr_file",
    "stdout_sha256", "stderr_sha256",
})

OLD_RUNTIME_FIELDS = frozenset({
    "identity_kind", "release_tag", "release_commit", "release_tag_object", "source_identity",
    "release_bundle_manifest_digest", "toolchain_manifest_digest", "environment_id",
    "runtime_profile_id", "python_version", "installed_distribution", "stateful_engine_present",
    "numerical_policy_module_present", "reported_numerical_policy",
})

MAX_BYTES = 256 * 1024 * 1024


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _same(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _exact_int(value: object, *, expected: int, field: str) -> int:
    if type(value) is not int or value != expected:
        _fail(f"policy rollback {field} must be the exact integer {expected}")
    return value


def _is_linklike(path: Path) -> bool:
    metadata = path.lstat()
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _claim_casefold(name: str, folded: set[str]) -> None:
    alias = name.casefold()
    if alias in folded:
        _fail("policy rollback contains a case-colliding path")
    folded.add(alias)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        _fail("policy rollback link is not portable")
    path = PurePosixPath(value)
    reserved = {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)]}
    if (path.is_absolute() or path.as_posix() != value
            or any(part in {".", ".."} or part.rstrip(" .") != part
                   or part.split(".")[0].upper() in reserved for part in path.parts)):
        _fail("policy rollback link escapes its sealed root")
    return value


def _author_path(value: object) -> str:
    """Normalize a recorded author path lexically without opening it."""

    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("policy rollback author path is invalid")
    normalized = value.replace("\\", "/")
    if normalized.startswith("//?/"):
        normalized = normalized[4:]
    path = PureWindowsPath(normalized) if re.match(r"^[A-Za-z]:/", normalized) else PurePosixPath(normalized)
    if (not path.is_absolute() or any(part in {".", ".."} for part in path.parts)
            or path.as_posix() != normalized or normalized.endswith("/")):
        _fail("policy rollback author path must be absolute and canonical")
    return normalized


def portable_stateful_policy_rollback_inventory(root: Path) -> dict[str, str]:
    """Inventory portable facts while omitting only explicit local build work."""

    scan = _fs(Path(os.path.abspath(root)))
    if _is_linklike(scan) or not scan.is_dir():
        _fail("policy rollback root must be an ordinary directory")
    resolved = scan.resolve()
    result: dict[str, str] = {}
    folded: set[str] = set()
    total = 0
    stack = list(scan.iterdir())
    while stack:
        path = stack.pop()
        if _is_linklike(path) or not path.resolve().is_relative_to(resolved):
            _fail("policy rollback forbids links, reparse points and escaped artifacts")
        name = _relative(path.relative_to(scan).as_posix())
        _claim_casefold(name, folded)
        metadata = path.stat()
        if stat.S_ISDIR(metadata.st_mode):
            if path.parent == scan and path.name == "local-only":
                continue
            children = list(path.iterdir())
            if not children:
                _fail("policy rollback contains an unreferenced empty directory")
            stack.extend(children)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail("policy rollback contains a nonregular artifact")
        if int(metadata.st_nlink) != 1:
            _fail("policy rollback forbids hard-linked artifacts")
        total += metadata.st_size
        if total > MAX_BYTES:
            _fail("policy rollback exceeds its portable byte budget")
        if name not in {"checksums.json", "COMPLETE", "INCOMPLETE"}:
            result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _last_json(path: Path) -> dict[str, Any]:
    lines = [line for line in _fs(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        _fail("policy rollback command has no structured output")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in policy rollback transcript")
            result[key] = value
        return result

    value = json.loads(lines[-1], object_pairs_hook=unique,
                       parse_constant=lambda _: _fail("nonfinite policy rollback transcript"))
    if not isinstance(value, dict):
        _fail("policy rollback transcript must end in an object")
    return value


def _zero_resources(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != RESOURCE_FIELDS:
        _fail("policy rollback resource snapshot has the wrong closed field set")
    if any(type(item) is not int or item != 0 for item in value.values()):
        _fail("policy rollback resource snapshot is not the exact zero baseline")
    return cast(dict[str, int], value)


def _capture_receipt(name: str, bundle: Any) -> dict[str, Any]:
    expected_path = CAPTURE_PATHS[name]
    schema = bundle.manifest.get("schema_version")
    result: dict[str, Any] = {
        "relative_path": expected_path,
        "schema_version": schema,
        "manifest_digest": bundle.manifest.get("manifest_digest"),
    }
    if name in {"candidate_c", "legacy_a"}:
        result["memory_plan_digest"] = bundle.manifest.get("memory_plan_digest")
    return result


def _fence_facts(bundle: PolicyStreamingCaptureBundle, *, candidate: bool) -> dict[str, Any]:
    counts = {
        "writes_started": sum(event.kind is K.KV_WRITE_STARTED for event in bundle.events),
        "writes_committed": sum(event.kind is K.KV_WRITE_COMMITTED for event in bundle.events),
        "writes_aborted": sum(event.kind is K.KV_WRITE_ABORTED for event in bundle.events),
        "releases": sum(event.kind is K.KV_PAGE_RELEASED for event in bundle.events),
    }
    resolved = [event for event in bundle.events if event.kind in {
        K.KV_WRITE_COMMITTED, K.KV_WRITE_ABORTED, K.KV_PAGE_RELEASED,
    }]
    terminals = [event for event in bundle.events if event.kind in {
        K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED,
    }]
    if len(terminals) != 1:
        _fail("policy rollback capture lacks exactly one physical terminal")
    terminal = terminals[0]
    facts = {
        **counts,
        "all_fences_completed": all(event.payload.get("fence_completed") is True for event in resolved),
        "terminal_physical_cleanup_complete": terminal.payload.get("physical_cleanup_complete") is True,
        "terminal_active_leases": terminal.payload.get("active_leases"),
        "terminal_pending_writes": terminal.payload.get("pending_writes"),
    }
    expected = {
        "writes_started": 8 if candidate else 0,
        "writes_committed": 8 if candidate else 0,
        "writes_aborted": 0,
        "releases": 1 if candidate else 0,
        "all_fences_completed": True,
        "terminal_physical_cleanup_complete": True,
        "terminal_active_leases": 0,
        "terminal_pending_writes": 0,
    }
    if facts != expected:
        _fail("policy rollback capture does not prove its exact fence/release boundary")
    return facts


def _validate_policy_result(
    result: Mapping[str, Any],
    bundle: PolicyStreamingCaptureBundle,
    *,
    candidate: bool,
) -> None:
    policy = candidate_policy("fp32") if candidate else legacy_policy("fp32")
    request_id = CANDIDATE_REQUEST if candidate else LEGACY_REQUEST
    expected_query = [6, 1, 1, 1, 1, 1, 1, 1] if candidate else list(range(6, 14))
    expected_valid = list(range(6, 14)) if candidate else [0] * 8
    if (type(result.get("schema_version")) is not int or result.get("schema_version") != 3
            or result.get("request_id") != request_id or result.get("outcome") != "FINISHED"
            or result.get("finish_reason") != "MAX_NEW_TOKENS" or result.get("error_code") is not None
            or result.get("input_token_ids") != EXPECTED_INPUT_TOKEN_IDS
            or result.get("output_token_ids") != EXPECTED_OUTPUT_TOKEN_IDS or result.get("text") != EXPECTED_TEXT
            or result.get("actual_seed") != 7 or result.get("next_output_position") != 8
            or result.get("rng_draws_consumed") != 0 or result.get("committed_draw_addresses") != []
            or result.get("kv_mode") != ("contiguous" if candidate else "none")
            or result.get("query_tokens_per_step") != expected_query
            or result.get("kv_valid_tokens_per_step") != expected_valid
            or result.get("capture_error") is not None
            or result.get("capture_manifest_digest") != bundle.manifest["manifest_digest"]
            or result.get("numerical_policy") != policy.to_dict()
            or result.get("numerical_policy_digest") != policy.digest
            or result.get("execution_binding") != bundle.manifest["execution_binding"]
            or result.get("execution_binding_digest") != bundle.manifest["execution_binding_digest"]
            or result.get("memory_plan_digest") != bundle.manifest["memory_plan_digest"]):
        _fail("policy rollback current result differs from its fixed policy/tokens/state/capture contract")
    _zero_resources(result.get("resource_status"))


def _safe_segment(
    value: object,
    result: Mapping[str, Any],
    bundle: PolicyStreamingCaptureBundle,
    *,
    candidate: bool,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "result_resource_status", "engine_status_before_close", "engine_close_returned",
        "engine_status_after_close", "terminal_outcome", "fence_facts", "post_close_admission",
    }:
        _fail("policy rollback safe boundary segment has the wrong closed field set")
    for field in ("result_resource_status", "engine_status_before_close", "engine_status_after_close"):
        _zero_resources(value[field])
    if value["result_resource_status"] != result["resource_status"] or value["engine_close_returned"] is not True:
        _fail("policy rollback engine did not close from its exact zero resource baseline")
    if value["terminal_outcome"] != "FINISHED" or value["fence_facts"] != _fence_facts(bundle, candidate=candidate):
        _fail("policy rollback safe boundary differs from replayed terminal/fence facts")
    rejection = value["post_close_admission"]
    expected_id = CANDIDATE_REJECTED_REQUEST if candidate else LEGACY_REJECTED_REQUEST
    if rejection != {
        "request_id": expected_id,
        "accepted": False,
        "error_code": "INVALID_ARGUMENT",
        "reason": "engine is closed",
    }:
        _fail("policy rollback did not prove admission closed after safe cleanup")


def _verify_commands(
    commands: Sequence[Mapping[str, Any]],
    new: Mapping[str, Any],
    old: Mapping[str, Any],
) -> None:
    if tuple(item.get("phase") for item in commands) != PHASES:
        _fail("policy rollback lacks the exact executed phase order")
    argv: dict[str, list[str]] = {}
    current_python = _author_path(new.get("python_executable"))
    for item in commands:
        if set(item) != COMMAND_FIELDS:
            _fail("policy rollback command record has an unknown or missing field")
        tokens = item.get("argv")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) and token for token in tokens):
            _fail("policy rollback lacks actual command argv")
        argv[str(item["phase"])] = [token.replace("\\", "/") for token in tokens]
        if item["phase"] not in {"resolve-tag", "archive"} and _author_path(tokens[0]) != current_python:
            _fail("policy rollback wrapper did not use the current bound Python interpreter")
    if argv["resolve-tag"][0] != argv["archive"][0] or PurePosixPath(argv["archive"][0]).name.lower() not in {"git", "git.exe"}:
        _fail("policy rollback old source was not resolved by one Git executable")
    if argv["resolve-tag"][1:] != ["rev-parse", f"refs/tags/{OLD_TAG}", f"refs/tags/{OLD_TAG}^{{commit}}"]:
        _fail("policy rollback did not resolve the immutable annotated tag")
    archive = argv["archive"]
    if len(archive) != 6 or archive[1:4] != ["archive", "--format=zip", "--output"] or archive[-1] != OLD_COMMIT:
        _fail("policy rollback archive is not the immutable old source")
    zip_path = _author_path(archive[4])
    suffix = "/local-only/v0.1.0.zip"
    if not zip_path.endswith(suffix):
        _fail("policy rollback archive is not isolated local work")
    output = zip_path[:-len(suffix)]
    checkout = output + "/local-only/v01"
    old_manifest = checkout + "/out/toolchain.json"
    old_digest = checkout + "/out/toolchain.sha256"
    new_root = _author_path(new.get("project_root"))
    if _author_path(old.get("project_root")) != checkout or new_root == checkout:
        _fail("policy rollback did not install a separate old checkout")
    old_runtime = old.get("runtime_profile_id")
    if old_runtime not in {"py312-win-amd64-cpu", "py312-linux-x86_64-cpu"}:
        _fail("policy rollback old runtime is not a pinned host CPU profile")
    old_environment = checkout + "/.illm/envs/rb01-policy"
    python_suffix = "/Scripts/python.exe" if old_runtime == "py312-win-amd64-cpu" else "/bin/python"
    if (old.get("profile") != "rb01-policy" or _author_path(old.get("environment_root")) != old_environment
            or _author_path(old.get("python_executable")) != old_environment + python_suffix
            or _author_path(old.get("python_executable")) == current_python):
        _fail("policy rollback old interpreter is not its fresh isolated environment")
    bootstrap = argv["bootstrap-old"]
    expected_bootstrap = [checkout + "/scripts/bootstrap.py", "--profile", "rb01-policy", "--runtime-profile", "v01-cpu",
                          "--locked", "--offline", "--artifact-cache"]
    if (len(bootstrap) != 14 or bootstrap[1:9] != expected_bootstrap
            or bootstrap[10:] != ["--write-manifest", old_manifest, "--write-manifest-digest", old_digest]):
        _fail("policy rollback old bootstrap bypasses source-bound offline installation")
    _author_path(bootstrap[9])

    current_manifest: str | None = None
    current_digest = output + "/current-toolchain.sha256"
    script = new_root + "/scripts/rehearse_policy_rollback.py"
    candidate = output + "/captures/candidate-c"
    legacy = output + "/captures/legacy-a"
    schema0 = output + "/captures/toy-schema0"
    schema2 = output + "/captures/historical-schema2"
    schema1 = output + "/captures/old-schema1"
    old_capture = checkout + "/out/policy-rollback-schema1"
    expected: dict[str, list[str]] = {
        "current-transition": ["python", script, "--internal-current-transition", "--output-root", output,
                               "--candidate-policy", "candidate-fp32", "--candidate-kv-mode", "contiguous",
                               "--legacy-policy", "legacy-fp32", "--legacy-kv-mode", "none",
                               "--prompt", "hello", "--max-new-tokens", "8", "--seed", "7"],
        "new-create-schema0": ["python", "-m", "illm", "demo", "toy-scheduler", "--seed", "7", "--steps", "4", "--capture", schema0],
        "new-read-schema0": ["python", "-m", "illm", "replay", schema0, "--mode", "trace"],
        "new-run-schema2": ["python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
                            "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                            "--request-id", SCHEMA2_REQUEST, "--kv-mode", "contiguous", "--telemetry", "request", "--capture", schema2],
        "new-read-schema2": ["python", "-m", "illm", "replay", schema2, "--mode", "trace"],
        "new-read-schema3-candidate": ["python", "-m", "illm", "replay", candidate, "--mode", "trace"],
        "new-read-schema3-legacy": ["python", "-m", "illm", "replay", legacy, "--mode", "trace"],
        "old-environment": ["python", "-I", "-c", OLD_ENVIRONMENT_PROBE],
        "old-generate-schema1": ["python", "-m", "illm", "generate", "--model", "fixture:tiny-decoder", "--device", "cpu",
                                 "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                                 "--request-id", OLD_REQUEST, "--telemetry", "request", "--capture", old_capture],
        "old-replay-schema1-trace": ["python", "-m", "illm", "replay", old_capture, "--mode", "trace"],
        "old-replay-schema1-state": ["python", "-m", "illm", "replay", old_capture, "--mode", "state", "--assert-state-digest"],
        "old-bundle-verify": ["python", "-m", "illm", "release", "verify", checkout + "/release/v0.1.0/manifest.json"],
        "old-reject-schema2": ["python", "-m", "illm", "replay", schema2, "--mode", "trace"],
        "old-reject-schema3": ["python", "-m", "illm", "replay", candidate, "--mode", "trace"],
        "new-read-schema1-state": ["python", "-m", "illm", "replay", schema1, "--mode", "state", "--assert-state-digest"],
    }
    for name, child in expected.items():
        tokens = argv[name]
        is_old = name.startswith("old-")
        source_root = checkout if is_old else new_root
        digest_file = old_digest if is_old else current_digest
        if len(tokens) < 8 or tokens[1:3] != [source_root + "/scripts/in_env.py", "--manifest"] or tokens[4:7] != ["--expected-manifest-digest-file", digest_file, "--"]:
            _fail("policy rollback command bypasses its source-bound manifest wrapper")
        manifest = _author_path(tokens[3])
        if is_old:
            if manifest != old_manifest:
                _fail("policy rollback old command entered a different toolchain")
        elif current_manifest is None:
            current_manifest = manifest
        elif current_manifest != manifest:
            _fail("policy rollback current commands entered different toolchains")
        if tokens[7:] != child:
            _fail("policy rollback executed a different command than its claimed phase")


OLD_ENVIRONMENT_PROBE = (
    "import importlib.metadata as m,importlib.util as u,json,os,pathlib,platform;"
    "p=json.loads(pathlib.Path(os.environ['ILLM_TOOLCHAIN_MANIFEST']).read_text(encoding='utf-8'));"
    "\ntry:n=u.find_spec('illm.numerical_policy') is not None\nexcept ModuleNotFoundError:n=False\n"
    "print(json.dumps({'illm_version':m.version('illm'),'stateful_engine_present':u.find_spec('illm.engine.streaming') is not None,"
    "'numerical_policy_module_present':n,'reported_numerical_policy':None,'environment_id':os.environ.get('ILLM_ENVIRONMENT_ID'),"
    "'toolchain_digest':p['manifest_digest'],'runtime_profile_id':p['runtime_profile_id'],'python_version':platform.python_version()},sort_keys=True))"
)


def _old_runtime_identity(probe: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "identity_kind": "source-built-installed-runtime",
        "release_tag": OLD_TAG,
        "release_commit": OLD_COMMIT,
        "release_tag_object": OLD_TAG_OBJECT,
        "source_identity": OLD_SOURCE,
        "release_bundle_manifest_digest": OLD_BUNDLE,
        "toolchain_manifest_digest": old["manifest_digest"],
        "environment_id": old["environment_id"],
        "runtime_profile_id": old["runtime_profile_id"],
        "python_version": old["python_version"],
        "installed_distribution": {"name": "illm", "version": "0.1.0"},
        "stateful_engine_present": False,
        "numerical_policy_module_present": False,
        "reported_numerical_policy": None,
    }


def _verify(root: Path, *, synthetic_test_fixture: bool = False) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir() or not _fs(root / "COMPLETE").is_file() or _fs(root / "INCOMPLETE").exists():
        _fail("policy rollback evidence is incomplete")
    root = _fs(root).resolve()
    inventory = portable_stateful_policy_rollback_inventory(root)
    report = _object(root / "rollback-report.json")
    if set(report) != REPORT_FIELDS:
        _fail("policy rollback report has an unknown or missing field")
    _signed(report, "report_digest")
    expected_mode, expected_status = (("synthetic_test_fixture", "synthetic_pass")
                                      if synthetic_test_fixture else ("release", "pass"))
    if (type(report.get("schema_version")) is not int or report["schema_version"] != REPORT_SCHEMA_VERSION
            or report.get("report_kind") != REPORT_KIND or report.get("status") != expected_status
            or report.get("evidence_mode") != expected_mode
            or report.get("rollback_kind") != "isolated-cpu-policy-rollback-rehearsal"
            or report.get("qualification_status") != "unqualified"
            or report.get("global_qualification_granted") is not False
            or report.get("gpu_qualification_granted") is not False):
        _fail("policy rollback report overclaims or has the wrong version/mode")
    if (root / "COMPLETE").read_text(encoding="ascii") != report["report_digest"] + "\n":
        _fail("policy rollback completion marker is stale")
    if _object(root / "checksums.json") != inventory:
        _fail("policy rollback sealed inventory is not exact")
    if report.get("files") != {key: value for key, value in inventory.items() if key != "rollback-report.json"}:
        _fail("policy rollback report inventory differs from the sealed files")

    source, versions = _verify_v02_source(root / "new-source", None)
    new = _verify_toolchain(root / "new-source", source, versions)
    old_source, old_versions = _verify_source(root / "old-source", OLD_SOURCE)
    old = _verify_toolchain(root / "old-source", old_source, old_versions)
    if not synthetic_test_fixture:
        for toolchain in (new, old):
            value = toolchain.get("environment_id")
            try:
                identifier = uuid.UUID(value) if isinstance(value, str) else None
            except ValueError:
                identifier = None
            if identifier is None or str(identifier) != value or identifier.version != 4:
                _fail("policy rollback actual runtime lacks a fresh UUID4 identity")
    if (old["installed_packages"].get("illm") != "0.1.0" or new["environment_id"] == old["environment_id"]
            or report.get("source_identity") != source
            or report.get("current_toolchain_manifest_digest") != new["manifest_digest"]
            or report.get("current_environment_id") != new["environment_id"]):
        _fail("policy rollback source/toolchain identities disagree")

    opened = {name: open_capture(root / path) for name, path in CAPTURE_PATHS.items()}
    candidate_capture = opened["candidate_c"]
    legacy_capture = opened["legacy_a"]
    schema2_capture = opened["historical_schema2"]
    schema0_capture = opened["toy_schema0"]
    old_capture = opened["old_schema1"]
    if (not isinstance(candidate_capture, PolicyStreamingCaptureBundle)
            or not isinstance(legacy_capture, PolicyStreamingCaptureBundle)
            or not isinstance(schema2_capture, StreamingCaptureBundle)
            or not isinstance(schema0_capture, CaptureBundle)
            or not isinstance(old_capture, ReferenceCaptureBundle)):
        _fail("policy rollback capture versions were relabeled or dispatched incorrectly")
    expected_captures = {
        name: _capture_receipt(name, opened[name]) for name in CAPTURE_PATHS
    }
    if report.get("captures") != expected_captures:
        _fail("policy rollback capture receipts differ from fully verified bundles")

    candidate = candidate_policy("fp32")
    legacy = legacy_policy("fp32")
    if (report.get("protocol_sha256") != PROTOCOL_SHA256
            or report.get("from_policy") != candidate.to_dict() or report.get("from_policy_digest") != candidate.digest
            or report.get("target_policy") != legacy.to_dict() or report.get("target_policy_digest") != legacy.digest
            or report.get("configs") != CONFIGS or report.get("configs_digest") != sha256_json(CONFIGS)):
        _fail("policy rollback from/target/config identity is not the frozen CPU contract")
    for bundle, policy, mode in ((candidate_capture, candidate, "contiguous"), (legacy_capture, legacy, "none")):
        if (bundle.manifest["numerical_policy"] != policy.to_dict()
                or bundle.manifest["numerical_policy_digest"] != policy.digest
                or bundle.manifest["memory_plan"]["kv_mode"] != mode
                or bundle.manifest["memory_plan"]["device"] != "cpu"
                or bundle.manifest["memory_plan"]["runtime_dtype"] != "fp32"):
            _fail("policy rollback schema3 capture has a different policy/MemoryPlan")
    binding = validate_execution_binding(cast(Mapping[str, Any], report.get("execution_binding")))
    if (report.get("execution_binding_digest") != sha256_json(binding)
            or binding != candidate_capture.manifest["execution_binding"]
            or binding != legacy_capture.manifest["execution_binding"]
            or binding["implementation_source_digest"] != source["project_root_digest"]
            or binding["protocol_sha256"] != PROTOCOL_SHA256
            or binding["toolchain"] != {"binding": "source-bound-manifest", "manifest_digest": new["manifest_digest"],
                                              "environment_id": new["environment_id"], "lock_digest": new["lock_digest"]}):
        _fail("policy rollback ExecutionBindingV1 differs across source/toolchain/captures")

    execution = _object(root / "execution.json")
    if set(execution) != {"schema_version", "report_kind", "commands"} or execution.get("schema_version") != 2 or execution.get("report_kind") != REPORT_KIND:
        _fail("policy rollback execution journal has an unsupported schema")
    commands = execution.get("commands")
    if not isinstance(commands, list) or not all(isinstance(item, dict) for item in commands):
        _fail("policy rollback lacks actual execution records")
    _verify_commands(cast(list[dict[str, Any]], commands), new, old)
    outputs: dict[str, dict[str, Any]] = {}
    for item in commands:
        name = item["phase"]
        expected_code = 78 if name in {"old-reject-schema2", "old-reject-schema3"} else 0
        if type(item.get("returncode")) is not int or item["returncode"] != expected_code:
            _fail("policy rollback command failed or a required rejection was absent")
        for stream in ("stdout", "stderr"):
            link = _relative(item.get(stream + "_file"))
            if link != f"logs/{name}.{stream}.txt" or item.get(stream + "_sha256") != sha256_file(root / link):
                _fail("policy rollback transcript checksum/link mismatch")
        if name not in {"resolve-tag", "archive"}:
            outputs[name] = _last_json(root / item["stderr_file" if expected_code else "stdout_file"])
    if (root / "logs/resolve-tag.stdout.txt").read_text(encoding="ascii").splitlines() != [OLD_TAG_OBJECT, OLD_COMMIT]:
        _fail("policy rollback did not resolve the fixed old annotated tag")
    if outputs["bootstrap-old"] != old:
        _fail("policy rollback old bootstrap stdout differs from embedded toolchain")

    transition = outputs["current-transition"]
    if (set(transition) != {"schema_version", "report_kind", "candidate_c", "legacy_a"}
            or transition.get("report_kind") != "illm-policy-rollback-current-transition-v1"):
        _fail("policy rollback current transition receipt has an unsupported schema")
    _exact_int(transition.get("schema_version"), expected=1, field="current transition schema_version")
    for name, bundle, is_candidate in (("candidate_c", candidate_capture, True), ("legacy_a", legacy_capture, False)):
        segment = transition[name]
        if not isinstance(segment, dict) or set(segment) != {"result", "engine_status_before_close", "engine_close_returned", "engine_status_after_close", "post_close_admission"}:
            _fail("policy rollback current transition segment has an unknown field")
        result = segment["result"]
        if not isinstance(result, dict):
            _fail("policy rollback current transition result must be an object")
        _validate_policy_result(result, bundle, candidate=is_candidate)
        expected_safe = {
            "result_resource_status": result["resource_status"],
            "engine_status_before_close": segment["engine_status_before_close"],
            "engine_close_returned": segment["engine_close_returned"],
            "engine_status_after_close": segment["engine_status_after_close"],
            "terminal_outcome": result["outcome"],
            "fence_facts": _fence_facts(bundle, candidate=is_candidate),
            "post_close_admission": segment["post_close_admission"],
        }
        safe = report.get("safe_boundary")
        if not isinstance(safe, dict) or safe.get(name) != expected_safe:
            _fail("policy rollback report safe boundary differs from raw current transition")
        _safe_segment(expected_safe, result, bundle, candidate=is_candidate)
    safe_boundary = report["safe_boundary"]
    if set(safe_boundary) != {"candidate_c", "legacy_a", "new_requests_after_last_safe_terminal"}:
        _fail("policy rollback safe boundary lacks the closed admission count")
    _exact_int(safe_boundary["new_requests_after_last_safe_terminal"], expected=0,
               field="new_requests_after_last_safe_terminal")
    if report.get("last_safe_request_id") != LEGACY_REQUEST:
        _fail("policy rollback last safe request is not explicit A+none")

    if not _same(outputs["new-read-schema3-candidate"], candidate_capture.request_story(CANDIDATE_REQUEST)):
        _fail("new reader did not replay candidate schema3")
    if not _same(outputs["new-read-schema3-legacy"], legacy_capture.request_story(LEGACY_REQUEST)):
        _fail("new reader did not replay legacy schema3")
    schema2_result = outputs["new-run-schema2"]
    if (schema2_result.get("schema_version") != 2 or schema2_result.get("request_id") != SCHEMA2_REQUEST
            or "numerical_policy" in schema2_result or "numerical_policy_digest" in schema2_result
            or schema2_result.get("capture_manifest_digest") != schema2_capture.manifest["manifest_digest"]
            or schema2_result.get("outcome") != "FINISHED"
            or schema2_result.get("finish_reason") != "MAX_NEW_TOKENS"
            or schema2_result.get("input_token_ids") != EXPECTED_INPUT_TOKEN_IDS
            or schema2_result.get("output_token_ids") != EXPECTED_OUTPUT_TOKEN_IDS
            or schema2_result.get("text") != EXPECTED_TEXT
            or schema2_result.get("query_tokens_per_step") != [6, 1, 1, 1, 1, 1, 1, 1]
            or schema2_result.get("kv_valid_tokens_per_step") != list(range(6, 14))
            or not _same(outputs["new-read-schema2"], schema2_capture.request_story(SCHEMA2_REQUEST))):
        _fail("historical schema2 was relabeled with a numerical policy or not replayed")
    _zero_resources(schema2_result.get("resource_status"))
    toy_trace = outputs["new-read-schema0"]
    if (toy_trace.get("mode") != "R0" or toy_trace.get("total_events") != len(schema0_capture.events)
            or toy_trace.get("displayed_events") != len(schema0_capture.events)):
        _fail("new reader did not replay schema0")

    probe = outputs["old-environment"]
    expected_old_identity = _old_runtime_identity(probe, old)
    if (set(probe) != {"illm_version", "stateful_engine_present", "numerical_policy_module_present",
                       "reported_numerical_policy", "environment_id", "toolchain_digest", "runtime_profile_id", "python_version"}
            or probe.get("illm_version") != "0.1.0" or probe.get("stateful_engine_present") is not False
            or probe.get("numerical_policy_module_present") is not False or probe.get("reported_numerical_policy") is not None
            or probe.get("environment_id") != old["environment_id"] or probe.get("toolchain_digest") != old["manifest_digest"]
            or probe.get("runtime_profile_id") != old["runtime_profile_id"] or probe.get("python_version") != old["python_version"]
            or set(expected_old_identity) != OLD_RUNTIME_FIELDS or report.get("old_runtime_identity") != expected_old_identity):
        _fail("policy rollback old source-built runtime identity is not independently bound")

    old_result = outputs["old-generate-schema1"]
    if any(field in old_result for field in ("numerical_policy", "numerical_policy_digest", "execution_binding", "execution_binding_digest")):
        _fail("policy rollback invented a numerical policy for the old binary")
    candidate_result = transition["candidate_c"]["result"]
    legacy_result = transition["legacy_a"]["result"]
    common = ("input_token_ids", "output_token_ids", "text", "finish_reason", "actual_seed", "model_spec_id",
              "sampling_spec_digest", "next_output_position", "rng_draws_consumed", "committed_draw_addresses")
    if (type(old_result.get("schema_version")) is not int or old_result.get("schema_version") != 1
            or old_result.get("request_id") != OLD_REQUEST
            or old_result.get("capture_manifest_digest") != old_capture.manifest["manifest_digest"]
            or old_result.get("finish_reason") != "MAX_NEW_TOKENS"
            or old_result.get("input_token_ids") != EXPECTED_INPUT_TOKEN_IDS
            or old_result.get("output_token_ids") != EXPECTED_OUTPUT_TOKEN_IDS or old_result.get("text") != EXPECTED_TEXT
            or any(not _same(candidate_result.get(field), legacy_result.get(field))
                   or not _same(candidate_result.get(field), old_result.get(field)) for field in common)):
        _fail("candidate C, explicit A and independent V0.1 differ on the full frozen request/result")
    for field in ("model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest"):
        values = [candidate_capture.manifest["identity"].get(field), legacy_capture.manifest["identity"].get(field),
                  old_capture.manifest["identity"].get(field)]
        if (any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values)
                or len(set(cast(list[str], values))) != 1):
            _fail("policy rollback paths use different model/weight/tokenizer identities")

    old_state = {
        "mode": "R1_REFERENCE", "asserted": True, "numerical_reexecution": False,
        "final_state_digest": old_capture.manifest["final_state_digest"], "status": "FINISHED",
        "output_tokens": old_capture.final_state["output_tokens"], "rng_draws_consumed": 0,
    }
    old_trace = outputs["old-replay-schema1-trace"]
    if (old_trace.get("mode") != "R0" or old_trace.get("total_events") != len(old_capture.events)
            or old_trace.get("displayed_events") != len(old_capture.events)
            or not _same(outputs["old-replay-schema1-state"], old_state)
            or not _same(outputs["new-read-schema1-state"], old_state)):
        _fail("old/new readers did not replay the same schema1 facts")
    for name in ("old-reject-schema2", "old-reject-schema3"):
        if outputs[name].get("code") != "UNSUPPORTED_SCHEMA":
            _fail("old reader did not fail closed on schema2 and schema3")
    if report.get("compatibility") != COMPATIBILITY:
        _fail("policy rollback compatibility matrix is incomplete or relabeled")

    old_bundle = verify_reference_release_bundle(root / "old-release-bundle/manifest.json", expected_source_identity=OLD_SOURCE)
    if (old_bundle["manifest_digest"] != OLD_BUNDLE or outputs["old-bundle-verify"] != old_bundle
            or expected_old_identity["release_bundle_manifest_digest"] != OLD_BUNDLE):
        _fail("policy rollback old release bundle is not the immutable published artifact")
    if (not isinstance(report.get("trigger_reason"), str) or not report["trigger_reason"].strip()
            or len(report["trigger_reason"]) > 1000
            or report.get("original_sources_unchanged") is not True
            or type(report.get("unrelated_processes_interrupted")) is not int
            or report.get("unrelated_processes_interrupted") != 0):
        _fail("policy rollback omits its trigger or overclaims service mutation")

    manifest = _object(root / "manifest.json")
    expected_manifest = {
        "schema_version": 2,
        "report_kind": REPORT_KIND,
        "source_identity": source,
        "protocol_sha256": PROTOCOL_SHA256,
        "current_toolchain_manifest_digest": new["manifest_digest"],
        "current_environment_id": new["environment_id"],
        "from_policy_digest": candidate.digest,
        "target_policy_digest": legacy.digest,
        "configs_digest": sha256_json(CONFIGS),
        "primary_capture_manifest_digest": candidate_capture.manifest["manifest_digest"],
        "old_release_tag": OLD_TAG,
        "old_release_commit": OLD_COMMIT,
        "old_release_tag_object": OLD_TAG_OBJECT,
        "old_bundle_manifest_digest": OLD_BUNDLE,
    }
    if set(manifest) != MANIFEST_FIELDS or manifest != expected_manifest:
        _fail("policy rollback manifest differs from fully verified facts")
    if (root / "current-toolchain.sha256").read_text(encoding="ascii") != new["manifest_digest"] + "\n":
        _fail("policy rollback current wrapper digest authority is stale")

    expected_roots = {"new-source", "old-source", "captures", "old-release-bundle", "logs"}
    expected_files = {"rollback-report.json", "execution.json", "commands.txt", "rollback-report.md", "manifest.json",
                      "current-toolchain.sha256"}
    actual_roots = {name.split("/", 1)[0] for name in inventory if "/" in name}
    if actual_roots != expected_roots or {name for name in inventory if "/" not in name} != expected_files:
        _fail("policy rollback contains unreferenced portable artifacts")
    authority_files = {"toolchain.json", "source-provenance.json", "locks/requirements-bootstrap.lock",
                       "locks/requirements-dev.lock", "locks/runtime-lock.json", "locks/pyproject.toml"}
    for name in ("new-source", "old-source"):
        if {path[len(name) + 1:] for path in inventory if path.startswith(name + "/")} != authority_files:
            _fail("policy rollback source authority inventory is not exact")
    if {name for name in inventory if name.startswith("logs/")} != {
            f"logs/{phase}.{stream}.txt" for phase in PHASES for stream in ("stdout", "stderr")}:
        _fail("policy rollback transcript inventory is not exact")
    return report


def verify_stateful_policy_rollback(root: Path) -> dict[str, Any]:
    """Verify an actual CPU rollback component without granting qualification."""

    try:
        return _verify(root)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "policy rollback evidence has invalid structure",
                        {"exception_type": type(exc).__name__}) from exc


__all__ = ["verify_stateful_policy_rollback", "portable_stateful_policy_rollback_inventory"]

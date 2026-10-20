"""Run the V0.2 single-request baseline without making a speedup claim.

The four fixed groups separate KV mode from capture/telemetry overhead. Formal
output requires an already-finalized toolchain manifest and produces COMPLETE
only after every semantic invariant and checksum passes. The explicit
development escape hatch is always unqualified and cannot produce that marker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, NoReturn, Sequence, cast

from _toolchain_common import bootstrap_digest, project_digest
from illm import __version__
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import METRIC_IDS, RequestObservation, build_streaming_metrics, registry_digest
from illm.sampling import SamplingSpec
from illm.stateful_benchmark import verify_completed_output as verify_completed_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATHS = {
    "tuning": PROJECT_ROOT / "fixtures" / "workloads" / "v02-single-request-tuning.json",
    "heldout": PROJECT_ROOT / "fixtures" / "workloads" / "v02-single-request-heldout.json",
}
EXPECTED_SUITE_DIGESTS = {
    "tuning": "9b0a020d3b4f2d9f83f0d6d8712a5a99d7ad8612f25f24f4a41171b121bd58ce",
    "heldout": "50ec684ffbc2e649a450b805f6ac11687ebd818a1cc22c5ad95faf041a5bac44",
}
REAL_MODEL_PROFILES = {
    (PROJECT_ROOT / "configs" / "models" / "v0_2.yaml").resolve(),
    (PROJECT_ROOT / "configs" / "models" / "v0_2_fp16.yaml").resolve(),
}
REAL_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
GROUPS = (
    ("none-off", "none", "off", False),
    ("none-request", "none", "request", True),
    ("contiguous-off", "contiguous", "off", False),
    ("contiguous-request", "contiguous", "request", True),
)
EVIDENCE_SCOPES = {
    "tiny": "tiny-cpu-learning-baseline",
    "real": "real-single-gpu-256x64",
}
_PROJECT_DIGEST_DIRECTORIES = (
    ".github",
    "docs",
    "evidence",
    "src",
    "scripts",
    "tests",
    "schemas",
    "registries",
    "runtime",
    "configs",
    "fixtures",
)
CORRECTNESS_FIELDS = (
    "output_token_ids",
    "outcome",
    "finish_reason",
    "next_output_position",
    "actual_seed",
    "sampling_spec_digest",
    "rng_draws_consumed",
    "committed_draw_addresses",
)
RESOURCE_FIELDS = {
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
}
MEMORY_PLAN_FIELDS = {
    "schema_version",
    "model_spec_id",
    "weights_digest",
    "weight_manifest_digest",
    "runtime_dtype",
    "device",
    "kv_mode",
    "kv_layout_id",
    "capacity_tokens",
    "memory_budget_bytes",
    "weights_bytes",
    "kv_bytes",
    "activation_bytes",
    "workspace_bytes",
    "io_bytes",
    "telemetry_bytes",
    "headroom_bytes",
    "required_bytes",
}
SERVER_TIMING_FIELDS = {
    "clock_domain",
    "accepted_ns",
    "token_ready_ns",
    "terminal_ready_ns",
    "tokenization_wall_time_ns",
    "per_forward_wall_time_ns",
    "server_e2e_wall_time_ns",
    "critical_capture_event_drop_count",
    "telemetry",
    "evidence_kind",
}
CLIENT_TIMING_FIELDS = {"clock_domain", "send_ns", "terminal_observed_ns", "elapsed_ns"}
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)) + b"\n")
    temporary.replace(path)


def _signed_document(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _fail(ErrorCode.INTERNAL, f"cannot sign a document that already contains {field}")
    value[field] = sha256_json(value)
    return value


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON facts without Python's bool-equals-int coercion."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _object(path: Path) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, f"JSON input is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.INVALID_ARGUMENT, f"JSON input must be an object: {path.name}")
    return cast(dict[str, Any], value)


def load_workload_suite(path: Path, *, suite: str, target: str) -> dict[str, Any]:
    """Load a predeclared public-synthetic suite and fail closed on drift."""

    if suite not in SUITE_PATHS or target not in {"tiny", "real"}:
        _fail(ErrorCode.INVALID_ARGUMENT, "suite and target must use the frozen V0.2 values")
    value = _object(path)
    expected_fields = {
        "schema_version", "suite_id", "suite", "public_synthetic", "sampling_spec", "targets", "suite_digest",
    }
    if set(value) != expected_fields:
        _fail(ErrorCode.INVALID_ARGUMENT, "workload suite fields differ from the frozen schema")
    expected_id = f"illm.v02.single-request.{suite}.v1"
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("suite") != suite
        or value.get("suite_id") != expected_id
        or value.get("public_synthetic") is not True
    ):
        _fail(ErrorCode.INVALID_ARGUMENT, "workload suite identity is invalid")
    digest = value.get("suite_digest")
    if (
        not isinstance(digest, str)
        or digest != EXPECTED_SUITE_DIGESTS[suite]
        or digest != sha256_json({key: item for key, item in value.items() if key != "suite_digest"})
    ):
        _fail(ErrorCode.INVALID_ARGUMENT, "workload suite digest mismatch")
    try:
        sampling = SamplingSpec.from_dict(cast(Mapping[str, Any], value["sampling_spec"]))
    except Exception as exc:
        from illm.engine.facade import as_illm_error

        raise as_illm_error(exc) from exc
    if sampling.mode != "greedy" or sampling.actual_seed != 7:
        _fail(ErrorCode.INVALID_ARGUMENT, "V0.2 baseline sampling must be frozen greedy")
    targets = value.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"tiny", "real"}:
        _fail(ErrorCode.INVALID_ARGUMENT, "workload suite must define tiny and real targets")
    target_value = cast(dict[str, Any], targets).get(target)
    if not isinstance(target_value, dict) or set(target_value) != {"required_capacity_tokens", "cases"}:
        _fail(ErrorCode.INVALID_ARGUMENT, "workload target fields are invalid")
    required_capacity = target_value.get("required_capacity_tokens")
    expected_capacity = 256 if target == "tiny" else 512
    if type(required_capacity) is not int or required_capacity != expected_capacity:
        _fail(ErrorCode.INVALID_ARGUMENT, "workload target capacity differs from the frozen profile")
    cases = target_value.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail(ErrorCode.INVALID_ARGUMENT, "workload target requires at least one case")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "source", "prompt_tokens", "max_new_tokens"}:
            _fail(ErrorCode.INVALID_ARGUMENT, "workload case fields are invalid")
        case_id = case.get("case_id")
        prompt_tokens = case.get("prompt_tokens")
        max_new_tokens = case.get("max_new_tokens")
        if not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None or case_id in seen:
            _fail(ErrorCode.INVALID_ARGUMENT, "workload case_id is invalid or duplicated")
        seen.add(case_id)
        if type(prompt_tokens) is not int or prompt_tokens < 1 or type(max_new_tokens) is not int or max_new_tokens < 1:
            _fail(ErrorCode.INVALID_ARGUMENT, "workload token counts must be positive integers")
        if prompt_tokens + max_new_tokens > required_capacity:
            _fail(ErrorCode.CONTEXT_LIMIT_EXCEEDED, "workload case exceeds its predeclared capacity")
        source = case.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
            _fail(ErrorCode.INVALID_ARGUMENT, "workload source is invalid")
        if target == "tiny":
            if set(source) != {"kind", "token_ids"} or source["kind"] != "public-token-ids-v1":
                _fail(ErrorCode.INVALID_ARGUMENT, "tiny workload requires public token IDs")
            token_ids = source.get("token_ids")
            if (
                not isinstance(token_ids, list)
                or len(token_ids) != prompt_tokens
                or any(type(token) is not int or token < 0 for token in token_ids)
            ):
                _fail(ErrorCode.INVALID_ARGUMENT, "tiny public token IDs do not match prompt_tokens")
        else:
            if set(source) != {"kind", "text", "add_bos"} or source["kind"] != "encoded-public-text-cycle-v1":
                _fail(ErrorCode.INVALID_ARGUMENT, "real workload requires the frozen encoded text-cycle source")
            if not isinstance(source.get("text"), str) or not source["text"] or type(source.get("add_bos")) is not bool:
                _fail(ErrorCode.INVALID_ARGUMENT, "real public source text is invalid")
            if prompt_tokens != 256 or max_new_tokens != 64:
                _fail(ErrorCode.INVALID_ARGUMENT, "real V0.2 workload must be exactly 256 prompt and 64 output tokens")
    return value


def validate_target_capacity(
    model: str | Path,
    target: str,
    inspected: Mapping[str, Any],
    required_capacity_tokens: int,
) -> int:
    """Bind target kind and fixed runtime capacity before allocating an engine."""

    fixture_id = inspected.get("fixture_id")
    maximum = inspected.get("max_context_tokens")
    if target == "tiny" and (str(model) != "fixture:tiny-decoder" or fixture_id != "tiny-decoder"):
        _fail(ErrorCode.INVALID_ARGUMENT, "tiny target requires fixture:tiny-decoder")
    spec = inspected.get("spec")
    if target == "real":
        if fixture_id is not None or not isinstance(model, (str, Path)) or Path(model).resolve() not in REAL_MODEL_PROFILES:
            _fail(ErrorCode.INVALID_ARGUMENT, "real target requires a frozen V0.2 local model profile")
        if not isinstance(spec, dict) or (
            spec.get("model_revision") != REAL_MODEL_REVISION
            or spec.get("tokenizer_revision") != REAL_MODEL_REVISION
            or spec.get("runtime_dtype") not in {"bf16", "fp16"}
        ):
            _fail(ErrorCode.INVALID_ARGUMENT, "real target model identity differs from the frozen V0.2 profile")
    if type(maximum) is not int or maximum < required_capacity_tokens:
        _fail(
            ErrorCode.CONTEXT_LIMIT_EXCEEDED,
            "model profile cannot satisfy the predeclared benchmark capacity",
            {"required_capacity_tokens": required_capacity_tokens, "profile_capacity_tokens": maximum},
        )
    return required_capacity_tokens


def _source_identity() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "project_root_digest": project_digest(PROJECT_ROOT),
        "bootstrap_digest": bootstrap_digest(PROJECT_ROOT),
    }


def _toolchain_binding(
    development_no_manifest: bool,
    source: Mapping[str, str],
) -> dict[str, Any]:
    if development_no_manifest:
        return {
            "binding": "development-unqualified",
            "formal_evidence_eligible": False,
            "manifest_digest": None,
            "environment_id": os.environ.get("ILLM_ENVIRONMENT_ID"),
            "python_executable": str(Path(sys.executable).resolve()),
            "manifest_environment_ignored": bool(os.environ.get("ILLM_TOOLCHAIN_MANIFEST")),
        }
    manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not manifest_name:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "run the streaming baseline through a finalized manifest runner")
    assert manifest_name is not None
    manifest_path = Path(manifest_name).resolve()
    try:
        toolchain = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest is unreadable") from exc
    required = {
        "schema_version",
        "finalized",
        "environment_id",
        "python_executable",
        "python_version",
        "project_root",
        "project_root_digest",
        "bootstrap_digest",
        "lock_digest",
        "runtime_profile_id",
        "runtime_lock_digest",
        "manifest_digest",
    }
    if (
        not isinstance(toolchain, dict)
        or not required <= set(toolchain)
        or type(toolchain.get("schema_version")) is not int
        or toolchain.get("schema_version") != 1
        or toolchain.get("finalized") is not True
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "streaming baseline requires a finalized selected runtime profile")
    digest = toolchain.get("manifest_digest")
    if (
        not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        or digest != sha256_json({key: value for key, value in toolchain.items() if key != "manifest_digest"})
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "streaming baseline toolchain manifest checksum mismatch")
    environment_id = toolchain.get("environment_id")
    python_executable = toolchain.get("python_executable")
    lock_identity = toolchain.get("lock_digest")
    runtime_lock = toolchain.get("runtime_lock_digest")
    if (
        not isinstance(environment_id, str)
        or os.environ.get("ILLM_ENVIRONMENT_ID") != environment_id
        or not isinstance(python_executable, str)
        or Path(sys.executable).resolve() != Path(python_executable).resolve()
        or not isinstance(lock_identity, str)
        or _DIGEST.fullmatch(lock_identity) is None
        or os.environ.get("ILLM_LOCK_DIGEST") != lock_identity
        or not isinstance(runtime_lock, str)
        or _DIGEST.fullmatch(runtime_lock) is None
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "streaming baseline interpreter differs from the finalized environment")
    if (
        Path(str(toolchain.get("project_root"))).resolve() != PROJECT_ROOT
        or toolchain.get("project_root_digest") != source["project_root_digest"]
        or toolchain.get("bootstrap_digest") != source["bootstrap_digest"]
        or toolchain.get("python_version") != platform.python_version()
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "streaming baseline manifest is not bound to current source/runtime")
    runtime_profile_id = toolchain.get("runtime_profile_id")
    if not isinstance(runtime_profile_id, str) or not runtime_profile_id:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "streaming baseline runtime profile identity is missing")
    return {
        "binding": "finalized-source-bound-manifest",
        "formal_evidence_eligible": True,
        "manifest_path": str(manifest_path),
        "manifest_digest": digest,
        "environment_id": environment_id,
        "python_executable": str(Path(python_executable).resolve()),
        "project_root": str(PROJECT_ROOT),
        "project_root_digest": source["project_root_digest"],
        "bootstrap_digest": source["bootstrap_digest"],
        "lock_digest": lock_identity,
        "runtime_profile_id": runtime_profile_id,
        "runtime_lock_digest": runtime_lock,
    }


def _hardware(torch: Any, device: str, cpu_threads: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": device,
        "cpu_threads": cpu_threads,
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cuda_runtime": torch.version.cuda,
        "live_free_memory_queried": False,
    }
    driver: dict[str, Any]
    if not device.startswith("cuda"):
        driver = {"kind": "not_applicable", "reason": "cpu_device"}
    else:
        properties = torch.cuda.get_device_properties(device)
        result.update({
            "gpu_name": properties.name,
            "total_device_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
        })
        try:
            process = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "NVIDIA driver version cannot be recorded") from exc
        versions = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if process.returncode != 0 or not versions:
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "NVIDIA driver version cannot be recorded")
        driver = {"kind": "nvidia", "reported_versions": sorted(set(versions))}
    result["driver"] = driver
    result["driver_digest"] = sha256_json(driver)
    result["hardware_digest"] = sha256_json(result)
    return result


def _materialize_cases(
    suite_value: Mapping[str, Any], target: str, tokenizer: Any, vocab_size: int,
) -> list[dict[str, Any]]:
    target_value = cast(dict[str, Any], cast(dict[str, Any], suite_value["targets"])[target])
    materialized: list[dict[str, Any]] = []
    for raw_case in cast(list[dict[str, Any]], target_value["cases"]):
        source = cast(dict[str, Any], raw_case["source"])
        prompt_tokens = cast(int, raw_case["prompt_tokens"])
        if source["kind"] == "public-token-ids-v1":
            token_ids = tuple(cast(list[int], source["token_ids"]))
        else:
            encoded = tokenizer.encode(source["text"], add_bos=source["add_bos"])
            if not encoded:
                _fail(ErrorCode.INVALID_ARGUMENT, "public synthetic source encoded to no tokens")
            repeats = math.ceil(prompt_tokens / len(encoded))
            token_ids = tuple((encoded * repeats)[:prompt_tokens])
        if len(token_ids) != prompt_tokens or any(type(token) is not int or not 0 <= token < vocab_size for token in token_ids):
            _fail(ErrorCode.INVALID_ARGUMENT, "materialized public prompt violates the model vocabulary or length")
        fixture = {
            "case_id": raw_case["case_id"],
            "source": source,
            "materialization_contract": (
                "literal-public-token-ids-v1"
                if source["kind"] == "public-token-ids-v1"
                else "tokenizer.encode-once(add_bos=frozen)-then-cycle-and-prefix-truncate-v1"
            ),
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": raw_case["max_new_tokens"],
            "public_input_token_ids": list(token_ids),
            "public_input_token_ids_digest": sha256_json({"token_ids": list(token_ids)}),
        }
        fixture["public_input_fixture_id"] = sha256_json(fixture)
        materialized.append(fixture)
    return materialized


def _gpu_memory_before(torch: Any, device: str) -> dict[str, int] | None:
    if not device.startswith("cuda"):
        return None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_before_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_before_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _gpu_memory_after(torch: Any, device: str, before: dict[str, int] | None) -> dict[str, int] | None:
    if before is None:
        return None
    torch.cuda.synchronize(device)
    return {
        **before,
        "allocated_after_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_after_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _correctness_fact(result: Any) -> dict[str, Any]:
    return {
        "output_token_ids": list(result.output_token_ids),
        "outcome": result.outcome,
        "finish_reason": result.finish_reason,
        "next_output_position": result.next_output_position,
        "actual_seed": result.actual_seed,
        "sampling_spec_digest": result.sampling_spec_digest,
        "rng_draws_consumed": result.rng_draws_consumed,
        "committed_draw_addresses": [list(address) for address in result.committed_draw_addresses],
    }


def _run_request(
    torch: Any,
    engine: Any,
    case: Mapping[str, Any],
    *,
    group_id: str,
    kv_mode: str,
    telemetry: str,
    capture_enabled: bool,
    phase: str,
    iteration: int,
    sampling: SamplingSpec,
    output: Path,
    target: str,
    suite: str,
    suite_digest: str,
    evidence_scope: str,
    capacity: int,
) -> tuple[dict[str, Any], RequestObservation | None]:
    from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk, TerminalChunk, TokenChunk

    case_id = cast(str, case["case_id"])
    request_id = f"bench-{group_id}-{case_id}-{phase}-{iteration}"
    capture = output / "captures" / group_id / case_id / f"{phase}-{iteration}" if capture_enabled else None
    before = _gpu_memory_before(torch, engine.device)
    client_send_ns = perf_counter_ns()
    chunks = tuple(engine.stream(GenerationRequest(
        model_spec_id=engine.model_spec_id,
        input_token_ids=tuple(cast(list[int], case["public_input_token_ids"])),
        max_new_tokens=cast(int, case["max_new_tokens"]),
        sampling=sampling,
        request_id=request_id,
        capture=capture,
        telemetry=telemetry,
    )))
    client_terminal_ns = perf_counter_ns()
    gpu_memory = _gpu_memory_after(torch, engine.device, before)
    if not chunks or not isinstance(chunks[0], MetadataChunk) or not isinstance(chunks[-1], TerminalChunk):
        _fail(ErrorCode.INTERNAL, "stream did not contain ordered metadata and terminal chunks")
    terminals = [chunk for chunk in chunks if isinstance(chunk, TerminalChunk)]
    tokens = [chunk for chunk in chunks if isinstance(chunk, TokenChunk)]
    if len(terminals) != 1 or [chunk.output_index for chunk in tokens] != list(range(len(tokens))):
        _fail(ErrorCode.INTERNAL, "stream token or terminal ordering is invalid")
    terminal = terminals[0]
    result = terminal.result
    if tuple(chunk.token_id for chunk in tokens) != result.output_token_ids:
        _fail(ErrorCode.ORACLE_DIVERGED, "stream token IDs differ from the terminal result")
    if result.outcome != "FINISHED":
        _fail(ErrorCode.ORACLE_DIVERGED, "baseline request did not finish successfully")
    if target == "real" and (
        len(result.output_token_ids) != case["max_new_tokens"] or result.finish_reason != "MAX_NEW_TOKENS"
    ):
        _fail(
            ErrorCode.ORACLE_DIVERGED,
            "real 256/64 workload terminated early; the predeclared workload was not completed",
            {"case_id": case_id, "committed_output_tokens": len(result.output_token_ids), "finish_reason": result.finish_reason},
        )
    if result.next_output_position != len(result.output_token_ids):
        _fail(ErrorCode.ORACLE_DIVERGED, "committed output position differs from output length")
    if result.rng_draws_consumed != 0 or result.committed_draw_addresses:
        _fail(ErrorCode.ORACLE_DIVERGED, "greedy baseline consumed RNG state")
    expected_query = (
        tuple(cast(int, case["prompt_tokens"]) + index for index in range(len(result.output_token_ids)))
        if kv_mode == "none"
        else (cast(int, case["prompt_tokens"]), *(1 for _ in result.output_token_ids[1:]))
    )
    if result.query_tokens_per_step != expected_query:
        _fail(ErrorCode.ORACLE_DIVERGED, "executor query-token facts differ from the KV-mode contract")
    if any(result.resource_status.values()) or any(engine.resource_status.values()):
        _fail(ErrorCode.ENGINE_UNHEALTHY, "request did not return all engine resources to baseline")
    if result.capture_error is not None:
        _fail(ErrorCode.CAPTURE_INCOMPLETE, "required benchmark capture is incomplete")
    capture_path: str | None = None
    if capture is not None:
        bundle = StreamingCaptureBundle.open(capture)
        if result.capture_manifest_digest != bundle.manifest.get("manifest_digest"):
            _fail(ErrorCode.CAPTURE_CORRUPT, "capture manifest digest differs from the generation result")
        if bundle.manifest.get("memory_plan_digest") != engine.memory_plan.digest:
            _fail(ErrorCode.CAPTURE_CORRUPT, "capture MemoryPlan digest differs from the engine")
        capture_path = capture.relative_to(output).as_posix()
    elif result.capture_manifest_digest is not None:
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture-off request unexpectedly produced a capture digest")
    if client_terminal_ns <= client_send_ns:
        _fail(ErrorCode.INTERNAL, "client timing pair is not strictly increasing")
    record: dict[str, Any] = {
        "schema_version": 1,
        "request_id": request_id,
        "group_id": group_id,
        "kv_mode": kv_mode,
        "telemetry": telemetry,
        "capture_enabled": capture_enabled,
        "case_id": case_id,
        "phase": phase,
        "iteration": iteration,
        "suite": suite,
        "suite_digest": suite_digest,
        "target": target,
        "evidence_scope": evidence_scope,
        "model": engine.model,
        "model_spec_id": engine.model_spec_id,
        "device": engine.device,
        "capacity_tokens": capacity,
        "sampling_spec_digest": sampling.digest,
        "public_input_fixture_id": case["public_input_fixture_id"],
        "prompt_tokens": case["prompt_tokens"],
        "requested_output_tokens": case["max_new_tokens"],
        "committed_output_tokens": len(result.output_token_ids),
        "correctness": _correctness_fact(result),
        "output_token_ids_digest": sha256_json({"token_ids": list(result.output_token_ids)}),
        "query_tokens_per_step": list(result.query_tokens_per_step),
        "kv_valid_tokens_per_step": list(result.kv_valid_tokens_per_step),
        "resource_status": result.resource_status,
        "memory_plan_digest": engine.memory_plan.digest,
        "server_timings": result.timings,
        "client_timing": {
            "clock_domain": "benchmark-client.perf_counter_ns",
            "send_ns": client_send_ns,
            "terminal_observed_ns": client_terminal_ns,
            "elapsed_ns": client_terminal_ns - client_send_ns,
        },
        "gpu_memory": gpu_memory,
        "gpu_memory_contract": "allocator allocated/peak/reserved facts only; never live-free admission",
        "capture": capture_path,
        "capture_manifest_digest": result.capture_manifest_digest,
    }
    record["record_digest"] = sha256_json(record)
    observation = None
    if phase == "measured":
        observation = RequestObservation(
            request_id=request_id,
            accepted_ns=cast(int, result.timings["accepted_ns"]),
            token_ready_ns=tuple(cast(list[int], result.timings["token_ready_ns"])),
            terminal_ready_ns=cast(int, result.timings["terminal_ready_ns"]),
            outcome=result.outcome,
            client_send_ns=client_send_ns,
            client_terminal_ns=client_terminal_ns,
            preaccept_reason=None,
            server_clock_domain=cast(str, result.timings["clock_domain"]),
            client_clock_domain="benchmark-client.perf_counter_ns",
        )
    return record, observation


def _group_summary(group_id: str, records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    measured = [record for record in records if record["phase"] == "measured"]
    client_values = [cast(int, cast(dict[str, Any], record["client_timing"])["elapsed_ns"]) for record in measured]
    server_values = [cast(int, cast(dict[str, Any], record["server_timings"])["server_e2e_wall_time_ns"]) for record in measured]
    query_values = [sum(cast(list[int], record["query_tokens_per_step"])) for record in measured]
    gpu_peaks = [
        cast(dict[str, int], record["gpu_memory"])["peak_allocated_bytes"]
        for record in measured if record["gpu_memory"] is not None
    ]
    return {
        "group_id": group_id,
        "warmup_count": sum(record["phase"] == "warmup" for record in records),
        "measurement_count": len(measured),
        "metrics": dict(metrics),
        "observed": {
            "client_e2e_ns": client_values,
            "server_e2e_ns": server_values,
            "total_query_tokens_per_request": query_values,
            "median_client_e2e_ns": float(statistics.median(client_values)),
            "median_server_e2e_ns": float(statistics.median(server_values)),
            "median_total_query_tokens_per_request": float(statistics.median(query_values)),
            "peak_device_allocated_bytes": max(gpu_peaks) if gpu_peaks else None,
        },
    }


def _metric_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | None]:
    left_metrics = cast(dict[str, Any], cast(dict[str, Any], left["metrics"])["metrics"])
    right_metrics = cast(dict[str, Any], cast(dict[str, Any], right["metrics"])["metrics"])
    differences: dict[str, float | None] = {}
    for metric_id in METRIC_IDS:
        left_value = cast(dict[str, Any], left_metrics[metric_id])["p99"]
        right_value = cast(dict[str, Any], right_metrics[metric_id])["p99"]
        differences[metric_id] = None if left_value is None or right_value is None else float(right_value) - float(left_value)
    return differences


def _comparison(
    kind: str,
    pairs: Sequence[tuple[str, Mapping[str, Any], str, Mapping[str, Any]]],
    *,
    evidence_scope: str,
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for left_id, left, right_id, right in pairs:
        left_observed = cast(dict[str, Any], left["observed"])
        right_observed = cast(dict[str, Any], right["observed"])
        values.append({
            "left_group": left_id,
            "right_group": right_id,
            "signed_difference_boundary": "right_minus_left",
            "observed_signed_differences": {
                "median_client_e2e_ns": float(right_observed["median_client_e2e_ns"]) - float(left_observed["median_client_e2e_ns"]),
                "median_server_e2e_ns": float(right_observed["median_server_e2e_ns"]) - float(left_observed["median_server_e2e_ns"]),
                "median_total_query_tokens_per_request": (
                    float(right_observed["median_total_query_tokens_per_request"])
                    - float(left_observed["median_total_query_tokens_per_request"])
                ),
                "metric_p99": _metric_differences(left, right),
            },
        })
    document = {
        "schema_version": 1,
        "comparison_kind": kind,
        "evidence_scope": evidence_scope,
        "interpretation": "observed signed differences only; no speedup or external-engine claim",
        "speedup_claim": None,
        "external_engine_comparison": None,
        "pairs": values,
    }
    return _signed_document(document, "comparison_digest")


def _checksum_inventory(output: Path) -> dict[str, str]:
    excluded = {"INCOMPLETE", "COMPLETE", "checksums.json", "DEVELOPMENT_ONLY"}
    inventory: dict[str, str] = {}
    normalized: set[str] = set()
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output).as_posix()
        if path.is_symlink():
            _fail(ErrorCode.CAPTURE_CORRUPT, f"benchmark output contains a symbolic link: {relative}")
        if not path.is_file() or relative in excluded:
            continue
        folded = relative.casefold()
        if folded in normalized:
            _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark output contains a case-colliding path")
        normalized.add(folded)
        inventory[relative] = sha256_file(path)
    return inventory


def _finalize_formal_output(output: Path) -> None:
    if (output / "DEVELOPMENT_ONLY").exists():
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "development benchmark output cannot be finalized")
    inventory = _checksum_inventory(output)
    _atomic_json(output / "checksums.json", inventory)
    if _object(output / "checksums.json") != _checksum_inventory(output):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark checksum self-check failed")
    (output / "COMPLETE").write_text(f"{sha256_file(output / 'checksums.json')}\n", encoding="ascii")
    (output / "INCOMPLETE").unlink()
    try:
        verify_completed_output(output)
    except BaseException:
        (output / "COMPLETE").unlink(missing_ok=True)
        _atomic_json(
            output / "INCOMPLETE",
            {"reason": "formal_benchmark_semantic_verification_failed", "formal": True},
        )
        raise


def run_benchmark(
    model: str | Path,
    *,
    device: str,
    output: Path,
    suite: str,
    target: str,
    warmups: int = 3,
    repetitions: int = 7,
    cpu_threads: int = 1,
    development_no_manifest: bool = False,
) -> dict[str, Any]:
    """Run the four fixed groups and persist either formal or development facts."""

    if suite not in SUITE_PATHS or target not in {"tiny", "real"}:
        _fail(ErrorCode.INVALID_ARGUMENT, "suite must be tuning/heldout and target must be tiny/real")
    if type(warmups) is not int or warmups < 1 or type(repetitions) is not int or repetitions < 3:
        _fail(ErrorCode.INVALID_ARGUMENT, "streaming baseline requires warmups>=1 and repetitions>=3")
    if type(cpu_threads) is not int or cpu_threads != 1:
        _fail(ErrorCode.INVALID_ARGUMENT, "V0.2 streaming baseline requires cpu_threads=1")
    if type(development_no_manifest) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "development_no_manifest must be explicit bool")
    if target == "tiny" and device != "cpu":
        _fail(ErrorCode.INVALID_ARGUMENT, "tiny target is the fixed CPU learning baseline")
    if target == "real" and not device.startswith("cuda"):
        _fail(ErrorCode.INVALID_ARGUMENT, "real target requires an explicit CUDA device")
    output = output.resolve()
    if not development_no_manifest and any(
        output.is_relative_to(PROJECT_ROOT / relative)
        for relative in _PROJECT_DIGEST_DIRECTORIES
    ):
        _fail(
            ErrorCode.INVALID_ARGUMENT,
            "formal baseline output cannot be created inside a source-digest directory",
        )
    source = _source_identity()
    toolchain = _toolchain_binding(development_no_manifest, source)
    evidence_scope = EVIDENCE_SCOPES[target]
    suite_value = load_workload_suite(SUITE_PATHS[suite], suite=suite, target=target)
    target_value = cast(dict[str, Any], cast(dict[str, Any], suite_value["targets"])[target])
    from illm.engine.facade import as_illm_error
    from illm.model import inspect_model

    try:
        inspected = inspect_model(model, project_root=PROJECT_ROOT)
    except Exception as exc:
        raise as_illm_error(exc) from exc
    capacity = validate_target_capacity(model, target, inspected, cast(int, target_value["required_capacity_tokens"]))
    if output.is_symlink() or output.exists() and (not output.is_dir() or any(output.iterdir())):
        _fail(ErrorCode.INVALID_ARGUMENT, "baseline output must be a new, empty, non-symlink directory")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "INCOMPLETE", {"reason": "benchmark_in_progress", "formal": not development_no_manifest})

    import torch
    from illm.engine.streaming import Engine

    original_threads = torch.get_num_threads()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    all_records: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    common_identity: dict[str, Any] | None = None
    materialized: list[dict[str, Any]] | None = None
    hardware: dict[str, Any] | None = None
    request_ids: set[str] = set()
    facts: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    try:
        sampling = SamplingSpec.from_dict(cast(Mapping[str, Any], suite_value["sampling_spec"]))
        for group_id, kv_mode, telemetry, capture_enabled in GROUPS:
            engine: Engine | None = None
            try:
                engine = Engine(
                    model,
                    device=device,
                    kv_mode=kv_mode,
                    max_context_tokens=capacity,
                    project_root=PROJECT_ROOT,
                )
                if materialized is None:
                    materialized = _materialize_cases(
                        suite_value,
                        target,
                        engine._loaded.tokenizer,
                        engine._loaded.spec.vocab_size,
                    )
                if hardware is None:
                    hardware = _hardware(torch, engine.device, cpu_threads)
                identity = dict(engine._identity)
                shared = {key: value for key, value in identity.items() if key != "kv_mode"}
                if common_identity is None:
                    common_identity = shared
                elif shared != common_identity:
                    _fail(ErrorCode.ORACLE_DIVERGED, "engine identity changed between fixed benchmark groups")
                records: list[dict[str, Any]] = []
                observations: list[RequestObservation] = []
                group_facts: dict[tuple[str, str, int], dict[str, Any]] = {}
                for phase, count in (("warmup", warmups), ("measured", repetitions)):
                    for case in materialized:
                        for iteration in range(count):
                            record, observation = _run_request(
                                torch,
                                engine,
                                case,
                                group_id=group_id,
                                kv_mode=kv_mode,
                                telemetry=telemetry,
                                capture_enabled=capture_enabled,
                                phase=phase,
                                iteration=iteration,
                                sampling=sampling,
                                output=output,
                                target=target,
                                suite=suite,
                                suite_digest=cast(str, suite_value["suite_digest"]),
                                evidence_scope=evidence_scope,
                                capacity=capacity,
                            )
                            if record["request_id"] in request_ids:
                                _fail(ErrorCode.INTERNAL, "benchmark request IDs are not globally unique")
                            request_ids.add(cast(str, record["request_id"]))
                            records.append(record)
                            all_records.append(record)
                            group_facts[(cast(str, case["case_id"]), phase, iteration)] = cast(
                                dict[str, Any], record["correctness"],
                            )
                            if observation is not None:
                                observations.append(observation)
                metrics = build_streaming_metrics(observations)
                summary = _group_summary(group_id, records, metrics)
                summary.update({
                    "kv_mode": kv_mode,
                    "telemetry": telemetry,
                    "capture_enabled": capture_enabled,
                    "memory_plan": engine.memory_plan.to_dict(),
                    "memory_plan_digest": engine.memory_plan.digest,
                    "engine_identity": identity,
                })
                groups[group_id] = summary
                facts[group_id] = group_facts
            finally:
                if engine is not None and not engine.close():
                    _fail(ErrorCode.ENGINE_UNHEALTHY, "benchmark engine did not close all background resources")

        baseline = facts["none-off"]
        if any(group_facts != baseline for group_facts in facts.values()):
            _fail(ErrorCode.ORACLE_DIVERGED, "KV or telemetry mode changed exact greedy generation facts")
        no_kv = _comparison(
            "no-kv-vs-contiguous",
            (
                ("none-off", groups["none-off"], "contiguous-off", groups["contiguous-off"]),
                ("none-request", groups["none-request"], "contiguous-request", groups["contiguous-request"]),
            ),
            evidence_scope=evidence_scope,
        )
        telemetry_comparison = _comparison(
            "telemetry-overhead",
            (
                ("none-off", groups["none-off"], "none-request", groups["none-request"]),
                ("contiguous-off", groups["contiguous-off"], "contiguous-request", groups["contiguous-request"]),
            ),
            evidence_scope=evidence_scope,
        )
        assert common_identity is not None and materialized is not None and hardware is not None
        inspected_spec = cast(dict[str, Any], inspected["spec"])
        model_identity = {
            "model_spec_id": common_identity["model_spec_id"],
            "model_revision": inspected_spec["model_revision"],
            "weights_digest": common_identity["weights_digest"],
            "weight_manifest_digest": common_identity["load_manifest_digest"],
            "tokenizer_digest": common_identity["tokenizer_digest"],
            "tokenizer_revision": inspected_spec["tokenizer_revision"],
            "runtime_dtype": common_identity["dtype"],
        }
        driver_identity = {
            "driver_id": "illm-v02-single-request-baseline-v1",
            "source_path": "scripts/benchmark_streaming.py",
            "source_sha256": sha256_file(Path(__file__).resolve()),
        }
        profile_path = Path(model).resolve() if target == "real" else None
        model_authority = {
            "authority_type": "fixture" if target == "tiny" else "profile",
            "fixture_id": inspected["fixture_id"],
            "profile_name": None if profile_path is None else profile_path.name,
            "profile_sha256": None if profile_path is None else sha256_file(profile_path),
            "max_context_tokens": inspected["max_context_tokens"],
            "spec": {
                "spec_id": inspected_spec["spec_id"],
                "model_revision": inspected_spec["model_revision"],
                "tokenizer_revision": inspected_spec["tokenizer_revision"],
                "runtime_dtype": inspected_spec["runtime_dtype"],
                "vocab_size": inspected_spec["vocab_size"],
            },
            "hardware_requirements": inspected["hardware_requirements"],
            "model_identity": model_identity,
        }
        metric_digest = registry_digest()
        authority = {
            "schema_version": 1,
            "authority_kind": "illm-v02-streaming-benchmark-authority-v1",
            "producer_source": source,
            "benchmark_driver": driver_identity,
            "suite": suite_value,
            "target_contract": {
                "target": target,
                "evidence_scope": evidence_scope,
                "required_capacity_tokens": capacity,
                "model_selector": str(model),
                "device_class": "cpu" if target == "tiny" else "cuda",
            },
            "model_authority": model_authority,
            "metric_registry_digest": metric_digest,
            "engine_version": __version__,
        }
        authority_digest = sha256_json(authority)
        workload = _signed_document({
            "schema_version": 1,
            "suite_id": suite_value["suite_id"],
            "suite_digest": suite_value["suite_digest"],
            "target": target,
            "evidence_scope": evidence_scope,
            "required_capacity_tokens": capacity,
            "privacy_scope": "public-synthetic-fixtures-only",
            "authority_digest": authority_digest,
            "cases": materialized,
        }, "materialized_workload_digest")
        _atomic_json(output / "raw" / "materialized-workload.json", workload)
        raw = _signed_document({
            "schema_version": 1,
            "evidence_kind": "development_measured" if development_no_manifest else "measured",
            "evidence_scope": evidence_scope,
            "suite": suite,
            "suite_digest": suite_value["suite_digest"],
            "target": target,
            "model": str(model),
            "model_spec_id": common_identity["model_spec_id"],
            "device": hardware["device"],
            "capacity_tokens": capacity,
            "sampling_spec_digest": sampling.digest,
            "warmups_per_case_group": warmups,
            "repetitions_per_case_group": repetitions,
            "cpu_threads": 1,
            "warmups_included": True,
            "authority_digest": authority_digest,
            "records": all_records,
        }, "raw_digest")
        _atomic_json(output / "raw" / "observations.json", raw)
        _atomic_json(output / "no-kv-vs-contiguous.json", no_kv)
        _atomic_json(output / "telemetry-overhead.json", telemetry_comparison)
        report = _signed_document({
            "schema_version": 1,
            "report_kind": "illm-v02-single-request-baseline",
            "status": "pass",
            "all_semantic_checks_passed": True,
            "evidence_kind": "development_measured" if development_no_manifest else "measured",
            "evidence_scope": evidence_scope,
            "development_no_manifest": development_no_manifest,
            "formal_evidence_eligible": not development_no_manifest,
            "evidence_stage": "development" if development_no_manifest else "qualification_candidate",
            "qualification_tier": "unqualified",
            "qualification_claim": None,
            "development_reasons": (
                ["explicit --development-no-manifest"] if development_no_manifest else []
            ),
            "performance_outcome": "baseline_only",
            "speedup_claim": None,
            "external_engine_comparison": None,
            "interpretation": "observed differences only; performance is not a release win condition",
            "suite": suite,
            "suite_id": suite_value["suite_id"],
            "suite_digest": suite_value["suite_digest"],
            "target": target,
            "model": str(model),
            "engine_version": __version__,
            "identity": {
                "toolchain": toolchain,
                "source": source,
                "benchmark_driver": driver_identity,
                "model": model_identity,
                "sampling_spec": sampling.to_dict(),
                "sampling_spec_digest": sampling.digest,
                "metric_registry_digest": metric_digest,
                "hardware_digest": hardware["hardware_digest"],
                "hardware_driver_digest": hardware["driver_digest"],
                "memory_plan_digests": {group_id: group["memory_plan_digest"] for group_id, group in groups.items()},
                "authority": authority,
                "authority_digest": authority_digest,
            },
            "hardware": hardware,
            "measurement_contract": {
                "groups": [group[0] for group in GROUPS],
                "fixed_matrix": [
                    {"group_id": group_id, "kv_mode": kv_mode, "telemetry": observation,
                     "required_capture": capture_enabled}
                    for group_id, kv_mode, observation, capture_enabled in GROUPS
                ],
                "warmups_per_case_group": warmups,
                "repetitions_per_case_group": repetitions,
                "cpu_threads": 1,
                "tf32_enabled": False,
                "sampling": "greedy exact; fixed seed shared across distinct request IDs",
                "server_clock_domain": "host.perf_counter_ns",
                "client_clock_domain": "benchmark-client.perf_counter_ns",
                "cross_clock_subtraction": False,
                "gpu_memory": "allocated/peak/reserved only; no live-free admission query",
                "load_time_excluded": True,
                "warmups_recorded_but_excluded_from_metrics": True,
                "observation_discard_policy": "none",
                "run_order": "group-order-then-phase-then-case-then-iteration-v1",
            },
            "correctness": {
                "exact_across_all_four_groups": True,
                "comparison_fields": list(next(iter(baseline.values())).keys()),
                "globally_unique_request_ids": len(request_ids),
                "resource_baseline_checked_after_every_request": True,
                "capture_complete_checked_for_request_groups": True,
                "real_workload_completed_256x64": target == "real",
                "accepted_request_failure_rate": {
                    "numerator": 0,
                    "denominator": len(all_records),
                    "value": 0.0,
                },
            },
            "groups": groups,
            "artifacts": {
                "raw": "raw/observations.json",
                "materialized_workload": "raw/materialized-workload.json",
                "no_kv_vs_contiguous": "no-kv-vs-contiguous.json",
                "telemetry_overhead": "telemetry-overhead.json",
                "captures": "captures/",
            },
        }, "report_digest")
        if development_no_manifest:
            _atomic_json(output / "report" / "dev-report.json", report)
            _atomic_json(output / "DEVELOPMENT_ONLY", {
                "qualification_tier": "unqualified",
                "formal_evidence_eligible": False,
                "development_no_manifest": True,
                "evidence_scope": evidence_scope,
                "report_digest": report["report_digest"],
            })
            (output / "INCOMPLETE").unlink()
        else:
            _atomic_json(output / "report" / "benchmark-report.json", report)
            _finalize_formal_output(output)
        return report
    except BaseException as exc:
        (output / "COMPLETE").unlink(missing_ok=True)
        try:
            _atomic_json(output / "INCOMPLETE", {
                "reason": "benchmark_failed",
                "exception_type": type(exc).__name__,
                "formal": not development_no_manifest,
            })
        except OSError:
            pass
        raise
    finally:
        torch.set_num_threads(original_threads)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(SUITE_PATHS), required=True)
    parser.add_argument("--target", choices=("tiny", "real"), required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--development-no-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            args.model,
            device=args.device,
            output=args.output,
            suite=args.suite,
            target=args.target,
            warmups=args.warmups,
            repetitions=args.repetitions,
            cpu_threads=args.cpu_threads,
            development_no_manifest=args.development_no_manifest,
        )
        print(json.dumps({
            "status": report["status"],
            "report_digest": report["report_digest"],
            "development_no_manifest": report["development_no_manifest"],
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        from illm.engine.facade import as_illm_error

        error = as_illm_error(exc)
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

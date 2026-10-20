"""Portable, tensor-free verification for V0.2 streaming benchmark evidence.

The reader recomputes persisted semantic facts and never executes a model.  It
uses only the Python standard library plus tensor-free iLLM contract modules.
Checksums establish internal consistency, not authenticity of fabricated input.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import METRIC_IDS, RequestObservation, build_streaming_metrics
from illm.sampling import SamplingSpec

__all__ = [
    "EVIDENCE_SCOPES",
    "EXPECTED_SUITE_DIGESTS",
    "GROUPS",
    "SUITE_PATHS",
    "verify_completed_output",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUITE_PATHS = {
    "tuning": PROJECT_ROOT / "fixtures" / "workloads" / "v02-single-request-tuning.json",
    "heldout": PROJECT_ROOT / "fixtures" / "workloads" / "v02-single-request-heldout.json",
}
EXPECTED_SUITE_DIGESTS = {
    "tuning": "9b0a020d3b4f2d9f83f0d6d8712a5a99d7ad8612f25f24f4a41171b121bd58ce",
    "heldout": "50ec684ffbc2e649a450b805f6ac11687ebd818a1cc22c5ad95faf041a5bac44",
}
REAL_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_METRIC_REGISTRY_DIGEST = "18618f35bd4a9141de6d6191619150d0ce8121d31f37c7a7f1547a0a0136e3ce"
REAL_PROFILE_AUTHORITIES = {
    "v0_2.yaml": {
        "profile_sha256": "1b2b81c473eb0ce75e2a079d2623ffb24977ae38f6e20180130ae4e8d2627aea",
        "runtime_dtype": "bf16",
    },
    "v0_2_fp16.yaml": {
        "profile_sha256": "e9ab221cec9572db8e581d30cdbfe6cdddc109e13e93e91f742aa3c73ec3b7aa",
        "runtime_dtype": "fp16",
    },
}
REAL_MATERIALIZED_INPUT_DIGESTS = {
    "tuning": "8ed85a2327c5c0ef2bb63a36c6dd43ccfb1e44d2dcb30bdea01a4126cfb6b6e0",
    "heldout": "64a2c12a3ba7c64bc81aa3f407f253d5f73dac319a17b11649437c018f0f5ccd",
}
TINY_MODEL_IDENTITY = {
    "model_spec_id": "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873",
    "model_revision": "sha256:c9e7f3974db381ff9031ac0ff2111acec97fd90e989df9ea17e5d052f555e486",
    "weights_digest": "3868a9840d0b7e418e26c6a31d5022fd829e68090fe75d617964eee7ea09ac92",
    "weight_manifest_digest": "dd0827b9ed7e4b1e2808e30552602a38cdda4666bf8b1417482b69ce1e5840bd",
    "tokenizer_digest": "f8972f8627f9f6af04431b14a6cedc02cba870e9494d76f72d3df82ccb161616",
    "tokenizer_revision": "sha256:f8972f8627f9f6af04431b14a6cedc02cba870e9494d76f72d3df82ccb161616",
    "runtime_dtype": "fp32",
}
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
_CUDA_DEVICE = re.compile(r"cuda(?::[0-9]+)?")

def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)

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
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence must be an object: {path.name}")
    return cast(dict[str, Any], value)

def _validate_workload_suite(value: dict[str, Any], *, suite: str, target: str) -> dict[str, Any]:
    """Validate an embedded or file-backed frozen public-synthetic suite."""

    if suite not in SUITE_PATHS or target not in {"tiny", "real"}:
        _fail(ErrorCode.INVALID_ARGUMENT, "suite and target must use the frozen V0.2 values")
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
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "workload sampling specification is invalid",
        ) from exc
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


def load_workload_suite(path: Path, *, suite: str, target: str) -> dict[str, Any]:
    """Load a predeclared public-synthetic suite and fail closed on drift."""

    return _validate_workload_suite(_object(path), suite=suite, target=target)

def _digest_files(project_root: Path, paths: Iterable[Path]) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted({item.resolve() for item in paths}, key=lambda item: item.as_posix()):
        if path.is_file():
            entries.append((path.relative_to(project_root).as_posix(), sha256_file(path)))
    return sha256_json(sorted(entries))

def _bootstrap_digest(project_root: Path) -> str:
    return _digest_files(
        project_root,
        [
            project_root / "scripts" / name
            for name in ("bootstrap.py", "_toolchain_common.py", "_runtime_artifacts.py")
        ],
    )

def _project_digest(project_root: Path) -> str:
    paths: list[Path] = [
        project_root / ".gitattributes",
        project_root / ".gitignore",
        project_root / "AI_INFERENCE_ENGINE_PROJECT_PLAN.md",
        project_root / "LICENSE",
        project_root / "README.md",
        project_root / "pyproject.toml",
        project_root / "requirements-bootstrap.lock",
        project_root / "requirements-dev.lock",
    ]
    for relative in (
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
    ):
        root = project_root / relative
        if root.exists():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
            )
    return _digest_files(project_root, paths)

def _source_identity() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "project_root_digest": _project_digest(PROJECT_ROOT),
        "bootstrap_digest": _bootstrap_digest(PROJECT_ROOT),
    }

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


def _verified_authority_snapshot(
    value: dict[str, Any],
    *,
    suite: str,
    target: str,
    evidence_scope: str,
    capacity: int,
    model: str,
    device: str,
    source: dict[str, Any],
    driver: dict[str, Any],
    model_identity: dict[str, Any],
    engine_version: str,
    metric_registry_digest: str,
    require_current_source: bool,
) -> dict[str, Any]:
    """Validate the self-contained producer authority embedded in evidence."""

    expected_fields = {
        "schema_version",
        "authority_kind",
        "producer_source",
        "benchmark_driver",
        "suite",
        "target_contract",
        "model_authority",
        "metric_registry_digest",
        "engine_version",
    }
    producer_source = value.get("producer_source")
    authority_driver = value.get("benchmark_driver")
    embedded_suite = value.get("suite")
    target_contract = value.get("target_contract")
    model_authority = value.get("model_authority")
    if (
        set(value) != expected_fields
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("authority_kind") != "illm-v02-streaming-benchmark-authority-v1"
        or not all(
            isinstance(item, dict)
            for item in (
                producer_source,
                authority_driver,
                embedded_suite,
                target_contract,
                model_authority,
            )
        )
        or value.get("metric_registry_digest") != EXPECTED_METRIC_REGISTRY_DIGEST
        or value.get("metric_registry_digest") != metric_registry_digest
        or not isinstance(value.get("engine_version"), str)
        or not value.get("engine_version")
        or value.get("engine_version") != engine_version
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark authority snapshot identity is invalid")
    producer_source = cast(dict[str, Any], producer_source)
    authority_driver = cast(dict[str, Any], authority_driver)
    embedded_suite = cast(dict[str, Any], embedded_suite)
    target_contract = cast(dict[str, Any], target_contract)
    model_authority = cast(dict[str, Any], model_authority)
    if (
        set(producer_source) != {"project_root", "project_root_digest", "bootstrap_digest"}
        or not isinstance(producer_source.get("project_root"), str)
        or not producer_source.get("project_root")
        or any(
            not isinstance(producer_source.get(name), str)
            or _DIGEST.fullmatch(cast(str, producer_source[name])) is None
            for name in ("project_root_digest", "bootstrap_digest")
        )
        or not _same_json(producer_source, source)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark producer source snapshot is invalid")
    if (
        set(authority_driver) != {"driver_id", "source_path", "source_sha256"}
        or authority_driver.get("driver_id") != "illm-v02-single-request-baseline-v1"
        or authority_driver.get("source_path") != "scripts/benchmark_streaming.py"
        or not isinstance(authority_driver.get("source_sha256"), str)
        or _DIGEST.fullmatch(cast(str, authority_driver["source_sha256"])) is None
        or not _same_json(authority_driver, driver)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark producer driver snapshot is invalid")
    try:
        suite_value = _validate_workload_suite(embedded_suite, suite=suite, target=target)
    except IllmError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "embedded benchmark suite authority is invalid") from exc
    expected_device_class = "cpu" if target == "tiny" else "cuda"
    if (
        set(target_contract)
        != {"target", "evidence_scope", "required_capacity_tokens", "model_selector", "device_class"}
        or target_contract.get("target") != target
        or target_contract.get("evidence_scope") != evidence_scope
        or type(target_contract.get("required_capacity_tokens")) is not int
        or target_contract.get("required_capacity_tokens") != capacity
        or target_contract.get("model_selector") != model
        or target_contract.get("device_class") != expected_device_class
        or (target == "tiny" and (model != "fixture:tiny-decoder" or device != "cpu"))
        or (target == "real" and _CUDA_DEVICE.fullmatch(device) is None)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark target authority is invalid")

    expected_model_fields = {
        "authority_type",
        "fixture_id",
        "profile_name",
        "profile_sha256",
        "max_context_tokens",
        "spec",
        "hardware_requirements",
        "model_identity",
    }
    spec = model_authority.get("spec")
    hardware_requirements = model_authority.get("hardware_requirements")
    authority_model_identity = model_authority.get("model_identity")
    if (
        set(model_authority) != expected_model_fields
        or not isinstance(spec, dict)
        or not isinstance(hardware_requirements, dict)
        or not isinstance(authority_model_identity, dict)
        or not _same_json(authority_model_identity, model_identity)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark model authority shape is invalid")
    spec = cast(dict[str, Any], spec)
    hardware_requirements = cast(dict[str, Any], hardware_requirements)
    expected_hardware_requirements = {
        "evidence_kind": "recorded",
        "minimum_compute_capability": [0, 0] if target == "tiny" else [8, 0],
        "minimum_gpu_memory_bytes": 0 if target == "tiny" else 6 * 1024**3,
        "policy": "checked_before_cuda_allocation; cpu_debug_does_not_qualify_gpu",
    }
    if (
        set(spec)
        != {"spec_id", "model_revision", "tokenizer_revision", "runtime_dtype", "vocab_size"}
        or spec.get("spec_id") != model_identity.get("model_spec_id")
        or spec.get("model_revision") != model_identity.get("model_revision")
        or spec.get("tokenizer_revision") != model_identity.get("tokenizer_revision")
        or spec.get("runtime_dtype") != model_identity.get("runtime_dtype")
        or not _same_json(hardware_requirements, expected_hardware_requirements)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark model authority facts are invalid")
    if target == "tiny":
        if (
            model_authority.get("authority_type") != "fixture"
            or model_authority.get("fixture_id") != "tiny-decoder"
            or model_authority.get("profile_name") is not None
            or model_authority.get("profile_sha256") is not None
            or type(model_authority.get("max_context_tokens")) is not int
            or model_authority.get("max_context_tokens") != 256
            or spec.get("vocab_size") != 259
            or not _same_json(model_identity, TINY_MODEL_IDENTITY)
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "tiny fixture authority differs from its registered contract")
    else:
        profile_name = model_authority.get("profile_name")
        registered = REAL_PROFILE_AUTHORITIES.get(profile_name) if isinstance(profile_name, str) else None
        if (
            model_authority.get("authority_type") != "profile"
            or model_authority.get("fixture_id") is not None
            or registered is None
            or model_authority.get("profile_sha256") != registered["profile_sha256"]
            or type(model_authority.get("max_context_tokens")) is not int
            or model_authority.get("max_context_tokens") != 512
            or spec.get("model_revision") != REAL_MODEL_REVISION
            or spec.get("tokenizer_revision") != REAL_MODEL_REVISION
            or spec.get("runtime_dtype") != registered["runtime_dtype"]
            or spec.get("vocab_size") != 151936
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "real profile authority differs from its registered contract")
    if require_current_source:
        try:
            current_source = _source_identity()
            current_driver_sha256 = sha256_file(PROJECT_ROOT / "scripts" / "benchmark_streaming.py")
        except OSError as exc:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "current source audit is unavailable") from exc
        if (
            not _same_json(producer_source, current_source)
            or authority_driver.get("source_sha256") != current_driver_sha256
        ):
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "benchmark authority differs from the current source checkout")
    return suite_value


def _verified_materialized_workload(
    output: Path,
    *,
    suite_value: dict[str, Any],
    suite: str,
    target: str,
    evidence_scope: str,
    capacity: int,
    authority_digest: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = _object(output / "raw" / "materialized-workload.json")
    digest = value.get("materialized_workload_digest")
    expected_fields = {
        "schema_version",
        "suite_id",
        "suite_digest",
        "target",
        "evidence_scope",
        "required_capacity_tokens",
        "privacy_scope",
        "authority_digest",
        "cases",
        "materialized_workload_digest",
    }
    if (
        set(value) != expected_fields
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("suite_id") != suite_value["suite_id"]
        or value.get("suite_digest") != suite_value["suite_digest"]
        or value.get("target") != target
        or value.get("evidence_scope") != evidence_scope
        or type(value.get("required_capacity_tokens")) is not int
        or value.get("required_capacity_tokens") != capacity
        or value.get("privacy_scope") != "public-synthetic-fixtures-only"
        or value.get("authority_digest") != authority_digest
        or not isinstance(digest, str)
        or digest != sha256_json({key: item for key, item in value.items() if key != "materialized_workload_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "materialized workload identity is invalid")
    cases_value = value.get("cases")
    fixed_cases = cast(dict[str, Any], cast(dict[str, Any], suite_value["targets"])[target])["cases"]
    if not isinstance(cases_value, list) or len(cases_value) != len(fixed_cases):
        _fail(ErrorCode.CAPTURE_CORRUPT, "materialized workload case count differs from the frozen suite")
    cases = cast(list[dict[str, Any]], cases_value)
    for materialized, frozen in zip(cases, cast(list[dict[str, Any]], fixed_cases), strict=True):
        if not isinstance(materialized, dict):
            _fail(ErrorCode.CAPTURE_CORRUPT, "materialized workload case is not an object")
        expected_case_fields = {
            "case_id",
            "source",
            "materialization_contract",
            "prompt_tokens",
            "max_new_tokens",
            "public_input_token_ids",
            "public_input_token_ids_digest",
            "public_input_fixture_id",
        }
        token_ids = materialized.get("public_input_token_ids")
        source = cast(dict[str, Any], frozen["source"])
        expected_contract = (
            "literal-public-token-ids-v1"
            if source["kind"] == "public-token-ids-v1"
            else "tokenizer.encode-once(add_bos=frozen)-then-cycle-and-prefix-truncate-v1"
        )
        if (
            set(materialized) != expected_case_fields
            or materialized.get("case_id") != frozen["case_id"]
            or not _same_json(materialized.get("source"), source)
            or materialized.get("materialization_contract") != expected_contract
            or type(materialized.get("prompt_tokens")) is not int
            or materialized.get("prompt_tokens") != frozen["prompt_tokens"]
            or type(materialized.get("max_new_tokens")) is not int
            or materialized.get("max_new_tokens") != frozen["max_new_tokens"]
            or not isinstance(token_ids, list)
            or len(token_ids) != frozen["prompt_tokens"]
            or any(type(token) is not int or token < 0 for token in token_ids)
            or materialized.get("public_input_token_ids_digest") != sha256_json({"token_ids": token_ids})
            or materialized.get("public_input_fixture_id")
            != sha256_json({key: item for key, item in materialized.items() if key != "public_input_fixture_id"})
            or (target == "tiny" and not _same_json(token_ids, source["token_ids"]))
            or (
                target == "real"
                and materialized.get("public_input_token_ids_digest")
                != REAL_MATERIALIZED_INPUT_DIGESTS[suite]
            )
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "materialized workload case differs from its frozen suite contract")
    return suite_value, cases

def _verify_memory_plan(
    plan: Any,
    digest: Any,
    *,
    group_id: str,
    kv_mode: str,
    capacity: int,
    device: str,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(plan, dict) or not isinstance(digest, str) or digest != sha256_json(plan):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} MemoryPlan digest is invalid")
    integer_fields = {
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
    if set(plan) != MEMORY_PLAN_FIELDS or any(
        type(plan.get(name)) is not int or plan[name] < 0 for name in integer_fields
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} MemoryPlan byte facts are invalid")
    components = sum(cast(int, plan[name]) for name in integer_fields - {"capacity_tokens", "memory_budget_bytes", "required_bytes"})
    if (
        type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
        or plan.get("model_spec_id") != model_identity.get("model_spec_id")
        or plan.get("weights_digest") != model_identity.get("weights_digest")
        or plan.get("weight_manifest_digest") != model_identity.get("weight_manifest_digest")
        or plan.get("runtime_dtype") != model_identity.get("runtime_dtype")
        or plan.get("device") != device
        or plan.get("kv_mode") != kv_mode
        or plan.get("kv_layout_id")
        != ("none-reference-full-context-v1" if kv_mode == "none" else "qwen2-contiguous-layer-major-v1")
        or plan.get("capacity_tokens") != capacity
        or plan.get("required_bytes") != components
        or cast(int, plan["required_bytes"]) > cast(int, plan["memory_budget_bytes"])
        or (kv_mode == "none" and plan.get("kv_bytes") != 0)
        or (kv_mode == "contiguous" and cast(int, plan["kv_bytes"]) <= 0)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} MemoryPlan is not bound to model/device/capacity")
    return cast(dict[str, Any], plan)

def _verify_capture_record(
    output: Path,
    record: Mapping[str, Any],
    *,
    expected_path: str,
    expected_query: list[int],
    expected_kv: list[int],
    group_identity: Mapping[str, Any],
    memory_plan: Mapping[str, Any],
    sampling: SamplingSpec,
) -> None:
    if record.get("capture") != expected_path:
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture path differs from the fixed group/request layout")
    try:
        bundle = StreamingCaptureBundle.open(output / expected_path)
    except IllmError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "benchmark capture failed canonical verification") from exc
    manifest = bundle.manifest
    identity = manifest.get("identity")
    request_id = record["request_id"]
    if (
        manifest.get("request_id") != request_id
        or record.get("capture_manifest_digest") != manifest.get("manifest_digest")
        or manifest.get("sampling_spec") != sampling.to_dict()
        or manifest.get("sampling_spec_digest") != sampling.digest
        or manifest.get("memory_plan") != memory_plan
        or manifest.get("memory_plan_digest") != record.get("memory_plan_digest")
        or manifest.get("logical_terminal_outcome") != "FINISHED"
        or manifest.get("terminal_outcome") != "FINISHED"
        or not isinstance(identity, dict)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture manifest is not bound to its raw request")
    for field in (
        "engine_version",
        "toolchain_digest",
        "device",
        "dtype",
        "model_spec_id",
        "weights_digest",
        "load_manifest_digest",
        "tokenizer_digest",
        "kv_mode",
        "memory_plan_evidence_kind",
    ):
        if identity.get(field) != group_identity.get(field):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"capture engine identity differs for {field}")
    story = bundle.request_story(cast(str, request_id))
    token_events = cast(list[dict[str, Any]], story["tokens"])
    if len(token_events) != len(expected_query):
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture output count differs from raw observations")
    for index, item in enumerate(token_events):
        payload = cast(dict[str, Any], item["payload"])
        if (
            payload.get("output_index") != index
            or payload.get("committed_output_count") != index + 1
            or payload.get("kv_valid_token_count") != expected_kv[index]
            or payload.get("next_output_position") != index + 1
            or type(payload.get("rng_draws_consumed")) is not int
            or payload.get("rng_draws_consumed") != 0
            or payload.get("draw_address") is not None
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "capture output/KV/RNG sequence differs from raw observations")
    timeline = cast(list[dict[str, Any]], story["timeline"])
    tokenized = [item for item in timeline if item.get("kind") == "request.tokenized"]
    accepted = [item for item in timeline if item.get("kind") == "request.accepted"]
    decided = [item for item in timeline if item.get("kind") == "request.terminal_decided"]
    completed = [item for item in timeline if item.get("kind") == "request.completed"]
    forwards = [item for item in timeline if item.get("kind") == "model.forward"]
    forward_queries = [cast(dict[str, Any], item["payload"]).get("query_tokens") for item in forwards]
    forward_durations = [cast(dict[str, Any], item["payload"]).get("duration_ns") for item in forwards]
    kv_story = cast(dict[str, Any], story["kv"])
    timings = cast(dict[str, Any], record["server_timings"])
    correctness = cast(dict[str, Any], record["correctness"])
    if not all(len(items) == 1 for items in (tokenized, accepted, decided, completed)):
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture terminal/timing event cardinality is invalid")
    tokenized_payload = cast(dict[str, Any], tokenized[0]["payload"])
    decided_payload = cast(dict[str, Any], decided[0]["payload"])
    completed_payload = cast(dict[str, Any], completed[0]["payload"])
    if (
        forward_queries != expected_query
        or not _same_json(forward_durations, timings["per_forward_wall_time_ns"])
        or accepted[0].get("monotonic_ns") != timings["accepted_ns"]
        or not _same_json(
            [cast(dict[str, Any], item["payload"])["server_ready_ns"] for item in token_events],
            timings["token_ready_ns"],
        )
        or completed_payload.get("server_terminal_ready_ns") != timings["terminal_ready_ns"]
        or cast(int, tokenized_payload.get("ended_ns")) - cast(int, tokenized_payload.get("started_ns"))
        != timings["tokenization_wall_time_ns"]
        or decided_payload.get("outcome") != correctness["outcome"]
        or decided_payload.get("reason") != correctness["finish_reason"]
        or decided_payload.get("committed_output_count") != record["committed_output_tokens"]
        or decided_payload.get("kv_valid_token_count") != (expected_kv[-1] if expected_kv else 0)
        or completed_payload.get("finish_reason") != correctness["finish_reason"]
        or completed_payload.get("committed_output_count") != record["committed_output_tokens"]
        or completed_payload.get("kv_valid_token_count") != (expected_kv[-1] if expected_kv else 0)
        or completed_payload.get("active_leases") != 0
        or completed_payload.get("pending_writes") != 0
        or completed_payload.get("physical_cleanup_complete") is not True
        or kv_story.get("final_valid_token_count") != (expected_kv[-1] if expected_kv else 0)
        or kv_story.get("final_active_allocation_bytes") != 0
        or kv_story.get("final_open_write_count") != 0
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture query/KV/terminal/timing facts differ from raw observations")

def _verified_record(
    output: Path,
    record: Any,
    *,
    group_id: str,
    kv_mode: str,
    telemetry: str,
    capture_enabled: bool,
    phase: str,
    iteration: int,
    case: Mapping[str, Any],
    suite: str,
    suite_digest: str,
    target: str,
    evidence_scope: str,
    model: str,
    model_spec_id: str,
    device: str,
    capacity: int,
    sampling: SamplingSpec,
    group_identity: Mapping[str, Any],
    memory_plan: Mapping[str, Any],
    memory_plan_digest: str,
) -> tuple[dict[str, Any], RequestObservation | None]:
    if not isinstance(record, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark raw record is not an object")
    expected_fields = {
        "schema_version",
        "request_id",
        "group_id",
        "kv_mode",
        "telemetry",
        "capture_enabled",
        "case_id",
        "phase",
        "iteration",
        "suite",
        "suite_digest",
        "target",
        "evidence_scope",
        "model",
        "model_spec_id",
        "device",
        "capacity_tokens",
        "sampling_spec_digest",
        "public_input_fixture_id",
        "prompt_tokens",
        "requested_output_tokens",
        "committed_output_tokens",
        "correctness",
        "output_token_ids_digest",
        "query_tokens_per_step",
        "kv_valid_tokens_per_step",
        "resource_status",
        "memory_plan_digest",
        "server_timings",
        "client_timing",
        "gpu_memory",
        "gpu_memory_contract",
        "capture",
        "capture_manifest_digest",
        "record_digest",
    }
    case_id = cast(str, case["case_id"])
    request_id = f"bench-{group_id}-{case_id}-{phase}-{iteration}"
    digest = record.get("record_digest")
    if (
        set(record) != expected_fields
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != 1
        or record.get("request_id") != request_id
        or record.get("group_id") != group_id
        or record.get("kv_mode") != kv_mode
        or record.get("telemetry") != telemetry
        or record.get("capture_enabled") is not capture_enabled
        or record.get("case_id") != case_id
        or record.get("phase") != phase
        or type(record.get("iteration")) is not int
        or record.get("iteration") != iteration
        or record.get("suite") != suite
        or record.get("suite_digest") != suite_digest
        or record.get("target") != target
        or record.get("evidence_scope") != evidence_scope
        or record.get("model") != model
        or record.get("model_spec_id") != model_spec_id
        or record.get("device") != device
        or type(record.get("capacity_tokens")) is not int
        or record.get("capacity_tokens") != capacity
        or record.get("sampling_spec_digest") != sampling.digest
        or record.get("public_input_fixture_id") != case["public_input_fixture_id"]
        or type(record.get("prompt_tokens")) is not int
        or record.get("prompt_tokens") != case["prompt_tokens"]
        or type(record.get("requested_output_tokens")) is not int
        or record.get("requested_output_tokens") != case["max_new_tokens"]
        or record.get("memory_plan_digest") != memory_plan_digest
        or record.get("gpu_memory_contract")
        != "allocator allocated/peak/reserved facts only; never live-free admission"
        or not isinstance(digest, str)
        or digest != sha256_json({key: item for key, item in record.items() if key != "record_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark raw record identity is invalid")
    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or set(correctness) != set(CORRECTNESS_FIELDS):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark correctness facts are invalid")
    output_tokens = correctness.get("output_token_ids")
    count = record.get("committed_output_tokens")
    requested = cast(int, case["max_new_tokens"])
    if (
        not isinstance(output_tokens, list)
        or any(type(token) is not int or token < 0 for token in output_tokens)
        or type(count) is not int
        or count != len(output_tokens)
        or not 1 <= count <= requested
        or record.get("output_token_ids_digest") != sha256_json({"token_ids": output_tokens})
        or correctness.get("outcome") != "FINISHED"
        or correctness.get("finish_reason") not in {"EOS", "MAX_NEW_TOKENS"}
        or (correctness.get("finish_reason") == "MAX_NEW_TOKENS" and count != requested)
        or (target == "real" and (count != requested or correctness.get("finish_reason") != "MAX_NEW_TOKENS"))
        or type(correctness.get("next_output_position")) is not int
        or correctness.get("next_output_position") != count
        or type(correctness.get("actual_seed")) is not int
        or correctness.get("actual_seed") != sampling.actual_seed
        or correctness.get("sampling_spec_digest") != sampling.digest
        or type(correctness.get("rng_draws_consumed")) is not int
        or correctness.get("rng_draws_consumed") != 0
        or not _same_json(correctness.get("committed_draw_addresses"), [])
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark output/terminal/RNG facts violate the frozen contract")
    prompt_tokens = cast(int, case["prompt_tokens"])
    expected_query = (
        list(range(prompt_tokens, prompt_tokens + count))
        if kv_mode == "none"
        else [prompt_tokens, *([1] * (count - 1))]
    )
    expected_kv = [0] * count if kv_mode == "none" else list(range(prompt_tokens, prompt_tokens + count))
    query_values = record.get("query_tokens_per_step")
    kv_values = record.get("kv_valid_tokens_per_step")
    if (
        not isinstance(query_values, list)
        or any(type(value) is not int for value in query_values)
        or not _same_json(query_values, expected_query)
        or not isinstance(kv_values, list)
        or any(type(value) is not int for value in kv_values)
        or not _same_json(kv_values, expected_kv)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark query/KV sequence violates the selected KV mode")
    resources = record.get("resource_status")
    if (
        not isinstance(resources, dict)
        or set(resources) != RESOURCE_FIELDS
        or any(type(value) is not int or value != 0 for value in resources.values())
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark request did not prove its resource baseline")
    timings = record.get("server_timings")
    client = record.get("client_timing")
    if not isinstance(timings, dict) or not isinstance(client, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark timing facts are not objects")
    accepted_ns = timings.get("accepted_ns")
    token_ready = timings.get("token_ready_ns")
    terminal_ns = timings.get("terminal_ready_ns")
    forward_ns = timings.get("per_forward_wall_time_ns")
    if (
        set(timings) != SERVER_TIMING_FIELDS
        or set(client) != CLIENT_TIMING_FIELDS
        or timings.get("clock_domain") != "host.perf_counter_ns"
        or type(accepted_ns) is not int
        or not isinstance(token_ready, list)
        or len(token_ready) != count
        or any(type(value) is not int for value in token_ready)
        or type(terminal_ns) is not int
        or terminal_ns < accepted_ns
        or any(value < accepted_ns or value > terminal_ns for value in token_ready)
        or any(left > right for left, right in zip(token_ready, token_ready[1:]))
        or type(timings.get("server_e2e_wall_time_ns")) is not int
        or timings.get("server_e2e_wall_time_ns") != terminal_ns - accepted_ns
        or type(timings.get("tokenization_wall_time_ns")) is not int
        or cast(int, timings["tokenization_wall_time_ns"]) < 0
        or not isinstance(forward_ns, list)
        or len(forward_ns) != count
        or any(type(value) is not int or value < 0 for value in forward_ns)
        or type(timings.get("critical_capture_event_drop_count")) is not int
        or timings.get("critical_capture_event_drop_count") != 0
        or timings.get("telemetry") != telemetry
        or timings.get("evidence_kind") != "measured"
        or client.get("clock_domain") != "benchmark-client.perf_counter_ns"
        or type(client.get("send_ns")) is not int
        or type(client.get("terminal_observed_ns")) is not int
        or cast(int, client["terminal_observed_ns"]) <= cast(int, client["send_ns"])
        or type(client.get("elapsed_ns")) is not int
        or client.get("elapsed_ns") != cast(int, client["terminal_observed_ns"]) - cast(int, client["send_ns"])
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark timing facts violate their independent clock domains")
    gpu_memory = record.get("gpu_memory")
    if target == "tiny" and gpu_memory is not None:
        _fail(ErrorCode.CAPTURE_CORRUPT, "tiny CPU baseline cannot contain GPU allocator facts")
    if target == "real":
        gpu_fields = {
            "allocated_before_bytes",
            "reserved_before_bytes",
            "allocated_after_bytes",
            "reserved_after_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        }
        if (
            not isinstance(gpu_memory, dict)
            or set(gpu_memory) != gpu_fields
            or any(type(value) is not int or value < 0 for value in gpu_memory.values())
            or gpu_memory["allocated_before_bytes"] > gpu_memory["reserved_before_bytes"]
            or gpu_memory["allocated_after_bytes"] > gpu_memory["reserved_after_bytes"]
            or gpu_memory["peak_allocated_bytes"]
            < max(gpu_memory["allocated_before_bytes"], gpu_memory["allocated_after_bytes"])
            or gpu_memory["peak_reserved_bytes"]
            < max(gpu_memory["reserved_before_bytes"], gpu_memory["reserved_after_bytes"])
            or gpu_memory["peak_allocated_bytes"] > gpu_memory["peak_reserved_bytes"]
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "real GPU baseline allocator facts are invalid")
    if capture_enabled:
        expected_capture = f"captures/{group_id}/{case_id}/{phase}-{iteration}"
        _verify_capture_record(
            output,
            record,
            expected_path=expected_capture,
            expected_query=expected_query,
            expected_kv=expected_kv,
            group_identity=group_identity,
            memory_plan=memory_plan,
            sampling=sampling,
        )
    elif record.get("capture") is not None or record.get("capture_manifest_digest") is not None:
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture-off benchmark record contains capture claims")
    observation = None
    if phase == "measured":
        observation = RequestObservation(
            request_id=request_id,
            accepted_ns=accepted_ns,
            token_ready_ns=tuple(cast(list[int], token_ready)),
            terminal_ready_ns=terminal_ns,
            outcome="FINISHED",
            client_send_ns=cast(int, client["send_ns"]),
            client_terminal_ns=cast(int, client["terminal_observed_ns"]),
            preaccept_reason=None,
            server_clock_domain="host.perf_counter_ns",
            client_clock_domain="benchmark-client.perf_counter_ns",
        )
    return cast(dict[str, Any], record), observation

def verify_completed_output(
    output: Path,
    *,
    require_current_source: bool = False,
) -> dict[str, Any]:
    """Recompute a formal baseline from its self-contained persisted facts.

    The default reader never dereferences producer paths or the current checkout.
    ``require_current_source`` is an explicit, non-portable audit mode.
    """

    if type(require_current_source) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "require_current_source must be bool")
    output = output.resolve()
    if (output / "DEVELOPMENT_ONLY").exists():
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "development benchmark output is never formal evidence")
    if not (output / "COMPLETE").is_file() or (output / "INCOMPLETE").exists():
        _fail(ErrorCode.CAPTURE_INCOMPLETE, "formal benchmark completion marker is absent or contradicted")
    checksums = _object(output / "checksums.json")
    if any(
        not isinstance(name, str)
        or Path(name).is_absolute()
        or ".." in Path(name).parts
        or not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
        for name, digest in checksums.items()
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark checksum inventory is invalid")
    actual = _checksum_inventory(output)
    if checksums != actual:
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark checksum inventory or file content differs")
    checksums_digest = sha256_file(output / "checksums.json")
    try:
        complete_digest = (output / "COMPLETE").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "benchmark completion marker is unreadable") from exc
    if complete_digest != checksums_digest:
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark completion marker does not bind the checksum inventory")
    report = _object(output / "report" / "benchmark-report.json")
    digest = report.get("report_digest")
    suite = report.get("suite")
    target = report.get("target")
    expected_report_fields = {
        "schema_version",
        "report_kind",
        "status",
        "all_semantic_checks_passed",
        "evidence_kind",
        "evidence_scope",
        "development_no_manifest",
        "formal_evidence_eligible",
        "evidence_stage",
        "qualification_tier",
        "qualification_claim",
        "development_reasons",
        "performance_outcome",
        "speedup_claim",
        "external_engine_comparison",
        "interpretation",
        "suite",
        "suite_id",
        "suite_digest",
        "target",
        "model",
        "engine_version",
        "identity",
        "hardware",
        "measurement_contract",
        "correctness",
        "groups",
        "artifacts",
        "report_digest",
    }
    if (
        set(report) != expected_report_fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("report_kind") != "illm-v02-single-request-baseline"
        or report.get("development_no_manifest") is not False
        or report.get("formal_evidence_eligible") is not True
        or report.get("evidence_stage") != "qualification_candidate"
        or report.get("qualification_tier") != "unqualified"
        or report.get("qualification_claim") is not None
        or report.get("development_reasons") != []
        or report.get("performance_outcome") != "baseline_only"
        or report.get("status") != "pass"
        or report.get("all_semantic_checks_passed") is not True
        or report.get("evidence_kind") != "measured"
        or report.get("speedup_claim") is not None
        or report.get("external_engine_comparison") is not None
        or report.get("interpretation") != "observed differences only; performance is not a release win condition"
        or not isinstance(suite, str)
        or suite not in EXPECTED_SUITE_DIGESTS
        or not isinstance(target, str)
        or target not in EVIDENCE_SCOPES
        or report.get("evidence_scope") != EVIDENCE_SCOPES[target]
        or report.get("suite_id") != f"illm.v02.single-request.{suite}.v1"
        or report.get("suite_digest") != EXPECTED_SUITE_DIGESTS[suite]
        or not isinstance(report.get("engine_version"), str)
        or not report.get("engine_version")
        or not isinstance(digest, str)
        or digest != sha256_json({key: value for key, value in report.items() if key != "report_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark report identity is invalid")

    evidence_scope = EVIDENCE_SCOPES[target]
    capacity = 256 if target == "tiny" else 512
    model = report.get("model")
    identity = report.get("identity")
    hardware = report.get("hardware")
    contract = report.get("measurement_contract")
    groups_value = report.get("groups")
    if (
        not isinstance(model, str)
        or not isinstance(identity, dict)
        or not isinstance(hardware, dict)
        or not isinstance(contract, dict)
        or not isinstance(groups_value, dict)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark report sections are invalid")
    device = hardware.get("device")
    if (
        not isinstance(device, str)
        or (target == "tiny" and (model != "fixture:tiny-decoder" or device != "cpu"))
        or (target == "real" and _CUDA_DEVICE.fullmatch(device) is None)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark target/model/device scope is invalid")

    toolchain = identity.get("toolchain")
    reported_source = identity.get("source")
    driver = identity.get("benchmark_driver")
    model_identity = identity.get("model")
    sampling_value = identity.get("sampling_spec")
    authority = identity.get("authority")
    authority_digest = identity.get("authority_digest")
    if (
        set(identity)
        != {
            "toolchain",
            "source",
            "benchmark_driver",
            "model",
            "sampling_spec",
            "sampling_spec_digest",
            "metric_registry_digest",
            "hardware_digest",
            "hardware_driver_digest",
            "memory_plan_digests",
            "authority",
            "authority_digest",
        }
        or not all(
            isinstance(value, dict)
            for value in (toolchain, reported_source, driver, model_identity, sampling_value, authority)
        )
        or not isinstance(authority_digest, str)
        or _DIGEST.fullmatch(authority_digest) is None
        or authority_digest != sha256_json(authority)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark source/model identity sections are invalid")
    toolchain = cast(dict[str, Any], toolchain)
    reported_source = cast(dict[str, Any], reported_source)
    driver = cast(dict[str, Any], driver)
    model_identity = cast(dict[str, Any], model_identity)
    sampling_value = cast(dict[str, Any], sampling_value)
    authority = cast(dict[str, Any], authority)
    expected_toolchain_fields = {
        "binding",
        "formal_evidence_eligible",
        "manifest_path",
        "manifest_digest",
        "environment_id",
        "python_executable",
        "project_root",
        "project_root_digest",
        "bootstrap_digest",
        "lock_digest",
        "runtime_profile_id",
        "runtime_lock_digest",
    }
    if (
        set(toolchain) != expected_toolchain_fields
        or toolchain.get("binding") != "finalized-source-bound-manifest"
        or toolchain.get("formal_evidence_eligible") is not True
        or not isinstance(toolchain.get("manifest_path"), str)
        or not isinstance(toolchain.get("environment_id"), str)
        or not toolchain.get("environment_id")
        or not isinstance(toolchain.get("python_executable"), str)
        or not isinstance(toolchain.get("runtime_profile_id"), str)
        or not toolchain.get("runtime_profile_id")
        or toolchain.get("project_root") != reported_source.get("project_root")
        or toolchain.get("project_root_digest") != reported_source.get("project_root_digest")
        or toolchain.get("bootstrap_digest") != reported_source.get("bootstrap_digest")
        or any(
            not isinstance(toolchain.get(name), str) or _DIGEST.fullmatch(cast(str, toolchain[name])) is None
            for name in ("manifest_digest", "lock_digest", "runtime_lock_digest")
        )
        or identity.get("metric_registry_digest") != EXPECTED_METRIC_REGISTRY_DIGEST
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark toolchain snapshot is invalid")
    suite_value = _verified_authority_snapshot(
        authority,
        suite=suite,
        target=target,
        evidence_scope=evidence_scope,
        capacity=capacity,
        model=model,
        device=device,
        source=reported_source,
        driver=driver,
        model_identity=model_identity,
        engine_version=cast(str, report["engine_version"]),
        metric_registry_digest=cast(str, identity["metric_registry_digest"]),
        require_current_source=require_current_source,
    )
    try:
        sampling = SamplingSpec.from_dict(sampling_value)
    except Exception as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "formal benchmark sampling identity is invalid") from exc
    if sampling.mode != "greedy" or sampling.actual_seed != 7 or identity.get("sampling_spec_digest") != sampling.digest:
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark sampling identity differs from the frozen suite")
    if sampling.to_dict() != suite_value["sampling_spec"]:
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark sampling identity differs from its frozen suite")

    driver_value = hardware.get("driver")
    hardware_projection = {key: item for key, item in hardware.items() if key != "hardware_digest"}
    expected_hardware_fields = {
        "device",
        "cpu_threads",
        "torch_version",
        "python_version",
        "platform",
        "machine",
        "processor",
        "cuda_runtime",
        "live_free_memory_queried",
        "driver",
        "driver_digest",
        "hardware_digest",
    }
    if target == "real":
        expected_hardware_fields |= {
            "gpu_name",
            "total_device_memory_bytes",
            "compute_capability",
            "multiprocessor_count",
        }
    if (
        set(hardware) != expected_hardware_fields
        or type(hardware.get("cpu_threads")) is not int
        or hardware.get("cpu_threads") != 1
        or any(
            not isinstance(hardware.get(name), str) or not cast(str, hardware[name])
            for name in ("torch_version", "python_version", "platform", "machine")
        )
        or hardware.get("processor") is not None
        and (not isinstance(hardware.get("processor"), str) or not hardware.get("processor"))
        or hardware.get("live_free_memory_queried") is not False
        or not isinstance(driver_value, dict)
        or hardware.get("driver_digest") != sha256_json(driver_value)
        or hardware.get("hardware_digest") != sha256_json(hardware_projection)
        or identity.get("hardware_digest") != hardware.get("hardware_digest")
        or identity.get("hardware_driver_digest") != hardware.get("driver_digest")
        or (
            target == "tiny"
            and (
                hardware.get("cuda_runtime") is not None
                or not _same_json(driver_value, {"kind": "not_applicable", "reason": "cpu_device"})
            )
        )
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark hardware/driver identity is invalid")
    if target == "real" and any(
        type(hardware.get(name)) is not expected
        for name, expected in (
            ("gpu_name", str),
            ("total_device_memory_bytes", int),
            ("compute_capability", list),
            ("multiprocessor_count", int),
        )
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "real baseline lacks required physical GPU facts")
    if target == "real":
        versions = driver_value.get("reported_versions")
        capability = cast(list[Any], hardware["compute_capability"])
        if (
            set(driver_value) != {"kind", "reported_versions"}
            or driver_value.get("kind") != "nvidia"
            or not isinstance(versions, list)
            or not versions
            or any(not isinstance(value, str) or not value.strip() for value in versions)
            or not isinstance(hardware.get("cuda_runtime"), str)
            or not cast(str, hardware["cuda_runtime"]).strip()
            or not cast(str, hardware["gpu_name"]).strip()
            or cast(int, hardware["total_device_memory_bytes"]) < 6 * 1024**3
            or cast(int, hardware["multiprocessor_count"]) <= 0
            or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
            or tuple(cast(list[int], capability)) < (8, 0)
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "real baseline physical GPU facts are invalid")

    model_digest_fields = {
        "model_spec_id",
        "weights_digest",
        "weight_manifest_digest",
        "tokenizer_digest",
    }
    if (
        set(model_identity)
        != {
            "model_spec_id",
            "model_revision",
            "weights_digest",
            "weight_manifest_digest",
            "tokenizer_digest",
            "tokenizer_revision",
            "runtime_dtype",
        }
        or any(
            not isinstance(model_identity.get(name), str) or _DIGEST.fullmatch(cast(str, model_identity[name])) is None
            for name in model_digest_fields
        )
        or model_identity.get("runtime_dtype") not in {"fp32", "bf16", "fp16"}
        or (target == "tiny" and not _same_json(model_identity, TINY_MODEL_IDENTITY))
        or (
            target == "real"
            and (
                model_identity.get("model_revision") != REAL_MODEL_REVISION
                or model_identity.get("tokenizer_revision") != REAL_MODEL_REVISION
                or model_identity.get("runtime_dtype") not in {"bf16", "fp16"}
            )
        )
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark model identity is invalid")

    expected_matrix = [
        {"group_id": group_id, "kv_mode": kv_mode, "telemetry": telemetry, "required_capture": capture_enabled}
        for group_id, kv_mode, telemetry, capture_enabled in GROUPS
    ]
    warmups = contract.get("warmups_per_case_group")
    repetitions = contract.get("repetitions_per_case_group")
    if (
        set(contract)
        != {
            "groups",
            "fixed_matrix",
            "warmups_per_case_group",
            "repetitions_per_case_group",
            "cpu_threads",
            "tf32_enabled",
            "sampling",
            "server_clock_domain",
            "client_clock_domain",
            "cross_clock_subtraction",
            "gpu_memory",
            "load_time_excluded",
            "warmups_recorded_but_excluded_from_metrics",
            "observation_discard_policy",
            "run_order",
        }
        or not _same_json(contract.get("groups"), [group[0] for group in GROUPS])
        or not _same_json(contract.get("fixed_matrix"), expected_matrix)
        or type(warmups) is not int
        or warmups < 1
        or type(repetitions) is not int
        or repetitions < 3
        or type(contract.get("cpu_threads")) is not int
        or contract.get("cpu_threads") != 1
        or contract.get("tf32_enabled") is not False
        or contract.get("sampling") != "greedy exact; fixed seed shared across distinct request IDs"
        or contract.get("server_clock_domain") != "host.perf_counter_ns"
        or contract.get("client_clock_domain") != "benchmark-client.perf_counter_ns"
        or contract.get("cross_clock_subtraction") is not False
        or contract.get("gpu_memory") != "allocated/peak/reserved only; no live-free admission query"
        or contract.get("load_time_excluded") is not True
        or contract.get("warmups_recorded_but_excluded_from_metrics") is not True
        or contract.get("observation_discard_policy") != "none"
        or contract.get("run_order") != "group-order-then-phase-then-case-then-iteration-v1"
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark measurement contract is invalid")

    _, materialized = _verified_materialized_workload(
        output,
        suite_value=suite_value,
        suite=suite,
        target=target,
        evidence_scope=evidence_scope,
        capacity=capacity,
        authority_digest=authority_digest,
    )
    if set(groups_value) != {group[0] for group in GROUPS}:
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark group matrix is incomplete")
    verified_groups: dict[str, dict[str, Any]] = {}
    shared_identity: dict[str, Any] | None = None
    for group_id, kv_mode, telemetry, capture_enabled in GROUPS:
        group = groups_value[group_id]
        if not isinstance(group, dict) or set(group) != {
            "group_id",
            "warmup_count",
            "measurement_count",
            "metrics",
            "observed",
            "kv_mode",
            "telemetry",
            "capture_enabled",
            "memory_plan",
            "memory_plan_digest",
            "engine_identity",
        }:
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} report shape is invalid")
        engine_identity = group.get("engine_identity")
        if not isinstance(engine_identity, dict):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} engine identity is invalid")
        if set(engine_identity) != {
            "engine_version",
            "git_commit",
            "git_dirty",
            "toolchain_digest",
            "toolchain_bound",
            "device",
            "dtype",
            "model_spec_id",
            "weights_digest",
            "load_manifest_digest",
            "tokenizer_digest",
            "kv_mode",
            "memory_plan_evidence_kind",
        }:
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} engine identity shape is invalid")
        common = {key: item for key, item in engine_identity.items() if key != "kv_mode"}
        if shared_identity is None:
            shared_identity = common
        elif common != shared_identity:
            _fail(ErrorCode.CAPTURE_CORRUPT, "engine identity changed across fixed groups")
        if (
            group.get("group_id") != group_id
            or group.get("kv_mode") != kv_mode
            or group.get("telemetry") != telemetry
            or group.get("capture_enabled") is not capture_enabled
            or engine_identity.get("kv_mode") != kv_mode
            or engine_identity.get("device") != device
            or engine_identity.get("model_spec_id") != model_identity["model_spec_id"]
            or engine_identity.get("weights_digest") != model_identity["weights_digest"]
            or engine_identity.get("load_manifest_digest") != model_identity["weight_manifest_digest"]
            or engine_identity.get("tokenizer_digest") != model_identity["tokenizer_digest"]
            or engine_identity.get("dtype") != model_identity["runtime_dtype"]
            or engine_identity.get("toolchain_digest") != toolchain["manifest_digest"]
            or engine_identity.get("toolchain_bound") is not True
            or not isinstance(engine_identity.get("git_commit"), str)
            or type(engine_identity.get("git_dirty")) is not bool
            or engine_identity.get("memory_plan_evidence_kind") != "static_conservative_estimate"
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} engine identity is not source/model/device bound")
        plan = _verify_memory_plan(
            group.get("memory_plan"),
            group.get("memory_plan_digest"),
            group_id=group_id,
            kv_mode=kv_mode,
            capacity=capacity,
            device=device,
            model_identity=model_identity,
        )
        verified_groups[group_id] = {
            "persisted": group,
            "engine_identity": engine_identity,
            "memory_plan": plan,
            "memory_plan_digest": group["memory_plan_digest"],
        }
    assert shared_identity is not None
    if (
        identity.get("memory_plan_digests")
        != {group_id: groups_value[group_id]["memory_plan_digest"] for group_id, *_ in GROUPS}
        or shared_identity.get("engine_version") != report["engine_version"]
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark report identity does not bind group plans")

    raw = _object(output / "raw" / "observations.json")
    raw_digest = raw.get("raw_digest")
    expected_raw_fields = {
        "schema_version",
        "evidence_kind",
        "evidence_scope",
        "suite",
        "suite_digest",
        "target",
        "model",
        "model_spec_id",
        "device",
        "capacity_tokens",
        "sampling_spec_digest",
        "warmups_per_case_group",
        "repetitions_per_case_group",
        "cpu_threads",
        "warmups_included",
        "authority_digest",
        "records",
        "raw_digest",
    }
    records_value = raw.get("records")
    if (
        set(raw) != expected_raw_fields
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("evidence_kind") != "measured"
        or raw.get("evidence_scope") != evidence_scope
        or raw.get("suite") != suite
        or raw.get("suite_digest") != EXPECTED_SUITE_DIGESTS[suite]
        or raw.get("target") != target
        or raw.get("model") != model
        or raw.get("model_spec_id") != model_identity["model_spec_id"]
        or raw.get("device") != device
        or type(raw.get("capacity_tokens")) is not int
        or raw.get("capacity_tokens") != capacity
        or raw.get("sampling_spec_digest") != sampling.digest
        or type(raw.get("warmups_per_case_group")) is not int
        or raw.get("warmups_per_case_group") != warmups
        or type(raw.get("repetitions_per_case_group")) is not int
        or raw.get("repetitions_per_case_group") != repetitions
        or type(raw.get("cpu_threads")) is not int
        or raw.get("cpu_threads") != 1
        or raw.get("warmups_included") is not True
        or raw.get("authority_digest") != authority_digest
        or not isinstance(records_value, list)
        or not isinstance(raw_digest, str)
        or raw_digest != sha256_json({key: item for key, item in raw.items() if key != "raw_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark raw envelope is invalid")

    records = cast(list[dict[str, Any]], records_value)
    expected_count = len(GROUPS) * len(materialized) * (warmups + repetitions)
    if len(records) != expected_count:
        _fail(ErrorCode.CAPTURE_CORRUPT, "formal benchmark raw record cardinality is invalid")
    cursor = 0
    request_ids: set[str] = set()
    facts: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    recomputed_groups: dict[str, dict[str, Any]] = {}
    for group_id, kv_mode, telemetry, capture_enabled in GROUPS:
        group_records: list[dict[str, Any]] = []
        observations: list[RequestObservation] = []
        group_facts: dict[tuple[str, str, int], dict[str, Any]] = {}
        group_contract = verified_groups[group_id]
        for phase, count in (("warmup", warmups), ("measured", repetitions)):
            for case in materialized:
                for iteration in range(count):
                    record, observation = _verified_record(
                        output,
                        records[cursor],
                        group_id=group_id,
                        kv_mode=kv_mode,
                        telemetry=telemetry,
                        capture_enabled=capture_enabled,
                        phase=phase,
                        iteration=iteration,
                        case=case,
                        suite=suite,
                        suite_digest=EXPECTED_SUITE_DIGESTS[suite],
                        target=target,
                        evidence_scope=evidence_scope,
                        model=model,
                        model_spec_id=cast(str, model_identity["model_spec_id"]),
                        device=device,
                        capacity=capacity,
                        sampling=sampling,
                        group_identity=cast(dict[str, Any], group_contract["engine_identity"]),
                        memory_plan=cast(dict[str, Any], group_contract["memory_plan"]),
                        memory_plan_digest=cast(str, group_contract["memory_plan_digest"]),
                    )
                    cursor += 1
                    request_id = cast(str, record["request_id"])
                    if request_id in request_ids:
                        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark request IDs are not globally unique")
                    request_ids.add(request_id)
                    group_records.append(record)
                    group_facts[(cast(str, case["case_id"]), phase, iteration)] = cast(
                        dict[str, Any], record["correctness"]
                    )
                    if observation is not None:
                        observations.append(observation)
        try:
            metrics = build_streaming_metrics(observations)
        except (TypeError, ValueError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "benchmark raw observations cannot form canonical metrics") from exc
        summary = _group_summary(group_id, group_records, metrics)
        persisted = cast(dict[str, Any], group_contract["persisted"])
        if any(not _same_json(persisted.get(key), summary[key]) for key in summary):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{group_id} report summary differs from recomputed raw metrics")
        recomputed_groups[group_id] = {**summary, **persisted}
        facts[group_id] = group_facts
    baseline = facts["none-off"]
    if any(group_facts != baseline for group_facts in facts.values()):
        _fail(ErrorCode.CAPTURE_CORRUPT, "KV/capture modes changed exact output/terminal/RNG facts")

    expected_no_kv = _comparison(
        "no-kv-vs-contiguous",
        (
            ("none-off", recomputed_groups["none-off"], "contiguous-off", recomputed_groups["contiguous-off"]),
            (
                "none-request",
                recomputed_groups["none-request"],
                "contiguous-request",
                recomputed_groups["contiguous-request"],
            ),
        ),
        evidence_scope=evidence_scope,
    )
    expected_telemetry = _comparison(
        "telemetry-overhead",
        (
            ("none-off", recomputed_groups["none-off"], "none-request", recomputed_groups["none-request"]),
            (
                "contiguous-off",
                recomputed_groups["contiguous-off"],
                "contiguous-request",
                recomputed_groups["contiguous-request"],
            ),
        ),
        evidence_scope=evidence_scope,
    )
    if (
        not _same_json(_object(output / "no-kv-vs-contiguous.json"), expected_no_kv)
        or not _same_json(_object(output / "telemetry-overhead.json"), expected_telemetry)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark comparison sidecars differ from recomputed raw facts")

    expected_correctness = {
        "exact_across_all_four_groups": True,
        "comparison_fields": list(CORRECTNESS_FIELDS),
        "globally_unique_request_ids": expected_count,
        "resource_baseline_checked_after_every_request": True,
        "capture_complete_checked_for_request_groups": True,
        "real_workload_completed_256x64": target == "real",
        "accepted_request_failure_rate": {"numerator": 0, "denominator": expected_count, "value": 0.0},
    }
    if not _same_json(report.get("correctness"), expected_correctness):
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark correctness summary differs from verified raw facts")
    if report.get("artifacts") != {
        "raw": "raw/observations.json",
        "materialized_workload": "raw/materialized-workload.json",
        "no_kv_vs_contiguous": "no-kv-vs-contiguous.json",
        "telemetry_overhead": "telemetry-overhead.json",
        "captures": "captures/",
    }:
        _fail(ErrorCode.CAPTURE_CORRUPT, "benchmark artifact routing is invalid")
    return report



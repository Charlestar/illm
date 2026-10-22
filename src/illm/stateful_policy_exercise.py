"""Portable reader for the policy-aware V0.2 lifecycle exercise.

This is a sibling of :mod:`illm.stateful_exercise`.  It does not reinterpret
the historical schema-2 bundle.  The reader imports no tensor runtime, trusts
no current working directory or environment, and recomputes the closed raw
records, schema-3 policy captures, policy/plan/binding identities and the
container inventory from sealed bytes.

``COMPLETE`` means only that this CPU component container is complete.  It is
not a global, GPU, release, or LTS qualification statement.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import sha256_file, sha256_json
from illm.capture.policy_streaming import (
    PolicyStreamingCaptureBundle,
    _play_policy,
    _privacy_check,
    _validate_policy_payload,
)
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.policy_events import POLICY_EVENT_CATALOG, PolicyStreamingEvent
from illm.domain.streaming_events import EVENT_ENVELOPE_VERSION, StreamingEventKind as K
from illm.metrics import registry_conformance, registry_digest, validate_metric_registry
from illm.numerical_policy import NumericalPolicyError, parse_policy
from illm.policy_capture_contract import (
    PROTOCOL_SHA256,
    validate_execution_binding,
    validate_policy_capture_metadata,
)


REPORT_KIND = "illm-v02-policy-streaming-lifecycle-exercise-v1"
SCOPE = "tiny-policy-state-machine-cpu-v2"
CYCLE_COUNT = 100
RACE_COUNT = 8
REQUIRED_FAULT_IDS = (
    "cancel-before-launch",
    "cancel-after-launch-delayed-fence",
    "finish-cancel-race",
    "capacity-exceeded-preaccept",
    "async-execution-error",
    "numerical-error",
    "oom",
    "unknown-fence-fail-stop",
    "optional-exporter-stall-drop",
    "required-capture-write-failure",
    "capture-truncation-detected",
    "forced-exit-quarantine",
)
COMPLETE_CAPTURE_IDS = (
    "success",
    "length",
    "eos",
    "cancel-after-launch",
    "no-kv",
    "exporter-stall",
)
REQUIRED_POLICY_CAPTURE_IDS = (
    "success",
    "length",
    "eos",
    "cancel-after-launch",
    "no-kv",
)
INCOMPLETE_CAPTURE_IDS = (
    "required-write-failure",
    "unknown-fence",
    "forced-exit",
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_PUBLIC_INPUT_IDS = [1, 107, 104, 111, 111, 114]
_ZERO_RESOURCE_FIELDS = frozenset(
    {
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
)
_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "report_kind",
        "scope",
        "status",
        "evidence_stage",
        "qualification_status",
        "qualification_scope",
        "development_no_manifest",
        "formal_evidence_eligible",
        "cpu_profile_evidence_eligible",
        "global_qualification_granted",
        "gpu_qualification_granted",
        "source_identity",
        "toolchain",
        "producer",
        "model_identity",
        "numerical_policy",
        "numerical_policy_digest",
        "execution_binding",
        "execution_binding_digest",
        "protocol_sha256",
        "memory_plan_digests",
        "metric_registry_digest",
        "actual_cycle_count",
        "finished_cycle_count",
        "cancelled_cycle_count",
        "race_count",
        "required_fault_scenario_ids",
        "required_fault_pass_count",
        "required_matrix_complete",
        "required_policy_capture_ids",
        "complete_capture_ids",
        "incomplete_capture_ids",
        "actual_gpu_faults_run",
        "actual_gpu_fault_claim",
        "simulation_boundary",
        "performance_claim",
        "raw_payload_scope",
        "raw_digests",
        "report_digest",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "input_token_ids",
        "output_token_ids",
        "text",
        "outcome",
        "finish_reason",
        "error_code",
        "actual_seed",
        "model_spec_id",
        "sampling_spec_digest",
        "kv_mode",
        "next_output_position",
        "rng_draws_consumed",
        "committed_draw_addresses",
        "query_tokens_per_step",
        "kv_valid_tokens_per_step",
        "resource_status",
        "timings",
        "capture_manifest_digest",
        "capture_error",
        "event_count",
        "numerical_policy",
        "numerical_policy_digest",
        "execution_binding",
        "execution_binding_digest",
        "memory_plan_digest",
    }
)
_TIMING_FIELDS = frozenset(
    {
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
)
_POLICY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "finalized",
        "event_envelope_version",
        "event_catalog",
        "capabilities",
        "privacy_mode",
        "replayable",
        "non_replayable_reasons",
        "critical_event_drop_count",
        "request_id",
        "identity",
        "sampling_spec",
        "sampling_spec_digest",
        "memory_plan",
        "memory_plan_digest",
        "numerical_policy",
        "numerical_policy_digest",
        "execution_binding",
        "execution_binding_digest",
        "protocol_sha256",
        "observation_profile",
        "event_count",
        "logical_terminal_outcome",
        "terminal_outcome",
        "files",
        "manifest_digest",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "implementation_source_scope",
        "implementation_source_digest",
        "source_identity_digest",
    }
)
_PRODUCER_FIELDS = frozenset({"path", "sha256"})
_MODEL_IDENTITY_FIELDS = frozenset(
    {"model_spec_id", "sampling_spec_digest", "runtime_dtype", "public_fixture"}
)
_MEMORY_PLAN_DIGEST_FIELDS = frozenset({"normal", "fault_subplans"})

__all__ = [
    "COMPLETE_CAPTURE_IDS",
    "CYCLE_COUNT",
    "INCOMPLETE_CAPTURE_IDS",
    "RACE_COUNT",
    "REPORT_KIND",
    "REQUIRED_FAULT_IDS",
    "REQUIRED_POLICY_CAPTURE_IDS",
    "SCOPE",
    "verify_policy_exercise_output",
]


def _fail(message: str, *, code: ErrorCode = ErrorCode.CAPTURE_CORRUPT,
          details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _closed(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{name} fields differ from the closed contract",
            details={"missing": sorted(expected - actual), "unknown": sorted(actual - expected)},
        )


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _fail(f"{name} must be a SHA-256 hex digest")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _plain_root(path: Path) -> Path:
    if not isinstance(path, Path):
        _fail("policy lifecycle output must be a Path", code=ErrorCode.INVALID_ARGUMENT)
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if _is_linklike(candidate):
            _fail("policy lifecycle path contains a link/reparse point")
    if not absolute.is_dir():
        _fail("policy lifecycle output root is not an ordinary directory")
    return absolute


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail("policy lifecycle JSON contains a non-finite number")
    return parsed


def _decode_json(text: str, *, artifact: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in policy lifecycle evidence", details={"artifact": artifact, "field": key})
            result[key] = value
        return result

    def reject(value: str) -> NoReturn:
        _fail("policy lifecycle JSON contains a non-finite constant", details={"artifact": artifact, "value": value})

    try:
        return json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=reject,
            parse_float=_finite_float,
        )
    except IllmError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "policy lifecycle JSON is malformed",
            {"artifact": artifact},
        ) from exc


def _object(path: Path) -> dict[str, Any]:
    try:
        value = _decode_json(path.read_text(encoding="utf-8"), artifact=path.name)
    except IllmError:
        raise
    except (OSError, UnicodeError) as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "policy lifecycle JSON is unreadable",
            {"artifact": path.name},
        ) from exc
    if type(value) is not dict:
        _fail("policy lifecycle JSON root must be an object", details={"artifact": path.name})
    return cast(dict[str, Any], value)


def _relative(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        _fail("policy lifecycle evidence path is not portable")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _fail("policy lifecycle evidence path escapes its sealed root")
    return value


def _plain_file(path: Path) -> None:
    if _is_linklike(path) or not path.is_file():
        _fail("policy lifecycle inventory contains a link or non-file artifact", details={"path": path.name})
    try:
        links = path.stat().st_nlink
    except OSError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy lifecycle file metadata is unreadable") from exc
    if links != 1:
        _fail("policy lifecycle evidence forbids hard-linked files", details={"path": path.name})


def _inventory(root: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    result: dict[str, str] = {}
    try:
        paths = sorted(root.rglob("*"))
    except OSError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy lifecycle inventory is unreadable") from exc
    for path in paths:
        if _is_linklike(path):
            _fail("policy lifecycle inventory contains a link/reparse point")
        if path.is_dir():
            continue
        _plain_file(path)
        relative = path.relative_to(root).as_posix()
        if relative not in ignored:
            result[relative] = sha256_file(path)
    return result


def _expected_inventory_paths(*, formal: bool) -> tuple[set[str], set[str]]:
    provenance = {
        "provenance/source-identity.json",
        "provenance/execution-binding.json",
        "provenance/numerical-policy.json",
        "provenance/exercise_policy_streaming.py",
        "provenance/metric-registry-v1.json",
        "provenance/metric-registry-conformance.json",
    }
    if formal:
        provenance.add("provenance/toolchain-manifest.json")
    evidence_files = {
        *provenance,
        "raw/cycles.json",
        "raw/captures.json",
        "raw/fault-matrix.json",
        "report/exercise-report.json",
    }
    capture_ids = (*COMPLETE_CAPTURE_IDS, *INCOMPLETE_CAPTURE_IDS, "truncated-copy")
    for capture_id in capture_ids:
        marker = "INCOMPLETE" if capture_id in INCOMPLETE_CAPTURE_IDS else "COMPLETE"
        evidence_files.update(
            {
                f"captures/{capture_id}/events.jsonl",
                f"captures/{capture_id}/checksums.json",
                f"captures/{capture_id}/capture-manifest.json",
                f"captures/{capture_id}/{marker}",
            }
        )
    directories = {
        "captures",
        "provenance",
        "raw",
        "report",
        *(f"captures/{capture_id}" for capture_id in capture_ids),
    }
    return evidence_files, directories


def _validate_exact_inventory(root: Path, *, formal: bool, indexed: Mapping[str, Any]) -> None:
    expected_files, expected_directories = _expected_inventory_paths(formal=formal)
    if set(indexed) != expected_files:
        _fail(
            "policy lifecycle indexed file inventory is not the fixed contract",
            details={
                "missing": sorted(expected_files - set(indexed)),
                "unknown": sorted(set(indexed) - expected_files),
            },
        )
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_linklike(path):
            _fail("policy lifecycle inventory contains a link/reparse point")
        if path.is_dir():
            actual_directories.add(relative)
        else:
            _plain_file(path)
            actual_files.add(relative)
    expected_outer_marker = "COMPLETE" if formal else "DEVELOPMENT_ONLY"
    expected_all_files = {*expected_files, "checksums.json", expected_outer_marker}
    if actual_files != expected_all_files or actual_directories != expected_directories:
        _fail(
            "policy lifecycle physical inventory is not exact",
            details={
                "missing_files": sorted(expected_all_files - actual_files),
                "unknown_files": sorted(actual_files - expected_all_files),
                "missing_directories": sorted(expected_directories - actual_directories),
                "unknown_directories": sorted(actual_directories - expected_directories),
            },
        )


def _resources(value: object, *, allow_owned: bool = False) -> dict[str, int]:
    if type(value) is not dict or set(value) != _ZERO_RESOURCE_FIELDS:
        _fail("policy lifecycle resource snapshot fields are invalid")
    result = cast(dict[str, Any], value)
    if any(type(item) is not int or item < 0 for item in result.values()):
        _fail("policy lifecycle resource snapshot values are invalid")
    if not allow_owned and any(result.values()):
        _fail("policy lifecycle terminal resources did not return to zero")
    return cast(dict[str, int], result)


def _quarantined_resources(
    value: object, *, memory_plan: Mapping[str, Any]
) -> dict[str, int]:
    resources = _resources(value, allow_owned=True)
    kv_bytes = _integer(memory_plan.get("kv_bytes"), name="quarantined MemoryPlan KV bytes", minimum=1)
    expected = {
        "active_requests": 0,
        "reservations": 1,
        "quarantined_requests": 1,
        "active_tickets": 1,
        "input_leases": 1,
        "output_leases": 1,
        "workspace_leases": 1,
        "kv_write_leases": 1,
        "kv_reserved_bytes": kv_bytes,
        "kv_valid_tokens": 0,
        "kv_staged_tokens": len(_PUBLIC_INPUT_IDS),
    }
    if resources != expected:
        _fail("quarantined policy resources differ from actual owned-ticket invariants")
    return resources


def _signed(value: Mapping[str, Any], field: str, *, name: str) -> None:
    if field not in value or value[field] != sha256_json({key: item for key, item in value.items() if key != field}):
        _fail(f"{name} digest mismatch")


def _registered_candidate(value: object, digest: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("numerical policy must be an object")
    try:
        policy = parse_policy(cast(Mapping[str, object], value))
    except NumericalPolicyError as exc:
        _fail("numerical policy is not a registered policy", details={"reason": str(exc)})
    if not policy.is_candidate or policy.storage_dtype != "fp32" or digest != policy.digest:
        _fail("lifecycle policy must be the registered candidate fp32 instance")
    return cast(dict[str, Any], policy.to_dict())


def _binding(value: object, digest: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("execution binding must be an object")
    normalized = validate_execution_binding(cast(Mapping[str, Any], value))
    if digest != sha256_json(normalized):
        _fail("execution binding digest mismatch")
    return normalized


def _validate_result(
    value: object,
    *,
    request_id: str,
    expected_outcome: str,
    expected_policy: Mapping[str, Any],
    policy_digest: str,
    expected_binding: Mapping[str, Any],
    binding_digest: str,
    allowed_plan_digests: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("policy lifecycle result must be an object")
    result = cast(dict[str, Any], value)
    _closed(result, _RESULT_FIELDS, name="policy lifecycle result")
    input_ids = result["input_token_ids"]
    output_ids = result["output_token_ids"]
    query_tokens = result["query_tokens_per_step"]
    valid_tokens = result["kv_valid_tokens_per_step"]
    addresses = result["committed_draw_addresses"]
    timings = result["timings"]
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 3
        or result["request_id"] != request_id
        or result["outcome"] != expected_outcome
        or type(input_ids) is not list
        or not input_ids
        or any(type(item) is not int or item < 0 for item in input_ids)
        or type(output_ids) is not list
        or any(type(item) is not int or item < 0 for item in output_ids)
        or type(result["text"]) is not str
        or type(result["actual_seed"]) is not int
        or result["actual_seed"] != 7
        or type(result["model_spec_id"]) is not str
        or _DIGEST.fullmatch(result["model_spec_id"]) is None
        or type(result["sampling_spec_digest"]) is not str
        or _DIGEST.fullmatch(result["sampling_spec_digest"]) is None
        or result["kv_mode"] not in {"none", "contiguous"}
        or type(result["next_output_position"]) is not int
        or result["next_output_position"] != len(output_ids)
        or type(result["rng_draws_consumed"]) is not int
        or result["rng_draws_consumed"] != 0
        or addresses != []
        or type(query_tokens) is not list
        or type(valid_tokens) is not list
        or len(query_tokens) != len(output_ids)
        or len(valid_tokens) != len(output_ids)
        or any(type(item) is not int or item < 1 for item in query_tokens)
        or any(type(item) is not int or item < 0 for item in valid_tokens)
        or result["numerical_policy"] != expected_policy
        or result["numerical_policy_digest"] != policy_digest
        or result["execution_binding"] != expected_binding
        or result["execution_binding_digest"] != binding_digest
        or result["memory_plan_digest"] not in allowed_plan_digests
        or type(result["event_count"]) is not int
        or result["event_count"] < 1
        or type(timings) is not dict
        or input_ids != _PUBLIC_INPUT_IDS
        or result["capture_manifest_digest"] is not None
        or result["capture_error"] is not None
    ):
        _fail("policy lifecycle result identity/state is invalid")
    _closed(cast(dict[str, Any], timings), _TIMING_FIELDS, name="policy lifecycle timings")
    if (
        timings["clock_domain"] != "host.perf_counter_ns"
        or timings["telemetry"] not in {"metrics", "request"}
        or timings["evidence_kind"] != "measured"
        or type(timings["critical_capture_event_drop_count"]) is not int
        or timings["critical_capture_event_drop_count"] < 0
        or type(timings["token_ready_ns"]) is not list
        or type(timings["per_forward_wall_time_ns"]) is not list
    ):
        _fail("policy lifecycle timing facts are invalid")
    resources = _resources(result["resource_status"])
    if expected_outcome == "FINISHED":
        if (
            result["finish_reason"] != "MAX_NEW_TOKENS"
            or result["error_code"] is not None
            or output_ids != [114]
            or result["text"] != "o"
        ):
            _fail("finished policy lifecycle result has invalid terminal fields")
    elif expected_outcome == "CANCELLED":
        if (
            result["finish_reason"] is not None
            or result["error_code"] is not None
            or output_ids
            or result["text"] != ""
        ):
            _fail("cancelled policy lifecycle result published output")
    elif expected_outcome == "FAILED":
        if result["finish_reason"] is not None or type(result["error_code"]) is not str or output_ids:
            _fail("failed policy lifecycle result has invalid terminal fields")
    else:
        _fail("policy lifecycle result outcome is unsupported")
    result["resource_status"] = resources
    return result


def _events(value: object, *, request_id: str) -> list[PolicyStreamingEvent]:
    if type(value) is not list:
        _fail("policy lifecycle events must be a list")
    raw = value
    if any(type(item) is not dict for item in raw):
        _fail("policy lifecycle event entries must be objects")
    try:
        events = [PolicyStreamingEvent.from_dict(cast(dict[str, Any], item)) for item in raw]
    except IllmError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "policy lifecycle event envelope is invalid",
        ) from exc
    for event in events:
        _privacy_check(event.payload, path="payload")
        _validate_policy_payload(event)
    if [event.producer_seq_no for event in events] != list(range(len(events))):
        _fail("policy lifecycle event sequence is incomplete")
    if any(
        event.event_catalog != POLICY_EVENT_CATALOG
        or (event.kind is not K.CLOCK_SNAPSHOT and event.payload.get("request_id") != request_id)
        for event in events
    ):
        _fail("policy lifecycle event identity/catalog differs")
    return events


def _events_file(path: Path, *, request_id: str) -> list[PolicyStreamingEvent]:
    try:
        encoded = path.read_bytes()
        if not encoded or not encoded.endswith(b"\n"):
            _fail("policy lifecycle event JSONL is empty or truncated")
        text = encoded.decode("utf-8")
    except IllmError:
        raise
    except (OSError, UnicodeError) as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "policy lifecycle event JSONL is unreadable",
            {"artifact": path.name},
        ) from exc
    return _events(
        [_decode_json(line, artifact=path.name) for line in text.splitlines()],
        request_id=request_id,
    )


def _terminal_event(events: Sequence[PolicyStreamingEvent]) -> PolicyStreamingEvent:
    terminal = [
        event
        for event in events
        if event.kind in {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}
    ]
    if len(terminal) != 1:
        _fail("policy lifecycle event stream must contain exactly one terminal event")
    return terminal[0]


def _validate_cycle_timings(
    result: Mapping[str, Any], events: Sequence[PolicyStreamingEvent]
) -> None:
    timings = cast(Mapping[str, Any], result["timings"])
    tokenized = [event for event in events if event.kind is K.REQUEST_TOKENIZED]
    accepted = [event for event in events if event.kind is K.REQUEST_ACCEPTED]
    output = [event for event in events if event.kind is K.REQUEST_OUTPUT_CHUNK]
    forwards = [event for event in events if event.kind is K.MODEL_FORWARD]
    terminal = _terminal_event(events)
    if len(tokenized) != 1 or len(accepted) != 1:
        _fail("policy lifecycle timing authority lacks unique tokenize/accept events")
    started_ns = tokenized[0].payload.get("started_ns")
    ended_ns = tokenized[0].payload.get("ended_ns")
    terminal_ready_ns = terminal.payload.get("server_terminal_ready_ns")
    token_ready_ns = [event.payload.get("server_ready_ns") for event in output]
    forward_durations = [event.payload.get("duration_ns") for event in forwards]
    if (
        type(started_ns) is not int
        or type(ended_ns) is not int
        or ended_ns < started_ns
        or type(terminal_ready_ns) is not int
        or terminal_ready_ns < accepted[0].monotonic_ns
        or any(type(value) is not int or value < 0 for value in token_ready_ns)
        or any(type(value) is not int or value < 0 for value in forward_durations)
    ):
        _fail("policy lifecycle timing events contain invalid measured values")
    expected = {
        "clock_domain": "host.perf_counter_ns",
        "accepted_ns": accepted[0].monotonic_ns,
        "token_ready_ns": token_ready_ns,
        "terminal_ready_ns": terminal_ready_ns,
        "tokenization_wall_time_ns": ended_ns - started_ns,
        "per_forward_wall_time_ns": forward_durations,
        "server_e2e_wall_time_ns": terminal_ready_ns - accepted[0].monotonic_ns,
        "critical_capture_event_drop_count": 0,
        "telemetry": "metrics",
        "evidence_kind": "measured",
    }
    if dict(timings) != expected:
        _fail("policy lifecycle timing facts differ from their ordered event authority")
    if result["query_tokens_per_step"] != [event.payload["query_tokens"] for event in forwards]:
        _fail("policy lifecycle query-token steps differ from model.forward events")
    if result["kv_valid_tokens_per_step"] != [
        event.payload["kv_valid_token_count"] for event in output
    ]:
        _fail("policy lifecycle KV-valid steps differ from output events")


def _verify_cycles(
    raw: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    policy_digest: str,
    binding: Mapping[str, Any],
    binding_digest: str,
    contiguous_plan: str,
    model_identity: Mapping[str, Any],
    capture_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_fields = frozenset(
        {
            "schema_version",
            "suite_id",
            "cycle_count",
            "alternation",
            "finished_count",
            "cancelled_count",
            "numerical_policy_digest",
            "execution_binding_digest",
            "memory_plan_digest",
            "records",
            "raw_digest",
        }
    )
    _closed(raw, expected_fields, name="policy cycle raw evidence")
    records_value = raw["records"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["suite_id"] != "illm.v02.policy.lifecycle.100-alternating.v1"
        or type(raw["cycle_count"]) is not int
        or raw["cycle_count"] != CYCLE_COUNT
        or raw["alternation"] != "even-finished-odd-cancel-before-launch"
        or type(raw["finished_count"]) is not int
        or raw["finished_count"] != 50
        or type(raw["cancelled_count"]) is not int
        or raw["cancelled_count"] != 50
        or raw["numerical_policy_digest"] != policy_digest
        or raw["execution_binding_digest"] != binding_digest
        or raw["memory_plan_digest"] != contiguous_plan
        or type(records_value) is not list
        or len(records_value) != CYCLE_COUNT
    ):
        _fail("policy 100-cycle raw evidence identity/count is invalid")
    _signed(raw, "raw_digest", name="policy cycle raw evidence")
    records = records_value
    result_records: list[dict[str, Any]] = []
    record_fields = frozenset(
        {
            "cycle_index",
            "request_id",
            "planned_outcome",
            "result",
            "events",
            "resources_after_terminal",
            "cycle_digest",
        }
    )
    for index, value in enumerate(records):
        if type(value) is not dict:
            _fail("policy lifecycle cycle record must be an object")
        record = cast(dict[str, Any], value)
        _closed(record, record_fields, name="policy lifecycle cycle record")
        request_id = f"policy-lifecycle-cycle-{index:03d}"
        planned = "FINISHED" if index % 2 == 0 else "CANCELLED"
        if (
            type(record["cycle_index"]) is not int
            or record["cycle_index"] != index
            or record["request_id"] != request_id
            or record["planned_outcome"] != planned
        ):
            _fail("policy lifecycle cycle ordering/identity is invalid", details={"cycle": index})
        _signed(record, "cycle_digest", name="policy lifecycle cycle record")
        result = _validate_result(
            record["result"],
            request_id=request_id,
            expected_outcome=planned,
            expected_policy=policy,
            policy_digest=policy_digest,
            expected_binding=binding,
            binding_digest=binding_digest,
            allowed_plan_digests=frozenset({contiguous_plan}),
        )
        if (
            result["model_spec_id"] != model_identity["model_spec_id"]
            or result["sampling_spec_digest"] != model_identity["sampling_spec_digest"]
            or result["model_spec_id"] != capture_metadata["identity"]["model_spec_id"]
            or result["sampling_spec_digest"] != capture_metadata["sampling_spec_digest"]
        ):
            _fail("policy lifecycle cycle model/SamplingSpec identity differs")
        resources = _resources(record["resources_after_terminal"])
        if result["resource_status"] != resources:
            _fail("policy lifecycle cycle result/resource snapshots differ")
        events = _events(record["events"], request_id=request_id)
        state = _play_policy(
            tuple(events), request_id, metadata=capture_metadata, complete=True
        )
        terminal = _terminal_event(events)
        output_events = [event for event in events if event.kind is K.REQUEST_OUTPUT_CHUNK]
        if (
            result["event_count"] != len(events)
            or state.terminal_outcome != planned
            or state.logical_terminal_outcome != planned
            or state.committed_output_count != len(result["output_token_ids"])
            or terminal.payload.get("finish_reason") != result["finish_reason"]
        ):
            _fail("policy lifecycle cycle terminal/event count is invalid")
        _validate_cycle_timings(result, events)
        if planned == "FINISHED":
            if len(output_events) != len(result["output_token_ids"]):
                _fail("finished policy cycle event/result output counts differ")
        elif output_events or any(event.kind is K.MODEL_FORWARD_STARTED for event in events):
            _fail("cancel-before-launch policy cycle launched or emitted output")
        result_records.append(record)
    return result_records


def _verify_incomplete_capture(
    path: Path,
    *,
    capture_id: str,
    expected_reason: str,
    policy: Mapping[str, Any],
    policy_digest: str,
    binding: Mapping[str, Any],
    binding_digest: str,
    allowed_plans: frozenset[str],
) -> dict[str, Any]:
    if not (path / "INCOMPLETE").is_file() or (path / "COMPLETE").exists():
        _fail("expected policy capture incomplete marker state is absent")
    manifest = _object(path / "capture-manifest.json")
    _closed(manifest, _POLICY_MANIFEST_FIELDS, name="incomplete policy capture manifest")
    _signed(manifest, "manifest_digest", name="incomplete policy capture manifest")
    reasons = manifest["non_replayable_reasons"]
    files = manifest["files"]
    expected_reasons = {
        "required-write-failure": ["CAPTURE_WRITE_FAILED"],
        "unknown-fence": ["FENCE_OR_RESOURCE_STATE_UNKNOWN"],
        "forced-exit": [
            "CONTROLLED_FORCED_EXIT",
            "ABANDONED_WITH_UNRESOLVED_OWNERSHIP",
        ],
    }[capture_id]
    expected_requests = {
        "required-write-failure": "policy-fault-required-write",
        "unknown-fence": "policy-fault-unknown-fence",
        "forced-exit": "policy-fault-forced-exit",
    }
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 3
        or manifest["finalized"] is not False
        or manifest["replayable"] is not False
        or reasons != expected_reasons
        or expected_reason != expected_reasons[0]
        or type(manifest["event_envelope_version"]) is not int
        or manifest["event_envelope_version"] != EVENT_ENVELOPE_VERSION
        or manifest["event_catalog"] != POLICY_EVENT_CATALOG
        or manifest["capabilities"] != ["R0"]
        or manifest["privacy_mode"] != "metadata_only"
        or type(manifest["event_count"]) is not int
        or manifest["event_count"] < 0
        or type(manifest["critical_event_drop_count"]) is not int
        or manifest["critical_event_drop_count"] < 0
        or manifest["request_id"] != expected_requests[capture_id]
        or manifest["numerical_policy"] != policy
        or manifest["numerical_policy_digest"] != policy_digest
        or manifest["execution_binding"] != binding
        or manifest["execution_binding_digest"] != binding_digest
        or manifest["memory_plan_digest"] not in allowed_plans
        or type(files) is not dict
        or set(files) != {"events.jsonl", "checksums.json"}
    ):
        _fail("incomplete policy capture identity/metadata is invalid")
    normalized = validate_policy_capture_metadata(
        identity=cast(Mapping[str, Any], manifest["identity"]),
        sampling_spec=cast(Mapping[str, Any], manifest["sampling_spec"]),
        memory_plan=cast(Mapping[str, Any], manifest["memory_plan"]),
        numerical_policy=cast(Mapping[str, Any], manifest["numerical_policy"]),
        execution_binding=cast(Mapping[str, Any], manifest["execution_binding"]),
        observation_profile=cast(str, manifest["observation_profile"]),
    )
    for name, value in normalized.items():
        if manifest[name] != value:
            _fail("incomplete policy capture normalized metadata differs")
    expected_names = {"INCOMPLETE", "events.jsonl", "checksums.json", "capture-manifest.json"}
    children = list(path.iterdir())
    if {item.name for item in children} != expected_names:
        _fail("incomplete policy capture inventory is not exact")
    for child in children:
        _plain_file(child)
    for name, expected in cast(dict[str, Any], files).items():
        if _digest(expected, name=f"incomplete capture {name} digest") != sha256_file(path / name):
            _fail("incomplete policy capture file checksum differs")
    checksums = _object(path / "checksums.json")
    if checksums != {"events.jsonl": files["events.jsonl"]}:
        _fail("incomplete policy capture checksum index differs")
    try:
        marker = (path / "INCOMPLETE").read_bytes()
    except OSError as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "incomplete policy capture marker is unreadable",
        ) from exc
    if marker != (expected_reasons[-1] + "\n").encode("utf-8"):
        _fail("incomplete policy capture marker/reason differs")
    events = _events_file(path / "events.jsonl", request_id=manifest["request_id"])
    state = _play_policy(
        tuple(events), manifest["request_id"], metadata=normalized, complete=False
    )
    if (
        manifest["event_count"] != len(events)
        or manifest["logical_terminal_outcome"] != state.logical_terminal_outcome
        or manifest["terminal_outcome"] != state.terminal_outcome
        or (
            capture_id == "required-write-failure"
            and manifest["critical_event_drop_count"] <= 0
        )
        or (
            capture_id != "required-write-failure"
            and manifest["critical_event_drop_count"] != 0
        )
    ):
        _fail("incomplete policy capture events/drop/state facts differ")
    try:
        PolicyStreamingCaptureBundle.open(path)
    except IllmError as exc:
        if exc.code is not ErrorCode.CAPTURE_INCOMPLETE:
            raise
    else:
        _fail("incomplete policy capture was accepted by complete reader")
    return manifest


def _verify_captures(
    root: Path,
    raw: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    policy_digest: str,
    binding: Mapping[str, Any],
    binding_digest: str,
    none_plan: str,
    contiguous_plan: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    expected_fields = frozenset(
        {
            "schema_version",
            "capture_contract",
            "complete_capture_ids",
            "required_policy_capture_ids",
            "incomplete_capture_ids",
            "corrupt_rejection_capture_ids",
            "numerical_policy_digest",
            "execution_binding_digest",
            "records",
            "raw_digest",
        }
    )
    _closed(raw, expected_fields, name="policy capture index")
    records_value = raw["records"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["capture_contract"] != "schema3-policy-metadata-only-R0"
        or raw["complete_capture_ids"] != list(COMPLETE_CAPTURE_IDS)
        or raw["required_policy_capture_ids"] != list(REQUIRED_POLICY_CAPTURE_IDS)
        or raw["incomplete_capture_ids"] != list(INCOMPLETE_CAPTURE_IDS)
        or raw["corrupt_rejection_capture_ids"] != ["truncated-copy"]
        or raw["numerical_policy_digest"] != policy_digest
        or raw["execution_binding_digest"] != binding_digest
        or type(records_value) is not list
    ):
        _fail("policy capture index identity/coverage is invalid")
    _signed(raw, "raw_digest", name="policy capture index")
    records = records_value
    by_id: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    complete_fields = frozenset(
        {
            "capture_id",
            "path",
            "request_id",
            "outcome",
            "finish_reason",
            "kv_mode",
            "manifest_digest",
            "event_count",
            "memory_plan_digest",
            "numerical_policy_digest",
            "execution_binding_digest",
            "resources_after_terminal",
            "capture_record_digest",
        }
    )
    incomplete_fields = frozenset(
        {
            "capture_id",
            "path",
            "expected_state",
            "expected_reason",
            "memory_plan_digest",
            "numerical_policy_digest",
            "execution_binding_digest",
            "capture_record_digest",
        }
    )
    corrupt_fields = frozenset(
        {
            "capture_id",
            "path",
            "expected_state",
            "expected_error_code",
            "source_manifest_digest",
            "capture_record_digest",
        }
    )
    for value in records:
        if type(value) is not dict:
            _fail("policy capture record must be an object")
        record = cast(dict[str, Any], value)
        capture_id = record.get("capture_id")
        if type(capture_id) is not str or capture_id in by_id:
            _fail("policy capture record identity is invalid")
        path_value = _relative(record.get("path"))
        if path_value != f"captures/{capture_id}":
            _fail("policy capture record path differs from fixed ID mapping")
        path = root / path_value
        if capture_id in COMPLETE_CAPTURE_IDS:
            _closed(record, complete_fields, name="complete policy capture record")
            _signed(record, "capture_record_digest", name="complete policy capture record")
            bundle = PolicyStreamingCaptureBundle.open(path)
            manifest = bundle.manifest
            terminal = _terminal_event(bundle.events)
            expected_plan = none_plan if capture_id == "no-kv" else contiguous_plan
            resources = _resources(record["resources_after_terminal"])
            if (
                record["manifest_digest"] != manifest["manifest_digest"]
                or record["request_id"] != manifest["request_id"]
                or record["outcome"] != manifest["terminal_outcome"]
                or type(record["event_count"]) is not int
                or record["event_count"] != len(bundle.events)
                or record["memory_plan_digest"] != expected_plan
                or manifest["memory_plan_digest"] != expected_plan
                or record["numerical_policy_digest"] != policy_digest
                or manifest["numerical_policy"] != policy
                or manifest["numerical_policy_digest"] != policy_digest
                or record["execution_binding_digest"] != binding_digest
                or manifest["execution_binding"] != binding
                or manifest["execution_binding_digest"] != binding_digest
                or manifest["protocol_sha256"] != PROTOCOL_SHA256
                or manifest["event_catalog"] != POLICY_EVENT_CATALOG
                or manifest["capabilities"] != ["R0"]
                or manifest["privacy_mode"] != "metadata_only"
                or bundle.request_story(cast(str, manifest["request_id"]))["kv"]["final_active_allocation_bytes"] != 0
                or resources != record["resources_after_terminal"]
                or record["finish_reason"] != terminal.payload.get("finish_reason")
            ):
                _fail("complete policy capture record differs from replayed schema3 facts")
            expected_finish = {"length": "MAX_NEW_TOKENS", "eos": "EOS"}.get(capture_id)
            if expected_finish is not None and record["finish_reason"] != expected_finish:
                _fail("policy capture finish reason differs from its required case")
            if capture_id == "cancel-after-launch" and record["outcome"] != "CANCELLED":
                _fail("cancel-after-launch policy capture did not cancel")
            if record["kv_mode"] != ("none" if capture_id == "no-kv" else "contiguous"):
                _fail("policy capture KV mode differs from its required case")
            manifests[capture_id] = manifest
        elif capture_id in INCOMPLETE_CAPTURE_IDS:
            _closed(record, incomplete_fields, name="incomplete policy capture record")
            _signed(record, "capture_record_digest", name="incomplete policy capture record")
            reason = record["expected_reason"]
            if type(reason) is not str or record["expected_state"] != "INCOMPLETE":
                _fail("incomplete policy capture expected state/reason is invalid")
            manifest = _verify_incomplete_capture(
                path,
                capture_id=capture_id,
                expected_reason=reason,
                policy=policy,
                policy_digest=policy_digest,
                binding=binding,
                binding_digest=binding_digest,
                allowed_plans=frozenset({contiguous_plan}),
            )
            if (
                record["memory_plan_digest"] != manifest["memory_plan_digest"]
                or record["numerical_policy_digest"] != policy_digest
                or record["execution_binding_digest"] != binding_digest
            ):
                _fail("incomplete policy capture record cross-binding differs")
            manifests[capture_id] = manifest
        elif capture_id == "truncated-copy":
            _closed(record, corrupt_fields, name="corrupt policy capture record")
            _signed(record, "capture_record_digest", name="corrupt policy capture record")
            code: str | None = None
            try:
                PolicyStreamingCaptureBundle.open(path)
            except IllmError as exc:
                code = exc.code.value
            if (
                code != ErrorCode.CAPTURE_CORRUPT.value
                or record["expected_state"] != "CORRUPT_REJECTED"
                or record["expected_error_code"] != code
                or record["source_manifest_digest"] != by_id.get("success", {}).get("manifest_digest")
            ):
                _fail("truncated policy capture rejection is not reproducible")
        else:
            _fail("policy capture evidence contains an unexpected case")
        by_id[capture_id] = record
    expected_ids = {*COMPLETE_CAPTURE_IDS, *INCOMPLETE_CAPTURE_IDS, "truncated-copy"}
    if set(by_id) != expected_ids:
        _fail("policy capture evidence case coverage is incomplete")
    if set(manifests) != {*COMPLETE_CAPTURE_IDS, *INCOMPLETE_CAPTURE_IDS}:
        _fail("policy capture manifest coverage is incomplete")
    return by_id, manifests


_OBSERVATION_FIELDS: dict[str, frozenset[str]] = {
    "cancel-before-launch": frozenset(
        {"cycle_index", "outcome", "model_forward_started_count", "resources_after_terminal"}
    ),
    "cancel-after-launch-delayed-fence": frozenset(
        {
            "cancellation_won",
            "outcome",
            "token_chunk_count",
            "resources_while_fence_pending",
            "resources_after_terminal",
            "capture_path",
            "capture_manifest_digest",
        }
    ),
    "finish-cancel-race": frozenset({"race_count", "iterations"}),
    "capacity-exceeded-preaccept": frozenset(
        {"error_code", "accepted_event_count", "resources_after_rejection", "memory_plan_digest"}
    ),
    "async-execution-error": frozenset(
        {"outcome", "error_code", "committed_output_count", "resources_after_terminal"}
    ),
    "numerical-error": frozenset(
        {"outcome", "error_code", "committed_output_count", "resources_after_terminal"}
    ),
    "oom": frozenset(
        {"outcome", "error_code", "committed_output_count", "resources_after_terminal"}
    ),
    "unknown-fence-fail-stop": frozenset(
        {
            "raised_error_code",
            "terminal_ready_event_count",
            "resources_quarantined",
            "engine_close_succeeded",
            "capture_path",
        }
    ),
    "optional-exporter-stall-drop": frozenset(
        {
            "outcome",
            "copy_drop_count",
            "exporter_error_count",
            "initial_close_succeeded",
            "final_close_succeeded",
            "resources_after_terminal",
            "independent_capture_path",
            "independent_capture_manifest_digest",
        }
    ),
    "required-capture-write-failure": frozenset(
        {
            "outcome",
            "capture_error",
            "critical_capture_event_drop_count",
            "resources_after_terminal",
            "capture_path",
            "capture_complete_present",
            "capture_incomplete_present",
        }
    ),
    "capture-truncation-detected": frozenset({"capture_path", "reader_error_code"}),
    "forced-exit-quarantine": frozenset(
        {
            "raised_error_code",
            "terminal_chunk_count",
            "resources_quarantined",
            "engine_close_succeeded",
            "capture_path",
        }
    ),
}


def _verify_faults(
    raw: Mapping[str, Any],
    *,
    cycles: Sequence[Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, Any]],
    capture_manifests: Mapping[str, Mapping[str, Any]],
    policy_digest: str,
    binding_digest: str,
    capacity_plan: str,
) -> None:
    fields = frozenset(
        {
            "schema_version",
            "matrix_id",
            "required_scenario_ids",
            "required_pass_count",
            "numerical_policy_digest",
            "execution_binding_digest",
            "scenarios",
            "nonrequired_actual_gpu_fault",
            "raw_digest",
        }
    )
    _closed(raw, fields, name="policy lifecycle fault matrix")
    scenarios_value = raw["scenarios"]
    optional = raw["nonrequired_actual_gpu_fault"]
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["matrix_id"] != "illm.v02.policy.stateful-failure-matrix.v1"
        or raw["required_scenario_ids"] != list(REQUIRED_FAULT_IDS)
        or type(raw["required_pass_count"]) is not int
        or raw["required_pass_count"] != len(REQUIRED_FAULT_IDS)
        or raw["numerical_policy_digest"] != policy_digest
        or raw["execution_binding_digest"] != binding_digest
        or type(scenarios_value) is not list
        or len(scenarios_value) != len(REQUIRED_FAULT_IDS)
        or optional
        != {
            "status": "not_run_resource_unavailable",
            "required_for_cpu_profile": False,
            "reason": "CPU policy state-machine component does not claim an actual GPU fault run",
            "simulation_results_must_not_be_relabelled_as_actual_gpu_faults": True,
        }
    ):
        _fail("policy lifecycle fault matrix identity/coverage is invalid")
    _signed(raw, "raw_digest", name="policy lifecycle fault matrix")
    scenario_fields = frozenset(
        {
            "scenario_id",
            "status",
            "required_for_cpu_profile",
            "mechanism",
            "evidence_kind",
            "actual_tiny_runtime",
            "actual_gpu_fault",
            "numerical_policy_digest",
            "execution_binding_digest",
            "observations",
            "scenario_digest",
        }
    )
    scenarios = scenarios_value
    for expected_id, value in zip(REQUIRED_FAULT_IDS, scenarios, strict=True):
        if type(value) is not dict:
            _fail("policy lifecycle fault scenario must be an object")
        scenario = cast(dict[str, Any], value)
        _closed(scenario, scenario_fields, name="policy lifecycle fault scenario")
        observations = scenario["observations"]
        if (
            scenario["scenario_id"] != expected_id
            or scenario["status"] != "pass"
            or scenario["required_for_cpu_profile"] is not True
            or type(scenario["mechanism"]) is not str
            or not scenario["mechanism"]
            or type(scenario["evidence_kind"]) is not str
            or not scenario["evidence_kind"]
            or scenario["actual_tiny_runtime"] is not True
            or scenario["actual_gpu_fault"] is not False
            or scenario["numerical_policy_digest"] != policy_digest
            or scenario["execution_binding_digest"] != binding_digest
            or type(observations) is not dict
        ):
            _fail("policy lifecycle fault scenario identity/status is invalid", details={"scenario": expected_id})
        _closed(cast(dict[str, Any], observations), _OBSERVATION_FIELDS[expected_id], name=f"{expected_id} observations")
        _signed(scenario, "scenario_digest", name="policy lifecycle fault scenario")
        observation = cast(dict[str, Any], observations)
        if expected_id == "cancel-before-launch":
            if (
                type(observation["cycle_index"]) is not int
                or observation["cycle_index"] != 1
                or observation["outcome"] != "CANCELLED"
                or type(observation["model_forward_started_count"]) is not int
                or observation["model_forward_started_count"] != 0
                or observation["resources_after_terminal"] != cycles[1]["resources_after_terminal"]
            ):
                _fail("cancel-before-launch policy facts are invalid")
        elif expected_id == "cancel-after-launch-delayed-fence":
            pending = _resources(observation["resources_while_fence_pending"], allow_owned=True)
            _resources(observation["resources_after_terminal"])
            if (
                observation["cancellation_won"] is not True
                or observation["outcome"] != "CANCELLED"
                or type(observation["token_chunk_count"]) is not int
                or observation["token_chunk_count"] != 0
                or pending["active_tickets"] != 1
                or pending["kv_write_leases"] != 1
                or observation["capture_path"] != captures["cancel-after-launch"]["path"]
                or observation["capture_manifest_digest"] != captures["cancel-after-launch"]["manifest_digest"]
            ):
                _fail("cancel-after-launch delayed fence facts are invalid")
        elif expected_id == "finish-cancel-race":
            iterations = observation["iterations"]
            if (
                type(iterations) is not list
                or type(observation["race_count"]) is not int
                or observation["race_count"] != RACE_COUNT
                or len(iterations) != RACE_COUNT
            ):
                _fail("finish/cancel race coverage is invalid")
            iteration_fields = frozenset(
                {
                    "iteration",
                    "outcome",
                    "cancel_outcome",
                    "cancel_won",
                    "terminal_event_count",
                    "token_chunk_count",
                    "resources_after_terminal",
                }
            )
            for index, item in enumerate(iterations):
                if type(item) is not dict:
                    _fail("race iteration must be an object")
                iteration = cast(dict[str, Any], item)
                _closed(iteration, iteration_fields, name="race iteration")
                _resources(iteration["resources_after_terminal"])
                if (
                    type(iteration["iteration"]) is not int
                    or iteration["iteration"] != index
                    or iteration["outcome"] not in {"FINISHED", "CANCELLED"}
                    or iteration["cancel_outcome"] != iteration["outcome"]
                    or type(iteration["cancel_won"]) is not bool
                    or iteration["cancel_won"] is not (iteration["outcome"] == "CANCELLED")
                    or type(iteration["terminal_event_count"]) is not int
                    or iteration["terminal_event_count"] != 1
                    or type(iteration["token_chunk_count"]) is not int
                    or iteration["token_chunk_count"]
                    != (0 if iteration["outcome"] == "CANCELLED" else 1)
                ):
                    _fail("race arbitration facts are invalid")
        elif expected_id == "capacity-exceeded-preaccept":
            _resources(observation["resources_after_rejection"])
            if (
                observation["error_code"] != ErrorCode.CONTEXT_LIMIT_EXCEEDED.value
                or type(observation["accepted_event_count"]) is not int
                or observation["accepted_event_count"] != 0
                or observation["memory_plan_digest"] != capacity_plan
            ):
                _fail("capacity preaccept policy facts are invalid")
        elif expected_id in {"async-execution-error", "numerical-error", "oom"}:
            expected_code = {
                "async-execution-error": ErrorCode.INTERNAL.value,
                "numerical-error": ErrorCode.NUMERICAL_ERROR.value,
                "oom": ErrorCode.RESOURCE_EXHAUSTED.value,
            }[expected_id]
            _resources(observation["resources_after_terminal"])
            if (
                observation["outcome"] != "FAILED"
                or observation["error_code"] != expected_code
                or type(observation["committed_output_count"]) is not int
                or observation["committed_output_count"] != 0
            ):
                _fail("known policy failure terminal/resource facts are invalid")
        elif expected_id == "unknown-fence-fail-stop":
            owned = _quarantined_resources(
                observation["resources_quarantined"],
                memory_plan=cast(
                    Mapping[str, Any], capture_manifests["unknown-fence"]["memory_plan"]
                ),
            )
            if (
                observation["raised_error_code"] != ErrorCode.ENGINE_UNHEALTHY.value
                or type(observation["terminal_ready_event_count"]) is not int
                or observation["terminal_ready_event_count"] != 0
                or observation["engine_close_succeeded"] is not False
                or observation["capture_path"] != captures["unknown-fence"]["path"]
            ):
                _fail("unknown-fence policy fail-stop facts are invalid")
        elif expected_id == "optional-exporter-stall-drop":
            _resources(observation["resources_after_terminal"])
            if (
                observation["outcome"] != "FINISHED"
                or type(observation["copy_drop_count"]) is not int
                or observation["copy_drop_count"] <= 0
                or type(observation["exporter_error_count"]) is not int
                or observation["initial_close_succeeded"] is not False
                or observation["final_close_succeeded"] is not True
                or observation["independent_capture_path"] != captures["exporter-stall"]["path"]
                or observation["independent_capture_manifest_digest"] != captures["exporter-stall"]["manifest_digest"]
            ):
                _fail("optional exporter policy isolation facts are invalid")
        elif expected_id == "required-capture-write-failure":
            _resources(observation["resources_after_terminal"])
            if (
                observation["outcome"] != "FINISHED"
                or observation["capture_error"] != "CAPTURE_WRITE_FAILED"
                or type(observation["critical_capture_event_drop_count"]) is not int
                or observation["critical_capture_event_drop_count"] <= 0
                or observation["capture_path"] != captures["required-write-failure"]["path"]
                or observation["capture_complete_present"] is not False
                or observation["capture_incomplete_present"] is not True
            ):
                _fail("required policy capture write failure facts are invalid")
        elif expected_id == "capture-truncation-detected":
            if observation != {
                "capture_path": captures["truncated-copy"]["path"],
                "reader_error_code": ErrorCode.CAPTURE_CORRUPT.value,
            }:
                _fail("policy capture truncation rejection facts are invalid")
        elif expected_id == "forced-exit-quarantine":
            owned = _quarantined_resources(
                observation["resources_quarantined"],
                memory_plan=cast(
                    Mapping[str, Any], capture_manifests["forced-exit"]["memory_plan"]
                ),
            )
            if (
                observation["raised_error_code"] != ErrorCode.ENGINE_UNHEALTHY.value
                or type(observation["terminal_chunk_count"]) is not int
                or observation["terminal_chunk_count"] != 0
                or observation["engine_close_succeeded"] is not False
                or observation["capture_path"] != captures["forced-exit"]["path"]
            ):
                _fail("forced-exit policy quarantine facts are invalid")


def _verify_provenance(root: Path, report: Mapping[str, Any], *, formal: bool) -> None:
    source = _object(root / "provenance" / "source-identity.json")
    _closed(source, _SOURCE_FIELDS, name="policy lifecycle source identity")
    _signed(source, "source_identity_digest", name="policy lifecycle source identity")
    if (
        type(source["schema_version"]) is not int
        or source["schema_version"] != 1
        or source["implementation_source_scope"] != "project-root-v1"
        or _DIGEST.fullmatch(cast(str, source["implementation_source_digest"])) is None
        or report["source_identity"] != source
    ):
        _fail("policy lifecycle source identity is invalid")
    producer = report["producer"]
    if type(producer) is not dict:
        _fail("policy lifecycle producer identity must be an object")
    _closed(cast(dict[str, Any], producer), _PRODUCER_FIELDS, name="policy lifecycle producer identity")
    source_copy = root / "provenance" / "exercise_policy_streaming.py"
    if (
        producer["path"] != "scripts/exercise_policy_streaming.py"
        or producer["sha256"] != sha256_file(source_copy)
    ):
        _fail("policy lifecycle producer source linkage is invalid")
    embedded_binding = _object(root / "provenance" / "execution-binding.json")
    if embedded_binding != report["execution_binding"]:
        _fail("embedded policy lifecycle execution binding differs from report")
    toolchain = cast(dict[str, Any], report["execution_binding"])["toolchain"]
    if report["toolchain"] != toolchain:
        _fail("policy lifecycle toolchain/report binding differs")
    manifest_path = root / "provenance" / "toolchain-manifest.json"
    if formal:
        manifest = _object(manifest_path)
        digest = manifest.get("manifest_digest")
        if (
            toolchain.get("binding") != "source-bound-manifest"
            or digest != toolchain.get("manifest_digest")
            or digest != sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
            or manifest.get("finalized") is not True
            or manifest.get("project_root_digest") != source["implementation_source_digest"]
            or manifest.get("environment_id") != toolchain.get("environment_id")
            or manifest.get("lock_digest") != toolchain.get("lock_digest")
        ):
            _fail("formal policy lifecycle toolchain manifest linkage is invalid")
    elif manifest_path.exists() or toolchain != {"binding": "unbound-development", "manifest_digest": None}:
        _fail("development policy lifecycle output contains a false formal binding")
    registry = _object(root / "provenance" / "metric-registry-v1.json")
    conformance = _object(root / "provenance" / "metric-registry-conformance.json")
    try:
        validate_metric_registry(registry)
    except ValueError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "embedded MetricRegistryV1 differs from frozen contract") from exc
    if conformance != registry_conformance(registry) or report["metric_registry_digest"] != registry_digest():
        _fail("policy lifecycle metric registry conformance/digest linkage is invalid")


def verify_policy_exercise_output(output: Path, *, require_formal: bool = True) -> dict[str, Any]:
    """Verify one closed policy lifecycle component without running a model.

    The returned ``cpu_profile_evidence_eligible`` flag is derived again from
    the marker and execution-binding facts.  It remains a CPU-component fact;
    ``qualification_status`` is always ``unqualified`` here.
    """

    if type(require_formal) is not bool:
        _fail("require_formal must be bool", code=ErrorCode.INVALID_ARGUMENT)
    root = _plain_root(output)
    if (root / "INCOMPLETE").exists():
        _fail("policy lifecycle exercise output is incomplete", code=ErrorCode.CAPTURE_INCOMPLETE)
    formal = (root / "COMPLETE").is_file()
    development = (root / "DEVELOPMENT_ONLY").is_file()
    if formal == development or formal is not require_formal:
        _fail("policy lifecycle completion/formality markers are inconsistent")
    checksums = _object(root / "checksums.json")
    _closed(checksums, frozenset({"schema_version", "files", "inventory_digest"}), name="policy lifecycle checksums")
    files = checksums["files"]
    if (
        type(checksums["schema_version"]) is not int
        or checksums["schema_version"] != 1
        or type(files) is not dict
        or checksums["inventory_digest"] != sha256_json(files)
        or files != _inventory(root)
    ):
        _fail("policy lifecycle checksum inventory is invalid")
    _validate_exact_inventory(root, formal=formal, indexed=cast(dict[str, Any], files))
    for name, digest in cast(dict[str, Any], files).items():
        relative = _relative(name)
        if _digest(digest, name=f"{relative} checksum") != sha256_file(root / relative):
            _fail("policy lifecycle file checksum differs", details={"path": relative})
    report = _object(root / "report" / "exercise-report.json")
    _closed(report, _REPORT_FIELDS, name="policy lifecycle report")
    _signed(report, "report_digest", name="policy lifecycle report")
    if (
        type(report["schema_version"]) is not int
        or report["schema_version"] != 1
        or report["report_kind"] != REPORT_KIND
        or report["scope"] != SCOPE
        or report["status"] != "pass"
        or report["qualification_status"] != "unqualified"
        or report["qualification_scope"] != "cpu-only"
        or report["global_qualification_granted"] is not False
        or report["gpu_qualification_granted"] is not False
        or report["actual_gpu_faults_run"] is not False
        or report["actual_gpu_fault_claim"] is not None
        or report["performance_claim"] is not None
        or report["raw_payload_scope"] != "public_tiny_fixture_only"
        or report["protocol_sha256"] != PROTOCOL_SHA256
        or type(report["actual_cycle_count"]) is not int
        or report["actual_cycle_count"] != CYCLE_COUNT
        or type(report["finished_cycle_count"]) is not int
        or report["finished_cycle_count"] != 50
        or type(report["cancelled_cycle_count"]) is not int
        or report["cancelled_cycle_count"] != 50
        or type(report["race_count"]) is not int
        or report["race_count"] != RACE_COUNT
        or report["required_fault_scenario_ids"] != list(REQUIRED_FAULT_IDS)
        or type(report["required_fault_pass_count"]) is not int
        or report["required_fault_pass_count"] != len(REQUIRED_FAULT_IDS)
        or report["required_matrix_complete"] is not True
        or report["required_policy_capture_ids"] != list(REQUIRED_POLICY_CAPTURE_IDS)
        or report["complete_capture_ids"] != list(COMPLETE_CAPTURE_IDS)
        or report["incomplete_capture_ids"] != list(INCOMPLETE_CAPTURE_IDS)
        or type(report["simulation_boundary"]) is not str
        or not report["simulation_boundary"]
    ):
        _fail("policy lifecycle report identity/status/coverage is invalid")
    policy = _registered_candidate(report["numerical_policy"], report["numerical_policy_digest"])
    policy_digest = cast(str, report["numerical_policy_digest"])
    binding = _binding(report["execution_binding"], report["execution_binding_digest"])
    binding_digest = cast(str, report["execution_binding_digest"])
    if binding["protocol_sha256"] != report["protocol_sha256"]:
        _fail("policy lifecycle protocol/execution binding differs")
    model_identity = report["model_identity"]
    if type(model_identity) is not dict:
        _fail("policy lifecycle model identity must be an object")
    _closed(cast(dict[str, Any], model_identity), _MODEL_IDENTITY_FIELDS, name="policy lifecycle model identity")
    if (
        model_identity["runtime_dtype"] != "fp32"
        or model_identity["public_fixture"] != "fixture:tiny-decoder"
        or type(model_identity["model_spec_id"]) is not str
        or _DIGEST.fullmatch(model_identity["model_spec_id"]) is None
        or type(model_identity["sampling_spec_digest"]) is not str
        or _DIGEST.fullmatch(model_identity["sampling_spec_digest"]) is None
    ):
        _fail("policy lifecycle model identity is invalid")
    plans = report["memory_plan_digests"]
    if type(plans) is not dict:
        _fail("policy lifecycle MemoryPlan digest table must be an object")
    _closed(cast(dict[str, Any], plans), _MEMORY_PLAN_DIGEST_FIELDS, name="policy lifecycle MemoryPlan table")
    normal = plans["normal"]
    fault_subplans = plans["fault_subplans"]
    if type(normal) is not dict or set(normal) != {"none", "contiguous"}:
        _fail("normal policy MemoryPlan digest table is invalid")
    if type(fault_subplans) is not dict or set(fault_subplans) != {"capacity-exceeded-preaccept"}:
        _fail("fault policy MemoryPlan digest table is invalid")
    none_plan = _digest(normal["none"], name="no-KV MemoryPlan digest")
    contiguous_plan = _digest(normal["contiguous"], name="contiguous MemoryPlan digest")
    capacity_plan = _digest(
        fault_subplans["capacity-exceeded-preaccept"],
        name="capacity fault MemoryPlan digest",
    )
    if len({none_plan, contiguous_plan, capacity_plan}) != 3:
        _fail("policy lifecycle distinct MemoryPlan scenarios share a digest")
    expected_formal = binding["toolchain"].get("binding") == "source-bound-manifest"
    if (
        expected_formal is not formal
        or report["development_no_manifest"] is formal
        or report["formal_evidence_eligible"] is not formal
        or report["cpu_profile_evidence_eligible"] is not formal
        or report["evidence_stage"] != ("source_bound_cpu_profile" if formal else "DEVELOPMENT_ONLY")
    ):
        _fail("policy lifecycle derived CPU evidence boundary is inconsistent")
    _verify_provenance(root, report, formal=formal)
    cycles = _object(root / "raw" / "cycles.json")
    captures = _object(root / "raw" / "captures.json")
    faults = _object(root / "raw" / "fault-matrix.json")
    capture_records, capture_manifests = _verify_captures(
        root,
        captures,
        policy=policy,
        policy_digest=policy_digest,
        binding=binding,
        binding_digest=binding_digest,
        none_plan=none_plan,
        contiguous_plan=contiguous_plan,
    )
    cycle_records = _verify_cycles(
        cycles,
        policy=policy,
        policy_digest=policy_digest,
        binding=binding,
        binding_digest=binding_digest,
        contiguous_plan=contiguous_plan,
        model_identity=cast(Mapping[str, Any], model_identity),
        capture_metadata=capture_manifests["success"],
    )
    _verify_faults(
        faults,
        cycles=cycle_records,
        captures=capture_records,
        capture_manifests=capture_manifests,
        policy_digest=policy_digest,
        binding_digest=binding_digest,
        capacity_plan=capacity_plan,
    )
    if report["raw_digests"] != {
        "cycles": cycles["raw_digest"],
        "captures": captures["raw_digest"],
        "fault_matrix": faults["raw_digest"],
    }:
        _fail("policy lifecycle report raw evidence linkage is invalid")
    first_result = cast(dict[str, Any], cycle_records[0]["result"])
    if (
        first_result["model_spec_id"] != model_identity["model_spec_id"]
        or first_result["sampling_spec_digest"] != model_identity["sampling_spec_digest"]
    ):
        _fail("policy lifecycle model identity differs from cycle evidence")
    if formal:
        try:
            complete = (root / "COMPLETE").read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy lifecycle COMPLETE marker is unreadable") from exc
        if complete != report["report_digest"] + "\n":
            _fail("policy lifecycle COMPLETE marker does not bind the report")
    else:
        marker = _object(root / "DEVELOPMENT_ONLY")
        if marker != {
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
            "qualification_scope": "cpu-only",
            "formal_evidence_eligible": False,
            "cpu_profile_evidence_eligible": False,
            "global_qualification_granted": False,
            "gpu_qualification_granted": False,
            "report_digest": report["report_digest"],
        }:
            _fail("development policy lifecycle marker/report boundary is invalid")
    return report

"""Portable reader for V0.2 lifecycle and fault-matrix evidence.

This module executes no model code and intentionally imports no tensor/runtime
library.  It recomputes sealed raw records, schema2 capture facts, source and
toolchain provenance, and MetricRegistryV1 conformance.  Controlled CPU fault
injections remain explicitly distinct from actual GPU fault evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEvent, StreamingEventKind as K
from illm.metrics import registry_conformance, registry_digest, validate_metric_registry


REPORT_KIND = "illm-v02-streaming-lifecycle-exercise-v1"
SCOPE = "tiny-state-machine-cpu-v1"
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
    "success", "length", "eos", "cancel-after-launch", "no-kv", "exporter-stall",
)
INCOMPLETE_CAPTURE_IDS = ("required-write-failure", "unknown-fence", "forced-exit")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_ZERO_RESOURCE_FIELDS = frozenset({
    "active_requests", "reservations", "quarantined_requests", "active_tickets",
    "input_leases", "output_leases", "workspace_leases", "kv_write_leases",
    "kv_reserved_bytes", "kv_valid_tokens", "kv_staged_tokens",
})

__all__ = [
    "COMPLETE_CAPTURE_IDS",
    "CYCLE_COUNT",
    "INCOMPLETE_CAPTURE_IDS",
    "RACE_COUNT",
    "REPORT_KIND",
    "REQUIRED_FAULT_IDS",
    "SCOPE",
    "verify_exercise_output",
]


def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(ErrorCode.CAPTURE_CORRUPT, "duplicate JSON field in lifecycle evidence", {"field": key})
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique,
            parse_constant=lambda item: _fail(ErrorCode.CAPTURE_CORRUPT, "nonfinite JSON evidence", {"value": item}),
        )
    except IllmError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence must be an object: {path.name}")
    return cast(dict[str, Any], value)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence path is not portable")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence path escapes its sealed root")
    return value


def _inventory(output: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    result: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence forbids symbolic links")
        if path.is_file() and path.name not in ignored:
            result[path.relative_to(output).as_posix()] = sha256_file(path)
    return result


def _resources_zero(value: Mapping[str, Any]) -> bool:
    return set(value) == _ZERO_RESOURCE_FIELDS and all(type(item) is int and item == 0 for item in value.values())


def _verify_source_and_toolchain(root: Path, report: Mapping[str, Any], formal: bool) -> None:
    source = _object(root / "provenance" / "source-identity.json")
    source_digest = source.get("source_identity_digest")
    project_files = source.get("project_files")
    bootstrap_files = source.get("bootstrap_files")
    if (
        source.get("schema_version") != 1
        or not isinstance(source_digest, str)
        or source_digest != sha256_json({key: value for key, value in source.items() if key != "source_identity_digest"})
        or any(not isinstance(source.get(name), str) or _DIGEST.fullmatch(cast(str, source[name])) is None
               for name in ("project_root_digest", "bootstrap_digest"))
        or type(source.get("git_dirty")) is not bool
        or not isinstance(source.get("git_commit"), str)
        or not isinstance(project_files, dict)
        or not isinstance(bootstrap_files, dict)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "embedded lifecycle source identity is invalid")
    project_inventory = cast(dict[str, Any], project_files)
    bootstrap_inventory = cast(dict[str, Any], bootstrap_files)
    if (
        any(_relative(name) != name or not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None
            for name, digest in project_inventory.items())
        or any(_relative(name) != name or not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None
               for name, digest in bootstrap_inventory.items())
        or source.get("project_root_digest") != sha256_json(sorted(project_inventory.items()))
        or source.get("bootstrap_digest") != sha256_json(sorted(bootstrap_inventory.items()))
        or set(bootstrap_inventory) != {
            "scripts/bootstrap.py", "scripts/_toolchain_common.py", "scripts/_runtime_artifacts.py",
        }
        or project_inventory.get("scripts/exercise_streaming.py")
        != sha256_file(root / "provenance" / "exercise_streaming.py")
        or any(project_inventory.get(name) != digest for name, digest in bootstrap_inventory.items())
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "embedded lifecycle source inventory/digests are invalid")
    toolchain = _object(root / "provenance" / "toolchain-binding.json")
    binding_digest = toolchain.get("binding_digest")
    if (
        toolchain.get("schema_version") != 1
        or not isinstance(binding_digest, str)
        or binding_digest != sha256_json({key: value for key, value in toolchain.items() if key != "binding_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "embedded lifecycle toolchain binding is invalid")
    producer = report.get("producer")
    if (
        not isinstance(producer, dict)
        or producer.get("producer_source_sha256") != sha256_file(root / "provenance" / "exercise_streaming.py")
        or producer.get("source_identity_digest") != source_digest
        or producer.get("toolchain_binding_digest") != binding_digest
        or report.get("source_identity") != source
        or report.get("toolchain") != toolchain
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle report source/toolchain linkage is invalid")
    manifest_path = root / "provenance" / "toolchain-manifest.json"
    if formal:
        manifest = _object(manifest_path)
        digest = manifest.get("manifest_digest")
        if (
            toolchain.get("binding") != "finalized-source-bound-manifest"
            or toolchain.get("formal_evidence_eligible") is not True
            or not isinstance(digest, str)
            or digest != sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
            or digest != toolchain.get("manifest_digest")
            or manifest.get("finalized") is not True
            or manifest.get("project_root_digest") != source.get("project_root_digest")
            or manifest.get("bootstrap_digest") != source.get("bootstrap_digest")
            or manifest.get("runtime_profile_id") != toolchain.get("runtime_profile_id")
            or manifest.get("runtime_lock_digest") != toolchain.get("runtime_lock_digest")
            or manifest.get("lock_digest") != toolchain.get("lock_digest")
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "formal lifecycle toolchain manifest linkage is invalid")
    elif manifest_path.exists() or toolchain.get("binding") != "development-unqualified" or toolchain.get("manifest_digest") is not None:
        _fail(ErrorCode.CAPTURE_CORRUPT, "development lifecycle output contains a false formal binding")


def _verified_events(value: object, request_id: str) -> list[StreamingEvent]:
    if not isinstance(value, list):
        _fail(ErrorCode.CAPTURE_CORRUPT, "cycle events must be a list")
    events = [StreamingEvent.from_dict(item) for item in value if isinstance(item, dict)]
    if len(events) != len(value) or [event.producer_seq_no for event in events] != list(range(len(events))):
        _fail(ErrorCode.CAPTURE_CORRUPT, "cycle event sequence is incomplete")
    if any(event.kind is not K.CLOCK_SNAPSHOT and event.payload.get("request_id") != request_id for event in events):
        _fail(ErrorCode.CAPTURE_CORRUPT, "cycle event request identity differs")
    return events


def _verify_cycles(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    records_value = raw.get("records")
    if (
        raw.get("schema_version") != 1
        or raw.get("suite_id") != "illm.v02.lifecycle.100-alternating.v1"
        or raw.get("cycle_count") != CYCLE_COUNT
        or raw.get("alternation") != "even-finished-odd-cancel-before-launch"
        or raw.get("finished_count") != 50
        or raw.get("cancelled_count") != 50
        or not isinstance(records_value, list)
        or len(records_value) != CYCLE_COUNT
        or raw.get("raw_digest") != sha256_json({key: value for key, value in raw.items() if key != "raw_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "100-cycle raw evidence identity/count is invalid")
    records = cast(list[dict[str, Any]], records_value)
    seen: set[str] = set()
    for index, record in enumerate(records):
        request_id = record.get("request_id")
        result = record.get("result")
        resources = record.get("resources_after_terminal")
        planned = "FINISHED" if index % 2 == 0 else "CANCELLED"
        if (
            not isinstance(request_id, str)
            or _SAFE_ID.fullmatch(request_id) is None
            or request_id in seen
            or request_id != f"lifecycle-cycle-{index:03d}"
            or record.get("cycle_index") != index
            or record.get("planned_outcome") != planned
            or record.get("cycle_digest") != sha256_json({key: value for key, value in record.items() if key != "cycle_digest"})
            or not isinstance(result, dict)
            or result.get("schema_version") != 2
            or result.get("request_id") != request_id
            or result.get("outcome") != planned
            or not isinstance(result.get("output_token_ids"), list)
            or result.get("next_output_position") != len(cast(list[Any], result["output_token_ids"]))
            or result.get("rng_draws_consumed") != 0
            or result.get("committed_draw_addresses") != []
            or not isinstance(resources, dict)
            or not _resources_zero(resources)
            or result.get("resource_status") != resources
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "cycle result/resource semantics are invalid", {"cycle": index})
        events = _verified_events(record.get("events"), request_id)
        terminal = [event for event in events if event.kind in {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}]
        token_events = [event for event in events if event.kind is K.REQUEST_OUTPUT_CHUNK]
        if len(terminal) != 1 or result.get("event_count") != len(events):
            _fail(ErrorCode.CAPTURE_CORRUPT, "cycle terminal/event count is invalid", {"cycle": index})
        if planned == "FINISHED":
            if result.get("finish_reason") != "MAX_NEW_TOKENS" or len(token_events) != 1:
                _fail(ErrorCode.CAPTURE_CORRUPT, "finished cycle output contract is invalid", {"cycle": index})
        elif result.get("finish_reason") is not None or token_events:
            _fail(ErrorCode.CAPTURE_CORRUPT, "cancelled cycle published output", {"cycle": index})
        seen.add(request_id)
    return records


def _verify_incomplete_capture(path: Path, expected_reason: str) -> dict[str, Any]:
    if not (path / "INCOMPLETE").is_file() or (path / "COMPLETE").exists():
        _fail(ErrorCode.CAPTURE_CORRUPT, "expected incomplete capture marker state is absent")
    manifest = _object(path / "capture-manifest.json")
    digest = manifest.get("manifest_digest")
    reasons = manifest.get("non_replayable_reasons")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("finalized") is not False
        or manifest.get("replayable") is not False
        or not isinstance(reasons, list)
        or expected_reason not in reasons
        or not isinstance(digest, str)
        or digest != sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
        or not isinstance(files, dict)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "incomplete capture metadata is invalid")
    for name, expected in files.items():
        relative = _relative(name)
        if not isinstance(expected, str) or sha256_file(path / relative) != expected:
            _fail(ErrorCode.CAPTURE_CORRUPT, "incomplete capture file checksum differs")
    try:
        StreamingCaptureBundle.open(path)
    except IllmError as exc:
        if exc.code is not ErrorCode.CAPTURE_INCOMPLETE:
            raise
    else:
        _fail(ErrorCode.CAPTURE_CORRUPT, "incomplete capture was accepted by portable reader")
    return manifest


def _verify_captures(root: Path, raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records_value = raw.get("records")
    if (
        raw.get("schema_version") != 1
        or raw.get("capture_contract") != "schema2-metadata-only-R0"
        or raw.get("complete_capture_ids") != list(COMPLETE_CAPTURE_IDS)
        or raw.get("incomplete_capture_ids") != list(INCOMPLETE_CAPTURE_IDS)
        or raw.get("corrupt_rejection_capture_ids") != ["truncated-copy"]
        or not isinstance(records_value, list)
        or raw.get("raw_digest") != sha256_json({key: value for key, value in raw.items() if key != "raw_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture evidence index is invalid")
    records = cast(list[dict[str, Any]], records_value)
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        capture_id = record.get("capture_id")
        path_value = _relative(record.get("path"))
        if (
            not isinstance(capture_id, str)
            or capture_id in by_id
            or record.get("capture_record_digest")
            != sha256_json({key: value for key, value in record.items() if key != "capture_record_digest"})
            or path_value != f"captures/{capture_id}"
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "capture record identity is invalid")
        path = root / path_value
        if capture_id in COMPLETE_CAPTURE_IDS:
            bundle = StreamingCaptureBundle.open(path)
            story = bundle.request_story(bundle.manifest["request_id"])
            resources = record.get("resources_after_terminal")
            if (
                record.get("manifest_digest") != bundle.manifest["manifest_digest"]
                or record.get("request_id") != bundle.manifest["request_id"]
                or record.get("outcome") != bundle.manifest["terminal_outcome"]
                or record.get("event_count") != len(bundle.events)
                or bundle.manifest.get("capabilities") != ["R0"]
                or bundle.manifest.get("privacy_mode") != "metadata_only"
                or not isinstance(resources, dict)
                or not _resources_zero(resources)
                or story["kv"]["final_active_allocation_bytes"] != 0
                or story["kv"]["final_open_write_count"] != 0
            ):
                _fail(ErrorCode.CAPTURE_CORRUPT, "complete capture record differs from replayed facts")
            expected_finish = {"length": "MAX_NEW_TOKENS", "eos": "EOS"}.get(capture_id)
            if expected_finish is not None and record.get("finish_reason") != expected_finish:
                _fail(ErrorCode.CAPTURE_CORRUPT, "capture finish reason differs from its required case")
            if capture_id == "cancel-after-launch" and record.get("outcome") != "CANCELLED":
                _fail(ErrorCode.CAPTURE_CORRUPT, "cancel-after-launch capture did not cancel")
            if capture_id == "no-kv" and record.get("kv_mode") != "none":
                _fail(ErrorCode.CAPTURE_CORRUPT, "no-KV capture does not declare none mode")
        elif capture_id in INCOMPLETE_CAPTURE_IDS:
            reason = record.get("expected_reason")
            if not isinstance(reason, str):
                _fail(ErrorCode.CAPTURE_CORRUPT, "incomplete capture expected reason is invalid")
            _verify_incomplete_capture(path, reason)
        elif capture_id == "truncated-copy":
            code: str | None = None
            try:
                StreamingCaptureBundle.open(path)
            except IllmError as exc:
                code = exc.code.value
            if code != ErrorCode.CAPTURE_CORRUPT.value or record.get("expected_error_code") != code:
                _fail(ErrorCode.CAPTURE_CORRUPT, "truncated capture rejection is not reproducible")
        else:
            _fail(ErrorCode.CAPTURE_CORRUPT, "capture evidence contains an unexpected case")
        by_id[capture_id] = record
    expected = {*COMPLETE_CAPTURE_IDS, *INCOMPLETE_CAPTURE_IDS, "truncated-copy"}
    if set(by_id) != expected:
        _fail(ErrorCode.CAPTURE_CORRUPT, "capture evidence case coverage is incomplete")
    return by_id


def _verify_faults(
    raw: Mapping[str, Any],
    cycles: Sequence[Mapping[str, Any]],
    captures: Mapping[str, Mapping[str, Any]],
) -> None:
    scenarios_value = raw.get("scenarios")
    optional = raw.get("nonrequired_actual_gpu_fault")
    if (
        raw.get("schema_version") != 1
        or raw.get("matrix_id") != "illm.v02.stateful-failure-matrix.v1"
        or raw.get("required_scenario_ids") != list(REQUIRED_FAULT_IDS)
        or raw.get("required_pass_count") != len(REQUIRED_FAULT_IDS)
        or not isinstance(scenarios_value, list)
        or len(scenarios_value) != len(REQUIRED_FAULT_IDS)
        or raw.get("raw_digest") != sha256_json({key: value for key, value in raw.items() if key != "raw_digest"})
        or optional != {
            "status": "not_run_resource_unavailable",
            "required_for_cpu_profile": False,
            "reason": "CPU state-machine profile does not claim an actual GPU fault run",
            "simulation_results_must_not_be_relabelled_as_actual_gpu_faults": True,
        }
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "fault matrix identity/coverage is invalid")
    scenarios = cast(list[dict[str, Any]], scenarios_value)
    for expected_id, scenario in zip(REQUIRED_FAULT_IDS, scenarios, strict=True):
        observations = scenario.get("observations")
        if (
            scenario.get("scenario_id") != expected_id
            or scenario.get("status") != "pass"
            or scenario.get("required_for_cpu_profile") is not True
            or scenario.get("actual_tiny_runtime") is not True
            or scenario.get("actual_gpu_fault") is not False
            or not isinstance(scenario.get("mechanism"), str)
            or not isinstance(scenario.get("evidence_kind"), str)
            or not isinstance(observations, dict)
            or scenario.get("scenario_digest")
            != sha256_json({key: value for key, value in scenario.items() if key != "scenario_digest"})
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "fault scenario identity/status is invalid", {"scenario": expected_id})
        observation = cast(dict[str, Any], observations)
        if expected_id == "cancel-before-launch":
            cycle = cycles[1]
            if observation.get("outcome") != "CANCELLED" or observation.get("model_forward_started_count") != 0 or observation.get("resources_after_terminal") != cycle["resources_after_terminal"]:
                _fail(ErrorCode.CAPTURE_CORRUPT, "cancel-before-launch raw linkage is invalid")
        elif expected_id == "cancel-after-launch-delayed-fence":
            pending = observation.get("resources_while_fence_pending")
            if not isinstance(pending, dict) or pending.get("active_tickets") != 1 or pending.get("kv_write_leases") != 1 or observation.get("outcome") != "CANCELLED" or observation.get("token_chunk_count") != 0 or observation.get("capture_manifest_digest") != captures["cancel-after-launch"]["manifest_digest"]:
                _fail(ErrorCode.CAPTURE_CORRUPT, "cancel-after-launch delayed fence facts are invalid")
        elif expected_id == "finish-cancel-race":
            iterations = observation.get("iterations")
            if observation.get("race_count") != RACE_COUNT or not isinstance(iterations, list) or len(iterations) != RACE_COUNT:
                _fail(ErrorCode.CAPTURE_CORRUPT, "finish/cancel race count is invalid")
            for index, item in enumerate(iterations):
                if not isinstance(item, dict) or item.get("iteration") != index or item.get("outcome") not in {"FINISHED", "CANCELLED"} or item.get("cancel_outcome") != item.get("outcome") or item.get("terminal_event_count") != 1 or not isinstance(item.get("resources_after_terminal"), dict) or not _resources_zero(cast(dict[str, Any], item["resources_after_terminal"])):
                    _fail(ErrorCode.CAPTURE_CORRUPT, "finish/cancel race arbitration facts are invalid")
        elif expected_id == "capacity-exceeded-preaccept":
            if observation.get("error_code") != ErrorCode.CONTEXT_LIMIT_EXCEEDED.value or observation.get("accepted_event_count") != 0 or not isinstance(observation.get("resources_after_rejection"), dict) or not _resources_zero(cast(dict[str, Any], observation["resources_after_rejection"])):
                _fail(ErrorCode.CAPTURE_CORRUPT, "capacity preaccept facts are invalid")
        elif expected_id in {"async-execution-error", "numerical-error", "oom"}:
            expected_code = {
                "async-execution-error": ErrorCode.INTERNAL.value,
                "numerical-error": ErrorCode.NUMERICAL_ERROR.value,
                "oom": ErrorCode.RESOURCE_EXHAUSTED.value,
            }[expected_id]
            if observation.get("outcome") != "FAILED" or observation.get("error_code") != expected_code or observation.get("committed_output_count") != 0 or not isinstance(observation.get("resources_after_terminal"), dict) or not _resources_zero(cast(dict[str, Any], observation["resources_after_terminal"])):
                _fail(ErrorCode.CAPTURE_CORRUPT, "known failure terminal/resource facts are invalid")
        elif expected_id == "unknown-fence-fail-stop":
            owned = observation.get("resources_quarantined")
            if observation.get("raised_error_code") != ErrorCode.ENGINE_UNHEALTHY.value or observation.get("terminal_ready_event_count") != 0 or observation.get("engine_close_succeeded") is not False or not isinstance(owned, dict) or owned.get("quarantined_requests") != 1 or owned.get("active_tickets") != 1 or observation.get("capture_path") != captures["unknown-fence"]["path"]:
                _fail(ErrorCode.CAPTURE_CORRUPT, "unknown-fence fail-stop facts are invalid")
        elif expected_id == "optional-exporter-stall-drop":
            if observation.get("outcome") != "FINISHED" or type(observation.get("copy_drop_count")) is not int or cast(int, observation["copy_drop_count"]) <= 0 or observation.get("initial_close_succeeded") is not False or observation.get("final_close_succeeded") is not True or observation.get("independent_capture_manifest_digest") != captures["exporter-stall"]["manifest_digest"]:
                _fail(ErrorCode.CAPTURE_CORRUPT, "optional exporter isolation facts are invalid")
        elif expected_id == "required-capture-write-failure":
            if observation.get("outcome") != "FINISHED" or observation.get("capture_error") != "CAPTURE_WRITE_FAILED" or type(observation.get("critical_capture_event_drop_count")) is not int or cast(int, observation["critical_capture_event_drop_count"]) <= 0 or observation.get("capture_path") != captures["required-write-failure"]["path"] or observation.get("capture_complete_present") is not False or observation.get("capture_incomplete_present") is not True:
                _fail(ErrorCode.CAPTURE_CORRUPT, "required capture write-failure facts are invalid")
        elif expected_id == "capture-truncation-detected":
            if observation != {"capture_path": captures["truncated-copy"]["path"], "reader_error_code": ErrorCode.CAPTURE_CORRUPT.value}:
                _fail(ErrorCode.CAPTURE_CORRUPT, "capture truncation rejection facts are invalid")
        elif expected_id == "forced-exit-quarantine":
            owned = observation.get("resources_quarantined")
            if observation.get("raised_error_code") != ErrorCode.ENGINE_UNHEALTHY.value or observation.get("terminal_chunk_count") != 0 or observation.get("engine_close_succeeded") is not False or not isinstance(owned, dict) or owned.get("quarantined_requests") != 1 or observation.get("capture_path") != captures["forced-exit"]["path"]:
                _fail(ErrorCode.CAPTURE_CORRUPT, "forced-exit quarantine facts are invalid")


def verify_exercise_output(output: Path, *, require_formal: bool = True) -> dict[str, Any]:
    """Recompute a formal or explicitly requested development evidence bundle."""

    if type(require_formal) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "require_formal must be bool")
    root = output.resolve()
    if not root.is_dir() or (root / "INCOMPLETE").exists():
        _fail(ErrorCode.CAPTURE_INCOMPLETE, "lifecycle exercise output is incomplete")
    formal = (root / "COMPLETE").is_file()
    development = (root / "DEVELOPMENT_ONLY").is_file()
    if formal == development or formal is not require_formal:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle completion/formality markers are inconsistent")
    checksums = _object(root / "checksums.json")
    files = checksums.get("files")
    if (
        checksums.get("schema_version") != 1
        or not isinstance(files, dict)
        or checksums.get("inventory_digest") != sha256_json(files)
        or files != _inventory(root)
        or any(not isinstance(name, str) or not isinstance(digest, str) or sha256_file(root / _relative(name)) != digest for name, digest in files.items())
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle checksum inventory is invalid")
    report = _object(root / "report" / "exercise-report.json")
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != REPORT_KIND
        or report.get("scope") != SCOPE
        or report.get("status") != "pass"
        or report.get("qualification_tier") != "unqualified"
        or report.get("required_matrix_complete") is not True
        or report.get("actual_gpu_faults_run") is not False
        or report.get("actual_gpu_fault_claim") is not None
        or report.get("report_digest") != sha256_json({key: value for key, value in report.items() if key != "report_digest"})
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle report identity/status is invalid")
    _verify_source_and_toolchain(root, report, formal)
    registry = _object(root / "provenance" / "metric-registry-v1.json")
    conformance = _object(root / "provenance" / "metric-registry-conformance.json")
    try:
        validate_metric_registry(registry)
    except ValueError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "embedded MetricRegistryV1 differs from frozen contract") from exc
    expected_conformance = registry_conformance(registry)
    if conformance != expected_conformance or report.get("metric_registry_digest") != registry_digest():
        _fail(ErrorCode.CAPTURE_CORRUPT, "MetricRegistryV1 conformance/digest linkage is invalid")
    cycles = _object(root / "raw" / "cycles.json")
    captures = _object(root / "raw" / "captures.json")
    faults = _object(root / "raw" / "fault-matrix.json")
    cycle_records = _verify_cycles(cycles)
    capture_records = _verify_captures(root, captures)
    _verify_faults(faults, cycle_records, capture_records)
    if report.get("raw_digests") != {
        "cycles": cycles["raw_digest"],
        "captures": captures["raw_digest"],
        "fault_matrix": faults["raw_digest"],
    }:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle report raw evidence linkage is invalid")
    if formal:
        if report.get("development_no_manifest") is not False or report.get("formal_evidence_eligible") is not True:
            _fail(ErrorCode.CAPTURE_CORRUPT, "formal lifecycle report lost eligibility boundary")
        try:
            complete = (root / "COMPLETE").read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "lifecycle COMPLETE marker is unreadable") from exc
        if complete != report["report_digest"]:
            _fail(ErrorCode.CAPTURE_CORRUPT, "COMPLETE does not bind lifecycle report")
    else:
        marker = _object(root / "DEVELOPMENT_ONLY")
        if (
            report.get("development_no_manifest") is not True
            or report.get("formal_evidence_eligible") is not False
            or marker != {
                "qualification_tier": "unqualified",
                "formal_evidence_eligible": False,
                "report_digest": report["report_digest"],
            }
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "development lifecycle marker/report boundary is invalid")
    return report

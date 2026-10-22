"""Actual candidate-policy CPU lifecycle and tamper regressions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.numerical_policy import candidate_policy
from illm.stateful_policy_exercise import (
    COMPLETE_CAPTURE_IDS,
    CYCLE_COUNT,
    INCOMPLETE_CAPTURE_IDS,
    RACE_COUNT,
    REQUIRED_FAULT_IDS,
    REQUIRED_POLICY_CAPTURE_IDS,
    verify_policy_exercise_output,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import exercise_policy_streaming as exercise


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _outer_inventory(root: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }


def _reseal_report(root: Path, report: dict[str, Any]) -> None:
    report.pop("report_digest", None)
    report["report_digest"] = sha256_json(report)
    _write(root / "report" / "exercise-report.json", report)
    _write(
        root / "DEVELOPMENT_ONLY",
        {
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
            "qualification_scope": "cpu-only",
            "formal_evidence_eligible": False,
            "cpu_profile_evidence_eligible": False,
            "global_qualification_granted": False,
            "gpu_qualification_granted": False,
            "report_digest": report["report_digest"],
        },
    )
    files = _outer_inventory(root)
    _write(
        root / "checksums.json",
        {"schema_version": 1, "files": files, "inventory_digest": sha256_json(files)},
    )


def _reseal_raw(root: Path, name: str, raw: dict[str, Any]) -> None:
    raw.pop("raw_digest", None)
    raw["raw_digest"] = sha256_json(raw)
    _write(root / "raw" / name, raw)
    report = _read(root / "report" / "exercise-report.json")
    key = {"cycles.json": "cycles", "captures.json": "captures", "fault-matrix.json": "fault_matrix"}[name]
    report["raw_digests"][key] = raw["raw_digest"]
    _reseal_report(root, report)


def _reseal_incomplete_capture(root: Path, capture_id: str) -> None:
    capture = root / "captures" / capture_id
    events_digest = sha256_file(capture / "events.jsonl")
    checksums = {"events.jsonl": events_digest}
    _write(capture / "checksums.json", checksums)
    manifest = _read(capture / "capture-manifest.json")
    manifest["files"] = {
        "events.jsonl": events_digest,
        "checksums.json": sha256_file(capture / "checksums.json"),
    }
    manifest.pop("manifest_digest", None)
    manifest["manifest_digest"] = sha256_json(manifest)
    _write(capture / "capture-manifest.json", manifest)
    _reseal_report(root, _read(root / "report" / "exercise-report.json"))


_BOUND_ENV = ("ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST")


def _run_development_output(output: Path) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as clean:
        for name in _BOUND_ENV:
            clean.delenv(name, raising=False)
        return exercise.run_policy_exercise(output=output, development_no_manifest=True)


@pytest.fixture(scope="module")
def development_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    output = tmp_path_factory.mktemp("stateful-policy-exercise") / "evidence"
    report = _run_development_output(output)
    return output, report


def test_development_fixture_temporarily_clears_parent_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index, name in enumerate(_BOUND_ENV):
        monkeypatch.setenv(name, f"parent-bound-{index}")
    observed: dict[str, str | None] = {}

    def fake_run(*, output: Path, development_no_manifest: bool) -> dict[str, Any]:
        assert output == tmp_path / "unused"
        assert development_no_manifest is True
        observed.update({name: os.environ.get(name) for name in _BOUND_ENV})
        return {"status": "pass"}

    monkeypatch.setattr(exercise, "run_policy_exercise", fake_run)
    assert _run_development_output(tmp_path / "unused") == {"status": "pass"}
    assert observed == {name: None for name in _BOUND_ENV}
    assert all(os.environ[name].startswith("parent-bound-") for name in _BOUND_ENV)


def test_actual_candidate_policy_lifecycle_has_complete_closed_coverage(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = development_output
    policy = candidate_policy("fp32")
    assert report["status"] == "pass"
    assert report["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert report["qualification_status"] == "unqualified"
    assert report["qualification_scope"] == "cpu-only"
    assert report["formal_evidence_eligible"] is False
    assert report["cpu_profile_evidence_eligible"] is False
    assert report["global_qualification_granted"] is False
    assert report["gpu_qualification_granted"] is False
    assert report["numerical_policy"] == policy.to_dict()
    assert report["numerical_policy_digest"] == policy.digest
    assert report["protocol_sha256"] == report["execution_binding"]["protocol_sha256"]
    assert report["actual_cycle_count"] == CYCLE_COUNT
    assert report["finished_cycle_count"] == report["cancelled_cycle_count"] == 50
    assert report["race_count"] == RACE_COUNT
    assert report["required_fault_scenario_ids"] == list(REQUIRED_FAULT_IDS)
    assert report["required_fault_pass_count"] == len(REQUIRED_FAULT_IDS)
    assert report["required_policy_capture_ids"] == list(REQUIRED_POLICY_CAPTURE_IDS)
    assert report["actual_gpu_faults_run"] is False
    assert report["actual_gpu_fault_claim"] is None
    assert report["performance_claim"] is None
    assert (output / "DEVELOPMENT_ONLY").is_file()
    assert not (output / "COMPLETE").exists() and not (output / "INCOMPLETE").exists()
    assert verify_policy_exercise_output(output, require_formal=False) == report

    plans = report["memory_plan_digests"]
    assert set(plans["normal"]) == {"none", "contiguous"}
    assert set(plans["fault_subplans"]) == {"capacity-exceeded-preaccept"}
    assert len({*plans["normal"].values(), *plans["fault_subplans"].values()}) == 3


def test_every_actual_cycle_is_schema3_candidate_and_returns_all_resources(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = development_output
    cycles = _read(output / "raw" / "cycles.json")
    assert cycles["cycle_count"] == CYCLE_COUNT and len(cycles["records"]) == CYCLE_COUNT
    for index, record in enumerate(cycles["records"]):
        result = record["result"]
        expected = "FINISHED" if index % 2 == 0 else "CANCELLED"
        assert result["schema_version"] == 3
        assert result["outcome"] == record["planned_outcome"] == expected
        assert result["numerical_policy_digest"] == report["numerical_policy_digest"]
        assert result["execution_binding_digest"] == report["execution_binding_digest"]
        assert result["memory_plan_digest"] == report["memory_plan_digests"]["normal"]["contiguous"]
        assert set(record["resources_after_terminal"].values()) == {0}
        terminal = [
            event
            for event in record["events"]
            if event["kind"] in {"request.completed", "request.cancelled", "request.failed"}
        ]
        assert len(terminal) == 1
        assert all(event["event_catalog"] == "illm-stateful-v02-policy-v1" for event in record["events"])


def test_schema3_capture_index_exposes_fixed_paths_and_real_replay(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = development_output
    captures = _read(output / "raw" / "captures.json")
    records = {record["capture_id"]: record for record in captures["records"]}
    assert set(COMPLETE_CAPTURE_IDS) <= set(records)
    assert set(INCOMPLETE_CAPTURE_IDS) <= set(records)
    for capture_id in COMPLETE_CAPTURE_IDS:
        record = records[capture_id]
        assert record["path"] == f"captures/{capture_id}"
        bundle = PolicyStreamingCaptureBundle.open(output / record["path"])
        assert bundle.manifest["schema_version"] == 3
        assert bundle.manifest["manifest_digest"] == record["manifest_digest"]
        assert bundle.manifest["numerical_policy_digest"] == report["numerical_policy_digest"]
        assert bundle.manifest["execution_binding_digest"] == report["execution_binding_digest"]
        assert bundle.manifest["event_catalog"] == "illm-stateful-v02-policy-v1"
        assert bundle.manifest["capabilities"] == ["R0"]
        assert bundle.manifest["privacy_mode"] == "metadata_only"
    assert records["no-kv"]["memory_plan_digest"] == report["memory_plan_digests"]["normal"]["none"]
    for capture_id in INCOMPLETE_CAPTURE_IDS:
        capture = output / records[capture_id]["path"]
        assert (capture / "INCOMPLETE").is_file() and not (capture / "COMPLETE").exists()
        with pytest.raises(IllmError) as raised:
            PolicyStreamingCaptureBundle.open(capture)
        assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE


def test_fault_table_preserves_real_quarantine_and_cpu_simulation_boundary(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = development_output
    faults = _read(output / "raw" / "fault-matrix.json")
    scenarios = {item["scenario_id"]: item for item in faults["scenarios"]}
    assert list(scenarios) == list(REQUIRED_FAULT_IDS)
    assert all(item["status"] == "pass" for item in scenarios.values())
    assert all(item["actual_tiny_runtime"] is True for item in scenarios.values())
    assert all(item["actual_gpu_fault"] is False for item in scenarios.values())
    unknown = scenarios["unknown-fence-fail-stop"]["observations"]["resources_quarantined"]
    forced = scenarios["forced-exit-quarantine"]["observations"]["resources_quarantined"]
    assert unknown["quarantined_requests"] == unknown["active_tickets"] == 1
    assert forced["quarantined_requests"] == forced["active_tickets"] == 1
    assert scenarios["capacity-exceeded-preaccept"]["observations"]["memory_plan_digest"] == (
        report["memory_plan_digests"]["fault_subplans"]["capacity-exceeded-preaccept"]
    )
    assert faults["nonrequired_actual_gpu_fault"]["status"] == "not_run_resource_unavailable"


def test_resigned_unknown_policy_and_cross_binding_are_rejected(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, _ = development_output
    unknown = tmp_path / "unknown-policy"
    shutil.copytree(original, unknown)
    report = _read(unknown / "report" / "exercise-report.json")
    report["numerical_policy"]["family"] = "unregistered-policy"
    report["numerical_policy_digest"] = sha256_json(report["numerical_policy"])
    _reseal_report(unknown, report)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(unknown, require_formal=False)

    crossed = tmp_path / "crossed-binding"
    shutil.copytree(original, crossed)
    report = _read(crossed / "report" / "exercise-report.json")
    binding = report["execution_binding"]
    binding["environment"]["machine"] = "forged-machine"
    binding["environment_digest"] = sha256_json(binding["environment"])
    report["execution_binding_digest"] = sha256_json(binding)
    _write(crossed / "provenance" / "execution-binding.json", binding)
    _reseal_report(crossed, report)
    with pytest.raises(IllmError, match="binding|cycle|capture"):
        verify_policy_exercise_output(crossed, require_formal=False)


@pytest.mark.parametrize("tamper", ["input", "event-private-field"])
def test_resigned_raw_cycles_cannot_escape_public_fixture_or_closed_event_payload(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    tamper: str,
) -> None:
    original, _ = development_output
    changed = tmp_path / f"raw-{tamper}"
    shutil.copytree(original, changed)
    cycles = _read(changed / "raw" / "cycles.json")
    record = cycles["records"][0]
    if tamper == "input":
        record["result"]["input_token_ids"] = [1, 2, 3]
        record["result"]["text"] = "private-text"
    else:
        record["events"][0]["payload"]["raw_prompt"] = "private-text"
    record.pop("cycle_digest")
    record["cycle_digest"] = sha256_json(record)
    _reseal_raw(changed, "cycles.json", cycles)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(changed, require_formal=False)


def test_resigned_unknown_capture_field_is_rejected_even_after_full_relink(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, _ = development_output
    tampered = tmp_path / "unknown-capture-field"
    shutil.copytree(original, tampered)
    capture = tampered / "captures" / "success"
    manifest = _read(capture / "capture-manifest.json")
    manifest["future_private_extension"] = {"accepted": True}
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = sha256_json(manifest)
    _write(capture / "capture-manifest.json", manifest)
    (capture / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
    raw = _read(tampered / "raw" / "captures.json")
    record = next(item for item in raw["records"] if item["capture_id"] == "success")
    record["manifest_digest"] = manifest["manifest_digest"]
    record.pop("capture_record_digest")
    record["capture_record_digest"] = sha256_json(record)
    _reseal_raw(tampered, "captures.json", raw)
    with pytest.raises(IllmError, match="fields|manifest"):
        verify_policy_exercise_output(tampered, require_formal=False)


def test_resigned_required_fault_not_run_and_unknown_report_field_fail_closed(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, _ = development_output
    table = tmp_path / "fault-table-not-run"
    shutil.copytree(original, table)
    faults = _read(table / "raw" / "fault-matrix.json")
    scenario = faults["scenarios"][0]
    scenario["status"] = "not_run"
    scenario.pop("scenario_digest")
    scenario["scenario_digest"] = sha256_json(scenario)
    _reseal_raw(table, "fault-matrix.json", faults)
    with pytest.raises(IllmError, match="scenario"):
        verify_policy_exercise_output(table, require_formal=False)

    unknown = tmp_path / "unknown-report-field"
    shutil.copytree(original, unknown)
    report = _read(unknown / "report" / "exercise-report.json")
    report["future_qualification_override"] = True
    _reseal_report(unknown, report)
    with pytest.raises(IllmError, match="fields"):
        verify_policy_exercise_output(unknown, require_formal=False)


@pytest.mark.parametrize("target", ["report", "checksums", "source"])
def test_resigned_boolean_schema_version_is_never_accepted_as_integer_one(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    target: str,
) -> None:
    original, _ = development_output
    tampered = tmp_path / f"bool-schema-{target}"
    shutil.copytree(original, tampered)
    if target == "report":
        report = _read(tampered / "report" / "exercise-report.json")
        report["schema_version"] = True
        _reseal_report(tampered, report)
    elif target == "checksums":
        checksums = _read(tampered / "checksums.json")
        checksums["schema_version"] = True
        _write(tampered / "checksums.json", checksums)
    else:
        source_path = tampered / "provenance" / "source-identity.json"
        source = _read(source_path)
        source["schema_version"] = True
        source.pop("source_identity_digest")
        source["source_identity_digest"] = sha256_json(source)
        _write(source_path, source)
        report = _read(tampered / "report" / "exercise-report.json")
        report["source_identity"] = source
        _reseal_report(tampered, report)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(tampered, require_formal=False)


def test_fully_resigned_extra_file_and_empty_directory_are_rejected(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, _ = development_output
    extra = tmp_path / "extra-file"
    shutil.copytree(original, extra)
    foreign = extra / "future" / "qualification-override.json"
    foreign.parent.mkdir()
    foreign.write_text('{"grant":true}\n', encoding="utf-8")
    files = _outer_inventory(extra)
    _write(
        extra / "checksums.json",
        {"schema_version": 1, "files": files, "inventory_digest": sha256_json(files)},
    )
    with pytest.raises(IllmError, match="inventory"):
        verify_policy_exercise_output(extra, require_formal=False)

    empty = tmp_path / "empty-directory"
    shutil.copytree(original, empty)
    (empty / "future-empty").mkdir()
    with pytest.raises(IllmError, match="inventory"):
        verify_policy_exercise_output(empty, require_formal=False)


@pytest.mark.parametrize(
    "tamper",
    ["private-event", "sequence-gap", "negative-event-count", "negative-drop-count", "marker-reason"],
)
def test_resigned_incomplete_capture_claims_are_replayed_and_fail_closed(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    tamper: str,
) -> None:
    original, _ = development_output
    changed = tmp_path / f"incomplete-{tamper}"
    shutil.copytree(original, changed)
    capture = changed / "captures" / "required-write-failure"
    manifest = _read(capture / "capture-manifest.json")
    if tamper == "private-event":
        (capture / "events.jsonl").write_bytes(canonical_json_bytes({"prompt": "private"}) + b"\n")
        manifest["event_count"] = 1
        _write(capture / "capture-manifest.json", manifest)
        _reseal_incomplete_capture(changed, "required-write-failure")
    elif tamper == "sequence-gap":
        events = [json.loads(line) for line in (capture / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        events[0]["producer_seq_no"] = 1
        (capture / "events.jsonl").write_bytes(
            b"".join(canonical_json_bytes(event) + b"\n" for event in events)
        )
        _reseal_incomplete_capture(changed, "required-write-failure")
    elif tamper == "negative-event-count":
        manifest["event_count"] = -1
        _write(capture / "capture-manifest.json", manifest)
        _reseal_incomplete_capture(changed, "required-write-failure")
    elif tamper == "negative-drop-count":
        manifest["critical_event_drop_count"] = -1
        _write(capture / "capture-manifest.json", manifest)
        _reseal_incomplete_capture(changed, "required-write-failure")
    else:
        (capture / "INCOMPLETE").write_text("FORGED_REASON\n", encoding="utf-8")
        _reseal_report(changed, _read(changed / "report" / "exercise-report.json"))
    with pytest.raises(IllmError):
        verify_policy_exercise_output(changed, require_formal=False)


@pytest.mark.parametrize(
    "tamper",
    ["terminal-kind", "model-spec", "sampling-spec", "timing"],
)
def test_resigned_cycle_must_match_full_model_sampling_timing_and_state_authority(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    tamper: str,
) -> None:
    original, _ = development_output
    changed = tmp_path / f"cycle-authority-{tamper}"
    shutil.copytree(original, changed)
    cycles = _read(changed / "raw" / "cycles.json")
    record = cycles["records"][0 if tamper == "terminal-kind" else 2]
    if tamper == "terminal-kind":
        template = next(
            event
            for event in cycles["records"][1]["events"]
            if event["kind"] == "request.cancelled"
        )
        replacement = json.loads(json.dumps(template))
        position = next(
            index
            for index, event in enumerate(record["events"])
            if event["kind"] == "request.completed"
        )
        original_terminal = record["events"][position]
        replacement["producer_seq_no"] = original_terminal["producer_seq_no"]
        replacement["monotonic_ns"] = original_terminal["monotonic_ns"]
        replacement["payload"]["request_id"] = record["request_id"]
        record["events"][position] = replacement
    elif tamper == "model-spec":
        record["result"]["model_spec_id"] = "f" * 64
    elif tamper == "sampling-spec":
        record["result"]["sampling_spec_digest"] = "e" * 64
    else:
        record["result"]["timings"]["accepted_ns"] += 1
    record.pop("cycle_digest")
    record["cycle_digest"] = sha256_json(record)
    _reseal_raw(changed, "cycles.json", cycles)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(changed, require_formal=False)


@pytest.mark.parametrize(
    "tamper",
    ["unknown-owned-zero", "forced-ticket-zero", "race-winner", "race-negative-chunks"],
)
def test_resigned_fault_resources_and_race_winner_claims_are_derived(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    tamper: str,
) -> None:
    original, _ = development_output
    changed = tmp_path / f"fault-derived-{tamper}"
    shutil.copytree(original, changed)
    faults = _read(changed / "raw" / "fault-matrix.json")
    by_id = {scenario["scenario_id"]: scenario for scenario in faults["scenarios"]}
    if tamper == "unknown-owned-zero":
        scenario = by_id["unknown-fence-fail-stop"]
        resources = scenario["observations"]["resources_quarantined"]
        for name in (
            "reservations",
            "input_leases",
            "output_leases",
            "workspace_leases",
            "kv_write_leases",
            "kv_reserved_bytes",
            "kv_staged_tokens",
        ):
            resources[name] = 0
    elif tamper == "forced-ticket-zero":
        scenario = by_id["forced-exit-quarantine"]
        scenario["observations"]["resources_quarantined"]["active_tickets"] = 0
    else:
        scenario = by_id["finish-cancel-race"]
        iteration = scenario["observations"]["iterations"][0]
        if tamper == "race-winner":
            iteration["cancel_won"] = not iteration["cancel_won"]
        else:
            iteration["token_chunk_count"] = -1
    scenario.pop("scenario_digest")
    scenario["scenario_digest"] = sha256_json(scenario)
    _reseal_raw(changed, "fault-matrix.json", faults)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(changed, require_formal=False)


@pytest.mark.parametrize("capture_id", ["success", "no-kv", "exporter-stall", "cancel-after-launch"])
def test_resigned_complete_capture_finish_reason_comes_from_terminal_event(
    development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    capture_id: str,
) -> None:
    original, _ = development_output
    changed = tmp_path / f"capture-finish-{capture_id}"
    shutil.copytree(original, changed)
    captures = _read(changed / "raw" / "captures.json")
    record = next(item for item in captures["records"] if item["capture_id"] == capture_id)
    record["finish_reason"] = None if record["finish_reason"] is not None else "MAX_NEW_TOKENS"
    record.pop("capture_record_digest")
    record["capture_record_digest"] = sha256_json(record)
    _reseal_raw(changed, "captures.json", captures)
    with pytest.raises(IllmError):
        verify_policy_exercise_output(changed, require_formal=False)

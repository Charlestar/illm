"""V0.2 lifecycle evidence producer and portable reader tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.stateful_exercise import (
    COMPLETE_CAPTURE_IDS,
    CYCLE_COUNT,
    INCOMPLETE_CAPTURE_IDS,
    REQUIRED_FAULT_IDS,
    verify_exercise_output,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import exercise_streaming as exercise


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _resign_development_bundle(root: Path, raw_name: str, raw: dict[str, Any]) -> None:
    raw_path = root / "raw" / raw_name
    raw.pop("raw_digest", None)
    raw["raw_digest"] = sha256_json(raw)
    _write(raw_path, raw)
    report_path = root / "report" / "exercise-report.json"
    report = _read(report_path)
    key = {"cycles.json": "cycles", "fault-matrix.json": "fault_matrix"}[raw_name]
    report["raw_digests"][key] = raw["raw_digest"]
    report.pop("report_digest", None)
    report["report_digest"] = sha256_json(report)
    _write(report_path, report)
    _write(root / "DEVELOPMENT_ONLY", {
        "qualification_tier": "unqualified",
        "formal_evidence_eligible": False,
        "report_digest": report["report_digest"],
    })
    exercise._write_checksums(root)


@pytest.fixture(scope="module")
def development_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    output = tmp_path_factory.mktemp("streaming-exercise") / "evidence"
    report = exercise.run_exercise(output=output, development_no_manifest=True)
    return output, report


def test_real_tiny_lifecycle_runs_hundred_cycles_and_complete_fault_matrix(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = development_output
    assert report["status"] == "pass"
    assert report["actual_cycle_count"] == CYCLE_COUNT
    assert report["finished_cycle_count"] == report["cancelled_cycle_count"] == 50
    assert report["required_fault_scenario_ids"] == list(REQUIRED_FAULT_IDS)
    assert report["required_fault_pass_count"] == len(REQUIRED_FAULT_IDS)
    assert report["required_matrix_complete"] is True
    assert report["actual_gpu_faults_run"] is False
    assert report["actual_gpu_fault_claim"] is None
    assert report["formal_evidence_eligible"] is False
    assert report["qualification_tier"] == "unqualified"
    assert (output / "DEVELOPMENT_ONLY").is_file()
    assert not (output / "COMPLETE").exists() and not (output / "INCOMPLETE").exists()
    assert verify_exercise_output(output, require_formal=False) == report

    cycles = _read(output / "raw" / "cycles.json")
    assert cycles["cycle_count"] == 100 and len(cycles["records"]) == 100
    for index, record in enumerate(cycles["records"]):
        expected = "FINISHED" if index % 2 == 0 else "CANCELLED"
        assert record["planned_outcome"] == record["result"]["outcome"] == expected
        assert set(record["resources_after_terminal"].values()) == {0}
        terminal = [
            event for event in record["events"]
            if event["kind"] in {"request.completed", "request.cancelled", "request.failed"}
        ]
        assert len(terminal) == 1

    faults = _read(output / "raw" / "fault-matrix.json")
    assert [item["scenario_id"] for item in faults["scenarios"]] == list(REQUIRED_FAULT_IDS)
    assert all(item["status"] == "pass" for item in faults["scenarios"])
    assert all(item["actual_tiny_runtime"] is True for item in faults["scenarios"])
    assert all(item["actual_gpu_fault"] is False for item in faults["scenarios"])
    assert faults["nonrequired_actual_gpu_fault"]["status"] == "not_run_resource_unavailable"


def test_capture_set_is_replayable_or_explicitly_incomplete(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, _ = development_output
    index = _read(output / "raw" / "captures.json")
    records = {record["capture_id"]: record for record in index["records"]}
    assert set(COMPLETE_CAPTURE_IDS) <= set(records)
    assert set(INCOMPLETE_CAPTURE_IDS) <= set(records)
    for capture_id in COMPLETE_CAPTURE_IDS:
        bundle = StreamingCaptureBundle.open(output / records[capture_id]["path"])
        assert bundle.manifest["manifest_digest"] == records[capture_id]["manifest_digest"]
        assert bundle.manifest["capabilities"] == ["R0"]
        assert bundle.manifest["privacy_mode"] == "metadata_only"
    assert records["length"]["finish_reason"] == "MAX_NEW_TOKENS"
    assert records["eos"]["finish_reason"] == "EOS"
    assert records["cancel-after-launch"]["outcome"] == "CANCELLED"
    assert records["no-kv"]["kv_mode"] == "none"
    for capture_id in INCOMPLETE_CAPTURE_IDS:
        path = output / records[capture_id]["path"]
        assert (path / "INCOMPLETE").is_file() and not (path / "COMPLETE").exists()
        with pytest.raises(IllmError) as raised:
            StreamingCaptureBundle.open(path)
        assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
    with pytest.raises(IllmError) as truncated:
        StreamingCaptureBundle.open(output / records["truncated-copy"]["path"])
    assert truncated.value.code is ErrorCode.CAPTURE_CORRUPT


def test_portable_reader_imports_no_torch_and_uses_no_scripts(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, _ = development_output
    code = (
        f"import sys;sys.path.insert(0,{str(ROOT / 'src')!r});from pathlib import Path;"
        "from illm.stateful_exercise import verify_exercise_output;"
        f"verify_exercise_output(Path({str(output)!r}),require_formal=False);"
        "assert 'torch' not in sys.modules;print('portable-pass')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "portable-pass"


def test_formal_mode_requires_current_finalized_manifest_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    monkeypatch.delenv("ILLM_LOCK_DIGEST", raising=False)
    output = tmp_path / "not-created"
    with pytest.raises(IllmError) as raised:
        exercise.run_exercise(output=output)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH
    assert not output.exists()


def test_development_output_cannot_be_read_as_formal(
    development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, _ = development_output
    with pytest.raises(IllmError) as raised:
        verify_exercise_output(output)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_resigned_cycle_resource_tamper_is_rejected(
    development_output: tuple[Path, dict[str, Any]], tmp_path: Path,
) -> None:
    original, _ = development_output
    tampered = tmp_path / "resigned-cycle-tamper"
    shutil.copytree(original, tampered)
    cycles = _read(tampered / "raw" / "cycles.json")
    record = cycles["records"][0]
    record["resources_after_terminal"]["active_requests"] = 1
    record["result"]["resource_status"]["active_requests"] = 1
    record.pop("cycle_digest")
    record["cycle_digest"] = sha256_json(record)
    _resign_development_bundle(tampered, "cycles.json", cycles)
    with pytest.raises(IllmError) as raised:
        verify_exercise_output(tampered, require_formal=False)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_resigned_required_not_run_and_registry_tamper_are_rejected(
    development_output: tuple[Path, dict[str, Any]], tmp_path: Path,
) -> None:
    original, _ = development_output
    not_run = tmp_path / "resigned-not-run"
    shutil.copytree(original, not_run)
    faults = _read(not_run / "raw" / "fault-matrix.json")
    scenario = faults["scenarios"][0]
    scenario["status"] = "not_run"
    scenario.pop("scenario_digest")
    scenario["scenario_digest"] = sha256_json(scenario)
    _resign_development_bundle(not_run, "fault-matrix.json", faults)
    with pytest.raises(IllmError, match="fault scenario"):
        verify_exercise_output(not_run, require_formal=False)

    registry_tamper = tmp_path / "resigned-registry"
    shutil.copytree(original, registry_tamper)
    registry_path = registry_tamper / "provenance" / "metric-registry-v1.json"
    registry = _read(registry_path)
    registry["metrics"][0]["unit"] = "ms"
    _write(registry_path, registry)
    exercise._write_checksums(registry_tamper)
    with pytest.raises(IllmError, match="MetricRegistryV1"):
        verify_exercise_output(registry_tamper, require_formal=False)


def test_production_script_does_not_import_test_frameworks() -> None:
    source = (ROOT / "scripts" / "exercise_streaming.py").read_text(encoding="utf-8")
    assert "import pytest" not in source
    assert "from tests" not in source

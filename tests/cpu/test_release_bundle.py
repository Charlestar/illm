from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.cli import main
from illm.domain.errors import IllmError
from illm.qualification import create_profile_evidence, parse_junit, verify_profile_evidence
from illm.release import verify_release_bundle, verify_release_manifest, write_verified_release_state
from illm.toy import run_toy_scheduler


def _extended_path(path: Path) -> Path:
    """Allow deeply nested sealed artifacts under an arbitrary pytest basetemp."""
    absolute = str(path.resolve())
    return Path("\\\\?\\" + absolute) if os.name == "nt" and not absolute.startswith("\\\\?\\") else path


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _resign(manifest_path: Path, *, seal: bool = False) -> None:
    manifest = _read(manifest_path)
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    _write(manifest_path, manifest)
    write_verified_release_state(verify_release_manifest(manifest_path), manifest_path.parent / "qualification" / "release-state.json")
    if seal:
        _write(manifest_path.parent / "SHA256SUMS.json", {
            item.relative_to(manifest_path.parent).as_posix(): sha256_file(item)
            for item in manifest_path.parent.rglob("*") if item.is_file() and item.name != "SHA256SUMS.json"
        })


@pytest.fixture
def release_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp_path = _extended_path(tmp_path)
    root = tmp_path / "release"
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.01"><testcase name="pass"/></testsuite>\n', encoding="utf-8")
    quality = tmp_path / "quality.json"
    _write(quality, {"schema_version": 0, "status": "pass", "issues": [], "checked_python_files": 20,
                     "type_check": {"tool": "mypy", "version": "1.18.2", "status": "pass", "returncode": 0}})
    profiles: list[dict[str, Any]] = []
    source = {"project_root_digest": "a" * 64, "bootstrap_digest": "b" * 64, "lock_digest": "c" * 64}
    base_toolchain: dict[str, Any] = {}
    capture = tmp_path / "capture"
    for system in ("Windows", "Linux"):
        monkeypatch.setattr(platform, "system", lambda system=system: system)
        run_toy_scheduler(seed=7, steps=4, capture_root=capture, project_root=Path.cwd())
        toolchain = {
            "schema_version": 0, "finalized": True, "profile": system.lower(), "os": system,
            "arch": platform.machine(), "python_version": platform.python_version(),
            "python_executable": sys.executable, "environment_id": "fixture-environment",
            "installed_packages": {"mypy": "1.18.2"}, **source,
        }
        toolchain["manifest_digest"] = sha256_json(toolchain)
        toolchain_path = tmp_path / "toolchain.json"
        _write(toolchain_path, toolchain)
        monkeypatch.setenv("ILLM_ENVIRONMENT_ID", str(toolchain["environment_id"]))
        monkeypatch.setenv("ILLM_LOCK_DIGEST", source["lock_digest"])
        evidence_root = tmp_path / f"{system}-evidence"
        profile = create_profile_evidence(toolchain_manifest=toolchain_path, junit=junit, quality=quality, capture=capture, output=evidence_root)
        destination = root / "qualification" / "profiles" / profile["profile_id"]
        shutil.copytree(evidence_root, destination / "evidence" / profile["evidence_digest"])
        _write(destination / "profile-result.json", profile)
        profiles.append(profile)
        if system == "Windows":
            base_toolchain = toolchain
            shutil.copytree(capture, root / "capture" / "toy-online-replay")
    test_report = {"schema_version": 0, **parse_junit(junit), "junit_sha256": sha256_file(junit)}
    _write(root / "test-report.json", test_report)
    cpu = {"profile_id": "cpu-clean-room-v1", "result": "pass", "reason": None, "required_for_functional": True,
           "environment_manifest_digest": base_toolchain["manifest_digest"], "evidence_digest": sha256_json(test_report)}
    profiles.append(cpu)
    _write(root / "qualification" / "profiles" / "cpu-clean-room-v1" / "profile-result.json", cpu)
    stall = {"schema_version": 0, "completed_scheduler_steps": 10_000, "offered_events": 10_001,
             "required_delivered_events": 10_001, "critical_event_drop_count": 0,
             "exporter_was_blocked": True, "producer_completed_while_exporter_blocked": True,
             "forward_progress": True, "exporter_queue_capacity": 3, "exporter_queue_high_watermark": 3,
             "exporter_in_flight_limit": 1, "exporter_drop_count": 9997,
             "traced_current_growth_bytes": 100, "traced_peak_bytes": 500,
             "memory_growth_limit_bytes": 1048576, "memory_bound_pass": True,
             "close_while_blocked_returned": False, "shutdown_after_release": True, "duration_s": 0.1}
    _write(root / "benchmark" / "event-stall" / "report.json", stall)
    for name in ("commands.txt", "teach-back.md", "known-limitations.md", "release-notes.md", "screenshots/terminal/demo.txt"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("fixture learning evidence\n", encoding="utf-8")
    rollback = root / "rollback"
    _write(rollback / "toolchain.json", base_toolchain)
    _write(rollback / "manifest.json", {"schema_version": 0, "source_release": "v0.0.0", "target_state": "pre-v0.0-empty-runtime",
          "toolchain_manifest_digest": base_toolchain["manifest_digest"], "environment_id": base_toolchain["environment_id"],
          "python_executable": base_toolchain["python_executable"], "installed_before": True, "installed_after": False,
          "success": True, "pip_returncode": 0, "background_processes_started": 0, "ports_opened": 0})
    for name in ("commands.txt", "uninstall.stdout.txt", "uninstall.stderr.txt", "rollback-report.md"):
        (rollback / name).write_text("rollback fixture\n", encoding="utf-8")
    _write(rollback / "checksums.json", {item.name: sha256_file(item) for item in rollback.iterdir() if item.is_file()})
    manifest = {"schema_version": 0, "spec_status": "implemented", "release_stage": "release_candidate",
                "qualification_tier": "functionally_qualified", "qualification_profiles": profiles,
                "performance_outcome": "not_applicable", "lts_state": "not_applicable",
                "toolchain_manifest_digest": base_toolchain["manifest_digest"],
                "capture_manifest_digest": _read(root / "capture" / "toy-online-replay" / "capture-manifest.json")["manifest_digest"],
                "test_report_digest": sha256_json(test_report), "event_stall_digest": sha256_json(stall)}
    manifest_path = root / "manifest.json"
    _write(manifest_path, manifest)
    _resign(manifest_path)
    return manifest_path


def test_full_bundle_verifies_after_relocation(release_bundle: Path, tmp_path: Path) -> None:
    destination = _extended_path(tmp_path) / "relocated"
    shutil.copytree(release_bundle.parent, destination)
    assert verify_release_bundle(destination / "manifest.json")["recomputed_qualification_tier"] == "functionally_qualified"
    assert main(["release", "verify", str(destination / "manifest.json")]) == 0


def test_manifest_alone_is_not_a_qualified_bundle(release_bundle: Path, tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    shutil.copy2(release_bundle, path)
    assert verify_release_manifest(path)["qualification_tier"] == "functionally_qualified"
    with pytest.raises(IllmError):
        verify_release_bundle(path)
    assert main(["release", "verify", str(path)]) == 64


@pytest.mark.parametrize("relative", [
    "capture/toy-online-replay/events.jsonl", "test-report.json", "rollback/uninstall.stdout.txt",
    "benchmark/event-stall/report.json", "qualification/profiles/linux-bootstrap-v1/profile-result.json",
])
def test_bundle_rejects_missing_or_modified_fact(release_bundle: Path, relative: str) -> None:
    (release_bundle.parent / relative).unlink()
    with pytest.raises(IllmError):
        verify_release_bundle(release_bundle)


def test_resigned_test_report_cannot_forge_testcase_facts(release_bundle: Path) -> None:
    report_path = release_bundle.parent / "test-report.json"
    report = _read(report_path)
    report["tests"] = 999
    _write(report_path, report)
    manifest = _read(release_bundle)
    manifest["test_report_digest"] = sha256_json(report)
    cpu = next(profile for profile in manifest["qualification_profiles"] if profile["profile_id"] == "cpu-clean-room-v1")
    cpu["evidence_digest"] = manifest["test_report_digest"]
    _write(release_bundle.parent / "qualification" / "profiles" / "cpu-clean-room-v1" / "profile-result.json", cpu)
    _write(release_bundle, manifest)
    _resign(release_bundle)
    with pytest.raises(IllmError, match="JUnit facts"):
        verify_release_bundle(release_bundle)


def test_released_bundle_requires_exact_checksum_coverage(release_bundle: Path) -> None:
    manifest = _read(release_bundle)
    manifest["spec_status"] = "released"
    manifest["release_stage"] = "released"
    _write(release_bundle, manifest)
    _resign(release_bundle)
    with pytest.raises(IllmError, match="SHA256SUMS"):
        verify_release_bundle(release_bundle)
    _resign(release_bundle, seal=True)
    assert verify_release_bundle(release_bundle)["release_stage"] == "released"
    (release_bundle.parent / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(IllmError, match="coverage"):
        verify_release_bundle(release_bundle)


def test_bundle_can_require_current_source_identity(release_bundle: Path) -> None:
    with pytest.raises(IllmError, match="different source"):
        verify_release_bundle(release_bundle, expected_source_identity={"project_root_digest": "f" * 64, "bootstrap_digest": "b" * 64, "lock_digest": "c" * 64})


def test_release_checksums_detect_changed_learning_artifact(release_bundle: Path) -> None:
    _resign(release_bundle, seal=True)
    (release_bundle.parent / "teach-back.md").write_text("modified", encoding="utf-8")
    with pytest.raises(IllmError, match="checksum mismatch"):
        verify_release_bundle(release_bundle)


def test_resigned_manifest_and_inventory_cannot_hide_failed_stall(release_bundle: Path) -> None:
    report_path = release_bundle.parent / "benchmark" / "event-stall" / "report.json"
    report = _read(report_path)
    report["producer_completed_while_exporter_blocked"] = False
    _write(report_path, report)
    manifest = _read(release_bundle)
    manifest["event_stall_digest"] = sha256_json(report)
    _write(release_bundle, manifest)
    _resign(release_bundle, seal=True)
    with pytest.raises(IllmError, match="forward progress"):
        verify_release_bundle(release_bundle)


@pytest.mark.parametrize("xml", [
    '<testsuite tests="1" failures="0" errors="0" skipped="1"><testcase name="skip"><skipped/></testcase></testsuite>',
    '<testsuite tests="1" failures="-1" errors="0" skipped="0"><testcase name="pass"/></testsuite>',
    '<testsuite tests="1" failures="0" errors="0" skipped="0" time="nan"><testcase name="pass"/></testsuite>',
    '<testsuite tests="99" failures="0" errors="0" skipped="0"><testcase name="pass"/></testsuite>',
    '<testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="failed"><failure/></testcase></testsuite>',
])
def test_junit_rejects_skipped_invalid_or_contradictory_results(tmp_path: Path, xml: str) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(xml, encoding="utf-8")
    with pytest.raises(IllmError):
        parse_junit(path)


def test_qualified_platform_cannot_be_relabeled(release_bundle: Path) -> None:
    profile = next(profile for profile in _read(release_bundle)["qualification_profiles"] if profile["profile_id"] == "linux-bootstrap-v1")
    root = release_bundle.parent / "qualification" / "profiles" / profile["profile_id"] / "evidence" / profile["evidence_digest"]
    index = _read(root / "evidence-index.json")
    index["arch"] = "incorrect-architecture"
    index["index_digest"] = sha256_json({key: value for key, value in index.items() if key != "index_digest"})
    _write(root / "evidence-index.json", index)
    with pytest.raises(IllmError, match="platform/finalization"):
        verify_profile_evidence(root)

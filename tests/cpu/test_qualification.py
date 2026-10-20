from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

import pytest

from illm.canonical import canonical_json_bytes, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.qualification import (
    create_profile_evidence,
    merge_profile_evidence,
    verify_profile_evidence,
)
from illm.toy import run_toy_scheduler


def _manifest_path() -> Path:
    value = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not value:
        pytest.skip("test must run through scripts/in_env.py")
    path = Path(value)
    if json.loads(path.read_text(encoding="utf-8"))["schema_version"] != 0:
        pytest.skip("V0.0 producer requires schema0; executed separately by legacy-format-contract CI")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite name="qualification" tests="1" failures="0" errors="0" skipped="0" time="0.01"><testcase name="pass"/></testsuite>\n',
        encoding="utf-8",
    )
    quality = tmp_path / "quality.json"
    quality.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 0,
                "status": "pass",
                "issues": [],
                "checked_python_files": 20,
                "type_check": {
                    "tool": "mypy",
                    "version": "1.18.2",
                    "returncode": 0,
                    "status": "pass",
                    "output": ["Success: no issues found"],
                },
            }
        )
        + b"\n"
    )
    capture = tmp_path / "capture"
    run_toy_scheduler(seed=19, steps=12, capture_root=capture, project_root=Path.cwd())
    return junit, quality, capture


def _profile(profile_id: str, result: str) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "result": result,
        "required_for_functional": True,
        "environment_manifest_digest": "a" * 64 if result == "pass" else None,
        "evidence_digest": "b" * 64 if result == "pass" else None,
        "reason": None if result == "pass" else "not_run_resource_unavailable",
    }


def test_platform_qualification_is_self_verifying_and_tamper_evident(tmp_path: Path) -> None:
    junit, quality, capture = _inputs(tmp_path)
    evidence = tmp_path / "evidence"
    profile = create_profile_evidence(
        toolchain_manifest=_manifest_path(),
        junit=junit,
        quality=quality,
        capture=capture,
        output=evidence,
    )
    verified_profile, index = verify_profile_evidence(evidence)
    assert verified_profile == profile
    assert profile["profile_id"] == (
        "windows-bootstrap-v1" if platform.system() == "Windows" else "linux-bootstrap-v1"
    )
    assert profile["evidence_digest"] == index["evidence_digest"]

    with (evidence / "quality.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(IllmError, match="checksum mismatch") as raised:
        verify_profile_evidence(evidence)
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT


def test_verified_profile_can_be_merged_without_overclaiming_linux(tmp_path: Path) -> None:
    junit, quality, capture = _inputs(tmp_path)
    evidence = tmp_path / "windows-evidence"
    created = create_profile_evidence(
        toolchain_manifest=_manifest_path(),
        junit=junit,
        quality=quality,
        capture=capture,
        output=evidence,
    )
    if created["profile_id"] != "windows-bootstrap-v1":
        pytest.skip("this assertion covers the current Windows development host")
    toolchain = json.loads(_manifest_path().read_text(encoding="utf-8"))
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    manifest = {
        "schema_version": 0,
        "spec_status": "accepted",
        "release_stage": "dev_snapshot",
        "qualification_tier": "learning_complete",
        "qualification_profiles": [
            _profile("cpu-clean-room-v1", "pass"),
            _profile("windows-bootstrap-v1", "pass"),
            _profile("linux-bootstrap-v1", "not_run_resource_unavailable"),
        ],
        "performance_outcome": "not_applicable",
        "lts_state": "not_applicable",
        "toolchain_manifest_digest": toolchain["manifest_digest"],
        "capture_manifest_digest": "c" * 64,
        "test_report_digest": "d" * 64,
        "event_stall_digest": "e" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    (release_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    verified = merge_profile_evidence(
        release_dir=release_dir,
        base_toolchain_manifest=_manifest_path(),
        evidence_roots=[evidence],
    )
    assert verified["recomputed_qualification_tier"] == "learning_complete"
    merged = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    windows = next(
        profile
        for profile in merged["qualification_profiles"]
        if profile["profile_id"] == "windows-bootstrap-v1"
    )
    assert windows["evidence_digest"] == created["evidence_digest"]


def test_deep_checkout_paths_support_sealing_verification_and_merge(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    junit, quality, capture = _inputs(inputs)
    original_manifest = _manifest_path()
    toolchain = json.loads(original_manifest.read_text(encoding="utf-8"))
    deep = tmp_path / ("nested-" + "a" * 70) / ("checkout-" + "b" * 70)

    def fixture_path(path: Path) -> Path:
        value = str(path.resolve())
        return Path("\\\\?\\" + value) if os.name == "nt" else path

    # Arrange files through the long-path API, then exercise public APIs with
    # ordinary paths as callers do. The sealed capture files exceed MAX_PATH.
    shutil.copytree(inputs, fixture_path(deep / "inputs"))
    long_manifest = deep / "inputs" / "toolchain.json"
    fixture_path(long_manifest).write_bytes(original_manifest.read_bytes())
    evidence = deep / "evidence"
    created = create_profile_evidence(
        toolchain_manifest=long_manifest,
        junit=deep / "inputs" / junit.name,
        quality=deep / "inputs" / quality.name,
        capture=deep / "inputs" / capture.name,
        output=evidence,
    )
    verified, index = verify_profile_evidence(evidence)
    assert verified == created
    assert index["source_identity"]["project_root_digest"] == toolchain["project_root_digest"]
    release_dir = deep / "release"
    fixture_path(release_dir).mkdir()
    manifest = {
        "schema_version": 0,
        "spec_status": "implemented",
        "release_stage": "release_candidate",
        "qualification_tier": "learning_complete",
        "qualification_profiles": [
            _profile("cpu-clean-room-v1", "pass"),
            _profile("windows-bootstrap-v1", "not_run_resource_unavailable"),
            _profile("linux-bootstrap-v1", "not_run_resource_unavailable"),
        ],
        "performance_outcome": "not_applicable",
        "lts_state": "not_applicable",
        "toolchain_manifest_digest": toolchain["manifest_digest"],
        "capture_manifest_digest": "c" * 64,
        "test_report_digest": "d" * 64,
        "event_stall_digest": "e" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    fixture_path(release_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    merged = merge_profile_evidence(
        release_dir=release_dir,
        base_toolchain_manifest=long_manifest,
        evidence_roots=[evidence],
    )
    assert merged["recomputed_qualification_tier"] == "learning_complete"
    imported = release_dir / "qualification" / "profiles" / created["profile_id"] / "evidence" / created["evidence_digest"]
    assert len(str(imported / "capture" / "capture-manifest.json")) > 260
    assert verify_profile_evidence(imported)[0] == created
    imported_toolchain = json.loads(fixture_path(imported / "toolchain.json").read_text(encoding="utf-8"))
    assert imported_toolchain == toolchain

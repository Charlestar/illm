from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import sha256_json
from illm.domain.errors import IllmError
from illm.release import recompute_qualification_tier, verify_release_manifest


def _profile(profile_id: str, result: str, *, required: bool = True) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "result": result,
        "required_for_functional": required,
        "environment_manifest_digest": "a" * 64 if result == "pass" else None,
        "evidence_digest": "b" * 64 if result == "pass" else None,
        "reason": None if result == "pass" else "not_run",
    }


def test_cpu_pass_permits_learning_but_not_functional_when_linux_missing() -> None:
    profiles = [
        _profile("cpu-clean-room-v1", "pass"),
        _profile("windows-bootstrap-v1", "pass"),
        _profile("linux-bootstrap-v1", "not_run_resource_unavailable"),
    ]
    assert recompute_qualification_tier(profiles) == "learning_complete"


def test_all_required_profiles_permit_functional_qualification() -> None:
    profiles = [
        _profile("cpu-clean-room-v1", "pass"),
        _profile("windows-bootstrap-v1", "pass"),
        _profile("linux-bootstrap-v1", "pass"),
    ]
    assert recompute_qualification_tier(profiles) == "functionally_qualified"


def test_claim_above_evidence_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 0,
        "spec_status": "accepted",
        "release_stage": "dev_snapshot",
        "qualification_tier": "functionally_qualified",
        "performance_outcome": "not_applicable",
        "lts_state": "not_applicable",
        "qualification_profiles": [
            _profile("cpu-clean-room-v1", "pass"),
            _profile("windows-bootstrap-v1", "pass"),
            _profile("linux-bootstrap-v1", "not_run_resource_unavailable"),
        ],
        "toolchain_manifest_digest": "c" * 64,
        "capture_manifest_digest": "d" * 64,
        "test_report_digest": "e" * 64,
        "event_stall_digest": "f" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IllmError, match="higher than evidence"):
        verify_release_manifest(path)


def test_release_manifest_digest_and_required_profile_set_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 0,
        "spec_status": "accepted",
        "release_stage": "dev_snapshot",
        "qualification_tier": "learning_complete",
        "performance_outcome": "not_applicable",
        "lts_state": "not_applicable",
        "qualification_profiles": [
            _profile("cpu-clean-room-v1", "pass"),
            _profile("windows-bootstrap-v1", "pass"),
        ],
        "toolchain_manifest_digest": "c" * 64,
        "capture_manifest_digest": "d" * 64,
        "test_report_digest": "e" * 64,
        "event_stall_digest": "f" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IllmError, match="profiles are missing"):
        verify_release_manifest(path)

    manifest["qualification_profiles"].append(
        _profile("linux-bootstrap-v1", "not_run_resource_unavailable")
    )
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest["spec_status"] = "implemented"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IllmError, match="digest mismatch"):
        verify_release_manifest(path)

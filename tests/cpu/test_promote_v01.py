"""Promotion transaction tests mock full evidence verification, not qualification."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import PROFILES, _inventory, _object, _write
from illm.reference_release import MANIFEST_FIELDS, _state


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import promote_v01 as promote


@pytest.fixture
def candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    project = tmp_path / "project"
    root = project / "release" / "v0.1.0"
    source = {"project_root_digest": "1" * 64, "bootstrap_digest": "2" * 64, "base_lock_digest": "3" * 64}
    monkeypatch.setattr(promote, "__file__", str(project / "scripts" / "promote_v01.py"))
    monkeypatch.setattr(promote, "reference_source_identity", lambda path: dict(source))
    spec = "# Synthetic promotion unit fixture\n\n```yaml\n" + "\n".join(f"{key}: {value}" for key, value in promote.RELEASED_AXES.items()) + "\n```\n"
    (project / promote.SPEC).parent.mkdir(parents=True)
    (project / promote.SPEC).write_text(spec, encoding="utf-8")
    root.mkdir(parents=True)
    (root / "spec.md").write_text(spec, encoding="utf-8")
    (root / "immutable-evidence.txt").write_text("Synthetic unit boundary. No model execution is represented.\n", encoding="utf-8")
    manifest: dict[str, Any] = {name: "4" * 64 for name in MANIFEST_FIELDS}
    manifest.update(schema_version=1, release_version="0.1.0", release_tag="v0.1.0", spec_status="implemented",
                    release_stage="release_candidate", qualification_tier="functionally_qualified",
                    performance_outcome="not_applicable", lts_state="not_applicable", source_identity=source,
                    primary_profile_id="tiny-reference-cpu-v1",
                    qualification_profiles=[{"profile_id": profile, "result": "pass", "required_for_functional": True,
                                             "environment_manifest_digest": "5" * 64, "evidence_digest": "6" * 64, "reason": None}
                                            for profile in sorted(PROFILES)])
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    _write(root / "manifest.json", manifest)
    _write(root / "qualification" / "release-state.json", _state(root / "manifest.json"))
    (root / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
    _write(root / "SHA256SUMS.json", _inventory(root))
    verified: list[dict[str, Any]] = []

    def boundary(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        assert path == root / "manifest.json" and expected_source_identity == source
        state = _state(path)
        assert _object(root / "qualification" / "release-state.json") == state
        assert (root / "COMPLETE").read_text(encoding="ascii") == state["manifest_digest"] + "\n"
        assert _object(root / "SHA256SUMS.json") == {name: digest for name, digest in _inventory(root).items() if name != "SHA256SUMS.json"}
        verified.append(state)
        return state

    monkeypatch.setattr(promote, "verify_reference_release_bundle", boundary)
    return {"project": project, "root": root, "source": source, "verified": verified, "boundary": boundary}


def test_promotion_changes_only_four_generated_files_and_reverifies(candidate: dict[str, Any]) -> None:
    root = candidate["root"]
    before = _inventory(root)
    spec_before = (candidate["project"] / promote.SPEC).read_bytes()
    result = promote.promote_reference_candidate(root)
    after = _inventory(root)
    assert result["spec_status"] == result["release_stage"] == "released"
    assert result["source_identity"] == candidate["source"]
    assert set(before) == set(after)
    assert {name for name in before if before[name] != after[name]} == promote.METADATA
    assert len(candidate["verified"]) == 2
    assert candidate["verified"][0]["release_stage"] == "release_candidate"
    assert (candidate["project"] / promote.SPEC).read_bytes() == spec_before


def test_already_released_is_verified_noop_without_rewriting_files(candidate: dict[str, Any]) -> None:
    root = candidate["root"]
    expected = promote.promote_reference_candidate(root)
    before = {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in promote.METADATA}
    assert promote.promote_reference_candidate(root) == expected
    assert before == {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in promote.METADATA}
    assert len(candidate["verified"]) == 3


@pytest.mark.parametrize("mutation", ["accepted", "comment-only", "duplicate", "copied-spec-only"])
def test_unreleased_or_unbound_frozen_spec_never_writes_metadata(candidate: dict[str, Any], mutation: str) -> None:
    path = candidate["project"] / promote.SPEC
    spec = path.read_text(encoding="utf-8")
    if mutation == "accepted":
        spec = spec.replace("spec_status: released", "spec_status: accepted")
    elif mutation == "comment-only":
        spec = spec.replace("release_stage: released", "release_stage: dev_snapshot # release_stage: released")
    elif mutation == "duplicate":
        spec = spec.replace("spec_status: released", "spec_status: released\nspec_status: accepted")
    else:
        spec += "\nChanged source not qualified by the candidate.\n"
    path.write_text(spec, encoding="utf-8")
    if mutation != "copied-spec-only":
        (candidate["root"] / "spec.md").write_text(spec, encoding="utf-8")
        _write(candidate["root"] / "SHA256SUMS.json", {name: digest for name, digest in _inventory(candidate["root"]).items() if name != "SHA256SUMS.json"})
    before = _inventory(candidate["root"])
    with pytest.raises(IllmError):
        promote.promote_reference_candidate(candidate["root"])
    assert _inventory(candidate["root"]) == before
    assert len(candidate["verified"]) == 1


def test_other_release_and_v00_are_rejected_before_verification(candidate: dict[str, Any]) -> None:
    for target in (candidate["root"].parent / "v0.0.0", candidate["project"] / "out" / "v0.1.0"):
        with pytest.raises(IllmError) as raised:
            promote.promote_reference_candidate(target)
        assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert candidate["verified"] == []


def test_failed_write_restores_original_bytes_and_verifies_them(candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    root = candidate["root"]
    before = _inventory(root)

    def write(path: Path, value: Mapping[str, Any]) -> None:
        if path.name == "release-state.json":
            raise OSError("synthetic disk failure")
        _write(path, value)

    monkeypatch.setattr(promote, "_write", write)
    with pytest.raises(IllmError, match="original release metadata was restored and verified"):
        promote.promote_reference_candidate(root)
    assert _inventory(root) == before
    assert len(candidate["verified"]) == 2


def test_failed_post_verification_restores_candidate(candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    before = _inventory(candidate["root"])
    calls = 0

    def boundary(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "synthetic final semantic failure")
        result: dict[str, Any] = candidate["boundary"](path, expected_source_identity=expected_source_identity)
        return result

    monkeypatch.setattr(promote, "verify_reference_release_bundle", boundary)
    with pytest.raises(IllmError, match="restored and verified"):
        promote.promote_reference_candidate(candidate["root"])
    assert calls == 3 and _inventory(candidate["root"]) == before


def test_restoration_failure_is_explicit_not_reported_as_success(candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_write(path: Path, value: Mapping[str, Any]) -> None:
        raise OSError("synthetic write failure")

    def failed_restore(path: Path, value: bytes) -> None:
        raise OSError("synthetic restore failure")

    monkeypatch.setattr(promote, "_write", failed_write)
    monkeypatch.setattr(promote, "_replace_bytes", failed_restore)
    with pytest.raises(IllmError, match="restoration could not be confirmed; do not publish") as raised:
        promote.promote_reference_candidate(candidate["root"])
    assert raised.value.code is ErrorCode.INTERNAL


def test_preflight_full_verification_failure_never_writes(candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    before = _inventory(candidate["root"])

    def reject(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "candidate belongs to different source")

    monkeypatch.setattr(promote, "verify_reference_release_bundle", reject)
    with pytest.raises(IllmError, match="different source"):
        promote.promote_reference_candidate(candidate["root"])
    assert _inventory(candidate["root"]) == before


def test_cli_reports_promotion_result(candidate: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert promote.main(["--release-dir", str(candidate["root"])]) == 0
    assert json.loads(capsys.readouterr().out)["release_stage"] == "released"

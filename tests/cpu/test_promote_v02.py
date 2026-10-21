"""Unit-only V0.2 promotion transaction tests; fake evidence is never qualification proof."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, cast

import pytest

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import _inventory, _object, _write
from illm.stateful_qualification import FORMAL_PROFILES
from illm.stateful_release import (
    BENCHMARKS,
    MANIFEST_FIELDS,
    PRIMARY,
    REPORT_KIND,
    TAG,
    VERSION,
    _state,
    verify_frozen_v02_spec,
    verify_stateful_release_bundle as real_verify_stateful_release_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import promote_v02 as promote


def _unit_spec() -> str:
    lines = [
        "status: implemented",
        "spec_status: implemented",
        "release_stage: dev_snapshot",
        "qualification_tier: unqualified",
        f"target_release: {TAG}",
        "performance_outcome: not_applicable",
        "lts_state: not_applicable",
        "spec_version: 2",
        "qualification_profiles:",
    ]
    for profile_id in sorted(FORMAL_PROFILES):
        lines.extend((
            f"  - profile_id: {profile_id}",
            "    result: not_run",
            "    reason: final_source_bound_gate_pending",
        ))
    return (
        "# Synthetic V0.2 promotion unit fixture\n\n```yaml\n"
        + "\n".join(lines)
        + "\n```\n\n## 12. Definition of Done\n\n### 12.1 Must\n\n"
        "- [ ] synthetic unit transaction gate\n\n### 12.2 Conditional\n\n"
        "- [ ] synthetic unit conditional gate\n"
    )


@pytest.fixture
def candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build only a schema-shaped transaction fixture and mock the full evidence reader."""
    project = tmp_path / "project"
    root = project / "release" / "v0.2.0"
    source = {"project_root_digest": "1" * 64, "bootstrap_digest": "2" * 64, "base_lock_digest": "3" * 64}
    monkeypatch.setattr(promote, "__file__", str(project / "scripts" / "promote_v02.py"))
    monkeypatch.setattr(promote, "stateful_source_identity", lambda path: dict(source))

    spec = _unit_spec()
    (project / promote.SPEC).parent.mkdir(parents=True)
    (project / promote.SPEC).write_text(spec, encoding="utf-8")
    root.mkdir(parents=True)
    (root / "spec.md").write_text(spec, encoding="utf-8")
    (root / "unit-only-evidence.txt").write_text(
        "Synthetic transaction fixture: no GPU, model, qualification, or release proof.\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {name: "4" * 64 for name in MANIFEST_FIELDS}
    manifest.update(
        schema_version=2,
        report_kind=REPORT_KIND,
        release_version=VERSION,
        release_tag=TAG,
        spec_status="implemented",
        release_stage="release_candidate",
        qualification_tier="functionally_qualified",
        performance_outcome="not_applicable",
        lts_state="not_applicable",
        source_identity=source,
        primary_profile_id=PRIMARY,
        qualification_profiles=[
            {
                "profile_id": profile,
                "simulated_profile_id": None,
                "evidence_mode": "formal",
                "result": "pass",
                "required_for_functional": True,
                "environment_manifest_digest": "5" * 64,
                "evidence_digest": "6" * 64,
                "reason": None,
            }
            for profile in sorted(FORMAL_PROFILES)
        ],
        capture_manifests={"unit_only": "not_real_capture_evidence"},
        execution_identities={"unit_only": "not_real_execution_evidence"},
        benchmark_report_digests={name: "7" * 64 for name in BENCHMARKS},
    )
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    _write(root / "manifest.json", manifest)
    _write(root / "qualification" / "release-state.json", _state(root / "manifest.json"))
    (root / "COMPLETE").write_text(str(manifest["manifest_digest"]) + "\n", encoding="ascii")
    _write(root / "SHA256SUMS.json", _inventory(root))
    verified: list[dict[str, Any]] = []

    def unit_only_boundary(
        path: Path, *, expected_source_identity: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        assert path == root / "manifest.json" and expected_source_identity == source
        state = _state(path)
        assert _object(root / "qualification" / "release-state.json") == state
        assert (root / "COMPLETE").read_text(encoding="ascii") == state["manifest_digest"] + "\n"
        assert _object(root / "SHA256SUMS.json") == {
            name: digest for name, digest in _inventory(root).items() if name != "SHA256SUMS.json"
        }
        verified.append(state)
        return state

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", unit_only_boundary)
    return {
        "project": project,
        "root": root,
        "source": source,
        "verified": verified,
        "boundary": unit_only_boundary,
    }


def test_promotion_changes_only_four_generated_files_and_reverifies(candidate: dict[str, Any]) -> None:
    root = cast(Path, candidate["root"])
    project = cast(Path, candidate["project"])
    before = _inventory(root)
    spec_before = (project / promote.SPEC).read_bytes()

    result = promote.promote_stateful_candidate(root)
    after = _inventory(root)

    assert result["spec_status"] == result["release_stage"] == "released"
    assert result["source_identity"] == candidate["source"]
    assert set(before) == set(after)
    assert {name for name in before if before[name] != after[name]} == promote.METADATA
    assert len(candidate["verified"]) == 2
    assert candidate["verified"][0]["release_stage"] == "release_candidate"
    assert (project / promote.SPEC).read_bytes() == spec_before
    assert (root / "spec.md").read_bytes() == spec_before
    verify_frozen_v02_spec(root / "spec.md")
    assert set(_object(root / "manifest.json")) == MANIFEST_FIELDS
    release_state = _object(root / "qualification" / "release-state.json")
    assert release_state["recomputed_qualification_tier"] == "functionally_qualified"
    assert release_state["verification_digest"] == result["verification_digest"]
    assert not (root.parent / promote.LOCK_NAME).exists()


def test_already_released_is_reverified_noop_without_rewriting(candidate: dict[str, Any]) -> None:
    root = cast(Path, candidate["root"])
    expected = promote.promote_stateful_candidate(root)
    before = {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in promote.METADATA}

    assert promote.promote_stateful_candidate(root) == expected
    assert before == {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in promote.METADATA}
    assert len(candidate["verified"]) == 3
    assert not (root.parent / promote.LOCK_NAME).exists()


@pytest.mark.parametrize(
    "mutation",
    ["draft", "released-target", "duplicate", "profile-pass", "wrong-spec-version", "checked-dod", "checkout-only"],
)
def test_non_snapshot_or_unbound_frozen_spec_never_writes_metadata(
    candidate: dict[str, Any], mutation: str,
) -> None:
    project = cast(Path, candidate["project"])
    root = cast(Path, candidate["root"])
    path = project / promote.SPEC
    spec = path.read_text(encoding="utf-8")
    if mutation == "draft":
        spec = spec.replace("spec_status: implemented", "spec_status: draft")
    elif mutation == "released-target":
        spec = spec.replace("release_stage: dev_snapshot", "release_stage: released")
    elif mutation == "duplicate":
        spec = spec.replace("spec_status: implemented", "spec_status: implemented\nspec_status: released")
    elif mutation == "profile-pass":
        spec = spec.replace("    result: not_run", "    result: pass", 1)
    elif mutation == "wrong-spec-version":
        spec = spec.replace("spec_version: 2", "spec_version: 999")
    elif mutation == "checked-dod":
        spec = spec.replace("- [ ] synthetic unit transaction gate", "- [x] synthetic unit transaction gate")
    elif mutation != "checkout-only":
        raise AssertionError("unknown test mutation")
    if mutation == "checkout-only":
        spec += "\nCheckout changed after candidate assembly.\n"
    path.write_text(spec, encoding="utf-8")
    if mutation != "checkout-only":
        (root / "spec.md").write_text(spec, encoding="utf-8")
        _write(root / "SHA256SUMS.json", {
            name: digest for name, digest in _inventory(root).items() if name != "SHA256SUMS.json"
        })
    before = _inventory(root)

    with pytest.raises(IllmError):
        promote.promote_stateful_candidate(root)

    assert _inventory(root) == before
    assert len(candidate["verified"]) == 1


def test_other_release_and_output_paths_are_rejected_before_verification(candidate: dict[str, Any]) -> None:
    root = cast(Path, candidate["root"])
    project = cast(Path, candidate["project"])
    for target in (root.parent / "v0.1.0", project / "out" / "v0.2.0"):
        with pytest.raises(IllmError) as raised:
            promote.promote_stateful_candidate(target)
        assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert candidate["verified"] == []


def test_linked_release_path_is_rejected_before_verification(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    original = promote._linked
    monkeypatch.setattr(promote, "_linked", lambda path: path == root or original(path))
    with pytest.raises(IllmError, match="cannot follow"):
        promote.promote_stateful_candidate(root)
    assert candidate["verified"] == []


def test_link_detector_rejects_junctions() -> None:
    class Junction:
        def is_symlink(self) -> bool:
            return False

        def is_junction(self) -> bool:
            return True

    assert promote._linked(cast(Path, Junction()))


def test_failed_write_restores_exact_original_bytes_and_reverifies(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    before = {name: (root / name).read_bytes() for name in promote.METADATA}
    original_replace_json = promote._replace_json

    def replace_json(path: Path, value: Mapping[str, Any]) -> None:
        if path.name == "release-state.json":
            raise OSError("synthetic disk failure")
        original_replace_json(path, value)

    monkeypatch.setattr(promote, "_replace_json", replace_json)
    with pytest.raises(IllmError, match="original release metadata was restored and verified"):
        promote.promote_stateful_candidate(root)
    assert before == {name: (root / name).read_bytes() for name in promote.METADATA}
    assert len(candidate["verified"]) == 2
    assert not (root.parent / promote.LOCK_NAME).exists()


def test_failed_post_verification_restores_candidate(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    before = _inventory(root)
    calls = 0

    def boundary(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "synthetic final semantic failure")
        result: dict[str, Any] = candidate["boundary"](
            path, expected_source_identity=expected_source_identity,
        )
        return result

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", boundary)
    with pytest.raises(IllmError, match="restored and verified"):
        promote.promote_stateful_candidate(root)
    assert calls == 3
    assert _inventory(root) == before


def test_unconfirmed_final_stage_is_restored(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    before = _inventory(root)
    calls = 0

    def boundary(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result: dict[str, Any] = candidate["boundary"](
            path, expected_source_identity=expected_source_identity,
        )
        if calls == 2:
            result = {**result, "spec_status": "implemented", "release_stage": "release_candidate"}
        return result

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", boundary)
    with pytest.raises(IllmError, match="restored and verified"):
        promote.promote_stateful_candidate(root)
    assert calls == 3
    assert _inventory(root) == before


def test_restoration_failure_is_internal_and_never_success(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_json(path: Path, value: Mapping[str, Any]) -> None:
        raise OSError("synthetic write failure")

    def failed_restore(path: Path, value: bytes) -> None:
        raise OSError("synthetic restore failure")

    monkeypatch.setattr(promote, "_replace_json", failed_json)
    monkeypatch.setattr(promote, "_replace_bytes", failed_restore)
    with pytest.raises(IllmError, match="restoration could not be confirmed; do not publish") as raised:
        promote.promote_stateful_candidate(cast(Path, candidate["root"]))
    assert raised.value.code is ErrorCode.INTERNAL


def test_source_change_before_writes_never_mutates_candidate(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    source = cast(dict[str, str], candidate["source"])
    changed = {**source, "project_root_digest": "9" * 64}
    values = iter((source, changed))
    monkeypatch.setattr(promote, "stateful_source_identity", lambda path: dict(next(values)))
    before = _inventory(root)
    with pytest.raises(IllmError, match="changed before promotion"):
        promote.promote_stateful_candidate(root)
    assert _inventory(root) == before


def test_source_change_after_writes_restores_candidate(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    source = cast(dict[str, str], candidate["source"])
    changed = {**source, "project_root_digest": "9" * 64}
    values = iter((source, source, changed))
    monkeypatch.setattr(promote, "stateful_source_identity", lambda path: dict(next(values)))
    before = _inventory(root)
    with pytest.raises(IllmError, match="restored and verified"):
        promote.promote_stateful_candidate(root)
    assert _inventory(root) == before


def test_source_change_during_final_verification_restores_candidate(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    source = cast(dict[str, str], candidate["source"])
    changed = {**source, "project_root_digest": "9" * 64}
    values = iter((source, source, source, changed))
    monkeypatch.setattr(promote, "stateful_source_identity", lambda path: dict(next(values)))
    before = _inventory(root)
    with pytest.raises(IllmError, match="restored and verified"):
        promote.promote_stateful_candidate(root)
    assert _inventory(root) == before


def test_preflight_full_verification_failure_never_writes(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    before = _inventory(root)

    def reject(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "candidate belongs to different source")

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", reject)
    with pytest.raises(IllmError, match="different source"):
        promote.promote_stateful_candidate(root)
    assert _inventory(root) == before


def test_unit_fixture_is_rejected_by_real_portable_reader(candidate: dict[str, Any]) -> None:
    """The transaction mock must never turn schema-shaped fake data into qualification."""
    with pytest.raises(IllmError):
        real_verify_stateful_release_bundle(
            cast(Path, candidate["root"]) / "manifest.json",
            expected_source_identity=cast(dict[str, str], candidate["source"]),
        )


def test_cli_reports_promotion_result(candidate: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert promote.main(["--release-dir", str(candidate["root"])]) == 0
    assert json.loads(capsys.readouterr().out)["release_stage"] == "released"


def test_existing_lock_is_never_removed_or_reinterpreted_as_stale(candidate: dict[str, Any]) -> None:
    root = cast(Path, candidate["root"])
    lock = root.parent / promote.LOCK_NAME
    receipt = b'{"pid":999999999,"owner_token":"not-this-transaction"}\n'
    lock.write_bytes(receipt)
    before = _inventory(root)
    with pytest.raises(IllmError, match="lock already exists") as raised:
        promote.promote_stateful_candidate(root)
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert lock.read_bytes() == receipt
    assert _inventory(root) == before
    assert candidate["verified"] == []


def test_preflight_exception_cleans_only_this_calls_lock(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    before = _inventory(root)

    def failed(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        assert (root.parent / promote.LOCK_NAME).is_file()
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "unit preflight failure while owned")

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", failed)
    with pytest.raises(IllmError, match="unit preflight failure while owned"):
        promote.promote_stateful_candidate(root)
    assert not (root.parent / promote.LOCK_NAME).exists()
    assert _inventory(root) == before


def test_already_released_noop_is_covered_by_the_same_lock(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    expected = promote.promote_stateful_candidate(root)
    before = _inventory(root)
    original = candidate["boundary"]

    def owned(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        assert (root.parent / promote.LOCK_NAME).is_file()
        result: dict[str, Any] = original(path, expected_source_identity=expected_source_identity)
        return result

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", owned)
    assert promote.promote_stateful_candidate(root) == expected
    assert _inventory(root) == before
    assert not (root.parent / promote.LOCK_NAME).exists()


def test_modified_owner_receipt_is_not_deleted_and_cleanup_is_internal(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    lock = root.parent / promote.LOCK_NAME
    before = _inventory(root)
    replacement = b"ownership no longer belongs to the original receipt\n"

    def changed(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
        lock.write_bytes(replacement)
        raise OSError("unit ownership interference")

    monkeypatch.setattr(promote, "verify_stateful_release_bundle", changed)
    with pytest.raises(IllmError, match="lock cleanup could not be confirmed; do not publish") as raised:
        promote.promote_stateful_candidate(root)
    assert raised.value.code is ErrorCode.INTERNAL
    assert lock.read_bytes() == replacement
    assert _inventory(root) == before


def test_lock_unlink_failure_never_reports_success(
    candidate: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cast(Path, candidate["root"])
    original = Path.unlink

    def failed(path: Path, missing_ok: bool = False) -> None:
        if path.name == promote.LOCK_NAME:
            raise OSError("unit lock removal failure")
        original(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failed)
    with pytest.raises(IllmError, match="do not publish") as raised:
        promote.promote_stateful_candidate(root)
    assert raised.value.code is ErrorCode.INTERNAL
    assert (root.parent / promote.LOCK_NAME).is_file()
    # Metadata can already be released; failed lock cleanup must override the
    # return value rather than falsely authorizing publication.
    assert _state(root / "manifest.json")["release_stage"] == "released"


_PROCESS_UNIT = r"""
import json,os,sys
from pathlib import Path
repo,project,ready,mode=map(str,sys.argv[1:5])
sys.path[:0]=[str(Path(repo)/'src'),str(Path(repo)/'scripts')]
import promote_v02 as promote
from illm.reference_qualification import _inventory,_object
source=json.loads(sys.argv[5])
project=Path(project)
root=project/'release/v0.2.0'
promote.__file__=str(project/'scripts/promote_v02.py')
promote.stateful_source_identity=lambda path:dict(source)
calls=0
def synthetic_boundary(path,*,expected_source_identity=None):
    global calls
    assert expected_source_identity==source and path==root/'manifest.json'
    calls+=1
    if calls==1:
        assert (root.parent/promote.LOCK_NAME).is_file()
        if mode=='crash':
            os._exit(37)
        Path(ready).write_text('synthetic process owns the lock\n',encoding='ascii')
        assert sys.stdin.readline().strip()=='continue'
    state=promote._state(path)
    assert _object(root/'qualification/release-state.json')==state
    assert (root/'COMPLETE').read_text(encoding='ascii')==state['manifest_digest']+'\n'
    assert _object(root/'SHA256SUMS.json')=={k:v for k,v in _inventory(root).items() if k!='SHA256SUMS.json'}
    return state
promote.verify_stateful_release_bundle=synthetic_boundary
result=promote.promote_stateful_candidate(root)
print(json.dumps({'evidence_mode':'synthetic_transaction_unit_only','release_stage':result['release_stage']}),flush=True)
"""


def _unit_process_command(candidate: Mapping[str, Any], ready: Path, mode: str) -> list[str]:
    return [sys.executable, "-I", "-B", "-c", _PROCESS_UNIT, str(ROOT), str(candidate["project"]),
            str(ready), mode, json.dumps(candidate["source"], sort_keys=True)]


def test_independent_process_competitor_cannot_snapshot_or_rollback_an_owner(
    candidate: dict[str, Any], tmp_path: Path,
) -> None:
    root = cast(Path, candidate["root"])
    ready = tmp_path / "independent-owner-ready"
    before = _inventory(root)
    process = subprocess.Popen(_unit_process_command(candidate, ready, "hold"), cwd=tmp_path,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file(), "independent owner did not acquire the lock"
        assert (root.parent / promote.LOCK_NAME).is_file()
        with pytest.raises(IllmError, match="lock already exists"):
            promote.promote_stateful_candidate(root)
        assert candidate["verified"] == []  # No competing preflight/snapshot.
        assert _inventory(root) == before
    finally:
        try:
            stdout, stderr = process.communicate("continue\n", timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    assert process.returncode == 0, stderr
    assert json.loads(stdout) == {"evidence_mode": "synthetic_transaction_unit_only", "release_stage": "released"}
    assert not (root.parent / promote.LOCK_NAME).exists()
    after = _inventory(root)
    # A later call may verify an already-released noop, never restore the old
    # candidate snapshot that it was prevented from taking during ownership.
    assert promote.promote_stateful_candidate(root)["release_stage"] == "released"
    assert _inventory(root) == after


def test_crashed_independent_owner_leaves_a_fail_closed_lock(
    candidate: dict[str, Any], tmp_path: Path,
) -> None:
    root = cast(Path, candidate["root"])
    before = _inventory(root)
    process = subprocess.run(_unit_process_command(candidate, tmp_path / "unused-ready", "crash"),
                             cwd=tmp_path, input="", capture_output=True, text=True, check=False, timeout=30)
    assert process.returncode == 37, process.stderr
    lock = root.parent / promote.LOCK_NAME
    receipt = lock.read_bytes()
    assert json.loads(receipt)["kind"] == "illm-v02-promotion-lock-v1"
    with pytest.raises(IllmError, match="lock already exists"):
        promote.promote_stateful_candidate(root)
    assert lock.read_bytes() == receipt
    assert _inventory(root) == before
    assert candidate["verified"] == []

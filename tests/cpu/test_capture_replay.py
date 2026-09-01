from __future__ import annotations

import json
from pathlib import Path

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture import CaptureBundle, CaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.replay import replay_events
from illm.toy import run_toy_scheduler


def _replay(bundle: CaptureBundle) -> str:
    return replay_events(bundle.initial_state, bundle.events, bundle.expected_step_digests).digest


def _resign_bundle(root: Path) -> None:
    checksums = {
        "checkpoint.json": sha256_file(root / "checkpoint.json"),
        "events.jsonl": sha256_file(root / "events.jsonl"),
        "state-digests.jsonl": sha256_file(root / "state-digests.jsonl"),
    }
    (root / "checksums.json").write_bytes(canonical_json_bytes(checksums) + b"\n")
    manifest = json.loads((root / "capture-manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {**checksums, "checksums.json": sha256_file(root / "checksums.json")}
    manifest["event_count"] = len((root / "events.jsonl").read_text(encoding="utf-8").splitlines())
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = sha256_json(manifest)
    (root / "capture-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")


def test_online_and_replay_digest_match(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    online = run_toy_scheduler(seed=7, steps=20, capture_root=root, project_root=Path.cwd())
    bundle = CaptureBundle.open(root)
    assert _replay(bundle) == online.state.digest == bundle.manifest["final_state_digest"]
    assert not online.state.waiting()
    assert all(request.status.name == "FINISHED" for request in online.state.requests)


def test_same_seed_has_same_semantic_event_sequence(tmp_path: Path) -> None:
    first = run_toy_scheduler(seed=13, steps=30, capture_root=tmp_path / "a", project_root=Path.cwd())
    second = run_toy_scheduler(seed=13, steps=30, capture_root=tmp_path / "b", project_root=Path.cwd())
    first_events = CaptureBundle.open(tmp_path / "a").events
    second_events = CaptureBundle.open(tmp_path / "b").events
    def semantic(event: DomainEvent) -> tuple[int, EventKind, dict[str, object]]:
        return event.seq_no, event.kind, dict(event.payload)
    assert [semantic(event) for event in first_events] == [semantic(event) for event in second_events]
    assert first.state.digest == second.state.digest


@pytest.mark.property
def test_100_seeds_by_100_steps_have_zero_digest_mismatch(tmp_path: Path) -> None:
    mismatch_count = 0
    for seed in range(100):
        root = tmp_path / f"seed-{seed:03d}"
        online = run_toy_scheduler(seed=seed, steps=100, capture_root=root, project_root=Path.cwd())
        replay_digest = _replay(CaptureBundle.open(root))
        mismatch_count += int(replay_digest != online.state.digest)
    assert mismatch_count == 0


def test_missing_event_is_not_silently_replayed(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    run_toy_scheduler(seed=3, steps=12, capture_root=root, project_root=Path.cwd())
    lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    del lines[3]
    (root / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _resign_bundle(root)
    with pytest.raises(IllmError, match="sequence gap") as raised:
        CaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_resigned_manifest_cannot_omit_a_fact_checksum(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    run_toy_scheduler(seed=3, steps=12, capture_root=root, project_root=Path.cwd())
    manifest = json.loads((root / "capture-manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["events.jsonl"]
    manifest.pop("manifest_digest")
    manifest["manifest_digest"] = sha256_json(manifest)
    (root / "capture-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(IllmError, match="exact v0 fact set") as raised:
        CaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_resigned_transaction_conflict_is_rejected_by_r1(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    run_toy_scheduler(seed=3, steps=12, capture_root=root, project_root=Path.cwd())
    values = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = next(item for item in values if item["kind_code"] == int(EventKind.REQUEST_SELECTED))
    selected["kind_code"] = int(EventKind.REQUEST_DEFERRED)
    selected["kind"] = EventKind.REQUEST_DEFERRED.wire_name
    (root / "events.jsonl").write_bytes(
        b"".join(canonical_json_bytes(item) + b"\n" for item in values)
    )
    _resign_bundle(root)
    bundle = CaptureBundle.open(root)
    with pytest.raises(IllmError, match="only a selected request may complete") as raised:
        _replay(bundle)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_resigned_noncontiguous_state_digest_records_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    run_toy_scheduler(seed=5, steps=8, capture_root=root, project_root=Path.cwd())
    records = [
        json.loads(line)
        for line in (root / "state-digests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    records[1]["step_id"] = 7
    (root / "state-digests.jsonl").write_bytes(
        b"".join(canonical_json_bytes(item) + b"\n" for item in records)
    )
    _resign_bundle(root)
    with pytest.raises(IllmError, match="not contiguous"):
        CaptureBundle.open(root)


def test_changed_fact_fails_checksum_validation(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    run_toy_scheduler(seed=4, steps=10, capture_root=root, project_root=Path.cwd())
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(IllmError, match="checksum mismatch"):
        CaptureBundle.open(root)


def test_unfinalized_capture_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    CaptureSession(root, 1, 2, Path.cwd())
    with pytest.raises(IllmError, match="not finalized") as raised:
        CaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE

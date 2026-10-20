from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle, ReferenceCaptureSession, replay_reference_events
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEvent, ReferenceEventKind as Kind
from illm.sampling import SamplingSpec


def _identity() -> dict[str, str]:
    return {
        "model_spec_id": "tiny-reference-v1", "weights_digest": "a" * 64,
        "load_manifest_digest": "b" * 64, "tokenizer_digest": "c" * 64,
        "input_spec_digest": "d" * 64,
    }


def _session(path: Path, mode: str = "greedy") -> ReferenceCaptureSession:
    spec = SamplingSpec() if mode == "greedy" else SamplingSpec(mode="sample", temperature=1, actual_seed=7)
    session = ReferenceCaptureSession(path, _identity(), "request-0", spec.to_dict())
    session.emit(Kind.REQUEST_ACCEPTED, {"max_new_tokens": 2})
    session.emit(Kind.REQUEST_TOKENIZED, {"input_tokens": 3, "input_digest": "e" * 64})
    return session


def _iteration(session: ReferenceCaptureSession, position: int) -> None:
    session.emit(Kind.MODEL_FORWARD_STARTED, {"output_position": position, "context_tokens": 3 + position})
    session.emit(Kind.MODEL_OP, {"output_position": position, "op_name": "layers.0.attention", "shape": [1, 3 + position, 32], "dtype": "fp32"})
    session.emit(Kind.MODEL_FORWARD_COMPLETED, {"output_position": position, "logits_digest": "f" * 64})
    session.emit(Kind.SAMPLING_STARTED, {"output_position": position, "sampling_spec_digest": session.state["sampling_spec_digest"]})
    session.emit(Kind.SAMPLING_SELECTED, {
        "output_position": position, "sampling_spec_digest": session.state["sampling_spec_digest"],
        "logits_digest": "f" * 64, "selected_digest": "1" * 64,
        "draw_address": None if session.state["sampling_spec"]["mode"] == "greedy" else [position, 0, 0],
    })


def _finished(path: Path, mode: str = "greedy") -> ReferenceCaptureSession:
    session = _session(path, mode)
    for position in range(2):
        _iteration(session, position)
    session.emit(Kind.REQUEST_FINISHED, {"output_tokens": 2, "output_digest": "2" * 64, "finish_reason": "MAX_NEW_TOKENS"})
    session.finalize()
    return session


def _write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _resign(path: Path) -> None:
    checksums = {name: sha256_file(path / name) for name in ("checkpoint.json", "events.jsonl", "state-digests.jsonl")}
    _write(path / "checksums.json", checksums)
    manifest = json.loads((path / "capture-manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {**checksums, "checksums.json": sha256_file(path / "checksums.json")}
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    _write(path / "capture-manifest.json", manifest)


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_reference_capture_reconstructs_control_state_without_numerical_rerun(tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _finished(tmp_path / "capture", mode)
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("R1 must not execute the numerical sampler")
    monkeypatch.setattr("illm.sampling.reference.prepare_sample", forbidden)
    monkeypatch.setattr("illm.sampling.reference.commit_sample", forbidden)
    bundle = ReferenceCaptureBundle.open(session.root)
    assert bundle.final_state == session.state == replay_reference_events(bundle.events, bundle.initial_state)
    assert bundle.final_state["next_output_position"] == bundle.final_state["output_tokens"] == 2
    assert bundle.final_state["rng_draws_consumed"] == (0 if mode == "greedy" else 2)
    assert bundle.final_state["committed_draw_addresses"] == ([] if mode == "greedy" else [[0, 0, 0], [1, 0, 0]])
    assert bundle.manifest["replay_capabilities"] == ["R0", "R1_REFERENCE"]
    assert bundle.manifest["privacy_mode"] == "metadata_only"
    assert bundle.manifest["run_status"] == "FINISHED"
    assert len(bundle.state_digests) == len(bundle.events)
    assert "duration_ns" not in next(event.payload for event in bundle.events if event.kind is Kind.MODEL_OP)


def test_reference_capture_remains_valid_after_relocation(tmp_path: Path) -> None:
    session = _finished(tmp_path / "original")
    shutil.copytree(session.root, tmp_path / "relocated")
    assert ReferenceCaptureBundle.open(tmp_path / "relocated", expected_identity=_identity()).final_state == session.state


def test_sampling_failure_and_request_terminal_do_not_commit_a_draw(tmp_path: Path) -> None:
    session = _session(tmp_path / "failed", "sample")
    session.emit(Kind.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": 3})
    session.emit(Kind.MODEL_FORWARD_COMPLETED, {"output_position": 0, "logits_digest": "f" * 64})
    session.emit(Kind.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": session.state["sampling_spec_digest"]})
    session.emit(Kind.SAMPLING_FAILED, {"output_position": 0, "error_code": "NUMERICAL_ERROR"})
    session.emit(Kind.REQUEST_FAILED, {"error_code": "NUMERICAL_ERROR"})
    manifest = session.finalize()
    bundle = ReferenceCaptureBundle.open(session.root)
    assert manifest["run_status"] == bundle.final_state["status"] == "FAILED"
    assert bundle.final_state["next_output_position"] == bundle.final_state["rng_draws_consumed"] == 0
    assert bundle.final_state["error_code"] == "NUMERICAL_ERROR"


def test_request_failure_can_finalize_before_tokenization(tmp_path: Path) -> None:
    session = ReferenceCaptureSession(tmp_path / "failed", _identity(), "request-0", SamplingSpec().to_dict())
    session.emit(Kind.REQUEST_ACCEPTED, {"max_new_tokens": 2})
    session.emit(Kind.REQUEST_FAILED, {"error_code": "INVALID_ARGUMENT"})
    assert session.finalize()["run_status"] == "FAILED"
    assert ReferenceCaptureBundle.open(session.root).final_state["input_tokens"] == 0


def test_illegal_order_duplicate_terminal_and_output_limit_fail_closed(tmp_path: Path) -> None:
    session = _session(tmp_path / "capture")
    with pytest.raises(IllmError, match="before forward"):
        session.emit(Kind.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": session.state["sampling_spec_digest"]})
    with pytest.raises(IllmError, match="full-context"):
        session.emit(Kind.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": 2})
    for position in range(2):
        _iteration(session, position)
    with pytest.raises(IllmError, match="eligible"):
        session.emit(Kind.MODEL_FORWARD_STARTED, {"output_position": 2, "context_tokens": 5})
    session.emit(Kind.REQUEST_FINISHED, {"output_tokens": 2, "output_digest": "2" * 64, "finish_reason": "MAX_NEW_TOKENS"})
    with pytest.raises(IllmError, match="after its terminal"):
        session.emit(Kind.REQUEST_FAILED, {"error_code": "INTERNAL"})
    session.finalize()


@pytest.mark.parametrize("mode,address", [("greedy", [0, 0, 0]), ("sample", None), ("sample", [0, 1, 0]), ("sample", [True, 0, 0])])
def test_wrong_random_address_or_greedy_draw_never_commits(tmp_path: Path, mode: str, address: Any) -> None:
    session = _session(tmp_path / "capture", mode)
    session.emit(Kind.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": 3})
    session.emit(Kind.MODEL_FORWARD_COMPLETED, {"output_position": 0, "logits_digest": "f" * 64})
    session.emit(Kind.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": session.state["sampling_spec_digest"]})
    with pytest.raises(IllmError):
        session.emit(Kind.SAMPLING_SELECTED, {"output_position": 0, "sampling_spec_digest": session.state["sampling_spec_digest"], "logits_digest": "f" * 64, "selected_digest": "1" * 64, "draw_address": address})
    assert session.state["next_output_position"] == session.state["rng_draws_consumed"] == 0


def test_default_metadata_contract_rejects_private_payloads_and_identity_fields(tmp_path: Path) -> None:
    session = _session(tmp_path / "capture")
    with pytest.raises(IllmError, match="metadata-only"):
        session.emit(Kind.REQUEST_FAILED, {"error_code": "INTERNAL", "prompt": "private prompt"})
    with pytest.raises(IllmError, match="metadata-only"):
        ReferenceCaptureSession(tmp_path / "bad", {**_identity(), "token_ids": [5, 7]}, "r", SamplingSpec().to_dict())
    _iteration(session, 0)
    session.emit(Kind.REQUEST_FINISHED, {"output_tokens": 1, "output_digest": "2" * 64, "finish_reason": "EOS"})
    session.finalize()
    events_text = (session.root / "events.jsonl").read_text(encoding="utf-8")
    for forbidden in ("private prompt", '"token_id"', '"token_ids"', '"logits"', '"prompt"', '"text"'):
        assert forbidden not in events_text


def test_incomplete_and_overwrite_capture_are_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path / "capture")
    with pytest.raises(IllmError) as raised:
        session.finalize()
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
    with pytest.raises(IllmError) as opened:
        ReferenceCaptureBundle.open(session.root)
    assert opened.value.code is ErrorCode.CAPTURE_INCOMPLETE
    with pytest.raises(IllmError, match="already contains"):
        _session(session.root)


def test_checksum_corruption_is_detected(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    with (session.root / "events.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(IllmError, match="checksum"):
        ReferenceCaptureBundle.open(session.root)


def test_resigned_missing_event_is_not_silently_replayed(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    events_path = session.root / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    del lines[3]
    events_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    _resign(session.root)
    with pytest.raises(IllmError):
        ReferenceCaptureBundle.open(session.root)


def test_resigned_control_state_and_changed_execution_identity_are_rejected(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    with pytest.raises(IllmError, match="different execution"):
        ReferenceCaptureBundle.open(session.root, expected_identity={**_identity(), "weights_digest": "9" * 64})
    records_path = session.root / "state-digests.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[4]["state"]["output_tokens"] = 99
    records[4]["state_digest"] = sha256_json(records[4]["state"])
    records_path.write_bytes(b"".join(canonical_json_bytes(record) + b"\n" for record in records))
    _resign(session.root)
    with pytest.raises(IllmError) as raised:
        ReferenceCaptureBundle.open(session.root)
    assert raised.value.code is ErrorCode.REPLAY_DIVERGED


def test_event_registry_unknown_schema_and_deep_immutability(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    bundle = ReferenceCaptureBundle.open(session.root)
    event = next(event for event in bundle.events if event.kind is Kind.MODEL_OP)
    assert isinstance(event.payload["shape"], tuple)
    exposed = event.to_dict()
    exposed["payload"]["shape"][0] = 99
    assert event.payload["shape"][0] == 1
    raw = event.to_dict()
    raw["schema_version"] = 2
    with pytest.raises(IllmError) as schema:
        ReferenceEvent.from_dict(raw)
    assert schema.value.code is ErrorCode.UNSUPPORTED_SCHEMA
    raw = event.to_dict()
    raw["kind_code"] = 1999
    with pytest.raises(IllmError) as kind:
        ReferenceEvent.from_dict(raw)
    assert kind.value.code is ErrorCode.UNSUPPORTED_SCHEMA
    registry = json.loads((Path(__file__).resolve().parents[2] / "registries" / "events-v1.json").read_text(encoding="utf-8"))
    assert {entry["code"]: entry["name"] for entry in registry["events"]} == {int(kind): kind.wire_name for kind in Kind}


def test_resigned_boolean_state_cannot_masquerade_as_numeric_count(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    path = session.root / "state-digests.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    record = next(record for record in records if record["state"]["output_tokens"] == 1)
    record["state"]["output_tokens"] = True
    path.write_bytes(b"".join(canonical_json_bytes(record) + b"\n" for record in records))
    _resign(session.root)
    with pytest.raises(IllmError) as raised:
        ReferenceCaptureBundle.open(session.root)
    assert raised.value.code is ErrorCode.REPLAY_DIVERGED


def test_resigned_jsonl_duplicate_key_is_rejected(tmp_path: Path) -> None:
    session = _finished(tmp_path / "capture")
    path = session.root / "events.jsonl"
    text = path.read_text(encoding="utf-8")
    text = text.replace('"seq_no":0', '"seq_no":999,"seq_no":0', 1)
    path.write_bytes(text.encode("utf-8"))
    _resign(session.root)
    with pytest.raises(IllmError, match="duplicate"):
        ReferenceCaptureBundle.open(session.root)

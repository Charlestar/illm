from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle, StreamingCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import EVENT_CATALOG, StreamingEvent, StreamingEventKind as K


def _identity() -> dict[str, Any]:
    return {
        "engine_version": "0.2.0",
        "git_commit": "a" * 40,
        "git_dirty": False,
        "toolchain_digest": "1" * 64,
        "device": "cpu",
        "dtype": "fp32",
        "model_spec_id": "2" * 64,
        "weights_digest": "3" * 64,
        "load_manifest_digest": "4" * 64,
        "tokenizer_digest": "5" * 64,
        "backend": "eager_reference",
    }


def _session(path: Path, request_id: str = "req-1") -> StreamingCaptureSession:
    return StreamingCaptureSession(
        path,
        _identity(),
        request_id,
        {"schema_version": 1, "mode": "greedy", "actual_seed": 7},
        {"schema_version": 1, "kv_cache_mode": "contiguous", "max_context_tokens": 4,
         "planned_kv_bytes": 128},
    )


def _begin(session: StreamingCaptureSession, *, max_new_tokens: int = 2) -> None:
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": max_new_tokens})
    session.emit(K.KV_ALLOCATION_COMMITTED, {
        "allocation_id": "kv-1", "capacity_tokens": 4, "allocated_bytes": 128,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "TOKENIZED", "to_phase": "PREFILLING"})


def _write(session: StreamingCaptureSession, *, write_id: str, start: int, count: int, valid: int) -> None:
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": write_id, "allocation_id": "kv-1", "start_token": start, "token_count": count,
        "lease_id": f"lease-{write_id}", "fence_id": f"fence-{write_id}",
    })
    session.emit(K.KV_WRITE_COMMITTED, {
        "write_id": write_id, "allocation_id": "kv-1", "lease_id": f"lease-{write_id}",
        "fence_id": f"fence-{write_id}", "fence_completed": True, "kv_valid_token_count": valid,
    })


def _complete_capture(path: Path) -> dict[str, Any]:
    session = _session(path)
    _begin(session)
    session.emit(K.MODEL_FORWARD_STARTED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    _write(session, write_id="prefill", start=0, count=2, valid=2)
    session.emit(K.MODEL_FORWARD_COMPLETED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.REQUEST_FIRST_TOKEN, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 2,
        "server_ready_ns": 30, "token_hmac": "opaque-token-0",
    })
    session.emit(K.REQUEST_OUTPUT_CHUNK, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 2,
        "server_ready_ns": 30, "token_hmac": "opaque-token-0",
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "PREFILLING", "to_phase": "DECODING"})
    session.emit(K.MODEL_FORWARD, {
        "phase": "decode", "query_tokens": 1, "context_tokens": 3, "output_position": 1,
        "duration_ns": 100,
    })
    _write(session, write_id="decode-0", start=2, count=1, valid=3)
    session.emit(K.REQUEST_OUTPUT_CHUNK, {
        "output_index": 1, "committed_output_count": 2, "kv_valid_token_count": 3,
        "server_ready_ns": 40, "token_hmac": "opaque-token-1",
    })
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
        "committed_output_count": 2, "kv_valid_token_count": 3,
    })
    session.emit(K.KV_PAGE_RELEASED, {
        "allocation_id": "kv-1", "released_bytes": 128, "fence_completed": True,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "DECODING", "to_phase": "FINISHED"})
    session.emit(K.REQUEST_COMPLETED, {
        "finish_reason": "MAX_NEW_TOKENS", "committed_output_count": 2, "kv_valid_token_count": 3,
    })
    return session.finalize()


def _resign(path: Path) -> None:
    facts = {"events.jsonl": sha256_file(path / "events.jsonl")}
    (path / "checksums.json").write_bytes(canonical_json_bytes(facts) + b"\n")
    manifest = json.loads((path / "capture-manifest.json").read_text(encoding="utf-8"))
    manifest["event_count"] = len((path / "events.jsonl").read_text(encoding="utf-8").splitlines())
    manifest["files"] = {**facts, "checksums.json": sha256_file(path / "checksums.json")}
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    (path / "capture-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    (path / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")


def test_complete_r0_capture_story_and_chrome_trace_are_metadata_only(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    manifest = _complete_capture(root)
    assert manifest["schema_version"] == 2
    assert manifest["event_envelope_version"] == 1
    assert manifest["event_catalog"] == EVENT_CATALOG
    assert manifest["capabilities"] == ["R0"]
    assert manifest["logical_terminal_outcome"] == "FINISHED"
    assert manifest["terminal_outcome"] == "FINISHED"
    assert not (root / "INCOMPLETE").exists()

    bundle = StreamingCaptureBundle.open(root)
    story = bundle.request_story("req-1")
    assert story["capability"] == "R0"
    assert [item["payload"]["output_index"] for item in story["tokens"]] == [0, 1]
    assert story["kv"]["final_valid_token_count"] == 3
    assert story["kv"]["final_active_allocation_bytes"] == 0
    assert story["kv"]["final_open_write_count"] == 0
    assert story["missing_or_dropped"] == []
    trace = bundle.chrome_trace()
    assert trace["displayTimeUnit"] == "ns"
    assert trace["metadata"]["derived"] is True
    assert len(trace["traceEvents"]) == len(bundle.events)
    encoded = canonical_json_bytes({"manifest": bundle.manifest, "story": story, "trace": trace}).lower()
    for forbidden in (b'"prompt"', b'"text"', b'"token_id"', b'"logits"', b'"key"'):
        assert forbidden not in encoded
    assert bundle.events[2].kind.wire_name == "kv.allocation_committed"
    assert any(item["name"] == "request.output_chunk" for item in trace["traceEvents"])
    summary = next(item for item in trace["traceEvents"] if item["ph"] == "X")
    assert summary["ts"] + summary["dur"] == pytest.approx(
        (summary["args"]["source_monotonic_ns"] - bundle.events[0].monotonic_ns) / 1_000.0,
    )


def test_events_are_frozen_and_reject_private_or_nonfinite_payloads() -> None:
    payload: dict[str, Any] = {"request_id": "req", "started_ns": 1, "ended_ns": 2, "input_tokens": 1,
                               "diagnostic": {"shape": [1, 2]}}
    event = StreamingEvent("producer", 0, 1, K.REQUEST_TOKENIZED, payload)
    payload["diagnostic"]["shape"].append(3)
    assert event.payload["diagnostic"]["shape"] == (1, 2)
    with pytest.raises(TypeError):
        event.payload["input_tokens"] = 2  # type: ignore[index]
    for private in ({"prompt": "secret"}, {"nested": {"token_ids": [1]}}):
        with pytest.raises(IllmError, match="metadata-only"):
            StreamingEvent("producer", 0, 1, K.REQUEST_TOKENIZED, {**payload, **private})
    with pytest.raises(IllmError, match="finite JSON"):
        StreamingEvent("producer", 0, 1, K.REQUEST_TOKENIZED, {**payload, "ratio": float("nan")})
    sensitive_fields: tuple[dict[str, Any], ...] = (
        {"output": "secret"},
        {"nested": {"hmac_key": "secret"}},
    )
    for sensitive in sensitive_fields:
        with pytest.raises(IllmError, match="metadata-only"):
            StreamingEvent("producer", 0, 1, K.REQUEST_TOKENIZED, {**payload, **sensitive})


def test_event_catalog_uses_one_namespace_separator_and_frozen_leaf_names() -> None:
    assert {kind: kind.wire_name for kind in K} == {
        K.REQUEST_TOKENIZED: "request.tokenized",
        K.REQUEST_ACCEPTED: "request.accepted",
        K.REQUEST_PHASE_CHANGED: "request.phase_changed",
        K.REQUEST_FIRST_TOKEN: "request.first_token",
        K.REQUEST_OUTPUT_CHUNK: "request.output_chunk",
        K.REQUEST_COMPLETED: "request.completed",
        K.REQUEST_CANCEL_REQUESTED: "request.cancel_requested",
        K.REQUEST_CANCELLED: "request.cancelled",
        K.REQUEST_FAILED: "request.failed",
        K.REQUEST_TERMINAL_DECIDED: "request.terminal_decided",
        K.MODEL_FORWARD_STARTED: "model.forward_started",
        K.MODEL_FORWARD_COMPLETED: "model.forward_completed",
        K.MODEL_FORWARD: "model.forward",
        K.KV_ALLOCATION_COMMITTED: "kv.allocation_committed",
        K.KV_WRITE_STARTED: "kv.write_started",
        K.KV_WRITE_COMMITTED: "kv.write_committed",
        K.KV_WRITE_ABORTED: "kv.write_aborted",
        K.KV_PAGE_RELEASED: "kv.page_released",
        K.CLOCK_SNAPSHOT: "clock_snapshot",
    }


def test_cancel_after_launch_requires_fence_abort_before_release(tmp_path: Path) -> None:
    root = tmp_path / "cancel"
    session = _session(root)
    _begin(session)
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "start_token": 0, "token_count": 2,
        "lease_id": "lease-1", "fence_id": "fence-1",
    })
    session.emit(K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": "cancel-1"})
    with pytest.raises(IllmError, match="duplicate cancel request"):
        session.emit(K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": "cancel-2"})
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "CANCELLED", "cancellation_id": "cancel-1",
        "committed_output_count": 0, "kv_valid_token_count": 0,
    })
    with pytest.raises(IllmError, match="completed fence"):
        session.emit(K.KV_WRITE_ABORTED, {
            "write_id": "prefill", "allocation_id": "kv-1", "lease_id": "lease-1",
            "fence_id": "fence-1", "fence_completed": False,
        })
    with pytest.raises(IllmError, match="before its write fence"):
        session.emit(K.KV_PAGE_RELEASED, {
            "allocation_id": "kv-1", "released_bytes": 128, "fence_completed": True,
        })
    session.emit(K.KV_WRITE_ABORTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "lease_id": "lease-1",
        "fence_id": "fence-1", "fence_completed": True,
    })
    with pytest.raises(IllmError, match="terminal-ready.*release"):
        session.emit(K.REQUEST_CANCELLED, {
            "cancellation_id": "cancel-1", "committed_output_count": 0, "kv_valid_token_count": 0,
            "server_terminal_ready_ns": 40,
        })
    session.emit(K.KV_PAGE_RELEASED, {
        "allocation_id": "kv-1", "released_bytes": 128, "fence_completed": True,
    })
    session.emit(K.REQUEST_CANCELLED, {
        "cancellation_id": "cancel-1", "committed_output_count": 0, "kv_valid_token_count": 0,
        "server_terminal_ready_ns": 41,
    })
    manifest = session.finalize()
    assert manifest["logical_terminal_outcome"] == "CANCELLED"
    assert manifest["terminal_outcome"] == "CANCELLED"
    story = StreamingCaptureBundle.open(root).request_story("req-1")
    assert story["logical_terminal_outcome"] == "CANCELLED"
    assert story["terminal_outcome"] == "CANCELLED"


def test_finalize_rejects_unclosed_write_and_preserves_incomplete_marker(tmp_path: Path) -> None:
    root = tmp_path / "dangling"
    session = _session(root)
    _begin(session)
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "start_token": 0, "token_count": 2,
        "lease_id": "lease-1", "fence_id": "fence-1",
    })
    with pytest.raises(IllmError, match="unresolved KV writes") as raised:
        session.finalize()
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
    assert (root / "INCOMPLETE").is_file() and not (root / "COMPLETE").exists()


def test_none_kv_capture_has_complete_forward_and_zero_resources(tmp_path: Path) -> None:
    root = tmp_path / "none-kv"
    session = StreamingCaptureSession(
        path=root,
        identity=_identity(),
        request_id="req-none",
        sampling_spec={"schema_version": 1, "mode": "greedy", "actual_seed": 7},
        memory_plan={"schema_version": 1, "kv_cache_mode": "none", "planned_kv_bytes": 0},
    )
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "TOKENIZED", "to_phase": "PREFILLING"})
    session.emit(K.MODEL_FORWARD_STARTED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.MODEL_FORWARD_COMPLETED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.REQUEST_FIRST_TOKEN, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 0,
        "server_ready_ns": 30, "token_hmac": "opaque-token-0",
    })
    session.emit(K.REQUEST_OUTPUT_CHUNK, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 0,
        "server_ready_ns": 30, "token_hmac": "opaque-token-0",
    })
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
        "committed_output_count": 1, "kv_valid_token_count": 0,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "PREFILLING", "to_phase": "FINISHED"})
    session.emit(K.REQUEST_COMPLETED, {
        "finish_reason": "MAX_NEW_TOKENS", "committed_output_count": 1,
        "kv_valid_token_count": 0, "server_terminal_ready_ns": 40,
    })
    session.finalize()

    bundle = StreamingCaptureBundle.open(root)
    story = bundle.request_story("req-none")
    assert story["kv"]["allocations"] == []
    assert story["kv"]["writes"] == []
    assert story["kv"]["final_active_allocation_bytes"] == 0


def test_mark_incomplete_persists_non_replayable_reason(tmp_path: Path) -> None:
    root = tmp_path / "incomplete"
    session = _session(root)
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    manifest = session.mark_incomplete("required sink write failed")
    assert manifest["finalized"] is False and manifest["replayable"] is False
    assert manifest["non_replayable_reasons"] == ["required sink write failed"]
    with pytest.raises(IllmError, match="incomplete/non-replayable") as raised:
        StreamingCaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE


def test_incomplete_loss_updates_are_monotone_and_cannot_reopen_event_stream(tmp_path: Path) -> None:
    root = tmp_path / "incomplete-loss"
    session = _session(root)
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    first = session.mark_incomplete("required sink write failed", critical_event_drop_count=1)
    fixed_prefix = (root / "events.jsonl").read_bytes()
    final = session.mark_incomplete("later critical facts not captured", critical_event_drop_count=7)
    assert first["critical_event_drop_count"] == 1
    assert final["critical_event_drop_count"] == 7
    assert len(final["non_replayable_reasons"]) == 2
    assert final["replayable"] is final["finalized"] is False
    assert (root / "events.jsonl").read_bytes() == fixed_prefix
    for invalid_count in (0, 6, -1, True):
        with pytest.raises(IllmError):
            session.mark_incomplete("cannot erase missing facts", critical_event_drop_count=invalid_count)
    with pytest.raises(IllmError):
        session.finalize()
    with pytest.raises(IllmError):
        session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    with pytest.raises(IllmError) as raised:
        StreamingCaptureBundle.open(root)
    assert raised.value.details is not None
    assert raised.value.details["reported_critical_event_drop_count"] == 7


def test_complete_capture_cannot_be_downgraded_or_rewritten(tmp_path: Path) -> None:
    root = tmp_path / "immutable-complete"
    _complete_capture(root)
    # A new writer also refuses to reuse the nonempty immutable directory.
    with pytest.raises(IllmError):
        _session(root)


def test_unknown_fence_preserves_logical_decision_without_fake_terminal_ready(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unknown-fence"
    session = _session(root)
    _begin(session)
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "start_token": 0, "token_count": 2,
        "lease_id": "lease-1", "fence_id": "fence-1",
    })
    session.emit(K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": "cancel-unknown"})
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "CANCELLED", "cancellation_id": "cancel-unknown",
        "committed_output_count": 0, "kv_valid_token_count": 0,
    })

    manifest = session.mark_incomplete("FENCE_OR_RESOURCE_STATE_UNKNOWN")
    assert manifest["finalized"] is False
    assert manifest["replayable"] is False
    assert manifest["logical_terminal_outcome"] == "CANCELLED"
    assert manifest["terminal_outcome"] is None
    assert manifest["non_replayable_reasons"] == ["FENCE_OR_RESOURCE_STATE_UNKNOWN"]
    assert (root / "INCOMPLETE").is_file() and not (root / "COMPLETE").exists()
    with pytest.raises(IllmError, match="incomplete/non-replayable") as raised:
        StreamingCaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE


def test_logical_terminal_decision_is_unique_and_ready_must_agree(tmp_path: Path) -> None:
    root = tmp_path / "logical-terminal"
    session = StreamingCaptureSession(
        path=root,
        identity=_identity(),
        request_id="req-logical",
        sampling_spec={"schema_version": 1, "mode": "greedy", "actual_seed": 7},
        memory_plan={"schema_version": 1, "kv_cache_mode": "none", "planned_kv_bytes": 0},
    )
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 1})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 1, "max_new_tokens": 1})
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
        "committed_output_count": 0, "kv_valid_token_count": 0,
    })
    with pytest.raises(IllmError, match="exactly one logical terminal decision"):
        session.emit(K.REQUEST_TERMINAL_DECIDED, {
            "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
            "committed_output_count": 0, "kv_valid_token_count": 0,
        })
    with pytest.raises(IllmError, match="production cannot continue"):
        session.emit(K.REQUEST_OUTPUT_CHUNK, {
            "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 0,
            "server_ready_ns": 3,
        })
    with pytest.raises(IllmError, match="cancel request cannot occur"):
        session.emit(K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": "too-late"})
    with pytest.raises(IllmError, match="disagrees with its logical terminal decision"):
        session.emit(K.REQUEST_FAILED, {
            "error_code": "INTERNAL", "committed_output_count": 0, "kv_valid_token_count": 0,
            "server_terminal_ready_ns": 3,
        })
    with pytest.raises(IllmError, match="unsupported metadata"):
        StreamingEvent("producer", 0, 1, K.REQUEST_TERMINAL_DECIDED, {
            "request_id": "req-logical", "outcome": "FINISHED",
            "committed_output_count": 0, "kv_valid_token_count": 0,
            "finish_reason": "MAX_NEW_TOKENS",
        })


def test_cancel_decision_suppresses_new_work_but_allows_inflight_cleanup(tmp_path: Path) -> None:
    root = tmp_path / "cancel-decision"
    session = _session(root)
    _begin(session)
    session.emit(K.MODEL_FORWARD_STARTED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "start_token": 0, "token_count": 2,
        "lease_id": "lease-1", "fence_id": "fence-1",
    })
    session.emit(K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": "cancel-1"})
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "CANCELLED", "cancellation_id": "cancel-1",
        "committed_output_count": 0, "kv_valid_token_count": 0,
    })

    for kind, payload in (
        (K.MODEL_FORWARD_STARTED, {
            "phase": "decode", "query_tokens": 1, "context_tokens": 3, "output_position": 1,
        }),
        (K.KV_WRITE_STARTED, {
            "write_id": "late", "allocation_id": "kv-1", "start_token": 0, "token_count": 1,
            "lease_id": "lease-late", "fence_id": "fence-late",
        }),
    ):
        with pytest.raises(IllmError, match="production cannot continue"):
            session.emit(kind, payload)
    with pytest.raises(IllmError, match="KV validity cannot advance"):
        session.emit(K.KV_WRITE_COMMITTED, {
            "write_id": "prefill", "allocation_id": "kv-1", "lease_id": "lease-1",
            "fence_id": "fence-1", "fence_completed": True, "kv_valid_token_count": 2,
        })

    session.emit(K.MODEL_FORWARD_COMPLETED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.MODEL_FORWARD, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
        "duration_ns": 10,
    })
    session.emit(K.KV_WRITE_ABORTED, {
        "write_id": "prefill", "allocation_id": "kv-1", "lease_id": "lease-1",
        "fence_id": "fence-1", "fence_completed": True,
    })
    session.emit(K.KV_PAGE_RELEASED, {
        "allocation_id": "kv-1", "released_bytes": 128, "fence_completed": True,
    })
    session.emit(K.REQUEST_CANCELLED, {
        "cancellation_id": "cancel-1", "committed_output_count": 0,
        "kv_valid_token_count": 0, "server_terminal_ready_ns": 20,
    })
    assert session.finalize()["terminal_outcome"] == "CANCELLED"


def test_reader_fails_closed_on_checksum_seq_gap_and_double_terminal(tmp_path: Path) -> None:
    checksum = tmp_path / "checksum"
    _complete_capture(checksum)
    with (checksum / "events.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(IllmError, match="checksum"):
        StreamingCaptureBundle.open(checksum)

    sequence = tmp_path / "sequence"
    _complete_capture(sequence)
    lines = (sequence / "events.jsonl").read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[2])
    value["producer_seq_no"] = 99
    lines[2] = canonical_json_bytes(value).decode("utf-8")
    (sequence / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _resign(sequence)
    with pytest.raises(IllmError, match="sequence"):
        StreamingCaptureBundle.open(sequence)

    terminal = tmp_path / "terminal"
    _complete_capture(terminal)
    values = [json.loads(line) for line in (terminal / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    last = values[-1]
    duplicate = dict(last)
    duplicate["producer_seq_no"] = len(values)
    duplicate["monotonic_ns"] = last["monotonic_ns"] + 1
    values.append(duplicate)
    (terminal / "events.jsonl").write_bytes(b"".join(canonical_json_bytes(item) + b"\n" for item in values))
    _resign(terminal)
    with pytest.raises(IllmError, match="after terminal|exactly one terminal"):
        StreamingCaptureBundle.open(terminal)


def test_wrong_request_story_and_unsupported_registry_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    _complete_capture(root)
    bundle = StreamingCaptureBundle.open(root)
    with pytest.raises(IllmError) as raised:
        bundle.request_story("other-request")
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    value = bundle.events[0].to_dict()
    value["event_catalog"] = "unknown-catalog"
    with pytest.raises(IllmError) as unsupported:
        StreamingEvent.from_dict(value)
    assert unsupported.value.code is ErrorCode.UNSUPPORTED_SCHEMA

    legacy = tmp_path / "legacy-family"
    legacy.mkdir()
    (legacy / "capture-manifest.json").write_text('{"schema_version":1}\n', encoding="utf-8")
    (legacy / "COMPLETE").write_text("legacy\n", encoding="ascii")
    with pytest.raises(IllmError) as wrong_family:
        StreamingCaptureBundle.open(legacy)
    assert wrong_family.value.code is ErrorCode.UNSUPPORTED_SCHEMA


def test_streaming_schemas_are_valid_json_and_name_exact_versions() -> None:
    root = Path(__file__).resolve().parents[2] / "schemas"
    event = json.loads((root / "streaming-event-v1.schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "streaming-capture-manifest-v2.schema.json").read_text(encoding="utf-8"))
    assert event["properties"]["event_catalog"]["const"] == EVENT_CATALOG
    assert event["properties"]["event_envelope_version"]["const"] == 1
    assert "request.output_chunk" in event["properties"]["kind"]["enum"]
    assert "request.terminal_decided" in event["properties"]["kind"]["enum"]
    assert "kv.write_committed" in event["properties"]["kind"]["enum"]
    assert "request.output.chunk" not in event["properties"]["kind"]["enum"]
    assert manifest["properties"]["schema_version"]["const"] == 2
    assert manifest["properties"]["capabilities"]["const"] == ["R0"]
    assert "logical_terminal_outcome" in manifest["required"]
    assert manifest["properties"]["logical_terminal_outcome"]["enum"] == [
        "FINISHED", "CANCELLED", "FAILED", None,
    ]

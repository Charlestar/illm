from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

import illm.capture.policy_streaming as policy_capture
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.policy_streaming import (
    PolicyStreamingCaptureBundle,
    PolicyStreamingCaptureSession,
)
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.policy_events import POLICY_EVENT_CATALOG, PolicyStreamingEvent
from illm.domain.streaming_events import StreamingEventKind as K
from illm.numerical_policy import candidate_policy
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.sampling import SamplingSpec


def _metadata(*, kv_mode: str = "none") -> dict[str, Any]:
    policy = candidate_policy("fp32").to_dict()
    environment = {
        "backend": "cpu",
        "python_version": "3.12.0",
        "platform": "fixture-platform",
        "machine": "fixture-machine",
        "torch_version": "fixture-torch",
        "cuda_arithmetic": "not_applicable",
        "cuda_execution_authorized": False,
    }
    binding = {
        "schema_version": 1,
        "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": "a" * 64,
        "environment": environment,
        "environment_digest": sha256_json(environment),
        "toolchain": {"binding": "unbound-development", "manifest_digest": None},
        "protocol_sha256": PROTOCOL_SHA256,
    }
    weights_bytes = 1024
    kv_bytes = 512 if kv_mode == "contiguous" else 0
    sidecar_bytes = 1024
    prefix_scratch_bytes = 256
    subtotal = (
        weights_bytes + kv_bytes + 256 + 128 + 64 + 32
        + sidecar_bytes + prefix_scratch_bytes
    )
    headroom_bytes = (((subtotal + 9) // 10 + 255) // 256) * 256
    plan = {
        "schema_version": 2,
        "model_spec_id": "2" * 64,
        "weights_digest": "3" * 64,
        "weight_manifest_digest": "4" * 64,
        "runtime_dtype": "fp32",
        "device": "cpu",
        "kv_mode": kv_mode,
        "kv_layout_id": (
            "qwen2-contiguous-layer-major-v1"
            if kv_mode == "contiguous"
            else "none-reference-full-context-v1"
        ),
        "capacity_tokens": 4,
        "memory_budget_bytes": 4096,
        "weights_bytes": weights_bytes,
        "kv_bytes": kv_bytes,
        "activation_bytes": 256,
        "workspace_bytes": 128,
        "io_bytes": 64,
        "telemetry_bytes": 32,
        "headroom_bytes": headroom_bytes,
        "required_bytes": subtotal + headroom_bytes,
        "numerical_policy": policy,
        "numerical_policy_digest": sha256_json(policy),
        "compute_dtype": "fp32",
        "sidecar_bytes": sidecar_bytes,
        "prefix_scratch_bytes": prefix_scratch_bytes,
        "execution_binding": binding,
        "execution_binding_digest": sha256_json(binding),
        "evidence_kind": "static_conservative_estimate",
    }
    identity = {
        "engine_version": "0.2.0",
        "git_commit": "b" * 40,
        "git_dirty": False,
        "toolchain_digest": sha256_json({"binding": "unbound-development"}),
        "toolchain_bound": False,
        "device": "cpu",
        "dtype": "fp32",
        "model_spec_id": plan["model_spec_id"],
        "weights_digest": plan["weights_digest"],
        "load_manifest_digest": plan["weight_manifest_digest"],
        "tokenizer_digest": "5" * 64,
        "kv_mode": kv_mode,
        "memory_plan_evidence_kind": "static_conservative_estimate",
    }
    return {
        "identity": identity,
        "sampling_spec": SamplingSpec(actual_seed=7).to_dict(),
        "memory_plan": plan,
        "numerical_policy": policy,
        "execution_binding": binding,
        "observation_profile": "request",
    }


def _session(
    path: Path, *, request_id: str = "req-policy", kv_mode: str = "none",
) -> PolicyStreamingCaptureSession:
    metadata = _metadata(kv_mode=kv_mode)
    return PolicyStreamingCaptureSession(
        path,
        metadata["identity"],
        request_id,
        metadata["sampling_spec"],
        metadata["memory_plan"],
        numerical_policy=metadata["numerical_policy"],
        execution_binding=metadata["execution_binding"],
        observation_profile=metadata["observation_profile"],
    )


def _complete(path: Path, *, include_forward_summary: bool = False) -> dict[str, Any]:
    session = _session(path)
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    session.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "TOKENIZED", "to_phase": "PREFILLING",
    })
    session.emit(K.MODEL_FORWARD_STARTED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    session.emit(K.MODEL_FORWARD_COMPLETED, {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0,
    })
    if include_forward_summary:
        session.emit(K.MODEL_FORWARD, {
            "phase": "prefill", "query_tokens": 2, "context_tokens": 2,
            "output_position": 0, "duration_ns": 1, "evidence_kind": "measured",
        })
    session.emit(K.REQUEST_FIRST_TOKEN, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 0,
        "server_ready_ns": 30, "token_hmac": "a" * 64,
    })
    session.emit(K.REQUEST_OUTPUT_CHUNK, {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 0,
        "server_ready_ns": 30, "token_hmac": "a" * 64,
    })
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
        "committed_output_count": 1, "kv_valid_token_count": 0,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "PREFILLING", "to_phase": "FINISHED",
    })
    session.emit(K.REQUEST_COMPLETED, {
        "finish_reason": "MAX_NEW_TOKENS", "committed_output_count": 1,
        "kv_valid_token_count": 0, "server_terminal_ready_ns": 40,
    })
    return session.finalize()


def _complete_contiguous(path: Path) -> dict[str, Any]:
    session = _session(path, kv_mode="contiguous")
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    session.emit(K.KV_ALLOCATION_COMMITTED, {
        "allocation_id": "kv-0", "capacity_tokens": 4, "allocated_bytes": 512,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "TOKENIZED", "to_phase": "PREFILLING",
    })
    forward = {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2,
        "output_position": 0,
    }
    session.emit(K.MODEL_FORWARD_STARTED, forward)
    session.emit(K.KV_WRITE_STARTED, {
        "write_id": "write-0", "allocation_id": "kv-0", "start_token": 0,
        "token_count": 2, "lease_id": "lease-0", "fence_id": "fence-0",
    })
    session.emit(K.MODEL_FORWARD_COMPLETED, forward)
    session.emit(K.KV_WRITE_COMMITTED, {
        "write_id": "write-0", "allocation_id": "kv-0", "lease_id": "lease-0",
        "fence_id": "fence-0", "fence_completed": True, "kv_valid_token_count": 2,
    })
    token = {
        "output_index": 0, "committed_output_count": 1, "kv_valid_token_count": 2,
        "server_ready_ns": 30, "token_hmac": "b" * 64,
    }
    session.emit(K.REQUEST_FIRST_TOKEN, token)
    session.emit(K.REQUEST_OUTPUT_CHUNK, token)
    session.emit(K.REQUEST_TERMINAL_DECIDED, {
        "outcome": "FINISHED", "reason": "MAX_NEW_TOKENS",
        "committed_output_count": 1, "kv_valid_token_count": 2,
    })
    session.emit(K.KV_PAGE_RELEASED, {
        "allocation_id": "kv-0", "released_bytes": 512, "fence_completed": True,
    })
    session.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "PREFILLING", "to_phase": "FINISHED",
    })
    session.emit(K.REQUEST_COMPLETED, {
        "finish_reason": "MAX_NEW_TOKENS", "committed_output_count": 1,
        "kv_valid_token_count": 2, "server_terminal_ready_ns": 40,
    })
    return session.finalize()


def _rewrite_manifest(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    manifest = json.loads((path / "capture-manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["manifest_digest"] = sha256_json({
        key: value for key, value in manifest.items() if key != "manifest_digest"
    })
    (path / "capture-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    (path / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")


def _resign_facts(path: Path) -> None:
    facts = {"events.jsonl": sha256_file(path / "events.jsonl")}
    (path / "checksums.json").write_bytes(canonical_json_bytes(facts) + b"\n")

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["files"] = {
            **facts,
            "checksums.json": sha256_file(path / "checksums.json"),
        }

    _rewrite_manifest(path, mutate)


def test_schema3_roundtrip_story_trace_and_old_reader_isolation(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    manifest = _complete(root)
    assert manifest["schema_version"] == 3
    assert manifest["event_catalog"] == POLICY_EVENT_CATALOG
    assert manifest["protocol_sha256"] == PROTOCOL_SHA256
    assert manifest["observation_profile"] == "request"
    assert not (root / "INCOMPLETE").exists()

    bundle = PolicyStreamingCaptureBundle.open(root)
    assert all(isinstance(event, PolicyStreamingEvent) for event in bundle.events)
    assert all(event.event_catalog == POLICY_EVENT_CATALOG for event in bundle.events)
    story = bundle.request_story("req-policy")
    trace = bundle.chrome_trace()
    assert story["schema_version"] == 2
    assert trace["metadata"]["schema_version"] == 2
    assert story["capture_identity"] == trace["metadata"]["capture_identity"]
    assert story["capture_identity"] == {
        "numerical_policy_digest": manifest["numerical_policy_digest"],
        "memory_plan_digest": manifest["memory_plan_digest"],
        "execution_binding_digest": manifest["execution_binding_digest"],
        "implementation_source_digest": "a" * 64,
        "environment_digest": manifest["execution_binding"]["environment_digest"],
        "protocol_sha256": PROTOCOL_SHA256,
        "observation_profile": "request",
    }
    assert story["logical_terminal_outcome"] == "FINISHED"
    assert story["kv"]["final_active_allocation_bytes"] == 0
    encoded = canonical_json_bytes({"manifest": bundle.manifest, "story": story, "trace": trace}).lower()
    for forbidden in (b'"prompt"', b'"text"', b'"token_id"', b'"logits"', b'"tensor"', b'"key"'):
        assert forbidden not in encoded
    assert b'"token_hmac"' in encoded
    with pytest.raises(IllmError) as raised:
        StreamingCaptureBundle.open(root)
    assert raised.value.code is ErrorCode.UNSUPPORTED_SCHEMA


def test_incomplete_loss_is_monotone_and_freezes_event_prefix(tmp_path: Path) -> None:
    root = tmp_path / "incomplete"
    session = _session(root)
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    first = session.mark_incomplete("sink unavailable", critical_event_drop_count=1)
    fixed_prefix = (root / "events.jsonl").read_bytes()
    second = session.mark_incomplete("later facts absent", critical_event_drop_count=3)
    assert first["critical_event_drop_count"] == 1
    assert second["critical_event_drop_count"] == 3
    assert second["non_replayable_reasons"] == ["sink unavailable", "later facts absent"]
    assert (root / "events.jsonl").read_bytes() == fixed_prefix
    for invalid in (2, 0, -1, True):
        with pytest.raises(IllmError):
            session.mark_incomplete("cannot regress", critical_event_drop_count=invalid)
    with pytest.raises(IllmError):
        session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    with pytest.raises(IllmError):
        session.finalize()
    with pytest.raises(IllmError) as raised:
        PolicyStreamingCaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
    assert raised.value.details is not None
    assert raised.value.details["reported_critical_event_drop_count"] == 3


def test_empty_directory_is_allowed_but_foreign_or_extra_artifacts_are_not(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    session = _session(empty)
    assert (empty / "INCOMPLETE").is_file()
    session.mark_incomplete("test stop")

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "owned.txt"
    sentinel.write_text("owner-data", encoding="utf-8")
    with pytest.raises(IllmError):
        _session(foreign)
    assert sentinel.read_text(encoding="utf-8") == "owner-data"

    complete = tmp_path / "complete"
    _complete(complete)
    extra = complete / "extra.txt"
    extra.write_text("not evidence", encoding="utf-8")
    with pytest.raises(IllmError, match="inventory"):
        PolicyStreamingCaptureBundle.open(complete)
    assert extra.read_text(encoding="utf-8") == "not evidence"


def test_linklike_output_or_artifact_is_rejected_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "linklike-root"
    original = policy_capture._is_linklike
    monkeypatch.setattr(
        policy_capture,
        "_is_linklike",
        lambda path: Path(path) == root.absolute() or original(Path(path)),
    )
    with pytest.raises(IllmError, match="symbolic link or reparse"):
        _session(root)
    assert not root.exists()

    monkeypatch.setattr(policy_capture, "_is_linklike", original)
    capture = tmp_path / "linked-artifact"
    _complete(capture)
    events = capture / "events.jsonl"
    preserved = events.read_bytes()
    monkeypatch.setattr(
        policy_capture,
        "_is_linklike",
        lambda path: Path(path) == events.absolute() or original(Path(path)),
    )
    with pytest.raises(IllmError, match="link"):
        PolicyStreamingCaptureBundle.open(capture)
    assert events.read_bytes() == preserved


def test_private_tensor_and_key_metadata_fail_but_sampling_fields_and_hmac_work(
    tmp_path: Path,
) -> None:
    private_values: tuple[dict[str, Any], ...] = (
        {"tensor": [1.0]},
        {"nested": {"tensors": []}},
        {"key": "secret"},
    )
    for private in private_values:
        session = _session(tmp_path / f"private-{len(list(tmp_path.iterdir()))}")
        with pytest.raises(IllmError, match="metadata-only"):
            session.emit(K.REQUEST_TOKENIZED, {
                "started_ns": 1, "ended_ns": 2, "input_tokens": 1, **private,
            })
        session.mark_incomplete("rejected private metadata")
    bundle_root = tmp_path / "valid"
    _complete(bundle_root)
    manifest = PolicyStreamingCaptureBundle.open(bundle_root).manifest
    assert manifest["sampling_spec"]["rng_algorithm"] == "philox4x32-10"
    assert manifest["sampling_spec"]["actual_seed"] == 7


@pytest.mark.parametrize("mode", ["duplicate", "nonfinite", "bool-count"])
def test_raw_json_and_integer_contract_fail_closed(tmp_path: Path, mode: str) -> None:
    root = tmp_path / mode
    _complete(root)
    manifest_path = root / "capture-manifest.json"
    if mode == "duplicate":
        original = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text('{"schema_version":3,' + original[1:], encoding="utf-8")
    elif mode == "nonfinite":
        original = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(original.replace('"event_count":10', '"event_count":NaN'),
                                 encoding="utf-8")
        assert "NaN" in manifest_path.read_text(encoding="utf-8")
    else:
        _rewrite_manifest(root, lambda manifest: manifest.__setitem__("event_count", True))
    with pytest.raises(IllmError):
        PolicyStreamingCaptureBundle.open(root)


def test_truncation_is_rejected_even_after_fact_hashes_are_resigned(tmp_path: Path) -> None:
    root = tmp_path / "truncated"
    _complete(root)
    events = root / "events.jsonl"
    encoded = events.read_bytes()
    assert encoded.endswith(b"\n")
    events.write_bytes(encoded[:-1])
    _resign_facts(root)
    with pytest.raises(IllmError, match="truncated"):
        PolicyStreamingCaptureBundle.open(root)


@pytest.mark.parametrize("non_object", [None, [], 7, True])
def test_resigned_non_object_event_lines_fail_as_capture_errors(
    tmp_path: Path, non_object: object,
) -> None:
    root = tmp_path / f"event-{type(non_object).__name__}"
    _complete(root)
    events_path = root / "events.jsonl"
    lines = events_path.read_bytes().splitlines(keepends=True)
    lines[0] = canonical_json_bytes(non_object) + b"\n"
    events_path.write_bytes(b"".join(lines))
    _resign_facts(root)
    with pytest.raises(IllmError, match="entries must be objects") as raised:
        PolicyStreamingCaptureBundle.open(root)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_cross_field_tamper_is_rejected_after_all_changed_digests_are_updated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cross-field"
    _complete(root)

    def mutate(manifest: dict[str, Any]) -> None:
        binding = manifest["execution_binding"]
        binding["environment"]["machine"] = "tampered-machine"
        binding["environment_digest"] = sha256_json(binding["environment"])
        manifest["execution_binding_digest"] = sha256_json(binding)
        # Deliberately do not rewrite the separately sealed MemoryPlan binding.

    _rewrite_manifest(root, mutate)
    with pytest.raises(IllmError, match="bindings disagree"):
        PolicyStreamingCaptureBundle.open(root)


def test_resigned_events_remain_bound_to_kv_mode_plan_and_capacity(tmp_path: Path) -> None:
    no_kv = tmp_path / "no-kv"
    contiguous = tmp_path / "contiguous"
    _complete(no_kv)
    _complete_contiguous(contiguous)
    no_kv_manifest = json.loads((no_kv / "capture-manifest.json").read_bytes())

    resigned_as_no_kv = tmp_path / "resigned-as-no-kv"
    shutil.copytree(contiguous, resigned_as_no_kv)

    def replace_policy_metadata(manifest: dict[str, Any]) -> None:
        for field in policy_capture.POLICY_METADATA_FIELDS:
            manifest[field] = no_kv_manifest[field]

    _rewrite_manifest(resigned_as_no_kv, replace_policy_metadata)
    with pytest.raises(IllmError, match="no-KV.*KV"):
        PolicyStreamingCaptureBundle.open(resigned_as_no_kv)

    resigned_capacity = tmp_path / "resigned-capacity"
    shutil.copytree(no_kv, resigned_capacity)

    def shrink_capacity(manifest: dict[str, Any]) -> None:
        manifest["memory_plan"]["capacity_tokens"] = 1
        manifest["memory_plan_digest"] = sha256_json(manifest["memory_plan"])

    _rewrite_manifest(resigned_capacity, shrink_capacity)
    with pytest.raises(IllmError, match="exceeds.*MemoryPlan capacity"):
        PolicyStreamingCaptureBundle.open(resigned_capacity)


def test_writer_rejects_forward_and_allocation_facts_that_disagree_with_plan(
    tmp_path: Path,
) -> None:
    no_kv = _session(tmp_path / "bad-forward")
    no_kv.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    no_kv.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    no_kv.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "TOKENIZED", "to_phase": "PREFILLING",
    })
    with pytest.raises(IllmError, match="forward coverage"):
        no_kv.emit(K.MODEL_FORWARD_STARTED, {
            "phase": "prefill", "query_tokens": 1, "context_tokens": 2,
            "output_position": 0,
        })
    no_kv.mark_incomplete("invalid forward rejected")

    contiguous = _session(tmp_path / "bad-allocation", kv_mode="contiguous")
    contiguous.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    contiguous.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    with pytest.raises(IllmError, match="allocation facts disagree"):
        contiguous.emit(K.KV_ALLOCATION_COMMITTED, {
            "allocation_id": "kv-0", "capacity_tokens": 3, "allocated_bytes": 512,
        })
    contiguous.mark_incomplete("invalid allocation rejected")


def test_policy_payload_is_closed_for_writer_and_resigned_reader(tmp_path: Path) -> None:
    writer = _session(tmp_path / "writer-private")
    with pytest.raises(IllmError, match="closed metadata contract"):
        writer.emit(K.REQUEST_TOKENIZED, {
            "started_ns": 1, "ended_ns": 2, "input_tokens": 1,
            "raw_prompt": "TOP-SECRET-PROMPT",
        })
    writer.mark_incomplete("private alias rejected")

    root = tmp_path / "reader-private"
    _complete(root)
    events_path = root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"]["raw_prompt"] = "TOP-SECRET-PROMPT"
    events_path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))
    _resign_facts(root)
    with pytest.raises(IllmError, match="closed metadata contract"):
        PolicyStreamingCaptureBundle.open(root)


@pytest.mark.parametrize(
    ("event_index", "field", "value"),
    [
        (6, "next_output_position", True),
        (6, "rng_draws_consumed", "0"),
        (6, "draw_address", {"raw_prompt": "TOP-SECRET-PROMPT"}),
        (6, "token_hmac", "TOP-SECRET-PROMPT"),
        (5, "token_hmac", "c" * 64),
        (9, "physical_cleanup_complete", False),
        (9, "active_leases", 1),
    ],
)
def test_resigned_optional_payload_values_fail_closed(
    tmp_path: Path, event_index: int, field: str, value: object,
) -> None:
    root = tmp_path / field
    _complete(root)
    events_path = root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[event_index]["payload"][field] = value
    events_path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))
    _resign_facts(root)
    with pytest.raises(IllmError):
        PolicyStreamingCaptureBundle.open(root)


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        (K.MODEL_FORWARD, "evidence_kind", {}),
        (K.MODEL_FORWARD, "evidence_kind", []),
        (K.KV_WRITE_STARTED, "measurement_boundary", {}),
        (K.KV_WRITE_STARTED, "measurement_boundary", []),
    ],
)
def test_writer_rejects_non_string_optional_enum_metadata_as_capture_error(
    tmp_path: Path, kind: K, field: str, value: object,
) -> None:
    session = _session(
        tmp_path / f"writer-{kind.name.lower()}-{type(value).__name__}",
        kv_mode="contiguous" if kind is K.KV_WRITE_STARTED else "none",
    )
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 1, "ended_ns": 2, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    if kind is K.KV_WRITE_STARTED:
        session.emit(K.KV_ALLOCATION_COMMITTED, {
            "allocation_id": "kv-0", "capacity_tokens": 4, "allocated_bytes": 512,
        })
    session.emit(K.REQUEST_PHASE_CHANGED, {
        "from_phase": "TOKENIZED", "to_phase": "PREFILLING",
    })
    forward = {
        "phase": "prefill", "query_tokens": 2, "context_tokens": 2,
        "output_position": 0,
    }
    session.emit(K.MODEL_FORWARD_STARTED, forward)
    if kind is K.MODEL_FORWARD:
        session.emit(K.MODEL_FORWARD_COMPLETED, forward)
        payload = {**forward, "duration_ns": 1, field: value}
    else:
        payload = {
            "write_id": "write-0", "allocation_id": "kv-0", "start_token": 0,
            "token_count": 2, "lease_id": "lease-0", "fence_id": "fence-0",
            field: value,
        }
    with pytest.raises(IllmError, match="unsupported"):
        session.emit(kind, payload)
    session.mark_incomplete("invalid optional enum metadata rejected")


@pytest.mark.parametrize(
    ("capture_kind", "event_index", "field", "value"),
    [
        ("summary", 5, "evidence_kind", {}),
        ("summary", 5, "evidence_kind", []),
        ("contiguous", 5, "measurement_boundary", {}),
        ("contiguous", 5, "measurement_boundary", []),
    ],
)
def test_resigned_reader_rejects_non_string_optional_enum_metadata_as_capture_error(
    tmp_path: Path, capture_kind: str, event_index: int, field: str, value: object,
) -> None:
    root = tmp_path / f"reader-{capture_kind}-{field}-{type(value).__name__}"
    if capture_kind == "summary":
        _complete(root, include_forward_summary=True)
    else:
        _complete_contiguous(root)
    events_path = root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events[event_index]["payload"][field] = value
    events_path.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))
    _resign_facts(root)
    with pytest.raises(IllmError, match="unsupported"):
        PolicyStreamingCaptureBundle.open(root)


def test_atomic_write_only_removes_a_temporary_file_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_uuid = type("FixedUuid", (), {"hex": "fixed"})()
    monkeypatch.setattr(uuid, "uuid4", lambda: fixed_uuid)
    target = tmp_path / "events.jsonl"
    foreign = tmp_path / f".events.jsonl.tmp-{os.getpid()}-fixed"
    foreign.write_bytes(b"foreign-owner-data")
    with pytest.raises(FileExistsError):
        policy_capture._atomic_bytes(target, b"new-data")
    assert foreign.read_bytes() == b"foreign-owner-data"
    assert not target.exists()


def test_manifest_and_event_schemas_are_new_closed_siblings() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_schema = json.loads(
        (root / "schemas" / "streaming-capture-manifest-v3.schema.json").read_text(encoding="utf-8")
    )
    event_schema = json.loads(
        (root / "schemas" / "policy-streaming-event-envelope-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest_schema["additionalProperties"] is False
    assert manifest_schema["properties"]["schema_version"]["const"] == 3
    assert manifest_schema["properties"]["event_catalog"]["const"] == POLICY_EVENT_CATALOG
    assert manifest_schema["properties"]["observation_profile"]["enum"] == [
        "off", "metrics", "request", "ops",
    ]
    assert event_schema["additionalProperties"] is False
    assert event_schema["properties"]["event_catalog"]["const"] == POLICY_EVENT_CATALOG


def test_copied_minimal_reader_runs_isolated_without_tensor_packages(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "illm"
    isolated = tmp_path / "isolated"
    package = isolated / "src" / "illm"
    for relative in (
        "canonical.py",
        "numerical_policy.py",
        "policy_capture_contract.py",
        "capture/streaming.py",
        "capture/policy_streaming.py",
        "domain/errors.py",
        "domain/streaming_events.py",
        "domain/policy_events.py",
        "sampling/reference.py",
    ):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    for package_dir in (package, package / "capture", package / "domain", package / "sampling"):
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

    capture = isolated / "capture"
    _complete(capture)
    script = (
        "import json,sys; from pathlib import Path; "
        f"sys.path.insert(0,{str(isolated / 'src')!r}); "
        "from illm.capture.policy_streaming import PolicyStreamingCaptureBundle as B; "
        f"b=B.open(Path({str(capture)!r})); "
        "print(json.dumps({'events':len(b.events),'torch':'torch' in sys.modules,"
        "'transformers':'transformers' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {"events": 10, "torch": False, "transformers": False}

"""Actual CPU inference, policy-aware R0 capture, and CLI replay boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from illm.capture.dispatch import open_streaming_capture
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle, PolicyStreamingCaptureSession
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.policy_events import POLICY_EVENT_CATALOG
from illm.domain.streaming_events import StreamingEventKind as K
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]


def _request(engine: Engine, capture: Path, *, request_id: str = "policy-story") -> GenerationRequest:
    return GenerationRequest(model_spec_id=engine.model_spec_id, prompt="CAPTURE_SECRET_SENTINEL",
                             max_new_tokens=4, request_id=request_id, capture=capture,
                             sampling=SamplingSpec(mode="greedy", actual_seed=7))


@pytest.mark.parametrize("mode", ["none", "contiguous"])
def test_actual_cpu_capture_is_policy_bound_r0_and_private(tmp_path: Path, mode: str) -> None:
    engine = Engine(project_root=ROOT, kv_mode=mode, numerical_policy=candidate_policy("fp32"))
    capture = tmp_path / "capture"
    try:
        result = engine.generate(_request(engine, capture))
        bundle = PolicyStreamingCaptureBundle.open(capture)
        assert result.capture_manifest_digest == bundle.manifest["manifest_digest"]
        assert bundle.manifest["schema_version"] == 3
        assert bundle.manifest["capabilities"] == ["R0"]
        assert bundle.manifest["event_catalog"] == POLICY_EVENT_CATALOG
        assert all(event.event_catalog == POLICY_EVENT_CATALOG for event in bundle.events)
        assert bundle.manifest["numerical_policy_digest"] == candidate_policy("fp32").digest
        assert bundle.manifest["memory_plan_digest"] == engine.memory_plan.digest
        assert bundle.manifest["observation_profile"] == "metrics"
        assert open_streaming_capture(capture).manifest == bundle.manifest
        story = bundle.request_story(result.request_id)
        assert story["schema_version"] == 2 and story["capability"] == "R0"
        assert story["kv"]["final_active_allocation_bytes"] == 0
        assert not any(result.resource_status.values())
        for file in capture.iterdir():
            assert b"CAPTURE_SECRET_SENTINEL" not in file.read_bytes()
        with pytest.raises(IllmError) as caught:
            StreamingCaptureBundle.open(capture)
        assert caught.value.code is ErrorCode.UNSUPPORTED_SCHEMA
    finally:
        assert engine.close(timeout=2.0)


def test_explicit_legacy_none_gets_new_binding_without_changing_reference(tmp_path: Path) -> None:
    engine = Engine(project_root=ROOT, kv_mode="none", numerical_policy=legacy_policy("fp32"))
    try:
        result = engine.generate(_request(engine, tmp_path / "capture"))
        manifest = PolicyStreamingCaptureBundle.open(tmp_path / "capture").manifest
        assert manifest["numerical_policy_digest"] == legacy_policy("fp32").digest
        assert manifest["memory_plan"]["sidecar_bytes"] == 0
        assert result.capture_error is None
    finally:
        assert engine.close(timeout=2.0)


def test_actual_cancel_capture_waits_for_cleanup(tmp_path: Path) -> None:
    engine = Engine(project_root=ROOT, numerical_policy=candidate_policy("fp32"))
    capture = tmp_path / "cancel"
    try:
        stream = engine.stream(_request(engine, capture))
        next(stream)
        next(stream)
        engine.cancel("policy-story", "stop-demo")
        list(stream)
        stream.close()
        bundle = PolicyStreamingCaptureBundle.open(capture)
        assert bundle.manifest["terminal_outcome"] == "CANCELLED"
        assert bundle.request_story("policy-story")["kv"]["final_open_write_count"] == 0
        assert engine.last_result is not None
        assert not any(engine.last_result.resource_status.values())
    finally:
        assert engine.close(timeout=2.0)


def test_capture_creation_failure_rejects_before_acceptance(tmp_path: Path) -> None:
    capture = tmp_path / "foreign"
    capture.mkdir()
    (capture / "sentinel").write_bytes(b"must remain")
    engine = Engine(project_root=ROOT, numerical_policy=candidate_policy("fp32"))
    try:
        with pytest.raises(IllmError) as caught:
            engine.generate(_request(engine, capture))
        assert caught.value.code is ErrorCode.CAPTURE_INCOMPLETE
        assert engine.last_events == ()
        assert (capture / "sentinel").read_bytes() == b"must remain"
    finally:
        assert engine.close(timeout=2.0)


def test_accepted_writer_failure_preserves_outcome_and_incomplete(tmp_path: Path) -> None:
    original = PolicyStreamingCaptureSession.emit

    def fail_output(self: PolicyStreamingCaptureSession, kind: K, payload: Any) -> Any:
        if kind is K.REQUEST_OUTPUT_CHUNK:
            raise OSError("injected required writer failure")
        return original(self, kind, payload)

    capture = tmp_path / "failed-writer"
    engine = Engine(project_root=ROOT, numerical_policy=candidate_policy("fp32"))
    try:
        with patch.object(PolicyStreamingCaptureSession, "emit", fail_output):
            result = engine.generate(_request(engine, capture))
        assert result.outcome == "FINISHED" and result.capture_error is not None
        assert not any(result.resource_status.values())
        assert (capture / "INCOMPLETE").is_file()
        assert not (capture / "COMPLETE").exists()
        with pytest.raises(IllmError) as caught:
            PolicyStreamingCaptureBundle.open(capture)
        assert caught.value.code is ErrorCode.CAPTURE_INCOMPLETE
    finally:
        assert engine.close(timeout=2.0)


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1")
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "illm", *args], cwd=ROOT,
                          env=environment, capture_output=True, text=True, encoding="utf-8", timeout=90, check=False)


def test_actual_cli_explicit_policy_capture_inspect_and_trace(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    run = _cli("run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "7",
               "--max-new-tokens", "4", "--numerical-policy", "candidate-fp32", "--capture", str(capture))
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["schema_version"] == 3
    replay = _cli("replay", str(capture), "--mode", "trace")
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout)["schema_version"] == 2
    trace = tmp_path / "trace.json"
    inspect = _cli("inspect", "request", str(capture), "--request", "req-0001", "--chrome-trace", str(trace))
    assert inspect.returncode == 0, inspect.stderr
    assert json.loads(trace.read_text(encoding="utf-8"))["metadata"]["capability"] == "R0"
    for mode in ("state", "execute", "what-if"):
        rejected = _cli("replay", str(capture), "--mode", mode)
        assert rejected.returncode != 0
        assert "UNSUPPORTED_SCHEMA" in rejected.stderr

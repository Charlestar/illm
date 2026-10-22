"""Actual tiny-CPU learning views for policy-bound schema-3 captures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, cast
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.dispatch import capture_schema
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import IllmError
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, GenerationResult
from illm.executor import EagerExecutor, ExecutionTicket
from illm.model.spec import ModelError
from illm.numerical_policy import (
    CANDIDATE_FAMILY,
    LEGACY_FAMILY,
    NumericalExecutionPolicyV1,
    candidate_policy,
    legacy_policy,
)
from illm.sampling import SamplingSpec
from illm.streaming_learning import (
    LEARNING_FILES,
    POLICY_LEARNING_GENERATOR_ID,
    WIDTH,
    derive_streaming_learning_assets,
    verify_streaming_learning_assets,
)


ROOT = Path(__file__).resolve().parents[2]
PROMPT_SENTINEL = "POLICY_CAPTURE_PRIVATE_PROMPT_SENTINEL"
HMAC_KEY_SENTINEL = b"HMAC_KEY_PRIVATE_0123456789ABCDE"
_BANNED_METADATA_KEYS = {
    "prompt", "prompt_text", "output", "output_text", "text", "text_delta",
    "final_text_delta", "token_id", "token_ids", "input_ids", "input_token_ids",
    "output_ids", "output_token_ids", "logits", "tensor", "tensors", "key",
    "keys", "key_bytes", "hmac_key", "token_hmac_key", "kv_key",
}

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "illm_policy_render_learning_test", ROOT / "scripts" / "render_streaming_learning.py",
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
render_main = cast(Callable[[Sequence[str]], int], _SCRIPT.main)


def _actual_policy_capture(
    root: Path,
    *,
    mode: str,
    request_id: str,
    policy: NumericalExecutionPolicyV1 | None = None,
) -> tuple[Path, GenerationResult]:
    selected_policy = policy or candidate_policy("fp32")
    engine = Engine(
        project_root=ROOT,
        kv_mode=mode,
        numerical_policy=selected_policy,
    )
    capture = root / f"capture-{mode}"
    try:
        with patch("illm.engine.streaming.secrets.token_bytes", return_value=HMAC_KEY_SENTINEL):
            result = engine.generate(
                GenerationRequest(
                    engine.model_spec_id,
                    4,
                    SamplingSpec(mode="greedy", actual_seed=7),
                    prompt=PROMPT_SENTINEL,
                    request_id=request_id,
                    capture=capture,
                    telemetry="ops",
                )
            )
        assert result.capture_error is None
        return capture, result
    finally:
        assert engine.close(timeout=2.0)


@pytest.fixture(scope="module")
def policy_capture(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, GenerationResult]]:
    root = tmp_path_factory.mktemp("policy-learning")
    yield _actual_policy_capture(root, mode="contiguous", request_id="policy-learning-fixed")


def _metadata_keys(value: object) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                result.add(key.casefold())
            result.update(_metadata_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_metadata_keys(item))
    return result


def _write_assets(directory: Path, assets: dict[str, bytes]) -> None:
    directory.mkdir()
    for name, data in assets.items():
        (directory / name).write_bytes(data)


@pytest.mark.parametrize(
    ("mode", "path_label"),
    [
        ("none", "B FULL RECOMPUTE / KV NONE"),
        ("contiguous", "C CACHED / KV CONTIGUOUS"),
    ],
)
def test_policy_learning_assets_bind_actual_b_and_c_paths(
    tmp_path: Path,
    mode: str,
    path_label: str,
) -> None:
    capture, result = _actual_policy_capture(tmp_path, mode=mode, request_id=f"learning-{mode}")
    bundle = PolicyStreamingCaptureBundle.open(capture)
    assets = derive_streaming_learning_assets(capture)
    assert assets == derive_streaming_learning_assets(capture)
    assert set(assets) == LEARNING_FILES
    assert capture_schema(capture) == 3
    prompt_tokens = len(result.input_token_ids)
    expected_queries = (
        [prompt_tokens + index for index in range(4)]
        if mode == "none"
        else [prompt_tokens, 1, 1, 1]
    )
    assert list(result.query_tokens_per_step) == expected_queries

    story = json.loads(assets["request-story.json"])
    trace = json.loads(assets["chrome-trace.json"])
    provenance = json.loads(assets["provenance.json"])
    expected_capture_identity = {
        "numerical_policy_digest": bundle.manifest["numerical_policy_digest"],
        "memory_plan_digest": bundle.manifest["memory_plan_digest"],
        "execution_binding_digest": bundle.manifest["execution_binding_digest"],
        "implementation_source_digest": bundle.manifest["execution_binding"]["implementation_source_digest"],
        "environment_digest": bundle.manifest["execution_binding"]["environment_digest"],
        "protocol_sha256": bundle.manifest["protocol_sha256"],
        "observation_profile": "ops",
    }
    assert story["schema_version"] == 2 and story["capability"] == "R0"
    assert trace["metadata"]["schema_version"] == 2 and trace["metadata"]["capability"] == "R0"
    assert story["capture_identity"] == trace["metadata"]["capture_identity"] == expected_capture_identity
    assert story["kv"]["final_active_allocation_bytes"] == 0
    assert story["kv"]["final_open_write_count"] == 0

    assert provenance["schema_version"] == 2
    assert provenance["generator"] == POLICY_LEARNING_GENERATOR_ID
    assert provenance["capture_schema_version"] == 3
    assert provenance["evidence_kind"] == "derived"
    assert provenance["capabilities"] == ["R0"]
    assert provenance["not_live_ui_screenshot"] is True
    assert provenance["not_gpu_kernel_trace"] is True
    assert provenance["identity"] == bundle.manifest["identity"]
    assert provenance["numerical_policy"] == bundle.manifest["numerical_policy"]
    assert provenance["execution_binding"] == bundle.manifest["execution_binding"]
    assert provenance["memory_plan"] == bundle.manifest["memory_plan"]
    assert provenance["protocol_sha256"] == bundle.manifest["protocol_sha256"]
    assert provenance["event_catalog"] == bundle.manifest["event_catalog"]

    svg = assets["request-story.svg"].decode("utf-8")
    assert CANDIDATE_FAMILY.upper() in svg
    assert "STORAGE: FP32" in svg
    assert path_label in svg
    assert bundle.manifest["numerical_policy_digest"][:12].upper() in svg
    assert bundle.manifest["execution_binding"]["implementation_source_digest"][:12].upper() in svg
    assert bundle.manifest["execution_binding"]["environment_digest"][:12].upper() in svg
    assert "NOT A LIVE UI OR GPU KERNEL TRACE" in svg
    png = assets["request-story.png"]
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert int.from_bytes(png[16:20], "big") == WIDTH

    directory = tmp_path / f"assets-{mode}"
    _write_assets(directory, assets)
    assert verify_streaming_learning_assets(capture, directory) == {
        "status": "pass",
        "evidence_kind": "derived",
        "file_count": 5,
        "source_capture_manifest_digest": bundle.manifest["manifest_digest"],
        "capture_schema_version": 3,
        "generator": POLICY_LEARNING_GENERATOR_ID,
        "capabilities": ["R0"],
        "gpu_kernel_trace": False,
    }


def test_policy_learning_assets_label_actual_legacy_capture_as_path_a(tmp_path: Path) -> None:
    policy = legacy_policy("fp32")
    capture, result = _actual_policy_capture(
        tmp_path,
        mode="none",
        request_id="learning-legacy-a",
        policy=policy,
    )
    bundle = PolicyStreamingCaptureBundle.open(capture)
    assets = derive_streaming_learning_assets(capture)
    provenance = json.loads(assets["provenance.json"])
    svg = assets["request-story.svg"].decode("utf-8")

    prompt_tokens = len(result.input_token_ids)
    assert list(result.query_tokens_per_step) == [prompt_tokens + index for index in range(4)]
    assert bundle.manifest["numerical_policy_digest"] == policy.digest
    assert bundle.manifest["memory_plan"]["kv_mode"] == "none"
    assert provenance["numerical_policy"]["family"] == LEGACY_FAMILY
    assert provenance["numerical_policy_digest"] == policy.digest
    assert provenance["memory_plan"]["kv_mode"] == "none"
    assert "A FROZEN V0.1 REFERENCE / KV NONE" in svg
    assert "B FULL RECOMPUTE" not in svg
    assert "C CACHED" not in svg


@pytest.mark.parametrize(("cancel_wins", "expected_outcome"), [(False, "FAILED"), (True, "CANCELLED")])
def test_policy_learning_assets_render_actual_uncompleted_forward_without_inventing_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_wins: bool,
    expected_outcome: str,
) -> None:
    request_id = f"learning-uncompleted-{expected_outcome.lower()}"
    engine = Engine(
        project_root=ROOT,
        kv_mode="contiguous",
        numerical_policy=candidate_policy("fp32"),
    )
    capture = tmp_path / f"capture-{expected_outcome.lower()}"

    def fail_resolve(_executor: EagerExecutor, _ticket: ExecutionTicket) -> tuple[float, ...]:
        if cancel_wins:
            cancellation = engine.cancel(request_id, "cancel-before-resolve-error")
            assert cancellation.won and cancellation.outcome == "CANCELLED"
        raise ModelError("EXECUTION_FAILED", "synthetic resolved operation failure")

    monkeypatch.setattr(EagerExecutor, "resolve", fail_resolve)
    try:
        result = engine.generate(
            GenerationRequest(
                engine.model_spec_id,
                4,
                SamplingSpec(mode="greedy", actual_seed=7),
                prompt="hello",
                request_id=request_id,
                capture=capture,
                telemetry="ops",
            )
        )
        assert result.outcome == expected_outcome
        assert result.output_token_ids == ()
        assert not any(result.resource_status.values())
        assert result.capture_error is None
    finally:
        assert engine.close(timeout=2.0)

    bundle = PolicyStreamingCaptureBundle.open(capture)
    assert bundle.manifest["terminal_outcome"] == expected_outcome
    assets = derive_streaming_learning_assets(capture)
    provenance = json.loads(assets["provenance.json"])
    story = json.loads(assets["request-story.json"])
    trace = json.loads(assets["chrome-trace.json"])
    svg = assets["request-story.svg"].decode("utf-8")

    assert provenance["completed_forward_count"] == 0
    assert provenance["uncompleted_forward_count"] == 1
    assert provenance["total_forward_count"] == 1
    assert story["terminal_outcome"] == expected_outcome
    assert story["kv"]["final_active_allocation_bytes"] == 0
    assert story["kv"]["final_open_write_count"] == 0
    forward_trace = [item for item in trace["traceEvents"] if item["name"] == "model.forward"]
    assert [item["ph"] for item in forward_trace] == ["B"]
    assert all("dur" not in item for item in forward_trace)
    assert "NO COMPLETION RECORDED" in svg
    assert f"OUTCOME: {expected_outcome}" in svg

    directory = tmp_path / f"views-{expected_outcome.lower()}"
    _write_assets(directory, assets)
    assert verify_streaming_learning_assets(capture, directory)["status"] == "pass"


def test_policy_learning_assets_do_not_disclose_prompt_token_ids_or_hmac_key(
    policy_capture: tuple[Path, GenerationResult],
) -> None:
    capture, result = policy_capture
    assets = derive_streaming_learning_assets(capture)
    combined = b"".join(assets.values())
    assert PROMPT_SENTINEL.encode("utf-8") not in combined
    assert HMAC_KEY_SENTINEL not in combined
    assert result.input_token_ids
    for name in ("request-story.json", "chrome-trace.json", "provenance.json"):
        value = json.loads(assets[name])
        assert not (_metadata_keys(value) & _BANNED_METADATA_KEYS)
    story = json.loads(assets["request-story.json"])
    assert story["tokens"]
    assert all("token_hmac" in item["payload"] for item in story["tokens"])


def test_resigned_capture_and_resigned_learning_view_tampering_are_rejected(
    policy_capture: tuple[Path, GenerationResult], tmp_path: Path,
) -> None:
    capture, _ = policy_capture
    resigned_capture = tmp_path / "resigned-capture"
    shutil.copytree(capture, resigned_capture)
    manifest_path = resigned_capture / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["numerical_policy_digest"] = "0" * 64
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    (resigned_capture / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
    with pytest.raises(IllmError):
        derive_streaming_learning_assets(resigned_capture)

    assets = derive_streaming_learning_assets(capture)
    directory = tmp_path / "resigned-learning"
    _write_assets(directory, assets)
    story_path = directory / "request-story.json"
    story = json.loads(story_path.read_bytes())
    story["capture_identity"]["numerical_policy_digest"] = "f" * 64
    story_bytes = canonical_json_bytes(story) + b"\n"
    story_path.write_bytes(story_bytes)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_bytes())
    provenance["files"]["request-story.json"] = hashlib.sha256(story_bytes).hexdigest()
    provenance["provenance_digest"] = sha256_json(
        {key: value for key, value in provenance.items() if key != "provenance_digest"}
    )
    provenance_path.write_bytes(canonical_json_bytes(provenance) + b"\n")
    with pytest.raises(IllmError):
        verify_streaming_learning_assets(capture, directory)


def test_policy_render_cli_is_exact_read_only_and_refuses_foreign_outputs(
    policy_capture: tuple[Path, GenerationResult], tmp_path: Path,
) -> None:
    capture, _ = policy_capture
    before_capture = {path.name: path.read_bytes() for path in capture.iterdir()}
    directory = tmp_path / "views"
    assert render_main([str(capture), "--output", str(directory)]) == 0
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert render_main([str(capture), "--output", str(directory), "--verify-existing"]) == 0
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    sentinel = foreign / "notes.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    assert render_main([str(capture), "--output", str(foreign)]) != 0
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
    assert render_main([str(capture), "--output", str(capture / "views")]) != 0
    assert {path.name: path.read_bytes() for path in capture.iterdir()} == before_capture


@pytest.mark.parametrize("duplicate_start", [False, True])
def test_schema2_learning_contract_remains_the_historical_shape(tmp_path: Path, duplicate_start: bool) -> None:
    engine = Engine(project_root=ROOT)
    capture = tmp_path / "schema2-capture"
    try:
        result = engine.generate(
            GenerationRequest(
                engine.model_spec_id,
                4,
                SamplingSpec(),
                prompt="hello",
                request_id="historical-schema2-learning",
                capture=capture,
            )
        )
        assert result.capture_error is None
    finally:
        assert engine.close(timeout=2.0)
    if duplicate_start:
        # The historical reader permits this R0 input. Do not tighten schema2
        # while introducing the separately strict policy-aware family.
        events = [json.loads(line) for line in (capture / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        start = next(index for index, event in enumerate(events) if event["kind"] == "model.forward_started")
        events.insert(start, dict(events[start]))
        for sequence, event in enumerate(events):
            event["producer_seq_no"] = sequence
        (capture / "events.jsonl").write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in events))
        facts = {"events.jsonl": hashlib.sha256((capture / "events.jsonl").read_bytes()).hexdigest()}
        (capture / "checksums.json").write_bytes(canonical_json_bytes(facts) + b"\n")
        manifest = json.loads((capture / "capture-manifest.json").read_text(encoding="utf-8"))
        manifest["event_count"] = len(events)
        manifest["files"] = {**facts, "checksums.json": hashlib.sha256((capture / "checksums.json").read_bytes()).hexdigest()}
        manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
        (capture / "capture-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        (capture / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
    assert capture_schema(capture) == 2
    bundle = StreamingCaptureBundle.open(capture)
    assets = derive_streaming_learning_assets(capture)
    provenance = json.loads(assets["provenance.json"])
    assert set(provenance) == {
        "schema_version", "generator", "evidence_kind", "not_live_ui_screenshot",
        "source_capture_manifest_digest", "source_model_spec_id",
        "source_memory_plan_digest", "source_sampling_spec_digest",
        "visible_forward_count", "total_forward_count", "files", "provenance_digest",
    }
    assert provenance["schema_version"] == 1
    assert provenance["generator"] == "illm-stateful-request-story-v1"
    assert "numerical_policy" not in provenance and "execution_binding" not in provenance
    directory = tmp_path / "schema2-views"
    _write_assets(directory, assets)
    assert verify_streaming_learning_assets(capture, directory) == {
        "status": "pass",
        "evidence_kind": "derived",
        "file_count": 5,
        "source_capture_manifest_digest": bundle.manifest["manifest_digest"],
    }

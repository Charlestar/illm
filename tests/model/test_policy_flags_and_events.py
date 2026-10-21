"""CPU policy flags and event catalogs are exact, isolated, and recoverable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from illm.domain.errors import IllmError
from illm.domain.events import DomainEvent
from illm.domain.policy_events import POLICY_EVENT_CATALOG, PolicyStreamingEvent
from illm.domain.streaming_events import EVENT_CATALOG, StreamingEvent, StreamingEventKind
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest
from illm.model import LoadedModel, load_model
from illm.model.policy import PolicyDecoder, PolicyOpRecord
from illm.numerical_policy import candidate_policy
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded() -> LoadedModel:
    return load_model(project_root=ROOT)


@dataclass(frozen=True, slots=True)
class _FlagState:
    threads: int
    deterministic: bool
    warn_only: bool
    precision: str
    tf32: bool
    cudnn_tf32: bool
    fp16_reduction: bool
    bf16_reduction: bool
    workspace: str | None


def _flags() -> _FlagState:
    return _FlagState(
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def _apply_flags(state: _FlagState) -> None:
    torch.backends.cuda.matmul.allow_tf32 = state.tf32
    torch.backends.cudnn.allow_tf32 = state.cudnn_tf32
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = state.fp16_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = state.bf16_reduction
    if state.workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = state.workspace
    # The bool allow_tf32 setter collapses medium/high. Restore the exact
    # three-state precision after all CUDA backend booleans.
    torch.set_float32_matmul_precision(state.precision)
    torch.use_deterministic_algorithms(state.deterministic, warn_only=state.warn_only)
    torch.set_num_threads(state.threads)


def _noncanonical_flags() -> _FlagState:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = "test-original-workspace"
    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_num_threads(2)
    return _flags()


@pytest.mark.parametrize("hook_failure", [False, True], ids=["normal", "hook-error"])
def test_cpu_policy_sets_and_restores_all_flags_without_cuda_initialization(
    loaded: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
    hook_failure: bool,
) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    original = _flags()
    lazy_init_calls: list[tuple[object, ...]] = []

    def forbidden_lazy_init(*args: object, **kwargs: object) -> None:
        del kwargs
        lazy_init_calls.append(args)
        raise AssertionError("CPU policy execution must not initialize CUDA")

    try:
        before = _noncanonical_flags()
        assert before.precision == "medium"
        monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_lazy_init)
        observed: list[str] = []

        def inspect(record: PolicyOpRecord, tensor: Tensor) -> None:
            del tensor
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled() is True
            assert torch.is_deterministic_algorithms_warn_only_enabled() is False
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is False
            assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"
            assert torch.is_autocast_enabled("cpu") is False
            observed.append(record.name)
            if hook_failure:
                raise RuntimeError("injected policy hook failure")

        input_ids = torch.tensor([[1, 107]], dtype=torch.int64)
        with torch.autocast(device_type="cpu", enabled=True):
            if hook_failure:
                with pytest.raises(RuntimeError, match="injected policy hook failure"):
                    decoder.forward(input_ids, op_hook=inspect)
            else:
                logits = decoder.forward(input_ids, op_hook=inspect)
                assert tuple(logits.shape) == (1, 2, loaded.spec.vocab_size)
            assert torch.is_autocast_enabled("cpu") is True

        assert observed
        assert lazy_init_calls == []
        assert _flags() == before
    finally:
        _apply_flags(original)


def test_policy_event_catalog_has_explicit_reader_dispatch() -> None:
    event = PolicyStreamingEvent(
        "stateful-engine",
        0,
        1,
        StreamingEventKind.REQUEST_TOKENIZED,
        {"request_id": "policy-event", "started_ns": 1, "ended_ns": 2, "input_tokens": 1},
    )
    wire = event.to_dict()
    assert wire["event_catalog"] == POLICY_EVENT_CATALOG
    assert PolicyStreamingEvent.from_dict(wire).to_dict() == wire
    with pytest.raises(IllmError, match="streaming event registry identity is unsupported"):
        StreamingEvent.from_dict(wire)

    historical = StreamingEvent(
        "stateful-engine",
        0,
        1,
        StreamingEventKind.REQUEST_TOKENIZED,
        {"request_id": "old-event", "started_ns": 1, "ended_ns": 2, "input_tokens": 1},
    ).to_dict()
    assert historical["event_catalog"] == EVENT_CATALOG
    assert StreamingEvent.from_dict(historical).to_dict() == historical
    with pytest.raises(IllmError, match="unsupported policy event catalog"):
        PolicyStreamingEvent.from_dict(historical)


@dataclass(slots=True)
class _CollectingSink:
    events: list[DomainEvent]

    def emit(self, event: DomainEvent) -> None:
        self.events.append(event)


def _request(engine: Engine, request_id: str) -> GenerationRequest:
    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        input_token_ids=(1, 107),
        max_new_tokens=1,
        sampling=SamplingSpec(mode="greedy", actual_seed=7),
        request_id=request_id,
        telemetry="request",
    )


def _exported_catalogs(events: list[DomainEvent]) -> set[str]:
    return {
        str(event.payload["streaming_event"]["event_catalog"])
        for event in events
    }


def test_policy_engine_and_exporter_use_new_catalog_without_relabeling_old_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    policy_sink = _CollectingSink([])
    policy_engine = Engine(project_root=ROOT, numerical_policy=candidate_policy("fp32"), exporters=(policy_sink,))
    try:
        policy_engine.generate(_request(policy_engine, "policy-export"))
        assert policy_engine.last_events
        assert all(isinstance(event, PolicyStreamingEvent) for event in policy_engine.last_events)
        assert {event.event_catalog for event in policy_engine.last_events} == {POLICY_EVENT_CATALOG}
    finally:
        assert policy_engine.close(timeout=5.0)
    assert policy_sink.events
    assert _exported_catalogs(policy_sink.events) == {POLICY_EVENT_CATALOG}

    historical_sink = _CollectingSink([])
    historical_engine = Engine(project_root=ROOT, exporters=(historical_sink,))
    try:
        historical_engine.generate(_request(historical_engine, "historical-export"))
        assert historical_engine.last_events
        assert all(type(event) is StreamingEvent for event in historical_engine.last_events)
        assert {event.event_catalog for event in historical_engine.last_events} == {EVENT_CATALOG}
    finally:
        assert historical_engine.close(timeout=5.0)
    assert historical_sink.events
    assert _exported_catalogs(historical_sink.events) == {EVENT_CATALOG}

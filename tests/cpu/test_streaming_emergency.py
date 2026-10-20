"""Deterministic CPU tests for forced second-interrupt ownership isolation."""

from __future__ import annotations

import argparse
import signal
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest
import torch

from illm.api import streaming_cli
from illm.capture.streaming import StreamingCaptureSession
from illm.domain.errors import EmergencyAbandon, ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K
import illm.engine.streaming as streaming_module
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk, TerminalChunk
import illm.executor.eager as eager_module
from illm.executor import CompletionFence, EagerExecutor
from illm.executor.fence import record_completion_fence
from illm.kv_cache import ContiguousKVStore
from illm.model import LoadedModel, load_model
from illm.model.incremental import IncrementalDecoder
from illm.model.spec import ModelError
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_TERMINALS = {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    value = load_model(project_root=ROOT)
    yield value
    torch.set_num_threads(original_threads)


def _request(engine: Engine, request_id: str, *, capture: Path | None = None) -> GenerationRequest:
    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        input_token_ids=(1, 104),
        max_new_tokens=1,
        sampling=SamplingSpec(actual_seed=7),
        request_id=request_id,
        capture=capture,
        telemetry="request" if capture is not None else "off",
    )


def _traceback_has_tensor(error: BaseException, name: str) -> bool:
    traceback = error.__traceback__
    while traceback is not None:
        value = traceback.tb_frame.f_locals.get(name)
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, tuple) and any(isinstance(item, torch.Tensor) for item in value):
            return True
        traceback = traceback.tb_next
    return False


def _assert_executor_fail_stopped(executor: EagerExecutor) -> None:
    with pytest.raises(ModelError) as future:
        executor.launch((1, 105))
    assert future.value.code == "EXECUTION_STATE_UNKNOWN"
    with pytest.raises(ModelError) as close:
        executor.close()
    assert close.value.code == "EXECUTION_STATE_UNKNOWN"


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_force_during_forward_quarantines_original_without_cleanup_proof(
    loaded: LoadedModel, kv_mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, kv_mode, 8)
    emergency = EmergencyAbandon("forced during forward")
    fence_records = 0
    cleanup_proofs = 0

    def forbidden_fence(_device: torch.device) -> NoReturn:
        nonlocal fence_records
        fence_records += 1
        pytest.fail("forced unwind must not record a cleanup fence")

    def forbidden_proof(*_values: object) -> NoReturn:
        nonlocal cleanup_proofs
        cleanup_proofs += 1
        pytest.fail("forced unwind must not enter normal failed-launch proof")

    def forced_forward(*_values: object, **_options: object) -> NoReturn:
        forward_local = torch.ones(1)
        assert forward_local.numel() == 1
        raise emergency

    monkeypatch.setattr(eager_module, "record_completion_fence", forbidden_fence)
    monkeypatch.setattr(executor, "_prove_failed_launch_complete", forbidden_proof)
    if kv_mode == "none":
        monkeypatch.setattr(type(loaded.reference), "forward", forced_forward)
    else:
        monkeypatch.setattr(IncrementalDecoder, "forward", forced_forward)

    with pytest.raises(EmergencyAbandon) as raised:
        executor.launch((1, 104))
    assert raised.value is emergency
    assert fence_records == cleanup_proofs == 0
    assert len(executor._quarantined) == 1
    assert emergency in executor._quarantined[0].resource_refs
    assert _traceback_has_tensor(emergency, "forward_local")
    status = executor.resource_status
    assert status.active_tickets == status.workspace_leases == 1
    assert status.input_leases == 1 and status.output_leases == 0
    if kv_mode == "contiguous":
        assert status.kv_write_leases == 1 and status.kv_staged_tokens == 2
    else:
        assert status.kv_write_leases == status.kv_staged_tokens == 0
    _assert_executor_fail_stopped(executor)


def test_force_after_begin_append_mutation_never_leaves_executor_reusable(
    loaded: LoadedModel, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, "contiguous", 8)
    emergency = EmergencyAbandon("forced before begin_append returned")
    original_begin = ContiguousKVStore.begin_append

    def begin_then_force(store: ContiguousKVStore, token_ids: tuple[int, ...]) -> NoReturn:
        lease = original_begin(store, token_ids)
        assert lease.state == "staged" and store.active_leases == 1
        raise emergency

    def forbidden_fence(_device: torch.device) -> NoReturn:
        pytest.fail("no model work started, so no completion fence may be recorded")

    monkeypatch.setattr(ContiguousKVStore, "begin_append", begin_then_force)
    monkeypatch.setattr(eager_module, "record_completion_fence", forbidden_fence)
    with pytest.raises(EmergencyAbandon) as raised:
        executor.launch((1, 104))
    assert raised.value is emergency
    assert len(executor._quarantined) == 1
    assert emergency in executor._quarantined[0].resource_refs
    traceback = emergency.__traceback__
    assert traceback is not None
    assert any(
        frame.tb_frame.f_locals.get("lease") is not None
        for frame in _traceback_chain(traceback)
    )
    status = executor.resource_status
    assert status.active_tickets == status.workspace_leases == 1
    assert status.kv_write_leases == 1 and status.kv_staged_tokens == 2
    _assert_executor_fail_stopped(executor)


def _traceback_chain(traceback: Any) -> Iterator[Any]:
    while traceback is not None:
        yield traceback
        traceback = traceback.tb_next


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_force_during_ticket_registration_retains_fence_and_tensor_owners(
    loaded: LoadedModel, kv_mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, kv_mode, 8)
    emergency = EmergencyAbandon("forced during ticket registration")
    original_record = record_completion_fence
    fence_records = 0

    def counted_record(device: torch.device) -> CompletionFence:
        nonlocal fence_records
        fence_records += 1
        return original_record(device)

    def interrupt_ticket(**values: Any) -> NoReturn:
        resource_refs = values["resource_refs"]
        assert all(isinstance(value, torch.Tensor) for value in resource_refs)
        raise emergency

    def forbidden_proof(*_values: object) -> NoReturn:
        pytest.fail("forced unwind must not start failed-launch cleanup proof")

    monkeypatch.setattr(eager_module, "record_completion_fence", counted_record)
    monkeypatch.setattr(eager_module, "ExecutionTicket", interrupt_ticket)
    monkeypatch.setattr(executor, "_prove_failed_launch_complete", forbidden_proof)
    with pytest.raises(EmergencyAbandon) as raised:
        executor.launch((1, 104))
    assert raised.value is emergency and fence_records == 1
    assert len(executor._quarantined) == 1
    refs = executor._quarantined[0].resource_refs
    assert emergency in refs
    assert any(hasattr(value, "wait") and hasattr(value, "is_complete") for value in refs)
    assert _traceback_has_tensor(emergency, "resource_refs")
    status = executor.resource_status
    assert status.active_tickets == status.input_leases == status.output_leases == status.workspace_leases == 1
    if kv_mode == "contiguous":
        assert status.kv_write_leases == 1
    _assert_executor_fail_stopped(executor)


class _EmergencyFence:
    def __init__(self, base: CompletionFence, error: EmergencyAbandon) -> None:
        self.base = base
        self.error = error
        self.wait_calls = 0
        self.completion_queries = 0

    def wait(self) -> None:
        self.wait_calls += 1
        raise self.error

    def is_complete(self) -> bool | None:
        self.completion_queries += 1
        pytest.fail("forced resolve must not query or wait on the fence again")

    @property
    def succeeded(self) -> bool | None:
        pytest.fail("forced resolve must not inspect fence success")


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_force_inside_failed_launch_cleanup_proof_is_not_swallowed(
    loaded: LoadedModel, kv_mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, kv_mode, 8)
    emergency = EmergencyAbandon("forced inside cleanup proof")
    original = RuntimeError("ordinary post-dispatch failure")
    record_calls = 0
    proof_fence: _EmergencyFence | None = None

    def fail_then_force(device: torch.device) -> CompletionFence:
        nonlocal record_calls, proof_fence
        record_calls += 1
        if record_calls == 1:
            raise original
        proof_fence = _EmergencyFence(record_completion_fence(device), emergency)
        return proof_fence

    monkeypatch.setattr(eager_module, "record_completion_fence", fail_then_force)
    with pytest.raises(EmergencyAbandon) as raised:
        executor.launch((1, 104))
    assert raised.value is emergency and record_calls == 2
    assert proof_fence is not None
    assert proof_fence.wait_calls == 1 and proof_fence.completion_queries == 0
    assert len(executor._quarantined) == 1
    refs = executor._quarantined[0].resource_refs
    assert original in refs and proof_fence in refs
    assert proof_fence.error is emergency
    _assert_executor_fail_stopped(executor)


def test_force_inside_abort_wait_propagates_without_a_second_query_or_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emergency = EmergencyAbandon("forced inside abort wait")
    engine_holder: dict[str, Engine] = {}
    fences: list[_CancelThenEmergencyFence] = []

    class CancelThenEmergencyExecutor(EagerExecutor):
        def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
            wrapped = _CancelThenEmergencyFence(fence, emergency, engine_holder)
            fences.append(wrapped)
            return wrapped

    monkeypatch.setattr(streaming_module, "EagerExecutor", CancelThenEmergencyExecutor)
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    engine_holder["engine"] = engine
    capture = tmp_path / "abort-emergency"
    stream = engine.stream(_request(engine, "abort-emergency", capture=capture))
    assert isinstance(next(stream), MetadataChunk)
    with pytest.raises(EmergencyAbandon) as raised:
        next(stream)
    assert raised.value is emergency and len(fences) == 1
    assert fences[0].wait_calls == 2
    assert fences[0].completion_queries == 1
    assert engine.last_result is None and engine.observations == ()
    assert not any(event.kind in PHYSICAL_TERMINALS for event in engine.last_events)
    assert engine.resource_status["quarantined_requests"] == 1
    assert (capture / "INCOMPLETE").is_file() and not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as metrics:
        engine.metrics()
    assert metrics.value.code is ErrorCode.CAPTURE_INCOMPLETE
    assert engine.close() is False


class _CancelThenEmergencyFence:
    def __init__(
        self,
        base: CompletionFence,
        error: EmergencyAbandon,
        engine_holder: dict[str, Engine],
    ) -> None:
        self.base = base
        self.error = error
        self.engine_holder = engine_holder
        self.wait_calls = 0
        self.completion_queries = 0

    def wait(self) -> None:
        self.wait_calls += 1
        if self.wait_calls == 1:
            self.base.wait()
            self.engine_holder["engine"].cancel("abort-emergency", "cancel-before-abort")
            return
        raise self.error

    def is_complete(self) -> bool | None:
        self.completion_queries += 1
        return self.base.is_complete()

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded


def test_force_during_resolve_waits_once_then_engine_quarantines_without_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fences: list[_EmergencyFence] = []
    emergency = EmergencyAbandon("forced during resolve wait")

    class EmergencyExecutor(EagerExecutor):
        def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
            wrapped = _EmergencyFence(fence, emergency)
            fences.append(wrapped)
            return wrapped

    monkeypatch.setattr(streaming_module, "EagerExecutor", EmergencyExecutor)
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    capture = tmp_path / "resolve-emergency"
    stream = engine.stream(_request(engine, "resolve-emergency", capture=capture))
    assert isinstance(next(stream), MetadataChunk)
    with pytest.raises(EmergencyAbandon) as raised:
        next(stream)
    assert raised.value is emergency
    assert len(fences) == 1
    assert fences[0].wait_calls == 1 and fences[0].completion_queries == 0
    assert engine.last_result is None and engine.observations == ()
    assert not any(event.kind in PHYSICAL_TERMINALS for event in engine.last_events)
    status = engine.resource_status
    assert status["quarantined_requests"] == status["reservations"] == 1
    assert status["active_tickets"] == 1 and status["kv_write_leases"] == 1
    assert (capture / "INCOMPLETE").is_file() and not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as metrics:
        engine.metrics()
    assert metrics.value.code is ErrorCode.CAPTURE_INCOMPLETE
    logical = engine.cancel("resolve-emergency", "after-emergency")
    assert logical.outcome == "FAILED" and logical.terminal_ready is False
    future = engine.stream(_request(engine, "future-after-resolve-emergency"))
    with pytest.raises(IllmError) as stopped:
        next(future)
    assert stopped.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert engine.close() is False


def test_force_after_capture_finalize_preserves_committed_server_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emergency = EmergencyAbandon("forced after capture finalize")
    original_finalize = StreamingCaptureSession.finalize

    def finalize_then_force(session: StreamingCaptureSession) -> NoReturn:
        manifest = original_finalize(session)
        assert manifest["finalized"] is True
        assert (session.root / "COMPLETE").is_file()
        raise emergency

    monkeypatch.setattr(StreamingCaptureSession, "finalize", finalize_then_force)
    engine = Engine(kv_mode="none", project_root=ROOT)
    capture = tmp_path / "post-finalize-emergency"
    stream = engine.stream(_request(engine, "post-finalize-emergency", capture=capture))
    observed: list[object] = [next(stream), next(stream)]
    assert isinstance(observed[0], MetadataChunk)
    assert not any(isinstance(chunk, TerminalChunk) for chunk in observed)
    with pytest.raises(EmergencyAbandon) as raised:
        next(stream)
    assert raised.value is emergency
    assert not any(isinstance(chunk, TerminalChunk) for chunk in observed)
    assert engine.last_result is None
    assert len(engine.observations) == 1
    assert engine.observations[0].request_id == "post-finalize-emergency"
    assert engine.observations[0].outcome == "FINISHED"
    terminal = engine.cancel("post-finalize-emergency", "after-client-interrupt")
    assert terminal.outcome == "FINISHED" and terminal.terminal_ready is True
    assert all(value == 0 for value in engine.resource_status.values())
    assert (capture / "COMPLETE").is_file()
    assert not (capture / "INCOMPLETE").exists()
    metrics = engine.metrics()
    assert metrics["request_totals"] == {"offered": 1, "accepted": 1, "preaccept": 0}
    assert engine.close() is True


def test_force_after_reclaim_before_server_terminal_record_fails_stopped_without_fake_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emergency = EmergencyAbandon("forced after reclaim before server terminal")
    original_emit = Engine._emit

    def interrupt_physical_terminal(
        engine: Engine, runtime: Any, kind: K, payload: dict[str, Any],
    ) -> Any:
        if kind in PHYSICAL_TERMINALS:
            raise emergency
        return original_emit(engine, runtime, kind, payload)

    monkeypatch.setattr(Engine, "_emit", interrupt_physical_terminal)
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    capture = tmp_path / "post-reclaim-emergency"
    stream = engine.stream(_request(engine, "post-reclaim-emergency", capture=capture))
    observed: list[object] = [next(stream), next(stream)]
    assert isinstance(observed[0], MetadataChunk)
    with pytest.raises(EmergencyAbandon) as raised:
        next(stream)
    assert raised.value is emergency
    assert not any(isinstance(chunk, TerminalChunk) for chunk in observed)
    assert engine.last_result is None and engine.observations == ()
    assert not any(event.kind in PHYSICAL_TERMINALS for event in engine.last_events)
    assert all(value == 0 for value in engine.resource_status.values())
    assert (capture / "INCOMPLETE").is_file()
    assert not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as metrics:
        engine.metrics()
    assert metrics.value.code is ErrorCode.CAPTURE_INCOMPLETE
    future = engine.stream(_request(engine, "future-post-reclaim-emergency"))
    with pytest.raises(IllmError) as stopped:
        next(future)
    assert stopped.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert engine.close() is True


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_force_before_metadata_quarantines_accepted_runtime_and_capture(
    tmp_path: Path, kv_mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    emergency = EmergencyAbandon(f"forced before metadata {kv_mode}")
    original_emit = Engine._emit

    def interrupt_accept(
        engine: Engine, runtime: Any, kind: K, payload: dict[str, Any],
    ) -> Any:
        if kind is K.REQUEST_ACCEPTED:
            raise emergency
        return original_emit(engine, runtime, kind, payload)

    monkeypatch.setattr(Engine, "_emit", interrupt_accept)
    engine = Engine(kv_mode=kv_mode, project_root=ROOT)
    capture = tmp_path / f"pre-metadata-{kv_mode}"
    stream = engine.stream(_request(engine, f"pre-metadata-{kv_mode}", capture=capture))
    with pytest.raises(EmergencyAbandon) as raised:
        next(stream)
    assert raised.value is emergency
    assert engine.last_result is None and engine.observations == ()
    assert not any(event.kind in PHYSICAL_TERMINALS for event in engine.last_events)
    status = engine.resource_status
    assert status["active_requests"] == 0
    assert status["quarantined_requests"] == status["reservations"] == 1
    if kv_mode == "contiguous":
        assert status["kv_reserved_bytes"] > 0
    assert (capture / "INCOMPLETE").is_file() and not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as metrics:
        engine.metrics()
    assert metrics.value.code is ErrorCode.CAPTURE_INCOMPLETE
    future = engine.stream(_request(engine, f"future-pre-metadata-{kv_mode}"))
    with pytest.raises(IllmError) as stopped:
        next(future)
    assert stopped.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert engine.close() is False


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_public_abandon_after_metadata_never_launches_again(
    tmp_path: Path, kv_mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launches = 0

    def forbidden_launch(_executor: EagerExecutor, _context: tuple[int, ...]) -> NoReturn:
        nonlocal launches
        launches += 1
        pytest.fail("an abandoned accepted request must not launch more model work")

    engine = Engine(kv_mode=kv_mode, project_root=ROOT)
    capture = tmp_path / f"public-abandon-{kv_mode}"
    request_id = f"public-abandon-{kv_mode}"
    stream = engine.stream(_request(engine, request_id, capture=capture))
    assert isinstance(next(stream), MetadataChunk)
    monkeypatch.setattr(EagerExecutor, "launch", forbidden_launch)
    engine.abandon(request_id, reason="CLI_SECOND_INTERRUPT")
    with pytest.raises(IllmError) as abandoned:
        next(stream)
    assert abandoned.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert launches == 0
    assert engine.last_result is None and engine.observations == ()
    assert not any(event.kind in PHYSICAL_TERMINALS for event in engine.last_events)
    status = engine.resource_status
    assert status["quarantined_requests"] == status["reservations"] == 1
    assert (capture / "INCOMPLETE").is_file() and not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as metrics:
        engine.metrics()
    assert metrics.value.code is ErrorCode.CAPTURE_INCOMPLETE
    future = engine.stream(_request(engine, f"future-{request_id}"))
    with pytest.raises(IllmError) as stopped:
        next(future)
    assert stopped.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert engine.close() is False


def test_second_interrupt_type_and_cli_exit_are_immediate_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {"abandonments": [], "closed": [], "stream_closed": False}

    class ControllerEngine:
        def cancel(self, _request_id: str, _cancellation_id: str) -> None:
            return None

    controller = streaming_cli._InterruptController(ControllerEngine(), "type-check")
    controller.handle(signal.SIGINT, None)
    with pytest.raises(EmergencyAbandon):
        controller.handle(signal.SIGINT, None)

    class FakeRequest:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeChunk:
        pass

    class FakeEngine:
        model_spec_id = "a" * 64

        def __init__(self, **_values: Any) -> None:
            return None

        def stream(self, _request: Any) -> Iterator[Any]:
            try:
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
                handler(signal.SIGINT, None)
                yield pytest.fail("forced second interrupt must unwind before a chunk")
            finally:
                state["stream_closed"] = True

        def abandon(self, request_id: str, *, reason: str) -> None:
            state["abandonments"].append((request_id, reason))

        def close(self, *, timeout: float) -> bool:
            state["closed"].append(timeout)
            return False

    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: (FakeEngine, FakeRequest, FakeChunk, FakeChunk, FakeChunk),
    )
    args = argparse.Namespace(
        command="run",
        model="fixture:tiny-decoder",
        model_config=None,
        device="cpu",
        prompt="hello",
        input_token_ids=None,
        max_new_tokens=1,
        stream=True,
        sampling_mode="greedy",
        temperature=None,
        top_k=0,
        top_p=1.0,
        seed=0,
        add_bos=None,
        kv_mode="none",
        max_context_tokens=None,
        memory_budget_bytes=1024,
        request_id="cli-emergency",
        capture=None,
        telemetry="off",
        cancel_after_tokens=None,
    )
    assert streaming_cli.dispatch(args) == 130
    assert state["abandonments"] == [("cli-emergency", "CLI_SECOND_INTERRUPT")]
    assert state["stream_closed"] is True and state["closed"] == [1.0]


def test_second_interrupt_during_controller_install_exits_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {"abandonments": [], "closed": []}

    class FakeRequest:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeChunk:
        pass

    class FakeEngine:
        model_spec_id = "a" * 64

        def __init__(self, **_values: Any) -> None:
            return None

        def abandon(self, request_id: str, *, reason: str) -> None:
            state["abandonments"].append((request_id, reason))

        def close(self, *, timeout: float) -> bool:
            state["closed"].append(timeout)
            return True

    def force_inside_install(controller: streaming_cli._InterruptController) -> NoReturn:
        controller.handle(signal.SIGINT, None)
        controller.handle(signal.SIGINT, None)
        raise AssertionError("the second handler invocation must force unwind")

    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: (FakeEngine, FakeRequest, FakeChunk, FakeChunk, FakeChunk),
    )
    monkeypatch.setattr(streaming_cli._InterruptController, "install", force_inside_install)
    args = argparse.Namespace(
        command="run",
        model="fixture:tiny-decoder",
        model_config=None,
        device="cpu",
        prompt="hello",
        input_token_ids=None,
        max_new_tokens=1,
        stream=True,
        sampling_mode="greedy",
        temperature=None,
        top_k=0,
        top_p=1.0,
        seed=0,
        add_bos=None,
        kv_mode="none",
        max_context_tokens=None,
        memory_budget_bytes=1024,
        request_id="install-emergency",
        capture=None,
        telemetry="off",
        cancel_after_tokens=None,
    )
    assert streaming_cli.dispatch(args) == 130
    assert state["abandonments"] == [("install-emergency", "CLI_SECOND_INTERRUPT")]
    assert state["closed"] == [1.0]

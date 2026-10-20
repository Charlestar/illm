"""V0.2 streaming-engine correctness, lifecycle, and failure boundaries."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch

import illm.engine.streaming as streaming_module
from illm.capture.streaming import StreamingCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import (
    GenerationRequest,
    MetadataChunk,
    TerminalChunk,
    TokenChunk,
)
from illm.executor import CompletionFence, EagerExecutor, ExecutionTicket
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads(
    (ROOT / "fixtures" / "tiny-decoder" / "oracle-v1" / "oracle-manifest.json").read_text(encoding="utf-8")
)
GOLDEN_CASES = tuple(ORACLE["greedy_cases"])


def _fifty_cases() -> tuple[tuple[str, tuple[int, ...], int, SamplingSpec], ...]:
    cases: list[tuple[str, tuple[int, ...], int, SamplingSpec]] = []
    for index in range(50):
        if index == 0:
            length, outputs = 1, 1
        elif index == 1:
            length, outputs = 2, 2
        elif index == 2:
            length, outputs = 255, 1
        else:
            length, outputs = 3 + ((index * 11) % 37), 1 + (index % 3)
        prompt = tuple(
            1 if position == 0 else 3 + ((index * 43 + position * 17) % 256)
            for position in range(length)
        )
        if index % 2:
            sampling = SamplingSpec(
                mode="sample",
                temperature=0.85,
                top_k=31,
                top_p=0.92,
                actual_seed=0xA11CE + index,
            )
        else:
            sampling = SamplingSpec(actual_seed=index)
        cases.append((f"engine-differential-{index:02d}", prompt, outputs, sampling))
    assert len(cases) == 50
    assert len({(prompt, outputs, sampling.digest) for _, prompt, outputs, sampling in cases}) == 50
    return tuple(cases)


DIFFERENTIAL_CASES = _fifty_cases()
assert len(GOLDEN_CASES) == 20


@pytest.fixture(scope="module", autouse=True)
def _single_torch_thread() -> Generator[None, None, None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope="module")
def engine_pair() -> Generator[tuple[Engine, Engine], None, None]:
    none = Engine(kv_mode="none", project_root=ROOT)
    contiguous = Engine(kv_mode="contiguous", project_root=ROOT)
    yield none, contiguous
    assert none.close()
    assert contiguous.close()


def _request(
    engine: Engine,
    request_id: str,
    *,
    input_ids: tuple[int, ...] = (1, 100),
    max_new_tokens: int = 2,
    sampling: SamplingSpec | None = None,
    capture: Path | None = None,
    telemetry: str = "off",
) -> GenerationRequest:
    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        input_token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        sampling=sampling or SamplingSpec(),
        request_id=request_id,
        capture=capture,
        telemetry=telemetry,
    )


def _assert_resource_baseline(engine: Engine) -> None:
    assert engine.resource_status
    assert set(engine.resource_status.values()) == {0}


@pytest.mark.parametrize(
    "fixture_id,prompt,output_steps,sampling",
    DIFFERENTIAL_CASES,
    ids=[case[0] for case in DIFFERENTIAL_CASES],
)
def test_fifty_cache_modes_match_sampling_and_committed_state(
    engine_pair: tuple[Engine, Engine],
    fixture_id: str,
    prompt: tuple[int, ...],
    output_steps: int,
    sampling: SamplingSpec,
) -> None:
    none, contiguous = engine_pair
    request_id = fixture_id
    none_result = none.generate(_request(
        none,
        request_id,
        input_ids=prompt,
        max_new_tokens=output_steps,
        sampling=sampling,
    ))
    cached_result = contiguous.generate(_request(
        contiguous,
        request_id,
        input_ids=prompt,
        max_new_tokens=output_steps,
        sampling=sampling,
    ))

    assert cached_result.output_token_ids == none_result.output_token_ids
    assert cached_result.text == none_result.text
    assert cached_result.outcome == none_result.outcome == "FINISHED"
    assert cached_result.finish_reason == none_result.finish_reason
    assert cached_result.next_output_position == none_result.next_output_position == len(cached_result.output_token_ids)
    assert cached_result.rng_draws_consumed == none_result.rng_draws_consumed
    assert cached_result.committed_draw_addresses == none_result.committed_draw_addresses
    if sampling.mode == "greedy":
        assert cached_result.rng_draws_consumed == 0
        assert cached_result.committed_draw_addresses == ()
    else:
        assert cached_result.rng_draws_consumed == len(cached_result.output_token_ids)
        assert cached_result.committed_draw_addresses == tuple(
            (position, 0, 0) for position in range(len(cached_result.output_token_ids))
        )
    output_count = len(cached_result.output_token_ids)
    assert cached_result.query_tokens_per_step == (len(prompt), *(1 for _ in range(output_count - 1)))
    assert cached_result.kv_valid_tokens_per_step == tuple(len(prompt) + index for index in range(output_count))
    assert none_result.query_tokens_per_step == tuple(len(prompt) + index for index in range(output_count))
    assert none_result.kv_valid_tokens_per_step == (0,) * output_count
    _assert_resource_baseline(none)
    _assert_resource_baseline(contiguous)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case["fixture_id"] for case in GOLDEN_CASES])
def test_twenty_v01_golden_sequences_remain_exact_at_engine_boundary(
    engine_pair: tuple[Engine, Engine], case: dict[str, Any],
) -> None:
    contiguous = engine_pair[1]
    expected = tuple(int(token) for token in case["expected_tokens"])
    prompt = tuple(int(token) for token in case["input_ids"])
    result = contiguous.generate(_request(
        contiguous,
        f"streaming-{case['fixture_id']}",
        input_ids=prompt,
        max_new_tokens=len(expected),
    ))
    assert result.output_token_ids == expected
    assert result.finish_reason == "MAX_NEW_TOKENS"
    assert result.next_output_position == len(expected)
    assert result.rng_draws_consumed == 0
    assert result.committed_draw_addresses == ()
    assert result.query_tokens_per_step == (len(prompt), *(1 for _ in expected[1:]))
    assert result.kv_valid_tokens_per_step == tuple(len(prompt) + index for index in range(len(expected)))
    _assert_resource_baseline(contiguous)


def test_stream_order_eos_length_and_kv_output_lag() -> None:
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    length_chunks = tuple(engine.stream(_request(
        engine, "length", input_ids=(1, 100), max_new_tokens=2,
    )))
    assert isinstance(length_chunks[0], MetadataChunk)
    assert [chunk.output_index for chunk in length_chunks if isinstance(chunk, TokenChunk)] == [0, 1]
    length_terminal = length_chunks[-1]
    assert isinstance(length_terminal, TerminalChunk)
    assert length_terminal.outcome == "FINISHED" and length_terminal.finish_reason == "MAX_NEW_TOKENS"
    assert length_terminal.result.kv_valid_tokens_per_step == (2, 3)
    assert length_terminal.result.output_token_ids
    assert length_terminal.result.kv_valid_tokens_per_step[-1] == (
        len(length_terminal.result.input_token_ids) + len(length_terminal.result.output_token_ids) - 1
    )

    eos_chunks = tuple(engine.stream(_request(
        engine, "eos", input_ids=(2,), max_new_tokens=4,
    )))
    eos_tokens = [chunk for chunk in eos_chunks if isinstance(chunk, TokenChunk)]
    eos_terminal = eos_chunks[-1]
    assert [chunk.output_index for chunk in eos_tokens] == [0]
    assert eos_tokens[0].token_id == 2
    assert isinstance(eos_terminal, TerminalChunk)
    assert eos_terminal.finish_reason == "EOS"
    assert eos_terminal.result.output_token_ids == (2,)
    assert eos_terminal.result.kv_valid_tokens_per_step == (1,)
    terminal_kinds = {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}
    assert sum(event.kind in terminal_kinds for event in engine.last_events) == 1
    _assert_resource_baseline(engine)
    assert engine.close()


class _ScriptedExecutor(EagerExecutor):
    tokens: tuple[int, ...] = (231, 192, 163)  # UTF-8 bytes E4 BD A0 plus ByteTokenizer offset.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._script_position = 0

    def resolve(self, ticket: ExecutionTicket) -> tuple[float, ...]:
        original = super().resolve(ticket)
        token = self.tokens[min(self._script_position, len(self.tokens) - 1)]
        self._script_position += 1
        values = [-100.0] * len(original)
        values[token] = 100.0
        return tuple(values)


def test_incremental_utf8_deltas_and_terminal_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming_module, "EagerExecutor", _ScriptedExecutor)
    engine = Engine(kv_mode="contiguous", project_root=ROOT)

    complete = tuple(engine.stream(_request(
        engine, "utf8-complete", max_new_tokens=3,
    )))
    token_chunks = [chunk for chunk in complete if isinstance(chunk, TokenChunk)]
    terminal = complete[-1]
    assert [chunk.token_id for chunk in token_chunks] == [231, 192, 163]
    assert [chunk.text_delta for chunk in token_chunks] == ["", "", "你"]
    assert isinstance(terminal, TerminalChunk)
    assert terminal.final_text_delta == ""
    assert "".join([*(chunk.text_delta for chunk in token_chunks), terminal.final_text_delta]) == terminal.result.text == "你"

    partial = engine.stream(_request(engine, "utf8-cancel", max_new_tokens=3))
    assert isinstance(next(partial), MetadataChunk)
    first = next(partial)
    assert isinstance(first, TokenChunk) and first.token_id == 231 and first.text_delta == ""
    assert engine.cancel("utf8-cancel", "cancel-partial").won
    cancelled = next(partial)
    assert isinstance(cancelled, TerminalChunk)
    assert cancelled.outcome == "CANCELLED"
    assert cancelled.final_text_delta == "�"
    assert cancelled.result.text == "�"
    with pytest.raises(StopIteration):
        next(partial)
    _assert_resource_baseline(engine)
    assert engine.close()


def test_preaccept_rejection_duplicate_id_and_single_active_limit() -> None:
    engine = Engine(kv_mode="contiguous", max_context_tokens=8, project_root=ROOT)
    too_large = _request(engine, "too-large", input_ids=(1,) * 8, max_new_tokens=1)
    with pytest.raises(IllmError) as capacity:
        next(engine.stream(too_large))
    assert capacity.value.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert not any(event.kind is K.REQUEST_ACCEPTED for event in engine.last_events)
    _assert_resource_baseline(engine)

    accepted = _request(engine, "unique", input_ids=(1,), max_new_tokens=1)
    engine.generate(accepted)
    with pytest.raises(IllmError) as duplicate:
        engine.generate(accepted)
    assert duplicate.value.code is ErrorCode.INVALID_ARGUMENT

    active = engine.stream(_request(engine, "active", input_ids=(1,), max_new_tokens=1))
    assert isinstance(next(active), MetadataChunk)
    contender = engine.stream(_request(engine, "contender", input_ids=(1,), max_new_tokens=1))
    with pytest.raises(IllmError) as overloaded:
        next(contender)
    assert overloaded.value.code is ErrorCode.OVERLOADED
    assert overloaded.value.details == {"reason": "SINGLE_ACTIVE_LIMIT"}
    engine.cancel("active", "cleanup")
    assert isinstance(next(active), TerminalChunk)
    with pytest.raises(StopIteration):
        next(active)
    _assert_resource_baseline(engine)
    assert engine.close()


def test_cancel_after_metadata_never_launches_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    stream = engine.stream(_request(engine, "before-launch", max_new_tokens=3))
    assert isinstance(next(stream), MetadataChunk)
    runtime = engine._active
    assert runtime is not None

    def forbidden_launch(_context: tuple[int, ...]) -> ExecutionTicket:
        pytest.fail("cancel-before-launch must not launch model work")

    monkeypatch.setattr(runtime.executor, "launch", forbidden_launch)
    first = engine.cancel("before-launch", "cancel-1")
    repeated = engine.cancel("before-launch", "cancel-1")
    losing = engine.cancel("before-launch", "cancel-2")
    assert first.won and repeated.won and not losing.won
    chunks = tuple(stream)
    assert len(chunks) == 1 and isinstance(chunks[0], TerminalChunk)
    assert chunks[0].outcome == "CANCELLED"
    assert chunks[0].result.output_token_ids == ()
    after = engine.cancel("before-launch", "cancel-1")
    assert after.outcome == "CANCELLED" and after.terminal_ready and after.won
    assert not engine.cancel("before-launch", "cancel-2").won
    _assert_resource_baseline(engine)
    assert engine.close()


class _DelayedFence:
    def __init__(self, base: CompletionFence) -> None:
        self.base = base
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test delayed fence timed out")
        self.base.wait()

    def is_complete(self) -> bool | None:
        return self.base.is_complete() if self.release.is_set() else False

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded if self.release.is_set() else None


def _delayed_executor(fences: list[_DelayedFence]) -> type[EagerExecutor]:
    class DelayedExecutor(EagerExecutor):
        def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
            delayed = _DelayedFence(fence)
            fences.append(delayed)
            return delayed

    return DelayedExecutor


def test_cancel_after_launch_waits_for_fence_then_aborts_and_reclaims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fences: list[_DelayedFence] = []
    monkeypatch.setattr(streaming_module, "EagerExecutor", _delayed_executor(fences))
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    chunks: list[Any] = []
    failures: list[BaseException] = []

    def consume() -> None:
        try:
            chunks.extend(engine.stream(_request(engine, "after-launch", max_new_tokens=3)))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=consume)
    worker.start()
    while not fences:
        worker.join(timeout=0.001)
    assert fences[0].entered.wait(timeout=2)
    cancellation = engine.cancel("after-launch", "cancel-after-launch")
    assert cancellation.won and cancellation.outcome == "CANCELLED"
    status = engine.resource_status
    assert status["active_tickets"] == 1
    assert status["kv_write_leases"] == 1
    assert status["kv_staged_tokens"] == 2
    assert not any(isinstance(chunk, TokenChunk) for chunk in chunks)
    fences[0].release.set()
    worker.join(timeout=3)
    assert not worker.is_alive() and failures == []
    terminals = [chunk for chunk in chunks if isinstance(chunk, TerminalChunk)]
    assert len(terminals) == 1 and terminals[0].outcome == "CANCELLED"
    assert not any(isinstance(chunk, TokenChunk) for chunk in chunks)
    kinds = [event.kind for event in engine.last_events]
    assert kinds.index(K.REQUEST_CANCEL_REQUESTED) < kinds.index(K.KV_WRITE_ABORTED)
    assert kinds.index(K.KV_WRITE_ABORTED) < kinds.index(K.KV_PAGE_RELEASED)
    assert kinds.index(K.KV_PAGE_RELEASED) < kinds.index(K.REQUEST_CANCELLED)
    _assert_resource_baseline(engine)
    assert engine.close()


def test_finish_cancel_race_has_one_logical_and_physical_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fences: list[_DelayedFence] = []
    monkeypatch.setattr(streaming_module, "EagerExecutor", _delayed_executor(fences))
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    terminal_kinds = {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}

    for index in range(12):
        chunks: list[Any] = []
        errors: list[BaseException] = []
        request_id = f"race-{index}"

        def consume() -> None:
            try:
                chunks.extend(engine.stream(_request(engine, request_id, max_new_tokens=1)))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        generation = threading.Thread(target=consume)
        generation.start()
        deadline = len(fences)
        while len(fences) == deadline:
            generation.join(timeout=0.001)
        fence = fences[-1]
        assert fence.entered.wait(timeout=2)
        barrier = threading.Barrier(3)
        cancel_results: list[Any] = []

        def release() -> None:
            barrier.wait(timeout=2)
            fence.release.set()

        def cancel() -> None:
            barrier.wait(timeout=2)
            cancel_results.append(engine.cancel(request_id, f"race-cancel-{index}"))

        release_thread = threading.Thread(target=release)
        cancel_thread = threading.Thread(target=cancel)
        release_thread.start()
        cancel_thread.start()
        barrier.wait(timeout=2)
        release_thread.join(timeout=2)
        cancel_thread.join(timeout=2)
        generation.join(timeout=3)
        assert not generation.is_alive() and errors == []
        terminals = [chunk for chunk in chunks if isinstance(chunk, TerminalChunk)]
        assert len(terminals) == 1
        assert terminals[0].outcome in {"FINISHED", "CANCELLED"}
        assert len(cancel_results) == 1 and cancel_results[0].outcome == terminals[0].outcome
        assert sum(event.kind in terminal_kinds for event in engine.last_events) == 1
        assert len([chunk for chunk in chunks if isinstance(chunk, TokenChunk)]) == (
            1 if terminals[0].outcome == "FINISHED" else 0
        )
        _assert_resource_baseline(engine)
    assert engine.close()


def test_hundred_finish_cancel_cycles_return_to_baseline() -> None:
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    for index in range(100):
        request = _request(engine, f"cycle-{index}", max_new_tokens=1)
        if index % 2 == 0:
            assert engine.generate(request).outcome == "FINISHED"
        else:
            stream = engine.stream(request)
            assert isinstance(next(stream), MetadataChunk)
            assert engine.cancel(request.request_id, f"cancel-{index}").won
            terminal = next(stream)
            assert isinstance(terminal, TerminalChunk) and terminal.outcome == "CANCELLED"
            with pytest.raises(StopIteration):
                next(stream)
        _assert_resource_baseline(engine)
    assert engine.close()


def test_generator_close_runs_safe_cleanup_without_publishing_a_fake_chunk() -> None:
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    stream = engine.stream(_request(engine, "generator-close", max_new_tokens=3))
    assert isinstance(next(stream), MetadataChunk)
    stream.close()
    terminal = engine.cancel("generator-close", "after-close")
    assert terminal.outcome == "CANCELLED" and terminal.terminal_ready
    assert sum(event.kind is K.REQUEST_CANCELLED for event in engine.last_events) == 1
    _assert_resource_baseline(engine)

    finished = engine.stream(_request(engine, "generator-close-after-token", max_new_tokens=1))
    assert isinstance(next(finished), MetadataChunk)
    assert isinstance(next(finished), TokenChunk)
    finished.close()
    outcome = engine.cancel("generator-close-after-token", "too-late")
    assert outcome.outcome == "FINISHED" and outcome.terminal_ready and not outcome.won
    assert sum(event.kind is K.REQUEST_COMPLETED for event in engine.last_events) == 1
    _assert_resource_baseline(engine)
    assert engine.close()


class _NonFiniteExecutor(EagerExecutor):
    def resolve(self, ticket: ExecutionTicket) -> tuple[float, ...]:
        return (float("nan"), *super().resolve(ticket)[1:])


class _OomExecutor(EagerExecutor):
    def launch(self, context_ids: tuple[int, ...]) -> ExecutionTicket:
        del context_ids
        raise MemoryError("synthetic pre-dispatch OOM")


def test_known_numerical_and_oom_failures_emit_failed_terminal_and_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = EagerExecutor
    engine = Engine(kv_mode="contiguous", project_root=ROOT)

    monkeypatch.setattr(streaming_module, "EagerExecutor", _NonFiniteExecutor)
    numerical = engine.generate(_request(engine, "numerical", max_new_tokens=2))
    assert numerical.outcome == "FAILED"
    assert numerical.error_code == ErrorCode.NUMERICAL_ERROR.value
    assert numerical.output_token_ids == ()
    _assert_resource_baseline(engine)

    monkeypatch.setattr(streaming_module, "EagerExecutor", _OomExecutor)
    oom = engine.generate(_request(engine, "oom", max_new_tokens=2))
    assert oom.outcome == "FAILED"
    assert oom.error_code == ErrorCode.RESOURCE_EXHAUSTED.value
    assert oom.output_token_ids == ()
    _assert_resource_baseline(engine)

    monkeypatch.setattr(streaming_module, "EagerExecutor", original)
    assert engine.generate(_request(engine, "after-known-failures", max_new_tokens=1)).outcome == "FINISHED"
    _assert_resource_baseline(engine)
    assert engine.close()


class _UnknownFence:
    def wait(self) -> None:
        return None

    def is_complete(self) -> bool | None:
        return None

    @property
    def succeeded(self) -> bool | None:
        return None


class _UnknownFenceExecutor(EagerExecutor):
    def _wrap_fence(self, _fence: CompletionFence) -> CompletionFence:
        return _UnknownFence()


def test_unknown_fence_fail_stops_and_quarantines_without_fake_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming_module, "EagerExecutor", _UnknownFenceExecutor)
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    capture = tmp_path / "unknown-fence"
    with pytest.raises(IllmError) as unknown:
        tuple(engine.stream(_request(engine, "unknown", max_new_tokens=1, capture=capture)))
    assert unknown.value.code is ErrorCode.ENGINE_UNHEALTHY
    status = engine.resource_status
    assert status["quarantined_requests"] == 1
    assert status["reservations"] == 1
    assert status["active_tickets"] == 1
    assert status["kv_write_leases"] == 1
    assert status["kv_staged_tokens"] > 0
    terminal_kinds = {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}
    assert not any(event.kind in terminal_kinds for event in engine.last_events)
    logical = engine.cancel("unknown", "after-fail-stop")
    assert logical.outcome == "FAILED" and not logical.terminal_ready and not logical.won
    assert (capture / "INCOMPLETE").is_file()
    assert not (capture / "COMPLETE").exists()
    with pytest.raises(IllmError) as stopped:
        engine.generate(_request(engine, "blocked-after-unknown", max_new_tokens=1))
    assert stopped.value.code is ErrorCode.ENGINE_UNHEALTHY
    assert engine.close() is False


def test_required_capture_creation_and_write_failures_remain_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine(kv_mode="contiguous", project_root=ROOT)
    real_session = StreamingCaptureSession

    class CannotCreate:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("synthetic create failure")

    monkeypatch.setattr(streaming_module, "StreamingCaptureSession", CannotCreate)
    create_path = tmp_path / "create-failure"
    with pytest.raises(IllmError) as creation:
        next(engine.stream(_request(engine, "capture-create", capture=create_path)))
    assert creation.value.code is ErrorCode.CAPTURE_INCOMPLETE
    assert not any(event.kind is K.REQUEST_ACCEPTED for event in engine.last_events)
    _assert_resource_baseline(engine)

    class FailAfterAcceptance:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = real_session(*args, **kwargs)
            self.failed = False

        def emit(self, kind: K, payload: dict[str, Any]) -> Any:
            if kind is K.MODEL_FORWARD_STARTED and not self.failed:
                self.failed = True
                raise OSError("synthetic required write failure")
            return self.inner.emit(kind, payload)

        def mark_incomplete(self, reason: str, *, critical_event_drop_count: int = 0) -> dict[str, Any]:
            return self.inner.mark_incomplete(reason, critical_event_drop_count=critical_event_drop_count)

        def finalize(self) -> dict[str, Any]:
            return self.inner.finalize()

    monkeypatch.setattr(streaming_module, "StreamingCaptureSession", FailAfterAcceptance)
    write_path = tmp_path / "write-failure"
    result = engine.generate(_request(engine, "capture-write", max_new_tokens=1, capture=write_path))
    assert result.outcome == "FINISHED"
    assert result.capture_manifest_digest is None
    assert result.capture_error == "CAPTURE_WRITE_FAILED"
    assert (write_path / "INCOMPLETE").is_file()
    assert not (write_path / "COMPLETE").exists()
    assert result.timings["critical_capture_event_drop_count"] > 0
    _assert_resource_baseline(engine)
    assert engine.close()


def test_required_capture_failure_at_acceptance_rejects_before_model_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    class FailAcceptance(StreamingCaptureSession):
        def emit(self, kind: K, payload: Any) -> Any:
            if kind is K.REQUEST_ACCEPTED:
                raise OSError("synthetic preaccept capture failure")
            return super().emit(kind, payload)

    monkeypatch.setattr(streaming_module, "StreamingCaptureSession", FailAcceptance)
    engine = Engine(project_root=ROOT)
    try:
        with pytest.raises(IllmError) as raised:
            engine.generate(_request(engine, "reject-at-acceptance", capture=tmp_path / "failed-acceptance"))
        assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
        assert not any(event.kind is K.REQUEST_ACCEPTED for event in engine.last_events)
        _assert_resource_baseline(engine)
    finally:
        assert engine.close()


class _BlockingExporter:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def emit(self, _event: Any) -> None:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test exporter timed out")


def test_optional_exporter_stall_never_blocks_or_invalidates_inference() -> None:
    exporter = _BlockingExporter()
    engine = Engine(
        kv_mode="contiguous",
        project_root=ROOT,
        exporters=(exporter,),
        exporter_capacity=1,
    )
    results: list[Any] = []
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            results.append(engine.generate(_request(
                engine, "exporter-stall", max_new_tokens=4, telemetry="request",
            )))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    worker = threading.Thread(target=generate)
    worker.start()
    assert exporter.entered.wait(timeout=2)
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert errors == [] and len(results) == 1
    assert results[0].outcome == "FINISHED"
    assert results[0].capture_error is None
    assert engine.exporter_status["copy_drop_count"] > 0
    _assert_resource_baseline(engine)
    assert engine.close(timeout=0.0) is False
    exporter.release.set()
    assert engine.close(timeout=2.0) is True

"""V0.2 incremental numerics, KV ownership, and fence gates."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

import illm.executor.eager as eager_module
from illm.executor import CompletionFence, EagerExecutor, ImmediateFence
from illm.kv_cache import CONTIGUOUS_LAYOUT_ID, ContiguousKVStore, KVLayoutSpec, KVWriteLease
from illm.model import LoadedModel, load_model
from illm.model.spec import ModelError


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    value = load_model(project_root=ROOT)
    yield value
    torch.set_num_threads(original)


def _token(logits: tuple[float, ...]) -> int:
    return max(range(len(logits)), key=logits.__getitem__)


def _fifty_cases() -> tuple[tuple[str, tuple[int, ...], int], ...]:
    cases: list[tuple[str, tuple[int, ...], int]] = []
    for index in range(50):
        if index == 0:
            length, outputs = 1, 1
        elif index == 1:
            length, outputs = 2, 2
        elif index == 2:
            length, outputs = 255, 1
        else:
            length, outputs = 3 + ((index * 11) % 37), 1 + (index % 3)
        token_ids = tuple(1 if position == 0 else 3 + ((index * 43 + position * 17) % 256) for position in range(length))
        cases.append((f"differential-{index:02d}", token_ids, outputs))
    assert len(cases) == 50
    assert len({(tokens, outputs) for _, tokens, outputs in cases}) == 50
    return tuple(cases)


DIFFERENTIAL_CASES = _fifty_cases()
ORACLE_MANIFEST = json.loads(
    (ROOT / "fixtures" / "tiny-decoder" / "oracle-v1" / "oracle-manifest.json").read_text(encoding="utf-8")
)
GOLDEN_CASES = tuple(ORACLE_MANIFEST["greedy_cases"])
assert len(DIFFERENTIAL_CASES) == 50
assert len(GOLDEN_CASES) == 20
assert len({case["fixture_id"] for case in GOLDEN_CASES}) == 20


@pytest.mark.parametrize(
    "fixture_id,prompt,output_steps",
    DIFFERENTIAL_CASES,
    ids=[case[0] for case in DIFFERENTIAL_CASES],
)
def test_cache_off_on_case_matches_every_step(
    loaded: LoadedModel, fixture_id: str, prompt: tuple[int, ...], output_steps: int,
) -> None:
    none = EagerExecutor(loaded, "none", 256)
    cached = EagerExecutor(loaded, "contiguous", 256)
    context = list(prompt)
    try:
        for step in range(output_steps):
            none_ticket = none.launch(tuple(context))
            cached_ticket = cached.launch(tuple(context))
            none_logits = none.resolve(none_ticket)
            cached_logits = cached.resolve(cached_ticket)
            torch.testing.assert_close(
                torch.tensor(cached_logits),
                torch.tensor(none_logits),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda message: f"{fixture_id} step={step}: {message}",
            )
            selected = _token(none_logits)
            assert _token(cached_logits) == selected, (fixture_id, step)
            assert none_ticket.query_tokens == len(context)
            expected_query = len(prompt) if step == 0 else 1
            assert cached_ticket.query_tokens == expected_query
            assert cached_ticket.before_valid_tokens == len(context) - expected_query
            assert cached_ticket.after_valid_tokens == len(context)
            assert cached.resource_status.kv_valid_tokens == cached_ticket.before_valid_tokens
            assert cached.resource_status.kv_staged_tokens == expected_query
            assert none.commit(none_ticket) == 0
            assert cached.commit(cached_ticket) == len(context)
            context.append(selected)
    finally:
        none.close()
        cached.close()
    assert cached.resource_status == cached.resource_status.__class__(0, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case["fixture_id"] for case in GOLDEN_CASES])
def test_v01_golden_sequence_remains_exact_with_contiguous_kv(
    loaded: LoadedModel, case: dict[str, Any],
) -> None:
    fixture_id = str(case["fixture_id"])
    input_ids = tuple(int(token) for token in case["input_ids"])
    expected_tokens = tuple(int(token) for token in case["expected_tokens"])
    executor = EagerExecutor(loaded, "contiguous", 256)
    context = list(input_ids)
    try:
        for output_position, expected in enumerate(expected_tokens):
            ticket = executor.launch(tuple(context))
            logits = executor.resolve(ticket)
            assert _token(logits) == expected, (fixture_id, output_position)
            assert ticket.query_tokens == (len(input_ids) if output_position == 0 else 1)
            assert executor.commit(ticket) == len(context)
            context.append(expected)
    finally:
        executor.close()


def test_new_kv_layout_does_not_change_v01_model_identity(loaded: LoadedModel) -> None:
    old_spec_id = loaded.spec.spec_id
    old_layout_id = loaded.spec.kv_layout_id
    layout = KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=17)
    assert layout.layout_id == CONTIGUOUS_LAYOUT_ID
    assert layout.model_spec_id == old_spec_id
    assert loaded.spec.spec_id == old_spec_id
    assert loaded.spec.kv_layout_id == old_layout_id == "none-reference-full-context-v1"
    assert layout.tensor_shape == (loaded.spec.num_layers, loaded.spec.num_kv_heads, 17, loaded.spec.head_dim)
    assert layout.reserved_bytes == layout.bytes_per_token * 17


def test_none_mode_calls_unchanged_full_context_forward_each_step(loaded: LoadedModel) -> None:
    executor = EagerExecutor(loaded, "none", 8)
    with patch.object(loaded.reference, "forward", wraps=loaded.reference.forward) as forward:
        first = executor.launch((1, 7))
        token = _token(executor.resolve(first))
        executor.commit(first)
        second = executor.launch((1, 7, token))
        executor.resolve(second)
        executor.commit(second)
    assert [int(call.args[0].shape[1]) for call in forward.call_args_list] == [2, 3]
    executor.close()


def test_kv_validity_cannot_advance_before_resolve(loaded: LoadedModel) -> None:
    executor = EagerExecutor(loaded, "contiguous", 8)
    ticket = executor.launch((1, 9))
    with pytest.raises(ModelError, match="resolve before commit"):
        executor.commit(ticket)
    assert executor.resource_status.kv_valid_tokens == 0
    assert executor.resource_status.kv_staged_tokens == 2
    executor.abort(ticket)
    assert executor.resource_status.kv_valid_tokens == 0
    executor.close()


class _MutableFence:
    def __init__(self, state: bool | None, succeeded: bool | None) -> None:
        self.state = state
        self._succeeded = succeeded

    def wait(self) -> None:
        if self.state is not True:
            raise RuntimeError("not complete")

    def is_complete(self) -> bool | None:
        return self.state

    @property
    def succeeded(self) -> bool | None:
        return self._succeeded


def _stage_all(store: ContiguousKVStore, token_ids: tuple[int, ...]) -> KVWriteLease:
    lease = store.begin_append(token_ids)
    shape = (1, store.layout.num_kv_heads, len(token_ids), store.layout.head_dim)
    for layer in range(store.layout.num_layers):
        lease.stage_layer(layer, torch.full(shape, float(layer + 1)), torch.full(shape, float(layer + 2)))
    return lease


def test_append_is_staged_until_successful_fence_and_abort_requires_known_completion(loaded: LoadedModel) -> None:
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=4))
    first = _stage_all(store, (1, 2))
    assert store.valid_tokens == 0
    assert first.commit(ImmediateFence()) == 2
    assert store.committed_token_ids == (1, 2)
    second = _stage_all(store, (3,))
    pending = _MutableFence(False, None)
    with pytest.raises(ModelError, match="pending or unknown"):
        second.commit(pending)
    with pytest.raises(ModelError, match="cannot be reused"):
        second.abort(pending)
    assert store.valid_tokens == 2
    assert store.staged_tokens == 1
    pending.state, pending._succeeded = True, True
    second.abort(pending)
    assert store.valid_tokens == 2
    assert store.staged_tokens == 0
    store.close()


class _DelayedFence:
    def __init__(self, base: CompletionFence) -> None:
        self.base = base
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        self.release.wait(timeout=5)
        if not self.release.is_set():
            raise RuntimeError("test fence timed out")
        self.base.wait()

    def is_complete(self) -> bool | None:
        return self.base.is_complete() if self.release.is_set() else False

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded if self.release.is_set() else None


def test_executor_does_not_reuse_resources_before_delayed_fence(
    loaded: LoadedModel, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, "contiguous", 16)
    delayed: list[_DelayedFence] = []

    def wrap(base: CompletionFence) -> CompletionFence:
        value = _DelayedFence(base)
        delayed.append(value)
        return value

    monkeypatch.setattr(executor, "_wrap_fence", wrap)
    ticket = executor.launch((1, 4, 5))
    failures: list[BaseException] = []

    def abort() -> None:
        try:
            executor.abort(ticket)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    worker = threading.Thread(target=abort)
    worker.start()
    assert delayed[0].entered.wait(timeout=2)
    assert executor.resource_status.active_tickets == 1
    assert executor.resource_status.kv_write_leases == 1
    assert executor.resource_status.kv_staged_tokens == 3
    delayed[0].release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert failures == []
    assert executor.resource_status.active_tickets == 0
    assert executor.resource_status.kv_write_leases == 0
    assert executor.resource_status.kv_valid_tokens == 0
    executor.close()


class _UnknownThenKnownFence:
    def __init__(self, base: CompletionFence) -> None:
        self.base = base
        self.known = False

    def wait(self) -> None:
        if not self.known:
            raise RuntimeError("completion unavailable")
        self.base.wait()

    def is_complete(self) -> bool | None:
        return self.base.is_complete() if self.known else None

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded if self.known else None


def test_unknown_completion_fails_stop_without_reuse(
    loaded: LoadedModel, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, "contiguous", 8)
    wrappers: list[_UnknownThenKnownFence] = []

    def wrap(base: CompletionFence) -> CompletionFence:
        value = _UnknownThenKnownFence(base)
        wrappers.append(value)
        return value

    monkeypatch.setattr(executor, "_wrap_fence", wrap)
    ticket = executor.launch((1, 8))
    with pytest.raises(ModelError) as raised:
        executor.abort(ticket)
    assert raised.value.code == "EXECUTION_STATE_UNKNOWN"
    assert executor.resource_status.active_tickets == 1
    assert executor.resource_status.kv_write_leases == 1
    with pytest.raises(ModelError) as unhealthy:
        executor.launch((1,))
    assert unhealthy.value.code == "EXECUTION_STATE_UNKNOWN"
    wrappers[0].known = True
    executor.abort(ticket)
    executor.close()


def test_decode_prefix_capacity_and_ticket_ownership_fail_closed(loaded: LoadedModel) -> None:
    first = EagerExecutor(loaded, "contiguous", 4)
    second = EagerExecutor(loaded, "none", 4)
    ticket = first.launch((1, 7))
    first.resolve(ticket)
    assert first.commit(ticket) == 2
    with pytest.raises(ModelError, match="committed KV prefix"):
        first.launch((1, 8, 9))
    with pytest.raises(ModelError) as capacity:
        first.launch((1, 7, 8, 9, 10))
    assert capacity.value.code == "RESOURCE_EXHAUSTED"
    foreign = first.launch((1, 7, 8))
    with pytest.raises(ModelError, match="different executor"):
        second.resolve(foreign)
    first.abort(foreign)
    first.close()
    second.close()


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_post_dispatch_fence_error_keeps_original_error_after_safe_cleanup(
    loaded: LoadedModel,
    kv_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, kv_mode, 8)
    calls = 0

    def fail_then_prove_complete(_device: torch.device) -> CompletionFence:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic fence-record failure")
        return ImmediateFence()

    monkeypatch.setattr(eager_module, "record_completion_fence", fail_then_prove_complete)
    with pytest.raises(RuntimeError, match="synthetic fence-record failure"):
        executor.launch((1, 7))
    assert calls == 2
    status = executor.resource_status
    assert status.active_tickets == 0
    assert status.input_leases == status.output_leases == status.workspace_leases == 0
    assert status.kv_write_leases == status.kv_staged_tokens == 0
    executor.close()
    closed = executor.resource_status
    assert closed == closed.__class__(0, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
def test_post_dispatch_unknown_completion_quarantines_refs_and_close_fails(
    loaded: LoadedModel,
    kv_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = EagerExecutor(loaded, kv_mode, 8)

    def cannot_record(_device: torch.device) -> CompletionFence:
        raise RuntimeError("synthetic completion proof failure")

    monkeypatch.setattr(eager_module, "record_completion_fence", cannot_record)
    with pytest.raises(ModelError) as unknown:
        executor.launch((1, 7))
    assert unknown.value.code == "EXECUTION_STATE_UNKNOWN"
    assert isinstance(unknown.value.__cause__, RuntimeError)
    status = executor.resource_status
    assert status.active_tickets == 1
    assert status.input_leases == status.output_leases == status.workspace_leases == 1
    if kv_mode == "contiguous":
        assert status.kv_write_leases == 1
        assert status.kv_staged_tokens == 2
        assert status.kv_reserved_bytes > 0
    else:
        assert status.kv_write_leases == status.kv_staged_tokens == status.kv_reserved_bytes == 0
    with pytest.raises(ModelError) as stopped:
        executor.launch((1, 8))
    assert stopped.value.code == "EXECUTION_STATE_UNKNOWN"
    with pytest.raises(ModelError) as close:
        executor.close()
    assert close.value.code == "EXECUTION_STATE_UNKNOWN"
    assert executor.resource_status == status

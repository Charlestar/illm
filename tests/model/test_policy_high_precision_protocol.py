"""Frozen D-high comparisons: seven full fixtures and one RoPE-only fixture."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
import torch
from torch import Tensor

from illm.executor import ImmediateFence
from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
from illm.model import LoadedModel, load_model
from illm.model.high_precision import HighPrecisionOracle, OpRecord
from illm.model.policy import PolicyDecoder, PolicyOpRecord
from illm.numerical_policy import candidate_policy


ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("high-00", [3]),
    ("high-01", [4, 5]),
    ("high-02", [2]),
    ("high-03", [0] * 7),
    ("high-04", [1, 2, 258, 3, 257, 4, 256]),
    ("high-06", [3 + ((43 * 6 + 17 * p) % 256) for p in range(31)]),
    ("high-07", [3 + ((43 * 7 + 17 * p) % 256) for p in range(255)]),
]
Trace = dict[tuple[str, int], Tensor]


def _candidate_hook(trace: Trace) -> Callable[[PolicyOpRecord, Tensor], None]:
    def hook(record: PolicyOpRecord, tensor: Tensor) -> None:
        key = (record.name, record.position)
        assert key not in trace
        assert tensor.dtype == torch.float32
        assert tensor.shape == record.shape
        trace[key] = tensor.detach().clone()
    return hook


def _high_hook(trace: Trace) -> Callable[[OpRecord, Tensor], None]:
    def hook(record: OpRecord, tensor: Tensor) -> None:
        key = (record.name, record.position)
        assert key not in trace
        assert tensor.dtype == torch.float64
        assert tensor.shape == record.shape
        trace[key] = tensor.detach().clone()
    return hook


@pytest.fixture(scope="module")
def models() -> Generator[tuple[LoadedModel, PolicyDecoder, PolicyDecoder, HighPrecisionOracle], None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    loaded = load_model(project_root=ROOT)
    yield loaded, PolicyDecoder(loaded.reference, candidate_policy("fp32")), PolicyDecoder(loaded.reference, candidate_policy("fp32")), HighPrecisionOracle.from_fixture(ROOT)
    torch.set_num_threads(previous)


@pytest.mark.parametrize("case_id,tokens", CASES, ids=[case[0] for case in CASES])
def test_frozen_full_cases_against_independent_binary64_formulas(
    models: tuple[LoadedModel, PolicyDecoder, PolicyDecoder, HighPrecisionOracle], case_id: str, tokens: list[int],
) -> None:
    loaded, b, c, high = models
    expected: Trace = {}
    b_trace: Trace = {}
    c_trace: Trace = {}
    inputs = torch.tensor([tokens], dtype=torch.int64)
    high_logits = high.forward(inputs, op_hook=_high_hook(expected))
    full_logits = b.forward(inputs, op_hook=_candidate_hook(b_trace))
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256))
    rows: list[Tensor] = []
    try:
        for token in tokens:
            lease = store.begin_append((token,))
            rows.append(c.forward_incremental(torch.tensor([[token]], dtype=torch.int64), lease, op_hook=_candidate_hook(c_trace)))
            lease.commit(ImmediateFence())
    finally:
        store.close()
    assert set(expected) == set(b_trace) == set(c_trace), case_id
    for key, expected_tensor in expected.items():
        for actual in (b_trace[key], c_trace[key]):
            assert actual.shape == expected_tensor.shape, (case_id, key)
            assert torch.isfinite(actual).all() and torch.isfinite(expected_tensor).all()
            # Expected precision is retained; no cast-to-FP32 oracle laundering.
            torch.testing.assert_close(actual.double(), expected_tensor, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(full_logits.double(), high_logits, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(torch.cat(rows, dim=1).double(), high_logits, atol=1e-4, rtol=1e-3)


def test_high05_is_rope_only_and_does_not_forge_a_nonzero_request_origin(
    models: tuple[LoadedModel, PolicyDecoder, PolicyDecoder, HighPrecisionOracle],
) -> None:
    _, b, c, high = models
    query = torch.tensor([[[[(2 * h + d - 9) / 16 for d in range(8)] for _ in range(2)]
                           for h in range(4)]], dtype=torch.float32)
    key = torch.tensor([[[[(3 * h - d + 5) / 32 for d in range(8)] for _ in range(2)]
                         for h in range(2)]], dtype=torch.float32)
    expected_q, expected_k = high.rope(query, key, torch.tensor([[31, 32]], dtype=torch.int64))
    for decoder in (b, c):
        for index, position in enumerate((31, 32)):
            q, k = decoder._rope_position(query[:, :, index:index + 1, :], key[:, :, index:index + 1, :], position)
            torch.testing.assert_close(q.double(), expected_q[:, :, index:index + 1, :], atol=1e-4, rtol=1e-3)
            torch.testing.assert_close(k.double(), expected_k[:, :, index:index + 1, :], atol=1e-4, rtol=1e-3)

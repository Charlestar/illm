"""All CPU protocol rows against ordinary, independently loaded Transformers."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from safetensors.torch import load_file

from illm.executor import ImmediateFence
from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
from illm.model import LoadedModel, load_model
from illm.model.policy import PolicyDecoder, PolicyOpRecord
from illm.model.reference import canonical_op_names
from illm.model.validation import TransformersOracle, collect_transformers_ops, load_transformers_oracle
from illm.numerical_policy import candidate_policy
from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample


ROOT = Path(__file__).resolve().parents[2]
INPUTS = json.loads((ROOT / "docs/implementation/v0.2-policy-inputs-draft.json").read_text(encoding="utf-8"))
GOLDENS = json.loads((ROOT / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_text(encoding="utf-8"))["greedy_cases"]
CASES: list[dict[str, Any]] = list(INPUTS["suites"][1]["cases"]) + [
    {"case_id": case["fixture_id"], "input_token_ids": case["input_ids"], "max_new_tokens": 8,
     "sampling_spec": SamplingSpec(mode="greedy", actual_seed=7).to_dict()} for case in GOLDENS
]
OP_GOLDENS = json.loads((ROOT / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_text(encoding="utf-8"))["op_cases"]
RowTrace = dict[tuple[str, int], Tensor]


def _capture(trace: RowTrace) -> Callable[[PolicyOpRecord, Tensor], None]:
    def hook(record: PolicyOpRecord, tensor: Tensor) -> None:
        key = (record.name, record.position)
        assert key not in trace, key
        assert record.shape == tuple(tensor.shape)
        assert record.dtype == "torch.float32"
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()
        trace[key] = tensor.detach().clone()
    return hook


def _native_row(name: str, tensor: Tensor, position: int) -> Tensor:
    if name.endswith(("attention_scores", "attention_probabilities")):
        return tensor[:, :, position:position + 1, :position + 1]
    if tensor.ndim == 4:
        return tensor[:, :, position:position + 1, :]
    assert tensor.ndim == 3, (name, tensor.shape)
    return tensor[:, position, :]


@pytest.fixture(scope="module")
def models() -> Generator[tuple[LoadedModel, PolicyDecoder, PolicyDecoder, TransformersOracle], None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    loaded = load_model(project_root=ROOT)
    b = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    c = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    native = load_transformers_oracle(loaded)
    yield loaded, b, c, native
    torch.set_num_threads(previous)


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_every_cpu_query_op_and_cached_prefix_against_b_and_d_native(
    models: tuple[LoadedModel, PolicyDecoder, PolicyDecoder, TransformersOracle], case: dict[str, Any],
) -> None:
    loaded, b, c, native = models
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256))
    context = tuple(case["input_token_ids"])
    spec = SamplingSpec(**case["sampling_spec"])
    state = SamplingState(case["case_id"], spec.digest)
    try:
        for step in range(case["max_new_tokens"]):
            query = context[store.valid_tokens:]
            lease = store.begin_append(query)
            b_trace: RowTrace = {}
            c_trace: RowTrace = {}
            full_ids = torch.tensor([context], dtype=torch.int64)
            b_logits = b.forward(full_ids, op_hook=_capture(b_trace))
            c_logits = c.forward_incremental(torch.tensor([query], dtype=torch.int64), lease, op_hook=_capture(c_trace))
            native_trace = collect_transformers_ops(native, full_ids, loaded)
            names = canonical_op_names(loaded.spec)
            assert set(b_trace) == {(name, p) for name in names for p in range(len(context))}
            assert set(c_trace) == {(name, p) for name in names for p in range(lease.start, lease.end)}
            assert b_logits.shape == (1, len(context), loaded.spec.vocab_size)
            assert c_logits.shape == (1, len(query), loaded.spec.vocab_size)
            for (name, p), actual in c_trace.items():
                expected_b = b_trace[name, p]
                expected_d = _native_row(name, native_trace[name], p)
                assert actual.shape == expected_b.shape == expected_d.shape
                torch.testing.assert_close(actual, expected_b, atol=1e-4, rtol=1e-3)
                torch.testing.assert_close(actual, expected_d, atol=1e-4, rtol=1e-3)
            # B's recomputed prefix must also independently satisfy native D,
            # not just the final row or the same helper used by C.
            for (name, p), actual in b_trace.items():
                torch.testing.assert_close(actual, _native_row(name, native_trace[name], p), atol=1e-4, rtol=1e-3)
            for layer in range(loaded.spec.num_layers):
                prefix_k, prefix_v = lease.prefix(layer)
                for p in range(lease.start):
                    torch.testing.assert_close(prefix_k[:, :, p:p + 1, :], b_trace[f"layer.{layer}.rope_k", p], atol=1e-4, rtol=1e-3)
                    torch.testing.assert_close(prefix_v[:, :, p:p + 1, :], b_trace[f"layer.{layer}.v_projection", p], atol=1e-4, rtol=1e-3)
                probabilities = native_trace[f"layer.{layer}.attention_probabilities"]
                future = torch.triu(torch.ones(len(context), len(context), dtype=torch.bool), diagonal=1)
                assert torch.count_nonzero(probabilities[:, :, future]) == 0
            logits = tuple(float(value) for value in b_logits[0, -1].tolist())
            other_logits = tuple(float(value) for value in c_logits[0, -1].tolist())
            selected = prepare_sample(logits, spec, state)
            assert prepare_sample(other_logits, spec, state).token_id == selected.token_id
            assert lease.commit(ImmediateFence()) == len(context)
            state = commit_sample(state, selected, spec=spec, logits=logits)
            context += (selected.token_id,)
            if selected.token_id == loaded.spec.eos_token_id:
                break
            assert state.next_output_position == step + 1
    finally:
        store.close()
    assert store.valid_tokens == store.staged_tokens == store.allocated_bytes == 0


@pytest.mark.parametrize("case", OP_GOLDENS, ids=[case["fixture_id"] for case in OP_GOLDENS])
def test_frozen_op_authority_directly_not_via_transitive_oracle_agreement(
    models: tuple[LoadedModel, PolicyDecoder, PolicyDecoder, TransformersOracle], case: dict[str, Any],
) -> None:
    loaded, b, c, _ = models
    directory = ROOT / "fixtures/tiny-decoder/oracle-v1"
    from illm.canonical import sha256_file
    assert sha256_file(directory / case["file"]) == case["sha256"]
    expected = load_file(str(directory / case["file"]), device="cpu")
    if case["position_ids"] == [255]:
        # This frozen vector is a direct-model RoPE boundary, not a stream
        # starting at255 with an invented committed KV prefix.
        for decoder in (b, c):
            for layer in range(loaded.spec.num_layers):
                q, k = decoder._rope_position(expected[f"layer.{layer}.q_projection"],
                                              expected[f"layer.{layer}.k_projection"], 255)
                torch.testing.assert_close(q, expected[f"layer.{layer}.rope_q"], atol=1e-4, rtol=1e-3)
                torch.testing.assert_close(k, expected[f"layer.{layer}.rope_k"], atol=1e-4, rtol=1e-3)
        return
    tokens = tuple(case["input_ids"])
    trace_b: RowTrace = {}
    trace_c: RowTrace = {}
    b.forward(torch.tensor([tokens], dtype=torch.int64), op_hook=_capture(trace_b))
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256))
    try:
        lease = store.begin_append(tokens)
        c.forward_incremental(torch.tensor([tokens], dtype=torch.int64), lease, op_hook=_capture(trace_c))
        lease.commit(ImmediateFence())
    finally:
        store.close()
    for key, actual in trace_b.items():
        name, p = key
        torch.testing.assert_close(actual, _native_row(name, expected[name], p), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(trace_c[key], _native_row(name, expected[name], p), atol=1e-4, rtol=1e-3)

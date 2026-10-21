"""Executable counterexamples for the frozen V0.2 numerical policy."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from illm.executor import ImmediateFence
from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
from illm.model import LoadedModel, load_model
from illm.model.policy import PolicyDecoder, PolicyOpRecord
from illm.model.reference import ReferenceDecoder, TORCH_DTYPES
from illm.model.spec import ModelError
from illm.model.validation import collect_transformers_ops, load_transformers_oracle
from illm.numerical_policy import candidate_policy


ROOT = Path(__file__).resolve().parents[2]
Trace = list[tuple[PolicyOpRecord, Tensor]]


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield load_model(project_root=ROOT)
    torch.set_num_threads(previous)


def _reference_for_dtype(loaded: LoadedModel, dtype: str) -> ReferenceDecoder:
    weights = {
        name: loaded.reference.canonical_weight(name).detach().clone()
        for name in loaded.spec.tensor_shapes()
    }
    return ReferenceDecoder(replace(loaded.spec, runtime_dtype=dtype), weights)


def _capture(trace: Trace) -> Callable[[PolicyOpRecord, Tensor], None]:
    def hook(record: PolicyOpRecord, tensor: Tensor) -> None:
        trace.append((record, tensor.detach().clone()))

    return hook


def _run(decoder: PolicyDecoder, ids: tuple[int, ...]) -> Trace:
    trace: Trace = []
    decoder(torch.tensor([ids], dtype=torch.int64), op_hook=_capture(trace))
    return trace


def _first_divergence(left: Trace, right: Trace) -> tuple[str, int] | None:
    assert len(left) == len(right)
    for (left_record, left_tensor), (right_record, right_tensor) in zip(left, right, strict=True):
        assert left_record == right_record
        if not torch.equal(left_tensor, right_tensor):
            return left_record.name, left_record.position
    return None


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    identity = f"{value.dtype}:{tuple(value.shape)}:".encode("ascii")
    return hashlib.sha256(identity + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _native_row(name: str, tensor: Tensor, position: int) -> Tensor:
    if name.endswith(("attention_scores", "attention_probabilities")):
        return tensor[:, :, position : position + 1, : position + 1]
    if tensor.ndim == 4:
        return tensor[:, :, position : position + 1, :]
    assert tensor.ndim == 3
    return tensor[:, position, :]


def test_transposed_square_q_weight_first_differs_at_q_projection(loaded: LoadedModel) -> None:
    baseline = _run(PolicyDecoder(loaded.reference, candidate_policy("fp32")), (3, 4))
    mutated = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    weight = mutated.canonical_sidecar("model.layers.0.self_attn.q_proj.weight")
    before = _tensor_digest(weight)
    transposed = weight.transpose(0, 1).contiguous()
    assert weight.shape[0] == weight.shape[1]
    assert not torch.equal(weight, transposed)
    with torch.no_grad():
        weight.copy_(transposed)
    assert _tensor_digest(weight) != before

    faulted = _run(mutated, (3, 4))
    assert _first_divergence(baseline, faulted) == ("layer.0.q_projection", 0)


def test_shifted_decode_position_first_differs_at_rope(loaded: LoadedModel) -> None:
    baseline = _run(PolicyDecoder(loaded.reference, candidate_policy("fp32")), (3, 4))
    mutated = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    original = mutated._rope_position

    def shifted(query: Tensor, key: Tensor, position: int) -> tuple[Tensor, Tensor]:
        return original(query, key, position + 1)

    mutated._rope_position = shifted  # type: ignore[method-assign]
    faulted = _run(mutated, (3, 4))
    assert _first_divergence(baseline, faulted) == ("layer.0.rope_q", 0)


def test_wrong_gqa_group_first_differs_at_scores_and_changes_context(loaded: LoadedModel) -> None:
    baseline = _run(PolicyDecoder(loaded.reference, candidate_policy("fp32")), (3, 4))
    mutated = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    original = mutated._attention_position

    def swapped(
        query: Tensor,
        key_prefix: Tensor,
        value_prefix: Tensor,
        *,
        position: int,
        layer: int,
        hook: Callable[[PolicyOpRecord, Tensor], None] | None,
    ) -> Tensor:
        return original(
            query,
            key_prefix.flip(1).contiguous(),
            value_prefix.flip(1).contiguous(),
            position=position,
            layer=layer,
            hook=hook,
        )

    mutated._attention_position = swapped  # type: ignore[method-assign]
    faulted = _run(mutated, (3, 4))
    assert _first_divergence(baseline, faulted) == ("layer.0.attention_scores", 0)
    baseline_map = {(record.name, record.position): tensor for record, tensor in baseline}
    faulted_map = {(record.name, record.position): tensor for record, tensor in faulted}
    assert not torch.equal(
        baseline_map[("layer.0.attention_context", 0)],
        faulted_map[("layer.0.attention_context", 0)],
    )


def test_future_visible_prefix_is_rejected_by_exact_support_shape(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    query = torch.zeros(
        (1, loaded.spec.num_attention_heads, loaded.spec.head_dim),
        dtype=TORCH_DTYPES[loaded.spec.runtime_dtype],
    )
    future_prefix = torch.zeros(
        (1, loaded.spec.num_kv_heads, 2, loaded.spec.head_dim),
        dtype=TORCH_DTYPES[loaded.spec.runtime_dtype],
    )
    with pytest.raises(ModelError, match="prefix shape differs from absolute position"):
        decoder._attention_position(
            query,
            future_prefix,
            future_prefix.clone(),
            position=0,
            layer=0,
            hook=None,
        )


def test_corrupted_actual_cached_key_is_caught_against_b_and_native(loaded: LoadedModel) -> None:
    b = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    c = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    native = load_transformers_oracle(loaded)
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=8))
    decode_lease = None
    try:
        prompt = (3, 4)
        prefill = store.begin_append(prompt)
        c.forward_incremental(torch.tensor([prompt], dtype=torch.int64), prefill)
        assert prefill.commit(ImmediateFence()) == len(prompt)

        context = (*prompt, 5)
        decode_lease = store.begin_append((5,))
        actual_key, _ = decode_lease.prefix(0)
        before = _tensor_digest(actual_key)
        with torch.no_grad():
            actual_key[0, 0, 0, 0] += 1.0
        assert _tensor_digest(actual_key) != before

        b_trace: Trace = []
        c_trace: Trace = []
        full_ids = torch.tensor([context], dtype=torch.int64)
        b.forward(full_ids, op_hook=_capture(b_trace))
        c.forward_incremental(
            torch.tensor([[5]], dtype=torch.int64),
            decode_lease,
            op_hook=_capture(c_trace),
        )
        native_trace = collect_transformers_ops(native, full_ids, loaded)
        position = len(prompt)
        b_map = {(record.name, record.position): tensor for record, tensor in b_trace}
        c_map = {(record.name, record.position): tensor for record, tensor in c_trace}
        score_name = "layer.0.attention_scores"
        expected_b = b_map[(score_name, position)]
        expected_native = _native_row(score_name, native_trace[score_name], position)
        torch.testing.assert_close(expected_b, expected_native, atol=1e-4, rtol=1e-3)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                c_map[(score_name, position)],
                expected_native,
                atol=1e-4,
                rtol=1e-3,
            )
        first = next(
            (record.name, record.position)
            for record, tensor in c_trace
            if not torch.equal(tensor, b_map[(record.name, record.position)])
        )
        assert first == (score_name, position)
    finally:
        if decode_lease is not None and decode_lease.state == "staged":
            decode_lease.abort(ImmediateFence())
        store.close()


@pytest.mark.parametrize("mutation", ["fused_bias", "storage_linear"], ids=["fused-bias", "early-cast"])
def test_linear_conformance_rejects_real_fused_bias_and_cast_mutations(
    loaded: LoadedModel,
    mutation: str,
) -> None:
    dtype = "fp32" if mutation == "fused_bias" else "bf16"
    reference = _reference_for_dtype(loaded, dtype)
    control = PolicyDecoder(reference, candidate_policy(dtype))
    mutated = PolicyDecoder(reference, candidate_policy(dtype))
    original_linear = F.linear

    def conformance_linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        if bias is not None:
            raise AssertionError("policy conformance rejected fused bias")
        if input.dtype != torch.float32 or weight.dtype != torch.float32:
            raise AssertionError("policy conformance rejected non-FP32 contraction")
        return original_linear(input, weight, bias)

    # The guard accepts the unmodified implementation during a complete forward.
    with patch("torch.nn.functional.linear", side_effect=conformance_linear):
        control(torch.tensor([[3, 4]], dtype=torch.int64))

    storage_dtype = TORCH_DTYPES[dtype]

    def wrong_linear_row(hidden: Tensor, name: str, *, bias: bool = False) -> Tensor:
        row = hidden.contiguous()
        weight = mutated.canonical_sidecar(name + ".weight")
        if mutation == "fused_bias":
            fused = mutated.canonical_sidecar(name + ".bias") if bias else None
            return F.linear(row.float(), weight, fused).to(dtype=storage_dtype)
        storage_row = row.to(dtype=storage_dtype)
        storage_weight = weight.to(dtype=storage_dtype)
        result = F.linear(storage_row, storage_weight, None)
        if bias:
            result = result + mutated.canonical_sidecar(name + ".bias").to(dtype=storage_dtype)
        return result

    mutated._linear_row = wrong_linear_row  # type: ignore[method-assign]
    expected = "fused bias" if mutation == "fused_bias" else "non-FP32 contraction"
    with patch("torch.nn.functional.linear", side_effect=conformance_linear):
        with pytest.raises(AssertionError, match=expected):
            mutated(torch.tensor([[3, 4]], dtype=torch.int64))

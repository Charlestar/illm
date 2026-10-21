"""CPU contract tests for the V0.2 rowwise numerical policy decoder."""

from __future__ import annotations

import math
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
from illm.model.policy import PolicyDecoder, PolicyOpHook, PolicyOpRecord
from illm.model.reference import ReferenceDecoder, TORCH_DTYPES, canonical_op_names
from illm.model.spec import ModelError
from illm.numerical_policy import candidate_policy, legacy_policy


ROOT = Path(__file__).resolve().parents[2]
Trace = dict[tuple[str, int], tuple[PolicyOpRecord, Tensor]]


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    value = load_model(project_root=ROOT)
    yield value


def _reference_for_dtype(loaded: LoadedModel, dtype: str) -> ReferenceDecoder:
    weights = {
        name: loaded.reference.canonical_weight(name).detach().clone()
        for name in loaded.spec.tensor_shapes()
    }
    return ReferenceDecoder(replace(loaded.spec, runtime_dtype=dtype), weights)


def _capture(trace: Trace) -> PolicyOpHook:
    def hook(record: PolicyOpRecord, tensor: Tensor) -> None:
        key = (record.name, record.position)
        assert key not in trace
        assert record.shape == tuple(tensor.shape)
        assert record.dtype == str(tensor.dtype)
        assert record.device == str(tensor.device)
        trace[key] = (record, tensor.detach().clone())

    return hook


def _expected_sidecar_bytes(reference: ReferenceDecoder) -> int:
    return sum(math.prod(shape) * 4 for shape in reference.spec.tensor_shapes().values())


def test_policy_sidecars_are_unique_fp32_storage_derived_and_tied(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    assert decoder.sidecar_bytes == _expected_sidecar_bytes(loaded.reference)
    for name in loaded.spec.tensor_shapes():
        sidecar = decoder.canonical_sidecar(name)
        storage = loaded.reference.canonical_weight(name)
        assert sidecar.dtype == torch.float32
        assert sidecar.device.type == "cpu"
        assert sidecar.is_contiguous()
        assert sidecar.data_ptr() != storage.data_ptr()
        assert torch.equal(sidecar, storage.float())
    assert loaded.spec.tie_word_embeddings
    assert (
        decoder.canonical_sidecar("lm_head.weight").data_ptr()
        == decoder.canonical_sidecar("model.embed_tokens.weight").data_ptr()
    )
    pointers = {name: decoder.canonical_sidecar(name).data_ptr() for name in loaded.spec.tensor_shapes()}
    decoder(torch.tensor([[3, 4]], dtype=torch.int64))
    assert pointers == {name: decoder.canonical_sidecar(name).data_ptr() for name in loaded.spec.tensor_shapes()}


def test_constructor_rejects_legacy_mismatch_and_non_cpu_before_sidecars(loaded: LoadedModel) -> None:
    with pytest.raises(ModelError, match="candidate policy family"):
        PolicyDecoder(loaded.reference, legacy_policy("fp32"))
    with pytest.raises(ModelError, match="storage dtype"):
        PolicyDecoder(loaded.reference, candidate_policy("bf16"))

    frozen = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    with pytest.raises(ModelError, match="placement and dtype are frozen"):
        frozen.to(device="cuda")

    meta = torch.empty((1,), device="meta")
    with patch.object(loaded.reference, "canonical_weight", return_value=meta):
        with pytest.raises(ModelError, match="CPU implementation rejects non-CPU"):
            PolicyDecoder(loaded.reference, candidate_policy("fp32"))


def test_full_trace_is_unique_per_position_ragged_and_all_position(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    trace: Trace = {}
    inputs = torch.tensor([[3, 4, 5]], dtype=torch.int64)
    logits = decoder(inputs, op_hook=_capture(trace))
    assert tuple(logits.shape) == (1, 3, loaded.spec.vocab_size)
    expected_names = set(canonical_op_names(loaded.spec))
    assert len(trace) == len(expected_names) * 3
    for position in range(3):
        assert {name for name, item_position in trace if item_position == position} == expected_names
        assert trace[("embedding", position)][0].shape == (1, loaded.spec.hidden_size)
        assert trace[("final_norm", position)][0].shape == (1, loaded.spec.hidden_size)
        assert trace[("logits", position)][0].shape == (1, loaded.spec.vocab_size)
        for layer in range(loaded.spec.num_layers):
            prefix = f"layer.{layer}"
            assert trace[(prefix + ".q_projection", position)][0].shape == (
                1,
                loaded.spec.num_attention_heads,
                1,
                loaded.spec.head_dim,
            )
            assert trace[(prefix + ".k_projection", position)][0].shape == (
                1,
                loaded.spec.num_kv_heads,
                1,
                loaded.spec.head_dim,
            )
            assert trace[(prefix + ".v_projection", position)][0].shape == (
                1,
                loaded.spec.num_kv_heads,
                1,
                loaded.spec.head_dim,
            )
            for name in ("attention_scores", "attention_probabilities"):
                assert trace[(prefix + "." + name, position)][0].shape == (
                    1,
                    loaded.spec.num_attention_heads,
                    1,
                    position + 1,
                )
            assert trace[(prefix + ".attention_context", position)][0].shape == (
                1,
                loaded.spec.num_attention_heads,
                1,
                loaded.spec.head_dim,
            )
    assert torch.equal(logits[0, 0], trace[("logits", 0)][1][0])
    assert torch.equal(logits[0, 1], trace[("logits", 1)][1][0])
    assert torch.equal(logits[0, 2], trace[("logits", 2)][1][0])


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "fp16"])
def test_full_and_incremental_share_exact_policy_arithmetic(
    loaded: LoadedModel,
    dtype: str,
) -> None:
    reference = _reference_for_dtype(loaded, dtype)
    decoder = PolicyDecoder(reference, candidate_policy(dtype))
    store = ContiguousKVStore(KVLayoutSpec.for_model(reference.spec, device="cpu", capacity=16))
    context = [3, 4, 5]
    try:
        inputs = torch.tensor([context], dtype=torch.int64)
        full = decoder(inputs)
        prefill = store.begin_append(tuple(context))
        cached = decoder.forward_incremental(inputs, prefill)
        assert full.dtype == TORCH_DTYPES[dtype]
        assert torch.equal(cached, full)
        assert prefill.commit(ImmediateFence()) == len(context)

        selected = int(full[0, -1].float().argmax().item())
        context.append(selected)
        full_decode = decoder(torch.tensor([context], dtype=torch.int64))
        decode = store.begin_append((selected,))
        trace: Trace = {}
        cached_decode = decoder.forward_incremental(
            torch.tensor([[selected]], dtype=torch.int64),
            decode,
            op_hook=_capture(trace),
        )
        assert tuple(cached_decode.shape) == (1, 1, loaded.spec.vocab_size)
        assert torch.equal(cached_decode[0, 0], full_decode[0, -1])
        assert decode.commit(ImmediateFence()) == len(context)
        for layer in range(loaded.spec.num_layers):
            score = trace[(f"layer.{layer}.attention_scores", len(context) - 1)][0]
            assert score.shape == (1, loaded.spec.num_attention_heads, 1, len(context))
    finally:
        store.close()


def test_every_linear_is_one_fp32_row_without_fused_bias(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    original: Callable[..., Tensor] = F.linear
    with patch("torch.nn.functional.linear", wraps=original) as linear:
        decoder(torch.tensor([[3, 4, 5]], dtype=torch.int64))
    calls_per_position = loaded.spec.num_layers * 7 + 1
    assert linear.call_count == 3 * calls_per_position
    for call in linear.call_args_list:
        hidden, weight, bias = call.args
        assert isinstance(hidden, Tensor) and isinstance(weight, Tensor)
        assert tuple(hidden.shape)[0] == 1 and hidden.ndim == 2 and hidden.is_contiguous()
        assert weight.ndim == 2 and weight.is_contiguous()
        assert hidden.dtype == weight.dtype == torch.float32
        assert bias is None


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_frozen_norm_rope_attention_and_storage_pointwise_boundaries(
    loaded: LoadedModel,
    dtype: str,
) -> None:
    reference = _reference_for_dtype(loaded, dtype)
    decoder = PolicyDecoder(reference, candidate_policy(dtype))
    trace: Trace = {}
    decoder(torch.tensor([[3]], dtype=torch.int64), op_hook=_capture(trace))
    storage_dtype = TORCH_DTYPES[dtype]
    embedding = trace[("embedding", 0)][1]

    promoted = embedding.float()
    variance = (promoted * promoted).mean(dim=-1, keepdim=True)
    expected_norm = (
        promoted
        * torch.rsqrt(variance + torch.tensor(reference.spec.rms_norm_eps, dtype=torch.float32))
    ).to(storage_dtype) * reference.canonical_weight("model.layers.0.input_layernorm.weight")
    actual_norm = trace[("layer.0.input_norm", 0)][1]
    assert torch.equal(actual_norm, expected_norm)

    expected_query = (
        F.linear(
            expected_norm.contiguous().float(),
            decoder.canonical_sidecar("model.layers.0.self_attn.q_proj.weight"),
            None,
        )
        + decoder.canonical_sidecar("model.layers.0.self_attn.q_proj.bias")
    ).to(storage_dtype).view(1, reference.spec.num_attention_heads, 1, reference.spec.head_dim)
    query = trace[("layer.0.q_projection", 0)][1]
    assert torch.equal(query, expected_query)

    indices = torch.arange(0, reference.spec.head_dim, 2, dtype=torch.float32)
    inverse = 1.0 / torch.pow(
        torch.tensor(reference.spec.rope_theta, dtype=torch.float32),
        indices / reference.spec.head_dim,
    )
    angles = torch.cat((inverse * torch.tensor(0.0), inverse * torch.tensor(0.0)))
    cosine, sine = torch.cos(angles).to(storage_dtype), torch.sin(angles).to(storage_dtype)
    query_row = query.squeeze(2)
    half = reference.spec.head_dim // 2
    rotated = torch.cat((-query_row[..., half:], query_row[..., :half]), dim=-1)
    expected_rope = (query_row * cosine + rotated * sine).unsqueeze(2)
    assert torch.equal(trace[("layer.0.rope_q", 0)][1], expected_rope)

    scores: list[Tensor] = []
    rope_query = trace[("layer.0.rope_q", 0)][1]
    rope_key = trace[("layer.0.rope_k", 0)][1]
    groups = reference.spec.num_attention_heads // reference.spec.num_kv_heads
    scale = torch.tensor(reference.spec.head_dim ** -0.5, dtype=torch.float32)
    for head in range(reference.spec.num_attention_heads):
        kv_head = head // groups
        scores.append(
            (
                rope_query[:, head, 0, :].contiguous().float()
                @ rope_key[:, kv_head, 0, :].contiguous().float().transpose(0, 1).contiguous()
                * scale
            ).to(storage_dtype)
        )
    expected_scores = torch.stack(scores, dim=1).unsqueeze(2)
    assert torch.equal(trace[("layer.0.attention_scores", 0)][1], expected_scores)
    assert torch.equal(
        trace[("layer.0.attention_probabilities", 0)][1],
        torch.ones_like(expected_scores),
    )
    gate = trace[("layer.0.mlp_gate", 0)][1]
    activated = trace[("layer.0.mlp_silu", 0)][1]
    up = trace[("layer.0.mlp_up", 0)][1]
    product = trace[("layer.0.mlp_product", 0)][1]
    assert torch.equal(activated, F.silu(gate))
    assert activated.dtype == storage_dtype
    assert torch.equal(product, activated * up)
    assert product.dtype == storage_dtype


def test_execution_flags_and_outer_autocast_restore_even_when_hook_raises(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    original_threads = torch.get_num_threads()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_precision = torch.get_float32_matmul_precision()
    outside_threads = 2 if original_threads != 2 else 3
    observed: list[tuple[int, bool, bool, str, bool]] = []

    def observe(_record: PolicyOpRecord, _tensor: Tensor) -> None:
        if not observed:
            observed.append(
                (
                    torch.get_num_threads(),
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.get_float32_matmul_precision(),
                    torch.is_autocast_enabled("cpu"),
                )
            )

    try:
        torch.set_num_threads(outside_threads)
        torch.use_deterministic_algorithms(False, warn_only=False)
        torch.set_float32_matmul_precision("medium")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")
            decoder(torch.tensor([[3]], dtype=torch.int64), op_hook=observe)
            assert torch.is_autocast_enabled("cpu")
        assert observed == [(1, True, False, "highest", False)]
        assert torch.get_num_threads() == outside_threads
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.get_float32_matmul_precision() == "medium"

        def fail(_record: PolicyOpRecord, _tensor: Tensor) -> None:
            raise RuntimeError("synthetic hook failure")

        with pytest.raises(RuntimeError, match="synthetic hook failure"):
            decoder(torch.tensor([[3]], dtype=torch.int64), op_hook=fail)
        assert torch.get_num_threads() == outside_threads
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)
        torch.set_num_threads(original_threads)


def test_incremental_lease_identity_sequence_and_prefix_fail_closed(
    loaded: LoadedModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=8))
    try:
        wrong = store.begin_append((3,))
        with pytest.raises(ModelError, match="differ from the KV append lease"):
            decoder.forward_incremental(torch.tensor([[4]], dtype=torch.int64), wrong)
        wrong.abort(ImmediateFence())

        prefill = store.begin_append((3,))
        decoder.forward_incremental(torch.tensor([[3]], dtype=torch.int64), prefill)
        prefill.commit(ImmediateFence())

        multi = store.begin_append((4, 5))
        with pytest.raises(ModelError, match="one query token"):
            decoder.forward_incremental(torch.tensor([[4, 5]], dtype=torch.int64), multi)
        multi.abort(ImmediateFence())

        corrupt = store.begin_append((4,))
        original_prefix = corrupt.prefix

        def wrong_prefix(layer: int) -> tuple[Tensor, Tensor]:
            key, value = original_prefix(layer)
            return key[:, :, :0, :], value[:, :, :0, :]

        monkeypatch.setattr(corrupt, "prefix", wrong_prefix)
        with pytest.raises(ModelError, match="prefix shape differs"):
            decoder.forward_incremental(torch.tensor([[4]], dtype=torch.int64), corrupt)
        monkeypatch.undo()
        corrupt.abort(ImmediateFence())
    finally:
        store.close()


def test_invalid_input_and_nonfinite_sidecar_fail_closed(loaded: LoadedModel) -> None:
    decoder = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    for inputs in (
        torch.tensor([[loaded.spec.vocab_size]], dtype=torch.int64),
        torch.tensor([[-1]], dtype=torch.int64),
        torch.zeros((2, 1), dtype=torch.int64),
        torch.zeros((1, 0), dtype=torch.int64),
        torch.zeros((1, loaded.spec.max_position_embeddings + 1), dtype=torch.int64),
        torch.tensor([[1.0]]),
    ):
        with pytest.raises(ModelError):
            decoder(inputs)

    broken = PolicyDecoder(loaded.reference, candidate_policy("fp32"))
    with torch.no_grad():
        broken.canonical_sidecar("model.layers.0.self_attn.q_proj.weight")[0, 0] = float("nan")
    with pytest.raises(ModelError) as raised:
        broken(torch.tensor([[3]], dtype=torch.int64))
    assert raised.value.code == "NUMERICAL_ERROR"

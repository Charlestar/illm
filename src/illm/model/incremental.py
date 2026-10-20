"""Inspectable Qwen2 prefill/decode path backed by contiguous K/V."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from illm.kv_cache import KVLayoutSpec, KVWriteLease
from illm.model.reference import OpHook, ReferenceDecoder, _validate_execution_mode
from illm.model.spec import ModelError


class IncrementalDecoder:
    """Reuse one V0.1 reference decoder's immutable weights with explicit K/V.

    The first call is an unchunked prefill.  Every later call is exactly one
    decode query token; the executor enforces that sequencing contract.
    """

    def __init__(self, reference: ReferenceDecoder, layout: KVLayoutSpec) -> None:
        if layout.model_spec_id != reference.spec.spec_id:
            raise ModelError("INVALID_MODEL_CONFIG", "KV layout is bound to a different model spec")
        if (
            layout.dtype != reference.spec.runtime_dtype
            or layout.num_layers != reference.spec.num_layers
            or layout.num_kv_heads != reference.spec.num_kv_heads
            or layout.head_dim != reference.spec.head_dim
        ):
            raise ModelError("INVALID_MODEL_CONFIG", "KV layout dimensions or dtype differ from model spec")
        embedding = reference.canonical_weight("model.embed_tokens.weight")
        if str(embedding.device) != layout.device:
            raise ModelError("INVALID_ARGUMENT", "KV layout and reference model device differ")
        self.reference = reference
        self.layout = layout

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, lease: KVWriteLease, *, op_hook: OpHook | None = None) -> Tensor:
        reference = self.reference
        spec = reference.spec
        embedding = reference.canonical_weight("model.embed_tokens.weight")
        _validate_execution_mode(embedding.device.type)
        if input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ModelError("INVALID_ARGUMENT", "incremental input_ids must be int64 [1,Q]")
        query_tokens = input_ids.shape[1]
        if query_tokens != lease.count or query_tokens < 1:
            raise ModelError("INVALID_ARGUMENT", "incremental query length differs from KV append lease")
        if input_ids.device != embedding.device:
            raise ModelError("INVALID_ARGUMENT", "input IDs and model must be on the same device")
        if bool(((input_ids < 0) | (input_ids >= spec.vocab_size)).any().item()):
            raise ModelError("INVALID_ARGUMENT", "input token is outside vocabulary")
        if lease.end > self.layout.capacity_tokens or lease.end > spec.max_position_embeddings:
            raise ModelError("RESOURCE_EXHAUSTED", "incremental positions exceed configured capacity")
        positions = torch.arange(lease.start, lease.end, device=embedding.device, dtype=torch.int64).unsqueeze(0)
        hidden = reference._record("embedding", F.embedding(input_ids, embedding), None, op_hook)
        groups = spec.num_attention_heads // spec.num_kv_heads
        head_dim = spec.head_dim
        query_positions = torch.arange(lease.start, lease.end, device=hidden.device, dtype=torch.int64)
        key_positions = torch.arange(lease.end, device=hidden.device, dtype=torch.int64)
        blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        causal_mask = torch.zeros((query_tokens, lease.end), device=hidden.device, dtype=hidden.dtype)
        causal_mask.masked_fill_(blocked, torch.finfo(hidden.dtype).min)

        for layer in range(spec.num_layers):
            prefix = f"model.layers.{layer}"
            op = f"layer.{layer}"
            residual = hidden
            norm = reference._record(
                f"{op}.input_norm",
                reference._norm(hidden, prefix + ".input_layernorm.weight"),
                layer,
                op_hook,
            )
            query = reference._linear(norm, prefix + ".self_attn.q_proj", bias=True).view(
                1, query_tokens, spec.num_attention_heads, head_dim
            ).transpose(1, 2)
            key = reference._linear(norm, prefix + ".self_attn.k_proj", bias=True).view(
                1, query_tokens, spec.num_kv_heads, head_dim
            ).transpose(1, 2)
            value = reference._linear(norm, prefix + ".self_attn.v_proj", bias=True).view(
                1, query_tokens, spec.num_kv_heads, head_dim
            ).transpose(1, 2)
            reference._record(f"{op}.q_projection", query, layer, op_hook)
            reference._record(f"{op}.k_projection", key, layer, op_hook)
            reference._record(f"{op}.v_projection", value, layer, op_hook)
            query, key = reference._rope(query, key, positions)
            reference._record(f"{op}.rope_q", query, layer, op_hook)
            reference._record(f"{op}.rope_k", key, layer, op_hook)
            previous_key, previous_value = lease.prefix(layer)
            lease.stage_layer(layer, key, value)
            all_key = torch.cat((previous_key, key), dim=2) if lease.start else key
            all_value = torch.cat((previous_value, value), dim=2) if lease.start else value
            repeated_key = all_key.repeat_interleave(groups, dim=1)
            repeated_value = all_value.repeat_interleave(groups, dim=1)
            scores = (query @ repeated_key.transpose(-2, -1)) * (head_dim ** -0.5)
            scores = reference._record(
                f"{op}.attention_scores",
                scores + causal_mask[None, None, :, :],
                layer,
                op_hook,
            )
            probabilities = reference._record(
                f"{op}.attention_probabilities",
                F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype),
                layer,
                op_hook,
            )
            context = reference._record(
                f"{op}.attention_context", probabilities @ repeated_value, layer, op_hook
            )
            attention = reference._record(
                f"{op}.attention_output",
                reference._linear(
                    context.transpose(1, 2).contiguous().view(1, query_tokens, -1),
                    prefix + ".self_attn.o_proj",
                ),
                layer,
                op_hook,
            )
            hidden = reference._record(f"{op}.attention_residual", residual + attention, layer, op_hook)
            residual = hidden
            norm = reference._record(
                f"{op}.post_attention_norm",
                reference._norm(hidden, prefix + ".post_attention_layernorm.weight"),
                layer,
                op_hook,
            )
            gate = reference._record(
                f"{op}.mlp_gate", reference._linear(norm, prefix + ".mlp.gate_proj"), layer, op_hook
            )
            up = reference._record(
                f"{op}.mlp_up", reference._linear(norm, prefix + ".mlp.up_proj"), layer, op_hook
            )
            activated = reference._record(f"{op}.mlp_silu", F.silu(gate), layer, op_hook)
            product = reference._record(f"{op}.mlp_product", activated * up, layer, op_hook)
            mlp = reference._record(
                f"{op}.mlp_output", reference._linear(product, prefix + ".mlp.down_proj"), layer, op_hook
            )
            hidden = reference._record(f"{op}.mlp_residual", residual + mlp, layer, op_hook)
        hidden = reference._record("final_norm", reference._norm(hidden, "model.norm.weight"), None, op_hook)
        logits = F.linear(hidden, reference.canonical_weight("lm_head.weight"))
        return reference._record("logits", logits, None, op_hook)

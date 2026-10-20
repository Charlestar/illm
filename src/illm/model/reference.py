"""Inspectable full-context Qwen2 eager path. Runtime dependency: PyTorch.

No Transformers runtime dependency, no parameter initialization RNG, no KV state.
Weights are registered once per canonical storage, including tied embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from illm.model.spec import ModelError, SupportedModelSpec


TORCH_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True, slots=True)
class OpRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layer: int | None
    device: str
    evidence_kind: str = "recorded"


OpHook = Callable[[OpRecord, Tensor], None]


def canonical_op_names(spec: SupportedModelSpec) -> tuple[str, ...]:
    names = ["embedding"]
    for layer in range(spec.num_layers):
        names.extend(f"layer.{layer}.{name}" for name in (
            "input_norm", "q_projection", "k_projection", "v_projection", "rope_q", "rope_k",
            "attention_scores", "attention_probabilities", "attention_context", "attention_output",
            "attention_residual", "post_attention_norm", "mlp_gate", "mlp_up", "mlp_silu",
            "mlp_product", "mlp_output", "mlp_residual",
        ))
    names.extend(("final_norm", "logits"))
    return tuple(names)


def _validate_execution_mode(device_type: str) -> None:
    if torch.is_autocast_enabled(device_type):
        raise ModelError("UNSUPPORTED_MODEL", "reference forward requires autocast disabled")
    if device_type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ModelError("UNSUPPORTED_MODEL", "reference CUDA forward requires TF32 matmul disabled")


class ReferenceDecoder(nn.Module):
    def __init__(self, spec: SupportedModelSpec, weights: Mapping[str, Tensor], *, device: str = "cpu") -> None:
        super().__init__()
        self.spec = spec
        expected = spec.tensor_shapes()
        if set(weights) != set(expected):
            raise ModelError("INVALID_WEIGHTS", f"weight coverage differs: missing={sorted(set(expected)-set(weights))} unexpected={sorted(set(weights)-set(expected))}")
        names: dict[str, str] = {}
        target_dtype = TORCH_DTYPES[spec.runtime_dtype]
        for index, name in enumerate(sorted(expected)):
            tensor = weights[name]
            if tuple(tensor.shape) != expected[name] or tensor.dtype != TORCH_DTYPES[spec.weight_dtype]:
                raise ModelError("INVALID_WEIGHTS", "weight shape/dtype differs from spec", field=name)
            if tensor.layout != torch.strided or not tensor.is_contiguous():
                raise ModelError("INVALID_WEIGHTS", "reference requires contiguous strided canonical weights", field=name)
            if not bool(torch.isfinite(tensor).all().item()):
                raise ModelError("NUMERICAL_ERROR", "weight tensor contains non-finite values", field=name)
            buffer_name = f"canonical_weight_{index:04d}"
            # clone once, not once per alias. No public mutable weights mapping.
            self.register_buffer(buffer_name, tensor.detach().to(device=device, dtype=target_dtype).clone(), persistent=True)
            names[name] = buffer_name
        self._weight_names = MappingProxyType(names)
        self.eval()

    def canonical_weight(self, name: str) -> Tensor:
        if name == "lm_head.weight" and self.spec.tie_word_embeddings:
            name = "model.embed_tokens.weight"
        try:
            value = getattr(self, self._weight_names[name])
        except (KeyError, AttributeError) as exc:
            raise ModelError("INVALID_WEIGHTS", "unknown canonical weight", field=name) from exc
        if not isinstance(value, Tensor):
            raise ModelError("INVALID_WEIGHTS", "canonical storage is no longer a tensor", field=name)
        return value

    def _record(self, name: str, tensor: Tensor, layer: int | None, hook: OpHook | None) -> Tensor:
        valid = torch.isfinite(tensor)
        if name.endswith(".attention_scores") and tensor.ndim == 4 and tensor.shape[-2] == tensor.shape[-1]:
            future = torch.triu(torch.ones(tensor.shape[-2:], device=tensor.device, dtype=torch.bool), diagonal=1)
            valid = valid | (future[None, None, :, :] & torch.isneginf(tensor))
        if not bool(valid.all().item()):
            raise ModelError("NUMERICAL_ERROR", "model op produced non-finite values", field=name)
        if hook is not None:
            hook(OpRecord(name, tuple(tensor.shape), str(tensor.dtype), layer, str(tensor.device)), tensor.detach())
        return tensor

    def _norm(self, hidden: Tensor, name: str) -> Tensor:
        original_dtype = hidden.dtype
        promoted = hidden.float()
        variance = (promoted * promoted).mean(dim=-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.spec.rms_norm_eps)
        return normalized.to(original_dtype) * self.canonical_weight(name)

    def _linear(self, hidden: Tensor, name: str, *, bias: bool = False) -> Tensor:
        return F.linear(hidden, self.canonical_weight(name + ".weight"),
                        self.canonical_weight(name + ".bias") if bias else None)

    def _rope(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        # Calculate frequencies/trigonometry in FP32 even for BF16/FP16 weights.
        with torch.autocast(device_type=query.device.type, enabled=False):
            indices = torch.arange(0, self.spec.head_dim, 2, device=query.device, dtype=torch.float32)
            inv_frequencies = 1.0 / (self.spec.rope_theta ** (indices / self.spec.head_dim))
            frequencies = torch.matmul(inv_frequencies[None, :, None], positions[:, None, :].float()).transpose(1, 2)
            angles = torch.cat((frequencies, frequencies), dim=-1)
            cosine = torch.cos(angles).to(query.dtype).unsqueeze(1)
            sine = torch.sin(angles).to(query.dtype).unsqueeze(1)
        half = self.spec.head_dim // 2
        q_rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        k_rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        return query * cosine + q_rotated * sine, key * cosine + k_rotated * sine

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, *, position_ids: Tensor | None = None, op_hook: OpHook | None = None) -> Tensor:
        embedding = self.canonical_weight("model.embed_tokens.weight")
        _validate_execution_mode(embedding.device.type)
        if input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ModelError("INVALID_ARGUMENT", "input_ids must be single-request int64 [1,T]")
        length = input_ids.shape[1]
        if not 1 <= length <= self.spec.max_position_embeddings:
            raise ModelError("INVALID_ARGUMENT", "context length is outside supported bounds")
        if input_ids.device != embedding.device:
            raise ModelError("INVALID_ARGUMENT", "input_ids and model must be on the same device")
        if bool(((input_ids < 0) | (input_ids >= self.spec.vocab_size)).any().item()):
            raise ModelError("INVALID_ARGUMENT", "input token is outside vocabulary")
        if position_ids is None:
            position_ids = torch.arange(length, device=embedding.device, dtype=torch.int64).unsqueeze(0)
        if position_ids.dtype != torch.int64 or tuple(position_ids.shape) != (1, length) or position_ids.device != embedding.device:
            raise ModelError("INVALID_ARGUMENT", "positions must be int64 [1,T] on model device")
        if bool(((position_ids < 0) | (position_ids >= self.spec.max_position_embeddings)).any().item()):
            raise ModelError("INVALID_ARGUMENT", "absolute position is outside supported bounds")
        if length > 1 and not bool((position_ids[:, 1:] == position_ids[:, :-1] + 1).all().item()):
            raise ModelError("INVALID_ARGUMENT", "full-context positions must be contiguous")
        hidden = self._record("embedding", F.embedding(input_ids, embedding), None, op_hook)
        causal_mask = torch.triu(torch.full((length, length), torch.finfo(hidden.dtype).min,
                                           device=hidden.device, dtype=hidden.dtype), diagonal=1)
        groups = self.spec.num_attention_heads // self.spec.num_kv_heads
        head_dim = self.spec.head_dim
        for layer in range(self.spec.num_layers):
            prefix = f"model.layers.{layer}"
            op = f"layer.{layer}"
            residual = hidden
            norm = self._record(f"{op}.input_norm", self._norm(hidden, prefix + ".input_layernorm.weight"), layer, op_hook)
            q = self._linear(norm, prefix + ".self_attn.q_proj", bias=True).view(1, length, self.spec.num_attention_heads, head_dim).transpose(1, 2)
            k = self._linear(norm, prefix + ".self_attn.k_proj", bias=True).view(1, length, self.spec.num_kv_heads, head_dim).transpose(1, 2)
            v = self._linear(norm, prefix + ".self_attn.v_proj", bias=True).view(1, length, self.spec.num_kv_heads, head_dim).transpose(1, 2)
            self._record(f"{op}.q_projection", q, layer, op_hook)
            self._record(f"{op}.k_projection", k, layer, op_hook)
            self._record(f"{op}.v_projection", v, layer, op_hook)
            q, k = self._rope(q, k, position_ids)
            self._record(f"{op}.rope_q", q, layer, op_hook)
            self._record(f"{op}.rope_k", k, layer, op_hook)
            repeated_k = k.repeat_interleave(groups, dim=1)
            repeated_v = v.repeat_interleave(groups, dim=1)
            scores = (q @ repeated_k.transpose(-2, -1)) * (head_dim ** -0.5)
            # FP16 finfo.min + a negative masked score may round to -Inf.
            # This is a legal causal mask value, not a raw-logit NaN/Inf.
            # No other op, unmasked position, +Inf or NaN gets this exception.
            scores = self._record(f"{op}.attention_scores", scores + causal_mask[None, None, :, :], layer, op_hook)
            probabilities = self._record(f"{op}.attention_probabilities", F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype), layer, op_hook)
            context = self._record(f"{op}.attention_context", probabilities @ repeated_v, layer, op_hook)
            attention = self._record(f"{op}.attention_output", self._linear(context.transpose(1, 2).contiguous().view(1, length, -1), prefix + ".self_attn.o_proj"), layer, op_hook)
            hidden = self._record(f"{op}.attention_residual", residual + attention, layer, op_hook)
            residual = hidden
            norm = self._record(f"{op}.post_attention_norm", self._norm(hidden, prefix + ".post_attention_layernorm.weight"), layer, op_hook)
            gate = self._record(f"{op}.mlp_gate", self._linear(norm, prefix + ".mlp.gate_proj"), layer, op_hook)
            up = self._record(f"{op}.mlp_up", self._linear(norm, prefix + ".mlp.up_proj"), layer, op_hook)
            activated = self._record(f"{op}.mlp_silu", F.silu(gate), layer, op_hook)
            product = self._record(f"{op}.mlp_product", activated * up, layer, op_hook)
            mlp = self._record(f"{op}.mlp_output", self._linear(product, prefix + ".mlp.down_proj"), layer, op_hook)
            hidden = self._record(f"{op}.mlp_residual", residual + mlp, layer, op_hook)
        hidden = self._record("final_norm", self._norm(hidden, "model.norm.weight"), None, op_hook)
        logits = F.linear(hidden, self.canonical_weight("lm_head.weight"))
        return self._record("logits", logits, None, op_hook)


def load_reference(spec: SupportedModelSpec, paths: Sequence[Path], *, device: str = "cpu") -> ReferenceDecoder:
    # Import lazily: inspect/config/tokenizer preparation itself needs no torch.
    from illm.model.loader import build_weight_manifest
    from safetensors.torch import load_file

    build_weight_manifest(paths, spec)  # Strict raw-header/range/coverage preflight.
    weights: dict[str, Tensor] = {}
    for path in paths:
        shard = load_file(str(path), device="cpu")
        if set(weights) & set(shard):
            raise ModelError("DUPLICATE_WEIGHT", "duplicate tensor after safetensors load")
        weights.update(shard)
    return ReferenceDecoder(spec, weights, device=device)

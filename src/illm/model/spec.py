"""Strict Qwen2 numerical contract. No torch/Transformers dependency."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


HF_FIELDS = frozenset({
    "architectures", "attention_dropout", "bos_token_id", "eos_token_id", "hidden_act",
    "hidden_size", "initializer_range", "intermediate_size", "max_position_embeddings",
    "max_window_layers", "model_type", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "rms_norm_eps", "rope_theta", "sliding_window",
    "tie_word_embeddings", "torch_dtype", "transformers_version", "use_cache",
    "use_mrope", "use_sliding_window", "vocab_size", "rope_scaling", "head_dim",
    "attention_bias", "mlp_bias", "_name_or_path",
})
DTYPES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
REVISION = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")


class ModelError(ValueError):
    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _integer(value: object, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ModelError("INVALID_MODEL_CONFIG", f"{field} must be integer >= {minimum}", field=field)
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError("INVALID_MODEL_CONFIG", f"{field} must be a finite positive number", field=field)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ModelError("INVALID_MODEL_CONFIG", f"{field} overflows float", field=field) from exc
    if not math.isfinite(result) or result <= 0:
        raise ModelError("INVALID_MODEL_CONFIG", f"{field} must be a finite positive number", field=field)
    return result


@dataclass(frozen=True, slots=True)
class SupportedModelSpec:
    architecture: str
    model_revision: str
    tokenizer_revision: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    weight_dtype: str
    runtime_dtype: str
    bos_token_id: int
    eos_token_id: int
    activation: str = "silu"
    qkv_bias: bool = True
    attention_output_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: None = None
    weight_layout_id: str = "hf-qwen2-out-in-v1"
    kv_layout_id: str = "none-reference-full-context-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported SupportedModelSpec schema")
        if self.architecture != "Qwen2ForCausalLM" or self.activation != "silu":
            raise ModelError("UNSUPPORTED_MODEL", "only dense Qwen2ForCausalLM with SiLU is supported")
        for field in ("model_revision", "tokenizer_revision"):
            value = getattr(self, field)
            if not isinstance(value, str) or not REVISION.fullmatch(value):
                raise ModelError("INVALID_MODEL_CONFIG", "revision must be immutable commit or sha256", field=field)
        for field in ("vocab_size", "hidden_size", "intermediate_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "max_position_embeddings"):
            _integer(getattr(self, field), field)
        if self.hidden_size != self.num_attention_heads * self.head_dim or self.head_dim % 2:
            raise ModelError("INVALID_MODEL_CONFIG", "hidden size/head count/even head dimension disagree", field="head_dim")
        if self.num_attention_heads % self.num_kv_heads:
            raise ModelError("INVALID_MODEL_CONFIG", "Q head count must be divisible by KV head count", field="num_kv_heads")
        if self.qkv_bias is not True or self.attention_output_bias is not False or self.mlp_bias is not False:
            raise ModelError("UNSUPPORTED_MODEL", "Qwen2 projection bias contract differs")
        if type(self.tie_word_embeddings) is not bool:
            raise ModelError("INVALID_MODEL_CONFIG", "tie_word_embeddings must be bool", field="tie_word_embeddings")
        if self.rope_scaling is not None or self.weight_layout_id != "hf-qwen2-out-in-v1" or self.kv_layout_id != "none-reference-full-context-v1":
            raise ModelError("UNSUPPORTED_MODEL", "unsupported RoPE or tensor layout")
        for field in ("rms_norm_eps", "rope_theta"):
            object.__setattr__(self, field, _number(getattr(self, field), field))
        if not isinstance(self.weight_dtype, str) or not isinstance(self.runtime_dtype, str) or self.weight_dtype not in {"fp32", "fp16", "bf16"} or self.runtime_dtype not in {"fp32", "fp16", "bf16"}:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported model dtype")
        for field in ("bos_token_id", "eos_token_id"):
            if _integer(getattr(self, field), field, minimum=0) >= self.vocab_size:
                raise ModelError("INVALID_MODEL_CONFIG", "special token is outside vocabulary", field=field)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def spec_id(self) -> str:
        return digest(self.to_dict())

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        h, i, kv = self.hidden_size, self.intermediate_size, self.num_kv_heads * self.head_dim
        shapes: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (self.vocab_size, h), "model.norm.weight": (h,)}
        for layer in range(self.num_layers):
            prefix = f"model.layers.{layer}"
            shapes[f"{prefix}.input_layernorm.weight"] = (h,)
            shapes[f"{prefix}.post_attention_layernorm.weight"] = (h,)
            for projection, width in (("q", h), ("k", kv), ("v", kv)):
                shapes[f"{prefix}.self_attn.{projection}_proj.weight"] = (width, h)
                shapes[f"{prefix}.self_attn.{projection}_proj.bias"] = (width,)
            shapes[f"{prefix}.self_attn.o_proj.weight"] = (h, h)
            shapes[f"{prefix}.mlp.gate_proj.weight"] = (i, h)
            shapes[f"{prefix}.mlp.up_proj.weight"] = (i, h)
            shapes[f"{prefix}.mlp.down_proj.weight"] = (h, i)
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocab_size, h)
        return dict(sorted(shapes.items()))

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(shape) for shape in self.tensor_shapes().values())

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any], *, model_revision: str, tokenizer_revision: str, runtime_dtype: str) -> SupportedModelSpec:
        if not isinstance(config, Mapping) or any(not isinstance(key, str) for key in config):
            raise ModelError("INVALID_MODEL_CONFIG", "config must be string-keyed object")
        unknown = set(config) - HF_FIELDS
        if unknown:
            raise ModelError("UNSUPPORTED_MODEL", f"unknown model config fields: {sorted(unknown)}", field=sorted(unknown)[0])
        required = {
            "architectures", "model_type", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
            "rope_theta", "hidden_act", "tie_word_embeddings", "max_position_embeddings",
            "torch_dtype", "bos_token_id", "eos_token_id",
        }
        missing = required - set(config)
        if missing:
            raise ModelError("INVALID_MODEL_CONFIG", f"missing model config fields: {sorted(missing)}", field=sorted(missing)[0])
        if config["architectures"] != ["Qwen2ForCausalLM"] or config["model_type"] != "qwen2":
            raise ModelError("UNSUPPORTED_MODEL", "exact qwen2 architecture is required", field="architectures")
        for name in ("use_sliding_window", "use_mrope", "mlp_bias"):
            if config.get(name, False) is not False:
                raise ModelError("UNSUPPORTED_MODEL", f"{name} is unsupported", field=name)
        if config.get("attention_bias", True) is not True:
            raise ModelError("UNSUPPORTED_MODEL", "QKV bias is required", field="attention_bias")
        if config.get("rope_scaling") is not None:
            raise ModelError("UNSUPPORTED_MODEL", "scaled RoPE is unsupported", field="rope_scaling")
        dropout = config.get("attention_dropout", 0.0)
        if isinstance(dropout, bool) or not isinstance(dropout, (float, int)) or dropout != 0.0:
            raise ModelError("UNSUPPORTED_MODEL", "attention dropout must be zero", field="attention_dropout")
        if "use_cache" in config and type(config["use_cache"]) is not bool:
            raise ModelError("INVALID_MODEL_CONFIG", "use_cache must be bool", field="use_cache")
        dtype = config["torch_dtype"]
        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported weight dtype", field="torch_dtype")
        hidden = _integer(config["hidden_size"], "hidden_size")
        heads = _integer(config["num_attention_heads"], "num_attention_heads")
        if hidden % heads:
            raise ModelError("INVALID_MODEL_CONFIG", "hidden size is not divisible by attention heads", field="hidden_size")
        explicit_head = config.get("head_dim", hidden // heads)
        return cls(
            architecture="Qwen2ForCausalLM", model_revision=model_revision, tokenizer_revision=tokenizer_revision,
            vocab_size=config["vocab_size"], hidden_size=hidden, intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"], num_attention_heads=heads, num_kv_heads=config["num_key_value_heads"],
            head_dim=explicit_head, rms_norm_eps=config["rms_norm_eps"], rope_theta=config["rope_theta"],
            tie_word_embeddings=config["tie_word_embeddings"], max_position_embeddings=config["max_position_embeddings"],
            activation=config["hidden_act"], weight_dtype=DTYPES[dtype], runtime_dtype=runtime_dtype,
            bos_token_id=config["bos_token_id"], eos_token_id=config["eos_token_id"],
        )

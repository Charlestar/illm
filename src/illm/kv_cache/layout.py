"""Independent V0.2 contiguous-KV layout contract."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from illm.model.spec import ModelError, SupportedModelSpec, digest


DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
CONTIGUOUS_LAYOUT_ID = "qwen2-contiguous-layer-major-v1"
_DEVICE = re.compile(r"^(?:cpu|cuda(?::[0-9]+)?)$")


@dataclass(frozen=True, slots=True)
class KVLayoutSpec:
    """Physical K/V layout without changing ``SupportedModelSpec`` identity."""

    schema_version: int
    layout_id: str
    model_spec_id: str
    dtype: str
    device: str
    capacity_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported KV layout schema")
        if self.layout_id != CONTIGUOUS_LAYOUT_ID:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported KV layout ID")
        if (
            not isinstance(self.model_spec_id, str)
            or len(self.model_spec_id) != 64
            or any(character not in "0123456789abcdef" for character in self.model_spec_id)
        ):
            raise ModelError("INVALID_MODEL_CONFIG", "KV layout model spec identity is invalid")
        if self.dtype not in DTYPE_BYTES:
            raise ModelError("UNSUPPORTED_MODEL", "unsupported KV dtype")
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ModelError("INVALID_ARGUMENT", "KV device must be cpu, cuda or cuda:<index>")
        for field in ("capacity_tokens", "num_layers", "num_kv_heads", "head_dim"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ModelError("INVALID_MODEL_CONFIG", f"{field} must be a positive integer", field=field)

    @classmethod
    def for_model(cls, spec: SupportedModelSpec, *, device: str, capacity: int) -> KVLayoutSpec:
        if type(capacity) is not int or not 1 <= capacity <= spec.max_position_embeddings:
            raise ModelError("INVALID_ARGUMENT", "KV capacity is outside the model context contract", field="capacity")
        return cls(
            schema_version=1,
            layout_id=CONTIGUOUS_LAYOUT_ID,
            model_spec_id=spec.spec_id,
            dtype=spec.runtime_dtype,
            device=device,
            capacity_tokens=capacity,
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
        )

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return (self.num_layers, self.num_kv_heads, self.capacity_tokens, self.head_dim)

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * DTYPE_BYTES[self.dtype]

    @property
    def reserved_bytes(self) -> int:
        return math.prod(self.tensor_shape) * DTYPE_BYTES[self.dtype] * 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

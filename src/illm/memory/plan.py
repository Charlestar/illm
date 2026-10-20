"""Deterministic, immutable V0.2 single-request memory planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from illm.kv_cache import KVLayoutSpec
from illm.kv_cache.layout import DTYPE_BYTES
from illm.model.loader import LoadedModel
from illm.model.spec import ModelError, digest


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Frozen capacity plan derived only from versioned model/config facts."""

    schema_version: int
    model_spec_id: str
    weights_digest: str
    weight_manifest_digest: str
    runtime_dtype: str
    device: str
    kv_mode: str
    kv_layout_id: str
    capacity_tokens: int
    memory_budget_bytes: int
    weights_bytes: int
    kv_bytes: int
    activation_bytes: int
    workspace_bytes: int
    io_bytes: int
    telemetry_bytes: int
    headroom_bytes: int
    required_bytes: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ModelError("INVALID_MODEL_CONFIG", "unsupported MemoryPlan schema")
        if self.kv_mode not in {"none", "contiguous"}:
            raise ModelError("INVALID_ARGUMENT", "MemoryPlan kv_mode must be none or contiguous")
        if (
            self.runtime_dtype not in DTYPE_BYTES
            or not isinstance(self.device, str)
            or not self.device
            or not isinstance(self.kv_layout_id, str)
            or not self.kv_layout_id
        ):
            raise ModelError("INVALID_MODEL_CONFIG", "MemoryPlan dtype/device binding is invalid")
        for field in ("model_spec_id", "weights_digest", "weight_manifest_digest"):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ModelError("INVALID_MODEL_CONFIG", f"MemoryPlan {field} is not a SHA-256 identity", field=field)
        for field in (
            "capacity_tokens",
            "memory_budget_bytes",
            "weights_bytes",
            "activation_bytes",
            "workspace_bytes",
            "io_bytes",
            "telemetry_bytes",
            "headroom_bytes",
            "required_bytes",
        ):
            value = getattr(self, field)
            minimum = 1 if field in {"capacity_tokens", "memory_budget_bytes", "weights_bytes", "required_bytes"} else 0
            if type(value) is not int or value < minimum:
                raise ModelError("INVALID_MODEL_CONFIG", f"MemoryPlan {field} is invalid", field=field)
        if type(self.kv_bytes) is not int or self.kv_bytes < 0:
            raise ModelError("INVALID_MODEL_CONFIG", "MemoryPlan kv_bytes is invalid", field="kv_bytes")
        components = (
            self.weights_bytes
            + self.kv_bytes
            + self.activation_bytes
            + self.workspace_bytes
            + self.io_bytes
            + self.telemetry_bytes
            + self.headroom_bytes
        )
        if self.required_bytes != components:
            raise ModelError("INVALID_MODEL_CONFIG", "MemoryPlan total differs from component sum")

    @property
    def fits_budget(self) -> bool:
        return self.required_bytes <= self.memory_budget_bytes

    def require_admission(self, requested_tokens: int) -> None:
        if type(requested_tokens) is not int or requested_tokens < 1:
            raise ModelError("INVALID_ARGUMENT", "requested token capacity must be a positive integer")
        if requested_tokens > self.capacity_tokens:
            raise ModelError(
                "RESOURCE_EXHAUSTED",
                "request exceeds precomputed MemoryPlan token capacity",
                field="capacity_tokens",
            )
        if not self.fits_budget:
            raise ModelError(
                "RESOURCE_EXHAUSTED",
                "precomputed MemoryPlan exceeds configured memory budget",
                field="memory_budget_bytes",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


def build_memory_plan(
    loaded: LoadedModel,
    kv_mode: str,
    capacity: int,
    memory_budget_bytes: int,
    *,
    device: str | None = None,
    telemetry_bytes: int = 1024 * 1024,
) -> MemoryPlan:
    """Build a plan without consulting allocator or live free-memory state."""

    spec = loaded.spec
    manifest = loaded.weight_manifest
    if kv_mode not in {"none", "contiguous"}:
        raise ModelError("INVALID_ARGUMENT", "kv_mode must be none or contiguous", field="kv_mode")
    if type(capacity) is not int or not 1 <= capacity <= min(loaded.max_context_tokens, spec.max_position_embeddings):
        raise ModelError("INVALID_ARGUMENT", "MemoryPlan capacity is outside model context contract", field="capacity")
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 1:
        raise ModelError("INVALID_ARGUMENT", "memory budget must be a positive integer", field="memory_budget_bytes")
    if type(telemetry_bytes) is not int or telemetry_bytes < 0:
        raise ModelError("INVALID_ARGUMENT", "telemetry bytes must be a nonnegative integer", field="telemetry_bytes")
    loaded_device = str(loaded.reference.canonical_weight("model.embed_tokens.weight").device)
    if device is None:
        device = loaded_device
    if not isinstance(device, str) or not device:
        raise ModelError("INVALID_ARGUMENT", "device binding must be a nonempty string", field="device")
    if device != loaded_device:
        raise ModelError("INVALID_ARGUMENT", "MemoryPlan device differs from loaded model device", field="device")
    weights_digest = manifest.get("weights_digest")
    manifest_digest = manifest.get("manifest_digest")
    storages = manifest.get("storages")
    if (
        not isinstance(weights_digest, str)
        or not isinstance(manifest_digest, str)
        or not isinstance(storages, list)
        or not storages
        or manifest.get("spec_id") != spec.spec_id
        or manifest.get("runtime_dtype") != spec.runtime_dtype
        or manifest.get("finalized") is not True
        or manifest_digest != digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    ):
        raise ModelError("INVALID_WEIGHTS", "weight manifest cannot bind a MemoryPlan")
    weights_bytes = 0
    for storage in storages:
        if not isinstance(storage, dict) or type(storage.get("final_storage_bytes")) is not int or storage["final_storage_bytes"] < 1:
            raise ModelError("INVALID_WEIGHTS", "weight storage byte count is invalid")
        weights_bytes += storage["final_storage_bytes"]
    recomputed_weights_digest = digest(
        [
            {
                "name": storage.get("canonical_name"),
                "shape": storage.get("shape"),
                "source_dtype": storage.get("source_dtype"),
                "checksum": storage.get("checksum"),
            }
            for storage in storages
        ]
    )
    if recomputed_weights_digest != weights_digest:
        raise ModelError("INVALID_WEIGHTS", "weight manifest weights digest is invalid")

    dtype_bytes = DTYPE_BYTES[spec.runtime_dtype]
    if kv_mode == "contiguous":
        kv_layout = KVLayoutSpec.for_model(spec, device=device, capacity=capacity)
        kv_layout_id = kv_layout.layout_id
        kv_bytes = kv_layout.reserved_bytes
    else:
        kv_layout_id = spec.kv_layout_id
        kv_bytes = 0

    # Conservative tensor-live-set planning for both full-context and cached
    # eager paths. Python loop locals can retain the preceding layer's tensors
    # until reassignment, and softmax/norm temporaries use FP32 even with FP16
    # weights. These are planned bytes, not measured allocator/free-memory facts.
    # CUDA library/internal allocator overhead is covered separately by fixed
    # workspace plus explicit headroom, never inferred from currently free VRAM.
    calculation_bytes = max(dtype_bytes, 4)
    vector_elements = capacity * (
        16 * spec.hidden_size
        + 8 * spec.intermediate_size
        + 4 * (spec.num_attention_heads + 2 * spec.num_kv_heads) * spec.head_dim
    )
    attention_elements = 16 * spec.num_attention_heads * capacity * capacity
    activation_bytes = _align_up((vector_elements + attention_elements) * calculation_bytes + 4 * capacity * capacity)
    workspace_bytes = _align_up(
        max(
            8 * spec.num_attention_heads * capacity * capacity * 4,
            4 * capacity * spec.intermediate_size * calculation_bytes,
            64 * 1024 * 1024 if device.startswith("cuda") else 1024 * 1024,
        )
    )
    # Reference forward materializes [1,T,V], not only the final [V]. Include
    # two whole logits buffers plus the exported final-row FP32/host scalars.
    io_bytes = _align_up(2 * capacity * 8 + 2 * capacity * spec.vocab_size * dtype_bytes + 64 * spec.vocab_size)
    # Required capture and in-memory canonical metadata scale with context;
    # optional exporter ring capacity is bounded independently of token count.
    telemetry_bytes = max(telemetry_bytes, capacity * 64 * 1024) if telemetry_bytes else 0
    subtotal = weights_bytes + kv_bytes + activation_bytes + workspace_bytes + io_bytes + telemetry_bytes
    headroom_bytes = _align_up((subtotal + 9) // 10)
    required_bytes = subtotal + headroom_bytes
    return MemoryPlan(
        schema_version=1,
        model_spec_id=spec.spec_id,
        weights_digest=weights_digest,
        weight_manifest_digest=manifest_digest,
        runtime_dtype=spec.runtime_dtype,
        device=device,
        kv_mode=kv_mode,
        kv_layout_id=kv_layout_id,
        capacity_tokens=capacity,
        memory_budget_bytes=memory_budget_bytes,
        weights_bytes=weights_bytes,
        kv_bytes=kv_bytes,
        activation_bytes=activation_bytes,
        workspace_bytes=workspace_bytes,
        io_bytes=io_bytes,
        telemetry_bytes=telemetry_bytes,
        headroom_bytes=headroom_bytes,
        required_bytes=required_bytes,
    )

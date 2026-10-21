"""Policy-aware static planning; old MemoryPlan schema1 is not reinterpreted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from illm.memory.plan import MemoryPlan, build_memory_plan
from illm.model.loader import LoadedModel
from illm.model.spec import ModelError, digest
from illm.numerical_policy import NumericalExecutionPolicyV1
from illm.policy_binding import ExecutionBindingV1, build_execution_binding


@dataclass(frozen=True, slots=True)
class PolicyMemoryPlan:
    """Additional compute storage belongs to the explicit policy, not KV mode."""

    base: MemoryPlan
    numerical_policy: NumericalExecutionPolicyV1
    sidecar_bytes: int
    prefix_scratch_bytes: int
    headroom_bytes: int
    required_bytes: int
    execution_binding: ExecutionBindingV1

    def __post_init__(self) -> None:
        self.numerical_policy.validate_kv_mode(self.base.kv_mode)
        if self.numerical_policy.storage_dtype != self.base.runtime_dtype:
            raise ModelError("INVALID_ARGUMENT", "policy MemoryPlan dtype differs")
        for name in ("sidecar_bytes", "prefix_scratch_bytes", "headroom_bytes", "required_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ModelError("INVALID_ARGUMENT", "policy MemoryPlan component is invalid", field=name)
        subtotal = self.base.required_bytes - self.base.headroom_bytes + self.sidecar_bytes + self.prefix_scratch_bytes
        if self.required_bytes != subtotal + self.headroom_bytes:
            raise ModelError("INVALID_ARGUMENT", "policy MemoryPlan total differs from components")

    @property
    def fits_budget(self) -> bool:
        return self.required_bytes <= self.base.memory_budget_bytes

    @property
    def capacity_tokens(self) -> int:
        return self.base.capacity_tokens

    @property
    def kv_bytes(self) -> int:
        return self.base.kv_bytes

    def require_admission(self, requested_tokens: int) -> None:
        self.base.require_admission(requested_tokens)
        if not self.fits_budget:
            raise ModelError("RESOURCE_EXHAUSTED", "policy MemoryPlan including sidecars exceeds budget")

    def to_dict(self) -> dict[str, Any]:
        value = self.base.to_dict()
        value.update(schema_version=2, numerical_policy=self.numerical_policy.to_dict(),
                     numerical_policy_digest=self.numerical_policy.digest,
                     compute_dtype="fp32" if self.numerical_policy.is_candidate else self.base.runtime_dtype,
                     sidecar_bytes=self.sidecar_bytes, prefix_scratch_bytes=self.prefix_scratch_bytes,
                     headroom_bytes=self.headroom_bytes, required_bytes=self.required_bytes,
                     execution_binding=self.execution_binding.to_dict(),
                     execution_binding_digest=self.execution_binding.digest,
                     evidence_kind="static_conservative_estimate")
        return value

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


def build_policy_memory_plan(loaded: LoadedModel, policy: NumericalExecutionPolicyV1,
                             kv_mode: str, capacity: int, memory_budget_bytes: int, *,
                             telemetry_bytes: int = 1024 * 1024,
                             project_root: Path | None = None) -> PolicyMemoryPlan:
    """Budget unique mirrors and canonical prefix copies before executor creation."""
    policy.validate_kv_mode(kv_mode)
    base = build_memory_plan(loaded, kv_mode, capacity, memory_budget_bytes, telemetry_bytes=telemetry_bytes)
    source_dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[loaded.spec.runtime_dtype]
    # The frozen implementation creates one FP32 mirror for every canonical
    # storage, including FP32 fixtures. Tied aliases are absent from storages.
    sidecars = base.weights_bytes // source_dtype_bytes * 4 if policy.is_candidate else 0
    spec = loaded.spec
    prefix = (2 * capacity * spec.num_kv_heads * spec.head_dim * (source_dtype_bytes + 4)
              + spec.num_attention_heads * capacity * 8) if policy.is_candidate else 0
    prefix = (prefix + 255) // 256 * 256
    subtotal = base.required_bytes - base.headroom_bytes + sidecars + prefix
    headroom = ((subtotal + 9) // 10 + 255) // 256 * 256
    binding = build_execution_binding(project_root or Path.cwd())
    return PolicyMemoryPlan(base, policy, sidecars, prefix, headroom, subtotal + headroom, binding)

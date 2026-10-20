"""Pure numerical contracts shared by execution and portable evidence readers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from illm.model.spec import ModelError


@dataclass(frozen=True, slots=True)
class TensorTolerance:
    atol: float = 1e-4
    rtol: float = 1e-3
    profile_id: str = "tiny-fp32-op-v1"

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 for value in (self.atol, self.rtol)):
            raise ModelError("INVALID_ARGUMENT", "tensor tolerances must be finite nonnegative numbers")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ModelError("INVALID_ARGUMENT", "tolerance profile ID is required")


FIXED_PROMPTS = (
    "hello", "a", "The capital of France is", "1 + 1 =", "Explain a token",
    "Inference", "KV cache", "Qwen", "你好", "模型", "A B C", "0", "12345",
    "\n", " ", "python", "def f(x):", "Once upon a time", "CPU and GPU", "🌍",
)


def tensor_tolerances(dtype: str) -> tuple[TensorTolerance, TensorTolerance]:
    # ADR-005 thresholds are declared before numerical qualification.
    if dtype == "bf16":
        return (TensorTolerance(0.0625, 0.02, "real-bf16-op-v1"),
                TensorTolerance(0.005, 0.02, "real-bf16-attention-probability-v1"))
    if dtype == "fp16":
        return (TensorTolerance(0.0078125, 0.01, "real-fp16-op-v1"),
                TensorTolerance(0.001, 0.01, "real-fp16-attention-probability-v1"))
    return TensorTolerance(), TensorTolerance()


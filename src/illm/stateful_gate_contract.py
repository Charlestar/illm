"""Tensor-free, shared V0.2 qualification execution contract.

Artifact IDs are portable paths relative to the gate root. The actual argv and
hashed transcripts remain independently required; these parameters are not
execution proof on their own.
"""

from __future__ import annotations

from typing import Any


STATEFUL_PROFILE_RUNTIME_IDS = {
    "state-machine-cpu-v1": "py312-win-amd64-cpu",
    "stateful-streaming-single-gpu-v1": "py312-linux-x86_64-cu126",
    "clean-room-release-v1": "py312-linux-x86_64-cpu",
}
MODEL_PROFILES = {"bf16": "configs/models/v0_2.yaml", "fp16": "configs/models/v0_2_fp16.yaml"}
BENCHMARK_GROUPS = ["none-off", "none-request", "contiguous-off", "contiguous-request"]


def phase_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    """Return a new ordered, exact semantic phase set for one profile."""
    runtime = STATEFUL_PROFILE_RUNTIME_IDS[profile_id]
    phases: dict[str, dict[str, Any]] = {
        "bootstrap": {"runtime_profile_id": runtime, "locked": True, "offline": True, "fresh_environment": True},
        "quality": {"scope": "all-python", "tool": "mypy"},
        "pytest": {"suite": "all", "junit_artifact": "junit.xml"},
    }
    if profile_id == "stateful-streaming-single-gpu-v1":
        for dtype, model in MODEL_PROFILES.items():
            phases[f"stateful-{dtype}"] = {
                "target": "real", "model_profile": model, "device": "cuda", "scope": "real-smoke",
                "runtime_dtype": dtype, "fixed_greedy_cases": 20,
            }
        for dtype in MODEL_PROFILES:
            for suite in ("tuning", "heldout"):
                phases[f"benchmark-{dtype}-{suite}"] = {
                    "target": "real", "suite": suite, "runtime_dtype": dtype, "device": "cuda",
                    "prompt_tokens": 256, "output_tokens": 64, "evidence_scope": "real-single-gpu-256x64",
                    "groups": list(BENCHMARK_GROUPS),
                }
    else:
        phases.update({
            "verify": {"fixture": "tiny-decoder", "device": "cpu"},
            "run": {
                "model": "fixture:tiny-decoder", "device": "cpu", "prompt": "hello", "max_new_tokens": 8,
                "stream": True, "request_id": "req-0001", "kv_mode": "contiguous", "telemetry": "request",
                "capture_artifact": "demo-capture",
            },
            "replay-trace": {"capture_artifact": "demo-capture", "mode": "R0", "request_id": "req-0001"},
            "inspect-request": {"capture_artifact": "demo-capture", "mode": "R0", "request_id": "req-0001"},
            "stateful-full": {
                "target": "tiny", "model": "fixture:tiny-decoder", "device": "cpu", "scope": "full",
                "differential_cases": 50, "fixed_greedy_cases": 20,
            },
            "exercise": {"scope": "tiny-state-machine-cpu-v1", "cycle_count": 100, "required_matrix": "all"},
            "learning-build": {"source_capture": "exercise/cancel-after-launch", "output_artifact": "learning"},
            "learning-verify": {"source_capture": "exercise/cancel-after-launch", "output_artifact": "learning"},
        })
    return phases

"""Tensor-free canonical sampling API; engine adapters own error/event mapping."""

from illm.sampling.reference import (
    CounterAddress,
    PhiloxDraw,
    SamplingDecision,
    SamplingError,
    SamplingSpec,
    SamplingState,
    commit_sample,
    fp32,
    philox4x32_10,
    philox_draw,
    prepare_sample,
    select_cdf,
)

__all__ = [
    "CounterAddress",
    "PhiloxDraw",
    "SamplingDecision",
    "SamplingError",
    "SamplingSpec",
    "SamplingState",
    "commit_sample",
    "fp32",
    "philox4x32_10",
    "philox_draw",
    "prepare_sample",
    "select_cdf",
]

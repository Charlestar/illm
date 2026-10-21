"""Explicit, unqualified CPU numerical policy; no CUDA or capture fallback."""

from __future__ import annotations

import json

from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, GenerationResult
from illm.numerical_policy import candidate_policy
from illm.sampling import SamplingSpec


def run(mode: str) -> GenerationResult:
    policy = candidate_policy("fp32")
    engine = Engine(device="cpu", kv_mode=mode, numerical_policy=policy)
    try:
        request = GenerationRequest(model_spec_id=engine.model_spec_id, prompt="hello", max_new_tokens=4,
                                    sampling=SamplingSpec(mode="greedy", actual_seed=7))
        result = engine.generate(request)
        if result.numerical_policy != policy or any(result.resource_status.values()):
            raise RuntimeError("policy identity or resource cleanup failed")
        print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
        return result
    finally:
        if not engine.close(timeout=1.0):
            raise RuntimeError("engine cleanup did not prove safe completion")


def main() -> int:
    b, c = run("none"), run("contiguous")
    if b.output_token_ids != c.output_token_ids or b.text != c.text:
        raise RuntimeError("candidate full-recompute and cached paths differ")
    if b.query_tokens_per_step != (6, 7, 8, 9) or c.query_tokens_per_step != (6, 1, 1, 1):
        raise RuntimeError("logical query work does not match KV mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

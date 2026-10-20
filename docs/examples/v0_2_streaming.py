"""Minimal V0.2 streaming API example for the synthetic tiny fixture.

Run this only inside a toolchain bootstrapped for the current source.  It is a
learning demo, not qualification or a language-quality example.
"""

from __future__ import annotations

import json

from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, TerminalChunk, TokenChunk
from illm.sampling import SamplingSpec


def main() -> int:
    engine = Engine(
        model="fixture:tiny-decoder",
        device="cpu",
        kv_mode="contiguous",  # Use "none" for the full-context oracle.
        max_context_tokens=None,
        memory_budget_bytes=6 * 1024**3,
    )
    request = GenerationRequest(
        model_spec_id=engine.model_spec_id,
        prompt="hello",
        max_new_tokens=4,
        sampling=SamplingSpec(mode="greedy", actual_seed=7),
        request_id="example-v02",
        telemetry="metrics",
    )

    text_parts: list[str] = []
    terminal: TerminalChunk | None = None
    stream = engine.stream(request)
    safely_closed = False
    try:
        for chunk in stream:
            print(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if isinstance(chunk, TokenChunk):
                text_parts.append(chunk.text_delta)
            elif isinstance(chunk, TerminalChunk):
                text_parts.append(chunk.final_text_delta)
                terminal = chunk
    finally:
        try:
            stream.close()
        finally:
            safely_closed = engine.close(timeout=1.0)

    if terminal is None:
        raise RuntimeError("stream ended without a terminal chunk")
    if "".join(text_parts) != terminal.result.text:
        raise RuntimeError("streamed text differs from the committed canonical decode")
    if terminal.result.query_tokens_per_step != (6, 1, 1, 1):
        raise RuntimeError("tiny contiguous execution did not use the expected prefill/decode split")
    if not safely_closed:
        raise RuntimeError("engine close did not prove safe completion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


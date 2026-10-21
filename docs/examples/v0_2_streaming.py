"""Minimal V0.2 generate/stream API example for the synthetic tiny fixture.

Run this only inside a toolchain bootstrapped for the current source.  It is a
learning demo, not qualification or a language-quality example.
"""

from __future__ import annotations

import json
from typing import Any

from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import (
    GenerationRequest,
    GenerationResult,
    TerminalChunk,
    TokenChunk,
)
from illm.sampling import SamplingSpec


EXPECTED_OUTPUT_TOKENS = (114, 114, 114, 114)
EXPECTED_TEXT = "oooo"


def _request(engine: Engine, request_id: str) -> GenerationRequest:
    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        prompt="hello",
        max_new_tokens=4,
        sampling=SamplingSpec(mode="greedy", actual_seed=7),
        request_id=request_id,
        telemetry="metrics",
    )


def _emit(example: str, payload: dict[str, Any]) -> None:
    print(json.dumps(
        {"example": example, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _assert_result(
    result: GenerationResult,
    *,
    kv_mode: str,
    query_tokens_per_step: tuple[int, ...],
    kv_valid_tokens_per_step: tuple[int, ...],
) -> None:
    if result.output_token_ids != EXPECTED_OUTPUT_TOKENS or result.text != EXPECTED_TEXT:
        raise RuntimeError("tiny greedy output differs from the frozen fixture result")
    if result.outcome != "FINISHED" or result.finish_reason != "MAX_NEW_TOKENS":
        raise RuntimeError("tiny greedy request did not finish at the requested length")
    if result.kv_mode != kv_mode:
        raise RuntimeError("result did not preserve the selected KV mode")
    if result.query_tokens_per_step != query_tokens_per_step:
        raise RuntimeError("query shape does not demonstrate the selected execution path")
    if result.kv_valid_tokens_per_step != kv_valid_tokens_per_step:
        raise RuntimeError("KV coverage does not match the selected execution path")
    if result.rng_draws_consumed != 0 or result.committed_draw_addresses:
        raise RuntimeError("greedy generation must not consume RNG draws")
    if any(result.resource_status.values()):
        raise RuntimeError("request resources did not return to baseline")


def _generate_without_kv() -> GenerationResult:
    engine = Engine(
        model="fixture:tiny-decoder",
        device="cpu",
        kv_mode="none",
        max_context_tokens=None,
        memory_budget_bytes=6 * 1024**3,
    )
    safely_closed = False
    try:
        result = engine.generate(_request(engine, "example-v02-generate-none"))
    finally:
        safely_closed = engine.close(timeout=1.0)
    if not safely_closed:
        raise RuntimeError("no-KV engine close did not prove safe completion")
    _assert_result(
        result,
        kv_mode="none",
        query_tokens_per_step=(6, 7, 8, 9),
        kv_valid_tokens_per_step=(0, 0, 0, 0),
    )
    _emit("generate-none", result.to_dict())
    return result


def _stream_with_contiguous_kv() -> GenerationResult:
    engine = Engine(
        model="fixture:tiny-decoder",
        device="cpu",
        kv_mode="contiguous",
        max_context_tokens=None,
        memory_budget_bytes=6 * 1024**3,
    )
    request = _request(engine, "example-v02-stream-contiguous")

    text_parts: list[str] = []
    chunk_kinds: list[str] = []
    streamed_tokens: list[int] = []
    output_indexes: list[int] = []
    terminal: TerminalChunk | None = None
    stream = engine.stream(request)
    safely_closed = False
    try:
        for chunk in stream:
            chunk_kinds.append(chunk.kind)
            _emit("stream-contiguous", chunk.to_dict())
            if isinstance(chunk, TokenChunk):
                text_parts.append(chunk.text_delta)
                streamed_tokens.append(chunk.token_id)
                output_indexes.append(chunk.output_index)
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
    if chunk_kinds != ["metadata", "token", "token", "token", "token", "terminal"]:
        raise RuntimeError("stream chunks were not metadata, ordered tokens, then terminal")
    if tuple(streamed_tokens) != EXPECTED_OUTPUT_TOKENS or output_indexes != [0, 1, 2, 3]:
        raise RuntimeError("stream token IDs or output indexes differ from the frozen fixture")
    if "".join(text_parts) != terminal.result.text:
        raise RuntimeError("streamed text differs from the committed canonical decode")
    if not safely_closed:
        raise RuntimeError("engine close did not prove safe completion")
    _assert_result(
        terminal.result,
        kv_mode="contiguous",
        query_tokens_per_step=(6, 1, 1, 1),
        kv_valid_tokens_per_step=(6, 7, 8, 9),
    )
    return terminal.result


def main() -> int:
    without_kv = _generate_without_kv()
    contiguous = _stream_with_contiguous_kv()
    if (without_kv.output_token_ids != contiguous.output_token_ids
            or without_kv.text != contiguous.text):
        raise RuntimeError("no-KV oracle and contiguous streaming output differ")
    _emit("comparison", {
        "output_token_ids": list(contiguous.output_token_ids),
        "text": contiguous.text,
        "none_query_tokens_per_step": list(without_kv.query_tokens_per_step),
        "contiguous_query_tokens_per_step": list(contiguous.query_tokens_per_step),
        "same_committed_output": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


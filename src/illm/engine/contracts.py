"""Tensor-free, explicitly versioned reference generation DTOs."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from illm.domain.errors import ErrorCode, IllmError
from illm.sampling import SamplingSpec


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str | Path = "fixture:tiny-decoder"
    prompt: str | None = None
    input_token_ids: tuple[int, ...] | None = None
    max_new_tokens: int = 8
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    add_bos: bool | None = None
    capture: Path | None = None
    telemetry: str = "request"

    def __post_init__(self) -> None:
        def invalid(message: str) -> None:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, message)

        if not isinstance(self.model, (str, Path)) or not str(self.model):
            invalid("model must be a nonempty local profile or fixture identifier")
        if (self.prompt is None) == (self.input_token_ids is None):
            invalid("provide exactly one of prompt and input_token_ids")
        if self.prompt is not None and not isinstance(self.prompt, str):
            invalid("prompt must be text")
        if self.input_token_ids is not None and (
            not isinstance(self.input_token_ids, tuple)
            or any(type(token) is not int or token < 0 for token in self.input_token_ids)
        ):
            invalid("input_token_ids must be an immutable tuple of nonnegative integers")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            invalid("max_new_tokens must be a positive integer")
        if not isinstance(self.sampling, SamplingSpec):
            invalid("sampling must be a canonical SamplingSpec")
        if not isinstance(self.request_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.request_id) is None:
            invalid("request_id must be an opaque identifier of at most128 safe ASCII characters")
        if self.add_bos is not None and type(self.add_bos) is not bool:
            invalid("add_bos must be bool or None")
        if self.capture is not None and not isinstance(self.capture, Path):
            invalid("capture must be a local Path")
        if not isinstance(self.telemetry, str) or self.telemetry not in {"off", "request", "ops"}:
            invalid("telemetry must be off, request or ops")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    finish_reason: str
    actual_seed: int
    model_spec_id: str
    sampling_spec_digest: str
    next_output_position: int
    rng_draws_consumed: int
    committed_draw_addresses: tuple[tuple[int, int, int], ...]
    timings: dict[str, Any]
    capture_manifest_digest: str | None
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        """Explicit result, not capture: output text/token IDs are user-visible."""
        return {
            "schema_version": 1, "request_id": self.request_id,
            "input_token_ids": list(self.input_token_ids), "output_token_ids": list(self.output_token_ids),
            "text": self.text, "finish_reason": self.finish_reason, "actual_seed": self.actual_seed,
            "model_spec_id": self.model_spec_id, "sampling_spec_digest": self.sampling_spec_digest,
            "next_output_position": self.next_output_position, "rng_draws_consumed": self.rng_draws_consumed,
            "committed_draw_addresses": [list(address) for address in self.committed_draw_addresses],
            "timings": dict(self.timings), "capture_manifest_digest": self.capture_manifest_digest,
            "event_count": self.event_count,
        }

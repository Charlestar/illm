"""Tensor-free V0.2 DTOs; V0.1 contracts remain at their original import path."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from illm.domain.errors import ErrorCode, IllmError
from illm.sampling import SamplingSpec


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model_spec_id: str
    max_new_tokens: int
    sampling: SamplingSpec
    prompt: str | None = None
    input_token_ids: tuple[int, ...] | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    add_bos: bool | None = None
    capture: Path | None = None
    telemetry: str = "metrics"

    def __post_init__(self) -> None:
        def invalid(message: str) -> None:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, message)

        if not isinstance(self.model_spec_id, str) or re.fullmatch(r"[0-9a-f]{64}", self.model_spec_id) is None:
            invalid("model_spec_id must identify the exact loaded model")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            invalid("max_new_tokens must be a positive integer")
        if not isinstance(self.sampling, SamplingSpec):
            invalid("sampling must be a canonical SamplingSpec")
        if (self.prompt is None) == (self.input_token_ids is None):
            invalid("provide exactly one of prompt and input_token_ids")
        if self.prompt is not None and not isinstance(self.prompt, str):
            invalid("prompt must be text")
        if self.input_token_ids is not None and (
            not isinstance(self.input_token_ids, tuple)
            or any(type(token) is not int or token < 0 for token in self.input_token_ids)
        ):
            invalid("input_token_ids must be an immutable tuple of nonnegative integers")
        if not isinstance(self.request_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", self.request_id) is None:
            invalid("request_id must be a nonempty opaque safe ASCII identifier")
        if self.add_bos is not None and type(self.add_bos) is not bool:
            invalid("add_bos must be bool or None")
        if self.capture is not None and not isinstance(self.capture, Path):
            invalid("capture must be a Path")
        if not isinstance(self.telemetry, str) or self.telemetry not in {"off", "metrics", "request", "ops"}:
            invalid("telemetry must be off, metrics, request or ops")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    input_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    text: str
    outcome: str
    finish_reason: str | None
    error_code: str | None
    actual_seed: int
    model_spec_id: str
    sampling_spec_digest: str
    kv_mode: str
    next_output_position: int
    rng_draws_consumed: int
    committed_draw_addresses: tuple[tuple[int, int, int], ...]
    query_tokens_per_step: tuple[int, ...]
    kv_valid_tokens_per_step: tuple[int, ...]
    resource_status: dict[str, int]
    timings: dict[str, Any]
    capture_manifest_digest: str | None
    capture_error: str | None
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        """User-facing result may contain text/IDs; canonical capture never does."""
        value = asdict(self)
        for name in ("input_token_ids", "output_token_ids", "query_tokens_per_step", "kv_valid_tokens_per_step"):
            value[name] = list(value[name])
        value["committed_draw_addresses"] = [list(address) for address in self.committed_draw_addresses]
        return {"schema_version": 2, **value}


@dataclass(frozen=True, slots=True)
class MetadataChunk:
    request_id: str
    model_spec_id: str
    sampling_spec_digest: str
    memory_plan_digest: str
    input_tokens: int
    kv_mode: str
    accepted_ns: int
    kind: str = field(default="metadata", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenChunk:
    request_id: str
    output_index: int
    token_id: int
    text_delta: str
    server_ready_ns: int
    kv_valid_token_count: int
    committed_output_count: int
    kind: str = field(default="token", init=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TerminalChunk:
    request_id: str
    outcome: str
    finish_reason: str | None
    error_code: str | None
    final_text_delta: str
    server_ready_ns: int
    result: GenerationResult
    kind: str = field(default="terminal", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "request_id": self.request_id, "outcome": self.outcome,
                "finish_reason": self.finish_reason, "error_code": self.error_code,
                "final_text_delta": self.final_text_delta, "server_ready_ns": self.server_ready_ns,
                "result": self.result.to_dict()}


@dataclass(frozen=True, slots=True)
class CancelResult:
    request_id: str
    cancellation_id: str
    outcome: str | None
    terminal_ready: bool
    won: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StreamChunk: TypeAlias = MetadataChunk | TokenChunk | TerminalChunk

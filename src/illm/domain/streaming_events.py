"""V0.2 metadata-only event envelope for stateful streaming R0 captures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping, cast

from illm.canonical import canonical_json_bytes, canonical_json_text
from illm.domain.errors import ErrorCode, IllmError


EVENT_ENVELOPE_VERSION = 1
EVENT_CATALOG = "illm-stateful-v02-v1"
EVENT_FIELDS = {
    "event_envelope_version", "event_catalog", "producer", "producer_seq_no",
    "monotonic_ns", "clock_domain", "kind", "kind_code", "replay_critical", "payload",
}

__all__ = [
    "EVENT_CATALOG",
    "EVENT_ENVELOPE_VERSION",
    "StreamingEvent",
    "StreamingEventKind",
    "thaw_metadata",
    "validate_metadata",
]


class StreamingEventKind(IntEnum):
    """Codes are new in V0.2 and must never be reused with another meaning."""

    REQUEST_TOKENIZED = 2000
    REQUEST_ACCEPTED = 2001
    REQUEST_PHASE_CHANGED = 2002
    REQUEST_FIRST_TOKEN = 2003
    REQUEST_OUTPUT_CHUNK = 2004
    REQUEST_COMPLETED = 2005
    REQUEST_CANCEL_REQUESTED = 2006
    REQUEST_CANCELLED = 2007
    REQUEST_FAILED = 2008
    REQUEST_TERMINAL_DECIDED = 2009

    MODEL_FORWARD_STARTED = 2100
    MODEL_FORWARD_COMPLETED = 2101
    MODEL_FORWARD = 2102

    KV_ALLOCATION_COMMITTED = 2200
    KV_WRITE_STARTED = 2201
    KV_WRITE_COMMITTED = 2202
    KV_WRITE_ABORTED = 2203
    KV_PAGE_RELEASED = 2204

    CLOCK_SNAPSHOT = 2300

    @property
    def wire_name(self) -> str:
        return _WIRE_NAMES[self]

    @property
    def replay_critical(self) -> bool:
        return self in REPLAY_CRITICAL_KINDS


# The namespace separator is a dot; underscores inside a leaf name are part of
# the frozen catalog spelling.  Keep this explicit so adding a multi-word event
# cannot silently change its wire identity.
_WIRE_NAMES: dict[StreamingEventKind, str] = {
    StreamingEventKind.REQUEST_TOKENIZED: "request.tokenized",
    StreamingEventKind.REQUEST_ACCEPTED: "request.accepted",
    StreamingEventKind.REQUEST_PHASE_CHANGED: "request.phase_changed",
    StreamingEventKind.REQUEST_FIRST_TOKEN: "request.first_token",
    StreamingEventKind.REQUEST_OUTPUT_CHUNK: "request.output_chunk",
    StreamingEventKind.REQUEST_COMPLETED: "request.completed",
    StreamingEventKind.REQUEST_CANCEL_REQUESTED: "request.cancel_requested",
    StreamingEventKind.REQUEST_CANCELLED: "request.cancelled",
    StreamingEventKind.REQUEST_FAILED: "request.failed",
    StreamingEventKind.REQUEST_TERMINAL_DECIDED: "request.terminal_decided",
    StreamingEventKind.MODEL_FORWARD_STARTED: "model.forward_started",
    StreamingEventKind.MODEL_FORWARD_COMPLETED: "model.forward_completed",
    StreamingEventKind.MODEL_FORWARD: "model.forward",
    StreamingEventKind.KV_ALLOCATION_COMMITTED: "kv.allocation_committed",
    StreamingEventKind.KV_WRITE_STARTED: "kv.write_started",
    StreamingEventKind.KV_WRITE_COMMITTED: "kv.write_committed",
    StreamingEventKind.KV_WRITE_ABORTED: "kv.write_aborted",
    StreamingEventKind.KV_PAGE_RELEASED: "kv.page_released",
    StreamingEventKind.CLOCK_SNAPSHOT: "clock_snapshot",
}


REPLAY_CRITICAL_KINDS = frozenset({
    StreamingEventKind.REQUEST_ACCEPTED,
    StreamingEventKind.REQUEST_PHASE_CHANGED,
    StreamingEventKind.REQUEST_OUTPUT_CHUNK,
    StreamingEventKind.REQUEST_COMPLETED,
    StreamingEventKind.REQUEST_CANCELLED,
    StreamingEventKind.REQUEST_FAILED,
    StreamingEventKind.REQUEST_TERMINAL_DECIDED,
    StreamingEventKind.KV_ALLOCATION_COMMITTED,
    StreamingEventKind.KV_WRITE_STARTED,
    StreamingEventKind.KV_WRITE_COMMITTED,
    StreamingEventKind.KV_WRITE_ABORTED,
    StreamingEventKind.KV_PAGE_RELEASED,
})


REQUIRED_PAYLOAD_FIELDS: dict[StreamingEventKind, frozenset[str]] = {
    StreamingEventKind.REQUEST_TOKENIZED: frozenset({"request_id", "started_ns", "ended_ns", "input_tokens"}),
    StreamingEventKind.REQUEST_ACCEPTED: frozenset({"request_id", "input_tokens", "max_new_tokens"}),
    StreamingEventKind.REQUEST_PHASE_CHANGED: frozenset({"request_id", "from_phase", "to_phase"}),
    StreamingEventKind.REQUEST_FIRST_TOKEN: frozenset({
        "request_id", "output_index", "committed_output_count", "kv_valid_token_count", "server_ready_ns",
    }),
    StreamingEventKind.REQUEST_OUTPUT_CHUNK: frozenset({
        "request_id", "output_index", "committed_output_count", "kv_valid_token_count", "server_ready_ns",
    }),
    StreamingEventKind.REQUEST_COMPLETED: frozenset({
        "request_id", "finish_reason", "committed_output_count", "kv_valid_token_count",
    }),
    StreamingEventKind.REQUEST_CANCEL_REQUESTED: frozenset({"request_id", "cancellation_id"}),
    StreamingEventKind.REQUEST_CANCELLED: frozenset({
        "request_id", "cancellation_id", "committed_output_count", "kv_valid_token_count",
    }),
    StreamingEventKind.REQUEST_FAILED: frozenset({
        "request_id", "error_code", "committed_output_count", "kv_valid_token_count",
    }),
    StreamingEventKind.REQUEST_TERMINAL_DECIDED: frozenset({
        "request_id", "outcome", "committed_output_count", "kv_valid_token_count",
    }),
    StreamingEventKind.MODEL_FORWARD_STARTED: frozenset({
        "request_id", "phase", "query_tokens", "context_tokens", "output_position",
    }),
    StreamingEventKind.MODEL_FORWARD_COMPLETED: frozenset({
        "request_id", "phase", "query_tokens", "context_tokens", "output_position",
    }),
    StreamingEventKind.MODEL_FORWARD: frozenset({
        "request_id", "phase", "query_tokens", "context_tokens", "output_position",
    }),
    StreamingEventKind.KV_ALLOCATION_COMMITTED: frozenset({
        "request_id", "allocation_id", "capacity_tokens", "allocated_bytes",
    }),
    StreamingEventKind.KV_WRITE_STARTED: frozenset({
        "request_id", "write_id", "allocation_id", "start_token", "token_count", "lease_id", "fence_id",
    }),
    StreamingEventKind.KV_WRITE_COMMITTED: frozenset({
        "request_id", "write_id", "allocation_id", "lease_id", "fence_id", "fence_completed",
        "kv_valid_token_count",
    }),
    StreamingEventKind.KV_WRITE_ABORTED: frozenset({
        "request_id", "write_id", "allocation_id", "lease_id", "fence_id", "fence_completed",
    }),
    StreamingEventKind.KV_PAGE_RELEASED: frozenset({
        "request_id", "allocation_id", "released_bytes", "fence_completed",
    }),
    StreamingEventKind.CLOCK_SNAPSHOT: frozenset({"host_monotonic_ns"}),
}

_NONNEGATIVE_INTS = frozenset({
    "started_ns", "ended_ns", "input_tokens", "max_new_tokens", "output_index",
    "committed_output_count", "kv_valid_token_count", "server_ready_ns", "query_tokens",
    "context_tokens", "output_position", "capacity_tokens", "allocated_bytes", "start_token",
    "token_count", "released_bytes", "host_monotonic_ns", "duration_ns",
    "kv_appended_output_count", "kv_valid_through_input_index", "active_leases", "pending_writes",
    "server_terminal_ready_ns",
})
_NONEMPTY_STRINGS = frozenset({
    "request_id", "cancellation_id", "from_phase", "to_phase", "finish_reason", "error_code",
    "outcome", "reason",
    "phase", "allocation_id", "write_id", "lease_id", "fence_id", "token_hmac",
})
_TERMINAL_DECISION_FIELDS = REQUIRED_PAYLOAD_FIELDS[
    StreamingEventKind.REQUEST_TERMINAL_DECIDED
] | frozenset({"reason", "cancellation_id", "error_code"})
_FORBIDDEN_PRIVATE_KEYS = frozenset({
    "prompt", "prompt_text", "output", "output_text", "text", "text_delta", "token_id",
    "token_ids", "input_ids", "input_token_ids", "output_ids", "output_token_ids", "logits",
    "key", "keys", "key_bytes", "hmac_key", "token_hmac_key", "kv_key",
})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def thaw_metadata(value: Any) -> Any:
    """Return ordinary JSON containers without exposing mutable event storage."""

    if isinstance(value, Mapping):
        return {str(key): thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_metadata(item) for item in value]
    return value


def _privacy_check(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming metadata keys must be nonempty strings")
            if key.lower() in _FORBIDDEN_PRIVATE_KEYS:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "metadata-only streaming capture forbids prompt/text/token/logit/KV contents",
                    {"field": f"{path}.{key}"},
                )
            _privacy_check(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _privacy_check(item, path=f"{path}[{index}]")


def validate_metadata(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) or not key for key in value):
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"{name} must be a string-keyed metadata object")
    result = cast(dict[str, Any], thaw_metadata(value))
    _privacy_check(result, path=name)
    try:
        canonical_json_bytes(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"{name} must contain finite JSON metadata") from exc
    return result


def _validate_payload(kind: StreamingEventKind, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_metadata(payload, name="payload")
    missing = REQUIRED_PAYLOAD_FIELDS[kind] - set(result)
    if missing:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "streaming event omits required metadata",
            {"kind": kind.wire_name, "missing": sorted(missing)},
        )
    for key in _NONNEGATIVE_INTS & set(result):
        if type(result[key]) is not int or result[key] < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"{key} must be a nonnegative integer")
    for key in _NONEMPTY_STRINGS & set(result):
        if not isinstance(result[key], str) or not result[key]:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"{key} must be a nonempty string")
    if kind is StreamingEventKind.REQUEST_TOKENIZED and result["ended_ns"] < result["started_ns"]:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "tokenization timestamps regress")
    if kind in {StreamingEventKind.REQUEST_ACCEPTED, StreamingEventKind.KV_ALLOCATION_COMMITTED,
                StreamingEventKind.KV_WRITE_STARTED}:
        positive = {
            StreamingEventKind.REQUEST_ACCEPTED: ("input_tokens", "max_new_tokens"),
            StreamingEventKind.KV_ALLOCATION_COMMITTED: ("capacity_tokens", "allocated_bytes"),
            StreamingEventKind.KV_WRITE_STARTED: ("token_count",),
        }[kind]
        if any(result[key] < 1 for key in positive):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event count/size must be positive")
    if kind in {StreamingEventKind.KV_WRITE_COMMITTED, StreamingEventKind.KV_WRITE_ABORTED,
                StreamingEventKind.KV_PAGE_RELEASED} and result["fence_completed"] is not True:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "KV commit/abort/release requires a completed fence")
    if (kind is StreamingEventKind.REQUEST_TERMINAL_DECIDED
            and result["outcome"] not in {"FINISHED", "CANCELLED", "FAILED"}):
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "logical terminal outcome is unsupported")
    if kind is StreamingEventKind.REQUEST_TERMINAL_DECIDED:
        unexpected = set(result) - _TERMINAL_DECISION_FIELDS
        if unexpected:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "logical terminal decision contains unsupported metadata",
                {"fields": sorted(unexpected)},
            )
    return result


@dataclass(frozen=True, slots=True)
class StreamingEvent:
    producer: str
    producer_seq_no: int
    monotonic_ns: int
    kind: StreamingEventKind
    payload: Mapping[str, Any]
    clock_domain: str = "host.perf_counter_ns"
    event_envelope_version: int = EVENT_ENVELOPE_VERSION
    event_catalog: str = EVENT_CATALOG

    def __post_init__(self) -> None:
        if type(self.event_envelope_version) is not int or self.event_envelope_version != EVENT_ENVELOPE_VERSION:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported streaming event envelope")
        if self.event_catalog != EVENT_CATALOG:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported streaming event catalog")
        if not isinstance(self.producer, str) or not self.producer:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event producer is invalid")
        if type(self.producer_seq_no) is not int or self.producer_seq_no < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming producer sequence is invalid")
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event timestamp is invalid")
        if self.clock_domain != "host.perf_counter_ns":
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event uses an unsupported clock domain")
        if not isinstance(self.kind, StreamingEventKind) or not isinstance(self.payload, Mapping):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event kind/payload is invalid")
        object.__setattr__(self, "payload", _freeze(_validate_payload(self.kind, self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_envelope_version": self.event_envelope_version,
            "event_catalog": self.event_catalog,
            "producer": self.producer,
            "producer_seq_no": self.producer_seq_no,
            "monotonic_ns": self.monotonic_ns,
            "clock_domain": self.clock_domain,
            "kind": self.kind.wire_name,
            "kind_code": int(self.kind),
            "replay_critical": self.kind.replay_critical,
            "payload": thaw_metadata(self.payload),
        }

    def to_json_line(self) -> str:
        return canonical_json_text(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamingEvent:
        if not isinstance(value, Mapping) or set(value) != EVENT_FIELDS:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event fields do not match envelope v1")
        if type(value["event_envelope_version"]) is not int or value["event_envelope_version"] != EVENT_ENVELOPE_VERSION:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported streaming event envelope")
        if value["event_catalog"] != EVENT_CATALOG or type(value["kind_code"]) is not int:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "streaming event registry identity is unsupported")
        try:
            kind = StreamingEventKind(value["kind_code"])
        except ValueError as exc:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unknown required streaming event kind") from exc
        if value["kind"] != kind.wire_name or value["replay_critical"] is not kind.replay_critical:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming event registry fields disagree")
        return cls(
            producer=value["producer"],
            producer_seq_no=value["producer_seq_no"],
            monotonic_ns=value["monotonic_ns"],
            kind=kind,
            payload=value["payload"],
            clock_domain=value["clock_domain"],
            event_envelope_version=value["event_envelope_version"],
            event_catalog=value["event_catalog"],
        )

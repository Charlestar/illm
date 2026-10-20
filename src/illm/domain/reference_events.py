"""Version-one metadata events for the synchronous reference engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

from illm.canonical import canonical_json_bytes, canonical_json_text
from illm.domain.errors import ErrorCode, IllmError


EVENT_FIELDS = {
    "schema_version", "producer", "seq_no", "monotonic_ns", "kind", "kind_code",
    "replay_critical", "payload",
}


class ReferenceEventKind(IntEnum):
    REQUEST_ACCEPTED = 1000
    REQUEST_TOKENIZED = 1001
    MODEL_FORWARD_STARTED = 1100
    MODEL_OP = 1101
    MODEL_FORWARD_COMPLETED = 1102
    SAMPLING_STARTED = 1200
    SAMPLING_SELECTED = 1201
    SAMPLING_FAILED = 1202
    REQUEST_FINISHED = 1300
    REQUEST_FAILED = 1301

    @property
    def wire_name(self) -> str:
        return self.name.lower().replace("_", ".")

    @property
    def replay_critical(self) -> bool:
        return True


PAYLOAD_FIELDS = {
    ReferenceEventKind.REQUEST_ACCEPTED: {"request_id", "max_new_tokens"},
    ReferenceEventKind.REQUEST_TOKENIZED: {"request_id", "input_tokens", "input_digest"},
    ReferenceEventKind.MODEL_FORWARD_STARTED: {"request_id", "output_position", "context_tokens"},
    ReferenceEventKind.MODEL_OP: {"request_id", "output_position", "op_name", "shape", "dtype"},
    ReferenceEventKind.MODEL_FORWARD_COMPLETED: {"request_id", "output_position", "logits_digest"},
    ReferenceEventKind.SAMPLING_STARTED: {"request_id", "output_position", "sampling_spec_digest"},
    ReferenceEventKind.SAMPLING_SELECTED: {
        "request_id", "output_position", "sampling_spec_digest", "logits_digest", "draw_address", "selected_digest",
    },
    ReferenceEventKind.SAMPLING_FAILED: {"request_id", "output_position", "error_code"},
    ReferenceEventKind.REQUEST_FINISHED: {"request_id", "output_tokens", "output_digest", "finish_reason"},
    ReferenceEventKind.REQUEST_FAILED: {"request_id", "error_code"},
}
SUMMARY_FIELDS = {"min", "max", "mean", "l2", "finite_count", "nonfinite_count"}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    producer: str
    seq_no: int
    monotonic_ns: int
    kind: ReferenceEventKind
    payload: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported reference event schema")
        if not isinstance(self.producer, str) or not self.producer:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event producer is invalid")
        if any(type(value) is not int or value < 0 for value in (self.seq_no, self.monotonic_ns)):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event sequence/timestamp is invalid")
        if not isinstance(self.kind, ReferenceEventKind) or not isinstance(self.payload, Mapping):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event kind/payload is invalid")
        fields = set(self.payload)
        required = PAYLOAD_FIELDS[self.kind]
        optional = {"summary", "duration_ns"} if self.kind is ReferenceEventKind.MODEL_OP else set()
        if not required <= fields or fields - required - optional:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference payload fields violate metadata-only contract")
        payload = _thaw(self.payload)
        if "summary" in payload and (
            not isinstance(payload["summary"], dict) or set(payload["summary"]) - SUMMARY_FIELDS
        ):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "op summary fields violate metadata-only contract")
        try:
            canonical_json_bytes(payload)
        except (ValueError, TypeError, OverflowError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference payload must be finite JSON metadata") from exc
        object.__setattr__(self, "payload", _freeze(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "producer": self.producer,
            "seq_no": self.seq_no, "monotonic_ns": self.monotonic_ns,
            "kind": self.kind.wire_name, "kind_code": int(self.kind),
            "replay_critical": True, "payload": _thaw(self.payload),
        }

    def to_json_line(self) -> str:
        return canonical_json_text(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceEvent:
        if not isinstance(value, Mapping) or set(value) != EVENT_FIELDS:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event fields do not match schema one")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported reference event schema")
        if type(value["kind_code"]) is not int:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "reference event kind code must be an integer")
        try:
            kind = ReferenceEventKind(value["kind_code"])
        except ValueError as exc:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unknown required reference event kind") from exc
        if value["kind"] != kind.wire_name or value["replay_critical"] is not True:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event registry identity mismatch")
        return cls(value["producer"], value["seq_no"], value["monotonic_ns"], kind, value["payload"])

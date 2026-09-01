from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Mapping

from illm.canonical import canonical_json_text
from illm.domain.errors import ErrorCode, IllmError


EVENT_FIELDS = {
    "schema_version",
    "producer",
    "seq_no",
    "monotonic_ns",
    "kind",
    "kind_code",
    "replay_critical",
    "payload",
}


class EventKind(IntEnum):
    REQUEST_ARRIVED = 100
    REQUEST_SELECTED = 101
    REQUEST_DEFERRED = 102
    REQUEST_COMPLETED = 103
    SCHEDULER_STEP_STARTED = 200
    SCHEDULER_STEP_COMPLETED = 201
    CHECKPOINT = 300
    CAPTURE_TRUNCATED = 301
    DIAGNOSTIC = 900

    @property
    def wire_name(self) -> str:
        return self.name.lower().replace("_", ".")

    @property
    def replay_critical(self) -> bool:
        return self is not EventKind.DIAGNOSTIC


@dataclass(frozen=True, slots=True)
class DomainEvent:
    producer: str
    seq_no: int
    monotonic_ns: int
    kind: EventKind
    payload: Mapping[str, Any]
    schema_version: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 0:
            raise ValueError("schema_version must be 0")
        if not isinstance(self.producer, str) or not self.producer:
            raise ValueError("producer must be a non-empty string")
        if isinstance(self.seq_no, bool) or not isinstance(self.seq_no, int):
            raise ValueError("seq_no must be an integer")
        if self.seq_no < 0:
            raise ValueError("seq_no must be non-negative")
        if isinstance(self.monotonic_ns, bool) or not isinstance(self.monotonic_ns, int):
            raise ValueError("monotonic_ns must be an integer")
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be non-negative")
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be an EventKind")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be an object")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "seq_no": self.seq_no,
            "monotonic_ns": self.monotonic_ns,
            "kind": self.kind.wire_name,
            "kind_code": int(self.kind),
            "replay_critical": self.kind.replay_critical,
            "payload": dict(self.payload),
        }

    def to_json_line(self) -> str:
        return canonical_json_text(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DomainEvent":
        if not isinstance(value, Mapping):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "DomainEvent must be an object")
        fields = set(value)
        if fields != EVENT_FIELDS:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "DomainEvent fields do not match schema v0",
                {"missing": sorted(EVENT_FIELDS - fields), "extra": sorted(fields - EVENT_FIELDS)},
            )
        if value.get("schema_version") != 0:
            raise IllmError(
                ErrorCode.UNSUPPORTED_SCHEMA,
                "unsupported DomainEvent schema",
                {"schema_version": value.get("schema_version")},
            )
        try:
            kind = EventKind(int(value["kind_code"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise IllmError(
                ErrorCode.UNSUPPORTED_SCHEMA,
                "unknown or missing event kind code",
                {"kind_code": value.get("kind_code")},
            ) from exc
        if value.get("kind") != kind.wire_name:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "event kind name/code mismatch",
                {"kind": value.get("kind"), "kind_code": int(kind)},
            )
        if value.get("replay_critical") is not kind.replay_critical:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "event replay-critical flag does not match registry",
                {"kind_code": int(kind)},
            )
        producer = value.get("producer")
        seq_no = value.get("seq_no")
        monotonic_ns = value.get("monotonic_ns")
        payload = value.get("payload")
        if not isinstance(producer, str) or not producer:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event producer is invalid")
        if isinstance(seq_no, bool) or not isinstance(seq_no, int) or seq_no < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event sequence number is invalid")
        if isinstance(monotonic_ns, bool) or not isinstance(monotonic_ns, int) or monotonic_ns < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event monotonic timestamp is invalid")
        if not isinstance(payload, dict):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event payload must be an object")
        return cls(producer, seq_no, monotonic_ns, kind, payload)

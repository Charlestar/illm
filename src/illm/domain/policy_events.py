"""Policy catalog dispatch without reinterpreting historical stateful events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import EVENT_CATALOG, StreamingEvent


POLICY_EVENT_CATALOG = "illm-stateful-v02-policy-v1"


@dataclass(frozen=True, slots=True)
class PolicyStreamingEvent(StreamingEvent):
    event_catalog: str = POLICY_EVENT_CATALOG

    def __post_init__(self) -> None:
        if self.event_catalog != POLICY_EVENT_CATALOG:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported policy event catalog")
        # Reuse unchanged field/payload validation, never serialize the temporary
        # old-catalog object or teach its reader to silently accept this family.
        validated = StreamingEvent(self.producer, self.producer_seq_no, self.monotonic_ns, self.kind,
                                   self.payload, self.clock_domain, self.event_envelope_version)
        object.__setattr__(self, "payload", validated.payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyStreamingEvent:
        if value.get("event_catalog") != POLICY_EVENT_CATALOG:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported policy event catalog")
        temporary = dict(value)
        temporary["event_catalog"] = EVENT_CATALOG
        event = StreamingEvent.from_dict(temporary)
        return cls(event.producer, event.producer_seq_no, event.monotonic_ns, event.kind, event.payload,
                   event.clock_domain, event.event_envelope_version)

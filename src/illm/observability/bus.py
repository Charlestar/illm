from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent


class EventSink(Protocol):
    def emit(self, event: DomainEvent) -> None: ...


@dataclass(slots=True)
class NoopSink:
    def emit(self, event: DomainEvent) -> None:
        del event


@dataclass(slots=True)
class BoundedMemorySink:
    capacity: int
    _queue: deque[DomainEvent] = field(init=False)
    dropped_diagnostic_count: int = 0
    critical_drop_count: int = 0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        self._queue = deque()

    def emit(self, event: DomainEvent) -> None:
        if len(self._queue) < self.capacity:
            self._queue.append(event)
            return
        if not event.kind.replay_critical:
            self.dropped_diagnostic_count += 1
            return
        for index, queued in enumerate(self._queue):
            if not queued.kind.replay_critical:
                del self._queue[index]
                self.dropped_diagnostic_count += 1
                self._queue.append(event)
                return
        self.critical_drop_count += 1
        raise IllmError(
            ErrorCode.CAPTURE_INCOMPLETE,
            "bounded sink has no safe slot for a replay-critical event",
            {"capacity": self.capacity, "seq_no": event.seq_no},
        )

    def drain(self) -> tuple[DomainEvent, ...]:
        result = tuple(self._queue)
        self._queue.clear()
        return result


@dataclass(slots=True)
class EventBus:
    required_sinks: tuple[EventSink, ...]
    best_effort_sinks: tuple[EventSink, ...] = ()
    exporter_error_count: int = 0

    def emit(self, event: DomainEvent) -> None:
        for sink in self.required_sinks:
            sink.emit(event)
        for sink in self.best_effort_sinks:
            try:
                sink.emit(event)
            except Exception:
                self.exporter_error_count += 1

from __future__ import annotations

import pytest

from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.observability import BoundedMemorySink, EventBus, NoopSink


def _event(seq: int, kind: EventKind) -> DomainEvent:
    return DomainEvent("test", seq, seq + 1, kind, {})


def test_diagnostic_events_drop_without_unbounded_growth() -> None:
    sink = BoundedMemorySink(capacity=3)
    for seq in range(10_000):
        sink.emit(_event(seq, EventKind.DIAGNOSTIC))
    assert len(sink.drain()) == 3
    assert sink.dropped_diagnostic_count == 9_997
    assert sink.critical_drop_count == 0


def test_critical_event_evicts_diagnostic_before_drop() -> None:
    sink = BoundedMemorySink(capacity=2)
    sink.emit(_event(0, EventKind.DIAGNOSTIC))
    sink.emit(_event(1, EventKind.REQUEST_ARRIVED))
    sink.emit(_event(2, EventKind.REQUEST_COMPLETED))
    drained = sink.drain()
    assert [event.seq_no for event in drained] == [1, 2]
    assert sink.dropped_diagnostic_count == 1
    assert sink.critical_drop_count == 0


def test_critical_overflow_fails_loudly() -> None:
    sink = BoundedMemorySink(capacity=1)
    sink.emit(_event(0, EventKind.REQUEST_ARRIVED))
    with pytest.raises(IllmError) as raised:
        sink.emit(_event(1, EventKind.REQUEST_COMPLETED))
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE
    assert sink.critical_drop_count == 1


def test_best_effort_exporter_failure_does_not_escape_domain_loop() -> None:
    class FailingSink:
        def emit(self, event: DomainEvent) -> None:
            del event
            raise RuntimeError("exporter stalled")

    required = BoundedMemorySink(capacity=2)
    bus = EventBus((required,), (FailingSink(), NoopSink()))
    event = _event(0, EventKind.REQUEST_ARRIVED)
    bus.emit(event)
    assert required.drain() == (event,)
    assert bus.exporter_error_count == 1

from __future__ import annotations

from threading import Event, Thread
from typing import Any

import pytest

from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.observability import BoundedMemorySink, EventBus, NoopSink


def _event(seq: int, kind: EventKind) -> DomainEvent:
    return DomainEvent("test", seq, seq + 1, kind, {})


@pytest.mark.parametrize("capacity", [False, True, 0, -1, 2.5, float("nan"), "2", None])
def test_bounded_queue_capacity_is_a_positive_integer(capacity: Any) -> None:
    with pytest.raises(ValueError):
        BoundedMemorySink(capacity=capacity)
    with pytest.raises(ValueError):
        EventBus((), best_effort_capacity=capacity)


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
    assert bus.close()
    assert bus.exporter_error_count == 1


def test_blocking_exporter_has_bounded_queue_and_does_not_block_required_events() -> None:
    entered = Event()
    release = Event()
    finished = Event()

    class BlockingSink:
        def emit(self, event: DomainEvent) -> None:
            del event
            entered.set()
            release.wait()

    required = BoundedMemorySink(capacity=2)
    bus = EventBus((required,), (BlockingSink(),), best_effort_capacity=3)
    delivered: list[int] = []

    def produce() -> None:
        for seq in range(10_000):
            bus.emit(_event(seq + 1, EventKind.REQUEST_COMPLETED))
            delivered.append(required.drain()[0].seq_no)
        finished.set()

    producer = Thread(target=produce, daemon=True)
    try:
        bus.emit(_event(0, EventKind.DIAGNOSTIC))
        required.drain()
        assert entered.wait(2)
        producer.start()
        assert finished.wait(5), "producer stalled behind optional exporter"
        assert not release.is_set()
        assert delivered == list(range(1, 10_001))
        assert required.critical_drop_count == 0
        assert bus.exporter_queue_high_watermark == 3
        assert bus.exporter_drop_count == 9_997
        assert bus.exporter_critical_drop_count == 9_997
        assert not bus.close(timeout=0.01)
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(2)
        assert bus.close(timeout=2)
    with pytest.raises(ValueError, match="closed"):
        bus.emit(_event(10_001, EventKind.DIAGNOSTIC))


def test_required_failure_still_propagates_before_optional_delivery() -> None:
    required = BoundedMemorySink(capacity=1)
    bus = EventBus((required,))
    bus.emit(_event(0, EventKind.REQUEST_ARRIVED))
    with pytest.raises(IllmError, match="no safe slot"):
        bus.emit(_event(1, EventKind.REQUEST_COMPLETED))
    assert bus.close()


def test_async_exporter_queue_prefers_critical_facts_to_diagnostics() -> None:
    entered = Event()
    release = Event()
    received: list[int] = []

    class RecordingSink:
        def emit(self, event: DomainEvent) -> None:
            entered.set()
            release.wait()
            received.append(event.seq_no)

    bus = EventBus((), (RecordingSink(),), best_effort_capacity=2)
    try:
        bus.emit(_event(0, EventKind.DIAGNOSTIC))
        assert entered.wait(2)
        bus.emit(_event(1, EventKind.DIAGNOSTIC))
        bus.emit(_event(2, EventKind.REQUEST_ARRIVED))
        bus.emit(_event(3, EventKind.REQUEST_COMPLETED))
        assert bus.exporter_drop_count == 1
        assert bus.exporter_critical_drop_count == 0
    finally:
        release.set()
        assert bus.close(timeout=2)
    assert received == [0, 2, 3]


def test_exporter_cannot_mutate_nested_required_fact_payload() -> None:
    class MutatingSink:
        def emit(self, event: DomainEvent) -> None:
            event.payload["selected"].append("corruption")

    required = BoundedMemorySink(capacity=1)
    original = DomainEvent("test", 0, 1, EventKind.SCHEDULER_STEP_COMPLETED, {"selected": ["req-0"]})
    bus = EventBus((required,), (MutatingSink(),))
    bus.emit(original)
    assert bus.close(timeout=2)
    assert required.drain()[0].payload["selected"] == ["req-0"]
    assert original.payload["selected"] == ["req-0"]

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Callable, Protocol

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
class _ExporterWorker:
    sink: EventSink
    capacity: int
    on_error: Callable[[], None]
    on_drop: Callable[[bool], None]
    _queue: deque[DomainEvent] = field(default_factory=deque, init=False)
    _condition: Condition = field(default_factory=Condition, init=False)
    _closing: bool = field(default=False, init=False)
    _high_watermark: int = field(default=0, init=False)
    _thread: Thread = field(init=False)

    def __post_init__(self) -> None:
        # A blocked plugin cannot be killed safely in Python. Keep exactly one
        # worker per exporter, make it daemon, and report a bounded close timeout.
        self._thread = Thread(target=self._run, name="illm-exporter", daemon=True)
        self._thread.start()

    def offer(self, event: DomainEvent) -> None:
        with self._condition:
            if self._closing:
                self.on_drop(event.kind.replay_critical)
                return
            if len(self._queue) == self.capacity:
                if event.kind.replay_critical:
                    for index, queued in enumerate(self._queue):
                        if not queued.kind.replay_critical:
                            del self._queue[index]
                            self.on_drop(False)
                            break
                    else:
                        self.on_drop(True)
                        return
                else:
                    self.on_drop(False)
                    return
            self._queue.append(event)
            self._high_watermark = max(self._high_watermark, len(self._queue))
            self._condition.notify()

    @property
    def high_watermark(self) -> int:
        with self._condition:
            return self._high_watermark

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closing:
                    self._condition.wait()
                if not self._queue:
                    return
                event = self._queue.popleft()
            try:
                # DomainEvent freezes its outer mapping, but JSON arrays/objects
                # inside it can still be mutable. Plugins get their own copy.
                self.sink.emit(replace(event, payload=deepcopy(dict(event.payload))))
            except Exception:
                self.on_error()

    def request_close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


@dataclass(slots=True)
class EventBus:
    required_sinks: tuple[EventSink, ...]
    best_effort_sinks: tuple[EventSink, ...] = ()
    exporter_error_count: int = 0
    best_effort_capacity: int = 256
    exporter_drop_count: int = field(default=0, init=False)
    exporter_critical_drop_count: int = field(default=0, init=False)
    _workers: tuple[_ExporterWorker, ...] = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.best_effort_capacity < 1:
            raise ValueError("best_effort_capacity must be positive")
        self._workers = tuple(
            _ExporterWorker(sink, self.best_effort_capacity, self._record_error, self._record_drop)
            for sink in self.best_effort_sinks
        )

    def _record_error(self) -> None:
        with self._lock:
            self.exporter_error_count += 1

    def _record_drop(self, critical: bool) -> None:
        with self._lock:
            self.exporter_drop_count += 1
            self.exporter_critical_drop_count += int(critical)

    @property
    def exporter_queue_high_watermark(self) -> int:
        return max((worker.high_watermark for worker in self._workers), default=0)

    def emit(self, event: DomainEvent) -> None:
        with self._lock:
            if self._closed:
                raise ValueError("event bus is closed")
        for sink in self.required_sinks:
            sink.emit(event)
        for worker in self._workers:
            worker.offer(event)

    def close(self, timeout: float = 1.0) -> bool:
        """Drain exporters within one shared timeout; False means a worker is stuck.

        Required sinks are synchronous and never depend on these workers. A
        permanently blocked exporter retains at most capacity queued events and
        one in-flight event. Calling close again can finish cooperative cleanup.
        """
        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        with self._lock:
            self._closed = True
        for worker in self._workers:
            worker.request_close()
        deadline = monotonic() + timeout
        stopped = True
        for worker in self._workers:
            stopped = worker.join(max(0.0, deadline - monotonic())) and stopped
        return stopped

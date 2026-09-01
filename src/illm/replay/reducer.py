from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, cast

from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.domain.state import RequestStatus, ToyRequest, ToyState


def _require_request(state: ToyState, request_id: str) -> ToyRequest:
    try:
        return state.request_map()[request_id]
    except KeyError as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "event references an unknown request",
            {"request_id": request_id, "step": state.current_step},
        ) from exc


def _step_id(payload: Mapping[str, Any]) -> int:
    value = payload.get("step_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event step_id is invalid")
    return value


def _request_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("request_id")
    if not isinstance(value, str) or not value:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event request_id is invalid")
    return value


def _apply_event(state: ToyState, event: DomainEvent) -> ToyState:
    """Apply one canonical event without importing or invoking the scheduler."""

    payload = event.payload
    if event.kind is EventKind.SCHEDULER_STEP_STARTED:
        step_id = _step_id(payload)
        if step_id != state.current_step + 1:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "scheduler steps are not contiguous",
                {"expected": state.current_step + 1, "actual": step_id},
            )
        return replace(state, current_step=step_id)

    if event.kind is EventKind.REQUEST_ARRIVED:
        request_id = _request_id(payload)
        arrival_step = payload.get("arrival_step")
        if isinstance(arrival_step, bool) or not isinstance(arrival_step, int) or arrival_step < 0:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event arrival_step is invalid")
        request_map = state.request_map()
        if request_id in request_map:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "duplicate request arrival",
                {"request_id": request_id},
            )
        request_map[request_id] = ToyRequest(request_id, arrival_step)
        return state.with_requests(request_map.values())

    if event.kind in {EventKind.REQUEST_SELECTED, EventKind.REQUEST_DEFERRED}:
        request = _require_request(state, _request_id(payload))
        if request.status is not RequestStatus.WAITING:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "only waiting requests may be selected or deferred",
                {"request_id": request.request_id, "status": request.status.name},
            )
        return state

    if event.kind is EventKind.REQUEST_COMPLETED:
        request_id = _request_id(payload)
        request_map = state.request_map()
        request = _require_request(state, request_id)
        if request.status is RequestStatus.FINISHED:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "request has more than one terminal event",
                {"request_id": request_id},
            )
        request_map[request_id] = replace(request, status=RequestStatus.FINISHED)
        return state.with_requests(request_map.values())

    if event.kind in {
        EventKind.SCHEDULER_STEP_COMPLETED,
        EventKind.CHECKPOINT,
        EventKind.CAPTURE_TRUNCATED,
        EventKind.DIAGNOSTIC,
    }:
        return state

    raise IllmError(
        ErrorCode.UNSUPPORTED_SCHEMA,
        "event kind is not supported by reducer v0",
        {"kind_code": int(event.kind)},
    )


def apply_event(state: ToyState, event: DomainEvent) -> ToyState:
    """Apply one event and normalize malformed payloads to CAPTURE_CORRUPT."""

    try:
        return _apply_event(state, event)
    except IllmError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            "event payload is invalid for its kind",
            {"kind_code": int(event.kind), "seq_no": event.seq_no},
        ) from exc


def first_difference(expected: object, actual: object, path: str = "$") -> str | None:
    """Return the first deterministic structural difference between two values."""

    if type(expected) is not type(actual):
        return path
    if isinstance(expected, dict):
        actual_dict = cast(dict[Any, Any], actual)
        expected_keys = set(expected)
        actual_keys = set(actual_dict)
        for key in sorted(expected_keys | actual_keys):
            if key not in expected_keys or key not in actual_keys:
                return f"{path}.{key}"
            difference = first_difference(expected[key], actual_dict[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        actual_list = cast(list[Any], actual)
        if len(expected) != len(actual_list):
            return f"{path}.length"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual_list)):
            difference = first_difference(expected_item, actual_item, f"{path}[{index}]")
            if difference:
                return difference
        return None
    return None if expected == actual else path


@dataclass(slots=True)
class ToyReplayReducer:
    """Transactional R1 reducer; it never imports or invokes the scheduler."""

    state: ToyState = field(default_factory=ToyState)
    active_step: int | None = None
    selected: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)

    def _require_active_step(self, event: DomainEvent) -> int:
        if self.active_step is None:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "event appears outside a scheduler step",
                {"kind_code": int(event.kind), "seq_no": event.seq_no},
            )
        event_step = _step_id(event.payload)
        if event_step != self.active_step:
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "event step_id does not match active scheduler step",
                {"expected": self.active_step, "actual": event_step},
            )
        return event_step

    def apply(self, event: DomainEvent) -> ToyState:
        if event.kind is EventKind.SCHEDULER_STEP_STARTED:
            if self.active_step is not None:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "scheduler steps overlap")
            self.active_step = _step_id(event.payload)
            self.selected.clear()
            self.deferred.clear()
            self.completed.clear()
            self.state = apply_event(self.state, event)
            return self.state

        if event.kind is EventKind.REQUEST_ARRIVED:
            if self.active_step is None:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "request arrival appears outside a scheduler step",
                    {"seq_no": event.seq_no},
                )
            arrival_step = event.payload.get("arrival_step")
            if arrival_step != self.active_step:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "arrival is recorded in the wrong step")
            self.state = apply_event(self.state, event)
            return self.state

        if event.kind in {EventKind.REQUEST_SELECTED, EventKind.REQUEST_DEFERRED}:
            self._require_active_step(event)
            request_id = _request_id(event.payload)
            if request_id in self.selected or request_id in self.deferred:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "request has duplicate or conflicting scheduler decisions",
                    {"request_id": request_id, "step_id": self.active_step},
                )
            reason_code = event.payload.get("reason_code")
            if not isinstance(reason_code, str) or not reason_code:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "scheduler reason_code is invalid")
            target = self.selected if event.kind is EventKind.REQUEST_SELECTED else self.deferred
            target.append(request_id)
            self.state = apply_event(self.state, event)
            return self.state

        if event.kind is EventKind.REQUEST_COMPLETED:
            self._require_active_step(event)
            request_id = _request_id(event.payload)
            if event.payload.get("terminal") != "FINISHED":
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "terminal event is invalid")
            if request_id not in self.selected or request_id in self.completed:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "only a selected request may complete exactly once in its step",
                    {"request_id": request_id, "step_id": self.active_step},
                )
            self.completed.append(request_id)
            self.state = apply_event(self.state, event)
            return self.state

        if event.kind is EventKind.SCHEDULER_STEP_COMPLETED:
            self._require_active_step(event)
            payload_selected = event.payload.get("selected")
            payload_deferred = event.payload.get("deferred")
            if not isinstance(payload_selected, list) or not all(
                isinstance(item, str) for item in payload_selected
            ):
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "step selected summary is invalid")
            if not isinstance(payload_deferred, list) or not all(
                isinstance(item, str) for item in payload_deferred
            ):
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "step deferred summary is invalid")
            waiting = [request.request_id for request in self.state.waiting()]
            if (
                payload_selected != self.selected
                or payload_deferred != self.deferred
                or self.completed != self.selected
                or set(waiting) != set(self.deferred)
                or len(waiting) != len(self.deferred)
            ):
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "scheduler step transaction is inconsistent",
                    {
                        "step_id": self.active_step,
                        "selected": self.selected,
                        "deferred": self.deferred,
                        "completed": self.completed,
                        "waiting": waiting,
                    },
                )
            self.state = apply_event(self.state, event)
            self.active_step = None
            return self.state

        if event.kind is EventKind.CAPTURE_TRUNCATED:
            raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "capture contains a truncation event")

        self.state = apply_event(self.state, event)
        return self.state

    def finish(self) -> ToyState:
        if self.active_step is not None:
            raise IllmError(
                ErrorCode.CAPTURE_INCOMPLETE,
                "capture ends inside a scheduler step",
                {"step_id": self.active_step},
            )
        return self.state


def replay_events(
    initial: ToyState,
    events: Iterable[DomainEvent],
    expected_step_digests: Iterable[Mapping[str, Any]],
) -> ToyState:
    """Replay R1 transactions and assert every recorded per-step state digest."""

    expected_by_step = {int(item["step_id"]): item for item in expected_step_digests}
    reducer = ToyReplayReducer(initial)
    for event in events:
        state = reducer.apply(event)
        if event.kind is EventKind.SCHEDULER_STEP_COMPLETED:
            step_id = _step_id(event.payload)
            expected_record = expected_by_step.get(step_id)
            expected_digest = None if expected_record is None else expected_record.get("state_digest")
            if expected_digest != state.digest:
                expected_state = None if expected_record is None else expected_record.get("state")
                raise IllmError(
                    ErrorCode.REPLAY_DIVERGED,
                    "state digest diverged at scheduler step",
                    {
                        "step_id": step_id,
                        "first_path": first_difference(expected_state, state.to_dict()),
                        "expected": expected_digest,
                        "actual": state.digest,
                    },
                )
    return reducer.finish()


def reduce_events(initial: ToyState, events: Iterable[DomainEvent]) -> ToyState:
    reducer = ToyReplayReducer(initial)
    for event in events:
        reducer.apply(event)
    return reducer.finish()

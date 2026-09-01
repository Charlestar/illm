from __future__ import annotations

import json

import pytest

from illm.canonical import canonical_json_text
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.domain.state import ToyRequest, ToyState
from illm.scheduler import build_plan


def test_canonical_json_has_stable_field_order() -> None:
    assert canonical_json_text({"z": 1, "a": {"y": 2, "b": 3}}) == '{"a":{"b":3,"y":2},"z":1}'


def test_event_canonical_round_trip() -> None:
    event = DomainEvent(
        "producer-a",
        7,
        123456,
        EventKind.REQUEST_DEFERRED,
        {"request_id": "req-1", "reason_code": "BATCH_CAPACITY"},
    )
    decoded = DomainEvent.from_dict(json.loads(event.to_json_line()))
    assert decoded == event
    assert decoded.kind.wire_name == "request.deferred"
    assert decoded.kind.replay_critical


def test_unknown_event_code_fails_closed() -> None:
    value = DomainEvent("p", 0, 1, EventKind.DIAGNOSTIC, {}).to_dict()
    value["kind_code"] = 99999
    value["kind"] = "future.event"
    with pytest.raises(IllmError, match="unknown or missing") as raised:
        DomainEvent.from_dict(value)
    assert raised.value.code is ErrorCode.UNSUPPORTED_SCHEMA


def test_event_extra_field_and_replay_flag_tamper_fail_closed() -> None:
    value = DomainEvent("p", 0, 1, EventKind.DIAGNOSTIC, {}).to_dict()
    value["extra"] = "future"
    with pytest.raises(IllmError, match="fields do not match") as extra:
        DomainEvent.from_dict(value)
    assert extra.value.code is ErrorCode.CAPTURE_CORRUPT

    value.pop("extra")
    value["replay_critical"] = True
    with pytest.raises(IllmError, match="replay-critical flag") as flag:
        DomainEvent.from_dict(value)
    assert flag.value.code is ErrorCode.CAPTURE_CORRUPT


def test_reference_scheduler_is_pure_and_covers_eligible_requests() -> None:
    state = ToyState(
        4,
        (
            ToyRequest("req-b", 2),
            ToyRequest("req-a", 2),
            ToyRequest("req-old", 1),
        ),
    )
    before = state.to_dict()
    plan = build_plan(state, 5)
    assert plan.selected == ("req-old",)
    assert tuple(item.request_id for item in plan.deferred) == ("req-a", "req-b")
    assert set(plan.selected).isdisjoint(item.request_id for item in plan.deferred)
    assert set(plan.selected) | {item.request_id for item in plan.deferred} == {
        request.request_id for request in state.waiting()
    }
    assert state.to_dict() == before

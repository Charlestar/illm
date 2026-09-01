from __future__ import annotations

from dataclasses import dataclass

from illm.domain.state import ToyState


@dataclass(frozen=True, slots=True)
class DeferredRequest:
    request_id: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class ToySchedulePlanV0:
    step_id: int
    selected: tuple[str, ...]
    deferred: tuple[DeferredRequest, ...]
    max_selected: int = 1


def build_plan(state: ToyState, step_id: int, *, max_selected: int = 1) -> ToySchedulePlanV0:
    """Pure reference scheduler: oldest request first, stable ID as tie-breaker."""

    if max_selected < 1:
        raise ValueError("max_selected must be positive")
    eligible = sorted(state.waiting(), key=lambda request: (request.arrival_step, request.request_id))
    selected = tuple(request.request_id for request in eligible[:max_selected])
    deferred = tuple(
        DeferredRequest(request.request_id, "BATCH_CAPACITY")
        for request in eligible[max_selected:]
    )
    return ToySchedulePlanV0(step_id, selected, deferred, max_selected)


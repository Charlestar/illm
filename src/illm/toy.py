from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from illm.capture import CaptureSession
from illm.domain.events import DomainEvent, EventKind
from illm.domain.state import ToyState
from illm.replay import ToyReplayReducer
from illm.scheduler import ToySchedulePlanV0, build_plan


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: int
    arrivals: tuple[str, ...]
    plan: ToySchedulePlanV0
    state_digest: str


@dataclass(frozen=True, slots=True)
class ToyRunResult:
    state: ToyState
    steps: tuple[StepResult, ...]
    event_count: int
    capture_manifest: dict[str, object]


def _arrival_counts(seed: int, steps: int) -> tuple[int, ...]:
    cutoff = max(1, steps // 3)
    total_requests = min(steps, cutoff + max(1, cutoff // 2))
    slots = [step for step in range(cutoff) for _ in range(2)]
    random.Random(seed).shuffle(slots)
    counts = [0] * steps
    for step in slots[:total_requests]:
        counts[step] += 1
    return tuple(counts)


def run_toy_scheduler(
    *,
    seed: int,
    steps: int,
    capture_root: Path,
    project_root: Path,
) -> ToyRunResult:
    if steps < 1:
        raise ValueError("steps must be positive")
    session = CaptureSession(capture_root, seed, steps, project_root)
    reducer = ToyReplayReducer(ToyState())
    sequence = 0
    request_index = 0
    results: list[StepResult] = []
    arrivals_by_step = _arrival_counts(seed, steps)

    def emit(kind: EventKind, payload: dict[str, object]) -> None:
        nonlocal sequence
        event = DomainEvent("toy-engine", sequence, time.monotonic_ns(), kind, payload)
        session.append(event)
        reducer.apply(event)
        sequence += 1

    for step_id, arrival_count in enumerate(arrivals_by_step):
        emit(EventKind.SCHEDULER_STEP_STARTED, {"step_id": step_id})
        arrival_ids: list[str] = []
        for _ in range(arrival_count):
            request_id = f"req-{request_index:04d}"
            request_index += 1
            arrival_ids.append(request_id)
            emit(
                EventKind.REQUEST_ARRIVED,
                {"request_id": request_id, "arrival_step": step_id},
            )
        plan = build_plan(reducer.state, step_id)
        for request_id in plan.selected:
            emit(
                EventKind.REQUEST_SELECTED,
                {"request_id": request_id, "step_id": step_id, "reason_code": "OLDEST_FIRST"},
            )
        for deferred in plan.deferred:
            emit(
                EventKind.REQUEST_DEFERRED,
                {
                    "request_id": deferred.request_id,
                    "step_id": step_id,
                    "reason_code": deferred.reason_code,
                },
            )
        for request_id in plan.selected:
            emit(
                EventKind.REQUEST_COMPLETED,
                {"request_id": request_id, "step_id": step_id, "terminal": "FINISHED"},
            )
        emit(
            EventKind.SCHEDULER_STEP_COMPLETED,
            {
                "step_id": step_id,
                "selected": list(plan.selected),
                "deferred": [item.request_id for item in plan.deferred],
            },
        )
        session.record_step_digest(step_id, reducer.state)
        results.append(StepResult(step_id, tuple(arrival_ids), plan, reducer.state.digest))

    state = reducer.finish()
    manifest = session.finalize(state)
    return ToyRunResult(state, tuple(results), sequence, manifest)

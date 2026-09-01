from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Iterable

from illm.canonical import sha256_json


class RequestStatus(IntEnum):
    WAITING = 1
    FINISHED = 2


@dataclass(frozen=True, order=True, slots=True)
class ToyRequest:
    request_id: str
    arrival_step: int
    status: RequestStatus = RequestStatus.WAITING

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")
        if not isinstance(self.status, RequestStatus):
            raise ValueError("status must be a RequestStatus")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_step": self.arrival_step,
            "status": self.status.name,
            "status_code": int(self.status),
        }


@dataclass(frozen=True, slots=True)
class ToyState:
    current_step: int = -1
    requests: tuple[ToyRequest, ...] = ()

    def __post_init__(self) -> None:
        if self.current_step < -1:
            raise ValueError("current_step must be -1 or greater")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique")

    def request_map(self) -> dict[str, ToyRequest]:
        return {request.request_id: request for request in self.requests}

    def waiting(self) -> tuple[ToyRequest, ...]:
        return tuple(request for request in self.requests if request.status is RequestStatus.WAITING)

    def completed(self) -> tuple[ToyRequest, ...]:
        return tuple(request for request in self.requests if request.status is RequestStatus.FINISHED)

    def with_requests(self, requests: Iterable[ToyRequest]) -> "ToyState":
        return replace(self, requests=tuple(sorted(requests)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 0,
            "current_step": self.current_step,
            "requests": [request.to_dict() for request in sorted(self.requests)],
            "waiting_count": len(self.waiting()),
            "completed_count": len(self.completed()),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

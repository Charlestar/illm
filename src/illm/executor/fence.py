"""Completion fences for explicit eager-executor resource ownership."""

from __future__ import annotations

from typing import Protocol

import torch


class CompletionFence(Protocol):
    """Minimal fence contract.

    ``is_complete`` deliberately has a third, unknown state.  Callers must not
    recycle a resource when completion cannot be established.
    """

    def wait(self) -> None: ...

    def is_complete(self) -> bool | None: ...

    @property
    def succeeded(self) -> bool | None: ...


class ImmediateFence:
    """Already-completed CPU work."""

    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error

    def wait(self) -> None:
        if self._error is not None:
            raise self._error

    def is_complete(self) -> bool | None:
        return True

    @property
    def succeeded(self) -> bool | None:
        return self._error is None


class EventFence:
    """A CUDA event recorded after all work owned by one ticket."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event
        self._complete = False
        self._succeeded: bool | None = None
        self._unknown = False

    @classmethod
    def record(cls, device: torch.device) -> EventFence:
        if device.type != "cuda":
            raise ValueError("EventFence requires a CUDA device")
        with torch.cuda.device(device):
            event = torch.cuda.Event(enable_timing=False)  # type: ignore[no-untyped-call]
            event.record(torch.cuda.current_stream(device))  # type: ignore[no-untyped-call]
        return cls(event)

    def wait(self) -> None:
        if self._complete:
            if self._succeeded is not True:
                raise RuntimeError("CUDA completion fence failed")
            return
        if self._unknown:
            raise RuntimeError("CUDA completion state is unknown")
        try:
            self._event.synchronize()
        except BaseException:
            # A failed synchronize does not prove that every address formerly
            # touched by the stream is safe for application-level reuse.
            self._succeeded = False
            self._unknown = True
            raise
        self._complete = True
        self._succeeded = True

    def is_complete(self) -> bool | None:
        if self._unknown:
            return None
        if self._complete:
            return True
        try:
            ready = bool(self._event.query())  # type: ignore[no-untyped-call]
        except BaseException:
            self._unknown = True
            self._succeeded = False
            return None
        if ready:
            self._complete = True
            self._succeeded = True
        return ready

    @property
    def succeeded(self) -> bool | None:
        return self._succeeded


def record_completion_fence(device: torch.device) -> CompletionFence:
    """Record a fence after the current eager launch."""

    return ImmediateFence() if device.type == "cpu" else EventFence.record(device)

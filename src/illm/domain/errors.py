from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    TOOLCHAIN_MISMATCH = "TOOLCHAIN_MISMATCH"
    TOOL_NOT_IN_LOCKED_ENV = "TOOL_NOT_IN_LOCKED_ENV"
    CAPTURE_INCOMPLETE = "CAPTURE_INCOMPLETE"
    CAPTURE_CORRUPT = "CAPTURE_CORRUPT"
    REPLAY_DIVERGED = "REPLAY_DIVERGED"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    INTERNAL = "INTERNAL"


EXIT_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 64,
    ErrorCode.TOOL_NOT_IN_LOCKED_ENV: 69,
    ErrorCode.TOOLCHAIN_MISMATCH: 70,
    ErrorCode.CAPTURE_INCOMPLETE: 74,
    ErrorCode.CAPTURE_CORRUPT: 75,
    ErrorCode.REPLAY_DIVERGED: 76,
    ErrorCode.UNSUPPORTED_SCHEMA: 78,
    ErrorCode.INTERNAL: 1,
}


@dataclass(slots=True)
class IllmError(Exception):
    code: ErrorCode
    message: str
    details: Mapping[str, Any] | None = None

    @property
    def exit_code(self) -> int:
        return EXIT_CODE[self.code]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            result["details"] = dict(self.details)
        return result


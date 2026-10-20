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
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    INVALID_MODEL_CONFIG = "INVALID_MODEL_CONFIG"
    INVALID_WEIGHTS = "INVALID_WEIGHTS"
    DUPLICATE_WEIGHT = "DUPLICATE_WEIGHT"
    MODEL_ARTIFACT_MISSING = "MODEL_ARTIFACT_MISSING"
    UNSUPPORTED_SAMPLING_PARAMETER = "UNSUPPORTED_SAMPLING_PARAMETER"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    ORACLE_DIVERGED = "ORACLE_DIVERGED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    ENGINE_BUSY = "ENGINE_BUSY"
    INTERNAL = "INTERNAL"


EXIT_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 64,
    ErrorCode.TOOL_NOT_IN_LOCKED_ENV: 69,
    ErrorCode.TOOLCHAIN_MISMATCH: 70,
    ErrorCode.CAPTURE_INCOMPLETE: 74,
    ErrorCode.CAPTURE_CORRUPT: 75,
    ErrorCode.REPLAY_DIVERGED: 76,
    ErrorCode.UNSUPPORTED_SCHEMA: 78,
    ErrorCode.UNSUPPORTED_MODEL: 80,
    ErrorCode.INVALID_MODEL_CONFIG: 81,
    ErrorCode.INVALID_WEIGHTS: 82,
    ErrorCode.DUPLICATE_WEIGHT: 83,
    ErrorCode.MODEL_ARTIFACT_MISSING: 84,
    ErrorCode.UNSUPPORTED_SAMPLING_PARAMETER: 85,
    ErrorCode.NUMERICAL_ERROR: 86,
    ErrorCode.ORACLE_DIVERGED: 87,
    ErrorCode.RESOURCE_EXHAUSTED: 88,
    ErrorCode.ENGINE_BUSY: 89,
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


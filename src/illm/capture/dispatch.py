"""Strict metadata-only dispatch; historical readers keep their own contracts."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from illm.domain.errors import ErrorCode, IllmError

if TYPE_CHECKING:
    from illm.capture.bundle import CaptureBundle
    from illm.capture.reference import ReferenceCaptureBundle
    from illm.capture.streaming import StreamingCaptureBundle
    from illm.capture.policy_streaming import PolicyStreamingCaptureBundle


def _plain(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture path is unreadable") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture path cannot contain links or reparse points")


def capture_schema(path: Path) -> int:
    """Read only a strict discriminator, not a claim that the capture is valid."""
    requested = Path(path).absolute()
    manifest_path = requested / "capture-manifest.json"
    _plain(manifest_path)
    if not requested.is_dir() or not manifest_path.is_file():
        raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "capture manifest is missing or unreadable")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "duplicate capture manifest field")
            result[key] = value
        return result

    def nonfinite(value: str) -> NoReturn:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "non-finite capture manifest value", {"value": value})

    try:
        header = json.loads(manifest_path.read_text(encoding="utf-8"),
                            object_pairs_hook=unique, parse_constant=nonfinite)
    except IllmError:
        raise
    except (ValueError, UnicodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture manifest is not valid JSON") from exc
    except OSError as exc:
        raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "capture manifest is unreadable") from exc
    schema = header.get("schema_version") if isinstance(header, dict) else None
    if type(schema) is not int or schema not in {0, 1, 2, 3}:
        raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported capture schema")
    return schema


def open_streaming_capture(path: Path) -> StreamingCaptureBundle | PolicyStreamingCaptureBundle:
    """Fully verify one R0 stateful family, without broadening either reader."""
    schema = capture_schema(path)
    if schema == 2:
        from illm.capture.streaming import StreamingCaptureBundle
        return StreamingCaptureBundle.open(path)
    if schema == 3:
        from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
        return PolicyStreamingCaptureBundle.open(path)
    raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "request story requires a stateful capture", {"schema_version": schema})


def open_capture(path: Path) -> CaptureBundle | ReferenceCaptureBundle | StreamingCaptureBundle | PolicyStreamingCaptureBundle:
    """Fully verify a known family; a discriminator alone never grants replay."""
    schema = capture_schema(path)
    if schema == 0:
        from illm.capture.bundle import CaptureBundle
        return CaptureBundle.open(path)
    if schema == 1:
        from illm.capture.reference import ReferenceCaptureBundle
        return ReferenceCaptureBundle.open(path)
    return open_streaming_capture(path)

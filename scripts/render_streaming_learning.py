"""Produce derived V0.2 Request Story PNG/SVG/trace from an actual R0 bundle."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Sequence

from illm.canonical import canonical_json_text
from illm.domain.errors import ErrorCode, IllmError
from illm.streaming_learning import derive_streaming_learning_assets, verify_streaming_learning_assets


def _plain_absolute(path: Path, *, label: str) -> Path:
    """Keep caller-visible path topology so capture/output links fail closed."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                f"{label} path is unreadable",
            ) from exc
        if stat.S_ISLNK(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & 0x400:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                f"{label} path cannot contain links or reparse points",
            )
    return absolute


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true", help="read-only exact regeneration check")
    args = parser.parse_args(argv)
    try:
        output = _plain_absolute(args.output, label="learning output")
        capture = _plain_absolute(args.capture, label="capture")
        if output.is_relative_to(capture) or capture.is_relative_to(output):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "learning output must be separate from the immutable capture")
        if args.verify_existing:
            print(canonical_json_text(verify_streaming_learning_assets(capture, output)))
            return 0
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "learning output must be a new or empty directory")
        assets = derive_streaming_learning_assets(capture)
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "INCOMPLETE"
        with marker.open("x", encoding="ascii") as stream:
            stream.write("derived_assets_in_progress\n")
        try:
            for name, data in assets.items():
                with (output / name).open("xb") as stream:
                    stream.write(data)
            marker.unlink()
            result = verify_streaming_learning_assets(capture, output)
        except BaseException:
            marker.touch(exist_ok=True)
            raise
        print(canonical_json_text(result))
        return 0
    except IllmError as exc:
        print(canonical_json_text(exc.to_dict()), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError) as exc:
        error = IllmError(ErrorCode.INVALID_ARGUMENT, "learning artifact I/O failed",
                          {"exception_type": type(exc).__name__})
        print(canonical_json_text(error.to_dict()), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

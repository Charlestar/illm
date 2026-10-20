"""Seal a source-bound V0.1 profile; publication and tags are separate actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import create_reference_profile_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--toolchain-manifest", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation-directory", type=Path, action="append", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--sampler-fixture", type=Path, required=True)
    parser.add_argument("--clean-room-report", type=Path)
    parser.add_argument("--gpu-failure-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile = create_reference_profile_evidence(profile_id=args.profile_id, toolchain_manifest=args.toolchain_manifest,
                                                    junit=args.junit, quality=args.quality, capture=args.capture,
                                                    validation_directories=args.validation_directory, benchmark=args.benchmark,
                                                    sampler_fixture=args.sampler_fixture, output=args.output,
                                                    clean_room_report=args.clean_room_report,
                                                    gpu_failure_evidence=args.gpu_failure_evidence)
        print(json.dumps(profile, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = IllmError(ErrorCode.INVALID_ARGUMENT, "reference qualification input is invalid", {"exception_type": type(exc).__name__})
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

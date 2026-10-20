"""Seal one completed source-bound V0.2 gate root; never bundle or promote."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.stateful_qualification import create_stateful_profile_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile = create_stateful_profile_evidence(
            profile_id=args.profile_id,
            gate_root=args.gate_root,
            output=args.output,
        )
        print(json.dumps(profile, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "V0.2 qualification input is invalid",
            {"exception_type": type(exc).__name__},
        )
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

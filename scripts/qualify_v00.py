from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.qualification import create_profile_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a finalized V0.0 platform qualification artifact")
    parser.add_argument("--toolchain-manifest", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        profile = create_profile_evidence(
            toolchain_manifest=args.toolchain_manifest,
            junit=args.junit,
            quality=args.quality,
            capture=args.capture,
            output=args.output,
        )
        print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError) as exc:
        failure = IllmError(ErrorCode.INVALID_ARGUMENT, str(exc))
        print(json.dumps(failure.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

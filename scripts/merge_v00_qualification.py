from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.qualification import merge_profile_evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge verified platform evidence into V0.0")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--base-toolchain-manifest", type=Path, required=True)
    parser.add_argument("--profile-evidence", type=Path, action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = merge_profile_evidence(
            release_dir=args.release_dir,
            base_toolchain_manifest=args.base_toolchain_manifest,
            evidence_roots=args.profile_evidence,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
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

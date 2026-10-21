"""Assemble a V0.2 candidate from complete real evidence; never tag or promote."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.stateful_qualification import stateful_source_identity
from illm.stateful_release import build_stateful_release_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, action="append", required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--documentation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_stateful_release_bundle(evidence_roots=args.evidence_root, rollback=args.rollback,
                                               documentation_root=args.documentation_root, output=args.output,
                                               expected_source_identity=stateful_source_identity(Path(__file__).resolve().parents[1]))
        print(json.dumps(result, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = IllmError(ErrorCode.INVALID_ARGUMENT, "V0.2 release input is invalid",
                            {"exception_type": type(exc).__name__})
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

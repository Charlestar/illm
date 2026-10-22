"""Seal a complete revision-2 CPU profile; never build a release or tag Git."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from illm.domain.errors import IllmError
from illm.stateful_policy_qualification import create_policy_stateful_profile_evidence
from illm.v02_profile_dispatch import verify_v02_profile_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--gate-root", type=Path)
    modes.add_argument("--verify-existing", type=Path)
    parser.add_argument("--profile-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.verify_existing is not None and (args.profile_id is not None or args.output is not None):
        parser.error("--verify-existing does not accept producer arguments")
    if args.gate_root is not None and (args.profile_id is None or args.output is None):
        parser.error("sealing requires --profile-id and --output")
    try:
        if args.verify_existing is not None:
            profile, _ = verify_v02_profile_evidence(args.verify_existing)
        else:
            profile = create_policy_stateful_profile_evidence(
                profile_id=args.profile_id, gate_root=args.gate_root, output=args.output,
            )
        print(json.dumps(profile, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "fail", "exception_type": type(exc).__name__, "message": str(exc)},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

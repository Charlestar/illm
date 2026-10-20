from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from illm.canonical import canonical_json_text
from illm.capture import CaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.release import verify_release_bundle, write_verified_release_state
from illm.replay import replay_events
from illm.toy import run_toy_scheduler


class IllmArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _parser() -> IllmArgumentParser:
    parser = IllmArgumentParser(prog="illm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo")
    demo_subparsers = demo.add_subparsers(dest="demo_name", required=True)
    toy = demo_subparsers.add_parser("toy-scheduler")
    toy.add_argument("--seed", type=int, required=True)
    toy.add_argument("--steps", type=int, required=True)
    toy.add_argument("--capture", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("capture", type=Path)
    replay.add_argument("--mode", choices=("trace", "state"), required=True)
    replay.add_argument("--assert-state-digest", action="store_true")
    replay.add_argument("--request-id")
    replay.add_argument("--step-id", type=int)

    release = subparsers.add_parser("release")
    release_subparsers = release.add_subparsers(dest="release_command", required=True)
    verify = release_subparsers.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--output", type=Path)
    return parser


def _demo(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--steps must be positive")
    project_root = Path.cwd().resolve()
    result = run_toy_scheduler(
        seed=args.seed,
        steps=args.steps,
        capture_root=args.capture,
        project_root=project_root,
    )
    print("step arrivals selected deferred reason")
    for step in result.steps:
        selected = ",".join(step.plan.selected) or "-"
        deferred = ",".join(item.request_id for item in step.plan.deferred) or "-"
        reasons = ",".join(sorted({item.reason_code for item in step.plan.deferred})) or "-"
        print(f"{step.step_id:>4} {','.join(step.arrivals) or '-':<18} {selected:<10} {deferred:<18} {reasons}")
    print(
        canonical_json_text(
            {
                "capture": str(args.capture.resolve()),
                "completed_requests": len(result.state.completed()),
                "event_count": result.event_count,
                "final_state_digest": result.state.digest,
            }
        )
    )
    return 0


def _replay(args: argparse.Namespace) -> int:
    bundle = CaptureBundle.open(args.capture)
    if args.mode == "trace":
        print("seq kind step request reason")
        displayed_events = 0
        for event in bundle.events:
            payload = event.payload
            if args.request_id and payload.get("request_id") != args.request_id:
                continue
            event_step = payload.get("step_id", payload.get("arrival_step"))
            if args.step_id is not None and event_step != args.step_id:
                continue
            displayed_events += 1
            print(
                f"{event.seq_no:>4} {event.kind.wire_name:<28} "
                f"{str(event_step if event_step is not None else '-'):<4} "
                f"{str(payload.get('request_id', '-')):<10} "
                f"{str(payload.get('reason_code', '-'))}"
            )
        print(
            canonical_json_text(
                {"mode": "R0", "displayed_events": displayed_events, "total_events": len(bundle.events)}
            )
        )
        return 0

    if args.request_id or args.step_id is not None:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "trace filters require --mode trace")

    expected_by_step = {int(item["step_id"]): item for item in bundle.expected_step_digests}
    state = replay_events(bundle.initial_state, bundle.events, bundle.expected_step_digests)
    expected_final = str(bundle.manifest["final_state_digest"])
    if args.assert_state_digest and state.digest != expected_final:
        raise IllmError(
            ErrorCode.REPLAY_DIVERGED,
            "final state digest does not match capture",
            {"expected": expected_final, "actual": state.digest},
        )
    print(
        canonical_json_text(
            {
                "mode": "R1_TOY",
                "asserted": bool(args.assert_state_digest),
                "steps": len(expected_by_step),
                "final_state_digest": state.digest,
            }
        )
    )
    return 0


def _release_verify(args: argparse.Namespace) -> int:
    result = verify_release_bundle(args.manifest)
    if args.output:
        write_verified_release_state(result, args.output)
    print(canonical_json_text(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "demo":
            return _demo(args)
        if args.command == "replay":
            return _replay(args)
        if args.command == "release" and args.release_command == "verify":
            return _release_verify(args)
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "unsupported command")
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError) as exc:
        error = IllmError(ErrorCode.INVALID_ARGUMENT, str(exc))
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

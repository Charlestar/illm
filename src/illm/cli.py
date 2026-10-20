from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from illm.canonical import canonical_json_text
from illm.capture import CaptureBundle
from illm.domain.errors import EXIT_CODE, ErrorCode, IllmError
from illm.reference_qualification import reference_source_identity
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

    model = subparsers.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    inspect = model_commands.add_parser("inspect")
    inspect.add_argument("--model", default="fixture:tiny-decoder")
    validate = model_commands.add_parser("validate")
    model_source = validate.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model")
    model_source.add_argument("--model-config", type=Path)
    validate.add_argument("--against", choices=("transformers",), required=True)
    validate.add_argument("--device", default="cpu")
    validate.add_argument("--output-directory", type=Path, default=Path("out/v0.1-validation"))

    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", default="fixture:tiny-decoder")
    input_source = generate.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--prompt")
    input_source.add_argument("--input-token-ids", help="JSON integer array, e.g. [1,104,105]")
    generate.add_argument("--max-new-tokens", type=int, default=8)
    generate.add_argument("--sampling-mode", choices=("greedy", "sample"), default="greedy")
    generate.add_argument("--temperature", type=float)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--request-id")
    generate.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=None)
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--capture", type=Path)
    generate.add_argument("--telemetry", choices=("off", "request", "ops"), default="request")

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
    verify.add_argument("--require-current-source", action="store_true")
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
    # Read only the version discriminator; the selected reader then verifies
    # exact fields, duplicate keys, file hashes, state digests and completeness.
    try:
        header = json.loads((args.capture / "capture-manifest.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "capture manifest is missing or unreadable") from exc
    except ValueError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture manifest is not valid JSON") from exc
    schema = header.get("schema_version") if isinstance(header, dict) else None
    if type(schema) is not int or schema not in {0, 1}:
        raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported capture schema")
    if schema == 1:
        return _reference_replay(args)
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


def _reference_replay(args: argparse.Namespace) -> int:
    from illm.capture.reference import ReferenceCaptureBundle

    bundle = ReferenceCaptureBundle.open(args.capture)
    if args.mode == "trace":
        print("seq kind output_position op")
        displayed = 0
        for event in bundle.events:
            if args.request_id and event.payload["request_id"] != args.request_id:
                continue
            if args.step_id is not None and event.payload.get("output_position") != args.step_id:
                continue
            displayed += 1
            print(f"{event.seq_no:>4} {event.kind.wire_name:<28} "
                  f"{str(event.payload.get('output_position', '-')):<4} {event.payload.get('op_name', '-')}")
        print(canonical_json_text({"mode": "R0", "displayed_events": displayed, "total_events": len(bundle.events),
                                   "capture_schema_version": 1, "privacy_mode": "metadata_only"}))
        return 0
    if args.request_id or args.step_id is not None:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "trace filters require --mode trace")
    # open() has already replayed every control transition and compared every
    # recorded state. This is never a numerical model rerun (R2 is unsupported).
    print(canonical_json_text({"mode": "R1_REFERENCE", "asserted": bool(args.assert_state_digest),
                               "numerical_reexecution": False, "final_state_digest": bundle.manifest["final_state_digest"],
                               "status": bundle.final_state["status"], "output_tokens": bundle.final_state["output_tokens"],
                               "rng_draws_consumed": bundle.final_state["rng_draws_consumed"]}))
    return 0


def _model(args: argparse.Namespace) -> int:
    if args.model_command == "inspect":
        from illm.model.loader import inspect_model

        print(canonical_json_text(inspect_model(args.model)))
        return 0
    from illm.model.validation import validate_model

    result = validate_model(args.model_config if args.model_config is not None else args.model,
                            device=args.device, output_directory=args.output_directory)
    print(canonical_json_text(result))
    return 0 if result["status"] == "pass" else EXIT_CODE[ErrorCode.ORACLE_DIVERGED]


def _generate(args: argparse.Namespace) -> int:
    from illm.engine import Engine, GenerationRequest
    from illm.sampling import SamplingSpec

    ids = None
    if args.input_token_ids is not None:
        value = json.loads(args.input_token_ids)
        if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "--input-token-ids must be a JSON array of nonnegative integers")
        ids = tuple(value)
    sampling = SamplingSpec(mode=args.sampling_mode,
                            temperature=args.temperature if args.temperature is not None else (1.0 if args.sampling_mode == "sample" else 0.0),
                            top_k=args.top_k, top_p=args.top_p, actual_seed=args.seed if args.seed is not None else secrets.randbits(64))
    request_options = {"request_id": args.request_id} if args.request_id is not None else {}
    request = GenerationRequest(model=args.model, prompt=args.prompt, input_token_ids=ids,
                                max_new_tokens=args.max_new_tokens, sampling=sampling, add_bos=args.add_bos,
                                capture=args.capture, telemetry=args.telemetry, **request_options)
    engine = Engine(args.model, device=args.device)
    try:
        result = engine.generate(request)
        print(canonical_json_text(result.to_dict()))
        return 0
    finally:
        engine.close()


def _release_verify(args: argparse.Namespace) -> int:
    if args.require_current_source:
        from illm.reference_release import verify_reference_release_bundle

        result = verify_reference_release_bundle(
            args.manifest,
            expected_source_identity=reference_source_identity(Path.cwd()),
        )
    else:
        result = verify_release_bundle(args.manifest)
    if args.output:
        write_verified_release_state(result, args.output)
    print(canonical_json_text(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # Canonical JSON is UTF-8, including redirected Windows consoles whose
    # inherited code page may otherwise reject generated Unicode characters.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="strict")
    try:
        args = _parser().parse_args(argv)
        if args.command == "demo":
            return _demo(args)
        if args.command == "replay":
            return _replay(args)
        if args.command == "model":
            return _model(args)
        if args.command == "generate":
            return _generate(args)
        if args.command == "release" and args.release_command == "verify":
            return _release_verify(args)
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "unsupported command")
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        from illm.engine.facade import as_illm_error
        from illm.model.spec import ModelError
        from illm.sampling import SamplingError

        error = as_illm_error(exc) if isinstance(exc, (ModelError, SamplingError, ImportError, RuntimeError)) else IllmError(ErrorCode.INVALID_ARGUMENT, str(exc))
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

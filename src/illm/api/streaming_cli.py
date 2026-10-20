"""V0.2 CLI adapter; root ``illm.cli`` owns top-level integration/error mapping."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, TextIO, TypeVar

from illm.canonical import canonical_json_bytes, canonical_json_text
from illm.domain.errors import EXIT_CODE, EmergencyAbandon, ErrorCode, IllmError
from illm.sampling import SamplingSpec


DEFAULT_MEMORY_BUDGET_BYTES = 6 * 1024**3
ParserT = TypeVar("ParserT", bound=argparse.ArgumentParser)


def add_streaming_commands(
    subparsers: argparse._SubParsersAction[ParserT],
) -> None:
    """Register V0.2 commands without changing the legacy root parser."""

    run = subparsers.add_parser("run")
    model_source = run.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model")
    model_source.add_argument("--model-config", type=Path)
    run.add_argument("--device", default="cpu")
    input_source = run.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--prompt")
    input_source.add_argument("--input-token-ids", help="JSON integer array, e.g. [1,104,105]")
    run.add_argument("--max-new-tokens", type=int, default=8)
    run.add_argument("--stream", action="store_true")
    run.add_argument("--sampling-mode", choices=("greedy", "sample"), default="greedy")
    run.add_argument("--temperature", type=float)
    run.add_argument("--top-k", type=int, default=0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--seed", type=int)
    run.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=None)
    run.add_argument("--kv-mode", choices=("none", "contiguous"), default="contiguous")
    run.add_argument("--max-context-tokens", type=int)
    run.add_argument("--memory-budget-bytes", type=int, default=DEFAULT_MEMORY_BUDGET_BYTES)
    run.add_argument("--request-id", default="req-0001")
    run.add_argument("--capture", type=Path)
    run.add_argument("--telemetry", choices=("off", "metrics", "request", "ops"), default="metrics")
    run.add_argument("--cancel-after-tokens", type=int)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--fixture", choices=("tiny-decoder",), required=True)
    verify.add_argument("--output-directory", type=Path)

    inspect = subparsers.add_parser("inspect")
    inspect_commands = inspect.add_subparsers(dest="inspect_command", required=True)
    request = inspect_commands.add_parser("request")
    request.add_argument("capture", type=Path)
    request.add_argument("--request", required=True)
    request.add_argument("--chrome-trace", type=Path)


def dispatch(args: argparse.Namespace) -> int | None:
    """Dispatch a V0.2 command, or return ``None`` for an unrelated command."""

    if args.command == "run":
        return _run(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "inspect" and args.inspect_command == "request":
        return _inspect_request(args)
    return None


def _streaming_api() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    from illm.engine.streaming import Engine
    from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk, TerminalChunk, TokenChunk

    return Engine, GenerationRequest, MetadataChunk, TokenChunk, TerminalChunk


def _parse_token_ids(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "--input-token-ids must be a JSON array of nonnegative integers",
        ) from exc
    if not isinstance(value, list) or any(type(token) is not int or token < 0 for token in value):
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "--input-token-ids must be a JSON array of nonnegative integers",
        )
    return tuple(value)


def _sampling(args: argparse.Namespace) -> SamplingSpec:
    temperature = args.temperature
    if temperature is None:
        temperature = 1.0 if args.sampling_mode == "sample" else 0.0
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    return SamplingSpec(
        mode=args.sampling_mode,
        temperature=temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        actual_seed=seed,
    )


def _validate_run_args(args: argparse.Namespace) -> None:
    if type(args.max_new_tokens) is not int or args.max_new_tokens < 1:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--max-new-tokens must be positive")
    if args.max_context_tokens is not None and (
        type(args.max_context_tokens) is not int or args.max_context_tokens < 1
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--max-context-tokens must be positive")
    if type(args.memory_budget_bytes) is not int or args.memory_budget_bytes < 1:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--memory-budget-bytes must be positive")
    if args.cancel_after_tokens is not None and (
        type(args.cancel_after_tokens) is not int or args.cancel_after_tokens < 0
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--cancel-after-tokens must be nonnegative")


@dataclass(slots=True)
class _InterruptController:
    engine: Any
    request_id: str
    stderr: TextIO = sys.stderr
    warning_after_seconds: float = 30.0
    _accepted: threading.Event = field(default_factory=threading.Event, init=False)
    _cancel_complete: threading.Event = field(default_factory=threading.Event, init=False)
    _finished: threading.Event = field(default_factory=threading.Event, init=False)
    _control_thread: threading.Thread | None = field(default=None, init=False)
    _count: int = field(default=0, init=False)
    _cancel_error: Exception | None = field(default=None, init=False)

    def install(self) -> Any:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handle)
        self._control_thread = threading.Thread(
            target=self._control,
            name="illm-cli-interrupt",
            daemon=True,
        )
        try:
            self._control_thread.start()
        except BaseException:
            signal.signal(signal.SIGINT, previous)
            raise
        return previous

    def handle(self, _signum: int, _frame: FrameType | None) -> None:
        # Python delivers this callback on the main thread at arbitrary
        # bytecode boundaries.  It must not re-enter engine/capture locks.
        self._count += 1
        if self._count == 1:
            return
        raise EmergencyAbandon

    @property
    def forced(self) -> bool:
        return self._count >= 2

    @property
    def interrupted(self) -> bool:
        return self._count >= 1

    def notify_accepted(self) -> None:
        """Publish the safe boundary after the metadata chunk is obtained."""

        self._accepted.set()

    def wait_for_cancel_if_requested(self) -> None:
        """Do not launch more work while a pending first interrupt is dispatched."""

        if not self.interrupted or self.forced or not self._accepted.is_set():
            return
        while not self._cancel_complete.wait(0.01):
            if self.forced:
                raise KeyboardInterrupt

    def raise_cancel_error(self) -> None:
        if self._cancel_error is not None:
            raise self._cancel_error

    def finish(self, previous_handler: Any) -> None:
        self._finished.set()
        signal.signal(signal.SIGINT, previous_handler)
        if self._control_thread is not None:
            self._control_thread.join()

    def _control(self) -> None:
        first_seen: float | None = None
        cancel_sent = False
        warned = False
        while not self._finished.wait(0.01):
            if not self.interrupted:
                continue
            if first_seen is None:
                first_seen = time.monotonic()
            if not cancel_sent and not self.forced and self._accepted.is_set():
                try:
                    self.engine.cancel(self.request_id, "cli-sigint-1")
                except Exception as exc:
                    self._cancel_error = exc
                finally:
                    cancel_sent = True
                    self._cancel_complete.set()
            if (
                not warned
                and first_seen is not None
                and time.monotonic() - first_seen >= self.warning_after_seconds
            ):
                print(
                    "warning: cancellation cleanup is still pending after 30 seconds; resource release is not claimed",
                    file=self.stderr,
                    flush=True,
                )
                warned = True


def _run(args: argparse.Namespace) -> int:
    _validate_run_args(args)
    input_token_ids = _parse_token_ids(args.input_token_ids)
    sampling = _sampling(args)
    Engine, GenerationRequest, MetadataChunk, TokenChunk, TerminalChunk = _streaming_api()
    model = args.model_config if args.model_config is not None else args.model
    engine = Engine(
        model=model,
        device=args.device,
        kv_mode=args.kv_mode,
        max_context_tokens=args.max_context_tokens,
        memory_budget_bytes=args.memory_budget_bytes,
        project_root=None,
    )
    controller: _InterruptController | None = None
    previous_handler: Any = None
    handler_installed = False
    terminal: Any | None = None
    stream: Any | None = None
    token_count = 0
    demonstration_cancelled = False
    forced_exit = False
    pending_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        request = GenerationRequest(
            model_spec_id=engine.model_spec_id,
            prompt=args.prompt,
            input_token_ids=input_token_ids,
            max_new_tokens=args.max_new_tokens,
            sampling=sampling,
            request_id=args.request_id,
            add_bos=args.add_bos,
            capture=args.capture,
            telemetry=args.telemetry,
        )
        controller = _InterruptController(engine, request.request_id)
        previous_handler = signal.getsignal(signal.SIGINT)
        handler_installed = True  # Restore even if install is interrupted after changing the handler.
        previous_handler = controller.install()
        try:
            stream = iter(engine.stream(request))
            while True:
                controller.wait_for_cancel_if_requested()
                controller.raise_cancel_error()
                if controller.forced:
                    raise KeyboardInterrupt
                try:
                    chunk = next(stream)
                except StopIteration:
                    break
                if isinstance(chunk, MetadataChunk):
                    controller.notify_accepted()
                    controller.wait_for_cancel_if_requested()
                    controller.raise_cancel_error()
                if args.stream:
                    print(canonical_json_text(chunk.to_dict()), flush=True)
                if isinstance(chunk, MetadataChunk):
                    if args.cancel_after_tokens == 0 and not demonstration_cancelled:
                        engine.cancel(request.request_id, "cli-cancel-after-tokens")
                        demonstration_cancelled = True
                elif isinstance(chunk, TokenChunk):
                    token_count += 1
                    if (
                        args.cancel_after_tokens is not None
                        and token_count >= args.cancel_after_tokens
                        and not demonstration_cancelled
                    ):
                        engine.cancel(request.request_id, "cli-cancel-after-tokens")
                        demonstration_cancelled = True
                elif isinstance(chunk, TerminalChunk):
                    if terminal is not None:
                        raise IllmError(ErrorCode.INTERNAL, "stream produced more than one terminal chunk")
                    terminal = chunk
                else:
                    raise IllmError(ErrorCode.INTERNAL, "stream produced an unknown chunk type")
            controller.raise_cancel_error()
        except KeyboardInterrupt:
            forced_exit = True
            if controller.forced:
                try:
                    engine.abandon(request.request_id, reason="CLI_SECOND_INTERRUPT")
                except BaseException as exc:
                    cleanup_error = exc
        except BaseException as exc:
            pending_error = exc
    except KeyboardInterrupt:
        forced_exit = True
        if controller is not None and controller.forced:
            try:
                engine.abandon(args.request_id, reason="CLI_SECOND_INTERRUPT")
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
    except BaseException as exc:
        pending_error = pending_error or exc
    finally:
        try:
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
        finally:
            if handler_installed and controller is not None:
                try:
                    controller.finish(previous_handler)
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            try:
                engine.close(timeout=1.0)
            except BaseException as exc:
                cleanup_error = cleanup_error or exc

    if forced_exit or (controller is not None and controller.forced):
        return 130
    if pending_error is not None:
        raise pending_error
    if cleanup_error is not None:
        raise cleanup_error
    if terminal is None:
        raise IllmError(ErrorCode.INTERNAL, "stream ended without a terminal chunk")
    if not args.stream:
        print(canonical_json_text(terminal.result.to_dict()))
    return _terminal_exit_code(terminal.outcome, terminal.error_code)


def _terminal_exit_code(outcome: object, error_code: object) -> int:
    if outcome in {"FINISHED", "CANCELLED"}:
        return 0
    if isinstance(error_code, str):
        try:
            return EXIT_CODE[ErrorCode(error_code)]
        except ValueError:
            pass
    return EXIT_CODE[ErrorCode.INTERNAL]


def _validate_fixture(output_directory: Path | None) -> Mapping[str, Any]:
    from illm.model.validation import validate_model

    return validate_model(
        "fixture:tiny-decoder",
        device="cpu",
        output_directory=output_directory,
    )


def _verify(args: argparse.Namespace) -> int:
    result = _validate_fixture(args.output_directory)
    print(canonical_json_text(result))
    return 0 if result.get("status") == "pass" else EXIT_CODE[ErrorCode.ORACLE_DIVERGED]


def _streaming_bundle(path: Path) -> Any:
    from illm.capture.streaming import StreamingCaptureBundle

    return StreamingCaptureBundle.open(path)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "--chrome-trace output already exists")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _inspect_request(args: argparse.Namespace) -> int:
    bundle = _streaming_bundle(args.capture)
    story = bundle.request_story(args.request)
    if args.chrome_trace is not None:
        capture_root = Path(bundle.root).resolve()
        trace_path = args.chrome_trace.resolve()
        if trace_path == capture_root or trace_path.is_relative_to(capture_root):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "--chrome-trace cannot modify the immutable capture")
        _write_new_json(trace_path, bundle.chrome_trace())
    print(canonical_json_text(story))
    return 0

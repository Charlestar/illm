from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

import pytest

from illm.api import streaming_cli
from illm.domain.errors import ErrorCode, IllmError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeRequest:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _FakeMetadata:
    def __init__(self, request_id: str = "req-0001") -> None:
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "metadata", "request_id": self.request_id, "accepted_ns": 10}


class _FakeToken:
    def __init__(self, output_index: int, request_id: str = "req-0001") -> None:
        self.output_index = output_index
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "token",
            "request_id": self.request_id,
            "output_index": self.output_index,
            "token_id": 100 + self.output_index,
        }


class _FakeResult:
    def __init__(self, request_id: str = "req-0001") -> None:
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, "request_id": self.request_id, "text": "hi"}


class _FakeTerminal:
    def __init__(
        self,
        request_id: str = "req-0001",
        *,
        outcome: str = "FINISHED",
        error_code: str | None = None,
    ) -> None:
        self.request_id = request_id
        self.outcome = outcome
        self.error_code = error_code
        self.result = _FakeResult(request_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "terminal",
            "request_id": self.request_id,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "result": self.result.to_dict(),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    streaming_cli.add_streaming_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def _run_real_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return subprocess.run(
        [sys.executable, "-m", "illm", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )


def _install_fake_api(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[Any],
    *,
    request_type: type[Any] = _FakeRequest,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "engines": [],
        "requests": [],
        "chunks": list(chunks),
        "stream_closed": False,
    }

    class RecordingRequest(request_type):  # type: ignore[misc]
        def __init__(self, **values: Any) -> None:
            super().__init__(**values)
            state["requests"].append(self)

    class FakeEngine:
        model_spec_id = "a" * 64

        def __init__(self, **values: Any) -> None:
            self.constructor = values
            self.cancellations: list[tuple[str, str]] = []
            self.abandonments: list[tuple[str, str]] = []
            self.close_timeouts: list[float] = []
            state["engines"].append(self)

        def stream(self, request: Any) -> Iterator[Any]:
            self.request = request
            try:
                yield from state["chunks"]
            finally:
                state["stream_closed"] = True

        def cancel(self, request_id: str, cancellation_id: str) -> None:
            self.cancellations.append((request_id, cancellation_id))

        def abandon(self, request_id: str, *, reason: str) -> None:
            self.abandonments.append((request_id, reason))

        def close(self, *, timeout: float) -> None:
            self.close_timeouts.append(timeout)

    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: (FakeEngine, RecordingRequest, _FakeMetadata, _FakeToken, _FakeTerminal),
    )
    return state


def test_nonstream_run_prints_only_result_and_forwards_frozen_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _install_fake_api(
        monkeypatch,
        [_FakeMetadata("public-1"), _FakeToken(0, "public-1"), _FakeTerminal("public-1")],
    )
    args = _parser().parse_args([
        "run", "--model-config", "models/tiny.json", "--device", "cpu",
        "--input-token-ids", "[1, 2]", "--max-new-tokens", "3", "--seed", "7",
        "--kv-mode", "none", "--max-context-tokens", "32", "--memory-budget-bytes", "4096",
        "--request-id", "public-1", "--capture", "out/capture", "--telemetry", "ops", "--no-add-bos",
    ])

    assert streaming_cli.dispatch(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "request_id": "public-1",
        "schema_version": 2,
        "text": "hi",
    }
    engine = state["engines"][0]
    assert engine.constructor == {
        "model": Path("models/tiny.json"),
        "device": "cpu",
        "kv_mode": "none",
        "max_context_tokens": 32,
        "memory_budget_bytes": 4096,
        "project_root": None,
    }
    request = state["requests"][0]
    assert request.model_spec_id == "a" * 64
    assert request.prompt is None and request.input_token_ids == (1, 2)
    assert request.max_new_tokens == 3 and request.request_id == "public-1"
    assert request.add_bos is False and request.capture == Path("out/capture")
    assert request.telemetry == "ops" and request.sampling.actual_seed == 7
    assert engine.close_timeouts == [1.0]


def test_stream_run_prints_every_chunk_as_jsonl_and_cancels_once_after_threshold(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _install_fake_api(
        monkeypatch,
        [_FakeMetadata(), _FakeToken(0), _FakeToken(1), _FakeTerminal(outcome="CANCELLED")],
    )
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--stream",
        "--seed", "0", "--cancel-after-tokens", "2",
    ])

    assert streaming_cli.dispatch(args) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["kind"] for record in records] == ["metadata", "token", "token", "terminal"]
    assert [record["output_index"] for record in records if record["kind"] == "token"] == [0, 1]
    engine = state["engines"][0]
    assert engine.cancellations == [("req-0001", "cli-cancel-after-tokens")]
    assert engine.close_timeouts == [1.0]


def test_cancel_after_zero_requests_cancel_after_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = _install_fake_api(monkeypatch, [_FakeMetadata(), _FakeTerminal(outcome="CANCELLED")])
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
        "--cancel-after-tokens", "0",
    ])
    assert streaming_cli.dispatch(args) == 0
    capsys.readouterr()
    assert state["engines"][0].cancellations == [("req-0001", "cli-cancel-after-tokens")]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--input-token-ids", "not-json"],
        ["--input-token-ids", "[true]"],
        ["--input-token-ids", "[-1]"],
        ["--max-new-tokens", "0", "--prompt", "hello"],
        ["--max-context-tokens", "0", "--prompt", "hello"],
        ["--memory-budget-bytes", "0", "--prompt", "hello"],
        ["--cancel-after-tokens", "-1", "--prompt", "hello"],
    ],
)
def test_invalid_run_values_fail_before_engine_construction(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: pytest.fail("tensor runtime must not be imported for invalid CLI input"),
    )
    if "--prompt" not in arguments:
        arguments = [*arguments]
    args = _parser().parse_args(["run", "--model", "fixture:tiny-decoder", *arguments])
    with pytest.raises(IllmError) as raised:
        streaming_cli.dispatch(args)
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT


def test_request_construction_failure_still_closes_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    class RejectingRequest(_FakeRequest):
        def __init__(self, **values: Any) -> None:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "synthetic request rejection")

    state = _install_fake_api(monkeypatch, [_FakeTerminal()], request_type=RejectingRequest)
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
    ])
    with pytest.raises(IllmError, match="synthetic request rejection"):
        streaming_cli.dispatch(args)
    assert state["engines"][0].close_timeouts == [1.0]


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([_FakeMetadata()], "without a terminal"),
        ([_FakeTerminal(), _FakeTerminal()], "more than one terminal"),
        ([object()], "unknown chunk"),
    ],
)
def test_malformed_stream_fails_closed_after_engine_close(
    chunks: list[Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_api(monkeypatch, chunks)
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
    ])
    with pytest.raises(IllmError, match=message) as raised:
        streaming_cli.dispatch(args)
    assert raised.value.code is ErrorCode.INTERNAL
    assert state["engines"][0].close_timeouts == [1.0]
    assert state["stream_closed"] is True


def test_interrupt_controller_defers_cancel_until_acceptance_and_never_abandons() -> None:
    class Engine:
        def __init__(self) -> None:
            self.cancellations: list[tuple[str, str, int | None]] = []

        def cancel(self, request_id: str, cancellation_id: str) -> None:
            self.cancellations.append((request_id, cancellation_id, threading.get_ident()))

    engine = Engine()
    controller = streaming_cli._InterruptController(engine, "req-7")
    worker = threading.Thread(target=controller._control)
    worker.start()
    try:
        controller.handle(signal.SIGINT, None)
        assert engine.cancellations == []
        controller.notify_accepted()
        assert controller._cancel_complete.wait(1.0)
        controller.raise_cancel_error()
        assert engine.cancellations == [
            ("req-7", "cli-sigint-1", worker.ident),
        ]
        assert worker.ident != threading.get_ident()
        with pytest.raises(KeyboardInterrupt):
            controller.handle(signal.SIGINT, None)
        assert len(engine.cancellations) == 1
        assert controller.forced is True
    finally:
        controller._finished.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()


def test_interrupt_warning_never_claims_resource_release() -> None:
    class Engine:
        def cancel(self, _request_id: str, _cancellation_id: str) -> None:
            return None

    stderr = io.StringIO()
    controller = streaming_cli._InterruptController(
        Engine(), "req-8", stderr=stderr, warning_after_seconds=0.0,
    )
    worker = threading.Thread(target=controller._control)
    worker.start()
    controller.handle(signal.SIGINT, None)
    deadline = time.monotonic() + 1.0
    while not stderr.getvalue() and time.monotonic() < deadline:
        time.sleep(0.01)
    controller._finished.set()
    worker.join(timeout=1.0)
    warning = stderr.getvalue()
    assert "cleanup is still pending" in warning
    assert "release is not claimed" in warning
    assert not worker.is_alive()


def test_first_interrupt_inside_preaccept_critical_section_cancels_from_control_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "critical": False,
        "cancelled": [],
        "closed": [],
        "stream_closed": False,
    }
    main_thread = threading.get_ident()

    class Engine:
        model_spec_id = "a" * 64

        def __init__(self, **_values: Any) -> None:
            return None

        def stream(self, _request: Any) -> Iterator[Any]:
            try:
                state["critical"] = True
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
                assert state["cancelled"] == []
                state["critical"] = False
                yield _FakeMetadata()
                assert state["cancelled"] == [
                    ("req-0001", "cli-sigint-1", state["control_thread"]),
                ]
                yield _FakeTerminal(outcome="CANCELLED")
            finally:
                state["critical"] = False
                state["stream_closed"] = True

        def cancel(self, request_id: str, cancellation_id: str) -> None:
            assert state["critical"] is False
            control_thread = threading.get_ident()
            assert control_thread != main_thread
            state["control_thread"] = control_thread
            state["cancelled"].append((request_id, cancellation_id, control_thread))

        def abandon(self, _request_id: str, *, reason: str) -> None:
            pytest.fail(f"first interrupt must not abandon: {reason}")

        def close(self, *, timeout: float) -> None:
            state["closed"].append(timeout)

    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: (Engine, _FakeRequest, _FakeMetadata, _FakeToken, _FakeTerminal),
    )
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
    ])
    assert streaming_cli.dispatch(args) == 0
    assert state["stream_closed"] is True
    assert state["closed"] == [1.0]


def test_second_interrupt_inside_critical_section_abandons_only_after_unwind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "critical": False,
        "cancelled": [],
        "abandoned": [],
        "closed": [],
        "stream_closed": False,
    }

    class Engine:
        model_spec_id = "a" * 64

        def __init__(self, **_values: Any) -> None:
            return None

        def stream(self, _request: Any) -> Iterator[Any]:
            try:
                state["critical"] = True
                handler = signal.getsignal(signal.SIGINT)
                assert callable(handler)
                handler(signal.SIGINT, None)
                handler(signal.SIGINT, None)
                yield pytest.fail("second interrupt must unwind before yielding")
            finally:
                state["critical"] = False
                state["stream_closed"] = True

        def cancel(self, request_id: str, cancellation_id: str) -> None:
            state["cancelled"].append((request_id, cancellation_id))

        def abandon(self, request_id: str, *, reason: str) -> None:
            assert state["critical"] is False
            state["abandoned"].append((request_id, reason))

        def close(self, *, timeout: float) -> None:
            state["closed"].append(timeout)

    monkeypatch.setattr(
        streaming_cli,
        "_streaming_api",
        lambda: (Engine, _FakeRequest, _FakeMetadata, _FakeToken, _FakeTerminal),
    )
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
    ])
    assert streaming_cli.dispatch(args) == 130
    assert state["cancelled"] == []
    assert state["abandoned"] == [("req-0001", "CLI_SECOND_INTERRUPT")]
    assert state["stream_closed"] is True
    assert state["closed"] == [1.0]


def test_stream_is_explicitly_closed_when_jsonl_print_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_api(monkeypatch, [_FakeMetadata(), _FakeTerminal()])

    def fail_print(*_values: Any, **_options: Any) -> None:
        raise OSError("synthetic stdout failure")

    monkeypatch.setattr(streaming_cli, "print", fail_print, raising=False)
    args = _parser().parse_args([
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello", "--seed", "0",
        "--stream",
    ])
    with pytest.raises(OSError, match="synthetic stdout failure"):
        streaming_cli.dispatch(args)
    assert state["stream_closed"] is True
    assert state["engines"][0].close_timeouts == [1.0]


@pytest.mark.parametrize(
    ("status", "expected"),
    [("pass", 0), ("fail", 87)],
)
def test_verify_adapter_uses_fixed_tiny_fixture(
    status: str,
    expected: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[Path | None] = []

    def validate(output: Path | None) -> dict[str, str]:
        seen.append(output)
        return {"fixture": "tiny-decoder", "status": status}

    monkeypatch.setattr(
        streaming_cli,
        "_validate_fixture",
        validate,
    )
    output = tmp_path / "validation"
    args = _parser().parse_args([
        "verify", "--fixture", "tiny-decoder", "--output-directory", str(output),
    ])
    assert streaming_cli.dispatch(args) == expected
    assert seen == [output]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_inspect_request_prints_story_and_writes_new_derived_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = tmp_path / "capture"
    trace = tmp_path / "derived" / "trace.json"

    class Bundle:
        root = capture

        def request_story(self, request_id: str) -> dict[str, Any]:
            assert request_id == "req-9"
            return {"capability": "R0", "request_id": request_id}

        def chrome_trace(self) -> dict[str, Any]:
            return {"traceEvents": [], "metadata": {"capability": "R0"}}

    monkeypatch.setattr(streaming_cli, "_streaming_bundle", lambda path: Bundle())
    args = _parser().parse_args([
        "inspect", "request", str(capture), "--request", "req-9", "--chrome-trace", str(trace),
    ])
    assert streaming_cli.dispatch(args) == 0
    assert json.loads(capsys.readouterr().out) == {"capability": "R0", "request_id": "req-9"}
    assert json.loads(trace.read_text(encoding="utf-8")) == {
        "metadata": {"capability": "R0"},
        "traceEvents": [],
    }


def test_inspect_never_mutates_capture_or_overwrites_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()

    class Bundle:
        root = capture

        def request_story(self, request_id: str) -> dict[str, Any]:
            return {"request_id": request_id}

        def chrome_trace(self) -> dict[str, Any]:
            return {"traceEvents": []}

    monkeypatch.setattr(streaming_cli, "_streaming_bundle", lambda path: Bundle())
    inside = _parser().parse_args([
        "inspect", "request", str(capture), "--request", "req", "--chrome-trace",
        str(capture / "derived.json"),
    ])
    with pytest.raises(IllmError, match="immutable capture"):
        streaming_cli.dispatch(inside)
    assert list(capture.iterdir()) == []

    existing = tmp_path / "existing.json"
    existing.write_text("preserve", encoding="utf-8")
    overwrite = _parser().parse_args([
        "inspect", "request", str(capture), "--request", "req", "--chrome-trace", str(existing),
    ])
    with pytest.raises(IllmError, match="already exists"):
        streaming_cli.dispatch(overwrite)
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_dispatch_leaves_unrelated_root_command_unhandled() -> None:
    assert streaming_cli.dispatch(argparse.Namespace(command="legacy")) is None


@pytest.mark.parametrize(
    ("outcome", "error_code", "expected"),
    [
        ("FINISHED", None, 0),
        ("CANCELLED", None, 0),
        ("FAILED", "RESOURCE_EXHAUSTED", 88),
        ("FAILED", "unknown", 1),
        ("DEADLINE_EXCEEDED", None, 1),
    ],
)
def test_terminal_exit_code_is_stable(outcome: str, error_code: str | None, expected: int) -> None:
    assert streaming_cli._terminal_exit_code(outcome, error_code) == expected


@pytest.mark.parametrize(
    ("kv_mode", "expected_query_tokens"),
    [("contiguous", [6, 1]), ("none", [6, 7])],
)
def test_real_tiny_cli_nonstream_distinguishes_incremental_and_full_context_execution(
    kv_mode: str,
    expected_query_tokens: list[int],
) -> None:
    process = _run_real_cli(
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello",
        "--max-new-tokens", "2", "--seed", "7", "--kv-mode", kv_mode,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["outcome"] == "FINISHED"
    assert result["query_tokens_per_step"] == expected_query_tokens
    assert len(result["output_token_ids"]) == 2
    assert all(value == 0 for value in result["resource_status"].values())


def test_real_tiny_cli_cancel_capture_inspect_trace_and_r0_only_replay(tmp_path: Path) -> None:
    capture = tmp_path / "cancel-capture"
    process = _run_real_cli(
        "run", "--model", "fixture:tiny-decoder", "--prompt", "hello",
        "--max-new-tokens", "4", "--seed", "7", "--stream", "--kv-mode", "contiguous",
        "--cancel-after-tokens", "2", "--request-id", "req-real-cancel", "--capture", str(capture),
    )
    assert process.returncode == 0, process.stderr
    chunks = [json.loads(line) for line in process.stdout.splitlines()]
    assert [chunk["kind"] for chunk in chunks] == ["metadata", "token", "token", "terminal"]
    assert [chunk["output_index"] for chunk in chunks[1:3]] == [0, 1]
    terminal = chunks[-1]
    assert terminal["outcome"] == "CANCELLED"
    assert all(value == 0 for value in terminal["result"]["resource_status"].values())
    assert (capture / "COMPLETE").is_file() and not (capture / "INCOMPLETE").exists()

    trace_path = tmp_path / "derived" / "trace.json"
    inspected = _run_real_cli(
        "inspect", "request", str(capture), "--request", "req-real-cancel",
        "--chrome-trace", str(trace_path),
    )
    assert inspected.returncode == 0, inspected.stderr
    story = json.loads(inspected.stdout)
    assert story["capability"] == "R0" and story["terminal_outcome"] == "CANCELLED"
    assert [item["payload"]["output_index"] for item in story["tokens"]] == [0, 1]
    assert story["kv"]["final_active_allocation_bytes"] == 0
    assert story["kv"]["final_open_write_count"] == 0
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["metadata"]["capability"] == "R0" and trace["metadata"]["derived"] is True

    replayed = _run_real_cli(
        "replay", str(capture), "--mode", "trace", "--request-id", "req-real-cancel",
    )
    assert replayed.returncode == 0, replayed.stderr
    replay_story = json.loads(replayed.stdout)
    assert replay_story["request_id"] == "req-real-cancel"
    assert replay_story["terminal_outcome"] == "CANCELLED"

    unsupported = _run_real_cli(
        "replay", str(capture), "--mode", "state", "--assert-state-digest",
    )
    assert unsupported.returncode == 78
    error = json.loads(unsupported.stderr)
    assert error["code"] == "UNSUPPORTED_SCHEMA"
    assert error["details"]["capabilities"] == ["R0"]

"""Produce policy-aware CPU lifecycle/capture evidence for V0.2.

Every real engine in this exercise uses the registered candidate FP32 policy.
Controlled CPU fault seams exercise lifecycle ownership; they are never
reported as actual GPU fault evidence.  The historical schema-2 producer and
reader are not imported or patched.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import secrets
import shutil
import sys
import threading
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterator, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle, PolicyStreamingCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K
from illm.engine.facade import as_illm_error
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk, TerminalChunk, TokenChunk
from illm.executor import CompletionFence, EagerExecutor, ExecutionTicket, ImmediateFence
from illm.metrics import metric_registry, registry_conformance, registry_digest
from illm.numerical_policy import NumericalExecutionPolicyV1, candidate_policy
from illm.policy_binding import build_execution_binding
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.sampling import SamplingSpec
from illm.stateful_policy_exercise import (
    COMPLETE_CAPTURE_IDS,
    CYCLE_COUNT,
    INCOMPLETE_CAPTURE_IDS,
    RACE_COUNT,
    REPORT_KIND,
    REQUIRED_FAULT_IDS,
    REQUIRED_POLICY_CAPTURE_IDS,
    SCOPE,
    verify_policy_exercise_output,
)

import illm.engine.streaming as streaming_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = candidate_policy("fp32")
_ZERO_RESOURCE_FIELDS = frozenset(
    {
        "active_requests",
        "reservations",
        "quarantined_requests",
        "active_tickets",
        "input_leases",
        "output_leases",
        "workspace_leases",
        "kv_write_leases",
        "kv_reserved_bytes",
        "kv_valid_tokens",
        "kv_staged_tokens",
    }
)
_PUBLIC_INPUT = (1, 107, 104, 111, 111, 114)


def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _fail(ErrorCode.INTERNAL, f"document already contains {field}")
    value[field] = sha256_json(value)
    return value


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _new_root(output: Path) -> Path:
    if not isinstance(output, Path):
        _fail(ErrorCode.INVALID_ARGUMENT, "output must be a Path")
    root = Path(os.path.abspath(os.fspath(output)))
    for ancestor in root.parents:
        if _is_linklike(ancestor):
            _fail(ErrorCode.INVALID_ARGUMENT, "output ancestry contains a link/reparse point")
    if root.exists() or root.is_symlink() or _is_linklike(root):
        _fail(ErrorCode.INVALID_ARGUMENT, "policy lifecycle output must not already exist")
    if not root.parent.is_dir():
        _fail(ErrorCode.INVALID_ARGUMENT, "policy lifecycle output parent must already exist")
    try:
        root.mkdir(exist_ok=False)
    except OSError as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "policy lifecycle output could not be created") from exc
    return root


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "policy lifecycle writer refuses to overwrite an artifact",
            {"path": path.name},
        ) from exc


def _replace_owned(path: Path, value: bytes) -> None:
    if _is_linklike(path) or not path.is_file():
        _fail(ErrorCode.INVALID_ARGUMENT, "owned policy lifecycle file is no longer ordinary")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(16)}")
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_json_bytes(dict(value)) + b"\n")


def _write_text(path: Path, value: str) -> None:
    _write_bytes(path, value.encode("ascii"))


def _resources_zero(value: Mapping[str, Any]) -> bool:
    return set(value) == _ZERO_RESOURCE_FIELDS and all(type(item) is int and item == 0 for item in value.values())


def _engine(
    *,
    kv_mode: str = "contiguous",
    max_context_tokens: int | None = None,
    exporters: Sequence[Any] = (),
    exporter_capacity: int = 256,
    expected_binding_digest: str | None = None,
) -> Engine:
    engine = Engine(
        "fixture:tiny-decoder",
        device="cpu",
        kv_mode=kv_mode,
        max_context_tokens=max_context_tokens,
        project_root=PROJECT_ROOT,
        exporters=exporters,
        exporter_capacity=exporter_capacity,
        numerical_policy=POLICY,
    )
    plan = engine.memory_plan.to_dict()
    if (
        engine.numerical_policy != POLICY
        or plan["schema_version"] != 2
        or (
            expected_binding_digest is not None
            and plan["execution_binding_digest"] != expected_binding_digest
        )
    ):
        _fail(ErrorCode.INTERNAL, "policy lifecycle engine did not select the candidate schema2 MemoryPlan")
    return engine


def _request(
    engine: Engine,
    request_id: str,
    *,
    max_new_tokens: int = 1,
    input_token_ids: tuple[int, ...] = _PUBLIC_INPUT,
    capture: Path | None = None,
    telemetry: str = "metrics",
) -> GenerationRequest:
    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        max_new_tokens=max_new_tokens,
        sampling=SamplingSpec(mode="greedy", actual_seed=7),
        input_token_ids=input_token_ids,
        request_id=request_id,
        capture=capture,
        telemetry=telemetry,
    )


def _events(engine: Engine) -> list[dict[str, Any]]:
    return [event.to_dict() for event in engine.last_events]


def _terminal(chunks: Sequence[object]) -> TerminalChunk:
    terminals = [chunk for chunk in chunks if isinstance(chunk, TerminalChunk)]
    if len(terminals) != 1:
        _fail(ErrorCode.INTERNAL, "policy stream did not produce exactly one terminal chunk")
    return terminals[0]


@contextmanager
def _patched(name: str, replacement: object) -> Iterator[None]:
    original = getattr(streaming_module, name)
    setattr(streaming_module, name, replacement)
    try:
        yield
    finally:
        setattr(streaming_module, name, original)


class _DelayedFence:
    def __init__(self, base: CompletionFence) -> None:
        self.base = base
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("controlled policy delayed fence timed out")
        self.base.wait()

    def is_complete(self) -> bool | None:
        return self.base.is_complete() if self.release.is_set() else False

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded if self.release.is_set() else None


def _delayed_executor(fences: list[_DelayedFence]) -> type[EagerExecutor]:
    class DelayedPolicyExecutor(EagerExecutor):
        def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
            delayed = _DelayedFence(fence)
            fences.append(delayed)
            return delayed

    return DelayedPolicyExecutor


def _await_fence(fences: list[_DelayedFence], index: int = 0) -> _DelayedFence:
    deadline = perf_counter_ns() + 5_000_000_000
    while len(fences) <= index and perf_counter_ns() < deadline:
        threading.Event().wait(0.001)
    if len(fences) <= index or not fences[index].entered.wait(timeout=5):
        _fail(ErrorCode.INTERNAL, "controlled policy execution never reached delayed fence")
    return fences[index]


class _NonFiniteExecutor(EagerExecutor):
    def resolve(self, ticket: ExecutionTicket) -> tuple[float, ...]:
        values = super().resolve(ticket)
        return (float("nan"), *values[1:])


class _OomExecutor(EagerExecutor):
    def launch(self, context_ids: tuple[int, ...]) -> ExecutionTicket:
        del context_ids
        raise MemoryError("controlled policy pre-dispatch OOM")


class _FailedFenceExecutor(EagerExecutor):
    def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
        del fence
        return ImmediateFence(RuntimeError("controlled policy asynchronous execution error"))


class _UnknownFence:
    def wait(self) -> None:
        return None

    def is_complete(self) -> bool | None:
        return None

    @property
    def succeeded(self) -> bool | None:
        return None


class _UnknownFenceExecutor(EagerExecutor):
    def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
        del fence
        return _UnknownFence()


class _BlockingExporter:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def emit(self, event: object) -> None:
        del event
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("controlled policy optional exporter timeout")


def _consume(engine: Engine, request: GenerationRequest) -> tuple[list[object], list[BaseException]]:
    chunks: list[object] = []
    errors: list[BaseException] = []
    try:
        chunks.extend(engine.stream(request))
    except BaseException as exc:
        errors.append(exc)
    return chunks, errors


def _scenario(
    scenario_id: str,
    *,
    binding_digest: str,
    mechanism: str,
    evidence_kind: str,
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    if scenario_id not in REQUIRED_FAULT_IDS:
        _fail(ErrorCode.INTERNAL, "unknown required policy fault scenario")
    return _signed(
        {
            "scenario_id": scenario_id,
            "status": "pass",
            "required_for_cpu_profile": True,
            "mechanism": mechanism,
            "evidence_kind": evidence_kind,
            "actual_tiny_runtime": True,
            "actual_gpu_fault": False,
            "numerical_policy_digest": POLICY.digest,
            "execution_binding_digest": binding_digest,
            "observations": dict(observations),
        },
        "scenario_digest",
    )


def _run_cycles() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    engine = _engine()
    records: list[dict[str, Any]] = []
    model_identity: dict[str, Any] | None = None
    memory_plan = engine.memory_plan.to_dict()
    binding = cast(dict[str, Any], memory_plan["execution_binding"])
    try:
        for index in range(CYCLE_COUNT):
            request_id = f"policy-lifecycle-cycle-{index:03d}"
            request = _request(engine, request_id)
            if index % 2 == 0:
                result = engine.generate(request)
                planned = "FINISHED"
            else:
                stream = engine.stream(request)
                first = next(stream)
                if not isinstance(first, MetadataChunk):
                    _fail(ErrorCode.INTERNAL, "policy cancel cycle was not accepted")
                cancellation = engine.cancel(request_id, f"cancel-{index:03d}")
                result = _terminal(list(stream)).result
                if not cancellation.won:
                    _fail(ErrorCode.INTERNAL, "policy cancel cycle did not win terminal arbitration")
                planned = "CANCELLED"
            resources = engine.resource_status
            events = _events(engine)
            result_value = result.to_dict()
            if (
                result.outcome != planned
                or not _resources_zero(resources)
                or result_value.get("schema_version") != 3
                or result_value.get("numerical_policy_digest") != POLICY.digest
                or result_value.get("execution_binding_digest") != memory_plan["execution_binding_digest"]
                or result_value.get("memory_plan_digest") != engine.memory_plan.digest
            ):
                _fail(ErrorCode.ENGINE_UNHEALTHY, "policy lifecycle cycle did not return valid candidate state")
            if model_identity is None:
                model_identity = {
                    "model_spec_id": result.model_spec_id,
                    "sampling_spec_digest": result.sampling_spec_digest,
                    "runtime_dtype": POLICY.storage_dtype,
                    "public_fixture": "fixture:tiny-decoder",
                }
            records.append(
                _signed(
                    {
                        "cycle_index": index,
                        "request_id": request_id,
                        "planned_outcome": planned,
                        "result": result_value,
                        "events": events,
                        "resources_after_terminal": resources,
                    },
                    "cycle_digest",
                )
            )
    finally:
        closed = engine.close()
    if not closed or model_identity is None:
        _fail(ErrorCode.ENGINE_UNHEALTHY, "policy cycle engine did not close cleanly")
    raw = _signed(
        {
            "schema_version": 1,
            "suite_id": "illm.v02.policy.lifecycle.100-alternating.v1",
            "cycle_count": CYCLE_COUNT,
            "alternation": "even-finished-odd-cancel-before-launch",
            "finished_count": 50,
            "cancelled_count": 50,
            "numerical_policy_digest": POLICY.digest,
            "execution_binding_digest": memory_plan["execution_binding_digest"],
            "memory_plan_digest": engine.memory_plan.digest,
            "records": records,
        },
        "raw_digest",
    )
    return raw, model_identity, binding


def _complete_capture(
    capture_root: Path,
    case_id: str,
    *,
    binding_digest: str,
    kv_mode: str = "contiguous",
    input_token_ids: tuple[int, ...] = _PUBLIC_INPUT,
    max_new_tokens: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / case_id
    engine = _engine(kv_mode=kv_mode, expected_binding_digest=binding_digest)
    resources: dict[str, int] = {}
    result: Any = None
    try:
        result = engine.generate(
            _request(
                engine,
                f"policy-capture-{case_id}",
                max_new_tokens=max_new_tokens,
                input_token_ids=input_token_ids,
                capture=path,
                telemetry="request",
            )
        )
        resources = engine.resource_status
    finally:
        closed = engine.close()
    if (
        result is None
        or not closed
        or not _resources_zero(resources)
        or result.capture_error is not None
        or result.numerical_policy != POLICY
    ):
        _fail(ErrorCode.CAPTURE_INCOMPLETE, f"{case_id} policy capture did not finish cleanly")
    bundle = PolicyStreamingCaptureBundle.open(path)
    record = _signed(
        {
            "capture_id": case_id,
            "path": f"captures/{case_id}",
            "request_id": result.request_id,
            "outcome": result.outcome,
            "finish_reason": result.finish_reason,
            "kv_mode": kv_mode,
            "manifest_digest": bundle.manifest["manifest_digest"],
            "event_count": len(bundle.events),
            "memory_plan_digest": bundle.manifest["memory_plan_digest"],
            "numerical_policy_digest": bundle.manifest["numerical_policy_digest"],
            "execution_binding_digest": bundle.manifest["execution_binding_digest"],
            "resources_after_terminal": resources,
        },
        "capture_record_digest",
    )
    return record, result.to_dict()


def _cancel_after_launch(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fences: list[_DelayedFence] = []
    path = capture_root / "cancel-after-launch"
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = _engine(expected_binding_digest=binding_digest)
        request = _request(
            engine,
            "policy-capture-cancel-after-launch",
            max_new_tokens=3,
            capture=path,
            telemetry="request",
        )
        chunks: list[object] = []
        errors: list[BaseException] = []

        def consume() -> None:
            values, failures = _consume(engine, request)
            chunks.extend(values)
            errors.extend(failures)

        worker = threading.Thread(target=consume, name="illm-policy-evidence-cancel-after-launch")
        worker.start()
        fence = _await_fence(fences)
        cancellation = engine.cancel(request.request_id, "cancel-after-launch")
        resources_while_pending = engine.resource_status
        fence.release.set()
        worker.join(timeout=10)
        if worker.is_alive() or errors:
            _fail(
                ErrorCode.INTERNAL,
                "policy cancel-after-launch worker did not terminate",
                {"errors": [type(item).__name__ for item in errors]},
            )
        terminal = _terminal(chunks)
        resources_after = engine.resource_status
        closed = engine.close()
    if (
        not cancellation.won
        or terminal.outcome != "CANCELLED"
        or not closed
        or resources_while_pending.get("active_tickets") != 1
        or resources_while_pending.get("kv_write_leases") != 1
        or not _resources_zero(resources_after)
    ):
        _fail(ErrorCode.ENGINE_UNHEALTHY, "policy delayed cancellation contract failed")
    bundle = PolicyStreamingCaptureBundle.open(path)
    record = _signed(
        {
            "capture_id": "cancel-after-launch",
            "path": "captures/cancel-after-launch",
            "request_id": terminal.request_id,
            "outcome": terminal.outcome,
            "finish_reason": terminal.finish_reason,
            "kv_mode": "contiguous",
            "manifest_digest": bundle.manifest["manifest_digest"],
            "event_count": len(bundle.events),
            "memory_plan_digest": bundle.manifest["memory_plan_digest"],
            "numerical_policy_digest": bundle.manifest["numerical_policy_digest"],
            "execution_binding_digest": bundle.manifest["execution_binding_digest"],
            "resources_after_terminal": resources_after,
        },
        "capture_record_digest",
    )
    scenario = _scenario(
        "cancel-after-launch-delayed-fence",
        binding_digest=binding_digest,
        mechanism="CompletionFence wrapper delays resolution after a real candidate-policy tiny forward",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-fence-delay-simulation",
        observations={
            "cancellation_won": cancellation.won,
            "outcome": terminal.outcome,
            "token_chunk_count": sum(isinstance(chunk, TokenChunk) for chunk in chunks),
            "resources_while_fence_pending": resources_while_pending,
            "resources_after_terminal": resources_after,
            "capture_path": record["path"],
            "capture_manifest_digest": record["manifest_digest"],
        },
    )
    return record, scenario


def _finish_cancel_race(binding_digest: str) -> dict[str, Any]:
    fences: list[_DelayedFence] = []
    results: list[dict[str, Any]] = []
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = _engine(expected_binding_digest=binding_digest)
        try:
            for index in range(RACE_COUNT):
                request = _request(engine, f"policy-race-{index:02d}")
                chunks: list[object] = []
                errors: list[BaseException] = []

                def consume() -> None:
                    values, failures = _consume(engine, request)
                    chunks.extend(values)
                    errors.extend(failures)

                worker = threading.Thread(target=consume, name=f"illm-policy-race-worker-{index}")
                worker.start()
                fence = _await_fence(fences, index)
                barrier = threading.Barrier(3)
                cancels: list[dict[str, Any]] = []

                def release() -> None:
                    barrier.wait(timeout=5)
                    fence.release.set()

                def cancel() -> None:
                    barrier.wait(timeout=5)
                    cancels.append(engine.cancel(request.request_id, f"race-cancel-{index}").to_dict())

                release_thread = threading.Thread(target=release, name=f"illm-policy-race-release-{index}")
                cancel_thread = threading.Thread(target=cancel, name=f"illm-policy-race-cancel-{index}")
                release_thread.start()
                cancel_thread.start()
                barrier.wait(timeout=5)
                release_thread.join(timeout=5)
                cancel_thread.join(timeout=5)
                worker.join(timeout=10)
                if any(thread.is_alive() for thread in (release_thread, cancel_thread, worker)) or errors:
                    _fail(ErrorCode.INTERNAL, "policy finish/cancel race threads did not terminate")
                terminal = _terminal(chunks)
                events = _events(engine)
                terminal_events = [
                    event
                    for event in events
                    if event["kind"] in {"request.completed", "request.cancelled", "request.failed"}
                ]
                resources = engine.resource_status
                if len(cancels) != 1 or len(terminal_events) != 1 or not _resources_zero(resources):
                    _fail(ErrorCode.ENGINE_UNHEALTHY, "policy race produced invalid terminal/resource state")
                results.append(
                    {
                        "iteration": index,
                        "outcome": terminal.outcome,
                        "cancel_outcome": cancels[0]["outcome"],
                        "cancel_won": cancels[0]["won"],
                        "terminal_event_count": len(terminal_events),
                        "token_chunk_count": sum(isinstance(chunk, TokenChunk) for chunk in chunks),
                        "resources_after_terminal": resources,
                    }
                )
        finally:
            closed = engine.close()
    if not closed:
        _fail(ErrorCode.ENGINE_UNHEALTHY, "policy race engine did not close")
    return _scenario(
        "finish-cancel-race",
        binding_digest=binding_digest,
        mechanism="barrier concurrently releases candidate-policy fence and submits cancellation",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-scheduling-simulation",
        observations={"race_count": RACE_COUNT, "iterations": results},
    )


def _capacity_failure(binding_digest: str) -> tuple[dict[str, Any], str]:
    engine = _engine(max_context_tokens=8, expected_binding_digest=binding_digest)
    capacity_plan = engine.memory_plan.digest
    error: IllmError | None = None
    try:
        engine.generate(
            _request(
                engine,
                "policy-capacity-preaccept",
                input_token_ids=(1, 3, 4, 5, 6, 7, 8, 9),
            )
        )
    except IllmError as exc:
        error = exc
    resources = engine.resource_status
    accepted_events = sum(event.kind is K.REQUEST_ACCEPTED for event in engine.last_events)
    closed = engine.close()
    if (
        error is None
        or error.code is not ErrorCode.CONTEXT_LIMIT_EXCEEDED
        or accepted_events
        or not closed
        or not _resources_zero(resources)
    ):
        _fail(ErrorCode.INTERNAL, "policy capacity failure did not reject before acceptance")
    scenario = _scenario(
        "capacity-exceeded-preaccept",
        binding_digest=binding_digest,
        mechanism="candidate-policy MemoryPlan capacity is exceeded before Engine acceptance",
        evidence_kind="actual-tiny-policy-runtime",
        observations={
            "error_code": error.code.value,
            "accepted_event_count": accepted_events,
            "resources_after_rejection": resources,
            "memory_plan_digest": capacity_plan,
        },
    )
    return scenario, capacity_plan


def _known_failure(
    scenario_id: str,
    executor: type[EagerExecutor],
    expected: ErrorCode,
    binding_digest: str,
) -> dict[str, Any]:
    with _patched("EagerExecutor", executor):
        engine = _engine(expected_binding_digest=binding_digest)
        result = engine.generate(_request(engine, f"policy-fault-{scenario_id}", max_new_tokens=2))
        resources = engine.resource_status
        closed = engine.close()
    if (
        result.outcome != "FAILED"
        or result.error_code != expected.value
        or result.output_token_ids
        or result.numerical_policy != POLICY
        or not closed
        or not _resources_zero(resources)
    ):
        _fail(ErrorCode.INTERNAL, f"{scenario_id} did not produce a safely reclaimed policy failure")
    mechanism = {
        "async-execution-error": "completed failure fence injected after a real candidate-policy tiny forward",
        "numerical-error": "nonfinite last-row logit injected after a candidate-policy tiny forward resolves",
        "oom": "MemoryError injected before candidate-policy executor dispatch",
    }[scenario_id]
    return _scenario(
        scenario_id,
        binding_digest=binding_digest,
        mechanism=mechanism,
        evidence_kind="actual-tiny-policy-runtime-with-controlled-fault-simulation",
        observations={
            "outcome": result.outcome,
            "error_code": result.error_code,
            "committed_output_count": len(result.output_token_ids),
            "resources_after_terminal": resources,
        },
    )


def _incomplete_record(capture_root: Path, capture_id: str, reason: str) -> dict[str, Any]:
    path = capture_root / capture_id
    manifest_path = path / "capture-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.INTERNAL, "incomplete policy capture manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        _fail(ErrorCode.INTERNAL, "incomplete policy capture manifest must be an object")
    return _signed(
        {
            "capture_id": capture_id,
            "path": f"captures/{capture_id}",
            "expected_state": "INCOMPLETE",
            "expected_reason": reason,
            "memory_plan_digest": manifest["memory_plan_digest"],
            "numerical_policy_digest": manifest["numerical_policy_digest"],
            "execution_binding_digest": manifest["execution_binding_digest"],
        },
        "capture_record_digest",
    )


def _unknown_fence(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "unknown-fence"
    with _patched("EagerExecutor", _UnknownFenceExecutor):
        engine = _engine(expected_binding_digest=binding_digest)
        error: IllmError | None = None
        try:
            tuple(
                engine.stream(
                    _request(
                        engine,
                        "policy-fault-unknown-fence",
                        capture=path,
                        telemetry="request",
                    )
                )
            )
        except IllmError as exc:
            error = exc
        resources = engine.resource_status
        terminal_events = sum(
            event.kind in {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED}
            for event in engine.last_events
        )
        closed = engine.close()
    if (
        error is None
        or error.code is not ErrorCode.ENGINE_UNHEALTHY
        or closed
        or resources.get("quarantined_requests") != 1
        or resources.get("active_tickets") != 1
        or terminal_events != 0
        or not (path / "INCOMPLETE").is_file()
        or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "unknown policy fence did not fail-stop and quarantine")
    record = _incomplete_record(capture_root, "unknown-fence", "FENCE_OR_RESOURCE_STATE_UNKNOWN")
    scenario = _scenario(
        "unknown-fence-fail-stop",
        binding_digest=binding_digest,
        mechanism="candidate-policy CompletionFence tri-state seam returns unknown after a real tiny forward",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-fence-state-simulation",
        observations={
            "raised_error_code": error.code.value,
            "terminal_ready_event_count": terminal_events,
            "resources_quarantined": resources,
            "engine_close_succeeded": closed,
            "capture_path": record["path"],
        },
    )
    return record, scenario


def _exporter_stall(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "exporter-stall"
    exporter = _BlockingExporter()
    engine = _engine(
        exporters=(exporter,),
        exporter_capacity=1,
        expected_binding_digest=binding_digest,
    )
    results: list[Any] = []
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            results.append(
                engine.generate(
                    _request(
                        engine,
                        "policy-fault-exporter-stall",
                        max_new_tokens=4,
                        capture=path,
                        telemetry="request",
                    )
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=generate, name="illm-policy-evidence-exporter-stall")
    worker.start()
    if not exporter.entered.wait(timeout=5):
        _fail(ErrorCode.INTERNAL, "policy optional exporter did not receive an event")
    worker.join(timeout=10)
    if worker.is_alive() or errors or len(results) != 1:
        _fail(ErrorCode.INTERNAL, "policy optional exporter blocked inference")
    result = results[0]
    status = engine.exporter_status
    resources = engine.resource_status
    initial_close = engine.close(timeout=0.0)
    exporter.release.set()
    final_close = engine.close(timeout=5.0)
    if (
        result.outcome != "FINISHED"
        or result.capture_error is not None
        or result.numerical_policy != POLICY
        or status["copy_drop_count"] <= 0
        or initial_close
        or not final_close
        or not _resources_zero(resources)
    ):
        _fail(ErrorCode.INTERNAL, "policy optional exporter isolation contract failed")
    bundle = PolicyStreamingCaptureBundle.open(path)
    record = _signed(
        {
            "capture_id": "exporter-stall",
            "path": "captures/exporter-stall",
            "request_id": result.request_id,
            "outcome": result.outcome,
            "finish_reason": result.finish_reason,
            "kv_mode": "contiguous",
            "manifest_digest": bundle.manifest["manifest_digest"],
            "event_count": len(bundle.events),
            "memory_plan_digest": bundle.manifest["memory_plan_digest"],
            "numerical_policy_digest": bundle.manifest["numerical_policy_digest"],
            "execution_binding_digest": bundle.manifest["execution_binding_digest"],
            "resources_after_terminal": resources,
        },
        "capture_record_digest",
    )
    scenario = _scenario(
        "optional-exporter-stall-drop",
        binding_digest=binding_digest,
        mechanism="bounded optional EventBus sink is held while candidate-policy inference completes",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-exporter-stall-simulation",
        observations={
            "outcome": result.outcome,
            "copy_drop_count": status["copy_drop_count"],
            "exporter_error_count": status["error_count"],
            "initial_close_succeeded": initial_close,
            "final_close_succeeded": final_close,
            "resources_after_terminal": resources,
            "independent_capture_path": record["path"],
            "independent_capture_manifest_digest": record["manifest_digest"],
        },
    )
    return record, scenario


def _required_write_failure(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "required-write-failure"
    real_session = PolicyStreamingCaptureSession

    class FailAfterAcceptance:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = real_session(*args, **kwargs)
            self.failed = False

        @property
        def root(self) -> Path:
            return self.inner.root

        def emit(self, kind: K, payload: Mapping[str, Any]) -> Any:
            if kind is K.MODEL_FORWARD_STARTED and not self.failed:
                self.failed = True
                raise OSError("controlled required policy capture write failure")
            return self.inner.emit(kind, payload)

        def mark_incomplete(
            self,
            reason: str,
            *,
            critical_event_drop_count: int = 0,
        ) -> dict[str, Any]:
            return self.inner.mark_incomplete(
                reason,
                critical_event_drop_count=critical_event_drop_count,
            )

        def finalize(self) -> dict[str, Any]:
            return self.inner.finalize()

    with _patched("PolicyStreamingCaptureSession", FailAfterAcceptance):
        engine = _engine(expected_binding_digest=binding_digest)
        result = engine.generate(
            _request(
                engine,
                "policy-fault-required-write",
                capture=path,
                telemetry="request",
            )
        )
        resources = engine.resource_status
        closed = engine.close()
    drops = result.timings["critical_capture_event_drop_count"]
    if (
        result.outcome != "FINISHED"
        or result.capture_error != "CAPTURE_WRITE_FAILED"
        or result.numerical_policy != POLICY
        or type(drops) is not int
        or drops <= 0
        or not closed
        or not _resources_zero(resources)
        or not (path / "INCOMPLETE").is_file()
        or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "required policy capture write failure was not preserved")
    record = _incomplete_record(capture_root, "required-write-failure", "CAPTURE_WRITE_FAILED")
    scenario = _scenario(
        "required-capture-write-failure",
        binding_digest=binding_digest,
        mechanism="required PolicyStreamingCaptureSession raises after acceptance",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-storage-fault-simulation",
        observations={
            "outcome": result.outcome,
            "capture_error": result.capture_error,
            "critical_capture_event_drop_count": drops,
            "resources_after_terminal": resources,
            "capture_path": record["path"],
            "capture_complete_present": False,
            "capture_incomplete_present": True,
        },
    )
    return record, scenario


def _truncate_capture(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = capture_root / "success"
    target = capture_root / "truncated-copy"
    shutil.copytree(source, target)
    source_bundle = PolicyStreamingCaptureBundle.open(source)
    events = target / "events.jsonl"
    value = events.read_bytes()
    if len(value) < 2:
        _fail(ErrorCode.INTERNAL, "success policy capture event stream is unexpectedly empty")
    _replace_owned(events, value[:-1])
    code: str | None = None
    try:
        PolicyStreamingCaptureBundle.open(target)
    except IllmError as exc:
        code = exc.code.value
    if code != ErrorCode.CAPTURE_CORRUPT.value:
        _fail(ErrorCode.INTERNAL, "truncated policy capture was not rejected as corrupt")
    record = _signed(
        {
            "capture_id": "truncated-copy",
            "path": "captures/truncated-copy",
            "expected_state": "CORRUPT_REJECTED",
            "expected_error_code": code,
            "source_manifest_digest": source_bundle.manifest["manifest_digest"],
        },
        "capture_record_digest",
    )
    scenario = _scenario(
        "capture-truncation-detected",
        binding_digest=binding_digest,
        mechanism="one byte removed from a copied checksummed schema3 policy capture",
        evidence_kind="portable-policy-artifact-corruption-simulation",
        observations={"capture_path": record["path"], "reader_error_code": code},
    )
    return record, scenario


def _forced_exit(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fences: list[_DelayedFence] = []
    path = capture_root / "forced-exit"
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = _engine(expected_binding_digest=binding_digest)
        request = _request(
            engine,
            "policy-fault-forced-exit",
            max_new_tokens=2,
            capture=path,
            telemetry="request",
        )
        chunks: list[object] = []
        errors: list[BaseException] = []

        def consume() -> None:
            values, failures = _consume(engine, request)
            chunks.extend(values)
            errors.extend(failures)

        worker = threading.Thread(target=consume, name="illm-policy-evidence-forced-exit")
        worker.start()
        fence = _await_fence(fences)
        engine.abandon(request.request_id, reason="CONTROLLED_FORCED_EXIT")
        fence.release.set()
        worker.join(timeout=10)
        resources = engine.resource_status
        closed = engine.close()
    raised = errors[0] if len(errors) == 1 else None
    if (
        worker.is_alive()
        or not isinstance(raised, IllmError)
        or raised.code is not ErrorCode.ENGINE_UNHEALTHY
        or closed
        or resources.get("quarantined_requests") != 1
        or not (path / "INCOMPLETE").is_file()
        or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "forced policy exit did not retain quarantined ownership")
    record = _incomplete_record(capture_root, "forced-exit", "CONTROLLED_FORCED_EXIT")
    scenario = _scenario(
        "forced-exit-quarantine",
        binding_digest=binding_digest,
        mechanism="Engine.abandon is called while a delayed candidate-policy ticket owns resources",
        evidence_kind="actual-tiny-policy-runtime-with-controlled-client-force-simulation",
        observations={
            "raised_error_code": raised.code.value,
            "terminal_chunk_count": sum(isinstance(chunk, TerminalChunk) for chunk in chunks),
            "resources_quarantined": resources,
            "engine_close_succeeded": closed,
            "capture_path": record["path"],
        },
    )
    return record, scenario


def _capture_evidence(
    capture_root: Path,
    binding_digest: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    success, _ = _complete_capture(
        capture_root,
        "success",
        binding_digest=binding_digest,
        max_new_tokens=2,
    )
    length, _ = _complete_capture(
        capture_root,
        "length",
        binding_digest=binding_digest,
        max_new_tokens=1,
    )
    eos, _ = _complete_capture(
        capture_root,
        "eos",
        binding_digest=binding_digest,
        input_token_ids=(2,),
        max_new_tokens=4,
    )
    no_kv, _ = _complete_capture(
        capture_root,
        "no-kv",
        binding_digest=binding_digest,
        kv_mode="none",
        max_new_tokens=2,
    )
    records.extend((success, length, eos, no_kv))
    cancel_record, cancel_scenario = _cancel_after_launch(capture_root, binding_digest)
    records.append(cancel_record)
    scenarios.append(cancel_scenario)
    exporter_record, exporter_scenario = _exporter_stall(capture_root, binding_digest)
    records.append(exporter_record)
    scenarios.append(exporter_scenario)
    write_record, write_scenario = _required_write_failure(capture_root, binding_digest)
    records.append(write_record)
    scenarios.append(write_scenario)
    unknown_record, unknown_scenario = _unknown_fence(capture_root, binding_digest)
    records.append(unknown_record)
    scenarios.append(unknown_scenario)
    truncated_record, truncated_scenario = _truncate_capture(capture_root, binding_digest)
    records.append(truncated_record)
    scenarios.append(truncated_scenario)
    forced_record, forced_scenario = _forced_exit(capture_root, binding_digest)
    records.append(forced_record)
    scenarios.append(forced_scenario)
    document = _signed(
        {
            "schema_version": 1,
            "capture_contract": "schema3-policy-metadata-only-R0",
            "complete_capture_ids": list(COMPLETE_CAPTURE_IDS),
            "required_policy_capture_ids": list(REQUIRED_POLICY_CAPTURE_IDS),
            "incomplete_capture_ids": list(INCOMPLETE_CAPTURE_IDS),
            "corrupt_rejection_capture_ids": ["truncated-copy"],
            "numerical_policy_digest": POLICY.digest,
            "execution_binding_digest": binding_digest,
            "records": records,
        },
        "raw_digest",
    )
    normal_plans = {
        "none": cast(str, no_kv["memory_plan_digest"]),
        "contiguous": cast(str, success["memory_plan_digest"]),
    }
    if any(
        record.get("memory_plan_digest") != normal_plans["contiguous"]
        for record in records
        if record.get("capture_id") in {"length", "eos", "cancel-after-launch", "exporter-stall"}
    ):
        _fail(ErrorCode.INTERNAL, "normal policy capture MemoryPlans are inconsistent")
    return document, scenarios, normal_plans


def _fault_evidence(
    cycle_raw: Mapping[str, Any],
    capture_scenarios: Sequence[dict[str, Any]],
    binding_digest: str,
) -> tuple[dict[str, Any], str]:
    records = cast(list[dict[str, Any]], cycle_raw["records"])
    cancel_cycle = records[1]
    capacity, capacity_plan = _capacity_failure(binding_digest)
    scenarios = [
        _scenario(
            "cancel-before-launch",
            binding_digest=binding_digest,
            mechanism="cancellation follows candidate-policy metadata acceptance and precedes first launch",
            evidence_kind="actual-tiny-policy-runtime",
            observations={
                "cycle_index": cancel_cycle["cycle_index"],
                "outcome": cast(dict[str, Any], cancel_cycle["result"])["outcome"],
                "model_forward_started_count": sum(
                    event["kind"] == "model.forward_started"
                    for event in cast(list[dict[str, Any]], cancel_cycle["events"])
                ),
                "resources_after_terminal": cancel_cycle["resources_after_terminal"],
            },
        ),
        _finish_cancel_race(binding_digest),
        capacity,
        _known_failure("async-execution-error", _FailedFenceExecutor, ErrorCode.INTERNAL, binding_digest),
        _known_failure("numerical-error", _NonFiniteExecutor, ErrorCode.NUMERICAL_ERROR, binding_digest),
        _known_failure("oom", _OomExecutor, ErrorCode.RESOURCE_EXHAUSTED, binding_digest),
        *capture_scenarios,
    ]
    by_id = {item["scenario_id"]: item for item in scenarios}
    if set(by_id) != set(REQUIRED_FAULT_IDS) or len(by_id) != len(scenarios):
        _fail(ErrorCode.INTERNAL, "required policy lifecycle fault matrix is incomplete or duplicated")
    ordered = [by_id[scenario_id] for scenario_id in REQUIRED_FAULT_IDS]
    raw = _signed(
        {
            "schema_version": 1,
            "matrix_id": "illm.v02.policy.stateful-failure-matrix.v1",
            "required_scenario_ids": list(REQUIRED_FAULT_IDS),
            "required_pass_count": len(REQUIRED_FAULT_IDS),
            "numerical_policy_digest": POLICY.digest,
            "execution_binding_digest": binding_digest,
            "scenarios": ordered,
            "nonrequired_actual_gpu_fault": {
                "status": "not_run_resource_unavailable",
                "required_for_cpu_profile": False,
                "reason": "CPU policy state-machine component does not claim an actual GPU fault run",
                "simulation_results_must_not_be_relabelled_as_actual_gpu_faults": True,
            },
        },
        "raw_digest",
    )
    return raw, capacity_plan


def _source_and_binding(
    development_no_manifest: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if type(development_no_manifest) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "development_no_manifest must be bool")
    try:
        execution = build_execution_binding(PROJECT_ROOT)
    except Exception as exc:
        raise as_illm_error(exc) from exc
    binding = execution.to_dict()
    toolchain = cast(dict[str, Any], binding["toolchain"])
    bound = toolchain.get("binding") == "source-bound-manifest"
    if bound == development_no_manifest:
        _fail(
            ErrorCode.TOOLCHAIN_MISMATCH,
            "formal policy lifecycle evidence requires a source-bound manifest; development requires an unbound process",
        )
    manifest: dict[str, Any] | None = None
    if bound:
        manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
        if not manifest_name:
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "source-bound execution lacks its selected manifest path")
        try:
            value = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "selected toolchain manifest is unreadable") from exc
        if not isinstance(value, dict):
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "selected toolchain manifest must be an object")
        manifest = cast(dict[str, Any], value)
    source = _signed(
        {
            "schema_version": 1,
            "implementation_source_scope": binding["implementation_source_scope"],
            "implementation_source_digest": binding["implementation_source_digest"],
        },
        "source_identity_digest",
    )
    return source, binding, manifest


def _prepare_provenance(
    root: Path,
    source: Mapping[str, Any],
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, str]:
    _write_json(root / "provenance" / "source-identity.json", source)
    _write_json(root / "provenance" / "execution-binding.json", binding)
    _write_json(root / "provenance" / "numerical-policy.json", POLICY.to_dict())
    source_copy = root / "provenance" / "exercise_policy_streaming.py"
    _write_bytes(source_copy, Path(__file__).resolve().read_bytes())
    if manifest is not None:
        _write_json(root / "provenance" / "toolchain-manifest.json", manifest)
    registry = cast(dict[str, Any], metric_registry())
    conformance = cast(dict[str, Any], registry_conformance(registry))
    _write_json(root / "provenance" / "metric-registry-v1.json", registry)
    _write_json(root / "provenance" / "metric-registry-conformance.json", conformance)
    return {
        "path": "scripts/exercise_policy_streaming.py",
        "sha256": sha256_file(source_copy),
    }


def _inventory(root: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if _is_linklike(path):
            _fail(ErrorCode.CAPTURE_CORRUPT, "policy lifecycle writer found a link/reparse point")
        if path.is_file():
            if path.stat().st_nlink != 1:
                _fail(ErrorCode.CAPTURE_CORRUPT, "policy lifecycle writer found a hard-linked file")
            relative = path.relative_to(root).as_posix()
            if relative not in ignored:
                result[relative] = sha256_file(path)
    return result


def _write_checksums(root: Path) -> None:
    files = _inventory(root)
    _write_json(
        root / "checksums.json",
        {"schema_version": 1, "files": files, "inventory_digest": sha256_json(files)},
    )


def run_policy_exercise(
    *,
    output: Path,
    development_no_manifest: bool = False,
) -> dict[str, Any]:
    """Run the fixed candidate-policy CPU lifecycle exercise."""

    source, initial_binding, manifest = _source_and_binding(development_no_manifest)
    root = _new_root(output)
    _write_json(
        root / "INCOMPLETE",
        {
            "reason": "policy_lifecycle_exercise_in_progress",
            "formal_container_requested": not development_no_manifest,
        },
    )
    try:
        producer = _prepare_provenance(root, source, initial_binding, manifest)
        cycles, model_identity, cycle_binding = _run_cycles()
        if cycle_binding != initial_binding:
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "policy cycle binding drifted after output creation")
        binding_digest = sha256_json(initial_binding)
        captures, capture_scenarios, normal_plans = _capture_evidence(root / "captures", binding_digest)
        faults, capacity_plan = _fault_evidence(cycles, capture_scenarios, binding_digest)
        if build_execution_binding(PROJECT_ROOT).to_dict() != initial_binding:
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "policy lifecycle source/toolchain changed during execution")
        _write_json(root / "raw" / "cycles.json", cycles)
        _write_json(root / "raw" / "captures.json", captures)
        _write_json(root / "raw" / "fault-matrix.json", faults)
        formal = not development_no_manifest
        report = _signed(
            {
                "schema_version": 1,
                "report_kind": REPORT_KIND,
                "scope": SCOPE,
                "status": "pass",
                "evidence_stage": "source_bound_cpu_profile" if formal else "DEVELOPMENT_ONLY",
                "qualification_status": "unqualified",
                "qualification_scope": "cpu-only",
                "development_no_manifest": development_no_manifest,
                "formal_evidence_eligible": formal,
                "cpu_profile_evidence_eligible": formal,
                "global_qualification_granted": False,
                "gpu_qualification_granted": False,
                "source_identity": dict(source),
                "toolchain": dict(cast(dict[str, Any], initial_binding["toolchain"])),
                "producer": producer,
                "model_identity": model_identity,
                "numerical_policy": POLICY.to_dict(),
                "numerical_policy_digest": POLICY.digest,
                "execution_binding": initial_binding,
                "execution_binding_digest": binding_digest,
                "protocol_sha256": PROTOCOL_SHA256,
                "memory_plan_digests": {
                    "normal": normal_plans,
                    "fault_subplans": {"capacity-exceeded-preaccept": capacity_plan},
                },
                "metric_registry_digest": registry_digest(),
                "actual_cycle_count": CYCLE_COUNT,
                "finished_cycle_count": 50,
                "cancelled_cycle_count": 50,
                "race_count": RACE_COUNT,
                "required_fault_scenario_ids": list(REQUIRED_FAULT_IDS),
                "required_fault_pass_count": len(REQUIRED_FAULT_IDS),
                "required_matrix_complete": True,
                "required_policy_capture_ids": list(REQUIRED_POLICY_CAPTURE_IDS),
                "complete_capture_ids": list(COMPLETE_CAPTURE_IDS),
                "incomplete_capture_ids": list(INCOMPLETE_CAPTURE_IDS),
                "actual_gpu_faults_run": False,
                "actual_gpu_fault_claim": None,
                "simulation_boundary": (
                    "controlled CPU injections exercise candidate-policy lifecycle paths; "
                    "they are not actual GPU fault evidence"
                ),
                "performance_claim": None,
                "raw_payload_scope": "public_tiny_fixture_only",
                "raw_digests": {
                    "cycles": cycles["raw_digest"],
                    "captures": captures["raw_digest"],
                    "fault_matrix": faults["raw_digest"],
                },
            },
            "report_digest",
        )
        _write_json(root / "report" / "exercise-report.json", report)
        _write_checksums(root)
        if formal:
            _write_text(root / "COMPLETE", cast(str, report["report_digest"]) + "\n")
        else:
            _write_json(
                root / "DEVELOPMENT_ONLY",
                {
                    "evidence_stage": "DEVELOPMENT_ONLY",
                    "qualification_status": "unqualified",
                    "qualification_scope": "cpu-only",
                    "formal_evidence_eligible": False,
                    "cpu_profile_evidence_eligible": False,
                    "global_qualification_granted": False,
                    "gpu_qualification_granted": False,
                    "report_digest": report["report_digest"],
                },
            )
        (root / "INCOMPLETE").unlink()
        verify_policy_exercise_output(root, require_formal=formal)
        return report
    except BaseException:
        (root / "COMPLETE").unlink(missing_ok=True)
        (root / "DEVELOPMENT_ONLY").unlink(missing_ok=True)
        incomplete = root / "INCOMPLETE"
        if not incomplete.exists():
            try:
                _write_json(
                    incomplete,
                    {
                        "reason": "policy_lifecycle_exercise_failed",
                        "formal_container_requested": not development_no_manifest,
                    },
                )
            except OSError:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-no-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_policy_exercise(
            output=args.output,
            development_no_manifest=args.development_no_manifest,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report_digest": report["report_digest"],
                    "evidence_stage": report["evidence_stage"],
                    "qualification_status": report["qualification_status"],
                    "output": str(args.output),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        error = as_illm_error(exc)
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

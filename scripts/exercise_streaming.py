"""Produce portable V0.2 lifecycle, capture, and fault-matrix evidence.

The producer runs the public tiny decoder through real Engine lifecycles.  A
fault whose hardware condition cannot be created safely on CPU is injected at
an existing implementation seam and is labelled as simulation in every raw
record.  In particular, no injected CPU fault is evidence of an actual GPU
failure.  Formal output requires the current finalized, source-bound toolchain
manifest; the explicit development escape hatch is always unqualified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterator, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle, StreamingCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K
from illm.engine.facade import as_illm_error
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk, TerminalChunk, TokenChunk
from illm.executor import CompletionFence, EagerExecutor, ExecutionTicket, ImmediateFence
from illm.metrics import metric_registry, registry_conformance, registry_digest
from illm.sampling import SamplingSpec
from illm.stateful_exercise import verify_exercise_output as verify_portable_exercise_output

import illm.engine.streaming as streaming_module
from _toolchain_common import bootstrap_digest, project_digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_KIND = "illm-v02-streaming-lifecycle-exercise-v1"
SCOPE = "tiny-state-machine-cpu-v1"
CYCLE_COUNT = 100
RACE_COUNT = 8
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ZERO_RESOURCE_FIELDS = frozenset({
    "active_requests", "reservations", "quarantined_requests", "active_tickets",
    "input_leases", "output_leases", "workspace_leases", "kv_write_leases",
    "kv_reserved_bytes", "kv_valid_tokens", "kv_staged_tokens",
})
REQUIRED_FAULT_IDS = (
    "cancel-before-launch",
    "cancel-after-launch-delayed-fence",
    "finish-cancel-race",
    "capacity-exceeded-preaccept",
    "async-execution-error",
    "numerical-error",
    "oom",
    "unknown-fence-fail-stop",
    "optional-exporter-stall-drop",
    "required-capture-write-failure",
    "capture-truncation-detected",
    "forced-exit-quarantine",
)
COMPLETE_CAPTURE_IDS = (
    "success", "length", "eos", "cancel-after-launch", "no-kv", "exporter-stall",
)
INCOMPLETE_CAPTURE_IDS = ("required-write-failure", "unknown-fence", "forced-exit")


def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _fail(ErrorCode.INTERNAL, f"document already contains {field}")
    value[field] = sha256_json(value)
    return value


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(dict(value)) + b"\n")


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("ascii"))


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(ErrorCode.CAPTURE_CORRUPT, "duplicate JSON field in lifecycle evidence", {"field": key})
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique,
            parse_constant=lambda item: _fail(ErrorCode.CAPTURE_CORRUPT, "nonfinite JSON evidence", {"value": item}),
        )
    except IllmError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence must be an object: {path.name}")
    return cast(dict[str, Any], value)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence path is not portable")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence path escapes its sealed root")
    return value


def _source_identity() -> dict[str, Any]:
    paths: list[Path] = [
        PROJECT_ROOT / ".gitattributes",
        PROJECT_ROOT / ".gitignore",
        PROJECT_ROOT / "AI_INFERENCE_ENGINE_PROJECT_PLAN.md",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "requirements-bootstrap.lock",
        PROJECT_ROOT / "requirements-dev.lock",
    ]
    for relative in (
        ".github", "docs", "evidence", "src", "scripts", "tests", "schemas",
        "registries", "runtime", "configs", "fixtures",
    ):
        directory = PROJECT_ROOT / relative
        if directory.exists():
            paths.extend(
                path for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
            )
    project_files = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in sorted({item.resolve() for item in paths}, key=lambda item: item.as_posix())
        if path.is_file()
    }
    bootstrap_names = ("bootstrap.py", "_toolchain_common.py", "_runtime_artifacts.py")
    bootstrap_files = {
        f"scripts/{name}": sha256_file(PROJECT_ROOT / "scripts" / name)
        for name in bootstrap_names
        if (PROJECT_ROOT / "scripts" / name).is_file()
    }
    project_root_digest = sha256_json(sorted(project_files.items()))
    bootstrap_source_digest = sha256_json(sorted(bootstrap_files.items()))
    if project_root_digest != project_digest(PROJECT_ROOT) or bootstrap_source_digest != bootstrap_digest(PROJECT_ROOT):
        _fail(ErrorCode.INTERNAL, "embedded source inventory differs from toolchain digest algorithm")
    commit = "unversioned"
    dirty = True
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        if revision.returncode == status.returncode == 0 and revision.stdout.strip():
            commit = revision.stdout.strip()
            dirty = bool(status.stdout.strip())
    except OSError:
        pass
    return _signed({
        "schema_version": 1,
        "project_root_digest": project_root_digest,
        "bootstrap_digest": bootstrap_source_digest,
        "project_files": project_files,
        "bootstrap_files": bootstrap_files,
        "git_commit": commit,
        "git_dirty": dirty,
    }, "source_identity_digest")


def _toolchain_binding(
    development_no_manifest: bool, source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if type(development_no_manifest) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "development_no_manifest must be bool")
    if development_no_manifest:
        return _signed({
            "schema_version": 1,
            "binding": "development-unqualified",
            "formal_evidence_eligible": False,
            "manifest_digest": None,
            "manifest_environment_ignored": bool(os.environ.get("ILLM_TOOLCHAIN_MANIFEST")),
            "python_executable": str(Path(sys.executable).resolve()),
        }, "binding_digest"), None

    manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not manifest_name:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "run lifecycle exercise through a finalized manifest runner")
    manifest_path = Path(manifest_name).resolve()
    manifest = _object(manifest_path)
    digest = manifest.get("manifest_digest")
    required = {
        "schema_version", "finalized", "environment_id", "python_executable", "python_version",
        "project_root", "project_root_digest", "bootstrap_digest", "lock_digest",
        "runtime_profile_id", "runtime_lock_digest", "manifest_digest",
    }
    if not required <= set(manifest) or manifest.get("schema_version") != 1 or manifest.get("finalized") is not True:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "lifecycle exercise requires a finalized selected runtime profile")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None or digest != sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest checksum mismatch")
    environment_id = manifest.get("environment_id")
    python_executable = manifest.get("python_executable")
    lock_digest = manifest.get("lock_digest")
    runtime_lock_digest = manifest.get("runtime_lock_digest")
    if (
        not isinstance(environment_id, str)
        or os.environ.get("ILLM_ENVIRONMENT_ID") != environment_id
        or not isinstance(python_executable, str)
        or Path(python_executable).resolve() != Path(sys.executable).resolve()
        or not isinstance(lock_digest, str)
        or _DIGEST.fullmatch(lock_digest) is None
        or os.environ.get("ILLM_LOCK_DIGEST") != lock_digest
        or not isinstance(runtime_lock_digest, str)
        or _DIGEST.fullmatch(runtime_lock_digest) is None
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "current interpreter/environment differs from finalized manifest")
    if (
        Path(str(manifest.get("project_root"))).resolve() != PROJECT_ROOT
        or manifest.get("project_root_digest") != source["project_root_digest"]
        or manifest.get("bootstrap_digest") != source["bootstrap_digest"]
        or manifest.get("python_version") != platform.python_version()
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest is not bound to current lifecycle source")
    runtime_profile_id = manifest.get("runtime_profile_id")
    if not isinstance(runtime_profile_id, str) or not runtime_profile_id:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "runtime profile identity is missing")
    return _signed({
        "schema_version": 1,
        "binding": "finalized-source-bound-manifest",
        "formal_evidence_eligible": True,
        "manifest_digest": digest,
        "environment_id": environment_id,
        "lock_digest": lock_digest,
        "runtime_profile_id": runtime_profile_id,
        "runtime_lock_digest": runtime_lock_digest,
        "python_version": platform.python_version(),
        "project_root_digest": source["project_root_digest"],
        "bootstrap_digest": source["bootstrap_digest"],
    }, "binding_digest"), manifest


def _inventory(output: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    result: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            _fail(ErrorCode.CAPTURE_CORRUPT, "lifecycle evidence forbids symbolic links")
        if path.is_file() and path.name not in ignored:
            result[path.relative_to(output).as_posix()] = sha256_file(path)
    return result


def _write_checksums(output: Path) -> None:
    files = _inventory(output)
    _atomic_json(output / "checksums.json", {
        "schema_version": 1,
        "files": files,
        "inventory_digest": sha256_json(files),
    })


def _resources_zero(value: Mapping[str, Any]) -> bool:
    return set(value) == _ZERO_RESOURCE_FIELDS and all(type(item) is int and item == 0 for item in value.values())


def _request(
    engine: Engine,
    request_id: str,
    *,
    max_new_tokens: int = 1,
    input_token_ids: tuple[int, ...] = (1, 107, 104, 111, 111, 114),
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
        _fail(ErrorCode.INTERNAL, "stream did not produce exactly one terminal chunk")
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
            raise RuntimeError("controlled delayed fence timed out")
        self.base.wait()

    def is_complete(self) -> bool | None:
        return self.base.is_complete() if self.release.is_set() else False

    @property
    def succeeded(self) -> bool | None:
        return self.base.succeeded if self.release.is_set() else None


def _delayed_executor(fences: list[_DelayedFence]) -> type[EagerExecutor]:
    class DelayedExecutor(EagerExecutor):
        def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
            delayed = _DelayedFence(fence)
            fences.append(delayed)
            return delayed

    return DelayedExecutor


def _await_fence(fences: list[_DelayedFence], index: int = 0) -> _DelayedFence:
    deadline = perf_counter_ns() + 5_000_000_000
    while len(fences) <= index and perf_counter_ns() < deadline:
        threading.Event().wait(0.001)
    if len(fences) <= index or not fences[index].entered.wait(timeout=5):
        _fail(ErrorCode.INTERNAL, "controlled execution never reached delayed fence")
    return fences[index]


class _NonFiniteExecutor(EagerExecutor):
    def resolve(self, ticket: ExecutionTicket) -> tuple[float, ...]:
        values = super().resolve(ticket)
        return (float("nan"), *values[1:])


class _OomExecutor(EagerExecutor):
    def launch(self, context_ids: tuple[int, ...]) -> ExecutionTicket:
        del context_ids
        raise MemoryError("controlled pre-dispatch OOM")


class _FailedFenceExecutor(EagerExecutor):
    def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
        del fence
        return ImmediateFence(RuntimeError("controlled asynchronous execution error"))


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
            raise RuntimeError("controlled optional exporter timeout")


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
    mechanism: str,
    evidence_kind: str,
    observations: Mapping[str, Any],
) -> dict[str, Any]:
    if scenario_id not in REQUIRED_FAULT_IDS:
        _fail(ErrorCode.INTERNAL, "unknown required fault scenario")
    return _signed({
        "scenario_id": scenario_id,
        "status": "pass",
        "required_for_cpu_profile": True,
        "mechanism": mechanism,
        "evidence_kind": evidence_kind,
        "actual_tiny_runtime": True,
        "actual_gpu_fault": False,
        "observations": dict(observations),
    }, "scenario_digest")


def _run_cycles() -> tuple[dict[str, Any], dict[str, Any]]:
    engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
    records: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    try:
        for index in range(CYCLE_COUNT):
            request_id = f"lifecycle-cycle-{index:03d}"
            request = _request(engine, request_id)
            if index % 2 == 0:
                result = engine.generate(request)
                planned = "FINISHED"
            else:
                stream = engine.stream(request)
                first = next(stream)
                if not isinstance(first, MetadataChunk):
                    _fail(ErrorCode.INTERNAL, "cancel cycle was not accepted before cancellation")
                cancellation = engine.cancel(request_id, f"cancel-{index:03d}")
                chunks = list(stream)
                result = _terminal(chunks).result
                if not cancellation.won:
                    _fail(ErrorCode.INTERNAL, "cancel cycle did not win terminal arbitration")
                planned = "CANCELLED"
            resources = engine.resource_status
            events = _events(engine)
            if result.outcome != planned or not _resources_zero(resources):
                _fail(ErrorCode.ENGINE_UNHEALTHY, "lifecycle cycle did not return to baseline")
            if identity is None:
                identity = {
                    "model_spec_id": result.model_spec_id,
                    "sampling_spec_digest": result.sampling_spec_digest,
                    "kv_mode": result.kv_mode,
                    "memory_plan_digest": engine.memory_plan.digest,
                }
            records.append(_signed({
                "cycle_index": index,
                "request_id": request_id,
                "planned_outcome": planned,
                "result": result.to_dict(),
                "events": events,
                "resources_after_terminal": resources,
            }, "cycle_digest"))
    finally:
        closed = engine.close()
    if not closed or identity is None:
        _fail(ErrorCode.ENGINE_UNHEALTHY, "cycle engine did not close cleanly")
    raw = _signed({
        "schema_version": 1,
        "suite_id": "illm.v02.lifecycle.100-alternating.v1",
        "cycle_count": CYCLE_COUNT,
        "alternation": "even-finished-odd-cancel-before-launch",
        "finished_count": 50,
        "cancelled_count": 50,
        "records": records,
    }, "raw_digest")
    return raw, identity


def _complete_capture(
    capture_root: Path,
    case_id: str,
    *,
    kv_mode: str = "contiguous",
    input_token_ids: tuple[int, ...] = (1, 107, 104, 111, 111, 114),
    max_new_tokens: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / case_id
    engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode=kv_mode, project_root=PROJECT_ROOT)
    try:
        result = engine.generate(_request(
            engine, f"capture-{case_id}", max_new_tokens=max_new_tokens,
            input_token_ids=input_token_ids, capture=path, telemetry="request",
        ))
        resources = engine.resource_status
    finally:
        closed = engine.close()
    if not closed or not _resources_zero(resources) or result.capture_error is not None:
        _fail(ErrorCode.CAPTURE_INCOMPLETE, f"{case_id} capture did not finish cleanly")
    bundle = StreamingCaptureBundle.open(path)
    record = _signed({
        "capture_id": case_id,
        "path": f"captures/{case_id}",
        "request_id": result.request_id,
        "outcome": result.outcome,
        "finish_reason": result.finish_reason,
        "kv_mode": kv_mode,
        "manifest_digest": bundle.manifest["manifest_digest"],
        "event_count": len(bundle.events),
        "resources_after_terminal": resources,
    }, "capture_record_digest")
    return record, result.to_dict()


def _cancel_after_launch(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fences: list[_DelayedFence] = []
    path = capture_root / "cancel-after-launch"
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        request = _request(engine, "capture-cancel-after-launch", max_new_tokens=3, capture=path, telemetry="request")
        chunks: list[object] = []
        errors: list[BaseException] = []

        def consume() -> None:
            values, failures = _consume(engine, request)
            chunks.extend(values)
            errors.extend(failures)

        worker = threading.Thread(target=consume, name="illm-evidence-cancel-after-launch")
        worker.start()
        fence = _await_fence(fences)
        cancellation = engine.cancel(request.request_id, "cancel-after-launch")
        resources_while_pending = engine.resource_status
        fence.release.set()
        worker.join(timeout=10)
        if worker.is_alive() or errors:
            _fail(ErrorCode.INTERNAL, "cancel-after-launch worker did not terminate", {"errors": [type(item).__name__ for item in errors]})
        terminal = _terminal(chunks)
        resources_after = engine.resource_status
        closed = engine.close()
    if (
        not cancellation.won or terminal.outcome != "CANCELLED" or not closed
        or resources_while_pending.get("active_tickets") != 1
        or resources_while_pending.get("kv_write_leases") != 1
        or not _resources_zero(resources_after)
    ):
        _fail(ErrorCode.ENGINE_UNHEALTHY, "delayed cancellation contract failed")
    bundle = StreamingCaptureBundle.open(path)
    record = _signed({
        "capture_id": "cancel-after-launch",
        "path": "captures/cancel-after-launch",
        "request_id": terminal.request_id,
        "outcome": terminal.outcome,
        "finish_reason": terminal.finish_reason,
        "kv_mode": "contiguous",
        "manifest_digest": bundle.manifest["manifest_digest"],
        "event_count": len(bundle.events),
        "resources_after_terminal": resources_after,
    }, "capture_record_digest")
    scenario = _scenario(
        "cancel-after-launch-delayed-fence",
        mechanism="CompletionFence wrapper delays resolution after a real tiny forward",
        evidence_kind="actual-tiny-runtime-with-controlled-fence-delay-simulation",
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


def _finish_cancel_race() -> dict[str, Any]:
    fences: list[_DelayedFence] = []
    results: list[dict[str, Any]] = []
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        try:
            for index in range(RACE_COUNT):
                request = _request(engine, f"race-{index:02d}")
                chunks: list[object] = []
                errors: list[BaseException] = []

                def consume() -> None:
                    values, failures = _consume(engine, request)
                    chunks.extend(values)
                    errors.extend(failures)

                worker = threading.Thread(target=consume, name=f"illm-evidence-race-worker-{index}")
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

                release_thread = threading.Thread(target=release, name=f"illm-evidence-race-release-{index}")
                cancel_thread = threading.Thread(target=cancel, name=f"illm-evidence-race-cancel-{index}")
                release_thread.start()
                cancel_thread.start()
                barrier.wait(timeout=5)
                release_thread.join(timeout=5)
                cancel_thread.join(timeout=5)
                worker.join(timeout=10)
                if any(thread.is_alive() for thread in (release_thread, cancel_thread, worker)) or errors:
                    _fail(ErrorCode.INTERNAL, "finish/cancel race threads did not terminate")
                terminal = _terminal(chunks)
                events = _events(engine)
                terminal_events = [event for event in events if event["kind"] in {
                    "request.completed", "request.cancelled", "request.failed",
                }]
                resources = engine.resource_status
                if len(cancels) != 1 or len(terminal_events) != 1 or not _resources_zero(resources):
                    _fail(ErrorCode.ENGINE_UNHEALTHY, "finish/cancel race produced an invalid terminal/resource state")
                results.append({
                    "iteration": index,
                    "outcome": terminal.outcome,
                    "cancel_outcome": cancels[0]["outcome"],
                    "cancel_won": cancels[0]["won"],
                    "terminal_event_count": len(terminal_events),
                    "token_chunk_count": sum(isinstance(chunk, TokenChunk) for chunk in chunks),
                    "resources_after_terminal": resources,
                })
        finally:
            closed = engine.close()
    if not closed:
        _fail(ErrorCode.ENGINE_UNHEALTHY, "race engine did not close")
    return _scenario(
        "finish-cancel-race",
        mechanism="barrier concurrently releases delayed fence and submits cancellation",
        evidence_kind="actual-tiny-runtime-with-controlled-scheduling-simulation",
        observations={"race_count": RACE_COUNT, "iterations": results},
    )


def _capacity_failure() -> dict[str, Any]:
    engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", max_context_tokens=8, project_root=PROJECT_ROOT)
    error: IllmError | None = None
    try:
        engine.generate(_request(engine, "capacity-preaccept", input_token_ids=(1, 3, 4, 5, 6, 7, 8, 9)))
    except IllmError as exc:
        error = exc
    resources = engine.resource_status
    accepted_events = sum(event.kind is K.REQUEST_ACCEPTED for event in engine.last_events)
    closed = engine.close()
    if error is None or error.code is not ErrorCode.CONTEXT_LIMIT_EXCEEDED or accepted_events or not closed or not _resources_zero(resources):
        _fail(ErrorCode.INTERNAL, "capacity failure did not reject before acceptance")
    return _scenario(
        "capacity-exceeded-preaccept",
        mechanism="fixed MemoryPlan capacity is exceeded before Engine acceptance",
        evidence_kind="actual-tiny-runtime",
        observations={
            "error_code": error.code.value,
            "accepted_event_count": accepted_events,
            "resources_after_rejection": resources,
        },
    )


def _known_failure(scenario_id: str, executor: type[EagerExecutor], expected: ErrorCode) -> dict[str, Any]:
    with _patched("EagerExecutor", executor):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        result = engine.generate(_request(engine, f"fault-{scenario_id}", max_new_tokens=2))
        resources = engine.resource_status
        closed = engine.close()
    if result.outcome != "FAILED" or result.error_code != expected.value or result.output_token_ids or not closed or not _resources_zero(resources):
        _fail(ErrorCode.INTERNAL, f"{scenario_id} did not produce a safely reclaimed failed terminal")
    mechanism = {
        "async-execution-error": "completed failure fence injected after a real tiny forward",
        "numerical-error": "nonfinite last-row logit injected after a real tiny forward resolves",
        "oom": "MemoryError injected before executor dispatch",
    }[scenario_id]
    return _scenario(
        scenario_id,
        mechanism=mechanism,
        evidence_kind="actual-tiny-runtime-with-controlled-fault-simulation",
        observations={
            "outcome": result.outcome,
            "error_code": result.error_code,
            "committed_output_count": len(result.output_token_ids),
            "resources_after_terminal": resources,
        },
    )


def _unknown_fence(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "unknown-fence"
    with _patched("EagerExecutor", _UnknownFenceExecutor):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        error: IllmError | None = None
        try:
            tuple(engine.stream(_request(engine, "fault-unknown-fence", capture=path, telemetry="request")))
        except IllmError as exc:
            error = exc
        resources = engine.resource_status
        terminal_events = sum(event.kind in {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED} for event in engine.last_events)
        closed = engine.close()
    if (
        error is None or error.code is not ErrorCode.ENGINE_UNHEALTHY or closed
        or resources.get("quarantined_requests") != 1 or resources.get("active_tickets") != 1
        or terminal_events != 0 or not (path / "INCOMPLETE").is_file() or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "unknown fence did not fail-stop and quarantine")
    record = _signed({
        "capture_id": "unknown-fence",
        "path": "captures/unknown-fence",
        "expected_state": "INCOMPLETE",
        "expected_reason": "FENCE_OR_RESOURCE_STATE_UNKNOWN",
    }, "capture_record_digest")
    scenario = _scenario(
        "unknown-fence-fail-stop",
        mechanism="CompletionFence tri-state seam returns unknown after a real tiny forward",
        evidence_kind="actual-tiny-runtime-with-controlled-fence-state-simulation",
        observations={
            "raised_error_code": error.code.value,
            "terminal_ready_event_count": terminal_events,
            "resources_quarantined": resources,
            "engine_close_succeeded": closed,
            "capture_path": record["path"],
        },
    )
    return record, scenario


def _exporter_stall(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "exporter-stall"
    exporter = _BlockingExporter()
    engine = Engine(
        "fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT,
        exporters=(exporter,), exporter_capacity=1,
    )
    results: list[Any] = []
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            results.append(engine.generate(_request(
                engine, "fault-exporter-stall", max_new_tokens=4, capture=path, telemetry="request",
            )))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=generate, name="illm-evidence-exporter-stall")
    worker.start()
    if not exporter.entered.wait(timeout=5):
        _fail(ErrorCode.INTERNAL, "optional exporter did not receive an event")
    worker.join(timeout=10)
    if worker.is_alive() or errors or len(results) != 1:
        _fail(ErrorCode.INTERNAL, "optional exporter blocked inference")
    result = results[0]
    status = engine.exporter_status
    resources = engine.resource_status
    initial_close = engine.close(timeout=0.0)
    exporter.release.set()
    final_close = engine.close(timeout=5.0)
    if (
        result.outcome != "FINISHED" or result.capture_error is not None
        or status["copy_drop_count"] <= 0 or initial_close or not final_close or not _resources_zero(resources)
    ):
        _fail(ErrorCode.INTERNAL, "optional exporter isolation contract failed")
    bundle = StreamingCaptureBundle.open(path)
    record = _signed({
        "capture_id": "exporter-stall",
        "path": "captures/exporter-stall",
        "request_id": result.request_id,
        "outcome": result.outcome,
        "finish_reason": result.finish_reason,
        "kv_mode": "contiguous",
        "manifest_digest": bundle.manifest["manifest_digest"],
        "event_count": len(bundle.events),
        "resources_after_terminal": resources,
    }, "capture_record_digest")
    scenario = _scenario(
        "optional-exporter-stall-drop",
        mechanism="bounded optional EventBus sink is held while a real tiny request completes",
        evidence_kind="actual-tiny-runtime-with-controlled-exporter-stall-simulation",
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


def _required_write_failure(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = capture_root / "required-write-failure"
    real_session = StreamingCaptureSession

    class FailAfterAcceptance:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = real_session(*args, **kwargs)
            self.failed = False

        def emit(self, kind: K, payload: Mapping[str, Any]) -> Any:
            if kind is K.MODEL_FORWARD_STARTED and not self.failed:
                self.failed = True
                raise OSError("controlled required capture write failure")
            return self.inner.emit(kind, payload)

        def mark_incomplete(self, reason: str, *, critical_event_drop_count: int = 0) -> dict[str, Any]:
            return self.inner.mark_incomplete(reason, critical_event_drop_count=critical_event_drop_count)

        def finalize(self) -> dict[str, Any]:
            return self.inner.finalize()

    with _patched("StreamingCaptureSession", FailAfterAcceptance):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        result = engine.generate(_request(
            engine, "fault-required-write", capture=path, telemetry="request",
        ))
        resources = engine.resource_status
        closed = engine.close()
    drops = result.timings["critical_capture_event_drop_count"]
    if (
        result.outcome != "FINISHED" or result.capture_error != "CAPTURE_WRITE_FAILED"
        or type(drops) is not int or drops <= 0 or not closed or not _resources_zero(resources)
        or not (path / "INCOMPLETE").is_file() or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "required capture write failure was not preserved as INCOMPLETE")
    record = _signed({
        "capture_id": "required-write-failure",
        "path": "captures/required-write-failure",
        "expected_state": "INCOMPLETE",
        "expected_reason": "CAPTURE_WRITE_FAILED",
    }, "capture_record_digest")
    scenario = _scenario(
        "required-capture-write-failure",
        mechanism="required StreamingCaptureSession writer raises after acceptance",
        evidence_kind="actual-tiny-runtime-with-controlled-storage-fault-simulation",
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


def _truncate_capture(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = capture_root / "success"
    target = capture_root / "truncated-copy"
    shutil.copytree(source, target)
    events = target / "events.jsonl"
    value = events.read_bytes()
    if len(value) < 2:
        _fail(ErrorCode.INTERNAL, "success capture event stream is unexpectedly empty")
    _atomic_bytes(events, value[:-1])
    code: str | None = None
    try:
        StreamingCaptureBundle.open(target)
    except IllmError as exc:
        code = exc.code.value
    if code != ErrorCode.CAPTURE_CORRUPT.value:
        _fail(ErrorCode.INTERNAL, "truncated capture was not rejected as corrupt")
    record = _signed({
        "capture_id": "truncated-copy",
        "path": "captures/truncated-copy",
        "expected_state": "CORRUPT_REJECTED",
        "expected_error_code": code,
    }, "capture_record_digest")
    scenario = _scenario(
        "capture-truncation-detected",
        mechanism="one byte removed from a copied, checksummed complete capture",
        evidence_kind="portable-artifact-corruption-simulation",
        observations={"capture_path": record["path"], "reader_error_code": code},
    )
    return record, scenario


def _forced_exit(capture_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fences: list[_DelayedFence] = []
    path = capture_root / "forced-exit"
    with _patched("EagerExecutor", _delayed_executor(fences)):
        engine = Engine("fixture:tiny-decoder", device="cpu", kv_mode="contiguous", project_root=PROJECT_ROOT)
        request = _request(engine, "fault-forced-exit", max_new_tokens=2, capture=path, telemetry="request")
        chunks: list[object] = []
        errors: list[BaseException] = []

        def consume() -> None:
            values, failures = _consume(engine, request)
            chunks.extend(values)
            errors.extend(failures)

        worker = threading.Thread(target=consume, name="illm-evidence-forced-exit")
        worker.start()
        fence = _await_fence(fences)
        engine.abandon(request.request_id, reason="CONTROLLED_FORCED_EXIT")
        fence.release.set()
        worker.join(timeout=10)
        resources = engine.resource_status
        closed = engine.close()
    raised = errors[0] if len(errors) == 1 else None
    if (
        worker.is_alive() or not isinstance(raised, IllmError) or raised.code is not ErrorCode.ENGINE_UNHEALTHY
        or closed or resources.get("quarantined_requests") != 1
        or not (path / "INCOMPLETE").is_file() or (path / "COMPLETE").exists()
    ):
        _fail(ErrorCode.INTERNAL, "forced exit did not retain quarantined ownership")
    record = _signed({
        "capture_id": "forced-exit",
        "path": "captures/forced-exit",
        "expected_state": "INCOMPLETE",
        "expected_reason": "CONTROLLED_FORCED_EXIT",
    }, "capture_record_digest")
    scenario = _scenario(
        "forced-exit-quarantine",
        mechanism="Engine.abandon is called while a delayed real tiny ticket owns resources",
        evidence_kind="actual-tiny-runtime-with-controlled-client-force-simulation",
        observations={
            "raised_error_code": raised.code.value,
            "terminal_chunk_count": sum(isinstance(chunk, TerminalChunk) for chunk in chunks),
            "resources_quarantined": resources,
            "engine_close_succeeded": closed,
            "capture_path": record["path"],
        },
    )
    return record, scenario


def _capture_evidence(capture_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    success, _ = _complete_capture(capture_root, "success", max_new_tokens=2)
    length, _ = _complete_capture(capture_root, "length", max_new_tokens=1)
    eos, _ = _complete_capture(capture_root, "eos", input_token_ids=(2,), max_new_tokens=4)
    no_kv, _ = _complete_capture(capture_root, "no-kv", kv_mode="none", max_new_tokens=2)
    records.extend((success, length, eos, no_kv))
    cancel_record, cancel_scenario = _cancel_after_launch(capture_root)
    records.append(cancel_record)
    scenarios.append(cancel_scenario)
    exporter_record, exporter_scenario = _exporter_stall(capture_root)
    records.append(exporter_record)
    scenarios.append(exporter_scenario)
    write_record, write_scenario = _required_write_failure(capture_root)
    records.append(write_record)
    scenarios.append(write_scenario)
    unknown_record, unknown_scenario = _unknown_fence(capture_root)
    records.append(unknown_record)
    scenarios.append(unknown_scenario)
    truncated_record, truncated_scenario = _truncate_capture(capture_root)
    records.append(truncated_record)
    scenarios.append(truncated_scenario)
    forced_record, forced_scenario = _forced_exit(capture_root)
    records.append(forced_record)
    scenarios.append(forced_scenario)
    document = _signed({
        "schema_version": 1,
        "capture_contract": "schema2-metadata-only-R0",
        "complete_capture_ids": list(COMPLETE_CAPTURE_IDS),
        "incomplete_capture_ids": list(INCOMPLETE_CAPTURE_IDS),
        "corrupt_rejection_capture_ids": ["truncated-copy"],
        "records": records,
    }, "raw_digest")
    return document, scenarios


def _fault_evidence(cycle_raw: Mapping[str, Any], capture_scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = cast(list[dict[str, Any]], cycle_raw["records"])
    cancel_cycle = records[1]
    scenarios = [
        _scenario(
            "cancel-before-launch",
            mechanism="cancel is submitted after metadata acceptance and before first model launch",
            evidence_kind="actual-tiny-runtime",
            observations={
                "cycle_index": cancel_cycle["cycle_index"],
                "outcome": cast(dict[str, Any], cancel_cycle["result"])["outcome"],
                "model_forward_started_count": sum(
                    event["kind"] == "model.forward_started" for event in cast(list[dict[str, Any]], cancel_cycle["events"])
                ),
                "resources_after_terminal": cancel_cycle["resources_after_terminal"],
            },
        ),
        _finish_cancel_race(),
        _capacity_failure(),
        _known_failure("async-execution-error", _FailedFenceExecutor, ErrorCode.INTERNAL),
        _known_failure("numerical-error", _NonFiniteExecutor, ErrorCode.NUMERICAL_ERROR),
        _known_failure("oom", _OomExecutor, ErrorCode.RESOURCE_EXHAUSTED),
        *capture_scenarios,
    ]
    by_id = {item["scenario_id"]: item for item in scenarios}
    if set(by_id) != set(REQUIRED_FAULT_IDS) or len(by_id) != len(scenarios):
        _fail(ErrorCode.INTERNAL, "required lifecycle fault matrix is incomplete or duplicated")
    ordered = [by_id[scenario_id] for scenario_id in REQUIRED_FAULT_IDS]
    return _signed({
        "schema_version": 1,
        "matrix_id": "illm.v02.stateful-failure-matrix.v1",
        "required_scenario_ids": list(REQUIRED_FAULT_IDS),
        "required_pass_count": len(REQUIRED_FAULT_IDS),
        "scenarios": ordered,
        "nonrequired_actual_gpu_fault": {
            "status": "not_run_resource_unavailable",
            "required_for_cpu_profile": False,
            "reason": "CPU state-machine profile does not claim an actual GPU fault run",
            "simulation_results_must_not_be_relabelled_as_actual_gpu_faults": True,
        },
    }, "raw_digest")


def _prepare_provenance(
    output: Path,
    source: Mapping[str, Any],
    toolchain: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, str]:
    _atomic_json(output / "provenance" / "source-identity.json", source)
    source_copy = output / "provenance" / "exercise_streaming.py"
    _atomic_bytes(source_copy, Path(__file__).resolve().read_bytes())
    _atomic_json(output / "provenance" / "toolchain-binding.json", toolchain)
    if manifest is not None:
        _atomic_json(output / "provenance" / "toolchain-manifest.json", manifest)
    registry = cast(dict[str, Any], metric_registry())
    conformance = cast(dict[str, Any], registry_conformance(registry))
    _atomic_json(output / "provenance" / "metric-registry-v1.json", registry)
    _atomic_json(output / "provenance" / "metric-registry-conformance.json", conformance)
    return {
        "producer_source_sha256": sha256_file(source_copy),
        "source_identity_digest": cast(str, source["source_identity_digest"]),
        "toolchain_binding_digest": cast(str, toolchain["binding_digest"]),
        "metric_registry_digest": registry_digest(),
    }


def verify_exercise_output(output: Path, *, require_formal: bool = True) -> dict[str, Any]:
    """Delegate portable verification to the tensor-free package reader."""

    return verify_portable_exercise_output(output, require_formal=require_formal)


def run_exercise(
    *,
    output: Path,
    development_no_manifest: bool = False,
) -> dict[str, Any]:
    """Run the fixed tiny lifecycle exercise and persist portable raw proofs."""

    if not isinstance(output, Path):
        _fail(ErrorCode.INVALID_ARGUMENT, "output must be a Path")
    source = _source_identity()
    toolchain, manifest = _toolchain_binding(development_no_manifest, source)
    root = output.resolve()
    if root.is_symlink() or root.exists() and (not root.is_dir() or any(root.iterdir())):
        _fail(ErrorCode.INVALID_ARGUMENT, "lifecycle output must be a new, empty, non-symlink directory")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "INCOMPLETE", {"reason": "lifecycle_exercise_in_progress", "formal": not development_no_manifest})
    try:
        provenance = _prepare_provenance(root, source, toolchain, manifest)
        cycles, model_identity = _run_cycles()
        captures, capture_scenarios = _capture_evidence(root / "captures")
        faults = _fault_evidence(cycles, capture_scenarios)
        _atomic_json(root / "raw" / "cycles.json", cycles)
        _atomic_json(root / "raw" / "captures.json", captures)
        _atomic_json(root / "raw" / "fault-matrix.json", faults)
        report = _signed({
            "schema_version": 1,
            "report_kind": REPORT_KIND,
            "scope": SCOPE,
            "status": "pass",
            "qualification_tier": "unqualified",
            "development_no_manifest": development_no_manifest,
            "formal_evidence_eligible": not development_no_manifest,
            "source_identity": dict(source),
            "toolchain": dict(toolchain),
            "producer": provenance,
            "model_identity": model_identity,
            "metric_registry_digest": registry_digest(),
            "actual_cycle_count": CYCLE_COUNT,
            "finished_cycle_count": 50,
            "cancelled_cycle_count": 50,
            "required_fault_scenario_ids": list(REQUIRED_FAULT_IDS),
            "required_fault_pass_count": len(REQUIRED_FAULT_IDS),
            "required_matrix_complete": True,
            "actual_gpu_faults_run": False,
            "actual_gpu_fault_claim": None,
            "simulation_boundary": "controlled CPU injections exercise implementation paths; they are not actual GPU fault evidence",
            "performance_claim": None,
            "raw_digests": {
                "cycles": cycles["raw_digest"],
                "captures": captures["raw_digest"],
                "fault_matrix": faults["raw_digest"],
            },
        }, "report_digest")
        _atomic_json(root / "report" / "exercise-report.json", report)
        _write_checksums(root)
        if development_no_manifest:
            _atomic_json(root / "DEVELOPMENT_ONLY", {
                "qualification_tier": "unqualified",
                "formal_evidence_eligible": False,
                "report_digest": report["report_digest"],
            })
            (root / "INCOMPLETE").unlink()
            verify_portable_exercise_output(root, require_formal=False)
        else:
            _atomic_text(root / "COMPLETE", cast(str, report["report_digest"]) + "\n")
            (root / "INCOMPLETE").unlink()
            verify_portable_exercise_output(root, require_formal=True)
        return report
    except BaseException as exc:
        (root / "COMPLETE").unlink(missing_ok=True)
        (root / "DEVELOPMENT_ONLY").unlink(missing_ok=True)
        try:
            _atomic_json(root / "INCOMPLETE", {
                "reason": "lifecycle_exercise_failed",
                "exception_type": type(exc).__name__,
                "formal": not development_no_manifest,
            })
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-no-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_exercise(output=args.output, development_no_manifest=args.development_no_manifest)
        print(json.dumps({
            "status": report["status"],
            "report_digest": report["report_digest"],
            "development_no_manifest": report["development_no_manifest"],
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        error = as_illm_error(exc)
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

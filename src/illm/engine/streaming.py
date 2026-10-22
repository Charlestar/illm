"""V0.2 single-request orchestration, separate from the frozen V0.1 oracle.

The state lock arbitrates output/cancel/finish, not GPU execution. Tickets own
their addresses until completion is proved. Capture is an independent required
writer; optional exporters receive bounded copies and cannot decide outcomes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter_ns
from typing import Any, Generator, Sequence

from illm import __version__
from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.streaming import StreamingCaptureSession
from illm.capture.policy_streaming import PolicyStreamingCaptureSession
from illm.domain.errors import EmergencyAbandon, ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.domain.streaming_events import StreamingEvent, StreamingEventKind as K
from illm.domain.policy_events import PolicyStreamingEvent
from illm.engine.facade import as_illm_error
from illm.engine.streaming_contracts import (
    CancelResult, GenerationRequest, GenerationResult, MetadataChunk, StreamChunk,
    TerminalChunk, TokenChunk,
)
from illm.executor import EagerExecutor, ExecutionTicket
from illm.memory import MemoryPlan, build_memory_plan
from illm.memory.policy import PolicyMemoryPlan, build_policy_memory_plan
from illm.metrics import RequestObservation, build_streaming_metrics
from illm.model.loader import LoadedModel, load_model
from illm.model.spec import ModelError
from illm.numerical_policy import NumericalExecutionPolicyV1, NumericalPolicyError
from illm.observability.bus import EventBus, EventSink
from illm.sampling import SamplingState, commit_sample, prepare_sample
from illm.tokenizer.streaming import DetokenizerFactory, IncrementalDetokenizer


@dataclass(slots=True)
class _RequestRuntime:
    request: GenerationRequest
    executor: EagerExecutor
    input_ids: tuple[int, ...]
    detokenizer: IncrementalDetokenizer
    sampling: SamplingState
    session: StreamingCaptureSession | PolicyStreamingCaptureSession | None
    tokenization_started_ns: int
    tokenization_ended_ns: int
    privacy_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    events: list[StreamingEvent] = field(default_factory=list)
    accepted_ns: int = 0
    phase: str = "TOKENIZED"
    output: tuple[int, ...] = ()
    token_ready_ns: list[int] = field(default_factory=list)
    query_tokens: list[int] = field(default_factory=list)
    valid_counts: list[int] = field(default_factory=list)
    forward_wall_ns: list[int] = field(default_factory=list)
    kv_valid_tokens: int = 0
    outcome: str | None = None
    finish_reason: str | None = None
    error_code: str | None = None
    cancellation_id: str | None = None
    terminal_ready: bool = False
    resources_reclaimed: bool = False
    capture_error: str | None = None
    capture_broken: bool = False
    capture_critical_drop_count: int = 0
    abandoned: bool = False
    ticket: ExecutionTicket | None = None
    open_write: dict[str, Any] | None = None
    allocation_announced: bool = False
    result: GenerationResult | None = None
    gate_released: bool = False


def _error(exc: Exception) -> IllmError:
    if isinstance(exc, ModelError) and exc.code in {"EXECUTION_STATE_UNKNOWN", "EXECUTION_FAILED"}:
        return IllmError(ErrorCode.ENGINE_UNHEALTHY if exc.code == "EXECUTION_STATE_UNKNOWN" else ErrorCode.INTERNAL,
                         "executor completion could not be established" if exc.code == "EXECUTION_STATE_UNKNOWN"
                         else "executor reported a completed failed operation")
    error = as_illm_error(exc)
    if error.code in {ErrorCode.UNSUPPORTED_MODEL, ErrorCode.UNSUPPORTED_SAMPLING_PARAMETER}:
        return IllmError(ErrorCode.UNSUPPORTED, error.message, error.details)
    return error


class Engine:
    def __init__(self, model: str | Path = "fixture:tiny-decoder", *, device: str = "cpu",
                 kv_mode: str = "contiguous", max_context_tokens: int | None = None,
                 memory_budget_bytes: int = 6 * 1024 ** 3, project_root: Path | None = None,
                 exporters: Sequence[EventSink] = (), exporter_capacity: int = 256,
                 numerical_policy: NumericalExecutionPolicyV1 | None = None) -> None:
        if kv_mode not in {"none", "contiguous"}:
            raise IllmError(ErrorCode.UNSUPPORTED, "kv_mode must be none or contiguous")
        if not isinstance(model, (str, Path)) or not str(model):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "model must be a local profile or fixture")
        if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "device must be cpu, cuda or cuda:<index>")
        if type(exporter_capacity) is not int or exporter_capacity < 1:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "exporter_capacity must be a positive integer")
        if type(memory_budget_bytes) is not int or memory_budget_bytes < 1:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "memory_budget_bytes must be a positive integer")
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens < 1):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "max_context_tokens must be a positive integer or None")
        if numerical_policy is not None:
            if not isinstance(numerical_policy, NumericalExecutionPolicyV1):
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "numerical_policy must be a registered policy object")
            if device != "cpu":
                raise IllmError(ErrorCode.UNSUPPORTED, "policy CPU implementation rejects CUDA before model loading")
            try:
                numerical_policy.validate_kv_mode(kv_mode)
            except NumericalPolicyError as exc:
                raise IllmError(ErrorCode.UNSUPPORTED, str(exc)) from exc
        self._numerical_policy = numerical_policy
        try:
            self._loaded: LoadedModel = load_model(model, device=device, project_root=project_root)
            capacity = self._loaded.max_context_tokens if max_context_tokens is None else max_context_tokens
            if type(capacity) is not int or not 1 <= capacity <= self._loaded.max_context_tokens:
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "max_context_tokens exceeds the selected profile")
            self._device = str(self._loaded.reference.canonical_weight("model.embed_tokens.weight").device)
            self._memory_plan: MemoryPlan | PolicyMemoryPlan = (
                build_policy_memory_plan(self._loaded, numerical_policy, kv_mode, capacity, memory_budget_bytes,
                                         project_root=(project_root or Path.cwd()).resolve())
                if numerical_policy is not None else build_memory_plan(self._loaded, kv_mode, capacity, memory_budget_bytes)
            )
            self.memory_plan.require_admission(1)
            self._execution_binding = (self._memory_plan.execution_binding
                                       if isinstance(self._memory_plan, PolicyMemoryPlan) else None)
            self._detokenizers = DetokenizerFactory(self._loaded.tokenizer, self._loaded.assets_directory / "tokenizer.json")
        except EmergencyAbandon:
            raise
        except Exception as exc:
            raise _error(exc) from exc
        self._model = str(model)
        self._kv_mode = kv_mode
        self._capacity = capacity
        self._gate = Lock()
        self._state_lock = RLock()
        self._event_lock = RLock()
        self._active: _RequestRuntime | None = None
        self._seen_ids: set[str] = set()
        self._terminals: dict[str, tuple[str, bool, str | None]] = {}
        self._quarantine: list[_RequestRuntime] = []
        self._closed = False
        self._unhealthy = False
        self._last_events: tuple[StreamingEvent, ...] = ()
        self._last_result: GenerationResult | None = None
        self._observations: list[RequestObservation] = []
        self._rejected_count = 0
        self._bus = EventBus((), tuple(exporters), best_effort_capacity=exporter_capacity)
        self._identity = self._build_identity((project_root or Path.cwd()).resolve())

    def _build_identity(self, root: Path) -> dict[str, Any]:
        commit, dirty = "unversioned", True
        try:
            commit_run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
            status_run = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
            if commit_run.returncode == status_run.returncode == 0:
                commit, dirty = commit_run.stdout.strip(), bool(status_run.stdout.strip())
        except OSError:
            pass
        toolchain_digest = sha256_json({"binding": "unbound-development"})
        bound = False
        manifest_path = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
        if manifest_path:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            toolchain_digest = str(manifest["manifest_digest"])
            bound = True
        return {"engine_version": __version__, "git_commit": commit, "git_dirty": dirty,
                "toolchain_digest": toolchain_digest, "toolchain_bound": bound,
                "device": self.device, "dtype": self._loaded.spec.runtime_dtype,
                "model_spec_id": self.model_spec_id,
                "weights_digest": self._loaded.weight_manifest["weights_digest"],
                "load_manifest_digest": self._loaded.weight_manifest["manifest_digest"],
                "tokenizer_digest": sha256_json(self._loaded.tokenizer.to_dict()),
                "kv_mode": self.kv_mode, "memory_plan_evidence_kind": "static_conservative_estimate"}

    @property
    def model_spec_id(self) -> str:
        return self._loaded.spec.spec_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def device(self) -> str:
        return self._device

    @property
    def kv_mode(self) -> str:
        return self._kv_mode

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def memory_plan(self) -> MemoryPlan | PolicyMemoryPlan:
        return self._memory_plan

    @property
    def numerical_policy(self) -> NumericalExecutionPolicyV1 | None:
        return self._numerical_policy

    @property
    def last_events(self) -> tuple[StreamingEvent, ...]:
        return self._last_events

    @property
    def last_result(self) -> GenerationResult | None:
        return self._last_result

    @property
    def exporter_status(self) -> dict[str, int]:
        return {"error_count": self._bus.exporter_error_count, "copy_drop_count": self._bus.exporter_drop_count,
                "queue_high_watermark": self._bus.exporter_queue_high_watermark}

    @property
    def observations(self) -> tuple[RequestObservation, ...]:
        with self._state_lock:
            return tuple(self._observations)

    def metrics(self) -> dict[str, Any]:
        with self._state_lock:
            if self._unhealthy or self._quarantine or (self._active is not None and not self._active.terminal_ready):
                unresolved = [*self._quarantine]
                if self._active is not None and all(self._active is not item for item in unresolved):
                    unresolved.append(self._active)
                raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "latency report has unresolved accepted requests",
                                {"unresolved_owned_requests": len(unresolved),
                                 "unresolved_accepted_requests": sum(bool(item.accepted_ns) for item in unresolved)})
            return build_streaming_metrics(self.observations)

    @property
    def resource_status(self) -> dict[str, int]:
        with self._state_lock:
            runtimes = [*self._quarantine]
            if self._active is not None and not self._active.terminal_ready and all(self._active is not item for item in runtimes):
                runtimes.append(self._active)
            totals = {name: 0 for name in ("active_tickets", "input_leases", "output_leases", "workspace_leases",
                                         "kv_write_leases", "kv_reserved_bytes", "kv_valid_tokens", "kv_staged_tokens")}
            for runtime in runtimes:
                for name, value in asdict(runtime.executor.resource_status).items():
                    totals[name] += value
            return {"active_requests": int(self._active is not None and not self._active.terminal_ready),
                    "reservations": len(runtimes), "quarantined_requests": len(self._quarantine), **totals}

    def _record_rejection(self, request: GenerationRequest, error: IllmError) -> None:
        # An offered duplicate ID is not the accepted request with that ID.
        self._rejected_count += 1
        offer_id = f"{request.request_id}.rejected-offer-{self._rejected_count}"
        reason = str(error.details.get("reason", error.code.value)) if error.details else error.code.value
        self._observations.append(RequestObservation(offer_id, None, (), None, None, None, None, reason))

    def _begin(self, request: GenerationRequest) -> _RequestRuntime:
        if not isinstance(request, GenerationRequest):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "stream requires the V0.2 GenerationRequest")
        if not self._gate.acquire(blocking=False):
            error = IllmError(ErrorCode.OVERLOADED, "one accepted request is already active", {"reason": "SINGLE_ACTIVE_LIMIT"})
            with self._state_lock:
                self._record_rejection(request, error)
            raise error
        executor: EagerExecutor | None = None
        try:
            with self._state_lock:
                if self._unhealthy:
                    raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "engine is fail-stopped; recreate it after external recovery")
                if self._closed:
                    raise IllmError(ErrorCode.INVALID_ARGUMENT, "engine is closed")
                if request.request_id in self._seen_ids:
                    raise IllmError(ErrorCode.INVALID_ARGUMENT, "request_id cannot be reused in this engine lifetime")
                self._seen_ids.add(request.request_id)
            if request.model_spec_id != self.model_spec_id:
                raise IllmError(ErrorCode.UNSUPPORTED, "request model_spec_id differs from loaded model")
            started = perf_counter_ns()
            if request.prompt is not None:
                input_ids = self._loaded.tokenizer.encode(request.prompt, add_bos=request.add_bos)
            else:
                input_ids = request.input_token_ids or ()
                if request.add_bos is True:
                    input_ids = (self._loaded.spec.bos_token_id, *input_ids)
            if not input_ids or any(token >= self._loaded.spec.vocab_size for token in input_ids):
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "input IDs are empty or outside vocabulary")
            ended = perf_counter_ns()
            if len(input_ids) + request.max_new_tokens > self.capacity:
                raise IllmError(ErrorCode.CONTEXT_LIMIT_EXCEEDED, "prompt plus requested output exceeds fixed context",
                                {"capacity_tokens": self.capacity, "requested_tokens": len(input_ids) + request.max_new_tokens})
            self.memory_plan.require_admission(len(input_ids) + request.max_new_tokens)
            executor = (EagerExecutor(self._loaded, self.kv_mode, self.capacity, numerical_policy=self.numerical_policy)
                        if self.numerical_policy is not None else EagerExecutor(self._loaded, self.kv_mode, self.capacity))
            session: StreamingCaptureSession | PolicyStreamingCaptureSession | None = None
            if request.capture is not None:
                try:
                    if self.numerical_policy is not None:
                        assert self._execution_binding is not None
                        session = PolicyStreamingCaptureSession(
                            request.capture, self._identity, request.request_id,
                            request.sampling.to_dict(), self.memory_plan.to_dict(),
                            numerical_policy=self.numerical_policy.to_dict(),
                            execution_binding=self._execution_binding.to_dict(),
                            observation_profile=request.telemetry,
                        )
                    else:
                        session = StreamingCaptureSession(request.capture, self._identity, request.request_id,
                                                          request.sampling.to_dict(), self.memory_plan.to_dict())
                except (OSError, IllmError) as exc:
                    raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "required capture could not be created before acceptance") from exc
            runtime = _RequestRuntime(request, executor, input_ids, self._detokenizers.new(),
                                      SamplingState.for_request(request.request_id, request.sampling), session, started, ended)
            with self._state_lock:
                self._active = runtime
                self._last_events = ()
                self._last_result = None
                self._emit(runtime, K.REQUEST_TOKENIZED, {"started_ns": started, "ended_ns": ended, "input_tokens": len(input_ids)})
                event = self._emit(runtime, K.REQUEST_ACCEPTED, {"input_tokens": len(input_ids), "max_new_tokens": request.max_new_tokens})
                runtime.accepted_ns = event.monotonic_ns
                self._emit(runtime, K.CLOCK_SNAPSHOT, {"host_monotonic_ns": perf_counter_ns()})
                if self.kv_mode == "contiguous":
                    self._emit(runtime, K.KV_ALLOCATION_COMMITTED,
                               {"allocation_id": "kv-0", "capacity_tokens": self.capacity,
                                "allocated_bytes": executor.resource_status.kv_reserved_bytes})
                    runtime.allocation_announced = True
            return runtime
        except EmergencyAbandon:
            self._unhealthy = True
            runtime_at_interrupt = self._active
            if runtime_at_interrupt is not None:
                runtime_at_interrupt.abandoned = True
                runtime_at_interrupt.capture_error = "CLI_SECOND_INTERRUPT"
                self._quarantine_runtime(runtime_at_interrupt, "CLI_SECOND_INTERRUPT")
                self._release_gate(runtime_at_interrupt)
            else:
                if executor is not None:
                    executor.close()  # Only empty reservations exist before acceptance.
                self._gate.release()
            raise
        except Exception as exc:
            if executor is not None:
                executor.close()
            error = _error(exc)
            with self._state_lock:
                self._active = None
                self._record_rejection(request, error)
            self._gate.release()
            raise error from exc

    def _emit(self, runtime: _RequestRuntime, kind: K, payload: dict[str, Any]) -> StreamingEvent:
        with self._event_lock:
            payload = {"request_id": runtime.request.request_id, **payload}
            event: StreamingEvent | None = None
            if runtime.session is not None and runtime.capture_broken:
                runtime.capture_critical_drop_count += int(kind.replay_critical)
            if runtime.session is not None and not runtime.capture_broken:
                try:
                    event = runtime.session.emit(kind, payload)
                except Exception:
                    runtime.capture_broken = True
                    runtime.capture_error = "CAPTURE_WRITE_FAILED"
                    runtime.capture_critical_drop_count += int(kind.replay_critical)
                    self._incomplete(runtime, "CAPTURE_WRITE_FAILED")
                    if runtime.accepted_ns == 0:
                        raise IllmError(ErrorCode.CAPTURE_INCOMPLETE,
                                        "required capture failed before acceptance was committed")
            if event is None:
                event_type = PolicyStreamingEvent if self.numerical_policy is not None else StreamingEvent
                event = event_type("stateful-engine", len(runtime.events), perf_counter_ns(), kind, payload)
            runtime.events.append(event)
            if runtime.request.telemetry in {"request", "ops"}:
                self._bus.emit(DomainEvent("stateful-export-copy", event.producer_seq_no, event.monotonic_ns,
                                          EventKind.DIAGNOSTIC, {"streaming_event": event.to_dict()}))
            return event

    def _incomplete(self, runtime: _RequestRuntime, reason: str) -> None:
        if runtime.session is not None:
            try:
                runtime.session.mark_incomplete(reason, critical_event_drop_count=runtime.capture_critical_drop_count)
            except Exception:
                runtime.capture_error = runtime.capture_error or "CAPTURE_INCOMPLETE_MARKER_FAILED"

    def _phase(self, runtime: _RequestRuntime, phase: str) -> None:
        if runtime.phase != phase:
            self._emit(runtime, K.REQUEST_PHASE_CHANGED, {"from_phase": runtime.phase, "to_phase": phase})
            runtime.phase = phase

    def cancel(self, request_id: str, cancellation_id: str) -> CancelResult:
        if any(not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None
               for value in (request_id, cancellation_id)):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "cancel IDs must be safe opaque identifiers")
        with self._state_lock:
            runtime = self._active
            if runtime is None or runtime.request.request_id != request_id:
                previous = self._terminals.get(request_id)
                if previous is None:
                    raise IllmError(ErrorCode.INVALID_ARGUMENT, "cancel request_id is unknown")
                outcome, ready, winner = previous
                return CancelResult(request_id, cancellation_id, outcome, ready, winner == cancellation_id)
            if runtime.outcome is None:
                runtime.outcome, runtime.cancellation_id = "CANCELLED", cancellation_id
                self._emit(runtime, K.REQUEST_CANCEL_REQUESTED, {"cancellation_id": cancellation_id})
                self._phase(runtime, "CANCEL_PENDING")
                self._decided(runtime)
            return CancelResult(request_id, cancellation_id, runtime.outcome, runtime.terminal_ready,
                                runtime.cancellation_id == cancellation_id)

    def abandon(self, request_id: str, reason: str = "CLI_SECOND_INTERRUPT") -> None:
        """Emergency fail-stop only: retain unresolved resources; never reclaim."""
        if not isinstance(reason, str) or not reason:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "abandon requires a nonempty reason")
        with self._state_lock:
            runtime = self._active
            if runtime is None or runtime.request.request_id != request_id:
                return
            runtime.abandoned = True
            runtime.capture_error = reason
            self._quarantine_runtime(runtime, reason)

    def _announce_write(self, runtime: _RequestRuntime, position: int, context_tokens: int) -> None:
        if self.kv_mode == "contiguous":
            write = {"write_id": f"write-{position}", "allocation_id": "kv-0",
                     "start_token": runtime.kv_valid_tokens,
                     "token_count": context_tokens - runtime.kv_valid_tokens,
                     "lease_id": f"lease-{position}", "fence_id": f"fence-{position}"}
            runtime.open_write = write
            self._emit(runtime, K.KV_WRITE_STARTED, {**write, "measurement_boundary": "host_before_launch"})

    def _decided(self, runtime: _RequestRuntime) -> None:
        payload: dict[str, Any] = {"outcome": runtime.outcome,
                                   "committed_output_count": len(runtime.output),
                                   "kv_valid_token_count": runtime.kv_valid_tokens}
        if runtime.cancellation_id is not None:
            payload["cancellation_id"] = runtime.cancellation_id
        if runtime.error_code is not None:
            payload["error_code"] = runtime.error_code
        if runtime.finish_reason is not None:
            payload["reason"] = runtime.finish_reason
        self._emit(runtime, K.REQUEST_TERMINAL_DECIDED, payload)

    def _resolve_write(self, runtime: _RequestRuntime, *, committed: bool) -> None:
        write = runtime.open_write
        if write is not None:
            payload = {name: write[name] for name in ("write_id", "allocation_id", "lease_id", "fence_id")}
            payload["fence_completed"] = True
            if committed:
                payload["kv_valid_token_count"] = runtime.kv_valid_tokens
            self._emit(runtime, K.KV_WRITE_COMMITTED if committed else K.KV_WRITE_ABORTED, payload)
            runtime.open_write = None

    def _step(self, runtime: _RequestRuntime) -> TokenChunk | None:
        position = len(runtime.output)
        context = (*runtime.input_ids, *runtime.output)
        phase = "PREFILLING" if position == 0 else "DECODING"
        query_tokens = len(context) if self.kv_mode == "none" or position == 0 else 1
        forward_payload = {"phase": phase, "query_tokens": query_tokens,
                           "context_tokens": len(context), "output_position": position}
        with self._state_lock:
            if runtime.abandoned:
                raise EmergencyAbandon
            if runtime.outcome is not None:
                return None
            self._phase(runtime, phase)
            self._emit(runtime, K.MODEL_FORWARD_STARTED, forward_payload)
            self._announce_write(runtime, position, len(context))
        started = perf_counter_ns()
        ticket = runtime.executor.launch(context)
        runtime.ticket = ticket
        runtime.query_tokens.append(ticket.query_tokens)
        if ticket.query_tokens != query_tokens:
            raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "executor query coverage differs from the launch contract")
        logits = runtime.executor.resolve(ticket)
        elapsed = perf_counter_ns() - started
        runtime.forward_wall_ns.append(elapsed)
        with self._state_lock:
            if runtime.abandoned:
                raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "request was abandoned with unresolved ownership")
            self._emit(runtime, K.MODEL_FORWARD_COMPLETED, {**forward_payload, "duration_ns": elapsed})
            self._emit(runtime, K.MODEL_FORWARD, {**forward_payload, "duration_ns": elapsed, "evidence_kind": "measured"})
            if runtime.outcome is not None:
                runtime.executor.abort(ticket)
                runtime.ticket = None
                self._resolve_write(runtime, committed=False)
                return None
            proposal = prepare_sample(logits, runtime.request.sampling, runtime.sampling)
            next_state = commit_sample(runtime.sampling, proposal, spec=runtime.request.sampling, logits=logits)
            runtime.kv_valid_tokens = runtime.executor.commit(ticket)
            runtime.ticket = None
            self._resolve_write(runtime, committed=True)
            runtime.sampling = next_state
            runtime.output = (*runtime.output, proposal.token_id)
            runtime.valid_counts.append(runtime.kv_valid_tokens)
            delta = runtime.detokenizer.push(proposal.token_id)
            ready = perf_counter_ns()
            runtime.token_ready_ns.append(ready)
            if position == 0:
                self._phase(runtime, "DECODING")
            payload = {"output_index": position, "committed_output_count": len(runtime.output),
                       "kv_valid_token_count": runtime.kv_valid_tokens, "server_ready_ns": ready,
                       "token_hmac": hmac.new(runtime.privacy_key, canonical_json_bytes([position, proposal.token_id]),
                                              hashlib.sha256).hexdigest(),
                       "next_output_position": next_state.next_output_position,
                       "rng_draws_consumed": next_state.rng_draws_consumed,
                       "draw_address": [position, 0, 0] if proposal.draw is not None else None}
            if position == 0:
                self._emit(runtime, K.REQUEST_FIRST_TOKEN, payload)
            self._emit(runtime, K.REQUEST_OUTPUT_CHUNK, payload)
            if proposal.token_id == self._loaded.spec.eos_token_id:
                runtime.outcome, runtime.finish_reason = "FINISHED", "EOS"
            elif len(runtime.output) == runtime.request.max_new_tokens:
                runtime.outcome, runtime.finish_reason = "FINISHED", "MAX_NEW_TOKENS"
            if runtime.outcome is not None:
                self._decided(runtime)
            return TokenChunk(runtime.request.request_id, position, proposal.token_id, delta, ready,
                              runtime.kv_valid_tokens, len(runtime.output))

    def _quarantine_runtime(self, runtime: _RequestRuntime, reason: str) -> None:
        with self._state_lock:
            self._unhealthy = True
            if runtime.outcome is None:
                runtime.outcome, runtime.error_code = "FAILED", ErrorCode.ENGINE_UNHEALTHY.value
                if runtime.accepted_ns:
                    self._decided(runtime)
            if all(item is not runtime for item in self._quarantine):
                self._quarantine.append(runtime)
            runtime.capture_broken = True
            runtime.capture_error = runtime.capture_error or reason
            self._incomplete(runtime, reason)
            self._terminals[runtime.request.request_id] = (runtime.outcome or "FAILED", False, runtime.cancellation_id)

    def _finish(self, runtime: _RequestRuntime) -> TerminalChunk:
        if runtime.abandoned:
            self._quarantine_runtime(runtime, "ABANDONED_WITH_UNRESOLVED_OWNERSHIP")
            raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "request abandoned; resources remain quarantined")
        try:
            if runtime.ticket is not None:
                runtime.executor.abort(runtime.ticket)
                runtime.ticket = None
                self._resolve_write(runtime, committed=False)
            runtime.executor.close()
            # A launch failure can have been fenced and aborted inside the
            # executor before returning a ticket. Only close success proves it.
            if runtime.open_write is not None:
                self._resolve_write(runtime, committed=False)
            status = asdict(runtime.executor.resource_status)
            if any(status.values()):
                raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "executor did not return to its resource baseline")
            runtime.resources_reclaimed = True
        except Exception as exc:
            self._quarantine_runtime(runtime, "FENCE_OR_RESOURCE_STATE_UNKNOWN")
            raise IllmError(ErrorCode.ENGINE_UNHEALTHY, "cannot prove safe resource reclamation; engine is fail-stopped") from exc
        with self._state_lock:
            if runtime.allocation_announced:
                self._emit(runtime, K.KV_PAGE_RELEASED,
                           {"allocation_id": "kv-0", "released_bytes": self.memory_plan.kv_bytes,
                            "fence_completed": True, "active_leases": 0, "pending_writes": 0})
            if runtime.outcome is None:
                raise IllmError(ErrorCode.INTERNAL, "accepted request has no logical terminal winner")
            final_delta = runtime.detokenizer.finish()
            self._phase(runtime, runtime.outcome)
            terminal_ready_ns = perf_counter_ns()
            terminal_payload: dict[str, Any] = {"committed_output_count": len(runtime.output),
                                               "kv_valid_token_count": runtime.kv_valid_tokens,
                                               "server_terminal_ready_ns": terminal_ready_ns,
                                               "active_leases": 0, "pending_writes": 0,
                                               "physical_cleanup_complete": True}
            kind = {"FINISHED": K.REQUEST_COMPLETED, "CANCELLED": K.REQUEST_CANCELLED, "FAILED": K.REQUEST_FAILED}[runtime.outcome]
            if runtime.outcome == "FINISHED":
                terminal_payload["finish_reason"] = runtime.finish_reason
            elif runtime.outcome == "CANCELLED":
                terminal_payload["cancellation_id"] = runtime.cancellation_id
            else:
                terminal_payload["error_code"] = runtime.error_code or ErrorCode.INTERNAL.value
            self._emit(runtime, kind, terminal_payload)
            # These server facts precede capture finalization/client delivery.
            # A later client interrupt cannot un-complete an already fenced
            # inference or erase its legitimate server latency observation.
            self._terminals[runtime.request.request_id] = (runtime.outcome, True, runtime.cancellation_id)
            self._observations.append(RequestObservation(runtime.request.request_id, runtime.accepted_ns,
                                                         tuple(runtime.token_ready_ns), terminal_ready_ns,
                                                         runtime.outcome, None, None, None, "host.perf_counter_ns", None))
            runtime.terminal_ready = True
            manifest_digest = None
            if runtime.session is not None and not runtime.capture_broken:
                try:
                    manifest_digest = str(runtime.session.finalize()["manifest_digest"])
                except Exception:
                    runtime.capture_broken = True
                    runtime.capture_error = "CAPTURE_FINALIZATION_FAILED"
                    self._incomplete(runtime, runtime.capture_error)
            elif runtime.session is not None:
                self._incomplete(runtime, runtime.capture_error or "CAPTURE_INCOMPLETE")
            result = GenerationResult(
                runtime.request.request_id, runtime.input_ids, runtime.output, runtime.detokenizer.text,
                runtime.outcome, runtime.finish_reason, runtime.error_code, runtime.request.sampling.actual_seed,
                self.model_spec_id, runtime.request.sampling.digest, self.kv_mode,
                runtime.sampling.next_output_position, runtime.sampling.rng_draws_consumed,
                tuple((address.absolute_output_pos, address.draw_kind, address.draw_index)
                      for address in runtime.sampling.committed_draw_addresses),
                tuple(runtime.query_tokens), tuple(runtime.valid_counts),
                {"active_requests": 0, "reservations": 0, "quarantined_requests": 0, **status},
                {"clock_domain": "host.perf_counter_ns", "accepted_ns": runtime.accepted_ns,
                 "token_ready_ns": list(runtime.token_ready_ns), "terminal_ready_ns": terminal_ready_ns,
                 "tokenization_wall_time_ns": runtime.tokenization_ended_ns - runtime.tokenization_started_ns,
                 "per_forward_wall_time_ns": list(runtime.forward_wall_ns),
                 "server_e2e_wall_time_ns": terminal_ready_ns - runtime.accepted_ns,
                 "critical_capture_event_drop_count": runtime.capture_critical_drop_count,
                 "telemetry": runtime.request.telemetry, "evidence_kind": "measured"},
                manifest_digest, runtime.capture_error, len(runtime.events), self.numerical_policy,
                self._execution_binding, self.memory_plan.digest if self.numerical_policy is not None else None,
            )
            runtime.result = result
            self._last_result = result
            self._last_events = tuple(runtime.events)
            return TerminalChunk(runtime.request.request_id, runtime.outcome, runtime.finish_reason,
                                 runtime.error_code, final_delta, terminal_ready_ns, result)

    def _release_gate(self, runtime: _RequestRuntime) -> None:
        with self._state_lock:
            if not runtime.gate_released:
                self._last_events = tuple(runtime.events)
                self._active = None
                runtime.gate_released = True
                self._gate.release()

    def stream(self, request: GenerationRequest) -> Generator[StreamChunk, None, None]:
        runtime = self._begin(request)
        try:
            yield MetadataChunk(request.request_id, self.model_spec_id, request.sampling.digest,
                                self.memory_plan.digest, len(runtime.input_ids), self.kv_mode, runtime.accepted_ns,
                                self.numerical_policy, self._execution_binding)
            while runtime.outcome is None:
                try:
                    chunk = self._step(runtime)
                except Exception as exc:
                    error = _error(exc)
                    with self._state_lock:
                        if runtime.outcome is None:
                            runtime.outcome, runtime.error_code = "FAILED", error.code.value
                            self._decided(runtime)
                    break
                if chunk is not None:
                    yield chunk
            terminal = self._finish(runtime)
            self._release_gate(runtime)
            yield terminal
        except EmergencyAbandon:
            runtime.capture_error = "CLI_SECOND_INTERRUPT"
            if runtime.resources_reclaimed:
                # Delivery can fail after safe physical completion. Preserve a
                # completed immutable server capture; an unfinished one already
                # has INCOMPLETE. Do not invent GPU quarantine for zero leases.
                if not runtime.terminal_ready:
                    self._unhealthy = True
                self._incomplete(runtime, "CLIENT_INTERRUPTED_AFTER_RESOURCE_RECLAIM")
            else:
                runtime.abandoned = True
                self._quarantine_runtime(runtime, "CLI_SECOND_INTERRUPT")
            raise
        except GeneratorExit:
            if not runtime.terminal_ready:
                self.cancel(request.request_id, "stream-close")
                self._finish(runtime)
            raise
        except BaseException:
            if runtime.resources_reclaimed and not runtime.terminal_ready:
                self._unhealthy = True
                self._incomplete(runtime, "SERVER_TERMINAL_RECORDING_INTERRUPTED")
            elif not runtime.terminal_ready and not runtime.abandoned and not self._unhealthy:
                with self._state_lock:
                    if runtime.outcome is None:
                        runtime.outcome, runtime.error_code = "FAILED", ErrorCode.INTERNAL.value
                        self._decided(runtime)
                self._finish(runtime)
            elif runtime.abandoned:
                self._quarantine_runtime(runtime, "ABANDONED_WITH_UNRESOLVED_OWNERSHIP")
            raise
        finally:
            self._release_gate(runtime)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        result: GenerationResult | None = None
        for chunk in self.stream(request):
            if isinstance(chunk, TerminalChunk):
                result = chunk.result
        if result is None:
            raise IllmError(ErrorCode.INTERNAL, "stream ended without its terminal result")
        return result

    def close(self, timeout: float = 1.0) -> bool:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "close timeout must be finite and nonnegative")
        if not self._gate.acquire(blocking=False):
            raise IllmError(ErrorCode.OVERLOADED, "close requires terminal completion or explicit abandonment",
                            {"reason": "SINGLE_ACTIVE_LIMIT"})
        try:
            self._closed = True
            return self._bus.close(timeout) and not self._quarantine
        finally:
            self._gate.release()

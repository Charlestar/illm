"""One active request, frozen eager weights, full context recomputed per token.

Required capture is synchronous. Optional exporters receive bounded diagnostic
copies only; blocked/raising exporters cannot decide generation outcomes.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
import struct
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.reference import ReferenceCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent, EventKind
from illm.domain.reference_events import ReferenceEvent, ReferenceEventKind as K
from illm.engine.contracts import GenerationRequest, GenerationResult
from illm.model.loader import LoadedModel, load_model
from illm.model.spec import ModelError
from illm.observability.bus import EventBus, EventSink
from illm.sampling import SamplingError, SamplingState, commit_sample, prepare_sample

if TYPE_CHECKING:
    from torch import Tensor
    from illm.model.reference import OpRecord


def as_illm_error(exc: Exception) -> IllmError:
    """Map adapter failures without exposing raw request text in diagnostics."""
    if isinstance(exc, IllmError):
        return exc
    if isinstance(exc, (ModelError, SamplingError)):
        try:
            code = ErrorCode(exc.code)
        except ValueError:
            code = ErrorCode.INTERNAL
        field = exc.field if isinstance(exc, ModelError) else None
        return IllmError(code, str(exc), {"field": field} if field else None)
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError" or (
        type(exc).__name__ == "AcceleratorError" and "out of memory" in str(exc).lower()
    ):
        return IllmError(ErrorCode.RESOURCE_EXHAUSTED, "reference execution exhausted memory; no retry performed")
    if isinstance(exc, ModuleNotFoundError):
        return IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "selected locked tensor runtime is not installed")
    return IllmError(ErrorCode.INTERNAL, "reference execution failed", {"exception_type": type(exc).__name__})


class Engine:
    def __init__(self, model: str | Path = "fixture:tiny-decoder", *, device: str = "cpu",
                 project_root: Path | None = None, exporters: Sequence[EventSink] = (),
                 exporter_capacity: int = 256) -> None:
        if not isinstance(model, (str, Path)) or not str(model):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "model must be a nonempty local profile or fixture identifier")
        if type(exporter_capacity) is not int or exporter_capacity < 1:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "exporter_capacity must be a positive integer")
        if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "device must be cpu, cuda or cuda:<index>")
        try:
            if device.startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    raise IllmError(ErrorCode.UNSUPPORTED_MODEL, "selected CUDA device is unavailable in this runtime")
            self._loaded: LoadedModel = load_model(model, device=device, project_root=project_root)
        except Exception as exc:
            raise as_illm_error(exc) from exc
        self.model = str(model)
        self.device = str(self._loaded.reference.canonical_weight("model.embed_tokens.weight").device)
        self._gate = Lock()
        self._closed = False
        self._bus = EventBus((), tuple(exporters), best_effort_capacity=exporter_capacity)
        self._last_events: tuple[ReferenceEvent, ...] = ()

    @property
    def last_events(self) -> tuple[ReferenceEvent, ...]:
        return self._last_events

    @property
    def model_spec_id(self) -> str:
        return self._loaded.spec.spec_id

    @property
    def exporter_status(self) -> dict[str, int]:
        return {"error_count": self._bus.exporter_error_count, "copy_drop_count": self._bus.exporter_drop_count,
                "queue_high_watermark": self._bus.exporter_queue_high_watermark}

    def close(self, timeout: float = 1.0) -> bool:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "close timeout must be finite and nonnegative")
        if not self._gate.acquire(blocking=False):
            raise IllmError(ErrorCode.ENGINE_BUSY, "cannot close an engine with an active request")
        try:
            self._closed = True
            return self._bus.close(timeout)
        finally:
            self._gate.release()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "generate requires GenerationRequest")
        if not self._gate.acquire(blocking=False):
            raise IllmError(ErrorCode.ENGINE_BUSY, "V0.1 permits one active request")
        try:
            self._last_events = ()
            if self._closed:
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "engine is closed")
            if str(request.model) != self.model:
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "request model differs from the engine's loaded model")
            return self._generate(request)
        finally:
            self._gate.release()

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        import torch

        started = perf_counter_ns()
        loaded = self._loaded
        tokenizer_spec = loaded.tokenizer.to_dict()
        identity = {
            "model_spec_id": loaded.spec.spec_id, "weights_digest": str(loaded.weight_manifest["weights_digest"]),
            "load_manifest_digest": str(loaded.weight_manifest["manifest_digest"]),
            "tokenizer_digest": sha256_json(tokenizer_spec),
            "input_spec_digest": sha256_json({"schema_version": 1, "tokenizer": tokenizer_spec,
                                               "input_kind": "text" if request.prompt is not None else "token_ids",
                                               "add_bos": request.add_bos, "template": "raw-text-no-chat-v1"}),
            "runtime_dtype": loaded.spec.runtime_dtype, "device": self.device, "backend": "illm_eager",
        }
        session = None
        if request.capture is not None:
            if request.capture.exists() and not request.capture.is_dir():
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "capture output must be a directory")
            try:
                session = ReferenceCaptureSession(request.capture, identity, request.request_id, request.sampling.to_dict())
            except OSError as exc:
                raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "required capture could not be initialized") from exc
        events: list[ReferenceEvent] = []
        terminal = False
        accepted = False
        # Destroyed with the call. Digest is useful within this capture, but is
        # neither a public prompt hash nor a cross-capture correlation handle.
        privacy_key = secrets.token_bytes(32)

        def private_digest(label: str, raw: bytes) -> str:
            return hmac.new(privacy_key, label.encode("ascii") + b"\0" + raw, hashlib.sha256).hexdigest()

        def emit(kind: K, payload: dict[str, Any]) -> None:
            nonlocal terminal, accepted
            if session is not None:
                event = session.emit(kind, payload)
            else:
                event = ReferenceEvent("reference-engine", len(events), perf_counter_ns(), kind,
                                       {"request_id": request.request_id, **payload})
            events.append(event)
            accepted |= kind is K.REQUEST_ACCEPTED
            terminal |= kind in {K.REQUEST_FINISHED, K.REQUEST_FAILED}
            if request.telemetry != "off":
                self._bus.emit(DomainEvent("reference-export-copy", event.seq_no, event.monotonic_ns,
                                           EventKind.DIAGNOSTIC, {"reference_event": event.to_dict()}))

        def synchronize() -> None:
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)

        try:
            emit(K.REQUEST_ACCEPTED, {"max_new_tokens": request.max_new_tokens})
            tokenization_started = perf_counter_ns()
            if request.prompt is not None:
                input_ids = loaded.tokenizer.encode(request.prompt, add_bos=request.add_bos)
            else:
                input_ids = request.input_token_ids or ()
                if request.add_bos is True:
                    input_ids = (loaded.spec.bos_token_id, *input_ids)
                if not input_ids:
                    raise IllmError(ErrorCode.INVALID_ARGUMENT, "empty token-ID input requires add_bos=True")
            if any(token >= loaded.spec.vocab_size for token in input_ids):
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "input token is outside vocabulary")
            if len(input_ids) + request.max_new_tokens - 1 > loaded.max_context_tokens:
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "requested full-context forwards exceed the model profile limit")
            tokenization_ns = perf_counter_ns() - tokenization_started
            emit(K.REQUEST_TOKENIZED, {"input_tokens": len(input_ids), "input_digest": private_digest("input", canonical_json_bytes(input_ids))})
            state = SamplingState.for_request(request.request_id, request.sampling)
            output: tuple[int, ...] = ()
            forward_times: list[int] = []
            sampling_times: list[int] = []
            host_export_times: list[int] = []
            finish_reason = "MAX_NEW_TOKENS"
            for position in range(request.max_new_tokens):
                context = (*input_ids, *output)
                emit(K.MODEL_FORWARD_STARTED, {"output_position": position, "context_tokens": len(context)})

                def hook(record: OpRecord, tensor: Tensor) -> None:
                    del tensor  # shape hooks never copy/persist tensor contents.
                    emit(K.MODEL_OP, {"output_position": position, "op_name": record.name,
                                      "shape": list(record.shape), "dtype": record.dtype})

                tensor_ids = torch.tensor([context], dtype=torch.int64, device=self.device)
                synchronize()
                forward_started = perf_counter_ns()
                logits = loaded.reference.forward(tensor_ids, op_hook=hook if request.telemetry == "ops" else None)
                synchronize()
                forward_times.append(perf_counter_ns() - forward_started)
                if tuple(logits.shape) != (1, len(context), loaded.spec.vocab_size):
                    raise IllmError(ErrorCode.INVALID_WEIGHTS, "reference returned an invalid logits shape")
                host_export_started = perf_counter_ns()
                raw_logits = cast(list[float], logits[0, -1].detach().float().cpu().tolist())
                host_export_times.append(perf_counter_ns() - host_export_started)
                logits_digest = private_digest("logits", struct.pack(f"<{len(raw_logits)}f", *raw_logits))
                emit(K.MODEL_FORWARD_COMPLETED, {"output_position": position, "logits_digest": logits_digest})
                emit(K.SAMPLING_STARTED, {"output_position": position, "sampling_spec_digest": request.sampling.digest})
                sampling_started = perf_counter_ns()
                try:
                    proposal = prepare_sample(raw_logits, request.sampling, state)
                    next_state = commit_sample(state, proposal, spec=request.sampling, logits=raw_logits)
                except SamplingError as exc:
                    emit(K.SAMPLING_FAILED, {"output_position": position, "error_code": exc.code})
                    raise
                next_output = (*output, proposal.token_id)
                draw_address = [position, 0, 0] if proposal.draw is not None else None
                emit(K.SAMPLING_SELECTED, {"output_position": position, "sampling_spec_digest": request.sampling.digest,
                                          "logits_digest": logits_digest, "draw_address": draw_address,
                                          "selected_digest": private_digest("selected", canonical_json_bytes([position, proposal.token_id]))})
                state, output = next_state, next_output
                sampling_times.append(perf_counter_ns() - sampling_started)
                if proposal.token_id == loaded.spec.eos_token_id:
                    finish_reason = "EOS"
                    break
            text = loaded.tokenizer.decode(output)
            emit(K.REQUEST_FINISHED, {"output_tokens": len(output), "finish_reason": finish_reason,
                                      "output_digest": private_digest("output", canonical_json_bytes(output))})
            manifest = session.finalize() if session is not None else None
            timings: dict[str, Any] = {
                "clock": "perf_counter_ns", "evidence_kind": "measured", "telemetry": request.telemetry,
                "synchronization": "cuda_device_boundary" if self.device.startswith("cuda") else "synchronous_cpu",
                "tokenization_wall_time_ns": tokenization_ns,
                "server_full_context_forward_wall_time_ns": sum(forward_times),
                "per_forward_wall_time_ns": forward_times, "sampling_wall_time_ns": sum(sampling_times),
                "logits_host_export_wall_time_ns": sum(host_export_times),
                "total_request_wall_time_ns": perf_counter_ns() - started,
            }
            return GenerationResult(request.request_id, input_ids, output, text, finish_reason,
                                    request.sampling.actual_seed, loaded.spec.spec_id, request.sampling.digest,
                                    state.next_output_position, state.rng_draws_consumed,
                                    tuple((address.absolute_output_pos, address.draw_kind, address.draw_index)
                                          for address in state.committed_draw_addresses),
                                    timings, str(manifest["manifest_digest"]) if manifest else None, len(events))
        except Exception as exc:
            error = as_illm_error(exc)
            if accepted and not terminal:
                try:
                    emit(K.REQUEST_FAILED, {"error_code": error.code.value})
                    if session is not None:
                        session.finalize()
                except Exception as capture_exc:
                    raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "request failed and required capture could not be finalized",
                                    {"request_error": error.code.value, "capture_error": type(capture_exc).__name__}) from capture_exc
            elif session is not None and terminal:
                raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "terminal outcome exists but required capture finalization failed") from exc
            raise error from exc
        finally:
            self._last_events = tuple(events)

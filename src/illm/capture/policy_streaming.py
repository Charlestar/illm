"""Schema-3 policy-bound R0 streaming captures.

This sibling format deliberately leaves the historical schema-2 reader and
writer unchanged. It serializes metadata only, imports no tensor runtime, and
binds every capture to its policy, plan, protocol, source, and environment.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import _Playback
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.policy_events import POLICY_EVENT_CATALOG, PolicyStreamingEvent
from illm.domain.streaming_events import (
    EVENT_ENVELOPE_VERSION,
    REQUIRED_PAYLOAD_FIELDS,
    StreamingEvent,
    StreamingEventKind as K,
    thaw_metadata,
)
from illm.policy_capture_contract import validate_policy_capture_metadata


__all__ = ["PolicyStreamingCaptureBundle", "PolicyStreamingCaptureSession"]

FACT_FILES = ("events.jsonl",)
CHECKSUM_FILE = "checksums.json"
MANIFEST_FILE = "capture-manifest.json"
COMPLETE_FILE = "COMPLETE"
INCOMPLETE_FILE = "INCOMPLETE"
CAPABILITIES = ["R0"]
POLICY_METADATA_FIELDS = {
    "identity", "sampling_spec", "sampling_spec_digest", "memory_plan",
    "memory_plan_digest", "numerical_policy", "numerical_policy_digest",
    "execution_binding", "execution_binding_digest", "protocol_sha256",
    "observation_profile",
}
MANIFEST_FIELDS = {
    "schema_version", "finalized", "event_envelope_version", "event_catalog",
    "capabilities", "privacy_mode", "replayable", "non_replayable_reasons",
    "critical_event_drop_count", "request_id", *POLICY_METADATA_FIELDS,
    "event_count", "logical_terminal_outcome", "terminal_outcome", "files",
    "manifest_digest",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PRIVATE_KEYS = frozenset({
    "prompt", "prompt_text", "output", "output_text", "text", "text_delta",
    "token_id", "token_ids", "input_ids", "input_token_ids", "output_ids",
    "output_token_ids", "logits", "tensor", "tensors", "key", "keys",
    "key_bytes", "hmac_key", "token_hmac_key", "kv_key",
})
_KV_KINDS = frozenset({
    K.KV_ALLOCATION_COMMITTED,
    K.KV_WRITE_STARTED,
    K.KV_WRITE_COMMITTED,
    K.KV_WRITE_ABORTED,
    K.KV_PAGE_RELEASED,
})
_FORWARD_FIELDS = ("phase", "query_tokens", "context_tokens", "output_position")
_POLICY_OPTIONAL_PAYLOAD_FIELDS: dict[K, frozenset[str]] = {
    K.REQUEST_FIRST_TOKEN: frozenset({
        "token_hmac", "next_output_position", "rng_draws_consumed", "draw_address",
    }),
    K.REQUEST_OUTPUT_CHUNK: frozenset({
        "token_hmac", "next_output_position", "rng_draws_consumed", "draw_address",
    }),
    K.REQUEST_COMPLETED: frozenset({
        "server_terminal_ready_ns", "active_leases", "pending_writes",
        "physical_cleanup_complete",
    }),
    K.REQUEST_CANCELLED: frozenset({
        "server_terminal_ready_ns", "active_leases", "pending_writes",
        "physical_cleanup_complete",
    }),
    K.REQUEST_FAILED: frozenset({
        "server_terminal_ready_ns", "active_leases", "pending_writes",
        "physical_cleanup_complete",
    }),
    K.REQUEST_TERMINAL_DECIDED: frozenset({"reason", "cancellation_id", "error_code"}),
    K.MODEL_FORWARD_COMPLETED: frozenset({"duration_ns"}),
    K.MODEL_FORWARD: frozenset({"duration_ns", "evidence_kind"}),
    K.KV_WRITE_STARTED: frozenset({"measurement_boundary"}),
    K.KV_PAGE_RELEASED: frozenset({"active_leases", "pending_writes"}),
    # Engine._emit attaches request_id before calling the capture session, while
    # direct portable writers may omit it for a clock-only event.
    K.CLOCK_SNAPSHOT: frozenset({"request_id"}),
}


def _fail(message: str, code: ErrorCode = ErrorCode.CAPTURE_CORRUPT,
          details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _plain_absolute(path: Path, *, argument: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if _is_linklike(candidate):
            _fail(
                "policy capture path contains a symbolic link or reparse point",
                ErrorCode.INVALID_ARGUMENT if argument else ErrorCode.CAPTURE_CORRUPT,
                {"path": str(candidate)},
            )
    return absolute


def _privacy_check(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _fail("policy capture metadata keys must be nonempty strings", details={"path": path})
            if key.lower() in _FORBIDDEN_PRIVATE_KEYS:
                _fail(
                    "metadata-only policy capture forbids prompt/text/token/logit/tensor/key contents",
                    details={"field": f"{path}.{key}"},
                )
            _privacy_check(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _privacy_check(item, path=f"{path}[{index}]")


def _detached_json(value: Any, *, name: str) -> Any:
    _privacy_check(value, path=name)
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT,
                        f"{name} must contain finite JSON metadata") from exc


def _validated_metadata(
    *, identity: Mapping[str, Any], sampling_spec: Mapping[str, Any],
    memory_plan: Mapping[str, Any], numerical_policy: Mapping[str, Any],
    execution_binding: Mapping[str, Any], observation_profile: str, argument: bool,
) -> dict[str, Any]:
    try:
        result = validate_policy_capture_metadata(
            identity=identity,
            sampling_spec=sampling_spec,
            memory_plan=memory_plan,
            numerical_policy=numerical_policy,
            execution_binding=execution_binding,
            observation_profile=observation_profile,
        )
    except IllmError:
        raise
    except Exception as exc:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT if argument else ErrorCode.CAPTURE_CORRUPT,
            "policy capture metadata contract is invalid",
        ) from exc
    if set(result) != POLICY_METADATA_FIELDS:
        _fail(
            "policy capture metadata contract returned an unexpected field set",
            ErrorCode.INVALID_ARGUMENT if argument else ErrorCode.CAPTURE_CORRUPT,
        )
    return cast(dict[str, Any], _detached_json(result, name="policy_capture_metadata"))


def _validate_policy_payload(event: StreamingEvent) -> None:
    allowed = REQUIRED_PAYLOAD_FIELDS[event.kind] | _POLICY_OPTIONAL_PAYLOAD_FIELDS.get(
        event.kind, frozenset()
    )
    unexpected = set(event.payload) - allowed
    if unexpected:
        _fail(
            "policy event payload contains fields outside its closed metadata contract",
            details={"kind": event.kind.wire_name, "fields": sorted(unexpected)},
        )
    token_hmac = event.payload.get("token_hmac")
    if token_hmac is not None and (
        not isinstance(token_hmac, str) or _DIGEST.fullmatch(token_hmac) is None
    ):
        _fail("policy event token_hmac must be a SHA-256 hex digest")
    evidence_kind = event.payload.get("evidence_kind")
    if (
        event.kind is K.MODEL_FORWARD
        and evidence_kind is not None
        and (type(evidence_kind) is not str or evidence_kind != "measured")
    ):
        _fail("policy model.forward evidence kind is unsupported")
    measurement_boundary = event.payload.get("measurement_boundary")
    if (
        event.kind is K.KV_WRITE_STARTED
        and measurement_boundary is not None
        and (
            type(measurement_boundary) is not str
            or measurement_boundary != "host_before_launch"
        )
    ):
        _fail("policy KV write measurement boundary is unsupported")
    for field_name in ("next_output_position", "rng_draws_consumed"):
        value = event.payload.get(field_name)
        if value is not None and (type(value) is not int or value < 0):
            _fail(f"policy event {field_name} must be a nonnegative integer")
    if "draw_address" in event.payload:
        address = event.payload["draw_address"]
        if address is not None and (
            not isinstance(address, (tuple, list))
            or len(address) != 3
            or any(type(item) is not int or item < 0 for item in address)
        ):
            _fail("policy event draw_address must be null or three nonnegative integers")
    if ("physical_cleanup_complete" in event.payload
            and event.payload["physical_cleanup_complete"] is not True):
        _fail("policy terminal cleanup flag must be true when recorded")
    for field_name in ("active_leases", "pending_writes"):
        if field_name in event.payload and event.payload[field_name] != 0:
            _fail(f"policy cleanup event must report {field_name}=0")


def _validate_policy_output_rng(
    payload: Mapping[str, Any], sampling_spec: Mapping[str, Any],
) -> None:
    committed = cast(int, payload["committed_output_count"])
    position = cast(int, payload["output_index"])
    if ("next_output_position" in payload
            and payload["next_output_position"] != committed):
        _fail("policy output next position disagrees with committed output count")
    mode = sampling_spec["mode"]
    if "rng_draws_consumed" in payload:
        expected_draws = 0 if mode == "greedy" else committed
        if payload["rng_draws_consumed"] != expected_draws:
            _fail("policy output RNG draw count disagrees with SamplingSpec")
    if "draw_address" in payload:
        expected_address: list[int] | None = None if mode == "greedy" else [position, 0, 0]
        actual_address = payload["draw_address"]
        if actual_address is not None:
            actual_address = list(cast(tuple[int, ...], actual_address))
        if actual_address != expected_address:
            _fail("policy output draw address disagrees with frozen RNG contract v1")


def _forward_signature(event: StreamingEvent) -> tuple[str, int, int, int]:
    payload = event.payload
    return (
        cast(str, payload["phase"]),
        cast(int, payload["query_tokens"]),
        cast(int, payload["context_tokens"]),
        cast(int, payload["output_position"]),
    )


def _validate_forward_shape(
    event: StreamingEvent, state: _Playback, *, kv_mode: str,
) -> tuple[str, int, int, int]:
    signature = _forward_signature(event)
    _, query_tokens, context_tokens, output_position = signature
    expected_position = state.committed_output_count
    expected_context = state.input_tokens + expected_position
    expected_query = expected_context if kv_mode == "none" or expected_position == 0 else 1
    if (
        output_position != expected_position
        or context_tokens != expected_context
        or query_tokens != expected_query
    ):
        _fail(
            "policy model forward coverage disagrees with accepted/committed request facts",
            details={
                "kind": event.kind.wire_name,
                "expected_output_position": expected_position,
                "actual_output_position": output_position,
                "expected_context_tokens": expected_context,
                "actual_context_tokens": context_tokens,
                "expected_query_tokens": expected_query,
                "actual_query_tokens": query_tokens,
                "kv_mode": kv_mode,
            },
        )
    return signature


def _play_policy(
    events: tuple[StreamingEvent, ...], request_id: str, *,
    metadata: Mapping[str, Any], complete: bool,
) -> _Playback:
    """Replay schema3 facts and bind them to its frozen plan.

    Historical schema2 continues to use ``_play`` directly.  This validator is
    deliberately sibling-only so adding policy/plan facts cannot reinterpret
    old bytes.
    """

    memory_plan = cast(Mapping[str, Any], metadata["memory_plan"])
    sampling_spec = cast(Mapping[str, Any], metadata["sampling_spec"])
    kv_mode = cast(str, memory_plan["kv_mode"])
    capacity_tokens = cast(int, memory_plan["capacity_tokens"])
    kv_bytes = cast(int, memory_plan["kv_bytes"])
    state = _Playback(request_id)
    allocation_count = 0
    open_forward: tuple[str, int, int, int] | None = None
    open_forward_has_write = False
    recent_completed: tuple[str, int, int, int] | None = None
    started_positions: set[int] = set()
    completed_positions: set[int] = set()
    first_token_payload: dict[str, Any] | None = None

    for index, event in enumerate(events):
        _validate_policy_payload(event)
        kind = event.kind
        payload = event.payload
        if kv_mode == "none" and kind in _KV_KINDS:
            _fail("no-KV policy capture contains a KV allocation/write/release event")

        if kind is K.REQUEST_ACCEPTED:
            requested = cast(int, payload["input_tokens"]) + cast(int, payload["max_new_tokens"])
            if requested > capacity_tokens:
                _fail(
                    "accepted request exceeds its frozen MemoryPlan capacity",
                    details={"requested_tokens": requested, "capacity_tokens": capacity_tokens},
                )
        elif kind is K.KV_ALLOCATION_COMMITTED:
            allocation_count += 1
            if (
                kv_mode != "contiguous"
                or allocation_count != 1
                or payload["capacity_tokens"] != capacity_tokens
                or payload["allocated_bytes"] != kv_bytes
            ):
                _fail(
                    "KV allocation facts disagree with the frozen contiguous MemoryPlan",
                    details={
                        "plan_capacity_tokens": capacity_tokens,
                        "event_capacity_tokens": payload["capacity_tokens"],
                        "plan_kv_bytes": kv_bytes,
                        "event_allocated_bytes": payload["allocated_bytes"],
                    },
                )
        elif kind is K.MODEL_FORWARD_STARTED:
            if not state.accepted:
                _fail("policy model forward precedes request acceptance")
            if open_forward is not None:
                _fail("policy capture has overlapping model forward starts")
            if kv_mode == "contiguous" and allocation_count != 1:
                _fail("contiguous model forward precedes its MemoryPlan-bound KV allocation")
            signature = _validate_forward_shape(event, state, kv_mode=kv_mode)
            position = signature[3]
            if position in started_positions:
                _fail("policy capture repeats a model forward output position")
            started_positions.add(position)
            open_forward = signature
            open_forward_has_write = False
            recent_completed = None
        elif kind is K.KV_WRITE_STARTED:
            if kv_mode != "contiguous" or open_forward is None or open_forward_has_write:
                _fail("policy KV write is not owned by exactly one active contiguous forward")
            _, query_tokens, context_tokens, _ = open_forward
            if (
                payload["token_count"] != query_tokens
                or payload["start_token"] + payload["token_count"] != context_tokens
            ):
                _fail("policy KV append coverage disagrees with its active model forward")
            open_forward_has_write = True
        elif kind is K.MODEL_FORWARD_COMPLETED:
            signature = _validate_forward_shape(event, state, kv_mode=kv_mode)
            if open_forward is None or signature != open_forward:
                _fail("policy model forward completion does not match its active start")
            if kv_mode == "contiguous" and not open_forward_has_write:
                _fail("contiguous model forward completed without its KV append lease")
            completed_positions.add(signature[3])
            recent_completed = signature
            open_forward = None
            open_forward_has_write = False
        elif kind is K.MODEL_FORWARD:
            signature = _validate_forward_shape(event, state, kv_mode=kv_mode)
            if recent_completed is None or signature != recent_completed:
                _fail("policy model.forward summary does not match the preceding completion")
            recent_completed = None
        elif kind is K.REQUEST_OUTPUT_CHUNK:
            if payload["output_index"] not in completed_positions:
                _fail("committed output lacks a completed model forward at the same position")
            _validate_policy_output_rng(payload, sampling_spec)
            if payload["output_index"] == 0 and first_token_payload != thaw_metadata(payload):
                _fail("first-token and output-zero metadata facts disagree")
        elif kind is K.REQUEST_FIRST_TOKEN:
            _validate_policy_output_rng(payload, sampling_spec)
            first_token_payload = cast(dict[str, Any], thaw_metadata(payload))

        state.apply(event, index)

    if complete:
        state.validate_complete()
        if kv_mode == "contiguous" and allocation_count != 1:
            _fail("complete contiguous capture lacks exactly one MemoryPlan-bound KV allocation")
        required_positions = set(range(state.committed_output_count))
        if not required_positions.issubset(completed_positions):
            _fail("complete policy capture omits a model forward for committed output")
        if state.terminal_outcome == "FINISHED" and open_forward is not None:
            _fail("finished policy capture ends with an unresolved model forward")
    return state


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "policy capture refuses to overwrite a foreign artifact",
            {"artifact": path.name},
        ) from exc


def _decode_json(text: str, *, artifact: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in policy capture",
                      details={"artifact": artifact, "field": key})
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail("non-finite JSON constant in policy capture",
              details={"artifact": artifact, "value": value})

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except IllmError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy capture JSON is malformed",
                        {"artifact": artifact}) from exc


def _read_json(path: Path) -> Any:
    try:
        return _decode_json(path.read_text(encoding="utf-8"), artifact=path.name)
    except IllmError:
        raise
    except (OSError, UnicodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy capture JSON is unreadable",
                        {"artifact": path.name}) from exc


def _assert_session_inventory(root: Path) -> None:
    allowed = {INCOMPLETE_FILE, *FACT_FILES, CHECKSUM_FILE, MANIFEST_FILE}
    try:
        children = list(root.iterdir())
    except OSError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "policy capture root is unreadable") from exc
    names = {child.name for child in children}
    if INCOMPLETE_FILE not in names or names - allowed:
        _fail("policy capture writer found foreign or terminal artifacts",
              ErrorCode.INVALID_ARGUMENT, {"artifacts": sorted(names)})
    for child in children:
        if _is_linklike(child) or not child.is_file():
            _fail("policy capture writer found a linked or non-file artifact",
                  ErrorCode.INVALID_ARGUMENT, {"artifact": child.name})


def _manifest(
    *, finalized: bool, replayable: bool, reasons: list[str], request_id: str,
    metadata: Mapping[str, Any], events: tuple[PolicyStreamingEvent, ...],
    logical_terminal_outcome: str | None, terminal_outcome: str | None,
    files: Mapping[str, str], critical_event_drop_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 3,
        "finalized": finalized,
        "event_envelope_version": EVENT_ENVELOPE_VERSION,
        "event_catalog": POLICY_EVENT_CATALOG,
        "capabilities": list(CAPABILITIES),
        "privacy_mode": "metadata_only",
        "replayable": replayable,
        "non_replayable_reasons": list(reasons),
        "critical_event_drop_count": critical_event_drop_count,
        "request_id": request_id,
        **cast(dict[str, Any], _detached_json(metadata, name="policy_capture_metadata")),
        "event_count": len(events),
        "logical_terminal_outcome": logical_terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "files": dict(files),
    }
    result["manifest_digest"] = sha256_json(result)
    return result


def _capture_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    binding = cast(Mapping[str, Any], manifest["execution_binding"])
    return {
        "numerical_policy_digest": manifest["numerical_policy_digest"],
        "memory_plan_digest": manifest["memory_plan_digest"],
        "execution_binding_digest": manifest["execution_binding_digest"],
        "implementation_source_digest": binding["implementation_source_digest"],
        "environment_digest": binding["environment_digest"],
        "protocol_sha256": manifest["protocol_sha256"],
        "observation_profile": manifest["observation_profile"],
    }


@dataclass(slots=True)
class PolicyStreamingCaptureSession:
    path: Path
    identity: Mapping[str, Any]
    request_id: str
    sampling_spec: Mapping[str, Any]
    memory_plan: Mapping[str, Any]
    numerical_policy: Mapping[str, Any] = field(kw_only=True)
    execution_binding: Mapping[str, Any] = field(kw_only=True)
    observation_profile: str = field(kw_only=True)
    _events: list[PolicyStreamingEvent] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)
    _finalized: bool = field(default=False, init=False)
    _critical_event_drop_count: int = field(default=0, init=False)
    _incomplete_reasons: list[str] = field(default_factory=list, init=False)
    _capture_metadata: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        requested = _plain_absolute(Path(self.path), argument=True)
        if not isinstance(self.request_id, str) or not self.request_id:
            _fail("policy streaming request ID must be nonempty", ErrorCode.INVALID_ARGUMENT)
        self._capture_metadata = _validated_metadata(
            identity=self.identity,
            sampling_spec=self.sampling_spec,
            memory_plan=self.memory_plan,
            numerical_policy=self.numerical_policy,
            execution_binding=self.execution_binding,
            observation_profile=self.observation_profile,
            argument=True,
        )
        if requested.exists():
            if _is_linklike(requested) or not requested.is_dir():
                _fail("policy capture output must be an ordinary directory",
                      ErrorCode.INVALID_ARGUMENT)
            try:
                if any(requested.iterdir()):
                    _fail("policy capture output must be a new or empty directory",
                          ErrorCode.INVALID_ARGUMENT)
            except OSError as exc:
                raise IllmError(ErrorCode.INVALID_ARGUMENT,
                                "policy capture output is unreadable") from exc
        else:
            try:
                requested.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                raise IllmError(ErrorCode.INVALID_ARGUMENT,
                                "policy capture output could not be created") from exc
        self.path = _plain_absolute(requested, argument=True)
        _write_exclusive(self.root / INCOMPLETE_FILE,
                         b"policy_streaming_capture_not_finalized\n")

    @property
    def root(self) -> Path:
        return self.path

    def emit(self, kind: K, payload: Mapping[str, Any]) -> PolicyStreamingEvent:
        if self._closed:
            _fail("policy streaming capture session is closed", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(kind, K):
            _fail("policy streaming event kind is invalid", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(payload, Mapping):
            _fail("policy streaming event payload must be metadata", ErrorCode.INVALID_ARGUMENT)
        event_payload = dict(payload)
        if kind is not K.CLOCK_SNAPSHOT:
            event_payload.setdefault("request_id", self.request_id)
        _privacy_check(event_payload, path="payload")
        event = PolicyStreamingEvent(
            "stateful-engine", len(self._events), time.perf_counter_ns(), kind, event_payload,
        )
        pending = cast(tuple[StreamingEvent, ...], tuple((*self._events, event)))
        _play_policy(
            pending, self.request_id, metadata=self._capture_metadata, complete=False,
        )
        self._events.append(event)
        return event

    def _write(self, *, finalized: bool, replayable: bool,
               reasons: list[str]) -> dict[str, Any]:
        _assert_session_inventory(self.root)
        events = tuple(self._events)
        state = _play_policy(
            cast(tuple[StreamingEvent, ...], events),
            self.request_id,
            metadata=self._capture_metadata,
            complete=finalized,
        )
        encoded = b"".join(canonical_json_bytes(event.to_dict()) + b"\n"
                           for event in events)
        _atomic_bytes(self.root / FACT_FILES[0], encoded)
        facts = {FACT_FILES[0]: sha256_file(self.root / FACT_FILES[0])}
        _atomic_bytes(self.root / CHECKSUM_FILE, canonical_json_bytes(facts) + b"\n")
        files = {**facts, CHECKSUM_FILE: sha256_file(self.root / CHECKSUM_FILE)}
        manifest = _manifest(
            finalized=finalized,
            replayable=replayable,
            reasons=reasons,
            request_id=self.request_id,
            metadata=self._capture_metadata,
            events=events,
            logical_terminal_outcome=state.logical_terminal_outcome,
            terminal_outcome=state.terminal_outcome,
            files=files,
            critical_event_drop_count=self._critical_event_drop_count,
        )
        _atomic_bytes(self.root / MANIFEST_FILE, canonical_json_bytes(manifest) + b"\n")
        return manifest

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            _fail("policy streaming capture session is closed", ErrorCode.INVALID_ARGUMENT)
        manifest = self._write(finalized=True, replayable=True, reasons=[])
        _write_exclusive(self.root / COMPLETE_FILE,
                         (manifest["manifest_digest"] + "\n").encode("ascii"))
        (self.root / INCOMPLETE_FILE).unlink()
        self._closed = True
        self._finalized = True
        return manifest

    def mark_incomplete(self, reason: str, *,
                        critical_event_drop_count: int = 0) -> dict[str, Any]:
        if self._finalized or (self._closed and not self._incomplete_reasons):
            _fail("a complete policy capture cannot be rewritten", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(reason, str) or not reason.strip():
            _fail("incomplete policy capture reason must be nonempty",
                  ErrorCode.INVALID_ARGUMENT)
        if (type(critical_event_drop_count) is not int
                or critical_event_drop_count < self._critical_event_drop_count):
            _fail("critical policy capture loss count must be a nonnegative monotone integer",
                  ErrorCode.INVALID_ARGUMENT)
        self._critical_event_drop_count = critical_event_drop_count
        normalized_reason = reason.strip()
        if normalized_reason not in self._incomplete_reasons:
            self._incomplete_reasons.append(normalized_reason)
        manifest = self._write(
            finalized=False,
            replayable=False,
            reasons=list(self._incomplete_reasons),
        )
        _atomic_bytes(self.root / INCOMPLETE_FILE,
                      (normalized_reason + "\n").encode("utf-8"))
        self._closed = True
        return manifest


@dataclass(frozen=True, slots=True)
class PolicyStreamingCaptureBundle:
    root: Path
    manifest: dict[str, Any]
    events: tuple[PolicyStreamingEvent, ...]
    _state: _Playback

    @classmethod
    def open(cls, path: Path) -> PolicyStreamingCaptureBundle:
        root = _plain_absolute(Path(path), argument=False)
        if not root.is_dir() or _is_linklike(root):
            _fail("policy capture root must be an ordinary directory")
        incomplete = root / INCOMPLETE_FILE
        if incomplete.exists() or incomplete.is_symlink() or _is_linklike(incomplete):
            reasons: list[str] = []
            reported_drop_count: int | None = None
            manifest_path = root / MANIFEST_FILE
            if manifest_path.is_file() and not _is_linklike(manifest_path):
                value = _read_json(manifest_path)
                if isinstance(value, dict) and isinstance(
                    value.get("non_replayable_reasons"), list
                ):
                    reasons = [
                        item for item in value["non_replayable_reasons"]
                        if isinstance(item, str)
                    ]
                if (isinstance(value, dict)
                        and type(value.get("critical_event_drop_count")) is int
                        and value["critical_event_drop_count"] >= 0):
                    reported_drop_count = value["critical_event_drop_count"]
            _fail(
                "policy capture is incomplete/non-replayable",
                ErrorCode.CAPTURE_INCOMPLETE,
                {
                    "reasons": reasons,
                    "reported_critical_event_drop_count": reported_drop_count,
                    "metadata_verification": "incomplete_claims_only",
                },
            )

        expected_names = {*FACT_FILES, CHECKSUM_FILE, MANIFEST_FILE, COMPLETE_FILE}
        try:
            children = list(root.iterdir())
        except OSError as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT,
                            "policy capture root is unreadable") from exc
        actual_names = {child.name for child in children}
        if actual_names != expected_names:
            _fail("policy capture artifact inventory is not exact")
        for child in children:
            if _is_linklike(child) or not child.is_file():
                _fail("policy capture contains a link or non-file artifact")

        manifest = _read_json(root / MANIFEST_FILE)
        if not isinstance(manifest, dict):
            _fail("policy capture manifest must be an object")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 3:
            _fail("unsupported policy capture schema", ErrorCode.UNSUPPORTED_SCHEMA)
        if set(manifest) != MANIFEST_FIELDS:
            _fail("policy capture manifest fields are invalid")
        projection = {key: value for key, value in manifest.items()
                      if key != "manifest_digest"}
        if (not isinstance(manifest["manifest_digest"], str)
                or _DIGEST.fullmatch(manifest["manifest_digest"]) is None
                or manifest["manifest_digest"] != sha256_json(projection)):
            _fail("policy capture manifest digest mismatch")
        if (manifest["finalized"] is not True
                or manifest["replayable"] is not True
                or manifest["non_replayable_reasons"] != []
                or type(manifest["critical_event_drop_count"]) is not int
                or manifest["critical_event_drop_count"] != 0):
            _fail("policy capture declares loss or incomplete finalization",
                  ErrorCode.CAPTURE_INCOMPLETE)
        if (type(manifest["event_envelope_version"]) is not int
                or manifest["event_envelope_version"] != EVENT_ENVELOPE_VERSION
                or manifest["event_catalog"] != POLICY_EVENT_CATALOG
                or manifest["capabilities"] != CAPABILITIES
                or manifest["privacy_mode"] != "metadata_only"):
            _fail("policy replay capability/privacy contract is unsupported",
                  ErrorCode.UNSUPPORTED_SCHEMA)
        if not isinstance(manifest["request_id"], str) or not manifest["request_id"]:
            _fail("policy capture request ID is invalid")

        normalized = _validated_metadata(
            identity=manifest["identity"],
            sampling_spec=manifest["sampling_spec"],
            memory_plan=manifest["memory_plan"],
            numerical_policy=manifest["numerical_policy"],
            execution_binding=manifest["execution_binding"],
            observation_profile=manifest["observation_profile"],
            argument=False,
        )
        if any(manifest[field] != normalized[field]
               for field in POLICY_METADATA_FIELDS):
            _fail("policy capture metadata or cross-field digest binding differs")

        files = manifest["files"]
        if not isinstance(files, dict) or set(files) != {*FACT_FILES, CHECKSUM_FILE}:
            _fail("policy capture manifest file coverage is invalid")
        for name, expected in files.items():
            if (not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None
                    or sha256_file(root / name) != expected):
                _fail("policy capture fact checksum mismatch",
                      details={"artifact": name})
        if _read_json(root / CHECKSUM_FILE) != {
            name: files[name] for name in FACT_FILES
        }:
            _fail("policy capture checksums disagree with manifest")
        try:
            raw_events = (root / FACT_FILES[0]).read_bytes()
            if not raw_events or not raw_events.endswith(b"\n"):
                _fail("policy capture events JSONL is empty or truncated")
            text = raw_events.decode("utf-8")
        except IllmError:
            raise
        except (OSError, UnicodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT,
                            "policy capture events JSONL is unreadable") from exc
        values = [_decode_json(line, artifact=FACT_FILES[0])
                  for line in text.splitlines()]
        if any(type(value) is not dict for value in values):
            _fail("policy capture event JSONL entries must be objects")
        events = tuple(PolicyStreamingEvent.from_dict(value) for value in values)
        if type(manifest["event_count"]) is not int or manifest["event_count"] != len(events):
            _fail("policy capture event count differs from manifest")
        state = _play_policy(
            cast(tuple[StreamingEvent, ...], events),
            manifest["request_id"],
            metadata=normalized,
            complete=True,
        )
        if manifest["logical_terminal_outcome"] != state.logical_terminal_outcome:
            _fail("policy capture logical terminal outcome differs from replayed R0 facts",
                  ErrorCode.REPLAY_DIVERGED)
        if manifest["terminal_outcome"] != state.terminal_outcome:
            _fail("policy capture terminal outcome differs from replayed R0 facts",
                  ErrorCode.REPLAY_DIVERGED)
        try:
            complete = (root / COMPLETE_FILE).read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT,
                            "policy capture completion marker is unreadable") from exc
        if complete != manifest["manifest_digest"] + "\n":
            _fail("policy capture completion marker is stale")
        for field in POLICY_METADATA_FIELDS:
            manifest[field] = normalized[field]
        return cls(root, manifest, events, state)

    def request_story(self, request_id: str) -> dict[str, Any]:
        if request_id != self.manifest["request_id"]:
            _fail(
                "request story ID is not present in this single-request policy capture",
                ErrorCode.INVALID_ARGUMENT,
            )
        phases: list[dict[str, Any]] = []
        tokens: list[dict[str, Any]] = []
        allocations: list[dict[str, Any]] = []
        writes: list[dict[str, Any]] = []
        releases: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        for event in self.events:
            payload = thaw_metadata(event.payload)
            item = {
                "producer_seq_no": event.producer_seq_no,
                "monotonic_ns": event.monotonic_ns,
                "kind": event.kind.wire_name,
                "payload": payload,
            }
            timeline.append(item)
            if event.kind is K.REQUEST_PHASE_CHANGED:
                phases.append(item)
            elif event.kind is K.REQUEST_OUTPUT_CHUNK:
                tokens.append(item)
            elif event.kind is K.KV_ALLOCATION_COMMITTED:
                allocations.append(item)
            elif event.kind in {
                K.KV_WRITE_STARTED,
                K.KV_WRITE_COMMITTED,
                K.KV_WRITE_ABORTED,
            }:
                writes.append(item)
            elif event.kind is K.KV_PAGE_RELEASED:
                releases.append(item)
        return {
            "schema_version": 2,
            "capability": "R0",
            "request_id": request_id,
            "capture_identity": _capture_identity(self.manifest),
            "logical_terminal_outcome": self._state.logical_terminal_outcome,
            "terminal_outcome": self._state.terminal_outcome,
            "event_count": len(self.events),
            "phases": phases,
            "tokens": tokens,
            "kv": {
                "allocations": allocations,
                "writes": writes,
                "releases": releases,
                "final_valid_token_count": self._state.kv_valid_token_count,
                "final_active_allocation_bytes": self._state.active_allocation_bytes,
                "final_open_write_count": len(self._state.open_writes),
            },
            "timeline": timeline,
            "missing_or_dropped": [],
        }

    def chrome_trace(self) -> dict[str, Any]:
        base = self.events[0].monotonic_ns if self.events else 0
        trace: list[dict[str, Any]] = []
        for event in self.events:
            payload = thaw_metadata(event.payload)
            item: dict[str, Any] = {
                "name": event.kind.wire_name,
                "cat": event.kind.wire_name.split(".", 1)[0],
                "ph": "i",
                "s": "t",
                "ts": (event.monotonic_ns - base) / 1_000.0,
                "pid": 1,
                "tid": 1,
                "args": {
                    **payload,
                    "producer": event.producer,
                    "source_event_kind": event.kind.wire_name,
                    "source_monotonic_ns": event.monotonic_ns,
                },
            }
            if event.kind is K.MODEL_FORWARD_STARTED:
                item["name"] = "model.forward"
                item["ph"] = "B"
                item.pop("s")
            elif event.kind is K.MODEL_FORWARD_COMPLETED:
                item["name"] = "model.forward"
                item["ph"] = "E"
                item.pop("s")
            elif event.kind is K.MODEL_FORWARD and type(payload.get("duration_ns")) is int:
                item["ph"] = "X"
                item["dur"] = payload["duration_ns"] / 1_000.0
                item["ts"] = max(
                    0.0,
                    (event.monotonic_ns - payload["duration_ns"] - base) / 1_000.0,
                )
                item.pop("s")
            trace.append(item)
        return {
            "traceEvents": trace,
            "displayTimeUnit": "ns",
            "metadata": {
                "schema_version": 2,
                "capability": "R0",
                "event_catalog": POLICY_EVENT_CATALOG,
                "source_clock_domain": "host.perf_counter_ns",
                "trace_timestamp_unit": "us",
                "request_id": self.manifest["request_id"],
                "capture_identity": _capture_identity(self.manifest),
                "derived": True,
            },
        }

"""V0.2 stateful streaming R0 capture, validation, and static learning views.

This module intentionally performs no model execution and imports no GPU/runtime
libraries.  A capture owns serialized metadata facts, never live engine objects.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import (
    EVENT_CATALOG,
    EVENT_ENVELOPE_VERSION,
    StreamingEvent,
    StreamingEventKind as K,
    thaw_metadata,
    validate_metadata,
)

__all__ = ["StreamingCaptureBundle", "StreamingCaptureSession"]


FACT_FILES = ("events.jsonl",)
CHECKSUM_FILE = "checksums.json"
MANIFEST_FILE = "capture-manifest.json"
COMPLETE_FILE = "COMPLETE"
INCOMPLETE_FILE = "INCOMPLETE"
CAPABILITIES = ["R0"]
MANIFEST_FIELDS = {
    "schema_version", "finalized", "event_envelope_version", "event_catalog", "capabilities",
    "privacy_mode", "replayable", "non_replayable_reasons", "critical_event_drop_count",
    "request_id", "identity", "sampling_spec", "sampling_spec_digest", "memory_plan",
    "memory_plan_digest", "event_count", "logical_terminal_outcome", "terminal_outcome",
    "files", "manifest_digest",
}
IDENTITY_REQUIRED = {
    "engine_version", "git_commit", "git_dirty", "toolchain_digest", "device", "dtype",
    "model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest",
}
TERMINALS = {
    K.REQUEST_COMPLETED: "FINISHED",
    K.REQUEST_CANCELLED: "CANCELLED",
    K.REQUEST_FAILED: "FAILED",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_AFTER_TERMINAL = {K.CLOCK_SNAPSHOT}


def _fail(message: str, code: ErrorCode = ErrorCode.CAPTURE_CORRUPT,
          details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_metadata(value, name="identity")
    missing = IDENTITY_REQUIRED - set(result)
    if missing:
        _fail("streaming capture identity is incomplete", details={"missing": sorted(missing)})
    for key in IDENTITY_REQUIRED - {"git_dirty"}:
        if not isinstance(result[key], str) or not result[key]:
            _fail("streaming capture identity string is invalid", details={"field": key})
    if type(result["git_dirty"]) is not bool:
        _fail("streaming capture git_dirty must be boolean")
    for key in ("toolchain_digest", "model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest"):
        if _DIGEST.fullmatch(result[key]) is None:
            _fail("streaming capture identity digest is invalid", details={"field": key})
    if result["dtype"] not in {"fp32", "fp16", "bf16"}:
        _fail("streaming capture dtype is unsupported")
    return result


def _metadata(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = validate_metadata(value, name=name)
    if not result:
        _fail(f"{name} must not be empty")
    return result


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_json(text: str, *, artifact: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in streaming capture", details={"artifact": artifact, "field": key})
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail("non-finite JSON constant in streaming capture", details={"artifact": artifact, "value": value})

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, IllmError):
            raise
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming capture JSON is malformed", {"artifact": artifact}) from exc


def _read_json(path: Path) -> Any:
    try:
        return _decode_json(path.read_text(encoding="utf-8"), artifact=path.name)
    except IllmError:
        raise
    except (OSError, UnicodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming capture JSON is unreadable", {"artifact": path.name}) from exc


@dataclass(slots=True)
class _Playback:
    request_id: str
    tokenized: bool = False
    accepted: bool = False
    input_tokens: int = 0
    max_new_tokens: int = 0
    phase: str = "RECEIVED"
    committed_output_count: int = 0
    kv_valid_token_count: int = 0
    first_token_seen: bool = False
    logical_terminal_outcome: str | None = None
    logical_committed_output_count: int | None = None
    logical_kv_valid_token_count: int | None = None
    terminal_outcome: str | None = None
    producer: str | None = None
    previous_ns: int = -1
    allocations: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_writes: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen_writes: set[str] = field(default_factory=set)
    cancellation_ids: set[str] = field(default_factory=set)

    def apply(self, event: StreamingEvent, expected_seq: int) -> None:
        if event.producer_seq_no != expected_seq:
            _fail("streaming producer sequence has a gap or duplicate", details={"expected": expected_seq, "actual": event.producer_seq_no})
        if self.producer is not None and event.producer != self.producer:
            _fail("V0.2 capture must contain one ordered producer")
        if event.monotonic_ns < self.previous_ns:
            _fail("streaming event timestamp regressed")
        self.producer = event.producer
        self.previous_ns = event.monotonic_ns
        kind = event.kind
        payload = event.payload
        if kind is not K.CLOCK_SNAPSHOT and payload["request_id"] != self.request_id:
            _fail("streaming event belongs to another request")
        if self.terminal_outcome is not None and kind not in _SAFE_AFTER_TERMINAL:
            _fail("streaming request has a non-cleanup event after terminal outcome")

        if kind is K.REQUEST_TOKENIZED:
            if self.tokenized or self.accepted:
                _fail("request tokenization is duplicated or occurs after acceptance")
            self.tokenized = True
            self.input_tokens = payload["input_tokens"]
            self.phase = "TOKENIZED"
        elif kind is K.REQUEST_ACCEPTED:
            if not self.tokenized or self.accepted or payload["input_tokens"] != self.input_tokens:
                _fail("request acceptance does not follow its tokenization/reservation boundary")
            self.accepted = True
            self.max_new_tokens = payload["max_new_tokens"]
        elif kind is K.REQUEST_PHASE_CHANGED:
            self._active()
            if payload["from_phase"] != self.phase or payload["to_phase"] == self.phase:
                _fail("request phase transition does not match recorded phase")
            self.phase = payload["to_phase"]
        elif kind is K.REQUEST_FIRST_TOKEN:
            self._producing()
            if self.first_token_seen or payload["output_index"] != 0 or payload["committed_output_count"] != 1:
                _fail("first-token marker is duplicated or does not describe output zero")
            if payload["kv_valid_token_count"] != self.kv_valid_token_count:
                _fail("first-token marker disagrees with valid KV coverage")
            self.first_token_seen = True
        elif kind is K.REQUEST_OUTPUT_CHUNK:
            self._producing()
            expected_count = self.committed_output_count + 1
            if payload["output_index"] != self.committed_output_count or payload["committed_output_count"] != expected_count:
                _fail("stream output index/count is not contiguous")
            if payload["kv_valid_token_count"] != self.kv_valid_token_count:
                _fail("stream output disagrees with committed KV coverage")
            if self.allocations:
                appended = max(0, self.kv_valid_token_count - self.input_tokens)
                if expected_count - appended not in {0, 1}:
                    _fail("committed output and KV-appended output violate the one-token lag")
            self.committed_output_count = expected_count
        elif kind is K.REQUEST_CANCEL_REQUESTED:
            self._active()
            if self.logical_terminal_outcome is not None:
                _fail("cancel request cannot occur after the logical terminal decision")
            if self.cancellation_ids:
                _fail("canonical stream contains a duplicate cancel request")
            self.cancellation_ids.add(payload["cancellation_id"])
        elif kind is K.REQUEST_TERMINAL_DECIDED:
            self._active()
            if self.logical_terminal_outcome is not None:
                _fail("accepted request must have exactly one logical terminal decision")
            if (payload["committed_output_count"] != self.committed_output_count
                    or payload["kv_valid_token_count"] != self.kv_valid_token_count):
                _fail("logical terminal counters disagree with committed stream/KV facts")
            self.logical_terminal_outcome = payload["outcome"]
            self.logical_committed_output_count = payload["committed_output_count"]
            self.logical_kv_valid_token_count = payload["kv_valid_token_count"]
        elif kind in TERMINALS:
            if not self.accepted or self.terminal_outcome is not None:
                _fail("accepted request must have exactly one terminal outcome")
            if self.logical_terminal_outcome is None:
                _fail("terminal-ready event lacks its logical terminal decision")
            if TERMINALS[kind] != self.logical_terminal_outcome:
                _fail("terminal-ready outcome disagrees with its logical terminal decision")
            if (payload["committed_output_count"] != self.logical_committed_output_count
                    or payload["kv_valid_token_count"] != self.logical_kv_valid_token_count):
                _fail("terminal-ready counters disagree with its logical terminal decision")
            if payload["committed_output_count"] != self.committed_output_count or payload["kv_valid_token_count"] != self.kv_valid_token_count:
                _fail("terminal counters disagree with committed stream/KV facts")
            if kind is K.REQUEST_CANCELLED and payload["cancellation_id"] not in self.cancellation_ids:
                _fail("cancelled terminal lacks its idempotent cancel request")
            if self.open_writes:
                _fail("terminal-ready precedes completion of an in-flight KV write fence")
            if any(not allocation["released"] for allocation in self.allocations.values()):
                _fail("terminal-ready precedes fenced KV resource release")
            self.terminal_outcome = TERMINALS[kind]
        elif kind is K.MODEL_FORWARD_STARTED:
            self._producing()
        elif kind in {K.MODEL_FORWARD_COMPLETED, K.MODEL_FORWARD}:
            self._active()
        elif kind is K.KV_ALLOCATION_COMMITTED:
            self._producing()
            if payload["allocation_id"] in self.allocations:
                _fail("KV allocation ID is duplicated")
            self.allocations[payload["allocation_id"]] = {
                "capacity_tokens": payload["capacity_tokens"],
                "allocated_bytes": payload["allocated_bytes"],
                "released": False,
            }
        elif kind is K.KV_WRITE_STARTED:
            self._producing()
            allocation = self._allocation(payload["allocation_id"])
            if allocation["released"] or payload["write_id"] in self.seen_writes:
                _fail("KV write uses a released allocation or duplicate write ID")
            if any(item["allocation_id"] == payload["allocation_id"] for item in self.open_writes.values()):
                _fail("contiguous KV allocation has overlapping in-flight writes")
            end = payload["start_token"] + payload["token_count"]
            if payload["start_token"] != self.kv_valid_token_count or end > allocation["capacity_tokens"]:
                _fail("KV append range is non-contiguous or exceeds capacity")
            self.seen_writes.add(payload["write_id"])
            self.open_writes[payload["write_id"]] = {
                "allocation_id": payload["allocation_id"], "start_token": payload["start_token"],
                "token_count": payload["token_count"], "lease_id": payload["lease_id"],
                "fence_id": payload["fence_id"],
            }
        elif kind in {K.KV_WRITE_COMMITTED, K.KV_WRITE_ABORTED}:
            if kind is K.KV_WRITE_COMMITTED and self.logical_terminal_outcome is not None:
                _fail("KV validity cannot advance after the logical terminal decision")
            write = self.open_writes.get(payload["write_id"])
            if write is None or any(payload[name] != write[name] for name in ("allocation_id", "lease_id", "fence_id")):
                _fail("KV write resolution does not match an open append lease")
            if kind is K.KV_WRITE_COMMITTED:
                expected_valid = write["start_token"] + write["token_count"]
                if payload["kv_valid_token_count"] != expected_valid:
                    _fail("KV write commit advances validity by the wrong append range")
                self.kv_valid_token_count = expected_valid
            del self.open_writes[payload["write_id"]]
        elif kind is K.KV_PAGE_RELEASED:
            allocation = self._allocation(payload["allocation_id"])
            if allocation["released"] or any(item["allocation_id"] == payload["allocation_id"] for item in self.open_writes.values()):
                _fail("KV allocation was released twice or before its write fence resolved")
            if payload["released_bytes"] != allocation["allocated_bytes"]:
                _fail("KV release byte count differs from its allocation")
            allocation["released"] = True
        elif kind is K.CLOCK_SNAPSHOT:
            return
        else:  # pragma: no cover - exhaustive enum defense
            _fail("unsupported streaming event kind", ErrorCode.UNSUPPORTED_SCHEMA)

    def _active(self) -> None:
        if not self.accepted or self.terminal_outcome is not None:
            _fail("streaming operation occurs outside an active accepted request")

    def _producing(self) -> None:
        self._active()
        if self.logical_terminal_outcome is not None:
            _fail("streaming production cannot continue after the logical terminal decision")

    def _allocation(self, allocation_id: str) -> dict[str, Any]:
        allocation = self.allocations.get(allocation_id)
        if allocation is None:
            _fail("KV event references an unknown allocation")
        return allocation

    def validate_complete(self) -> None:
        if not self.tokenized or not self.accepted:
            _fail("streaming capture lacks tokenized/accepted facts", ErrorCode.CAPTURE_INCOMPLETE)
        if self.open_writes:
            _fail("streaming capture ends with unresolved KV writes", ErrorCode.CAPTURE_INCOMPLETE,
                  {"write_ids": sorted(self.open_writes)})
        unreleased = sorted(name for name, item in self.allocations.items() if not item["released"])
        if unreleased:
            _fail("streaming capture ends with unreleased KV allocations", ErrorCode.CAPTURE_INCOMPLETE,
                  {"allocation_ids": unreleased})
        if self.logical_terminal_outcome is None:
            _fail("streaming capture lacks its accepted request logical terminal decision", ErrorCode.CAPTURE_INCOMPLETE)
        if self.terminal_outcome is None:
            _fail("streaming capture lacks its accepted request terminal-ready event", ErrorCode.CAPTURE_INCOMPLETE)
        if self.logical_terminal_outcome != self.terminal_outcome:
            _fail("streaming logical and terminal-ready outcomes disagree", ErrorCode.REPLAY_DIVERGED)
        if self.first_token_seen != (self.committed_output_count > 0):
            _fail("streaming capture first-token marker disagrees with committed output", ErrorCode.CAPTURE_INCOMPLETE)

    @property
    def active_allocation_bytes(self) -> int:
        return sum(int(item["allocated_bytes"]) for item in self.allocations.values() if not item["released"])


def _play(events: tuple[StreamingEvent, ...], request_id: str, *, complete: bool) -> _Playback:
    state = _Playback(request_id)
    for index, event in enumerate(events):
        state.apply(event, index)
    if complete:
        state.validate_complete()
    return state


def _manifest(
    *, finalized: bool, replayable: bool, reasons: list[str], request_id: str,
    identity: Mapping[str, Any], sampling_spec: Mapping[str, Any], memory_plan: Mapping[str, Any],
    events: tuple[StreamingEvent, ...], logical_terminal_outcome: str | None,
    terminal_outcome: str | None, files: Mapping[str, str], critical_event_drop_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 2,
        "finalized": finalized,
        "event_envelope_version": EVENT_ENVELOPE_VERSION,
        "event_catalog": EVENT_CATALOG,
        "capabilities": list(CAPABILITIES),
        "privacy_mode": "metadata_only",
        "replayable": replayable,
        "non_replayable_reasons": reasons,
        "critical_event_drop_count": critical_event_drop_count,
        "request_id": request_id,
        "identity": thaw_metadata(identity),
        "sampling_spec": thaw_metadata(sampling_spec),
        "sampling_spec_digest": sha256_json(thaw_metadata(sampling_spec)),
        "memory_plan": thaw_metadata(memory_plan),
        "memory_plan_digest": sha256_json(thaw_metadata(memory_plan)),
        "event_count": len(events),
        "logical_terminal_outcome": logical_terminal_outcome,
        "terminal_outcome": terminal_outcome,
        "files": dict(files),
    }
    result["manifest_digest"] = sha256_json(result)
    return result


@dataclass(slots=True)
class StreamingCaptureSession:
    path: Path
    identity: Mapping[str, Any]
    request_id: str
    sampling_spec: Mapping[str, Any]
    memory_plan: Mapping[str, Any]
    _events: list[StreamingEvent] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)
    _finalized: bool = field(default=False, init=False)
    _critical_event_drop_count: int = field(default=0, init=False)
    _incomplete_reasons: list[str] = field(default_factory=list, init=False)
    _identity: dict[str, Any] = field(init=False)
    _sampling_spec: dict[str, Any] = field(init=False)
    _memory_plan: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        requested_path = Path(self.path)
        if requested_path.is_symlink():
            _fail("streaming capture root cannot be a symbolic link", ErrorCode.INVALID_ARGUMENT)
        self.path = requested_path.resolve()
        if not isinstance(self.request_id, str) or not self.request_id:
            _fail("streaming request ID must be nonempty", ErrorCode.INVALID_ARGUMENT)
        self._identity = _identity(self.identity)
        self._sampling_spec = _metadata(self.sampling_spec, "sampling_spec")
        self._memory_plan = _metadata(self.memory_plan, "memory_plan")
        if self.root.exists() and any(self.root.iterdir()):
            _fail("streaming capture output must be a new or empty directory", ErrorCode.INVALID_ARGUMENT)
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(self.root / INCOMPLETE_FILE, b"streaming_capture_not_finalized\n")

    @property
    def root(self) -> Path:
        """Resolved bundle path; ``path`` is the constructor-facing name."""

        return self.path

    def emit(self, kind: K, payload: Mapping[str, Any]) -> StreamingEvent:
        if self._closed:
            _fail("streaming capture session is closed", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(kind, K):
            _fail("streaming event kind is invalid", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(payload, Mapping):
            _fail("streaming event payload must be metadata", ErrorCode.INVALID_ARGUMENT)
        event_payload = dict(payload)
        if kind is not K.CLOCK_SNAPSHOT:
            event_payload.setdefault("request_id", self.request_id)
        event = StreamingEvent("stateful-engine", len(self._events), time.perf_counter_ns(), kind, event_payload)
        _play(tuple((*self._events, event)), self.request_id, complete=False)
        self._events.append(event)
        return event

    def _write(self, *, finalized: bool, replayable: bool, reasons: list[str]) -> dict[str, Any]:
        events = tuple(self._events)
        state = _play(events, self.request_id, complete=finalized)
        encoded = b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in events)
        _atomic_bytes(self.root / FACT_FILES[0], encoded)
        facts = {FACT_FILES[0]: sha256_file(self.root / FACT_FILES[0])}
        _atomic_bytes(self.root / CHECKSUM_FILE, canonical_json_bytes(facts) + b"\n")
        files = {**facts, CHECKSUM_FILE: sha256_file(self.root / CHECKSUM_FILE)}
        manifest = _manifest(
            finalized=finalized, replayable=replayable, reasons=reasons, request_id=self.request_id,
            identity=self._identity, sampling_spec=self._sampling_spec, memory_plan=self._memory_plan,
            events=events, logical_terminal_outcome=state.logical_terminal_outcome,
            terminal_outcome=state.terminal_outcome, files=files,
            critical_event_drop_count=self._critical_event_drop_count,
        )
        _atomic_bytes(self.root / MANIFEST_FILE, canonical_json_bytes(manifest) + b"\n")
        return manifest

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            _fail("streaming capture session is closed", ErrorCode.INVALID_ARGUMENT)
        try:
            manifest = self._write(finalized=True, replayable=True, reasons=[])
            _atomic_bytes(self.root / COMPLETE_FILE, (manifest["manifest_digest"] + "\n").encode("ascii"))
            (self.root / INCOMPLETE_FILE).unlink()
            self._closed = True
            self._finalized = True
            return manifest
        except BaseException:
            # INCOMPLETE is deliberately retained across validation, I/O, interrupt,
            # or process failures.  A reader must never infer success from prefix data.
            raise

    def mark_incomplete(self, reason: str, *, critical_event_drop_count: int = 0) -> dict[str, Any]:
        if self._finalized or (self._closed and not self._incomplete_reasons):
            _fail("a complete streaming capture cannot be rewritten", ErrorCode.INVALID_ARGUMENT)
        if not isinstance(reason, str) or not reason.strip():
            _fail("incomplete capture reason must be nonempty", ErrorCode.INVALID_ARGUMENT)
        if type(critical_event_drop_count) is not int or critical_event_drop_count < self._critical_event_drop_count:
            _fail("critical capture loss count must be a nonnegative monotone integer", ErrorCode.INVALID_ARGUMENT)
        self._critical_event_drop_count = critical_event_drop_count
        if reason.strip() not in self._incomplete_reasons:
            self._incomplete_reasons.append(reason.strip())
        # Once incomplete, emit/finalize remain closed. Only failure metadata
        # may be updated as the engine counts later canonical events that this
        # required writer did not receive. The captured event prefix is fixed.
        manifest = self._write(finalized=False, replayable=False, reasons=list(self._incomplete_reasons))
        _atomic_bytes(self.root / INCOMPLETE_FILE, (reason.strip() + "\n").encode("utf-8"))
        (self.root / COMPLETE_FILE).unlink(missing_ok=True)
        self._closed = True
        return manifest


@dataclass(frozen=True, slots=True)
class StreamingCaptureBundle:
    root: Path
    manifest: dict[str, Any]
    events: tuple[StreamingEvent, ...]
    _state: _Playback

    @classmethod
    def open(cls, path: Path) -> StreamingCaptureBundle:
        requested_path = Path(path)
        if requested_path.is_symlink():
            _fail("streaming capture root must not be a symbolic link")
        root = requested_path.resolve()
        if not root.is_dir():
            _fail("streaming capture root must be an ordinary directory")
        if (root / INCOMPLETE_FILE).exists():
            reasons: list[str] = []
            reported_drop_count: int | None = None
            if (root / MANIFEST_FILE).is_file():
                value = _read_json(root / MANIFEST_FILE)
                if isinstance(value, dict) and isinstance(value.get("non_replayable_reasons"), list):
                    reasons = [item for item in value["non_replayable_reasons"] if isinstance(item, str)]
                if isinstance(value, dict) and type(value.get("critical_event_drop_count")) is int and value["critical_event_drop_count"] >= 0:
                    reported_drop_count = value["critical_event_drop_count"]
            _fail("streaming capture is incomplete/non-replayable", ErrorCode.CAPTURE_INCOMPLETE,
                  {"reasons": reasons, "reported_critical_event_drop_count": reported_drop_count,
                   "metadata_verification": "incomplete_claims_only"})
        manifest_path = root / MANIFEST_FILE
        if manifest_path.is_symlink() or not manifest_path.is_file():
            _fail("streaming capture manifest is missing or linked")
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            _fail("streaming capture manifest must be an object")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
            _fail("unsupported streaming capture schema", ErrorCode.UNSUPPORTED_SCHEMA)
        expected_names = {*FACT_FILES, CHECKSUM_FILE, MANIFEST_FILE, COMPLETE_FILE}
        actual_names: set[str] = set()
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file():
                _fail("streaming capture contains a link or non-file artifact")
            actual_names.add(path.name)
        if actual_names != expected_names:
            _fail("streaming capture artifact inventory is not exact")
        if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
            _fail("streaming capture manifest fields are invalid")
        projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if manifest["manifest_digest"] != sha256_json(projection):
            _fail("streaming capture manifest digest mismatch")
        if (manifest["finalized"] is not True or manifest["replayable"] is not True
                or manifest["non_replayable_reasons"] != []
                or type(manifest["critical_event_drop_count"]) is not int
                or manifest["critical_event_drop_count"] != 0):
            _fail("streaming capture declares loss or incomplete finalization", ErrorCode.CAPTURE_INCOMPLETE)
        if (type(manifest["event_envelope_version"]) is not int
                or manifest["event_envelope_version"] != EVENT_ENVELOPE_VERSION
                or manifest["event_catalog"] != EVENT_CATALOG or manifest["capabilities"] != CAPABILITIES
                or manifest["privacy_mode"] != "metadata_only"):
            _fail("streaming replay capability/privacy contract is unsupported", ErrorCode.UNSUPPORTED_SCHEMA)
        if not isinstance(manifest["request_id"], str) or not manifest["request_id"]:
            _fail("streaming manifest request ID is invalid")
        identity = _identity(manifest["identity"])
        sampling = _metadata(manifest["sampling_spec"], "sampling_spec")
        memory = _metadata(manifest["memory_plan"], "memory_plan")
        if manifest["sampling_spec_digest"] != sha256_json(sampling) or manifest["memory_plan_digest"] != sha256_json(memory):
            _fail("streaming manifest sampling/memory digest mismatch")
        files = manifest["files"]
        if not isinstance(files, dict) or set(files) != {*FACT_FILES, CHECKSUM_FILE}:
            _fail("streaming manifest file coverage is invalid")
        for name, expected in files.items():
            if not isinstance(expected, str) or _DIGEST.fullmatch(expected) is None or sha256_file(root / name) != expected:
                _fail("streaming capture fact checksum mismatch", details={"artifact": name})
        if _read_json(root / CHECKSUM_FILE) != {name: files[name] for name in FACT_FILES}:
            _fail("streaming checksums disagree with manifest")
        try:
            lines = (root / FACT_FILES[0]).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "streaming events JSONL is unreadable") from exc
        values = [_decode_json(line, artifact=FACT_FILES[0]) for line in lines]
        events = tuple(StreamingEvent.from_dict(value) for value in values)
        if type(manifest["event_count"]) is not int or manifest["event_count"] != len(events):
            _fail("streaming event count differs from manifest")
        state = _play(events, manifest["request_id"], complete=True)
        if manifest["logical_terminal_outcome"] != state.logical_terminal_outcome:
            _fail("streaming logical terminal outcome differs from replayed R0 facts", ErrorCode.REPLAY_DIVERGED)
        if manifest["terminal_outcome"] != state.terminal_outcome:
            _fail("streaming terminal outcome differs from replayed R0 facts", ErrorCode.REPLAY_DIVERGED)
        if (root / COMPLETE_FILE).read_text(encoding="ascii") != manifest["manifest_digest"] + "\n":
            _fail("streaming completion marker is stale")
        # Retain the fully validated canonical objects, not caller-owned aliases.
        manifest["identity"], manifest["sampling_spec"], manifest["memory_plan"] = identity, sampling, memory
        return cls(root, manifest, events, state)

    def request_story(self, request_id: str) -> dict[str, Any]:
        if request_id != self.manifest["request_id"]:
            _fail("request story ID is not present in this single-request capture", ErrorCode.INVALID_ARGUMENT)
        phases: list[dict[str, Any]] = []
        tokens: list[dict[str, Any]] = []
        allocations: list[dict[str, Any]] = []
        writes: list[dict[str, Any]] = []
        releases: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []
        for event in self.events:
            payload = thaw_metadata(event.payload)
            item = {"producer_seq_no": event.producer_seq_no, "monotonic_ns": event.monotonic_ns,
                    "kind": event.kind.wire_name, "payload": payload}
            timeline.append(item)
            if event.kind is K.REQUEST_PHASE_CHANGED:
                phases.append(item)
            elif event.kind is K.REQUEST_OUTPUT_CHUNK:
                tokens.append(item)
            elif event.kind is K.KV_ALLOCATION_COMMITTED:
                allocations.append(item)
            elif event.kind in {K.KV_WRITE_STARTED, K.KV_WRITE_COMMITTED, K.KV_WRITE_ABORTED}:
                writes.append(item)
            elif event.kind is K.KV_PAGE_RELEASED:
                releases.append(item)
        return {
            "schema_version": 1,
            "capability": "R0",
            "request_id": request_id,
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
                "schema_version": 1,
                "capability": "R0",
                "event_catalog": EVENT_CATALOG,
                "source_clock_domain": "host.perf_counter_ns",
                "trace_timestamp_unit": "us",
                "request_id": self.manifest["request_id"],
                "derived": True,
            },
        }

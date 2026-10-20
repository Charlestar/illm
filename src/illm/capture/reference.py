"""Metadata-only reference captures and control replay; no numerical execution."""

from __future__ import annotations

import json
import math
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEvent, ReferenceEventKind
from illm.sampling import SamplingError, SamplingSpec


FACT_FILES = ("checkpoint.json", "events.jsonl", "state-digests.jsonl")
MANIFEST_FILES = (*FACT_FILES, "checksums.json")
IDENTITY_REQUIRED = {"model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest", "input_spec_digest"}
IDENTITY_OPTIONAL = {
    "model_id", "tokenizer_spec_digest", "input_template_digest", "runtime_dtype", "device", "backend",
    "build_digest", "model_numerical_contract", "sampling_numerical_contract",
}
STATE_FIELDS = {
    "schema_version", "request_id", "status", "identity", "sampling_spec", "sampling_spec_digest",
    "max_new_tokens", "input_tokens", "input_digest", "output_tokens", "output_digest",
    "next_output_position", "rng_draws_consumed", "committed_draw_addresses", "last_logits_digest",
    "last_selected_digest", "finish_reason", "error_code", "op_count",
}
MANIFEST_FIELDS = {
    "schema_version", "finalized", "replay_capabilities", "privacy_mode", "critical_event_drop_count",
    "non_replayable_reasons", "request_id", "identity", "sampling_spec", "event_count", "run_status",
    "final_state_digest", "files", "manifest_digest",
}


def _fail(message: str, code: ErrorCode = ErrorCode.CAPTURE_CORRUPT) -> None:
    raise IllmError(code, message)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_digest(value: object, name: str) -> None:
    if not _is_digest(value):
        _fail(f"{name} must be a canonical SHA256 digest")


def _count(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identity(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not IDENTITY_REQUIRED <= set(value) or set(value) - IDENTITY_REQUIRED - IDENTITY_OPTIONAL:
        _fail("reference identity fields do not match metadata-only contract")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str) or not item:
            _fail("reference identity values must be nonempty strings")
        if key.endswith("_digest"):
            _require_digest(item, key)
        result[key] = item
    return result


def _initial_state(identity: Mapping[str, Any], request_id: str, sampling_spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request_id, str) or not request_id:
        _fail("reference request ID must be nonempty")
    try:
        spec = SamplingSpec.from_dict(sampling_spec)
    except SamplingError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference SamplingSpec is unsupported", {"sampling_error": exc.code}) from exc
    return {
        "schema_version": 1, "request_id": request_id, "status": "NEW", "identity": _identity(identity),
        "sampling_spec": spec.to_dict(), "sampling_spec_digest": spec.digest, "max_new_tokens": None,
        "input_tokens": 0, "input_digest": None, "output_tokens": 0, "output_digest": None,
        "next_output_position": 0, "rng_draws_consumed": 0, "committed_draw_addresses": [],
        "last_logits_digest": None, "last_selected_digest": None, "finish_reason": None,
        "error_code": None, "op_count": 0,
    }


def _validate_checkpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != STATE_FIELDS:
        _fail("reference checkpoint fields are invalid")
    expected = _initial_state(value["identity"], value["request_id"], value["sampling_spec"])
    if canonical_json_bytes(dict(value)) != canonical_json_bytes(expected):
        _fail("reference checkpoint is not a canonical initial request state")
    return expected


def _error_code(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", value) is None:
        _fail("reference failure code must be a stable identifier")
    return str(value)


def _apply_transition(state: Mapping[str, Any], event: ReferenceEvent) -> dict[str, Any]:
    result = deepcopy(dict(state))
    payload = event.payload
    kind = event.kind
    status = str(result["status"])
    if payload["request_id"] != result["request_id"]:
        _fail("reference event belongs to another request")
    if status in {"FINISHED", "FAILED"}:
        _fail("reference request has events after its terminal outcome")
    if "output_position" in payload and _count(payload["output_position"], "output_position") != result["next_output_position"]:
        _fail("reference event output position does not match committed state")

    if kind is ReferenceEventKind.REQUEST_ACCEPTED:
        if status != "NEW":
            _fail("reference request was accepted more than once")
        result.update(status="ACCEPTED", max_new_tokens=_count(payload["max_new_tokens"], "max_new_tokens", 1))
    elif status == "NEW":
        _fail("reference request event precedes acceptance")
    elif kind is ReferenceEventKind.REQUEST_FAILED:
        error_code = _error_code(payload["error_code"])
        if status == "SAMPLING_FAILED" and result["error_code"] != error_code:
            _fail("request failure differs from recorded sampling failure")
        result.update(status="FAILED", error_code=error_code)
    elif kind is ReferenceEventKind.REQUEST_TOKENIZED:
        if status != "ACCEPTED":
            _fail("reference tokenization occurs outside accepted state")
        _require_digest(payload["input_digest"], "input_digest")
        result.update(status="READY", input_tokens=_count(payload["input_tokens"], "input_tokens", 1), input_digest=payload["input_digest"])
    elif kind is ReferenceEventKind.MODEL_FORWARD_STARTED:
        if status != "READY" or result["output_tokens"] >= result["max_new_tokens"]:
            _fail("reference forward starts outside an eligible request iteration")
        if _count(payload["context_tokens"], "context_tokens", 1) != result["input_tokens"] + result["output_tokens"]:
            _fail("reference full-context forward length is inconsistent")
        result.update(status="FORWARD", last_logits_digest=None)
    elif kind is ReferenceEventKind.MODEL_OP:
        if status != "FORWARD":
            _fail("model op occurs outside forward")
        if not isinstance(payload["op_name"], str) or re.fullmatch(r"[a-zA-Z0-9_.]{1,160}", payload["op_name"]) is None:
            _fail("model op name is invalid")
        shape = payload["shape"]
        if not isinstance(shape, (tuple, list)):
            _fail("model op shape must be a list of dimensions")
        size = 1
        for dimension in shape:
            size *= _count(dimension, "model op dimension")
        if payload["dtype"] not in {"fp32", "float32", "torch.float32", "fp16", "float16", "torch.float16", "bf16", "bfloat16", "torch.bfloat16"}:
            _fail("model op dtype is unsupported")
        if "duration_ns" in payload:
            _count(payload["duration_ns"], "model op duration")
        summary = payload.get("summary", {})
        for key, value in summary.items():
            if key in {"finite_count", "nonfinite_count"}:
                _count(value, key)
            elif value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                _fail("model op summary must contain finite scalar metadata")
        if "finite_count" in summary and "nonfinite_count" in summary and summary["finite_count"] + summary["nonfinite_count"] != size:
            _fail("model op summary counts do not cover its shape")
        result["op_count"] += 1
    elif kind is ReferenceEventKind.MODEL_FORWARD_COMPLETED:
        if status != "FORWARD":
            _fail("reference forward completion is out of order")
        _require_digest(payload["logits_digest"], "logits_digest")
        result.update(status="FORWARD_DONE", last_logits_digest=payload["logits_digest"])
    elif kind is ReferenceEventKind.SAMPLING_STARTED:
        if status != "FORWARD_DONE" or payload["sampling_spec_digest"] != result["sampling_spec_digest"]:
            _fail("sampling begins before forward completion or with a changed SamplingSpec")
        result["status"] = "SAMPLING"
    elif kind is ReferenceEventKind.SAMPLING_FAILED:
        if status != "SAMPLING":
            _fail("sampling failure occurs outside an active sampling decision")
        result.update(status="SAMPLING_FAILED", error_code=_error_code(payload["error_code"]))
    elif kind is ReferenceEventKind.SAMPLING_SELECTED:
        if status != "SAMPLING" or payload["sampling_spec_digest"] != result["sampling_spec_digest"]:
            _fail("sampling selection is out of order or changes SamplingSpec")
        if payload["logits_digest"] != result["last_logits_digest"]:
            _fail("sampling selection does not bind the completed forward logits")
        _require_digest(payload["selected_digest"], "selected_digest")
        address = payload["draw_address"]
        if result["sampling_spec"]["mode"] == "greedy":
            if address is not None:
                _fail("greedy selection must not consume a random draw")
        else:
            expected_address = [result["next_output_position"], 0, 0]
            if not isinstance(address, (tuple, list)) or len(address) != 3 or any(type(item) is not int for item in address) or list(address) != expected_address:
                _fail("sampling draw address differs from its committed output position")
            result["committed_draw_addresses"].append(expected_address)
            result["rng_draws_consumed"] += 1
        result["output_tokens"] += 1
        result["next_output_position"] += 1
        result.update(status="READY", last_selected_digest=payload["selected_digest"])
    elif kind is ReferenceEventKind.REQUEST_FINISHED:
        if status != "READY" or result["output_tokens"] < 1:
            _fail("reference finish occurs before a committed token or during an iteration")
        if _count(payload["output_tokens"], "output_tokens") != result["output_tokens"]:
            _fail("reference terminal output length differs from committed selections")
        reason = payload["finish_reason"]
        if reason not in {"MAX_NEW_TOKENS", "EOS"} or (reason == "MAX_NEW_TOKENS" and result["output_tokens"] != result["max_new_tokens"]):
            _fail("reference finish reason does not agree with request limits")
        _require_digest(payload["output_digest"], "output_digest")
        result.update(status="FINISHED", finish_reason=reason, output_digest=payload["output_digest"])
    else:
        _fail("unsupported reference control transition", ErrorCode.UNSUPPORTED_SCHEMA)
    return result


def _apply(state: Mapping[str, Any], event: ReferenceEvent) -> dict[str, Any]:
    try:
        return _apply_transition(state, event)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event metadata has an invalid type") from exc


def replay_reference_events(events: Iterable[ReferenceEvent], initial_state: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct recorded controls; never import a model or compute a sample."""
    state = _validate_checkpoint(initial_state)
    previous_timestamp = -1
    producer: str | None = None
    for sequence, event in enumerate(events):
        if event.seq_no != sequence or event.monotonic_ns < previous_timestamp:
            _fail("reference event sequence gap or timestamp regression")
        if producer is not None and producer != event.producer:
            _fail("reference capture must have one ordered producer")
        state = _apply(state, event)
        previous_timestamp = event.monotonic_ns
        producer = event.producer
    if state["status"] not in {"FINISHED", "FAILED"}:
        _fail("reference capture lacks a terminal request outcome", ErrorCode.CAPTURE_INCOMPLETE)
    return state


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _decode_json(text: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in reference capture")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail(f"non-finite JSON constant in reference capture: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference JSON is malformed") from exc


def _read_json(path: Path) -> Any:
    try:
        return _decode_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference capture JSON is unreadable", {"file": path.name}) from exc


@dataclass(slots=True)
class ReferenceCaptureSession:
    root: Path
    identity: Mapping[str, Any]
    request_id: str
    sampling_spec: Mapping[str, Any]
    initial_state: dict[str, Any] = field(init=False)
    state: dict[str, Any] = field(init=False)
    _events: list[ReferenceEvent] = field(default_factory=list, init=False)
    _records: list[dict[str, Any]] = field(default_factory=list, init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.initial_state = _initial_state(self.identity, self.request_id, self.sampling_spec)
        self.state = deepcopy(self.initial_state)
        if self.root.exists() and any((self.root / name).exists() for name in (*MANIFEST_FILES, "capture-manifest.json", "COMPLETE", "INCOMPLETE")):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "reference capture output already contains capture artifacts")
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(self.root / "INCOMPLETE", b"reference_capture_not_finalized\n")

    def emit(self, kind: ReferenceEventKind, payload: Mapping[str, Any]) -> ReferenceEvent:
        if self._finalized:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "reference capture is already finalized")
        event_payload = dict(payload)
        event_payload.setdefault("request_id", self.request_id)
        event = ReferenceEvent("reference-engine", len(self._events), time.monotonic_ns(), kind, event_payload)
        next_state = _apply(self.state, event)
        self._events.append(event)
        self.state = next_state
        self._records.append({"seq_no": event.seq_no, "state_digest": sha256_json(self.state), "state": deepcopy(self.state)})
        return event

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "reference capture is already finalized")
        final_state = replay_reference_events(self._events, self.initial_state)
        if final_state != self.state:
            _fail("online reference capture state differs from canonical replay", ErrorCode.REPLAY_DIVERGED)
        _atomic_bytes(self.root / "checkpoint.json", canonical_json_bytes(self.initial_state) + b"\n")
        _atomic_bytes(self.root / "events.jsonl", b"".join(canonical_json_bytes(event.to_dict()) + b"\n" for event in self._events))
        _atomic_bytes(self.root / "state-digests.jsonl", b"".join(canonical_json_bytes(record) + b"\n" for record in self._records))
        checksums = {name: sha256_file(self.root / name) for name in FACT_FILES}
        _atomic_bytes(self.root / "checksums.json", canonical_json_bytes(checksums) + b"\n")
        manifest: dict[str, Any] = {
            "schema_version": 1, "finalized": True, "replay_capabilities": ["R0", "R1_REFERENCE"],
            "privacy_mode": "metadata_only", "critical_event_drop_count": 0, "non_replayable_reasons": [],
            "request_id": self.request_id, "identity": self.initial_state["identity"],
            "sampling_spec": self.initial_state["sampling_spec"], "event_count": len(self._events),
            "run_status": final_state["status"], "final_state_digest": sha256_json(final_state),
            "files": {**checksums, "checksums.json": sha256_file(self.root / "checksums.json")},
        }
        manifest["manifest_digest"] = sha256_json(manifest)
        _atomic_bytes(self.root / "capture-manifest.json", canonical_json_bytes(manifest) + b"\n")
        _atomic_bytes(self.root / "COMPLETE", b"complete\n")
        (self.root / "INCOMPLETE").unlink()
        self._finalized = True
        return manifest


@dataclass(frozen=True, slots=True)
class ReferenceCaptureBundle:
    root: Path
    manifest: dict[str, Any]
    initial_state: dict[str, Any]
    events: tuple[ReferenceEvent, ...]
    final_state: dict[str, Any]
    state_digests: tuple[dict[str, Any], ...]

    @classmethod
    def open(cls, root: Path, *, expected_identity: Mapping[str, Any] | None = None) -> ReferenceCaptureBundle:
        root = root.resolve()
        if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
            _fail("reference capture is not finalized", ErrorCode.CAPTURE_INCOMPLETE)
        manifest = _read_json(root / "capture-manifest.json")
        if not isinstance(manifest, dict):
            _fail("reference capture manifest must be an object")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            _fail("unsupported reference capture schema", ErrorCode.UNSUPPORTED_SCHEMA)
        if set(manifest) != MANIFEST_FIELDS or manifest.get("finalized") is not True:
            _fail("reference capture manifest fields/finalization are invalid")
        projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        try:
            if manifest["manifest_digest"] != sha256_json(projection):
                _fail("reference manifest digest mismatch")
        except (ValueError, TypeError, OverflowError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference manifest is not canonical JSON") from exc
        if manifest["replay_capabilities"] != ["R0", "R1_REFERENCE"] or manifest["privacy_mode"] != "metadata_only":
            _fail("unsupported reference replay capability/privacy contract", ErrorCode.UNSUPPORTED_SCHEMA)
        if type(manifest["critical_event_drop_count"]) is not int or manifest["critical_event_drop_count"] != 0 or manifest["non_replayable_reasons"] != []:
            _fail("reference capture declares critical loss", ErrorCode.CAPTURE_INCOMPLETE)
        files = manifest["files"]
        if not isinstance(files, dict) or set(files) != set(MANIFEST_FILES):
            _fail("reference capture lacks exact fact-file checksum coverage")
        for name, expected in files.items():
            path = root / name
            if not _is_digest(expected) or path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
                _fail("reference fact checksum mismatch")
        if _read_json(root / "checksums.json") != {name: files[name] for name in FACT_FILES}:
            _fail("reference checksums disagree with manifest")
        initial = _validate_checkpoint(_read_json(root / "checkpoint.json"))
        if initial["request_id"] != manifest["request_id"] or initial["identity"] != manifest["identity"] or initial["sampling_spec"] != manifest["sampling_spec"]:
            _fail("reference checkpoint/manifest execution identity mismatch")
        if expected_identity is not None and initial["identity"] != _identity(expected_identity):
            _fail("reference capture belongs to a different execution identity")
        try:
            event_values = [_decode_json(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            records = [_decode_json(line) for line in (root / "state-digests.jsonl").read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, ValueError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "reference event/state JSONL is unreadable") from exc
        events = tuple(ReferenceEvent.from_dict(value) for value in event_values)
        if type(manifest["event_count"]) is not int or len(events) != manifest["event_count"] or len(records) != len(events):
            _fail("reference event/state record count mismatch")
        final_state = replay_reference_events(events, initial)
        state = initial
        for event, record in zip(events, records):
            state = _apply(state, event)
            if not isinstance(record, dict) or set(record) != {"seq_no", "state_digest", "state"} or type(record["seq_no"]) is not int or record["seq_no"] != event.seq_no:
                _fail("reference state records are not canonical and contiguous")
            if record["state_digest"] != sha256_json(state) or sha256_json(record["state"]) != sha256_json(state):
                raise IllmError(ErrorCode.REPLAY_DIVERGED, "reference control state diverged", {"seq_no": event.seq_no})
        if manifest["final_state_digest"] != sha256_json(final_state) or manifest["run_status"] != final_state["status"]:
            _fail("reference final state/terminal status mismatch", ErrorCode.REPLAY_DIVERGED)
        return cls(root, manifest, initial, events, final_state, tuple(records))

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from illm.canonical import canonical_json_bytes, canonical_json_text, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent
from illm.domain.state import RequestStatus, ToyRequest, ToyState


FACT_FILES = ("checkpoint.json", "events.jsonl", "state-digests.jsonl")
MANIFEST_FACT_FILES = (*FACT_FILES, "checksums.json")
SHA256_HEX_LENGTH = 64


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_identity(project_root: Path) -> tuple[str, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unborn"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return revision, dirty


def _state_from_dict(value: dict[str, Any]) -> ToyState:
    requests = []
    for item in value.get("requests", []):
        try:
            status = RequestStatus[item["status"]]
        except (KeyError, TypeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "invalid checkpoint request") from exc
        requests.append(ToyRequest(str(item["request_id"]), int(item["arrival_step"]), status))
    return ToyState(int(value.get("current_step", -1)), tuple(sorted(requests)))


@dataclass(slots=True)
class CaptureSession:
    root: Path
    seed: int
    steps: int
    project_root: Path
    _events: list[DomainEvent] = field(init=False, default_factory=list)
    _state_digests: list[dict[str, Any]] = field(init=False, default_factory=list)
    _initial_state: ToyState = field(init=False, default_factory=ToyState)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("COMPLETE", "capture-manifest.json"):
            (self.root / name).unlink(missing_ok=True)
        _atomic_write(
            self.root / "INCOMPLETE",
            canonical_json_bytes({"schema_version": 0, "reason": "capture_not_finalized"}),
        )
        self._events.clear()
        self._state_digests.clear()
        self._initial_state = ToyState()

    def append(self, event: DomainEvent) -> None:
        self._events.append(event)

    def record_step_digest(self, step_id: int, state: ToyState) -> None:
        self._state_digests.append(
            {"step_id": step_id, "state_digest": state.digest, "state": state.to_dict()}
        )

    def finalize(self, final_state: ToyState, *, critical_event_drop_count: int = 0) -> dict[str, Any]:
        if final_state.waiting():
            raise IllmError(
                ErrorCode.CAPTURE_INCOMPLETE,
                "toy run ended with accepted requests lacking a terminal event",
                {"waiting": [request.request_id for request in final_state.waiting()]},
            )
        checkpoint = self._initial_state.to_dict()
        events = "".join(f"{event.to_json_line()}\n" for event in self._events).encode("utf-8")
        digests = "".join(f"{canonical_json_text(item)}\n" for item in self._state_digests).encode("utf-8")
        _atomic_write(self.root / "checkpoint.json", canonical_json_bytes(checkpoint) + b"\n")
        _atomic_write(self.root / "events.jsonl", events)
        _atomic_write(self.root / "state-digests.jsonl", digests)
        checksums = {name: sha256_file(self.root / name) for name in FACT_FILES}
        _atomic_write(self.root / "checksums.json", canonical_json_bytes(checksums) + b"\n")
        revision, dirty = _git_identity(self.project_root)
        manifest: dict[str, Any] = {
            "schema_version": 0,
            "finalized": True,
            "replay_capabilities": ["R0", "R1_TOY"],
            "privacy_mode": "metadata_only",
            "critical_event_drop_count": critical_event_drop_count,
            "non_replayable_reasons": [],
            "seed": self.seed,
            "steps": self.steps,
            "event_count": len(self._events),
            "final_state_digest": final_state.digest,
            "files": {**checksums, "checksums.json": sha256_file(self.root / "checksums.json")},
            "run": {
                "git_revision": revision,
                "git_dirty": dirty,
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "os": platform.system(),
                "arch": platform.machine(),
                "seed": self.seed,
                "event_schema_version": 0,
                "privacy_mode": "metadata_only",
                "critical_event_drop_count": critical_event_drop_count,
            },
        }
        manifest["manifest_digest"] = sha256_json(manifest)
        _atomic_write(self.root / "capture-manifest.json", canonical_json_bytes(manifest) + b"\n")
        _atomic_write(self.root / "COMPLETE", b"complete\n")
        (self.root / "INCOMPLETE").unlink(missing_ok=True)
        return manifest


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    root: Path
    manifest: dict[str, Any]
    initial_state: ToyState
    events: tuple[DomainEvent, ...]
    expected_step_digests: tuple[dict[str, Any], ...]

    @classmethod
    def open(cls, root: Path) -> "CaptureBundle":
        root = root.resolve()
        if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
            raise IllmError(
                ErrorCode.CAPTURE_INCOMPLETE,
                "capture is not finalized",
                {"path": str(root)},
            )
        manifest_path = root / "capture-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture manifest is unreadable") from exc
        if manifest.get("schema_version") != 0:
            raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported capture schema")
        stored_digest = manifest.pop("manifest_digest", None)
        actual_digest = sha256_json(manifest)
        manifest["manifest_digest"] = stored_digest
        if not manifest.get("finalized") or stored_digest != actual_digest:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture manifest digest mismatch")
        if manifest.get("critical_event_drop_count") != 0 or manifest.get("non_replayable_reasons"):
            raise IllmError(
                ErrorCode.CAPTURE_INCOMPLETE,
                "capture declares replay-critical loss",
                {"reasons": manifest.get("non_replayable_reasons", [])},
            )
        raw_files = manifest.get("files")
        if not isinstance(raw_files, dict) or set(raw_files) != set(MANIFEST_FACT_FILES):
            raise IllmError(
                ErrorCode.CAPTURE_CORRUPT,
                "capture manifest does not cover the exact v0 fact set",
                {
                    "expected": sorted(MANIFEST_FACT_FILES),
                    "actual": sorted(raw_files) if isinstance(raw_files, dict) else None,
                },
            )
        expected_files = dict(raw_files)
        for name, expected in expected_files.items():
            if not isinstance(expected, str) or len(expected) != SHA256_HEX_LENGTH or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "capture manifest contains an invalid checksum",
                    {"file": name},
                )
            path = root / name
            if not path.is_file() or sha256_file(path) != expected:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "capture checksum mismatch",
                    {"file": name},
                )
        try:
            checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
            checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
            event_values = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            digest_values = [
                json.loads(line)
                for line in (root / "state-digests.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture fact file is unreadable") from exc
        if not isinstance(checksums, dict) or checksums != {
            name: expected_files[name] for name in FACT_FILES
        }:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "checksums.json disagrees with capture manifest")
        events = tuple(DomainEvent.from_dict(value) for value in event_values)
        producer_sequences: dict[str, int] = {}
        producer_timestamps: dict[str, int] = {}
        for event in events:
            expected_seq = producer_sequences.get(event.producer, 0)
            if event.seq_no != expected_seq:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "producer sequence gap",
                    {"producer": event.producer, "expected": expected_seq, "actual": event.seq_no},
                )
            previous_timestamp = producer_timestamps.get(event.producer, -1)
            if event.monotonic_ns < previous_timestamp:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "producer monotonic timestamp moved backwards",
                    {"producer": event.producer, "seq_no": event.seq_no},
                )
            producer_sequences[event.producer] = expected_seq + 1
            producer_timestamps[event.producer] = event.monotonic_ns
        if len(events) != int(manifest.get("event_count", -1)):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "event count mismatch")
        steps = manifest.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "capture step count is invalid")
        if len(digest_values) != steps:
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "state digest record count mismatch")
        for expected_step, record in enumerate(digest_values):
            if not isinstance(record, dict) or record.get("step_id") != expected_step:
                raise IllmError(
                    ErrorCode.CAPTURE_CORRUPT,
                    "state digest steps are not contiguous",
                    {"expected": expected_step, "actual": record.get("step_id") if isinstance(record, dict) else None},
                )
            digest = record.get("state_digest")
            if not isinstance(digest, str) or len(digest) != SHA256_HEX_LENGTH or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "state digest record is invalid")
            if not isinstance(record.get("state"), dict):
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "state digest record lacks structural state")
        if digest_values[-1]["state_digest"] != manifest.get("final_state_digest"):
            raise IllmError(ErrorCode.CAPTURE_CORRUPT, "final state digest is not bound to step records")
        return cls(root, manifest, _state_from_dict(checkpoint), events, tuple(digest_values))

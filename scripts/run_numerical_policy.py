"""Run the approved CPU-only slice of the V0.2 numerical-policy protocol.

The driver is intentionally a development gate, not a release qualifier.  It
must itself run through ``scripts/in_env.py`` with a source-bound finalized
manifest.  Environment freshness is an external orchestration gate and is not
claimed by this driver.
GPU stages are reserved by the frozen CLI but are rejected before any project,
Torch, or model module can be imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, BinaryIO, Mapping, NoReturn, Sequence, cast
import uuid
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_COMMIT = "04144ab9ddb8d86c482ad6db8a618bded1f1e7aa"
PROTOCOL_ID = "v02-numerical-policy-protocol-draft-1"
PROTOCOL_SHA256 = "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"
APPROVAL_SHA256 = "2046ec2425b57d050ff61d7ad636e747b42eb6b08bd88a6883ae2188fae2ab7f"
INPUT_BYTES_SHA256 = "93d0de1c5c606a58830eeef2d6b95f4a2b389b4defd17735507d2e95b9adb754"
SPEC_DRAFT_SHA256 = "73a05a4f0e5e338d40ff406279695ef9a9596e2c55471bb8ffc8043493350334"
ADR_DRAFT_SHA256 = "33133ac0a5d888e3175be9e2de1cdd1f2aa180e71d93d6591641af9023004c77"
GPU_STAGES = frozenset({"gpu-pilot", "gpu-correctness", "gpu-benchmarks"})
CPU_STAGES = ("inputs", "cpu", "verify")
PENDING_WORK = (
    "capture manifest schema3 writer and reader",
    "formal portable release qualification",
    "all GPU pilot, correctness, benchmark, lifecycle, and rollback evidence",
)

_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "approval_kind",
        "approved_commit",
        "approval_date",
        "protocol_id",
        "protocol_sha256",
        "spec_draft_sha256",
        "adr_draft_sha256",
        "input_bytes_sha256",
        "authorized_stages",
        "gpu_authorized",
        "gpu_budget_status",
        "human_answer",
    }
)
_APPROVAL_EXPECTED: Mapping[str, object] = {
    "schema_version": 1,
    "approval_kind": "illm-human-policy-approval-v1",
    "approved_commit": APPROVED_COMMIT,
    "approval_date": "2026-10-21",
    "protocol_id": PROTOCOL_ID,
    "protocol_sha256": PROTOCOL_SHA256,
    "spec_draft_sha256": SPEC_DRAFT_SHA256,
    "adr_draft_sha256": ADR_DRAFT_SHA256,
    "input_bytes_sha256": INPUT_BYTES_SHA256,
    "authorized_stages": ["inputs", "cpu", "verify"],
    "gpu_authorized": False,
    "gpu_budget_status": "unreconciled",
    "human_answer": "批准规范与 CPU 实现，GPU 暂缓（推荐）",
}


class DriverFailure(Exception):
    """A fail-closed protocol-driver error with a stable machine code."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"code": self.code, "message": self.message, "status": "fail"}
        if self.details:
            result["details"] = self.details
        return result


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise DriverFailure("INVALID_ARGUMENT", message)


@dataclass(frozen=True, slots=True)
class ToolchainContext:
    project_root: Path
    environment_root: Path
    python_executable: Path
    environment_id: str
    lock_digest: str
    manifest_digest: str
    project_root_digest: str
    python_version: str
    runtime_profile_id: str | None

    def portable_identity(self) -> dict[str, object]:
        return {
            "environment_id": self.environment_id,
            "lock_digest": self.lock_digest,
            "manifest_digest": self.manifest_digest,
            "project_root_digest": self.project_root_digest,
            "python_version": self.python_version,
            "runtime_profile_id": self.runtime_profile_id,
        }


@dataclass(frozen=True, slots=True)
class InputMaterial:
    source: Path
    input_manifest_digest: str
    suite_digests: tuple[str, str]
    case_counts: tuple[int, int]
    policy_bytes: Mapping[str, bytes]
    policy_digests: Mapping[str, str]


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DriverFailure("INTERNAL_ERROR", "value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DriverFailure("EVIDENCE_INVALID", "evidence file is unreadable", path=path.name) from exc
    return digest.hexdigest()


def _unique_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DriverFailure("AUTHORITY_MISMATCH", f"{label} contains a duplicate field", field=key)
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} contains a non-finite number", value=value)

    try:
        parsed = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except DriverFailure:
        raise
    except (TypeError, ValueError) as exc:
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} must contain an object")
    return cast(dict[str, object], parsed)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_exclusive(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DriverFailure("PATH_NOT_NEW", "refusing to overwrite an existing artifact", path=path.name) from exc
    except OSError as exc:
        raise DriverFailure("EVIDENCE_WRITE_FAILED", "could not write evidence artifact", path=path.name) from exc


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _argument_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _ensure_plain_chain(path: Path, stop: Path) -> None:
    current = path
    while True:
        if current.exists() and _is_linklike(current):
            raise DriverFailure("PATH_ESCAPE", "path contains a symlink or reparse point", path=current.name)
        if current == stop:
            return
        if stop not in current.parents:
            raise DriverFailure("PATH_ESCAPE", "path escapes workspace out/")
        current = current.parent


def _under_out(path: Path, project_root: Path, *, label: str) -> Path:
    out_argument = project_root / "out"
    if not out_argument.is_dir() or _is_linklike(out_argument):
        raise DriverFailure("PATH_ESCAPE", "workspace out/ must be an ordinary directory")
    out_root = out_argument.resolve()
    raw = _argument_path(path, project_root)
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(out_root)
    except ValueError as exc:
        raise DriverFailure("PATH_ESCAPE", f"--{label} must be strictly inside workspace out/") from exc
    if resolved == out_root:
        raise DriverFailure("PATH_ESCAPE", f"--{label} cannot be workspace out/ itself")
    _ensure_plain_chain(raw.absolute().parent, out_root)
    if raw.exists() and _is_linklike(raw):
        raise DriverFailure("PATH_ESCAPE", f"--{label} cannot be a symlink or reparse point")
    return resolved


def _regular_file_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or _is_linklike(path) or not path.is_file():
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} must be an ordinary file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DriverFailure("AUTHORITY_MISMATCH", f"{label} is unreadable") from exc


class _AttemptLock:
    def __init__(self, attempt: Path) -> None:
        self.path = attempt.with_name(f"{attempt.name}.lock")
        self._handle: BinaryIO | None = None
        self._identity: tuple[int, int] | None = None
        self._token = _canonical_bytes(
            {"kind": "illm-policy-attempt-lock-v1", "pid": os.getpid(), "token": uuid.uuid4().hex}
        ) + b"\n"

    def __enter__(self) -> _AttemptLock:
        try:
            self._handle = self.path.open("x+b")
            stat = os.fstat(self._handle.fileno())
            self._identity = (stat.st_dev, stat.st_ino)
            if self._handle.write(self._token) != len(self._token):
                raise OSError("short attempt-lock write")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except FileExistsError as exc:
            raise DriverFailure(
                "ATTEMPT_LOCKED",
                "attempt lock already exists; it is never cleared as stale or foreign",
                lock=self.path.name,
            ) from exc
        except OSError as exc:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
            cleanup_error: str | None = None
            try:
                if self._path_is_owned():
                    self.path.unlink()
            except OSError as cleanup:
                cleanup_error = type(cleanup).__name__
            raise DriverFailure(
                "ATTEMPT_LOCK_FAILED",
                "could not acquire and durably initialize the attempt lock",
                owned_lock_cleanup_error=cleanup_error,
            ) from exc
        return self

    def _path_is_owned(self) -> bool:
        if self._identity is None or not self.path.exists() or _is_linklike(self.path):
            return False
        try:
            stat = self.path.lstat()
        except OSError:
            return False
        return (stat.st_dev, stat.st_ino) == self._identity

    def release(self) -> None:
        """Remove exactly this live lock, failing closed on replacement or cleanup failure."""

        if self._handle is None:
            raise DriverFailure("ATTEMPT_LOCK_FAILED", "attempt lock handle is not live")
        try:
            self._handle.close()
        except OSError as exc:
            raise DriverFailure("ATTEMPT_LOCK_FAILED", "could not close the attempt lock") from exc
        finally:
            self._handle = None
        if not self._path_is_owned():
            raise DriverFailure(
                "ATTEMPT_LOCK_TAMPERED",
                "attempt lock path changed or disappeared; foreign state was not removed",
            )
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise DriverFailure("ATTEMPT_LOCK_FAILED", "could not re-read the owned attempt lock") from exc
        if raw != self._token:
            raise DriverFailure(
                "ATTEMPT_LOCK_TAMPERED",
                "attempt lock bytes changed; foreign state was not removed",
            )
        try:
            self.path.unlink()
        except OSError as exc:
            raise DriverFailure(
                "ATTEMPT_LOCK_FAILED",
                "owned attempt lock could not be removed; stage cannot report success",
            ) from exc

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("inputs", "cpu", "verify", "gpu-pilot", "gpu-correctness", "gpu-benchmarks"),
        required=True,
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_toolchain(project_root: Path) -> ToolchainContext:
    manifest_text = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    environment_id = os.environ.get("ILLM_ENVIRONMENT_ID")
    lock_identity = os.environ.get("ILLM_LOCK_DIGEST")
    if not manifest_text or not environment_id or not lock_identity:
        raise DriverFailure(
            "TOOLCHAIN_MISMATCH",
            "run this driver through scripts/in_env.py with a source-bound finalized manifest",
        )
    manifest_argument = Path(manifest_text)
    if manifest_argument.is_symlink() or _is_linklike(manifest_argument) or not manifest_argument.is_file():
        raise DriverFailure("TOOLCHAIN_MISMATCH", "toolchain manifest must be an ordinary file")
    manifest_path = manifest_argument.resolve()
    scripts = str(project_root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        runner = importlib.import_module("in_env")
        if Path(str(runner.__file__)).resolve() != (project_root / "scripts/in_env.py").resolve():
            raise DriverFailure("TOOLCHAIN_MISMATCH", "in_env verifier was imported from an unexpected source")
        manifest = runner._load_manifest(manifest_path)
        if manifest.get("schema_version") != 1:
            raise DriverFailure("TOOLCHAIN_MISMATCH", "policy stages require a runtime-bound schema1 manifest")
        profile_value = manifest.get("runtime_profile_id")
        if type(profile_value) is not str:
            raise DriverFailure("TOOLCHAIN_MISMATCH", "runtime-bound manifest has no profile identity")
        runtime_lock = runner.load_runtime_lock(project_root, profile_value)
        if runtime_lock.get("target", {}).get("torch_channel") != "cpu":
            raise DriverFailure("TOOLCHAIN_MISMATCH", "CPU policy stages require a CPU-only runtime profile")
        verified_root, environment_root = runner._verify_manifest(manifest, None)
    except Exception as exc:
        raise DriverFailure("TOOLCHAIN_MISMATCH", "finalized manifest verification failed") from exc
    if verified_root.resolve() != project_root.resolve():
        raise DriverFailure("TOOLCHAIN_MISMATCH", "manifest belongs to a different workspace")
    python = Path(str(manifest["python_executable"])).resolve()
    if os.path.normcase(str(Path(sys.executable).resolve())) != os.path.normcase(str(python)):
        raise DriverFailure("TOOLCHAIN_MISMATCH", "driver is not running under the finalized interpreter")
    if environment_id != manifest["environment_id"] or lock_identity != manifest["lock_digest"]:
        raise DriverFailure("TOOLCHAIN_MISMATCH", "in_env handshake variables differ from the manifest")
    runtime_profile = manifest.get("runtime_profile_id")
    if runtime_profile is not None and type(runtime_profile) is not str:
        raise DriverFailure("TOOLCHAIN_MISMATCH", "runtime profile identity is malformed")
    return ToolchainContext(
        project_root=verified_root.resolve(),
        environment_root=environment_root.resolve(),
        python_executable=python,
        environment_id=str(manifest["environment_id"]),
        lock_digest=str(manifest["lock_digest"]),
        manifest_digest=str(manifest["manifest_digest"]),
        project_root_digest=str(manifest["project_root_digest"]),
        python_version=str(manifest["python_version"]),
        runtime_profile_id=runtime_profile,
    )


def _confirm_toolchain(toolchain: ToolchainContext) -> None:
    confirmed = _load_toolchain(toolchain.project_root)
    if confirmed != toolchain:
        raise DriverFailure("TOOLCHAIN_MISMATCH", "toolchain identity changed during the stage")


def _validate_authorities(protocol: Path, approval: Path, project_root: Path) -> dict[str, object]:
    protocol_path = _argument_path(protocol, project_root)
    approval_path = _argument_path(approval, project_root)
    protocol_raw = _regular_file_bytes(protocol_path, "protocol")
    approval_raw = _regular_file_bytes(approval_path, "approval transcript")
    if _sha256_bytes(protocol_raw) != PROTOCOL_SHA256:
        raise DriverFailure("AUTHORITY_MISMATCH", "protocol raw bytes differ from the approved digest")
    if _sha256_bytes(approval_raw) != APPROVAL_SHA256:
        raise DriverFailure("AUTHORITY_MISMATCH", "approval transcript raw bytes differ from the received answer")
    approval_value = _unique_json(approval_raw, "approval transcript")
    if set(approval_value) != _APPROVAL_FIELDS or approval_value != _APPROVAL_EXPECTED:
        raise DriverFailure("AUTHORITY_MISMATCH", "approval transcript fields differ from the received human answer")
    frozen_files = {
        "spec draft": (project_root / "docs/implementation/v0.2-spec-v3-amendment-draft.md", SPEC_DRAFT_SHA256),
        "ADR draft": (project_root / "docs/implementation/v0.2-adr-007-draft.md", ADR_DRAFT_SHA256),
        "input draft": (project_root / "docs/implementation/v0.2-policy-inputs-draft.json", INPUT_BYTES_SHA256),
    }
    for label, (path, expected) in frozen_files.items():
        raw = _regular_file_bytes(path, label)
        if _sha256_bytes(raw) != expected:
            raise DriverFailure("AUTHORITY_MISMATCH", f"{label} raw bytes differ from approval")
    return approval_value


def _load_input_material(project_root: Path) -> InputMaterial:
    # These project modules are deliberately tensor-free and import only the
    # standard library plus other tensor-free contract modules.
    inputs_module = importlib.import_module("illm.policy_inputs")
    policy_module = importlib.import_module("illm.numerical_policy")
    source = project_root / "docs/implementation/v0.2-policy-inputs-draft.json"
    manifest = inputs_module.load_policy_inputs(source)
    policy_bytes: dict[str, bytes] = {}
    policy_digests: dict[str, str] = {}
    for dtype in ("bf16", "fp16", "fp32"):
        policy = policy_module.candidate_policy(dtype)
        raw = _canonical_bytes(policy.to_dict())
        if _sha256_bytes(raw) != policy.digest:
            raise DriverFailure("POLICY_IDENTITY_MISMATCH", "candidate descriptor digest drifted", dtype=dtype)
        policy_bytes[dtype] = raw
        policy_digests[dtype] = str(policy.digest)
    return InputMaterial(
        source=source,
        input_manifest_digest=str(manifest.manifest_digest),
        suite_digests=cast(tuple[str, str], tuple(manifest.suite_digests)),
        case_counts=cast(tuple[int, int], tuple(manifest.case_counts)),
        policy_bytes=policy_bytes,
        policy_digests=policy_digests,
    )


def _journal_projection(journal: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in journal.items() if key != "journal_digest"}


def _set_journal_digest(journal: dict[str, object]) -> None:
    journal["journal_digest"] = _sha256_bytes(_canonical_bytes(_journal_projection(journal)))


def _new_journal(toolchain: ToolchainContext, material: InputMaterial) -> dict[str, object]:
    journal: dict[str, object] = {
        "schema_version": 1,
        "journal_kind": "illm-policy-development-attempt-v1",
        "status": "in_progress",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "source_identity": {
            "approved_commit": APPROVED_COMMIT,
            "project_root_digest": toolchain.project_root_digest,
        },
        "protocol": {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256},
        "approval": {
            "approval_kind": "illm-human-policy-approval-v1",
            "sha256": APPROVAL_SHA256,
            "interpretation": "transcript_of_received_human_answer_not_self_authorization",
        },
        "toolchain": toolchain.portable_identity(),
        "input_identity": {
            "raw_bytes_sha256": INPUT_BYTES_SHA256,
            "manifest_digest": material.input_manifest_digest,
        },
        "candidate_policy_digests": dict(sorted(material.policy_digests.items())),
        "stages": [],
        "pending": list(PENDING_WORK),
    }
    _set_journal_digest(journal)
    return journal


def _write_new_journal(path: Path, journal: dict[str, object]) -> None:
    _write_exclusive(path, _canonical_bytes(journal) + b"\n")


def _load_journal(path: Path) -> dict[str, object]:
    raw = _regular_file_bytes(path, "attempt journal")
    journal = _unique_json(raw, "attempt journal")
    expected_fields = {
        "schema_version", "journal_kind", "status", "evidence_mode", "qualification_tier",
        "source_identity", "protocol", "approval", "toolchain", "input_identity",
        "candidate_policy_digests", "stages", "pending", "journal_digest",
    }
    if set(journal) != expected_fields:
        raise DriverFailure("JOURNAL_INVALID", "attempt journal fields differ from schema")
    reported = journal.get("journal_digest")
    computed = _sha256_bytes(_canonical_bytes(_journal_projection(journal)))
    if type(reported) is not str or reported != computed:
        raise DriverFailure("JOURNAL_INVALID", "attempt journal digest mismatch")
    if (
        journal.get("schema_version") != 1
        or journal.get("journal_kind") != "illm-policy-development-attempt-v1"
        or journal.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or journal.get("qualification_tier") != "unqualified"
        or type(journal.get("stages")) is not list
    ):
        raise DriverFailure("JOURNAL_INVALID", "attempt journal identity is invalid")
    return journal


def _validate_journal_identity(
    journal: Mapping[str, object], toolchain: ToolchainContext, material: InputMaterial
) -> list[dict[str, object]]:
    expected = _new_journal(toolchain, material)
    for field in (
        "schema_version", "journal_kind", "evidence_mode", "qualification_tier", "source_identity",
        "protocol", "approval", "toolchain", "input_identity", "candidate_policy_digests", "pending",
    ):
        if journal.get(field) != expected[field]:
            raise DriverFailure("JOURNAL_IDENTITY_DRIFT", "attempt source/protocol/policy/toolchain identity drifted", field=field)
    stages_value = journal.get("stages")
    if type(stages_value) is not list or any(not isinstance(item, dict) for item in stages_value):
        raise DriverFailure("JOURNAL_INVALID", "attempt stages must be objects")
    stages = cast(list[dict[str, object]], stages_value)
    if len(stages) > len(CPU_STAGES):
        raise DriverFailure("JOURNAL_INVALID", "attempt contains too many stages")
    stage_fields = {
        "sequence", "stage", "depends_on", "result", "evidence_mode", "qualification_tier",
        "output_root", "files_digest", "checksums_sha256",
    }
    for index, record in enumerate(stages):
        stage = CPU_STAGES[index]
        root = record.get("output_root")
        digests = (record.get("files_digest"), record.get("checksums_sha256"))
        if (
            set(record) != stage_fields
            or record.get("sequence") != index + 1
            or record.get("stage") != stage
            or record.get("depends_on") != list(CPU_STAGES[:index])
            or record.get("result") != "pass"
            or record.get("evidence_mode") != "DEVELOPMENT_ONLY"
            or record.get("qualification_tier") != "unqualified"
            or type(root) is not str
            or Path(root).is_absolute()
            or not root.startswith("out/")
            or ".." in Path(root).parts
            or any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                   for value in digests)
        ):
            raise DriverFailure("JOURNAL_INVALID", "attempt stage record differs from the ordered schema", stage=stage)
    expected_status = "verified_development_only_unqualified" if len(stages) == 3 else "in_progress"
    if journal.get("status") != expected_status:
        raise DriverFailure("JOURNAL_INVALID", "attempt status differs from its completed stage prefix")
    return stages


def _expected_prior_stages(stage: str) -> list[str]:
    if stage == "inputs":
        return []
    if stage == "cpu":
        return ["inputs"]
    if stage == "verify":
        return ["inputs", "cpu"]
    raise DriverFailure("INVALID_ARGUMENT", "unsupported CPU stage")


def _begin_output(output: Path, stage: str) -> None:
    if output.exists() or output.is_symlink():
        raise DriverFailure("PATH_NOT_NEW", "--output must be a new directory")
    created = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        _write_exclusive(
            output / "INCOMPLETE",
            _canonical_bytes({"kind": "illm-policy-stage-marker-v1", "stage": stage, "status": "in_progress"}) + b"\n",
        )
    except BaseException as exc:
        if created:
            # This directory was created by this call.  If durable marker
            # initialization fails, remove only its own partial marker and
            # empty directory rather than leaving an unmarked output root.
            try:
                entries = list(output.iterdir())
                if all(item.name == "INCOMPLETE" and not item.is_dir() for item in entries):
                    for item in entries:
                        item.unlink()
                    output.rmdir()
            except OSError:
                pass
        if isinstance(exc, DriverFailure):
            raise
        raise DriverFailure("EVIDENCE_WRITE_FAILED", "could not create stage output") from exc


def _mark_failure(output: Path, stage: str, failure: DriverFailure) -> None:
    complete = output / "COMPLETE"
    if complete.exists():
        complete.unlink(missing_ok=True)
    if output.is_dir():
        _atomic_json(
            output / "INCOMPLETE",
            {"kind": "illm-policy-stage-marker-v1", "stage": stage, "status": "failed", "error_code": failure.code},
        )


def _evidence_files(output: Path) -> list[Path]:
    excluded = {"INCOMPLETE", "COMPLETE", "checksums.json"}
    files: list[Path] = []
    for path in sorted(output.rglob("*"), key=lambda item: item.as_posix()):
        if path.name in excluded and path.parent == output:
            continue
        if path.is_symlink() or _is_linklike(path):
            raise DriverFailure("EVIDENCE_INVALID", "stage output contains a link", path=path.name)
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise DriverFailure("EVIDENCE_INVALID", "stage output contains a non-regular entry", path=path.name)
    return files


def _inventory(output: Path, stage: str) -> dict[str, object]:
    entries = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _evidence_files(output)
    ]
    return {
        "schema_version": 1,
        "kind": "illm-policy-stage-checksums-v1",
        "stage": stage,
        "files": entries,
        "files_digest": _sha256_bytes(_canonical_bytes(entries)),
    }


def _seal_output(output: Path, stage: str) -> dict[str, object]:
    inventory = _inventory(output, stage)
    _write_exclusive(output / "checksums.json", _canonical_bytes(inventory) + b"\n")
    marker = {
        "kind": "illm-policy-stage-complete-v1",
        "stage": stage,
        "files_digest": inventory["files_digest"],
    }
    _write_exclusive(output / "COMPLETE", _canonical_bytes(marker) + b"\n")
    try:
        (output / "INCOMPLETE").unlink()
    except OSError as exc:
        (output / "COMPLETE").unlink(missing_ok=True)
        raise DriverFailure("EVIDENCE_WRITE_FAILED", "could not finalize stage marker") from exc
    return inventory


def _verify_sealed_output(
    root: Path, expected_stage: str, expected_digest: str, project_root: Path
) -> dict[str, object]:
    checked = _under_out(root, project_root, label="journal-stage-root")
    if not checked.is_dir() or _is_linklike(checked):
        raise DriverFailure("EVIDENCE_INVALID", "journal stage root is not an ordinary directory")
    if (checked / "INCOMPLETE").exists():
        raise DriverFailure("EVIDENCE_INVALID", "journal stage root is incomplete", stage=expected_stage)
    checksums = _unique_json(_regular_file_bytes(checked / "checksums.json", "checksums"), "checksums")
    marker = _unique_json(_regular_file_bytes(checked / "COMPLETE", "completion marker"), "completion marker")
    if (
        checksums.get("schema_version") != 1
        or checksums.get("kind") != "illm-policy-stage-checksums-v1"
        or checksums.get("stage") != expected_stage
        or checksums.get("files_digest") != expected_digest
        or marker != {
            "kind": "illm-policy-stage-complete-v1",
            "stage": expected_stage,
            "files_digest": expected_digest,
        }
    ):
        raise DriverFailure("EVIDENCE_INVALID", "sealed stage identity differs from journal", stage=expected_stage)
    actual = _inventory(checked, expected_stage)
    if checksums != actual:
        raise DriverFailure("EVIDENCE_INVALID", "sealed stage files or digests changed", stage=expected_stage)
    return checksums


def _portable_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise DriverFailure("PATH_ESCAPE", "portable evidence path escapes workspace") from exc


def _stage_record(stage: str, output: Path, inventory: Mapping[str, object], project_root: Path) -> dict[str, object]:
    return {
        "sequence": CPU_STAGES.index(stage) + 1,
        "stage": stage,
        "depends_on": _expected_prior_stages(stage),
        "result": "pass",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "output_root": _portable_relative(output, project_root),
        "files_digest": inventory["files_digest"],
        "checksums_sha256": _sha256_file(output / "checksums.json"),
    }


def _inputs_report(material: InputMaterial, toolchain: ToolchainContext) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "illm-policy-inputs-development-snapshot-v1",
        "status": "pass",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "qualification_result": "not_evaluated",
        "approved_commit": APPROVED_COMMIT,
        "protocol": {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256},
        "input": {
            "raw_bytes_sha256": INPUT_BYTES_SHA256,
            "manifest_digest": material.input_manifest_digest,
            "suite_digests": list(material.suite_digests),
            "case_counts": list(material.case_counts),
            "snapshot": "policy-inputs.json",
        },
        "candidate_policy_digests": dict(sorted(material.policy_digests.items())),
        "toolchain": toolchain.portable_identity(),
        "limitations": list(PENDING_WORK),
    }


def _run_inputs(output: Path, material: InputMaterial, toolchain: ToolchainContext) -> None:
    raw = material.source.read_bytes()
    if _sha256_bytes(raw) != INPUT_BYTES_SHA256:
        raise DriverFailure("AUTHORITY_MISMATCH", "input proposal changed after validation")
    _write_exclusive(output / "policy-inputs.json", raw)
    policies = output / "candidate-policies"
    policies.mkdir()
    for dtype, descriptor in sorted(material.policy_bytes.items()):
        _write_exclusive(policies / f"{dtype}.json", descriptor)
    report = _inputs_report(material, toolchain)
    _atomic_json(output / "inputs-report.json", report)


def _portable_argv(command: Sequence[str], project_root: Path, python: Path) -> list[str]:
    portable: list[str] = []
    for index, value in enumerate(command):
        if index == 0 and os.path.normcase(str(Path(value).resolve())) == os.path.normcase(str(python.resolve())):
            portable.append("{finalized_python}")
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                portable.append(candidate.resolve().relative_to(project_root.resolve()).as_posix())
            except ValueError:
                portable.append("{external_path_redacted}")
        else:
            portable.append(value.replace("\\", "/"))
    return portable


def _execute_phase(
    name: str,
    command: list[str],
    *,
    project_root: Path,
    output: Path,
    environment: Mapping[str, str],
    python: Path,
) -> dict[str, object]:
    process = subprocess.run(
        command,
        cwd=project_root,
        env=dict(environment),
        capture_output=True,
        check=False,
        shell=False,
    )
    logs = output / "logs"
    stdout_path = logs / f"{name}.stdout.txt"
    stderr_path = logs / f"{name}.stderr.txt"
    _write_exclusive(stdout_path, process.stdout)
    _write_exclusive(stderr_path, process.stderr)
    return {
        "phase": name,
        "argv": _portable_argv(command, project_root, python),
        "returncode": process.returncode,
        "stdout_file": stdout_path.relative_to(output).as_posix(),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_file": stderr_path.relative_to(output).as_posix(),
        "stderr_sha256": _sha256_file(stderr_path),
    }


def _run_cpu(output: Path, toolchain: ToolchainContext, material: InputMaterial) -> None:
    logs = output / "logs"
    logs.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    python = str(toolchain.python_executable)
    commands = (
        (
            "quality",
            [python, "scripts/check_quality.py", "--output", str(output / "quality.json")],
        ),
        (
            "pytest",
            [
                python,
                "-m",
                "pytest",
                "tests",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(output / "pytest-temp"),
                "--junitxml",
                str(output / "junit.xml"),
                "--tb=short",
            ],
        ),
        ("example", [python, "docs/examples/v0_2_policy_cpu.py"]),
    )
    records: list[dict[str, object]] = []
    for name, command in commands:
        record = _execute_phase(
            name,
            command,
            project_root=toolchain.project_root,
            output=output,
            environment=environment,
            python=toolchain.python_executable,
        )
        records.append(record)
        _atomic_json(
            output / "execution.json",
            {
                "schema_version": 1,
                "kind": "illm-policy-cpu-development-execution-v1",
                "commands": records,
            },
        )
        if record["returncode"] != 0:
            raise DriverFailure("CPU_PHASE_FAILED", "CPU development phase failed", phase=name)
    required = (output / "quality.json", output / "junit.xml")
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise DriverFailure("CPU_PHASE_FAILED", "CPU phase returned without required quality/JUnit evidence")
    # Pytest's base directory is scratch space, not evidence.  It can contain
    # intentional symlink/fault fixtures, so remove only this exact driver-owned
    # subtree after the suite has passed; on failure it remains for diagnosis.
    pytest_temp = output / "pytest-temp"
    if pytest_temp.exists() or pytest_temp.is_symlink():
        if pytest_temp.is_symlink() or _is_linklike(pytest_temp) or pytest_temp.resolve().parent != output.resolve():
            raise DriverFailure("CPU_PHASE_FAILED", "pytest scratch root is not the expected ordinary subtree")
        try:
            shutil.rmtree(pytest_temp)
        except OSError as exc:
            raise DriverFailure("CPU_PHASE_FAILED", "could not remove driver-owned pytest scratch data") from exc
    report = {
        "schema_version": 1,
        "kind": "illm-policy-cpu-development-gate-v1",
        "status": "pass",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "qualification_result": "not_evaluated",
        "performance_outcome": "not_evaluated",
        "gpu_result": "not_run",
        "raw_tensor_evidence": "not_captured",
        "approved_commit": APPROVED_COMMIT,
        "protocol": {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256},
        "input_manifest_digest": material.input_manifest_digest,
        "candidate_policy_digests": dict(sorted(material.policy_digests.items())),
        "toolchain": toolchain.portable_identity(),
        "commands": records,
        "required_artifacts": {
            "quality": {"path": "quality.json", "sha256": _sha256_file(output / "quality.json")},
            "junit": {"path": "junit.xml", "sha256": _sha256_file(output / "junit.xml")},
        },
        "limitations": list(PENDING_WORK),
    }
    _atomic_json(output / "cpu-report.json", report)
    _validate_cpu_stage(output, toolchain, material)


def _validate_inputs_stage(
    root: Path, material: InputMaterial, toolchain: ToolchainContext
) -> None:
    if _sha256_file(root / "policy-inputs.json") != INPUT_BYTES_SHA256:
        raise DriverFailure("EVIDENCE_INVALID", "input snapshot is not byte-identical to authority")
    for dtype, expected in material.policy_digests.items():
        if _sha256_file(root / "candidate-policies" / f"{dtype}.json") != expected:
            raise DriverFailure("EVIDENCE_INVALID", "candidate policy descriptor changed", dtype=dtype)
    report = _unique_json(_regular_file_bytes(root / "inputs-report.json", "inputs report"), "inputs report")
    if report != _inputs_report(material, toolchain):
        raise DriverFailure(
            "EVIDENCE_INVALID",
            "inputs report identity, result axes, source material, or toolchain differs",
        )


def _evidence_json(path: Path, label: str) -> dict[str, object]:
    try:
        return _unique_json(_regular_file_bytes(path, label), label)
    except DriverFailure as exc:
        raise DriverFailure("EVIDENCE_INVALID", f"{label} is not a strict ordinary JSON object") from exc


def _validate_quality(path: Path) -> dict[str, object]:
    quality = _evidence_json(path, "quality report")
    if set(quality) != {"schema_version", "checked_python_files", "issues", "type_check", "status"}:
        raise DriverFailure("EVIDENCE_INVALID", "quality report fields differ from check_quality schema")
    checked = quality.get("checked_python_files")
    if (
        type(quality.get("schema_version")) is not int
        or quality.get("schema_version") != 0
        or type(checked) is not int
        or checked <= 0
        or quality.get("issues") != []
        or quality.get("status") != "pass"
    ):
        raise DriverFailure("EVIDENCE_INVALID", "quality report does not prove a passing nonempty source check")
    type_check = quality.get("type_check")
    if not isinstance(type_check, dict) or set(type_check) != {"tool", "version", "returncode", "status", "output"}:
        raise DriverFailure("EVIDENCE_INVALID", "quality report type-check fields differ from schema")
    version = type_check.get("version")
    output = type_check.get("output")
    if (
        type_check.get("tool") != "mypy"
        or type(version) is not str
        or not version
        or type(type_check.get("returncode")) is not int
        or type_check.get("returncode") != 0
        or type_check.get("status") != "pass"
        or type(output) is not list
        or any(type(line) is not str for line in output)
    ):
        raise DriverFailure("EVIDENCE_INVALID", "quality report does not prove a passing pinned mypy run")
    return quality


def _junit_integer(suite: ET.Element, field: str) -> int:
    raw = suite.attrib.get(field)
    if raw is None or not raw.isdecimal():
        raise DriverFailure("EVIDENCE_INVALID", "JUnit suite counter is absent or non-decimal", field=field)
    value = int(raw)
    if value < 0:
        raise DriverFailure("EVIDENCE_INVALID", "JUnit suite counter is negative", field=field)
    return value


def _validate_junit(path: Path) -> dict[str, int]:
    raw = _regular_file_bytes(path, "JUnit report")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise DriverFailure("EVIDENCE_INVALID", "JUnit report cannot declare a document type or entity")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise DriverFailure("EVIDENCE_INVALID", "JUnit report is not well-formed XML") from exc
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [child for child in list(root) if child.tag == "testsuite"]
        if len(suites) != len(list(root)):
            raise DriverFailure("EVIDENCE_INVALID", "JUnit testsuites root contains an unexpected child")
    else:
        raise DriverFailure("EVIDENCE_INVALID", "JUnit root must be testsuite or testsuites")
    if not suites:
        raise DriverFailure("EVIDENCE_INVALID", "JUnit report contains no suites")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    testcase_total = 0
    for suite in suites:
        cases = [child for child in list(suite) if child.tag == "testcase"]
        counters = {field: _junit_integer(suite, field) for field in totals}
        try:
            duration = float(suite.attrib.get("time", "0"))
        except ValueError as exc:
            raise DriverFailure("EVIDENCE_INVALID", "JUnit suite duration is invalid") from exc
        if not math.isfinite(duration) or duration < 0:
            raise DriverFailure("EVIDENCE_INVALID", "JUnit suite duration is non-finite or negative")
        if counters["tests"] != len(cases):
            raise DriverFailure("EVIDENCE_INVALID", "JUnit suite test count differs from testcase elements")
        observed = {"failures": 0, "errors": 0, "skipped": 0}
        for case in cases:
            if not case.attrib.get("name") or not case.attrib.get("classname"):
                raise DriverFailure("EVIDENCE_INVALID", "JUnit testcase lacks name or classname")
            for child in list(case):
                if child.tag == "failure":
                    observed["failures"] += 1
                elif child.tag == "error":
                    observed["errors"] += 1
                elif child.tag == "skipped":
                    observed["skipped"] += 1
        if any(counters[field] != observed[field] for field in observed):
            raise DriverFailure("EVIDENCE_INVALID", "JUnit suite counters differ from testcase outcomes")
        for field, value in counters.items():
            totals[field] += value
        testcase_total += len(cases)
    if testcase_total == 0:
        raise DriverFailure("EVIDENCE_INVALID", "JUnit report contains no testcase evidence")
    if (
        totals["tests"] != testcase_total
        or totals["tests"] <= totals["skipped"]
        or totals["failures"] != 0
        or totals["errors"] != 0
    ):
        raise DriverFailure("EVIDENCE_INVALID", "JUnit report does not prove a passing accounted test suite")
    return totals


def _expected_cpu_commands(root: Path, project_root: Path) -> list[tuple[str, list[str]]]:
    relative = _portable_relative(root, project_root)
    return [
        (
            "quality",
            ["{finalized_python}", "scripts/check_quality.py", "--output", f"{relative}/quality.json"],
        ),
        (
            "pytest",
            [
                "{finalized_python}", "-m", "pytest", "tests", "-p", "no:cacheprovider",
                "--basetemp", f"{relative}/pytest-temp", "--junitxml", f"{relative}/junit.xml", "--tb=short",
            ],
        ),
        ("example", ["{finalized_python}", "docs/examples/v0_2_policy_cpu.py"]),
    ]


def _validate_execution(root: Path, project_root: Path) -> list[dict[str, object]]:
    execution = _evidence_json(root / "execution.json", "CPU execution report")
    if set(execution) != {"schema_version", "kind", "commands"} or (
        type(execution.get("schema_version")) is not int
        or execution.get("schema_version") != 1
        or execution.get("kind") != "illm-policy-cpu-development-execution-v1"
    ):
        raise DriverFailure("EVIDENCE_INVALID", "CPU execution report identity differs from schema")
    commands_value = execution.get("commands")
    if type(commands_value) is not list or any(not isinstance(item, dict) for item in commands_value):
        raise DriverFailure("EVIDENCE_INVALID", "CPU execution commands must be an object list")
    commands = cast(list[dict[str, object]], commands_value)
    expected = _expected_cpu_commands(root, project_root)
    if len(commands) != len(expected):
        raise DriverFailure("EVIDENCE_INVALID", "CPU execution must contain exactly three ordered commands")
    fields = {
        "phase", "argv", "returncode", "stdout_file", "stdout_sha256", "stderr_file", "stderr_sha256",
    }
    for record, (phase, argv) in zip(commands, expected, strict=True):
        stdout_name = f"logs/{phase}.stdout.txt"
        stderr_name = f"logs/{phase}.stderr.txt"
        stdout = root / stdout_name
        stderr = root / stderr_name
        if (
            set(record) != fields
            or record.get("phase") != phase
            or record.get("argv") != argv
            or type(record.get("returncode")) is not int
            or record.get("returncode") != 0
            or record.get("stdout_file") != stdout_name
            or record.get("stderr_file") != stderr_name
            or record.get("stdout_sha256") != _sha256_file(stdout)
            or record.get("stderr_sha256") != _sha256_file(stderr)
        ):
            raise DriverFailure("EVIDENCE_INVALID", "CPU command transcript or log digest differs", phase=phase)
    return commands


def _validate_cpu_stage(root: Path, toolchain: ToolchainContext, material: InputMaterial) -> None:
    for path in (root / "quality.json", root / "junit.xml", root / "execution.json", root / "cpu-report.json"):
        if path.is_symlink() or not path.is_file():
            raise DriverFailure("EVIDENCE_INVALID", "CPU stage is missing required evidence", path=path.name)
    quality = _validate_quality(root / "quality.json")
    _validate_junit(root / "junit.xml")
    commands = _validate_execution(root, toolchain.project_root)
    quality_stdout = _evidence_json(root / "logs/quality.stdout.txt", "quality stdout")
    if quality_stdout != quality:
        raise DriverFailure("EVIDENCE_INVALID", "quality stdout and quality artifact differ")
    report = _evidence_json(root / "cpu-report.json", "CPU report")
    report_fields = {
        "schema_version", "kind", "status", "evidence_mode", "qualification_tier", "qualification_result",
        "performance_outcome", "gpu_result", "raw_tensor_evidence", "approved_commit", "protocol",
        "input_manifest_digest", "candidate_policy_digests", "toolchain", "commands", "required_artifacts",
        "limitations",
    }
    required = {
        "quality": {"path": "quality.json", "sha256": _sha256_file(root / "quality.json")},
        "junit": {"path": "junit.xml", "sha256": _sha256_file(root / "junit.xml")},
    }
    if (
        set(report) != report_fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "illm-policy-cpu-development-gate-v1"
        or report.get("status") != "pass"
        or report.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or report.get("qualification_tier") != "unqualified"
        or report.get("qualification_result") != "not_evaluated"
        or report.get("performance_outcome") != "not_evaluated"
        or report.get("gpu_result") != "not_run"
        or report.get("raw_tensor_evidence") != "not_captured"
        or report.get("approved_commit") != APPROVED_COMMIT
        or report.get("protocol") != {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256}
        or report.get("input_manifest_digest") != material.input_manifest_digest
        or report.get("candidate_policy_digests") != dict(sorted(material.policy_digests.items()))
        or report.get("toolchain") != toolchain.portable_identity()
        or report.get("commands") != commands
        or report.get("required_artifacts") != required
        or report.get("limitations") != list(PENDING_WORK)
    ):
        raise DriverFailure("EVIDENCE_INVALID", "CPU report identity, result axes, or linked evidence differs")


def _run_verify(
    output: Path,
    stages: Sequence[Mapping[str, object]],
    toolchain: ToolchainContext,
    material: InputMaterial,
) -> None:
    verified: list[dict[str, object]] = []
    for record in stages:
        stage = record.get("stage")
        root_value = record.get("output_root")
        digest = record.get("files_digest")
        if type(stage) is not str or type(root_value) is not str or type(digest) is not str:
            raise DriverFailure("JOURNAL_INVALID", "stage record identity is malformed")
        root = toolchain.project_root / root_value
        checksums = _verify_sealed_output(root, stage, digest, toolchain.project_root)
        if record.get("checksums_sha256") != _sha256_file(root / "checksums.json"):
            raise DriverFailure("EVIDENCE_INVALID", "journal checksum-file identity changed", stage=stage)
        if stage == "inputs":
            _validate_inputs_stage(root, material, toolchain)
        elif stage == "cpu":
            _validate_cpu_stage(root, toolchain, material)
        else:
            raise DriverFailure("JOURNAL_INVALID", "verify encountered an unexpected prior stage", stage=stage)
        files = checksums.get("files")
        verified.append(
            {
                "stage": stage,
                "output_root": root_value,
                "files_digest": digest,
                "file_count": len(files) if isinstance(files, list) else None,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "illm-policy-cpu-development-verification-v1",
        "status": "pass",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "qualification_result": "not_evaluated",
        "performance_outcome": "not_evaluated",
        "verified_stages": verified,
        "approved_commit": APPROVED_COMMIT,
        "protocol": {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256},
        "input_manifest_digest": material.input_manifest_digest,
        "candidate_policy_digests": dict(sorted(material.policy_digests.items())),
        "toolchain": toolchain.portable_identity(),
        "limitations": list(PENDING_WORK),
    }
    _atomic_json(output / "verification-report.json", report)


def run_stage(argv: Sequence[str] | None = None) -> int:
    """Execute one authorized development stage or fail closed."""

    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    # This branch must stay before toolchain loading, authority reads, or any
    # project import.  GPU authorization is false and the budget is unknown.
    if args.stage in GPU_STAGES:
        raise DriverFailure(
            "GPU_NOT_AUTHORIZED",
            "GPU stages are unconditionally disabled: human approval deferred GPU work and budget is unreconciled",
            stage=args.stage,
        )

    project_root = PROJECT_ROOT.resolve()
    attempt = _under_out(args.attempt, project_root, label="attempt")
    output = _under_out(args.output, project_root, label="output")
    if output.exists() or output.is_symlink():
        raise DriverFailure("PATH_NOT_NEW", "--output must be a new directory")
    if attempt == output or attempt in output.parents or output in attempt.parents:
        raise DriverFailure("PATH_CONFLICT", "--attempt and --output must be separate sibling paths")

    output_started = False
    rollback_journal: dict[str, object] | None = None
    try:
        with _AttemptLock(attempt):
            toolchain = _load_toolchain(project_root)
            _validate_authorities(args.protocol, args.approval, project_root)
            material = _load_input_material(project_root)
            if args.stage == "inputs":
                if attempt.exists() or attempt.is_symlink():
                    raise DriverFailure("JOURNAL_ORDER", "inputs requires a new attempt journal")
                journal = _new_journal(toolchain, material)
                _write_new_journal(attempt, journal)
            else:
                if not attempt.is_file() or attempt.is_symlink() or _is_linklike(attempt):
                    raise DriverFailure("JOURNAL_ORDER", "later stages require the ordinary inputs attempt journal")
                journal = _load_journal(attempt)
            stages = _validate_journal_identity(journal, toolchain, material)
            observed = [record.get("stage") for record in stages]
            expected = _expected_prior_stages(args.stage)
            if observed != expected:
                raise DriverFailure(
                    "JOURNAL_ORDER",
                    "stage cannot be skipped, repeated, overwritten, or reordered",
                    expected=expected,
                    observed=observed,
                )
            rollback_journal = _unique_json(_canonical_bytes(journal), "attempt rollback snapshot")

            try:
                _begin_output(output, args.stage)
                output_started = True
                if args.stage == "inputs":
                    _run_inputs(output, material, toolchain)
                elif args.stage == "cpu":
                    _run_cpu(output, toolchain, material)
                elif args.stage == "verify":
                    _run_verify(output, stages, toolchain, material)
                else:  # pragma: no cover - argparse and the early GPU guard close this branch
                    raise DriverFailure("INVALID_ARGUMENT", "unsupported stage")
                _confirm_toolchain(toolchain)
                inventory = _seal_output(output, args.stage)
                stages.append(_stage_record(args.stage, output, inventory, project_root))
                journal["stages"] = stages
                if args.stage == "verify":
                    journal["status"] = "verified_development_only_unqualified"
                _set_journal_digest(journal)
                _atomic_json(attempt, journal)
            except DriverFailure as failure:
                _mark_failure(output, args.stage, failure)
                raise
            except Exception as exc:
                internal_failure = DriverFailure(
                    "STAGE_FAILED", "stage failed before completion", exception_type=type(exc).__name__
                )
                _mark_failure(output, args.stage, internal_failure)
                raise internal_failure from exc
    except DriverFailure as failure:
        if failure.code in {"ATTEMPT_LOCK_FAILED", "ATTEMPT_LOCK_TAMPERED"} and output_started:
            _mark_failure(output, args.stage, failure)
            if rollback_journal is not None:
                _set_journal_digest(rollback_journal)
                _atomic_json(attempt, rollback_journal)
        raise
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with stable exit classes."""

    try:
        return run_stage(argv)
    except DriverFailure as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        if exc.code == "GPU_NOT_AUTHORIZED":
            return 77
        if exc.code in {"INVALID_ARGUMENT", "PATH_ESCAPE", "PATH_CONFLICT", "PATH_NOT_NEW"}:
            return 64
        if exc.code == "TOOLCHAIN_MISMATCH":
            return 70
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        failure = DriverFailure("INTERNAL_ERROR", "driver failed closed", exception_type=type(exc).__name__)
        print(json.dumps(failure.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

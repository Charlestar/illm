"""Promote this checkout's verified V0.2 candidate metadata; never edit source or Git."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

from illm.canonical import canonical_json_bytes, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import _fs, _inventory, _object
from illm.stateful_qualification import stateful_source_identity
from illm.stateful_release import _state, verify_frozen_v02_spec, verify_stateful_release_bundle


METADATA = frozenset({"manifest.json", "COMPLETE", "qualification/release-state.json", "SHA256SUMS.json"})
SPEC = "docs/specs/versions/v0.2-stateful-streaming.md"
LOCK_NAME = ".v0.2.0-promotion.lock"


def _frozen_spec(project: Path, candidate: Path) -> None:
    path = project / SPEC
    if path.read_bytes() != (candidate / "spec.md").read_bytes():
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "candidate SPEC differs from the frozen checkout")
    verify_frozen_v02_spec(path)


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _release_root(project: Path, candidate: Path) -> Path:
    target = project / "release" / "v0.2.0"
    candidate = Path(candidate)
    if any(_linked(path) for path in (candidate, target.parent, target)):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion cannot follow release directory links")
    target_absolute = target.absolute()
    if (candidate.absolute() != target_absolute or target.resolve() != target_absolute
            or candidate.resolve() != target_absolute):
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "promotion only accepts this checkout's ordinary release/v0.2.0 directory",
        )
    return target


def _replace_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one allowed metadata file, preserving rollback bytes."""
    path = _fs(path)
    temporary = path.with_name(f".{path.name}.promotion-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    _replace_bytes(path, canonical_json_bytes(dict(value)) + b"\n")


def _released(result: Mapping[str, Any]) -> bool:
    return result.get("spec_status") == "released" and result.get("release_stage") == "released"


def _owned_lock_identity(path: Path, stream: BinaryIO, identity: os.stat_result) -> None:
    named = path.lstat()
    opened = os.fstat(stream.fileno())
    if (not stat.S_ISREG(named.st_mode) or not stat.S_ISREG(opened.st_mode)
            or not identity.st_ino or not os.path.samestat(identity, opened)
            or not os.path.samestat(identity, named)):
        raise OSError("promotion lock ownership cannot be confirmed")


def _remove_owned_lock(path: Path, stream: BinaryIO, identity: os.stat_result,
                       payload: bytes | None) -> None:
    """Never remove a missing, replaced, linked, or modified owner receipt."""
    _owned_lock_identity(path, stream, identity)
    if payload is not None:
        stream.seek(0)
        if stream.read(len(payload) + 1) != payload:
            raise OSError("promotion lock owner receipt changed")
    # Closing before unlink is required on Windows. Verify the named identity
    # again after closing; cooperating callers never replace another lock.
    stream.close()
    named = path.lstat()
    if not stat.S_ISREG(named.st_mode) or not os.path.samestat(identity, named):
        raise OSError("promotion lock identity changed during cleanup")
    path.unlink()


@contextmanager
def _promotion_lock(target: Path) -> Iterator[None]:
    """Exclusive across processes; stale locks require explicit operator review."""
    path = _fs(target.parent / LOCK_NAME)
    payload = canonical_json_bytes({"schema_version": 1, "kind": "illm-v02-promotion-lock-v1",
                                    "pid": os.getpid(), "owner_token": secrets.token_hex(32)}) + b"\n"
    try:
        stream = path.open("x+b")
    except FileExistsError as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT,
                        "promotion lock already exists; another or crashed transaction must not be cleared automatically",
                        {"lock_path": str(path)}) from exc
    identity: os.stat_result | None = None
    receipt: bytes | None = None
    try:
        identity = os.fstat(stream.fileno())
        if stream.write(payload) != len(payload):
            raise OSError("promotion lock receipt write was incomplete")
        stream.flush()
        receipt = payload
        os.fsync(stream.fileno())
        yield
    finally:
        try:
            if identity is None:
                raise OSError("promotion lock creation identity is unavailable")
            _remove_owned_lock(path, stream, identity, receipt)
        except BaseException as exc:
            try:
                stream.close()
            except BaseException:
                pass
            raise IllmError(
                ErrorCode.INTERNAL,
                "promotion lock cleanup could not be confirmed; do not publish or automatically remove the lock",
                {"exception_type": type(exc).__name__, "lock_path": str(path)},
            ) from exc


def _promote_locked(project: Path, target: Path) -> dict[str, Any]:
    source = stateful_source_identity(project)
    result = verify_stateful_release_bundle(target / "manifest.json", expected_source_identity=source)
    _frozen_spec(project, target)
    if _released(result):
        if stateful_source_identity(project) != source:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed during promotion verification")
        return result
    if result.get("spec_status") != "implemented" or result.get("release_stage") != "release_candidate":
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "only an implemented release candidate can be promoted")

    inventory = _inventory(target)
    original = {name: _fs(target / name).read_bytes() for name in METADATA}
    manifest = _object(target / "manifest.json")
    manifest.update(spec_status="released", release_stage="released")
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    if stateful_source_identity(project) != source:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed before promotion")

    try:
        _replace_json(target / "manifest.json", manifest)
        _replace_json(target / "qualification" / "release-state.json", _state(target / "manifest.json"))
        _replace_bytes(target / "COMPLETE", (str(manifest["manifest_digest"]) + "\n").encode("ascii"))
        checksums = {name: digest for name, digest in _inventory(target).items() if name != "SHA256SUMS.json"}
        _replace_json(target / "SHA256SUMS.json", checksums)

        after = _inventory(target)
        if (set(after) != set(inventory)
                or any(after[name] != digest for name, digest in inventory.items() if name not in METADATA)):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion changed a non-metadata release artifact")
        if stateful_source_identity(project) != source:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed during promotion")
        verified = verify_stateful_release_bundle(target / "manifest.json", expected_source_identity=source)
        if not _released(verified):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion verification did not confirm released metadata")
        if stateful_source_identity(project) != source:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed during final verification")
        return verified
    except BaseException as exc:
        failures: list[str] = []
        for name, content in original.items():
            try:
                _replace_bytes(target / name, content)
            except BaseException:
                failures.append(name)
        try:
            if _inventory(target) != inventory:
                failures.append("restored_inventory")
            restored = verify_stateful_release_bundle(target / "manifest.json", expected_source_identity=source)
            if restored.get("spec_status") != "implemented" or restored.get("release_stage") != "release_candidate":
                failures.append("restored_candidate_state")
        except BaseException:
            failures.append("restored_bundle_verification")
        if failures:
            raise IllmError(
                ErrorCode.INTERNAL,
                "promotion failed and original metadata restoration could not be confirmed; do not publish",
                {"exception_type": type(exc).__name__, "restoration_failures": failures},
            ) from exc
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "promotion failed; original release metadata was restored and verified",
            {"exception_type": type(exc).__name__},
        ) from exc


def promote_stateful_candidate(candidate: Path) -> dict[str, Any]:
    """Serialize preflight, noop, four metadata updates, and rollback as one owner."""
    project = Path(__file__).resolve().parents[1]
    target = _release_root(project, candidate)
    with _promotion_lock(target):
        return _promote_locked(project, target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", "--candidate", dest="candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(promote_stateful_candidate(args.candidate), sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        failure = IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "promotion input could not be verified; no promotion confirmed",
            {"exception_type": type(exc).__name__},
        )
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

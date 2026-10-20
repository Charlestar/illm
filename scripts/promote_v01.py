"""Promote this checkout's verified V0.1 candidate metadata; never edit source or Git."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import _fs, _inventory, _object, _write, reference_source_identity
from illm.reference_release import _state, verify_reference_release_bundle


METADATA = {"manifest.json", "COMPLETE", "qualification/release-state.json", "SHA256SUMS.json"}
SPEC = "docs/specs/versions/v0.1-reference-runner.md"
RELEASED_AXES = {"status": "released", "spec_status": "released", "release_stage": "released",
                 "qualification_tier": "functionally_qualified", "performance_outcome": "not_applicable",
                 "lts_state": "not_applicable", "target_release": "v0.1.0"}


def _released_spec(project: Path, candidate: Path) -> None:
    actual = (project / SPEC).read_bytes()
    if actual != (candidate / "spec.md").read_bytes():
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "candidate SPEC differs from the frozen checkout")
    block = re.search(r"(?ms)^```yaml[ \t]*\r?\n(.*?)^```[ \t]*\r?$", actual.decode("utf-8"))
    axes: dict[str, str] = {}
    if block is not None:
        for line in block[1].splitlines():
            match = re.fullmatch(r"([a-z_]+):[ \t]*([^#]*?)[ \t]*(?:#.*)?", line)
            if match is not None:
                key, value = match.groups()
                if key in axes:
                    raise IllmError(ErrorCode.INVALID_ARGUMENT, "frozen SPEC has duplicate top-level metadata")
                axes[key] = value.strip()
    if any(axes.get(key) != value for key, value in RELEASED_AXES.items()):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion requires the already-qualified frozen SPEC to declare every released axis")


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


def promote_reference_candidate(candidate: Path) -> dict[str, Any]:
    """Reverify, change only four generated metadata files, then reverify again."""
    project = Path(__file__).resolve().parents[1]
    target = project / "release" / "v0.1.0"
    if target.resolve() != target.absolute() or candidate.resolve() != target.resolve():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion only accepts this checkout's ordinary release/v0.1.0 directory")
    if any(path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()) for path in (target.parent, target)):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion cannot follow release directory links")
    source = reference_source_identity(project)
    result = verify_reference_release_bundle(target / "manifest.json", expected_source_identity=source)
    _released_spec(project, target)
    if result["spec_status"] == "released" and result["release_stage"] == "released":
        return result
    if result["spec_status"] != "implemented" or result["release_stage"] != "release_candidate":
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "only an implemented release candidate can be promoted")
    inventory = _inventory(target)
    original = {name: _fs(target / name).read_bytes() for name in METADATA}
    manifest = _object(target / "manifest.json")
    manifest.update(spec_status="released", release_stage="released")
    manifest["manifest_digest"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    if reference_source_identity(project) != source:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed before promotion")
    try:
        _write(target / "manifest.json", manifest)
        _write(target / "qualification" / "release-state.json", _state(target / "manifest.json"))
        _replace_bytes(target / "COMPLETE", (manifest["manifest_digest"] + "\n").encode("ascii"))
        _write(target / "SHA256SUMS.json", {name: digest for name, digest in _inventory(target).items() if name != "SHA256SUMS.json"})
        after = _inventory(target)
        if set(after) != set(inventory) or any(after[name] != digest for name, digest in inventory.items() if name not in METADATA):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion changed a non-metadata release artifact")
        if reference_source_identity(project) != source:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified source changed during promotion")
        return verify_reference_release_bundle(target / "manifest.json", expected_source_identity=source)
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
            verify_reference_release_bundle(target / "manifest.json", expected_source_identity=source)
        except BaseException:
            failures.append("restored_bundle_verification")
        if failures:
            raise IllmError(ErrorCode.INTERNAL, "promotion failed and original metadata restoration could not be confirmed; do not publish",
                            {"exception_type": type(exc).__name__, "restoration_failures": failures}) from exc
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "promotion failed; original release metadata was restored and verified",
                        {"exception_type": type(exc).__name__}) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", "--candidate", dest="candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(promote_reference_candidate(args.candidate), sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = IllmError(ErrorCode.INVALID_ARGUMENT, "promotion input could not be verified; no promotion confirmed",
                            {"exception_type": type(exc).__name__})
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit committed V0.2 released metadata, not Git tags or fresh GPU execution.

Default mode requires this final V0.2 source. CI may audit the bundle as a
historical artifact on later source versions; a V0.2 tag event never gets that
exception. The exact release commit still requires current-source verification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any, Sequence

from illm.domain.errors import ErrorCode, IllmError
from illm.stateful_qualification import stateful_source_identity
from illm.stateful_release import TAG, VERSION, verify_stateful_release_bundle


PROJECT = Path(__file__).resolve().parents[1]


def check_snapshot(manifest: Path, project: Path, *, ci: bool = False, event_ref: str = "") -> dict[str, Any]:
    """Revalidate all facts, released axes and the appropriate source boundary."""
    project = project.absolute()
    target = project / "release" / TAG / "manifest.json"
    if (project.resolve() != project or manifest.absolute() != target or target.parent.resolve() != target.parent
            or target.is_symlink()):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "snapshot check requires this checkout's ordinary release/v0.2.0/manifest.json")
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = metadata.get("version")
    if metadata.get("name") != "illm" or not isinstance(version, str) or not version:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "snapshot source project identity is invalid")
    current = not ci or version == VERSION or event_ref == f"refs/tags/{TAG}"
    if current and version != VERSION:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "V0.2 publication requires final package0.2.0, not development or later source")
    expected = stateful_source_identity(project) if current else None
    result = verify_stateful_release_bundle(target, expected_source_identity=expected)
    if result["release_tag"] != TAG or result["release_stage"] != "released" or result["spec_status"] != "released":
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "a candidate is not committed released V0.2 metadata")
    return {"status": "pass", "release_tag": TAG, "released_metadata": True,
            "source_binding": "current" if current else "historical", "git_tag_verified": False,
            "manifest_digest": result["manifest_digest"], "verification_digest": result["verification_digest"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ci", action="store_true", help="allow historical audit on later source, but never on a v0.2.0 tag event")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check_snapshot(args.manifest, PROJECT, ci=args.ci,
                                        event_ref=os.environ.get("GITHUB_REF", "")), sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError) as exc:
        failure = IllmError(ErrorCode.INVALID_ARGUMENT, "release snapshot could not be verified",
                            {"exception_type": type(exc).__name__})
        print(json.dumps(failure.to_dict(), sort_keys=True), file=sys.stderr)
        return failure.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

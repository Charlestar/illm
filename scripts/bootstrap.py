from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
import venv
from pathlib import Path
from typing import NoReturn, Sequence

from _toolchain_common import (
    PROFILE_PATTERN,
    SCHEMA_VERSION,
    ToolchainFailure,
    atomic_write_json,
    atomic_write_text,
    bootstrap_digest,
    environment_entrypoint,
    environment_python,
    expected_package_versions,
    installed_packages_probe_code,
    is_within,
    lock_digest,
    manifest_digest,
    project_digest,
)


class ToolchainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ToolchainFailure("INVALID_ARGUMENT", message)


def _parser() -> ToolchainArgumentParser:
    parser = ToolchainArgumentParser(description="Build an offline iLLM toolchain")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--write-manifest", type=Path, required=True)
    parser.add_argument("--write-manifest-digest", type=Path)
    return parser


def _safe_remove_environment(environment_root: Path, environments_root: Path) -> None:
    resolved = environment_root.resolve()
    parent = environments_root.resolve()
    if resolved.parent != parent or not is_within(resolved, parent) or resolved == parent:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "refusing to remove an environment outside the managed root",
            environment_root=str(resolved),
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _run(command: list[str], *, cwd: Path) -> None:
    process = subprocess.run(command, cwd=cwd, text=True)
    if process.returncode != 0:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "locked environment installation failed",
            command=command,
            returncode=process.returncode,
        )


def _install(project_root: Path, environment_root: Path) -> tuple[Path, Path, dict[str, str]]:
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False)
    builder.create(environment_root)
    python = environment_python(environment_root).resolve()
    if not python.is_file():
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "venv did not create an interpreter")
    wheelhouse = project_root / "vendor" / "wheels"
    bootstrap_lock = project_root / "requirements-bootstrap.lock"
    development_lock = project_root / "requirements-dev.lock"
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--require-hashes",
            "-r",
            str(bootstrap_lock),
        ],
        cwd=project_root,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--require-hashes",
            "-r",
            str(development_lock),
        ],
        cwd=project_root,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-build-isolation",
            "--no-deps",
            str(project_root),
        ],
        cwd=project_root,
    )
    entrypoint = environment_entrypoint(environment_root, "illm").resolve()
    if not entrypoint.is_file():
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "locked illm entrypoint is missing")
    package_probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            installed_packages_probe_code(),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    installed_packages = json.loads(package_probe.stdout)
    expected_packages = expected_package_versions(project_root)
    if installed_packages != expected_packages:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "installed package set does not exactly match locked packages",
            expected=expected_packages,
            actual=installed_packages,
        )
    return python, entrypoint, installed_packages


def bootstrap(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if not args.locked:
        raise ToolchainFailure("INVALID_ARGUMENT", "--locked is required for a finalized manifest")
    if not PROFILE_PATTERN.fullmatch(args.profile):
        raise ToolchainFailure("INVALID_ARGUMENT", "profile contains unsupported characters")
    if sys.version_info < (3, 11):
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "Python 3.11 or newer is required",
            actual=platform.python_version(),
        )
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = args.write_manifest.resolve()
    if not is_within(manifest_path, project_root):
        raise ToolchainFailure("INVALID_ARGUMENT", "manifest must be written inside the project root")
    digest_path = args.write_manifest_digest.resolve() if args.write_manifest_digest else None
    if digest_path and not is_within(digest_path, project_root):
        raise ToolchainFailure("INVALID_ARGUMENT", "manifest digest must be inside project root")
    environments_root = (project_root / ".illm" / "envs").resolve()
    environments_root.mkdir(parents=True, exist_ok=True)
    environment_root = (environments_root / args.profile).resolve()
    _safe_remove_environment(environment_root, environments_root)
    environment_root.mkdir(parents=True)
    atomic_write_json(environment_root / "INCOMPLETE", {"reason": "bootstrap_in_progress"})

    expected_lock_digest = lock_digest(project_root)
    expected_bootstrap_digest = bootstrap_digest(project_root)
    expected_project_digest = project_digest(project_root)
    environment_id = str(uuid.uuid4())
    try:
        python, entrypoint, installed_packages = _install(project_root, environment_root)
        sentinel = {
            "schema_version": SCHEMA_VERSION,
            "state": "ready",
            "environment_id": environment_id,
            "lock_digest": expected_lock_digest,
            "python_executable": str(python),
        }
        atomic_write_json(environment_root / ".illm-toolchain.json", sentinel)
        (environment_root / "INCOMPLETE").unlink(missing_ok=True)
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "finalized": True,
            "profile": args.profile,
            "os": platform.system(),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "project_root": str(project_root),
            "environment_root": str(environment_root),
            "python_executable": str(python),
            "entrypoints": {"illm": str(entrypoint)},
            "installed_packages": installed_packages,
            "environment_id": environment_id,
            "bootstrap_digest": expected_bootstrap_digest,
            "lock_digest": expected_lock_digest,
            "project_root_digest": expected_project_digest,
        }
        manifest["manifest_digest"] = manifest_digest(manifest)
        atomic_write_json(manifest_path, manifest)
        if digest_path:
            atomic_write_text(digest_path, str(manifest["manifest_digest"]) + "\n", encoding="ascii")
        return manifest
    except Exception as exc:
        manifest_path.unlink(missing_ok=True)
        (environment_root / ".illm-toolchain.json").unlink(missing_ok=True)
        atomic_write_json(
            environment_root / "INCOMPLETE",
            {"reason": "bootstrap_failed", "exception_type": type(exc).__name__},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = bootstrap(argv)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    except ToolchainFailure as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 64 if exc.code == "INVALID_ARGUMENT" else 70
    except Exception as exc:
        failure = ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "bootstrap failed before manifest finalization",
            exception_type=type(exc).__name__,
        )
        print(json.dumps(failure.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

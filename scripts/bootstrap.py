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
from typing import Any, NoReturn, Sequence

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
from _runtime_artifacts import (
    artifact_identities,
    artifact_path,
    isolated_environment,
    load_runtime_lock,
    runtime_probe,
    verify_runtime_cache,
    verify_runtime_host,
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
    parser.add_argument("--runtime-profile", help="explicit v01-cpu/v01-cuda alias or runtime profile ID")
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--offline", action="store_true", help="bootstrap always operates offline; explicit for tutorials")
    parser.add_argument("--require-new-environment", action="store_true",
                        help="atomically refuse an existing managed environment instead of replacing it")
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
    process = subprocess.run(command, cwd=cwd, text=True, env=isolated_environment())
    if process.returncode != 0:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "locked environment installation failed",
            command=command,
            returncode=process.returncode,
        )


def _install(
    project_root: Path, environment_root: Path,
    runtime_lock: dict[str, Any] | None = None, artifact_cache: Path | None = None,
) -> tuple[Path, Path, dict[str, str]]:
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
            "--no-compile",
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
            "--no-compile",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "--require-hashes",
            "-r",
            str(development_lock),
        ],
        cwd=project_root,
    )
    if runtime_lock is not None:
        if artifact_cache is None:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact cache is required")
        runtime_requirements = environment_root / ".runtime-requirements.txt"
        atomic_write_text(runtime_requirements, "\n".join(
            f"{artifact['name']} @ {artifact_path(artifact_cache, artifact).as_uri()} --hash=sha256:{artifact['sha256']}"
            for artifact in runtime_lock["artifacts"]
        ) + "\n")
        _run([
            str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-compile", "--no-index",
            "--only-binary", ":all:", "--no-deps", "--require-hashes", "-r", str(runtime_requirements),
        ], cwd=project_root)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
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
    expected_packages = expected_package_versions(project_root, runtime_lock)
    if installed_packages != expected_packages:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "installed package set does not exactly match locked packages",
            expected=expected_packages,
            actual=installed_packages,
        )
    _run([str(python), "-m", "pip", "check", "--disable-pip-version-check"], cwd=project_root)
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
    runtime_lock = load_runtime_lock(project_root, args.runtime_profile) if args.runtime_profile else None
    if args.artifact_cache is not None and runtime_lock is None:
        raise ToolchainFailure("INVALID_ARGUMENT", "--artifact-cache requires --runtime-profile")
    artifact_cache = (args.artifact_cache or project_root / ".illm" / "cache" / "runtime").resolve()
    if runtime_lock is not None:
        verify_runtime_host(runtime_lock)
        expected_package_versions(project_root, runtime_lock)
        verify_runtime_cache(runtime_lock, artifact_cache)
    manifest_path = args.write_manifest.resolve()
    if not is_within(manifest_path, project_root):
        raise ToolchainFailure("INVALID_ARGUMENT", "manifest must be written inside the project root")
    digest_path = args.write_manifest_digest.resolve() if args.write_manifest_digest else None
    if digest_path and not is_within(digest_path, project_root):
        raise ToolchainFailure("INVALID_ARGUMENT", "manifest digest must be inside project root")
    environments_root = (project_root / ".illm" / "envs").resolve()
    if args.require_new_environment and not environments_root.is_relative_to(project_root):
        raise ToolchainFailure("INVALID_ARGUMENT", "fresh environment root must remain inside the project")
    environments_root.mkdir(parents=True, exist_ok=True)
    environment_root = (environments_root / args.profile).resolve()
    if args.require_new_environment:
        if (environments_root / args.profile).is_symlink() or environment_root.parent != environments_root:
            raise ToolchainFailure("INVALID_ARGUMENT", "fresh environment cannot follow an escaped link")
        try:
            environment_root.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ToolchainFailure("INVALID_ARGUMENT", "fresh environment already exists and will not be replaced") from exc
    else:
        _safe_remove_environment(environment_root, environments_root)
        environment_root.mkdir(parents=True)
    atomic_write_json(environment_root / "INCOMPLETE", {"reason": "bootstrap_in_progress"})

    expected_lock_digest = lock_digest(project_root, runtime_lock)
    expected_bootstrap_digest = bootstrap_digest(project_root)
    expected_project_digest = project_digest(project_root)
    environment_id = str(uuid.uuid4())
    try:
        python, entrypoint, installed_packages = _install(project_root, environment_root, runtime_lock, artifact_cache)
        probe = runtime_probe(python, runtime_lock) if runtime_lock else None
        sentinel = {
            "schema_version": 1 if runtime_lock else SCHEMA_VERSION,
            "state": "ready",
            "environment_id": environment_id,
            "lock_digest": expected_lock_digest,
            "python_executable": str(python),
        }
        if runtime_lock:
            sentinel.update({"runtime_profile_id": runtime_lock["profile_id"], "runtime_lock_digest": runtime_lock["lock_digest"]})
        atomic_write_json(environment_root / ".illm-toolchain.json", sentinel)
        (environment_root / "INCOMPLETE").unlink(missing_ok=True)
        manifest: dict[str, object] = {
            "schema_version": 1 if runtime_lock else SCHEMA_VERSION,
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
        if runtime_lock:
            manifest.update({
                "runtime_profile_id": runtime_lock["profile_id"],
                "runtime_lock_digest": runtime_lock["lock_digest"],
                "runtime_artifacts": artifact_identities(runtime_lock),
                "artifact_cache": str(artifact_cache), "runtime_probe": probe,
            })
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
            reason=str(exc),
        )
        print(json.dumps(failure.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

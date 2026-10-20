from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, NoReturn, Sequence, cast

from _toolchain_common import (
    PROFILE_PATTERN,
    SCHEMA_VERSION,
    TOOLCHAIN_FIELDS,
    ToolchainFailure,
    bootstrap_digest,
    environment_python,
    environment_entrypoint,
    expected_package_versions,
    installed_packages_probe_code,
    is_within,
    lock_digest,
    manifest_digest,
    normalized_path,
    project_digest,
)
from _runtime_artifacts import (
    RUNTIME_MANIFEST_FIELDS,
    artifact_identities,
    isolated_environment,
    load_runtime_lock,
    runtime_probe,
    verify_runtime_cache,
)


def _split_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    try:
        separator = argv.index("--")
    except ValueError as exc:
        raise ToolchainFailure("INVALID_ARGUMENT", "runner requires -- before the child command") from exc
    runner_args = list(argv[:separator])
    child_args = list(argv[separator + 1 :])
    if not child_args:
        raise ToolchainFailure("INVALID_ARGUMENT", "child command must not be empty")
    return runner_args, child_args


class ToolchainArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ToolchainFailure("INVALID_ARGUMENT", message)


def _parser() -> ToolchainArgumentParser:
    parser = ToolchainArgumentParser(description="Run inside a finalized iLLM toolchain")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-digest-file", type=Path)
    return parser


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "toolchain manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "toolchain manifest must be an object")
    manifest = cast(dict[str, Any], value)
    version = manifest.get("schema_version")
    if version not in {SCHEMA_VERSION, 1} or isinstance(version, bool) or manifest.get("finalized") is not True:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest is not a finalized supported toolchain")
    fields = TOOLCHAIN_FIELDS | RUNTIME_MANIFEST_FIELDS if version == 1 else TOOLCHAIN_FIELDS
    if set(manifest) != fields:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "toolchain manifest fields do not match schema",
            missing=sorted(fields - set(manifest)),
            extra=sorted(set(manifest) - fields),
        )
    if manifest.get("manifest_digest") != manifest_digest(manifest):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest digest mismatch")
    return manifest


def _verify_handshake(manifest: dict[str, Any], python: Path, environment_root: Path) -> None:
    sentinel = environment_root / ".illm-toolchain.json"
    handshake_code = (
        "import json,pathlib,platform,sys;"
        f"p=pathlib.Path({str(sentinel)!r});"
        "d=json.loads(p.read_text(encoding='utf-8'));"
        "print(json.dumps({'sys_executable':str(pathlib.Path(sys.executable).resolve()),"
        "'environment_id':d.get('environment_id'),'lock_digest':d.get('lock_digest'),"
        "'sentinel_python':d.get('python_executable'),'state':d.get('state'),"
        "'os':platform.system(),'arch':platform.machine(),'python_version':platform.python_version(),"
        "'runtime_profile_id':d.get('runtime_profile_id'),'runtime_lock_digest':d.get('runtime_lock_digest')},sort_keys=True))"
    )
    process = subprocess.run(
        [str(python), "-I", "-c", handshake_code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "environment handshake failed",
            returncode=process.returncode,
        )
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "environment handshake returned invalid data") from exc
    expected = {
        "sys_executable": normalized_path(python),
        "environment_id": manifest["environment_id"],
        "lock_digest": manifest["lock_digest"],
        "sentinel_python": normalized_path(python),
        "state": "ready",
        "os": manifest["os"],
        "arch": manifest["arch"],
        "python_version": manifest["python_version"],
        "runtime_profile_id": manifest.get("runtime_profile_id"),
        "runtime_lock_digest": manifest.get("runtime_lock_digest"),
    }
    actual = dict(result)
    actual["sys_executable"] = normalized_path(Path(str(actual.get("sys_executable", ""))))
    actual["sentinel_python"] = normalized_path(Path(str(actual.get("sentinel_python", ""))))
    if actual != expected:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "environment handshake identity mismatch",
            expected=expected,
            actual=actual,
        )
    packages = subprocess.run(
        [str(python), "-I", "-c", installed_packages_probe_code()],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        installed_packages = json.loads(packages.stdout) if packages.returncode == 0 else None
    except json.JSONDecodeError:
        installed_packages = None
    if installed_packages != manifest["installed_packages"]:
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "installed package identity mismatch",
            expected=manifest["installed_packages"],
            actual=installed_packages,
        )


def _verify_manifest(manifest: dict[str, Any], expected_digest_file: Path | None) -> tuple[Path, Path]:
    project_root = Path(str(manifest["project_root"])).resolve()
    actual_project_root = Path(__file__).resolve().parents[1]
    if normalized_path(project_root) != normalized_path(actual_project_root):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "project root identity mismatch")
    environment_root = Path(str(manifest["environment_root"])).resolve()
    managed_root = (project_root / ".illm" / "envs").resolve()
    if not is_within(environment_root, managed_root) or environment_root == managed_root:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "environment root escapes managed root")
    python = Path(str(manifest["python_executable"])).resolve()
    if not python.is_file() or not is_within(python, environment_root):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest interpreter is missing or outside environment")
    if normalized_path(python) != normalized_path(environment_python(environment_root)):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest interpreter path is not canonical")
    try:
        uuid.UUID(str(manifest["environment_id"]))
    except (ValueError, AttributeError) as exc:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "environment ID is not a UUID") from exc
    if not PROFILE_PATTERN.fullmatch(str(manifest["profile"])):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest profile is invalid")
    entrypoints = manifest.get("entrypoints")
    expected_entrypoint = environment_entrypoint(environment_root, "illm").resolve()
    if not isinstance(entrypoints, dict) or set(entrypoints) != {"illm"} or normalized_path(
        Path(str(entrypoints.get("illm", "")))
    ) != normalized_path(expected_entrypoint):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "locked entrypoint identity mismatch")
    runtime_lock = None
    if manifest["schema_version"] == 1:
        runtime_lock = load_runtime_lock(project_root, str(manifest["runtime_profile_id"]))
        if (
            manifest["runtime_lock_digest"] != runtime_lock["lock_digest"]
            or manifest["runtime_artifacts"] != artifact_identities(runtime_lock)
            or manifest["os"] != runtime_lock["target"]["os"]
            or manifest["python_version"] != runtime_lock["target"]["python_validation_patch"]
        ):
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest runtime profile/artifact identity mismatch")
    expected_packages = expected_package_versions(project_root, runtime_lock)
    if manifest.get("installed_packages") != expected_packages:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest package set does not match lock files")
    checks = {
        "bootstrap_digest": bootstrap_digest(project_root),
        "lock_digest": lock_digest(project_root, runtime_lock),
        "project_root_digest": project_digest(project_root),
    }
    mismatches = {
        key: {"expected": manifest.get(key), "actual": actual}
        for key, actual in checks.items()
        if manifest.get(key) != actual
    }
    if mismatches:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "toolchain source digest mismatch", mismatches=mismatches)
    if expected_digest_file:
        try:
            expected_digest = expected_digest_file.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest digest sidecar is unreadable") from exc
        if expected_digest != manifest["manifest_digest"]:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "manifest digest sidecar mismatch")
    _verify_handshake(manifest, python, environment_root)
    if runtime_lock:
        verify_runtime_cache(runtime_lock, Path(str(manifest["artifact_cache"])))
        if runtime_probe(python, runtime_lock) != manifest["runtime_probe"]:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime import/device probe identity mismatch")
    return project_root, environment_root


def _resolve_command(manifest: dict[str, Any], environment_root: Path, child: list[str]) -> list[str]:
    command = child[0]
    if command.lower() in {"python", "python.exe"}:
        executable = Path(str(manifest["python_executable"])).resolve()
    elif command in manifest.get("entrypoints", {}):
        executable = Path(str(manifest["entrypoints"][command])).resolve()
    else:
        candidate = Path(command)
        if not candidate.is_absolute():
            raise ToolchainFailure(
                "TOOL_NOT_IN_LOCKED_ENV",
                "command is not a locked environment entrypoint",
                command=command,
            )
        executable = candidate.resolve()
    allowed = {
        normalized_path(Path(str(manifest["python_executable"]))),
        *(normalized_path(Path(str(path))) for path in manifest.get("entrypoints", {}).values()),
    }
    if normalized_path(executable) not in allowed or not is_within(executable, environment_root):
        raise ToolchainFailure(
            "TOOL_NOT_IN_LOCKED_ENV",
            "resolved command escapes the locked environment",
            command=command,
        )
    if not executable.is_file():
        raise ToolchainFailure("TOOL_NOT_IN_LOCKED_ENV", "locked entrypoint is missing", command=command)
    return [str(executable), *child[1:]]


def run(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    runner_args, child = _split_args(raw)
    args = _parser().parse_args(runner_args)
    manifest = _load_manifest(args.manifest.resolve())
    project_root, environment_root = _verify_manifest(manifest, args.expected_manifest_digest_file)
    command = _resolve_command(manifest, environment_root, child)
    child_environment = isolated_environment()
    child_environment["VIRTUAL_ENV"] = str(environment_root)
    child_environment["ILLM_ENVIRONMENT_ID"] = str(manifest["environment_id"])
    child_environment["ILLM_LOCK_DIGEST"] = str(manifest["lock_digest"])
    child_environment["ILLM_TOOLCHAIN_MANIFEST"] = str(args.manifest.resolve())
    child_environment["PATH"] = str(Path(command[0]).parent) + os.pathsep + child_environment.get("PATH", "")
    process = subprocess.run(command, cwd=project_root, env=child_environment, shell=False)
    return process.returncode


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except ToolchainFailure as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        if exc.code == "INVALID_ARGUMENT":
            return 64
        if exc.code == "TOOL_NOT_IN_LOCKED_ENV":
            return 69
        return 70
    except Exception as exc:
        failure = ToolchainFailure(
            "TOOLCHAIN_MISMATCH",
            "runner failed before child execution",
            exception_type=type(exc).__name__,
        )
        print(json.dumps(failure.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

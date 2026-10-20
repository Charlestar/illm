from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 0
PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
LOCK_REQUIREMENT_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
TOOLCHAIN_FIELDS = {
    "schema_version",
    "finalized",
    "profile",
    "os",
    "arch",
    "python_version",
    "project_root",
    "environment_root",
    "python_executable",
    "entrypoints",
    "installed_packages",
    "environment_id",
    "bootstrap_digest",
    "lock_digest",
    "project_root_digest",
    "manifest_digest",
}


class ToolchainFailure(Exception):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_entries(entries: Iterable[tuple[str, str]]) -> str:
    return sha256_bytes(canonical_bytes(sorted(entries)))


def digest_files(project_root: Path, paths: Iterable[Path]) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted({item.resolve() for item in paths}, key=lambda item: item.as_posix()):
        if path.is_file():
            entries.append((path.relative_to(project_root).as_posix(), sha256_file(path)))
    return digest_entries(entries)


def lock_digest(project_root: Path, runtime_lock: dict[str, Any] | None = None) -> str:
    locks = [project_root / name for name in ("requirements-bootstrap.lock", "requirements-dev.lock")]
    wheels = sorted((project_root / "vendor" / "wheels").glob("*.whl"))
    if not all(path.is_file() for path in locks) or not wheels:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "offline lock or wheelhouse is missing")
    base = digest_files(project_root, [*locks, *wheels])
    if runtime_lock is None:
        return base
    return sha256_bytes(canonical_bytes({"base_lock_digest": base, "runtime_lock_digest": runtime_lock["lock_digest"]}))


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_package_versions(project_root: Path, runtime_lock: dict[str, Any] | None = None) -> dict[str, str]:
    versions: dict[str, str] = {}
    for lock in [project_root / name for name in ("requirements-bootstrap.lock", "requirements-dev.lock")]:
        for line in lock.read_text(encoding="utf-8").splitlines():
            match = LOCK_REQUIREMENT_PATTERN.match(line.strip())
            if match:
                name, version = normalize_distribution_name(match.group(1)), match.group(2)
                if name in versions and versions[name] != version:
                    raise ToolchainFailure("TOOLCHAIN_MISMATCH", "shared dependency pins conflict", package=name)
                versions[name] = version
    if runtime_lock is not None:
        for artifact in runtime_lock["artifacts"]:
            name, version = str(artifact["name"]), str(artifact["version"])
            if name in versions and versions[name] != version:
                raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime and shared dependency pins conflict", package=name)
            versions[name] = version
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    versions[normalize_distribution_name(str(project["name"]))] = str(project["version"])
    return dict(sorted(versions.items()))


def installed_packages_probe_code() -> str:
    return (
        "import importlib.metadata as m,json,re;"
        "n=lambda s:re.sub(r'[-_.]+','-',s).lower();"
        "print(json.dumps({n(d.metadata['Name']):d.version for d in m.distributions() "
        "if d.metadata.get('Name')},sort_keys=True))"
    )


def bootstrap_digest(project_root: Path) -> str:
    return digest_files(
        project_root,
        [project_root / "scripts" / name for name in ("bootstrap.py", "_toolchain_common.py", "_runtime_artifacts.py")],
    )


def project_digest(project_root: Path) -> str:
    paths: list[Path] = [
        project_root / ".gitattributes",
        project_root / ".gitignore",
        project_root / "AI_INFERENCE_ENGINE_PROJECT_PLAN.md",
        project_root / "LICENSE",
        project_root / "README.md",
        project_root / "pyproject.toml",
        project_root / "requirements-bootstrap.lock",
        project_root / "requirements-dev.lock",
    ]
    for relative in (
        ".github",
        "docs",
        "evidence",
        "src",
        "scripts",
        "tests",
        "schemas",
        "registries",
        "runtime",
        "configs",
        "fixtures",
    ):
        root = project_root / relative
        if root.exists():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
            )
    return digest_files(project_root, paths)


def manifest_digest(manifest: dict[str, Any]) -> str:
    projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    return sha256_bytes(canonical_bytes(projection))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding=encoding, newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def environment_python(environment_root: Path) -> Path:
    if os.name == "nt":
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def environment_entrypoint(environment_root: Path, name: str) -> Path:
    if os.name == "nt":
        candidates = [
            environment_root / "Scripts" / f"{name}.exe",
            environment_root / "Scripts" / f"{name}.cmd",
            environment_root / "Scripts" / name,
        ]
    else:
        candidates = [environment_root / "bin" / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]

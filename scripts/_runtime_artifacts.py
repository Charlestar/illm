"""Explicit runtime artifacts shared by online fetch and offline toolchain paths.

Only the fetch command imports network functionality. This module is stdlib-only
so a clean Python interpreter can validate the profile before creating a venv.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
import zipfile

from _toolchain_common import (
    ToolchainFailure,
    canonical_bytes,
    is_within,
    normalize_distribution_name,
    sha256_bytes,
    sha256_file,
)

RUNTIME_MANIFEST_FIELDS = {
    "runtime_profile_id", "runtime_lock_digest", "runtime_artifacts", "artifact_cache", "runtime_probe"
}
ARTIFACT_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime authority is unreadable", path=str(path)) from exc
    if not isinstance(value, dict):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime authority must be an object")
    return cast(dict[str, Any], value)


def load_runtime_lock(project_root: Path, requested: str) -> dict[str, Any]:
    registry = _read_object(project_root / "runtime" / "profiles.json")
    if type(registry.get("schema_version")) is not int or registry.get("schema_version") != 1 or not isinstance(registry.get("profiles"), dict):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "unsupported runtime profile registry")
    if requested in {"v01-cpu", "v01-cuda"}:
        aliases = registry.get("aliases", {}).get(requested, {})
        requested = str(aliases.get(platform.system(), ""))
    selected = registry["profiles"].get(requested)
    if not requested or not isinstance(selected, dict):
        raise ToolchainFailure("INVALID_ARGUMENT", "unknown runtime profile", profile=requested)
    relative = selected.get("artifact_lock")
    if not isinstance(relative, str):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime profile has no artifact lock")
    lock_path = (project_root / relative).resolve()
    authority_root = (project_root / "runtime" / "artifacts").resolve()
    if not is_within(lock_path, authority_root):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime lock escapes artifact authority")
    lock = _read_object(lock_path)
    projection = {key: value for key, value in lock.items() if key != "lock_digest"}
    actual = sha256_bytes(canonical_bytes(projection))
    if (
        type(lock.get("schema_version")) is not int or lock.get("schema_version") != 1
        or lock.get("kind") != "illm-runtime-artifact-lock"
        or lock.get("profile_id") != requested
        or lock.get("lock_digest") != actual
        or selected.get("lock_digest") != actual
    ):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact lock identity mismatch")
    if not isinstance(lock.get("target"), dict) or not isinstance(lock.get("artifacts"), list):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact lock fields are invalid")
    names: set[str] = set()
    for artifact in lock["artifacts"]:
        if not isinstance(artifact, dict):
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact must be an object")
        name = artifact.get("name")
        filename = artifact.get("filename")
        sha = artifact.get("sha256")
        metadata_sha = artifact.get("metadata_sha256")
        size = artifact.get("size_bytes")
        url = urlparse(str(artifact.get("url", "")))
        if (
            not isinstance(name, str) or normalize_distribution_name(name) != name or name in names
            or not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.+%-]+\.whl", filename)
            or not isinstance(sha, str) or not SHA256_PATTERN.fullmatch(sha)
            or not isinstance(metadata_sha, str) or not SHA256_PATTERN.fullmatch(metadata_sha)
            or not isinstance(size, int) or isinstance(size, bool) or size <= 0
            or url.scheme != "https" or url.hostname not in ARTIFACT_HOSTS
            or url.username is not None or url.password is not None or url.fragment or url.query or url.port not in {None, 443}
            or artifact.get("cache_key") != f"sha256/{sha[:2]}/{sha}/{filename}"
            or not isinstance(artifact.get("version"), str)
        ):
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "invalid runtime artifact identity", name=name)
        names.add(name)
    if not {"torch", "transformers", "tokenizers", "safetensors"}.issubset(names):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime lock lacks required model/oracle dependencies")
    return lock


def verify_runtime_host(lock: dict[str, Any]) -> None:
    target = lock["target"]
    required_arch = "AMD64" if target["os"] == "Windows" else "x86_64"
    if (
        platform.system() != target["os"]
        or platform.machine().lower() != required_arch.lower()
        or sys.implementation.name != target["python_implementation"]
        or platform.python_version() != target["python_validation_patch"]
    ):
        raise ToolchainFailure(
            "TOOLCHAIN_MISMATCH", "runtime profile host identity mismatch",
            expected={"os": target["os"], "arch": required_arch, "python": target["python_validation_patch"]},
            actual={"os": platform.system(), "arch": platform.machine(), "python": platform.python_version()},
        )
    if target["os"] == "Linux":
        libc_name, libc_version = platform.libc_ver()
        floor = tuple(int(part) for part in target["glibc_floor"].split("."))
        actual = tuple(int(part) for part in libc_version.split(".")) if libc_version else ()
        if libc_name != "glibc" or actual < floor:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime profile requires newer glibc")


def runtime_packages(lock: dict[str, Any]) -> dict[str, str]:
    return {str(item["name"]): str(item["version"]) for item in lock["artifacts"]}


def artifact_identities(lock: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: artifact[key] for key in ("name", "version", "filename", "sha256", "size_bytes")}
        for artifact in lock["artifacts"]
    ]


def artifact_path(cache_root: Path, artifact: dict[str, Any]) -> Path:
    root = cache_root.resolve()
    path = (root / str(artifact["cache_key"])).resolve()
    if not is_within(path, root) or path == root:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact cache path escapes root")
    return path


def filesystem_path(path: Path) -> Path:
    """Use Windows extended paths for long cross-platform wheel filenames."""
    absolute = str(path.absolute())
    if os.name == "nt" and len(absolute) >= 240 and not absolute.startswith("\\\\?\\"):
        if absolute.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + absolute[2:])
        return Path("\\\\?\\" + absolute)
    return path


def verify_artifact(path: Path, artifact: dict[str, Any], *, metadata: bool = True) -> None:
    path = filesystem_path(path)
    if not path.is_file():
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact cache miss; run scripts/fetch_runtime.py explicitly", artifact=artifact["filename"])
    if path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact hash/size mismatch", artifact=artifact["filename"])
    if metadata:
        try:
            with zipfile.ZipFile(path) as wheel:
                members = [item for item in wheel.infolist() if item.filename.count("/") == 1 and item.filename.endswith(".dist-info/METADATA")]
                if len(members) != 1 or members[0].file_size > 4 * 1024 * 1024:
                    raise ValueError("invalid wheel METADATA member")
                raw = wheel.read(members[0])
            if sha256_bytes(raw) != artifact["metadata_sha256"]:
                raise ValueError("wheel metadata digest differs from reviewed lock")
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime wheel metadata mismatch", artifact=artifact["filename"]) from exc


def verify_runtime_cache(lock: dict[str, Any], cache_root: Path) -> None:
    for artifact in lock["artifacts"]:
        verify_artifact(artifact_path(cache_root, artifact), artifact)


def isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in list(environment):
        if key.upper().startswith("PIP_") or key.upper() in {"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"}:
            environment.pop(key, None)
    environment.update({
        "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    return environment


def runtime_probe(python: Path, lock: dict[str, Any]) -> dict[str, Any]:
    cuda = lock["target"]["torch_channel"] != "cpu"
    code = (
        "import json,torch,transformers,safetensors,tokenizers;"
        "r={'torch':torch.__version__,'transformers':transformers.__version__,"
        "'safetensors':safetensors.__version__,'tokenizers':tokenizers.__version__,'cuda_build':torch.version.cuda};"
        f"gpu={cuda!r};"
        "device='cuda' if gpu else 'cpu';"
        "x=torch.tensor([1.,2.],device=device);assert x.sum().item()==3.;"
        "r['smoke_sum']=3.0;r['device']=device;"
        "r['gpu_name']=torch.cuda.get_device_name(0) if gpu else None;"
        "r['device_capability']=list(torch.cuda.get_device_capability(0)) if gpu else None;"
        "r['compiled_arches']=torch.cuda.get_arch_list() if gpu else [];"
        "torch.cuda.synchronize() if gpu else None;print(json.dumps(r,sort_keys=True))"
    )
    process = subprocess.run([str(python), "-I", "-c", code], check=False, capture_output=True, text=True, timeout=120, env=isolated_environment())
    if process.returncode != 0:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime import/device smoke probe failed", stderr=process.stderr[-4000:])
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime probe returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime probe must return an object")
    packages = runtime_packages(lock)
    for name in ("torch", "transformers", "safetensors", "tokenizers"):
        if result.get(name) != packages[name]:
            raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime imported package version differs from lock", package=name)
    if (cuda and (result.get("cuda_build") != "12.6" or result.get("device_capability") != [8, 6] or "sm_86" not in result.get("compiled_arches", []))) or (not cuda and result.get("cuda_build") is not None):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime device/build identity differs from profile")
    return cast(dict[str, Any], result)

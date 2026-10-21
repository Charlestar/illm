"""Source/environment identity for policy executions; no tensor dependency."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from illm.model.spec import ModelError


_FILES = (".gitattributes", ".gitignore", "AI_INFERENCE_ENGINE_PROJECT_PLAN.md", "LICENSE", "README.md",
          "pyproject.toml", "requirements-bootstrap.lock", "requirements-dev.lock")
_DIRECTORIES = (".github", "docs", "evidence", "src", "scripts", "tests", "schemas", "registries",
                "runtime", "configs", "fixtures")


def _project_digest(root: Path) -> str:
    paths = [root / name for name in _FILES]
    for directory in _DIRECTORIES:
        parent = root / directory
        if parent.exists():
            paths.extend(path for path in parent.rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts
                         and not any(part.endswith(".egg-info") for part in path.parts))
    entries: list[tuple[str, str]] = []
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ModelError("INVALID_ARGUMENT", "policy source cannot contain escaped or linked files")
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return sha256_json(sorted(entries))


@dataclass(frozen=True, slots=True)
class ExecutionBindingV1:
    _canonical_bytes: bytes

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._canonical_bytes))

    @property
    def digest(self) -> str:
        return sha256_bytes(self._canonical_bytes)


def build_execution_binding(project_root: Path) -> ExecutionBindingV1:
    """Bind current source and CPU software, rejecting a stale supplied manifest."""
    root = project_root.resolve()
    if not (root / "src/illm").is_dir() or not (root / "pyproject.toml").is_file():
        raise ModelError("INVALID_ARGUMENT", "explicit policy execution requires its local source workspace")
    source_digest = _project_digest(root)
    toolchain: dict[str, Any] = {"binding": "unbound-development", "manifest_digest": None}
    manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if manifest_name:
        manifest = json.loads(Path(manifest_name).read_text(encoding="utf-8"))
        if (manifest.get("finalized") is not True
                or manifest.get("manifest_digest") != sha256_json({k: v for k, v in manifest.items() if k != "manifest_digest"})
                or Path(str(manifest.get("project_root"))).resolve() != root
                or manifest.get("project_root_digest") != source_digest
                or manifest.get("environment_id") != os.environ.get("ILLM_ENVIRONMENT_ID")
                or manifest.get("lock_digest") != os.environ.get("ILLM_LOCK_DIGEST")
                or manifest.get("python_version") != platform.python_version()
                or Path(str(manifest.get("python_executable"))).resolve() != Path(sys.executable).resolve()):
            raise ModelError("INVALID_ARGUMENT", "policy execution rejects stale or mismatched environment manifest")
        toolchain = {"binding": "source-bound-manifest", "manifest_digest": manifest["manifest_digest"],
                     "environment_id": manifest["environment_id"], "lock_digest": manifest["lock_digest"]}
    environment = {"backend": "cpu", "python_version": platform.python_version(), "platform": platform.platform(),
                   "machine": platform.machine(), "torch_version": importlib.metadata.version("torch"),
                   "cuda_arithmetic": "not_applicable", "cuda_execution_authorized": False}
    return ExecutionBindingV1(canonical_json_bytes({
        "schema_version": 1, "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": source_digest, "environment": environment,
        "environment_digest": sha256_json(environment), "toolchain": toolchain,
        "protocol_sha256": "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5",
    }))

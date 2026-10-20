from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import tomllib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "in_env.py"


def _manifest_path() -> Path:
    value = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not value:
        pytest.skip("test must run through scripts/in_env.py")
    return Path(value)


def _run(manifest: Path, child: list[str], *, path_prefix: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if path_prefix:
        environment["PATH"] = str(path_prefix) + os.pathsep + environment.get("PATH", "")
    return subprocess.run(
        [sys.executable, str(RUNNER), "--manifest", str(manifest), "--", *child],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _fake_command(directory: Path, name: str, marker: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        (directory / f"{name}.bat").write_text(f"@echo fake>{marker}\n@exit /b 99\n", encoding="utf-8")
    else:
        path = directory / name
        path.write_text(f"#!/bin/sh\necho fake > {marker}\nexit 99\n", encoding="utf-8")
        path.chmod(0o755)


def _resign(manifest: dict[str, object]) -> None:
    projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_runner_uses_manifest_interpreter_despite_path_poisoning(tmp_path: Path) -> None:
    manifest_path = _manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fake = tmp_path / "fake-bin"
    marker = tmp_path / "called"
    _fake_command(fake, "python", marker)
    process = _run(
        manifest_path,
        ["python", "-c", "import json,sys;print(json.dumps({'python':sys.executable}))"],
        path_prefix=fake,
    )
    assert process.returncode == 0, process.stderr
    assert Path(json.loads(process.stdout)["python"]).resolve() == Path(manifest["python_executable"]).resolve()
    expected_packages = {
        "colorama": "0.4.6",
        "illm": tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"],
        "iniconfig": "2.3.0",
        "packaging": "26.3",
        "mypy": "1.18.2",
        "mypy-extensions": "1.1.0",
        "pathspec": "1.1.1",
        "pip": "26.2.1",
        "pluggy": "1.6.0",
        "pygments": "2.21.0",
        "pytest": "9.1.1",
        "setuptools": "83.0.0",
        "typing-extensions": "4.16.0",
        "wheel": "0.47.0",
    }
    if manifest["schema_version"] == 1:
        expected_packages.update({artifact["name"]: artifact["version"] for artifact in manifest["runtime_artifacts"]})
    assert manifest["installed_packages"] == expected_packages
    assert not marker.exists()


def test_runner_never_falls_back_to_global_entrypoint(tmp_path: Path) -> None:
    fake = tmp_path / "fake-bin"
    marker = tmp_path / "called"
    _fake_command(fake, "not-in-lock", marker)
    process = _run(_manifest_path(), ["not-in-lock"], path_prefix=fake)
    assert process.returncode == 69
    assert "TOOL_NOT_IN_LOCKED_ENV" in process.stderr
    assert not marker.exists()


def test_manifest_console_entrypoint_ignores_poisoned_path(tmp_path: Path) -> None:
    fake = tmp_path / "fake-bin"
    marker = tmp_path / "called"
    _fake_command(fake, "illm", marker)
    process = _run(_manifest_path(), ["illm", "--help"], path_prefix=fake)
    assert process.returncode == 0, process.stderr
    assert "usage: illm" in process.stdout
    assert not marker.exists()


def test_manifest_tamper_fails_before_child_import(tmp_path: Path) -> None:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest["environment_id"] = "tampered"
    tampered = tmp_path / "toolchain.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    marker = tmp_path / "imported"
    process = _run(
        tampered,
        ["python", "-c", f"import illm;open({str(marker)!r},'w').write('bad')"],
    )
    assert process.returncode == 70
    assert "TOOLCHAIN_MISMATCH" in process.stderr
    assert not marker.exists()


def test_resigned_environment_id_tamper_is_caught_by_handshake(tmp_path: Path) -> None:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest["environment_id"] = "00000000-0000-0000-0000-000000000000"
    _resign(manifest)
    tampered = tmp_path / "toolchain.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    marker = tmp_path / "child-ran"
    process = _run(
        tampered,
        ["python", "-c", f"open({str(marker)!r},'w').write('bad')"],
    )
    assert process.returncode == 70
    assert "handshake identity mismatch" in process.stderr
    assert not marker.exists()


def test_resigned_interpreter_path_escape_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest["python_executable"] = str(RUNNER)
    _resign(manifest)
    tampered = tmp_path / "toolchain.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    process = _run(tampered, ["python", "-c", "raise SystemExit(0)"])
    assert process.returncode == 70
    assert "outside environment" in process.stderr


def test_resigned_entrypoint_and_package_set_tamper_fail_before_child(tmp_path: Path) -> None:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest["entrypoints"]["illm"] = manifest["python_executable"]
    _resign(manifest)
    tampered_entrypoint = tmp_path / "entrypoint.json"
    tampered_entrypoint.write_text(json.dumps(manifest), encoding="utf-8")
    process = _run(tampered_entrypoint, ["python", "-c", "raise SystemExit(0)"])
    assert process.returncode == 70
    assert "entrypoint identity mismatch" in process.stderr

    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    del manifest["installed_packages"]["pytest"]
    _resign(manifest)
    tampered_packages = tmp_path / "packages.json"
    tampered_packages.write_text(json.dumps(manifest), encoding="utf-8")
    process = _run(tampered_packages, ["python", "-c", "raise SystemExit(0)"])
    assert process.returncode == 70
    assert "package set does not match lock files" in process.stderr


def test_unfinalized_manifest_fails_before_child(tmp_path: Path) -> None:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    manifest["finalized"] = False
    path = tmp_path / "unfinalized.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    process = _run(path, ["python", "-c", "raise SystemExit(0)"])
    assert process.returncode == 70
    assert "not a finalized" in process.stderr


def test_child_exit_code_is_forwarded() -> None:
    process = _run(_manifest_path(), ["python", "-c", "raise SystemExit(17)"])
    assert process.returncode == 17

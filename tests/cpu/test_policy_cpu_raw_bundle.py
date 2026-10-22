"""Closed container checks; numeric comparisons have their own raw tests."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.policy_cpu_numerical_contract import NumericalEvidenceError, write_json_exclusive
from illm.policy_cpu_raw_bundle import (
    AUTHORITY_NAMES, BUNDLE_KIND, COMMAND_NAMES, WORKER_NAMES,
    file_identity, inventory, safe_relative, verify_bundle, verify_payload, verify_seal, worker_program,
)


ROOT = Path(__file__).resolve().parents[2]


def _put(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(path, value)


def _reseal(root: Path, binding: dict[str, Any]) -> None:
    manifest = {"schema_version": 1, "bundle_kind": BUNDLE_KIND, "evidence_stage": "DEVELOPMENT_ONLY",
                "qualification_status": "unqualified", "binding": binding, "binding_digest": sha256_json(binding),
                "files": inventory(root, exclude=frozenset({"bundle-manifest.json", "COMPLETE"}))}
    manifest["manifest_digest"] = sha256_json(manifest)
    for leaf in ("bundle-manifest.json", "COMPLETE"):
        path = root / leaf
        if path.exists():
            path.unlink()
    _put(root / "bundle-manifest.json", manifest)
    (root / "COMPLETE").write_bytes((str(manifest["manifest_digest"]) + "\n").encode("ascii"))


def _sealed_shell(root: Path) -> dict[str, Any]:
    """Only a container unit fixture, never a numerical worker or pass oracle."""
    root.mkdir()
    binding: dict[str, Any] = {"kind": "unit-test-no-execution-identity"}
    report = {"comparison_records": {"path": "comparisons.jsonl", "sha256": sha256_bytes(b""), "count": 0}}
    _put(root / "numerical-report.json", report)
    (root / "comparisons.jsonl").write_bytes(b"")
    (root / "reader/illm").mkdir(parents=True)
    (root / "reader/illm/__init__.py").write_bytes(b"# unit-test container only\n")
    for name in AUTHORITY_NAMES:
        (root / "authorities").mkdir(exist_ok=True)
        (root / "authorities" / name).write_bytes(b"unit-container-placeholder")
    for name in WORKER_NAMES:
        worker_root = root / "workers" / name
        _put(worker_root / "worker.json", {"binding": binding, "binding_digest": sha256_json(binding),
                                           "process": {"python_executable": "python-unit"}})
        for leaf in ("COMPLETE", "pack/data.bin", "pack/index.jsonl", "pack/manifest.json", "pack/COMPLETE"):
            path = worker_root / leaf
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unit-container-placeholder")
    commands = []
    reader_result = {"schema_version": 1, "report_digest": sha256_json(report),
                     "comparison_records": report["comparison_records"], "module_files": {"illm": "illm/__init__.py"},
                     "forbidden_import_attempts": [], "forbidden_loaded_modules": [], "network_events": []}
    (root / "logs").mkdir()
    driver_spec = importlib.util.spec_from_file_location("unit_raw_driver_program", ROOT / "scripts/validate_policy_cpu_raw.py")
    assert driver_spec is not None and driver_spec.loader is not None
    driver = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver)
    roles = {"native": ("runtime", "native"), "candidate": ("runtime", "candidate"),
             "formula": ("formula", None), "mutations": ("mutation", None),
             "reproduction": ("runtime", "candidate-reproduction")}
    for name in COMMAND_NAMES:
        (root / "logs" / f"{name}.stdout.log").write_bytes(canonical_json_bytes(reader_result) + b"\n")
        (root / "logs" / f"{name}.stderr.log").write_bytes(b"")
        if name in roles:
            module, role = roles[name]
            argv = ["python-unit", "-I", "-X", "utf8", "-c", worker_program(module), "/unit-project/src",
                    "--project-root", "/unit-project", "--output", "/unit-evidence/workers/" + name]
            if role is not None:
                argv.extend(("--role", role))
            cwd = "/unit-project"
        else:
            reader = "/unit-project/src" if name == "reader-source" else "/unit-evidence/reader"
            argv = ["python-unit", "-I", "-S", "-X", "utf8", "-c", driver._READER_PROGRAM,
                    reader, "/unit-evidence", "payload"]
            cwd = "/unit-evidence/authorities" if name == "reader-source" else "/unit-evidence/reader"
        command = {"label": name, "argv": argv, "cwd": cwd,
                   "returncode": 0, **{stream: file_identity(root / "logs" / f"{name}.{stream}.log",
                                                           f"logs/{name}.{stream}.log")
                                      for stream in ("stdout", "stderr")}}
        _put(root / "logs" / f"{name}.command.json", command)
        commands.append(command)
    _put(root / "commands.json", commands)
    _reseal(root, binding)
    return binding


def test_seal_is_closed_and_read_only_not_a_numerical_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "bundle"
    _sealed_shell(root)
    before = inventory(root)
    assert verify_seal(root)["qualification_status"] == "unqualified"
    sentinel = {"numeric_test_boundary": "not-executed"}
    monkeypatch.setattr("illm.policy_cpu_raw_bundle.verify_payload", lambda _: sentinel)
    assert verify_bundle(root) is sentinel
    assert inventory(root) == before


@pytest.mark.parametrize("change", ["extra", "changed", "missing", "incomplete", "empty-dir"])
def test_container_mutations_fail_closed(tmp_path: Path, change: str) -> None:
    root = tmp_path / "bundle"
    _sealed_shell(root)
    if change == "extra":
        (root / "extra.json").write_bytes(b"{}")
    elif change == "changed":
        (root / "reader/illm/__init__.py").write_bytes(b"# altered\n")
    elif change == "missing":
        (root / "workers/native/pack/data.bin").unlink()
    elif change == "incomplete":
        (root / "INCOMPLETE").write_bytes(b"interrupted\n")
    else:
        (root / "unlisted-empty").mkdir()
    with pytest.raises(NumericalEvidenceError):
        verify_seal(root)


def test_resigning_extra_member_does_not_extend_closed_catalog(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    binding = _sealed_shell(root)
    (root / "extra.json").write_bytes(b"{}")
    _reseal(root, binding)
    with pytest.raises(NumericalEvidenceError, match="catalog"):
        verify_seal(root)


@pytest.mark.parametrize("relative", ["../a", "/a", "a//b", "a/./b", "a/../b", "C:/a", "a\\b", ""])
def test_member_paths_are_closed_posix(relative: str) -> None:
    with pytest.raises(NumericalEvidenceError):
        safe_relative(relative)


def test_payload_rejects_qualification_without_recomputing(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _sealed_shell(root)
    for name, source in (("protocol.md", "v0.2-numerical-policy-protocol.md"),
                         ("approval.json", "v0.2-policy-approval.json")):
        (root / "authorities" / name).write_bytes((ROOT / "docs/implementation" / source).read_bytes())
    path = root / "numerical-report.json"
    path.write_bytes(canonical_json_bytes({"report_kind": "illm-stateful-numerical-policy-validation-v1",
                                         "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "qualified",
                                         "status": "pass"}))
    with pytest.raises(NumericalEvidenceError, match="cannot assert"):
        verify_payload(root)


def test_plain_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    original, alias = tmp_path / "source", tmp_path / "alias"
    original.write_bytes(b"raw")
    try:
        os.link(original, alias)
    except OSError:
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(NumericalEvidenceError, match="hard-link"):
        file_identity(alias, "alias")


def test_bundle_imports_without_tensor_packages_or_site() -> None:
    program = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "import illm.policy_cpu_raw_bundle;"
        "assert not any(name.split('.')[0] in ('torch','transformers','safetensors','tokenizers') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(ROOT / "src")],
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()


def test_collection_requires_source_bound_environment_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = ROOT / "scripts/validate_policy_cpu_raw.py"
    spec = importlib.util.spec_from_file_location("illm_raw_driver_unit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    target = tmp_path / "not-created"
    with pytest.raises(NumericalEvidenceError, match="source-bound"):
        module.collect(ROOT, target)
    assert not target.exists()


def test_individual_transcript_cannot_be_resigned_to_conflict(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    binding = _sealed_shell(root)
    path = root / "logs/native.command.json"
    value = json.loads(path.read_bytes())
    value["returncode"] = 9
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    _reseal(root, binding)
    with pytest.raises(NumericalEvidenceError, match="individual"):
        verify_seal(root)


@pytest.mark.parametrize("change", ["program", "role", "output", "reader-program", "reader-cwd"])
def test_resigned_transcript_cannot_change_execution_shape(tmp_path: Path, change: str) -> None:
    root = tmp_path / "bundle"
    binding = _sealed_shell(root)
    commands = json.loads((root / "commands.json").read_bytes())
    index = 5 if change.startswith("reader-") else 0
    command = commands[index]
    if change == "program":
        command["argv"][5] = "print('not a worker')"
    elif change == "role":
        command["argv"][-1] = "candidate"
    elif change == "output":
        command["argv"][10] = "/foreign/native"
    elif change == "reader-program":
        command["argv"][6] = "print('not a reader')"
    else:
        command["cwd"] = "/foreign"
    for path, value in ((root / "commands.json", commands),
                        (root / "logs" / f"{command['label']}.command.json", command)):
        path.write_bytes(canonical_json_bytes(value) + b"\n")
    _reseal(root, binding)
    with pytest.raises(NumericalEvidenceError, match="command"):
        verify_seal(root)


def test_complete_hardlink_rejected_before_read(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _sealed_shell(root)
    external = tmp_path / "external-marker"
    external.write_bytes((root / "COMPLETE").read_bytes())
    (root / "COMPLETE").unlink()
    try:
        os.link(external, root / "COMPLETE")
    except OSError:
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(NumericalEvidenceError, match="hard-link"):
        verify_seal(root)


def test_all_command_interpreters_cannot_be_resigned_away_from_workers(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    binding = _sealed_shell(root)
    commands = json.loads((root / "commands.json").read_bytes())
    for command in commands:
        command["argv"][0] = "forged-python"
        (root / "logs" / f"{command['label']}.command.json").write_bytes(canonical_json_bytes(command) + b"\n")
    (root / "commands.json").write_bytes(canonical_json_bytes(commands) + b"\n")
    _reseal(root, binding)
    with pytest.raises(NumericalEvidenceError, match="actual worker process"):
        verify_seal(root)

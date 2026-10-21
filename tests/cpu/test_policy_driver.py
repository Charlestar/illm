"""CPU-only numerical-policy driver tests; none of these are release evidence."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_numerical_policy as driver


_AUTHORITY_FILES = (
    "docs/implementation/v0.2-numerical-policy-protocol.md",
    "docs/implementation/v0.2-policy-approval.json",
    "docs/implementation/v0.2-spec-v3-amendment-draft.md",
    "docs/implementation/v0.2-adr-007-draft.md",
    "docs/implementation/v0.2-policy-inputs-draft.json",
)


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, driver.ToolchainContext]:
    project = tmp_path / "project"
    (project / "out").mkdir(parents=True)
    for relative in _AUTHORITY_FILES:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    context = driver.ToolchainContext(
        project_root=project,
        environment_root=project / ".illm/envs/unit",
        python_executable=Path(sys.executable).resolve(),
        environment_id="11111111-1111-4111-8111-111111111111",
        lock_digest="a" * 64,
        manifest_digest="b" * 64,
        project_root_digest="c" * 64,
        python_version="3.12.10",
        runtime_profile_id="unit-cpu",
    )
    monkeypatch.setattr(driver, "PROJECT_ROOT", project)
    monkeypatch.setattr(driver, "_load_toolchain", lambda root: context)
    return project, context


def _arguments(project: Path, stage: str, output_name: str) -> list[str]:
    return [
        "--stage",
        stage,
        "--protocol",
        "docs/implementation/v0.2-numerical-policy-protocol.md",
        "--approval",
        "docs/implementation/v0.2-policy-approval.json",
        "--attempt",
        "out/attempt.json",
        "--output",
        f"out/{output_name}",
    ]


def _fake_cpu_process(project: Path, *, fault: str | None = None) -> Any:
    def process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert Path(command[0]).resolve() == Path(sys.executable).resolve()
        assert kwargs["cwd"] == project
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        stdout = b"synthetic unit stdout\n"
        if "scripts/check_quality.py" in command:
            target = Path(command[command.index("--output") + 1])
            quality = {
                "schema_version": 0,
                "checked_python_files": 1,
                "issues": ["synthetic fault"] if fault == "quality" else [],
                "type_check": {
                    "tool": "mypy",
                    "version": "1.18.2",
                    "returncode": 0,
                    "status": "pass",
                    "output": ["Success: synthetic unit check"],
                },
                "status": "fail" if fault == "quality" else "pass",
            }
            stdout = (json.dumps(quality, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            target.write_bytes(stdout)
        elif "pytest" in command:
            target = Path(command[command.index("--junitxml") + 1])
            if fault == "junit-no-testcase":
                junit = '<testsuite tests="1" failures="0" errors="0" skipped="0"/>\n'
            else:
                junit = (
                    '<testsuite tests="1" failures="0" errors="0" skipped="0">'
                    '<testcase classname="unit.synthetic" name="test_pass"/></testsuite>\n'
                )
            target.write_text(junit, encoding="utf-8")
            scratch = Path(command[command.index("--basetemp") + 1])
            scratch.mkdir()
            (scratch / "synthetic-unit-scratch.txt").write_text("not evidence\n", encoding="ascii")
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    return process


def _run_inputs_cpu(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0
    monkeypatch.setattr(subprocess, "run", _fake_cpu_process(project))
    assert driver.run_stage(_arguments(project, "cpu", "cpu")) == 0


@pytest.mark.parametrize("stage", ["gpu-pilot", "gpu-correctness", "gpu-benchmarks"])
def test_gpu_stage_rejects_before_project_or_torch_import(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    imported: list[str] = []
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_value: dict[str, object] | None = None,
        locals_value: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch" or name.startswith("torch.") or name.startswith("illm"):
            imported.append(name)
            raise AssertionError(f"GPU guard imported forbidden module: {name}")
        return original_import(name, globals_value, locals_value, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    arguments = [
        "--stage", stage,
        "--protocol", "missing-protocol",
        "--approval", "missing-approval",
        "--attempt", "outside-attempt",
        "--output", "outside-output",
    ]
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(arguments)
    assert raised.value.code == "GPU_NOT_AUTHORIZED"
    assert imported == []
    assert driver.main(arguments) == 77


def test_inputs_cpu_verify_are_ordered_portable_and_unqualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    imported: list[str] = []
    original_import = builtins.__import__

    def observe_import(
        name: str,
        globals_value: dict[str, object] | None = None,
        locals_value: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        imported.append(name)
        return original_import(name, globals_value, locals_value, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", observe_import)
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0
    # The input registry may traverse the stdlib-only model specification
    # package, but it must not import the tensor runtime.
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)

    copied = (project / "out/inputs/policy-inputs.json").read_bytes()
    assert copied == (project / "docs/implementation/v0.2-policy-inputs-draft.json").read_bytes()
    for dtype, digest in {
        "bf16": "c90519d8e1de749be6727cbb898e42b78a189d2f4c813e224dc9b343ce762497",
        "fp16": "b3a7f22a649b70d24e1014089ac5d5f8856cb5cb4fb7bdac9f469685cb171cb4",
        "fp32": "4e15f0d894637ef4a9533ee4c31adcc18de0b7fddefde98efe2798b9a2a29553",
    }.items():
        assert driver._sha256_file(project / f"out/inputs/candidate-policies/{dtype}.json") == digest

    monkeypatch.setattr(subprocess, "run", _fake_cpu_process(project))
    assert driver.run_stage(_arguments(project, "cpu", "cpu")) == 0
    assert driver.run_stage(_arguments(project, "verify", "verify")) == 0

    journal_raw = (project / "out/attempt.json").read_text(encoding="utf-8")
    journal = json.loads(journal_raw)
    assert [record["stage"] for record in journal["stages"]] == ["inputs", "cpu", "verify"]
    assert [record["sequence"] for record in journal["stages"]] == [1, 2, 3]
    assert journal["status"] == "verified_development_only_unqualified"
    assert journal["evidence_mode"] == "DEVELOPMENT_ONLY"
    assert journal["qualification_tier"] == "unqualified"
    assert str(project.resolve()) not in journal_raw
    assert all(record["output_root"].startswith("out/") for record in journal["stages"])

    cpu = json.loads((project / "out/cpu/cpu-report.json").read_text(encoding="utf-8"))
    assert cpu["kind"] == "illm-policy-cpu-development-gate-v1"
    assert cpu["status"] == "pass"
    assert cpu["qualification_tier"] == "unqualified"
    assert cpu["gpu_result"] == "not_run"
    assert cpu["raw_tensor_evidence"] == "not_captured"
    assert [record["phase"] for record in cpu["commands"]] == ["quality", "pytest", "example"]
    assert all(record["argv"][0] == "{finalized_python}" for record in cpu["commands"])
    assert str(project.resolve()) not in json.dumps(cpu, ensure_ascii=False)
    assert not (project / "out/cpu/pytest-temp").exists()

    verified = json.loads((project / "out/verify/verification-report.json").read_text(encoding="utf-8"))
    assert [record["stage"] for record in verified["verified_stages"]] == ["inputs", "cpu"]
    for root in (project / "out/inputs", project / "out/cpu", project / "out/verify"):
        assert (root / "COMPLETE").is_file()
        assert (root / "checksums.json").is_file()
        assert not (root / "INCOMPLETE").exists()
    assert not (project / "out/attempt.json.lock").exists()


def test_cpu_failure_preserves_incomplete_and_does_not_advance_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0

    def fail_quality(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 2, b"", b"synthetic failure\n")

    monkeypatch.setattr(subprocess, "run", fail_quality)
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "cpu", "failed-cpu"))
    assert raised.value.code == "CPU_PHASE_FAILED"
    output = project / "out/failed-cpu"
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()
    assert (output / "logs/quality.stderr.txt").read_text(encoding="utf-8") == "synthetic failure\n"
    journal = json.loads((project / "out/attempt.json").read_text(encoding="utf-8"))
    assert [record["stage"] for record in journal["stages"]] == ["inputs"]


@pytest.mark.parametrize("fault", ["quality", "junit-no-testcase"])
def test_cpu_rejects_forged_quality_or_unaccounted_junit_before_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0
    monkeypatch.setattr(subprocess, "run", _fake_cpu_process(project, fault=fault))
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "cpu", f"failed-{fault}"))
    assert raised.value.code == "EVIDENCE_INVALID"
    output = project / f"out/failed-{fault}"
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()
    journal = json.loads((project / "out/attempt.json").read_text(encoding="utf-8"))
    assert [record["stage"] for record in journal["stages"]] == ["inputs"]


def test_cpu_rejects_phase_log_digest_drift_before_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0
    monkeypatch.setattr(subprocess, "run", _fake_cpu_process(project))
    original = driver._execute_phase

    def corrupt_record(name: str, command: list[str], **kwargs: Any) -> dict[str, object]:
        record = original(name, command, **kwargs)
        if name == "example":
            record["stdout_sha256"] = "0" * 64
        return record

    monkeypatch.setattr(driver, "_execute_phase", corrupt_record)
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "cpu", "failed-phase-log"))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert (project / "out/failed-phase-log/INCOMPLETE").is_file()
    assert not (project / "out/failed-phase-log/COMPLETE").exists()


def test_cpu_validator_rejects_report_not_linked_to_execution_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, context = _project(tmp_path, monkeypatch)
    _run_inputs_cpu(project, monkeypatch)
    report_path = project / "out/cpu/cpu-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["required_artifacts"]["junit"]["sha256"] = "0" * 64
    report["commands"][2]["argv"] = ["{finalized_python}", "forged-example.py"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    material = driver._load_input_material(project)
    with pytest.raises(driver.DriverFailure) as raised:
        driver._validate_cpu_stage(project / "out/cpu", context, material)
    assert raised.value.code == "EVIDENCE_INVALID"


@pytest.mark.parametrize("mutation", ["qualification_result", "toolchain"])
def test_verify_rejects_resealed_inputs_report_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    _run_inputs_cpu(project, monkeypatch)
    root = project / "out/inputs"
    report_path = root / "inputs-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "qualification_result":
        report["qualification_result"] = "pass"
    else:
        report["toolchain"]["environment_id"] = "forged-environment"
    driver._atomic_json(report_path, report)

    (root / "checksums.json").unlink()
    (root / "COMPLETE").unlink()
    driver._write_exclusive(root / "INCOMPLETE", b"synthetic hostile reseal\n")
    inventory = driver._seal_output(root, "inputs")
    attempt_path = project / "out/attempt.json"
    journal = json.loads(attempt_path.read_text(encoding="utf-8"))
    journal["stages"][0]["files_digest"] = inventory["files_digest"]
    journal["stages"][0]["checksums_sha256"] = driver._sha256_file(root / "checksums.json")
    driver._set_journal_digest(journal)
    driver._atomic_json(attempt_path, journal)

    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "verify", f"failed-inputs-{mutation}"))
    assert raised.value.code == "EVIDENCE_INVALID"
    output = project / f"out/failed-inputs-{mutation}"
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()


def test_verify_detects_changed_sealed_file_and_remains_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    _run_inputs_cpu(project, monkeypatch)
    target = project / "out/inputs/policy-inputs.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "verify", "failed-verify"))
    assert raised.value.code == "EVIDENCE_INVALID"
    assert (project / "out/failed-verify/INCOMPLETE").is_file()
    assert not (project / "out/failed-verify/COMPLETE").exists()
    journal = json.loads((project / "out/attempt.json").read_text(encoding="utf-8"))
    assert [record["stage"] for record in journal["stages"]] == ["inputs", "cpu"]


def test_existing_attempt_lock_is_preserved_without_authority_or_output_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    lock = project / "out/attempt.json.lock"
    lock.write_bytes(b"foreign-or-stale-lock-must-remain")
    monkeypatch.setattr(
        driver,
        "_load_toolchain",
        lambda root: (_ for _ in ()).throw(AssertionError("toolchain must not run while locked")),
    )
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "inputs", "never-created"))
    assert raised.value.code == "ATTEMPT_LOCKED"
    assert lock.read_bytes() == b"foreign-or-stale-lock-must-remain"
    assert not (project / "out/never-created").exists()


def test_owned_lock_initialization_failure_closes_and_removes_only_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"

    def fail_fsync(descriptor: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    lock = driver._AttemptLock(attempt)
    with pytest.raises(driver.DriverFailure) as raised:
        lock.__enter__()
    assert raised.value.code == "ATTEMPT_LOCK_FAILED"
    assert not lock.path.exists()


def test_owned_lock_release_failure_is_not_silently_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt.json"
    lock = driver._AttemptLock(attempt)
    lock.__enter__()
    original = Path.unlink

    def refuse_owned_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == lock.path:
            raise PermissionError("synthetic lock cleanup failure")
        original(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse_owned_unlink)
    with pytest.raises(driver.DriverFailure) as raised:
        lock.release()
    assert raised.value.code == "ATTEMPT_LOCK_FAILED"
    assert lock.path.is_file()
    assert lock.path.read_bytes() == lock._token


def test_lock_exit_failure_rolls_back_stage_and_marks_output_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    original = driver._AttemptLock.release

    def fail_after_release(lock: driver._AttemptLock) -> None:
        original(lock)
        raise driver.DriverFailure("ATTEMPT_LOCK_FAILED", "synthetic post-release failure")

    monkeypatch.setattr(driver._AttemptLock, "release", fail_after_release)
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "inputs", "failed-lock-exit"))
    assert raised.value.code == "ATTEMPT_LOCK_FAILED"
    output = project / "out/failed-lock-exit"
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()
    journal = json.loads((project / "out/attempt.json").read_text(encoding="utf-8"))
    assert journal["stages"] == []


def test_output_marker_initialization_failure_leaves_no_unmarked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    original = driver._write_exclusive

    def fail_marker(path: Path, value: bytes) -> None:
        if path.name == "INCOMPLETE":
            raise driver.DriverFailure("EVIDENCE_WRITE_FAILED", "synthetic marker failure")
        original(path, value)

    monkeypatch.setattr(driver, "_write_exclusive", fail_marker)
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "inputs", "failed-marker"))
    assert raised.value.code == "EVIDENCE_WRITE_FAILED"
    assert not (project / "out/failed-marker").exists()


def test_authority_tamper_fails_before_journal_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    approval = project / "docs/implementation/v0.2-policy-approval.json"
    approval.write_bytes(approval.read_bytes() + b" ")
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "inputs", "never-created"))
    assert raised.value.code == "AUTHORITY_MISMATCH"
    assert not (project / "out/attempt.json").exists()
    assert not (project / "out/never-created").exists()


def test_toolchain_or_source_drift_rejects_later_stage_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, context = _project(tmp_path, monkeypatch)
    assert driver.run_stage(_arguments(project, "inputs", "inputs")) == 0
    drifted = driver.ToolchainContext(
        project_root=context.project_root,
        environment_root=context.environment_root,
        python_executable=context.python_executable,
        environment_id=context.environment_id,
        lock_digest=context.lock_digest,
        manifest_digest="d" * 64,
        project_root_digest="e" * 64,
        python_version=context.python_version,
        runtime_profile_id=context.runtime_profile_id,
    )
    monkeypatch.setattr(driver, "_load_toolchain", lambda root: drifted)
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(_arguments(project, "cpu", "never-created"))
    assert raised.value.code == "JOURNAL_IDENTITY_DRIFT"
    assert not (project / "out/never-created").exists()


def test_attempt_and_output_must_be_plain_paths_strictly_under_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project(tmp_path, monkeypatch)
    arguments = _arguments(project, "inputs", "inside")
    arguments[arguments.index("--attempt") + 1] = str(tmp_path / "outside.json")
    with pytest.raises(driver.DriverFailure) as raised:
        driver.run_stage(arguments)
    assert raised.value.code == "PATH_ESCAPE"
    assert not (project / "out/inside").exists()

    link = project / "out/link"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    arguments = _arguments(project, "inputs", "link/output")
    with pytest.raises(driver.DriverFailure) as linked:
        driver.run_stage(arguments)
    assert linked.value.code == "PATH_ESCAPE"


def test_non_gpu_stage_requires_finalized_manifest_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    monkeypatch.delenv("ILLM_LOCK_DIGEST", raising=False)
    with pytest.raises(driver.DriverFailure) as raised:
        driver._load_toolchain(ROOT)
    assert raised.value.code == "TOOLCHAIN_MISMATCH"

"""V0.2 profile sealer/portable-reader contract tests.

Synthetic bundles in this file are explicitly non-release fixtures.  They use
``synthetic_test_fixture``/``synthetic_pass`` and the public reader must reject
them even when their internal checksums are self-consistent.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Any, Iterator, Mapping, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import metric_registry, registry_conformance
from illm.reference_qualification import (
    BASE_LOCKS,
    BOOTSTRAP_FILES,
    PROJECT_FILES,
    RUNTIME_LOCKS,
    _inventory,
    _source_provenance,
)
from illm.stateful_gate_contract import phase_parameters
import illm.stateful_qualification as qualification


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _locked_versions(provenance: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in BASE_LOCKS:
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line.strip())
            if match:
                versions[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    for item in cast(list[dict[str, Any]], runtime["artifacts"]):
        versions[str(item["name"])] = str(item["version"])
    versions["illm"] = str(provenance["project_version"])
    return dict(sorted(versions.items()))


def _synthetic_authority(tmp_path: Path, runtime_id: str, *, version: str = "0.2.0.dev0") -> tuple[dict[str, Any], dict[str, Any]]:
    source = tmp_path / "synthetic-source-never-executed"
    selected = {*PROJECT_FILES, *BOOTSTRAP_FILES, f"runtime/artifacts/{runtime_id}.json"}
    for name in selected:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    pyproject = (source / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = re.sub(r'^version = "[^"]+"', f'version = "{version}"', pyproject, count=1, flags=re.MULTILINE)
    (source / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    wheel = source / "vendor" / "wheels" / "synthetic-unit-only.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"not-an-installable-wheel\n")
    provenance = _source_provenance(source)
    runtime = json.loads((source / "runtime" / "artifacts" / f"{runtime_id}.json").read_text(encoding="utf-8"))
    versions = _locked_versions(provenance, runtime)
    system = str(runtime["target"]["os"])
    toolchain: dict[str, Any] = {
        "schema_version": 1,
        "finalized": True,
        "profile": "synthetic-test-fixture-never-executed",
        "os": system,
        "arch": {"Windows": "AMD64", "Linux": "x86_64"}[system],
        "python_version": runtime["target"]["python_validation_patch"],
        "project_root": "/absent/synthetic-test-source",
        "environment_root": "/absent/synthetic-test-environment",
        "python_executable": "/absent/synthetic-test-python",
        "entrypoints": {"illm": "/absent/synthetic-test-illm"},
        "installed_packages": versions,
        "environment_id": "synthetic-test-fixture-never-executed",
        "bootstrap_digest": provenance["source_identity"]["bootstrap_digest"],
        "lock_digest": sha256_json({
            "base_lock_digest": provenance["source_identity"]["base_lock_digest"],
            "runtime_lock_digest": RUNTIME_LOCKS[runtime_id],
        }),
        "project_root_digest": provenance["source_identity"]["project_root_digest"],
        "runtime_profile_id": runtime_id,
        "runtime_lock_digest": RUNTIME_LOCKS[runtime_id],
        "runtime_artifacts": [
            {key: item[key] for key in ("name", "version", "filename", "sha256", "size_bytes")}
            for item in runtime["artifacts"]
        ],
        "artifact_cache": "/absent/synthetic-test-cache",
        "runtime_probe": {
            "torch": versions["torch"],
            "transformers": versions["transformers"],
            "safetensors": versions["safetensors"],
            "tokenizers": versions["tokenizers"],
            "device": "cpu" if not runtime_id.endswith("cu126") else "cuda",
            "cuda_build": None if not runtime_id.endswith("cu126") else "12.6",
            "gpu_name": None if not runtime_id.endswith("cu126") else "Synthetic Test GPU",
            "device_capability": None if not runtime_id.endswith("cu126") else [8, 0],
            "compiled_arches": [] if not runtime_id.endswith("cu126") else ["sm_80"],
            "smoke_sum": 3.0,
        },
    }
    toolchain["manifest_digest"] = sha256_json(toolchain)
    return provenance, toolchain


def _junit(path: Path) -> None:
    suite = ET.Element(
        "testsuite", tests=str(len(qualification.CRITICAL_TESTS)), failures="0", errors="0", skipped="0", time="1.0",
    )
    for name in sorted(qualification.CRITICAL_TESTS):
        ET.SubElement(suite, "testcase", name=name, classname="synthetic.unit.fixture")
    ET.ElementTree(suite).write(path, encoding="utf-8")


def _components() -> dict[str, Any]:
    return {
        "metric_registry": registry_conformance(),
        "stateful": {"tiny-full": {"status": "pass", "fixture": "synthetic_test_fixture"}},
        "exercise": {"status": "pass", "fixture": "synthetic_test_fixture"},
        "learning": {"status": "pass", "fixture": "synthetic_test_fixture"},
        "demo_capture_manifest_digest": "d" * 64,
    }


def _synthetic_gate_argv(phase: str, parameters: Mapping[str, Any], runtime_id: str) -> list[str]:
    project = "/absent/synthetic-v02-project"
    gate = "/absent/synthetic-v02-gate"
    interpreter = "/absent/synthetic-v02-python"
    if phase == "bootstrap":
        return [
            interpreter, f"{project}/scripts/bootstrap.py", "--profile", "synthetic-v02-gate",
            "--runtime-profile", runtime_id, "--locked", "--offline", "--require-new-environment",
            "--write-manifest", f"{gate}/toolchain.json", "--write-manifest-digest",
            f"{gate}/toolchain.sha256",
        ]
    prefix = [
        interpreter, f"{project}/scripts/in_env.py", "--manifest", f"{gate}/toolchain.json",
        "--expected-manifest-digest-file", f"{gate}/toolchain.sha256", "--",
    ]
    if phase == "quality":
        child = ["python", "scripts/check_quality.py", "--output", f"{gate}/quality.json"]
    elif phase == "pytest":
        child = [
            "python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--basetemp",
            f"{gate}/pytest-temp", "--junitxml", f"{gate}/junit.xml", "--tb=short",
        ]
    elif phase == "verify":
        child = ["python", "-m", "illm", "verify", "--fixture", "tiny-decoder"]
    elif phase == "run":
        child = [
            "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--prompt", "hello", "--max-new-tokens", "8", "--stream", "--request-id", "req-0001",
            "--kv-mode", "contiguous", "--telemetry", "request", "--capture", f"{gate}/demo-capture",
        ]
    elif phase == "replay-trace":
        child = ["python", "-m", "illm", "replay", f"{gate}/demo-capture", "--mode", "trace"]
    elif phase == "inspect-request":
        child = [
            "python", "-m", "illm", "inspect", "request", f"{gate}/demo-capture", "--request", "req-0001",
        ]
    elif phase == "stateful-full":
        child = [
            "python", "scripts/validate_stateful.py", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--scope", "full", "--output", f"{gate}/stateful-full",
        ]
    elif phase in {"stateful-bf16", "stateful-fp16"}:
        dtype = phase.removeprefix("stateful-")
        child = [
            "python", "scripts/validate_stateful.py", "--model", str(parameters["model_profile"]),
            "--device", "cuda:0", "--scope", "real-smoke", "--output", f"{gate}/stateful-{dtype}",
        ]
    elif phase == "exercise":
        child = ["python", "scripts/exercise_streaming.py", "--output", f"{gate}/exercise"]
    elif phase in {"learning-build", "learning-verify"}:
        child = [
            "python", "scripts/render_streaming_learning.py", f"{gate}/exercise/captures/cancel-after-launch",
            "--output", f"{gate}/learning",
        ]
        if phase == "learning-verify":
            child.append("--verify-existing")
    elif phase.startswith("benchmark-"):
        dtype, suite = phase.removeprefix("benchmark-").split("-", 1)
        model = {"bf16": "configs/models/v0_2.yaml", "fp16": "configs/models/v0_2_fp16.yaml"}[dtype]
        child = [
            "python", "scripts/benchmark_streaming.py", "--model", model, "--device", "cuda:0",
            "--target", "real", "--suite", suite, "--warmups", "3", "--repetitions", "7",
            "--output", f"{gate}/benchmark-{dtype}-{suite}",
        ]
    else:
        raise AssertionError(f"unknown fixture phase {phase}")
    return [*prefix, *child]


def _synthetic_bundle(tmp_path: Path, *, version: str = "0.2.0.dev0") -> tuple[Path, dict[str, Any], dict[str, str], dict[str, Any]]:
    root = tmp_path / "synthetic-profile-never-release-eligible"
    root.mkdir(parents=True)
    provenance, toolchain = _synthetic_authority(tmp_path, "py312-win-amd64-cpu", version=version)
    _write(root / "source-provenance.json", provenance)
    _write(root / "toolchain.json", toolchain)
    locks = root / "locks"
    locks.mkdir()
    for name in BASE_LOCKS:
        shutil.copy2(ROOT / name, locks / name)
    source_pyproject = tmp_path / "synthetic-source-never-executed" / "pyproject.toml"
    shutil.copy2(source_pyproject, locks / "pyproject.toml")
    shutil.copy2(ROOT / "runtime" / "artifacts" / "py312-win-amd64-cpu.json", locks / "runtime-lock.json")
    _junit(root / "junit.xml")
    source_files = cast(dict[str, Any], provenance["project_files"])
    _write(root / "quality.json", {
        "schema_version": 0,
        "status": "pass",
        "issues": [],
        "checked_python_files": sum(name.startswith("src/illm/") and name.endswith(".py") for name in source_files),
        "type_check": {"tool": "mypy", "version": toolchain["installed_packages"]["mypy"],
                       "status": "pass", "returncode": 0, "output": ["synthetic fixture"]},
    })
    _write(root / "metrics" / "metric-registry-v1.json", metric_registry())
    _write(root / "metrics" / "metric-registry-conformance.json", registry_conformance())
    for relative in (
        "stateful/tiny-full/fixture.txt", "exercise/fixture.txt", "demo-capture/fixture.txt",
        "learning/fixture.txt", "gate/fixture.txt",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic_test_fixture\n", encoding="ascii")
    components = _components()
    component_digests = qualification._component_digests(components)
    facts = _inventory(root)
    identity = cast(dict[str, str], provenance["source_identity"])
    evidence_digest = sha256_json({
        "schema_version": 2,
        "profile_id": qualification.SYNTHETIC_PROFILE_ID,
        "source_identity": identity,
        "components": component_digests,
        "files": facts,
    })
    profile: dict[str, Any] = {
        "schema_version": 2,
        "profile_id": qualification.SYNTHETIC_PROFILE_ID,
        "simulated_profile_id": "state-machine-cpu-v1",
        "evidence_mode": "synthetic_test_fixture",
        "result": "synthetic_pass",
        "required_for_functional": False,
        "environment_manifest_digest": toolchain["manifest_digest"],
        "evidence_digest": evidence_digest,
        "components": component_digests,
        "reason": "unit_test_only_never_release_eligible",
    }
    profile["profile_digest"] = sha256_json(profile)
    index: dict[str, Any] = {
        "schema_version": 2,
        "profile_id": qualification.SYNTHETIC_PROFILE_ID,
        "simulated_profile_id": "state-machine-cpu-v1",
        "evidence_mode": "synthetic_test_fixture",
        "os": toolchain["os"],
        "arch": toolchain["arch"],
        "python_version": toolchain["python_version"],
        "source_identity": identity,
        "runtime_profile_id": toolchain["runtime_profile_id"],
        "runtime_lock_digest": toolchain["runtime_lock_digest"],
        "files": facts,
        "components": component_digests,
        "evidence_digest": evidence_digest,
    }
    index["index_digest"] = sha256_json(index)
    _write(root / "profile-result.json", profile)
    _write(root / "evidence-index.json", index)
    (root / "COMPLETE").write_text("complete\n", encoding="ascii")
    _write(root / "SHA256SUMS.json", _inventory(root))
    return root, toolchain, identity, components


def _gate_report(root: Path, profile_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    runtime_id = qualification.FORMAL_PROFILE_RUNTIMES[profile_id]
    toolchain = {"manifest_digest": "a" * 64}
    identity = {name: character * 64 for name, character in zip(sorted(qualification.SOURCE_FIELDS), "bcd", strict=True)}
    commands: list[dict[str, Any]] = []
    for phase, parameters in phase_parameters(profile_id).items():
        stdout = root / "logs" / f"{phase}.stdout.txt"
        stderr = root / "logs" / f"{phase}.stderr.txt"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text("{}\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        commands.append({
            "phase": phase,
            "argv": _synthetic_gate_argv(phase, parameters, runtime_id),
            "parameters": parameters,
            "returncode": 0,
            "stdout_file": stdout.relative_to(root).as_posix(),
            "stderr_file": stderr.relative_to(root).as_posix(),
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        })
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_kind": qualification.REPORT_KIND,
        "profile_id": profile_id,
        "runtime_profile_id": runtime_id,
        "status": "pass",
        "toolchain_manifest_digest": toolchain["manifest_digest"],
        "source_identity": identity,
        "fresh_environment": True,
        "offline_bootstrap": True,
        "network_used": "not_observed",
        "network_blocking": "not_enforced",
        "network_policy": "offline_resolution_and_local_model_only",
        "commands": commands,
    }
    report["report_digest"] = sha256_json(report)
    _write(root / "gate-report.json", report)
    return report, toolchain, identity


def _resign_gate(root: Path, report: dict[str, Any]) -> None:
    report.pop("report_digest", None)
    report["report_digest"] = sha256_json(report)
    _write(root / "gate-report.json", report)


def test_profiles_and_shared_phase_contract_are_exact() -> None:
    assert qualification.FORMAL_PROFILE_RUNTIMES == {
        "state-machine-cpu-v1": "py312-win-amd64-cpu",
        "stateful-streaming-single-gpu-v1": "py312-linux-x86_64-cu126",
        "clean-room-release-v1": "py312-linux-x86_64-cpu",
    }
    assert [name for name, _ in qualification._expected_gate_parameters(
        "state-machine-cpu-v1", "py312-win-amd64-cpu",
    )] == list(phase_parameters("state-machine-cpu-v1"))
    assert len(phase_parameters("state-machine-cpu-v1")) == 11
    assert len(phase_parameters("clean-room-release-v1")) == 11
    assert len(phase_parameters("stateful-streaming-single-gpu-v1")) == 9
    bootstrap = _synthetic_gate_argv(
        "bootstrap", phase_parameters("state-machine-cpu-v1")["bootstrap"], "py312-win-amd64-cpu",
    )
    assert bootstrap.count("--require-new-environment") == 1
    assert qualification._verify_bootstrap_argv(bootstrap, "py312-win-amd64-cpu")[2].endswith(
        "/synthetic-v02-gate"
    )


def test_gate_argv_lexically_supports_windows_without_touching_author_paths() -> None:
    runtime = "py312-win-amd64-cpu"
    bootstrap = [
        r"C:\Absent\Python\python.exe", r"C:\Absent\Project\scripts\bootstrap.py",
        "--profile", "synthetic-v02-gate", "--runtime-profile", runtime, "--locked", "--offline",
        "--require-new-environment", "--write-manifest", r"C:\Absent\Gate\toolchain.json",
        "--write-manifest-digest", r"C:\Absent\Gate\toolchain.sha256",
    ]
    interpreter, project, gate = qualification._verify_bootstrap_argv(bootstrap, runtime)
    assert (interpreter, project, gate) == (
        "C:/Absent/Python/python.exe", "C:/Absent/Project", "C:/Absent/Gate",
    )
    parameters = phase_parameters("state-machine-cpu-v1")["quality"]
    qualification._verify_wrapped_argv(
        "quality",
        [
            r"C:\Absent\Python\python.exe", r"C:\Absent\Project\scripts\in_env.py", "--manifest",
            r"C:\Absent\Gate\toolchain.json", "--expected-manifest-digest-file",
            r"C:\Absent\Gate\toolchain.sha256", "--", "python", "scripts/check_quality.py", "--output",
            r"C:\Absent\Gate\quality.json",
        ],
        parameters,
        interpreter=interpreter,
        project_root=project,
        gate_root=gate,
    )


def test_v02_source_reader_accepts_only_v02_package_identity(tmp_path: Path) -> None:
    valid, _, identity, _ = _synthetic_bundle(tmp_path / "valid")
    parsed, _ = qualification._verify_v02_source(valid, identity)
    assert parsed == identity

    old, _, _, _ = _synthetic_bundle(tmp_path / "old", version="0.1.0")
    with pytest.raises(IllmError, match="not V0.2"):
        qualification._verify_v02_source(old, None)


def test_synthetic_fixture_can_exercise_internals_but_public_reader_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, toolchain, identity, components = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(qualification, "_verify_components", lambda *_: components)
    monkeypatch.setattr(qualification, "_verify_gate", lambda *_: None)
    profile, index = qualification._verify_stateful_profile_evidence(
        root, identity, allow_synthetic_test_fixture=True,
    )
    assert profile["evidence_mode"] == "synthetic_test_fixture"
    assert profile["result"] == "synthetic_pass"
    assert profile["required_for_functional"] is False
    assert index["runtime_profile_id"] == toolchain["runtime_profile_id"]
    with pytest.raises(IllmError, match="unknown or non-release"):
        qualification.verify_stateful_profile_evidence(root)


def test_exact_runtime_map_rejects_cross_platform_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _, components = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(qualification, "_verify_components", lambda *_: components)
    monkeypatch.setattr(qualification, "_verify_gate", lambda *_: None)
    with pytest.raises(IllmError, match="requires exact runtime") as raised:
        qualification._verify_facts(root, "clean-room-release-v1", None)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_gate_rejects_bool_numeric_parameter_and_extra_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    monkeypatch.setattr(qualification, "_verify_gate_phase_output", lambda *_args, **_kwargs: None)
    qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})

    report["commands"][0]["parameters"]["locked"] = 1
    report.pop("report_digest")
    report["report_digest"] = sha256_json(report)
    _write(gate / "gate-report.json", report)
    with pytest.raises(IllmError, match="schema/order/parameters"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})

    report["commands"][0]["parameters"]["locked"] = True
    report.pop("report_digest")
    report["report_digest"] = sha256_json(report)
    _write(gate / "gate-report.json", report)
    (gate / "extra.txt").write_text("resigned-extra\n", encoding="utf-8")
    with pytest.raises(IllmError, match="inventory is not exact"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})


def test_gate_tree_excludes_only_pytest_scratch_without_traversing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    evidence = gate / "logs"
    scratch = gate / "pytest-temp"
    evidence.mkdir(parents=True)
    (evidence / "phase.stdout.txt").write_text("pass\n", encoding="utf-8")
    (scratch / "legitimate-empty-tmp-path").mkdir(parents=True)

    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path) -> Iterator[Path]:
        if path == scratch:
            raise AssertionError("pytest scratch contents must not be traversed")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    qualification._check_gate_tree(gate)

    (gate / "unreferenced-empty-evidence").mkdir()
    with pytest.raises(IllmError, match="empty directories"):
        qualification._check_gate_tree(gate)


def test_gate_tree_rejects_nonordinary_pytest_scratch_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    scratch = gate / "pytest-temp"
    scratch.mkdir(parents=True)
    (gate / "evidence.txt").write_text("pass\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def scratch_is_symlink(path: Path) -> bool:
        return path == scratch or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", scratch_is_symlink)
    with pytest.raises(IllmError, match="pytest scratch"):
        qualification._check_gate_tree(gate)


def test_gate_tree_does_not_promote_pytest_symlink_fixture_to_evidence(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    scratch = gate / "pytest-temp"
    scratch.mkdir(parents=True)
    (gate / "evidence.txt").write_text("pass\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("not evidence\n", encoding="utf-8")
    try:
        (scratch / "deliberate-test-symlink").symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit creating symbolic links")

    qualification._check_gate_tree(gate)
    with pytest.raises(IllmError, match="links and escaped paths"):
        qualification._check_tree(gate)


def test_gate_copy_never_copies_or_hashes_pytest_scratch(tmp_path: Path) -> None:
    gate = tmp_path / "gate"
    _gate_report(gate, "state-machine-cpu-v1")
    scratch = gate / "pytest-temp"
    (scratch / "empty-test-temp").mkdir(parents=True)
    destination = tmp_path / "sealed-gate"

    qualification._copy_gate(gate / "gate-report.json", destination)

    assert not (destination / "pytest-temp").exists()
    assert "pytest-temp" not in _inventory(destination)


def test_quality_transcript_cannot_substitute_bool_for_numeric_zero(tmp_path: Path) -> None:
    _write(tmp_path / "quality.json", {"status": "pass", "error_count": 0})
    stdout = tmp_path / "stdout.txt"
    _write(stdout, {"status": "pass", "error_count": False})
    with pytest.raises(IllmError, match="does not bind"):
        qualification._verify_gate_phase_output("quality", stdout, tmp_path, {}, {}, argv=[])


def test_qualification_trees_reject_unhashed_nonregular_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    entry = gate / "logs" / "unhashed-nonregular"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"unit stand-in\n")
    original_is_file = Path.is_file

    def stand_in(path: Path) -> bool:
        return False if path == entry else original_is_file(path)

    monkeypatch.setattr(Path, "is_file", stand_in)
    with pytest.raises(IllmError, match="nonregular"):
        qualification._check_tree(gate)
    with pytest.raises(IllmError, match="nonregular"):
        qualification._check_gate_tree(gate)


def test_gate_phase_order_and_transcript_hashes_are_not_labels_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    report, toolchain, identity = _gate_report(gate, "stateful-streaming-single-gpu-v1")
    monkeypatch.setattr(qualification, "_verify_gate_phase_output", lambda *_args, **_kwargs: None)
    qualification._verify_gate(gate, "stateful-streaming-single-gpu-v1", toolchain, identity, {})
    report["commands"][3], report["commands"][4] = report["commands"][4], report["commands"][3]
    report.pop("report_digest")
    report["report_digest"] = sha256_json(report)
    _write(gate / "gate-report.json", report)
    with pytest.raises(IllmError, match="schema/order/parameters"):
        qualification._verify_gate(gate, "stateful-streaming-single-gpu-v1", toolchain, identity, {})

    report, toolchain, identity = _gate_report(gate, "stateful-streaming-single-gpu-v1")
    stdout = gate / cast(str, report["commands"][3]["stdout_file"])
    stdout.write_text('{"status":"fake-pass"}\n', encoding="utf-8")
    with pytest.raises(IllmError, match="checksum"):
        qualification._verify_gate(gate, "stateful-streaming-single-gpu-v1", toolchain, identity, {})


def test_gate_argv_rejects_substrings_duplicate_flags_and_python_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    monkeypatch.setattr(qualification, "_verify_gate_phase_output", lambda *_args, **_kwargs: None)

    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    report["commands"][0]["argv"][8] = "x--require-new-environment"
    _resign_gate(gate, report)
    with pytest.raises(IllmError, match="bootstrap argv"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})

    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    report["commands"][0]["argv"][1:2] = ["-c", "print('scripts/bootstrap.py --require-new-environment')"]
    _resign_gate(gate, report)
    with pytest.raises(IllmError, match="bootstrap argv"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})

    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    stateful = next(item for item in report["commands"] if item["phase"] == "stateful-full")
    output_index = stateful["argv"].index("--output")
    stateful["argv"][output_index:output_index] = ["--scope", "real-smoke"]
    _resign_gate(gate, report)
    with pytest.raises(IllmError, match="extra, omitted, or reordered"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})


def test_gate_argv_binds_wrapper_boundary_and_artifact_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "gate"
    monkeypatch.setattr(qualification, "_verify_gate_phase_output", lambda *_args, **_kwargs: None)

    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    quality = next(item for item in report["commands"] if item["phase"] == "quality")
    quality["argv"][6] = "--not-a-boundary"
    _resign_gate(gate, report)
    with pytest.raises(IllmError, match="boundary"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})

    report, toolchain, identity = _gate_report(gate, "state-machine-cpu-v1")
    quality = next(item for item in report["commands"] if item["phase"] == "quality")
    quality["argv"][-1] = "/absent/different-gate/quality.json"
    _resign_gate(gate, report)
    with pytest.raises(IllmError, match="exactly bind"):
        qualification._verify_gate(gate, "state-machine-cpu-v1", toolchain, identity, {})


def test_sealer_rejects_unknown_or_nonexact_gate_before_live_claim(tmp_path: Path) -> None:
    with pytest.raises(IllmError, match="unknown V0.2"):
        qualification.create_stateful_profile_evidence(
            profile_id="tiny-reference-cpu-v1", gate_root=tmp_path, output=tmp_path / "never",
        )
    gate = tmp_path / "gate"
    gate.mkdir()
    with pytest.raises(IllmError, match="inventory"):
        qualification.create_stateful_profile_evidence(
            profile_id="state-machine-cpu-v1", gate_root=gate, output=tmp_path / "never",
        )
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize(
    ("phase", "report", "component_key"),
    [
        (
            "stateful-full",
            {
                "status": "pass",
                "scope": "full",
                "report_digest": "a" * 64,
                "formal_evidence_eligible": True,
                "complete_report_field": "stateful-proof",
            },
            "tiny-full",
        ),
        (
            "stateful-bf16",
            {
                "status": "pass",
                "scope": "real-smoke",
                "report_digest": "b" * 64,
                "formal_evidence_eligible": True,
                "complete_report_field": "bf16-proof",
            },
            "bf16",
        ),
        (
            "exercise",
            {
                "status": "pass",
                "report_digest": "c" * 64,
                "development_no_manifest": False,
                "complete_report_field": "lifecycle-proof",
            },
            "exercise",
        ),
        (
            "benchmark-fp16-heldout",
            {
                "status": "pass",
                "report_digest": "d" * 64,
                "development_no_manifest": False,
                "complete_report_field": "benchmark-proof",
            },
            "fp16-heldout",
        ),
    ],
)
def test_gate_accepts_exact_real_producer_summary_or_complete_report(
    tmp_path: Path,
    phase: str,
    report: dict[str, Any],
    component_key: str,
) -> None:
    """Mirror the final JSON objects printed by the three producer CLIs."""

    output = f"/author/gate/{phase}"
    argv = ["python", "producer.py", "--output", output]
    components: dict[str, Any] = {
        "stateful": {component_key: report},
        "exercise": report,
        "benchmarks": {component_key: report},
    }
    if phase.startswith("stateful-"):
        summary = {
            "status": report["status"],
            "scope": report["scope"],
            "report_digest": report["report_digest"],
            "formal_evidence_eligible": report["formal_evidence_eligible"],
            "output": output,
        }
    else:
        summary = {
            "status": report["status"],
            "report_digest": report["report_digest"],
            "development_no_manifest": report["development_no_manifest"],
            "output": output,
        }
    stdout = tmp_path / "stdout.txt"
    for value in (summary, report):
        stdout.write_bytes(canonical_json_bytes(value) + b"\n")
        qualification._verify_gate_phase_output(
            phase,
            stdout,
            root=tmp_path,
            toolchain={},
            components=components,
            argv=argv,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_output", "bool_as_int", "early_object", "duplicate"])
def test_gate_rejects_tampered_or_noncanonical_producer_summary(tmp_path: Path, mutation: str) -> None:
    report: dict[str, Any] = {
        "status": "pass",
        "scope": "full",
        "report_digest": "a" * 64,
        "formal_evidence_eligible": True,
        "complete_report_field": "stateful-proof",
    }
    output = "/author/gate/stateful-full"
    summary: dict[str, Any] = {
        "status": "pass",
        "scope": "full",
        "report_digest": "a" * 64,
        "formal_evidence_eligible": True,
        "output": output,
    }
    if mutation == "missing":
        summary.pop("scope")
    elif mutation == "extra":
        summary["unverified_claim"] = "pass"
    elif mutation == "wrong_output":
        summary["output"] = "/author/gate/different-artifact"
    elif mutation == "bool_as_int":
        summary["formal_evidence_eligible"] = 1
    stdout = tmp_path / "stdout.txt"
    if mutation == "early_object":
        stdout.write_text(json.dumps({"unrelated": True}) + "\n" + json.dumps(summary) + "\n", encoding="utf-8")
    elif mutation == "duplicate":
        stdout.write_text(
            '{"status":"fail","status":"pass","scope":"full","report_digest":"'
            + "a" * 64
            + f'","formal_evidence_eligible":true,"output":"{output}"}}\n',
            encoding="utf-8",
        )
    else:
        stdout.write_bytes(canonical_json_bytes(summary) + b"\n")
    components = {"stateful": {"tiny-full": report}}
    with pytest.raises(IllmError):
        qualification._verify_gate_phase_output(
            "stateful-full",
            stdout,
            root=tmp_path,
            toolchain={},
            components=components,
            argv=["python", "producer.py", "--output", output],
        )


def test_portable_import_does_not_load_torch_or_project_scripts() -> None:
    code = (
        f"import sys;sys.path.insert(0,{str(ROOT / 'src')!r});"
        "import illm.stateful_qualification as q;"
        "assert 'torch' not in sys.modules;"
        "assert 'validate_stateful' not in sys.modules;"
        "assert q.FORMAL_PROFILE_RUNTIMES['state-machine-cpu-v1'].endswith('cpu');"
        "print('portable-pass')"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "portable-pass"


def test_qualify_cli_does_not_import_test_frameworks() -> None:
    source = (ROOT / "scripts" / "qualify_v02.py").read_text(encoding="utf-8")
    assert "import pytest" not in source
    assert "from tests" not in source
    assert qualification._V02_VERSION.fullmatch("0.2.0")
    assert qualification._V02_VERSION.fullmatch("0.2.0.dev7")
    assert qualification._V02_VERSION.fullmatch("0.1.0") is None
    assert platform.system() in {"Windows", "Linux", "Darwin"}

"""Revision-2 policy profile contract tests.

The gate fixtures below are transcript-only unit data.  They do not execute a
phase, construct a public profile, or establish CPU, GPU, or release fitness.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, cast
import xml.etree.ElementTree as ET

import pytest

from illm import stateful_policy_qualification as policy_qualification
from illm import stateful_qualification as legacy_qualification
from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.policy_cpu_numerical_contract import NumericalEvidenceError
from illm.stateful_policy_gate_contract import (
    POLICY_GPU_DEFERRED_REASON,
    POLICY_GPU_PROFILE_ID,
    POLICY_PROFILE_RUNTIME_IDS,
    REQUIRED_RESULT_AXES,
    policy_phase_commands,
    policy_phase_parameters,
)
from illm.stateful_policy_profile_gate import GATE_KIND, verify_policy_profile_gate
from illm.v02_profile_dispatch import verify_v02_profile_evidence


ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = "state-machine-cpu-v1"
RUNTIME_ID = POLICY_PROFILE_RUNTIME_IDS[PROFILE_ID]
PHASES = tuple(policy_phase_parameters(PROFILE_ID))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _dispatch_root(root: Path, schema: object) -> Path:
    root.mkdir()
    _write(root / "profile-result.json", {"schema_version": schema})
    return root


def test_dispatch_uses_only_the_reader_selected_by_the_exact_integer_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def legacy_reader(
        root: Path, *, expected_source_identity: object = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del root, expected_source_identity
        calls.append("schema2")
        return {"reader": "schema2"}, {"index": "schema2"}

    def policy_reader(
        root: Path, *, expected_source_identity: object = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del root, expected_source_identity
        calls.append("schema3")
        return {"reader": "schema3"}, {"index": "schema3"}

    monkeypatch.setattr(legacy_qualification, "verify_stateful_profile_evidence", legacy_reader)
    monkeypatch.setattr(policy_qualification, "verify_policy_stateful_profile_evidence", policy_reader)

    assert verify_v02_profile_evidence(_dispatch_root(tmp_path / "v2", 2))[0] == {"reader": "schema2"}
    assert calls == ["schema2"]
    calls.clear()
    assert verify_v02_profile_evidence(_dispatch_root(tmp_path / "v3", 3))[0] == {"reader": "schema3"}
    assert calls == ["schema3"]


@pytest.mark.parametrize("schema", [None, True, 2.0, "2", 0, 1, 4])
def test_dispatch_never_falls_back_for_missing_noninteger_or_unknown_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: object,
) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal called
        called = True
        raise AssertionError("an unsupported schema must not reach either reader")

    monkeypatch.setattr(legacy_qualification, "verify_stateful_profile_evidence", forbidden)
    monkeypatch.setattr(policy_qualification, "verify_policy_stateful_profile_evidence", forbidden)
    root = _dispatch_root(tmp_path / "profile", schema)

    with pytest.raises(IllmError) as raised:
        verify_v02_profile_evidence(root)

    assert raised.value.code is ErrorCode.UNSUPPORTED_SCHEMA
    assert called is False


def test_schema2_policy_shadow_fields_are_rejected_by_the_legacy_reader_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-2 envelope cannot acquire revision-2 policy shadow fields."""

    root = tmp_path / "mixed-schema"
    root.mkdir()
    profile: dict[str, Any] = {name: None for name in legacy_qualification.PROFILE_FIELDS}
    profile["schema_version"] = 2
    profile["contract_revision"] = 2
    index: dict[str, Any] = {name: None for name in legacy_qualification.INDEX_FIELDS}
    index["schema_version"] = 2
    _write(root / "profile-result.json", profile)
    _write(root / "evidence-index.json", index)
    _write_bytes(root / "COMPLETE", b"unit-only-not-a-profile\n")
    checksums = {
        name: sha256_bytes((root / name).read_bytes())
        for name in ("COMPLETE", "evidence-index.json", "profile-result.json")
    }
    _write(root / "SHA256SUMS.json", checksums)

    policy_called = False

    def forbidden_policy(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal policy_called
        policy_called = True
        raise AssertionError("schema 2 must never fall forward to the policy reader")

    monkeypatch.setattr(policy_qualification, "verify_policy_stateful_profile_evidence", forbidden_policy)

    with pytest.raises(IllmError) as raised:
        verify_v02_profile_evidence(root)

    assert raised.value.code is ErrorCode.UNSUPPORTED_SCHEMA
    assert policy_called is False


def _synthetic_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Build closed low-level transcript bytes, never a qualification profile."""

    root.mkdir()
    lexical_output = ROOT / "out" / "unit-policy-transcript-not-evidence"
    interpreter = ROOT / ".illm" / "envs" / "unit-policy-transcript" / "Scripts" / "python.exe"
    parameters = policy_phase_parameters(PROFILE_ID)
    child_commands = policy_phase_commands(PROFILE_ID, lexical_output)
    prefix = [
        str(interpreter),
        str(ROOT / "scripts" / "in_env.py"),
        "--manifest",
        str(lexical_output / "toolchain.json"),
        "--expected-manifest-digest-file",
        str(lexical_output / "toolchain.sha256"),
        "--",
    ]
    commands: list[dict[str, Any]] = []
    bootstrap = [
        str(interpreter),
        str(ROOT / "scripts" / "bootstrap.py"),
        "--profile",
        "unit-policy-transcript",
        "--runtime-profile",
        RUNTIME_ID,
        "--locked",
        "--offline",
        "--require-new-environment",
        "--write-manifest",
        str(lexical_output / "toolchain.json"),
        "--write-manifest-digest",
        str(lexical_output / "toolchain.sha256"),
    ]
    for phase in PHASES:
        stdout_relative = f"logs/{phase}.stdout.txt"
        stderr_relative = f"logs/{phase}.stderr.txt"
        _write_bytes(root / stdout_relative, b"")
        _write_bytes(root / stderr_relative, b"")
        argv = bootstrap if phase == "bootstrap" else [*prefix, *child_commands[phase]]
        commands.append({
            "phase": phase,
            "argv": argv,
            "parameters": parameters[phase],
            "returncode": 0,
            "stdout_file": stdout_relative,
            "stderr_file": stderr_relative,
            "stdout_sha256": sha256_bytes(b""),
            "stderr_sha256": sha256_bytes(b""),
        })
    identity = {
        "project_root_digest": "1" * 64,
        "bootstrap_digest": "2" * 64,
        "base_lock_digest": "3" * 64,
    }
    toolchain: dict[str, Any] = {
        "runtime_profile_id": RUNTIME_ID,
        "manifest_digest": "4" * 64,
        "profile": "unit-policy-transcript",
        "project_root": str(ROOT),
    }
    report: dict[str, Any] = {
        "schema_version": 3,
        "report_kind": GATE_KIND,
        "contract_revision": 2,
        "runtime_profile_id": RUNTIME_ID,
        "profile_id": PROFILE_ID,
        "status": "pass",
        "fresh_environment": True,
        "offline_bootstrap": True,
        "network_used": "not_observed",
        "network_blocking": "not_enforced",
        "network_policy": "offline_resolution_and_local_model_only",
        "toolchain_manifest_digest": toolchain["manifest_digest"],
        "source_identity": identity,
        "protocol_sha256": parameters["bootstrap"]["contract"]["protocol"]["sha256"],
        "approval_sha256": parameters["bootstrap"]["contract"]["approval"]["sha256"],
        "commands": commands,
        "qualification_status": "unqualified",
        "global_qualification_granted": False,
        "gpu_qualification_granted": False,
        **{axis: "incomplete" for axis in REQUIRED_RESULT_AXES},
    }
    report["report_digest"] = sha256_json(report)
    _write(root / "gate-report.json", report)
    _write(root / "execution.json", {
        "schema_version": 3,
        "contract_revision": 2,
        "profile_id": PROFILE_ID,
        "commands": commands,
    })
    return report, toolchain, identity


def _resign_gate(root: Path, report: dict[str, Any]) -> None:
    report.pop("report_digest", None)
    report["report_digest"] = sha256_json(report)
    _write(root / "gate-report.json", report)
    _write(root / "execution.json", {
        "schema_version": 3,
        "contract_revision": 2,
        "profile_id": PROFILE_ID,
        "commands": report["commands"],
    })


def _command(report: dict[str, Any], phase: str) -> dict[str, Any]:
    commands = cast(list[dict[str, Any]], report["commands"])
    return next(command for command in commands if command["phase"] == phase)


def test_low_level_gate_accepts_exact_fifteen_phase_transcript_without_granting_qualification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gate"
    report, toolchain, identity = _synthetic_gate(root)

    result = verify_policy_profile_gate(
        root, profile_id=PROFILE_ID, toolchain=toolchain, source_identity=identity,
    )

    assert tuple(command["phase"] for command in report["commands"]) == PHASES
    assert len(PHASES) == result["command_count"] == 15
    assert result["qualification_status"] == "unqualified"
    assert result["global_qualification_granted"] is False
    assert result["gpu_qualification_granted"] is False
    for phase in ("demo-b", "demo-c"):
        command = _command(report, phase)
        assert command["parameters"]["seed"] == 7
        assert command["argv"][command["argv"].index("--seed") + 1] == "7"


def _mutate_phase_injection(root: Path, report: dict[str, Any]) -> None:
    injected = dict(cast(list[dict[str, Any]], report["commands"])[-1])
    injected["phase"] = "injected-pass"
    cast(list[dict[str, Any]], report["commands"]).append(injected)
    _resign_gate(root, report)


def _mutate_phase_order(root: Path, report: dict[str, Any]) -> None:
    commands = cast(list[dict[str, Any]], report["commands"])
    commands[6], commands[7] = commands[7], commands[6]
    _resign_gate(root, report)


def _mutate_parameter(root: Path, report: dict[str, Any]) -> None:
    _command(report, "demo-b")["parameters"]["seed"] = 8
    _resign_gate(root, report)


def _mutate_seed_argv(root: Path, report: dict[str, Any]) -> None:
    argv = cast(list[str], _command(report, "demo-b")["argv"])
    argv[argv.index("--seed") + 1] = "8"
    _resign_gate(root, report)


def _mutate_bool_returncode(root: Path, report: dict[str, Any]) -> None:
    _command(report, "policy-cpu")["returncode"] = True
    _resign_gate(root, report)


def _mutate_closed_log_path(root: Path, report: dict[str, Any]) -> None:
    _command(report, "policy-inputs")["stdout_file"] = "logs/../policy-inputs.stdout.txt"
    _resign_gate(root, report)


def _mutate_unknown_command_field(root: Path, report: dict[str, Any]) -> None:
    _command(report, "policy-verify")["injected_claim"] = "pass"
    _resign_gate(root, report)


GateMutation = Callable[[Path, dict[str, Any]], None]


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_phase_injection,
        _mutate_phase_order,
        _mutate_parameter,
        _mutate_seed_argv,
        _mutate_bool_returncode,
        _mutate_closed_log_path,
        _mutate_unknown_command_field,
    ],
    ids=lambda mutation: cast(GateMutation, mutation).__name__.removeprefix("_mutate_"),
)
def test_gate_rejects_fully_resigned_phase_parameter_seed_bool_path_and_injection_mutations(
    tmp_path: Path, mutation: GateMutation,
) -> None:
    root = tmp_path / "gate"
    report, toolchain, identity = _synthetic_gate(root)
    mutation(root, report)

    with pytest.raises(NumericalEvidenceError):
        verify_policy_profile_gate(
            root, profile_id=PROFILE_ID, toolchain=toolchain, source_identity=identity,
        )


def test_gate_rejects_changed_log_bytes_when_the_recorded_hash_is_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "gate"
    _, toolchain, identity = _synthetic_gate(root)
    _write_bytes(root / "logs/policy-cpu.stdout.txt", b"injected transcript\n")

    with pytest.raises(NumericalEvidenceError, match="log path or bytes"):
        verify_policy_profile_gate(
            root, profile_id=PROFILE_ID, toolchain=toolchain, source_identity=identity,
        )


def _load_gate_runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    specification = importlib.util.spec_from_file_location(
        "run_v02_policy_gate_unit", ROOT / "scripts/run_v02_policy_gate.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_gpu_runner_fails_before_subprocess_output_or_environment_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_gate_runner(monkeypatch)
    fake_project = tmp_path / "unit-project-never-created"
    output = fake_project / "out" / "gpu-never-created"
    profile_name = "gpu-unit-never-created"
    environment = fake_project / ".illm" / "envs" / profile_name
    subprocess_called = False

    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("GPU rejection must precede every subprocess")

    monkeypatch.setattr(runner, "PROJECT_ROOT", fake_project)
    monkeypatch.setattr(
        runner,
        "_contract",
        lambda: (POLICY_PROFILE_RUNTIME_IDS, policy_phase_parameters, policy_phase_commands),
    )
    subprocess_module = cast(Any, runner).subprocess
    monkeypatch.setattr(subprocess_module, "run", forbidden_subprocess)
    run_gate = cast(Callable[[list[str]], int], getattr(runner, "run_gate"))

    with pytest.raises(ValueError, match=POLICY_GPU_DEFERRED_REASON):
        run_gate([
            "--profile",
            profile_name,
            "--profile-id",
            POLICY_GPU_PROFILE_ID,
            "--output",
            str(output),
        ])

    assert subprocess_called is False
    assert not output.exists()
    assert not environment.exists()
    assert not fake_project.exists()


def test_junit_requires_new_policy_regressions_to_have_executed_not_skipped(tmp_path: Path) -> None:
    # This exercises only the named-regression projection, not a public profile.
    document = ET.Element("testsuite")
    for name in sorted(policy_qualification.REQUIRED_EXECUTED_REGRESSIONS):
        ET.SubElement(document, "testcase", name=name, classname="unit_only")
    path = tmp_path / "junit.xml"
    ET.ElementTree(document).write(path, encoding="utf-8")
    assert policy_qualification._required_regressions(path)["status"] == "pass"
    case = next(document.iter("testcase"))
    ET.SubElement(case, "skipped")
    ET.ElementTree(document).write(path, encoding="utf-8")
    with pytest.raises(NumericalEvidenceError, match="actually executed"):
        policy_qualification._required_regressions(path)


def test_junit_entity_declarations_cannot_supply_regression_names(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_bytes(b'<!DOCTYPE testsuite [<!ENTITY name "x">]><testsuite/>')
    with pytest.raises(NumericalEvidenceError, match="XML declarations"):
        policy_qualification._required_regressions(path)


def test_component_stdout_projection_preserves_bootstrap_pip_prefix_and_manifest_receipt(tmp_path: Path) -> None:
    # Projection-only unit data; none of these simplified components can form a
    # public passing profile. Bootstrap actually emits pip text before JSON.
    toolchain = {"project_root": "C:/unit-project", "manifest_digest": "a" * 64}
    gate = {"gate_relative": "out/unit-gate"}
    numerical = {"status": "incomplete", "cpu_results": {"hard_gate_result": "pass"}, "report_kind": "unit-only"}
    raw_manifest = {"manifest_digest": "b" * 64}
    exercise = {"status": "pass", "report_digest": "c" * 64, "evidence_stage": "source_bound_cpu_profile"}
    portable = {"report_sha256": "d" * 64, "capture_manifest_digest": "e" * 64}
    learning = {"status": "pass", "evidence_kind": "derived"}
    _write_bytes(tmp_path / "gate/logs/bootstrap.stdout.txt", b"Looking in links: vendor/wheels\n"
                 + canonical_json_bytes(toolchain) + b"\n")
    for phase in ("policy-inputs", "policy-cpu", "policy-verify"):
        _write_bytes(tmp_path / "gate/logs" / (phase + ".stdout.txt"), b"")
    raw_reader = {"schema_version": 1, "report_digest": "f" * 64}
    _write(tmp_path / "raw-numerical/logs/reader-copy.stdout.log", raw_reader)
    outputs = {
        "raw-numerical-build": {**numerical, "manifest_digest": raw_manifest["manifest_digest"],
                                "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified"},
        "raw-numerical-verify": raw_reader,
        "policy-exercise": {**exercise, "qualification_status": "unqualified", "output": "C:/unit-project/out/unit-gate/exercise"},
        "learning-build": learning, "learning-verify": learning,
    }
    portable_stdout = {
        "report_kind": "illm-policy-capture-portable-development-v1", "status": "pass",
        "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified",
        "report_sha256": portable["report_sha256"], "manifest_digest": portable["capture_manifest_digest"],
        "output": "C:/unit-project/out/unit-gate/portable",
    }
    outputs["capture-portable-build"] = portable_stdout
    outputs["capture-portable-verify"] = {**portable_stdout, "source_capture_argument_used": False}
    for phase, value in outputs.items():
        _write(tmp_path / "gate/logs" / (phase + ".stdout.txt"), value)
    parameters: dict[str, Any] = {
        "toolchain": toolchain, "gate": gate, "numerical": numerical, "raw_manifest": raw_manifest,
        "exercise": exercise, "portable": portable, "learning": learning,
    }
    policy_qualification._phase_outputs(tmp_path, **parameters)
    _write(tmp_path / "gate/logs/bootstrap.stdout.txt", {**toolchain, "manifest_digest": "0" * 64})
    with pytest.raises(NumericalEvidenceError, match="bootstrap stdout"):
        policy_qualification._phase_outputs(tmp_path, **parameters)


def test_profile_dispatch_and_readers_import_without_site_or_tensor_runtime() -> None:
    program = r'''
import builtins,json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
forbidden=('torch','transformers','tokenizers','safetensors')
original=builtins.__import__
attempted=[]
network=[]
def guarded(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.',1)[0] in forbidden:
        attempted.append(name)
        raise RuntimeError('forbidden tensor import')
    return original(name,globals,locals,fromlist,level)
def audit(event,args):
    if event.startswith('socket.'):
        network.append(event)
        raise RuntimeError('forbidden network')
builtins.__import__=guarded
sys.addaudithook(audit)
import illm.v02_profile_dispatch,illm.stateful_policy_qualification
import illm.stateful_policy_exercise,illm.stateful_policy_rollback
loaded=[name for name in sys.modules if name.split('.',1)[0] in forbidden]
assert not attempted and not loaded and not network
print(json.dumps({'attempted':attempted,'loaded':loaded,'network':network},sort_keys=True))
'''
    process = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(ROOT / "src")],
                             cwd=ROOT, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
    assert process.stderr == b""
    assert process.stdout.strip() == b'{"attempted": [], "loaded": [], "network": []}'

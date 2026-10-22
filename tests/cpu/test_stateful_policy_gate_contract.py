from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from illm.stateful_policy_gate_contract import (
    APPROVAL_PATH,
    APPROVAL_SHA256,
    CANDIDATE_POLICY_DIGESTS,
    CAPTURE_EVENT_CATALOG,
    INPUT_BYTES_SHA256,
    INPUT_MANIFEST_DIGEST,
    POLICY_CPU_PROFILE_IDS,
    POLICY_GPU_DEFERRED_REASON,
    POLICY_GPU_PROFILE_ID,
    POLICY_PROFILE_RUNTIME_IDS,
    POLICY_QUALIFICATION_CONTRACT_REVISION,
    POLICY_QUALIFICATION_SCHEMA_VERSION,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    REQUIRED_POLICY_CAPTURE_IDS,
    REQUIRED_RESULT_AXES,
    policy_phase_commands,
    policy_phase_parameters,
)


PHASES = (
    "bootstrap",
    "policy-inputs",
    "policy-cpu",
    "policy-verify",
    "raw-numerical-build",
    "raw-numerical-verify",
    "demo-b",
    "demo-c",
    "replay-c",
    "inspect-c",
    "policy-exercise",
    "capture-portable-build",
    "capture-portable-verify",
    "learning-build",
    "learning-verify",
)


@pytest.mark.parametrize("profile_id", sorted(POLICY_CPU_PROFILE_IDS))
def test_cpu_profiles_have_one_ordered_revision2_contract(profile_id: str, tmp_path: Path) -> None:
    parameters = policy_phase_parameters(profile_id)
    commands = policy_phase_commands(profile_id, tmp_path / "gate")

    assert tuple(parameters) == PHASES
    assert tuple(commands) == PHASES[1:]
    assert "quality" not in commands
    assert "pytest" not in commands
    assert parameters["policy-cpu"]["nested_execution"] == ["quality", "pytest", "example"]
    assert parameters["policy-cpu"]["test_scope"] == "all"
    assert parameters["policy-cpu"]["cuda_execution_authorized"] is False
    assert json.loads(json.dumps(parameters)) == parameters

    bootstrap = parameters["bootstrap"]
    assert bootstrap["runtime_profile_id"] == POLICY_PROFILE_RUNTIME_IDS[profile_id]
    assert bootstrap["fresh_environment"] is True
    assert bootstrap["locked"] is True
    assert bootstrap["offline"] is True
    contract = bootstrap["contract"]
    assert contract["qualification_schema_version"] == POLICY_QUALIFICATION_SCHEMA_VERSION == 3
    assert contract["contract_revision"] == POLICY_QUALIFICATION_CONTRACT_REVISION == 2
    assert contract["protocol"] == {
        "protocol_id": "v02-numerical-policy-protocol-draft-1",
        "path": PROTOCOL_PATH,
        "sha256": PROTOCOL_SHA256,
    }
    assert contract["approval"] == {"path": APPROVAL_PATH, "sha256": APPROVAL_SHA256}
    assert contract["input_manifest"]["raw_sha256"] == INPUT_BYTES_SHA256
    assert contract["input_manifest"]["manifest_digest"] == INPUT_MANIFEST_DIGEST
    assert contract["result_axes"] == list(REQUIRED_RESULT_AXES)
    assert contract["result_axis_values"] == ["pass", "fail", "incomplete", "not_run"]
    assert contract["capture"]["manifest_schema_version"] == 3
    assert contract["capture"]["event_catalog"] == CAPTURE_EVENT_CATALOG
    assert contract["capture"]["capabilities"] == ["R0"]
    assert contract["memory_plan_schema_version"] == 2
    assert contract["stream_api_contract_revision"] == 2
    assert contract["generation_result_schema_version"] == 3
    assert contract["rollback_component"] == {
        "report_kind": "illm-stateful-policy-rollback-v2",
        "schema_version": 2,
        "required_by_release": True,
        "evaluated_by_profile_gate": False,
    }


def test_policy_registry_identity_is_closed_and_has_three_candidates_and_legacy_adapters() -> None:
    contract = policy_phase_parameters("state-machine-cpu-v1")["bootstrap"]["contract"]
    registry = contract["policy_registry"]
    assert set(registry) == {"schema_version", "candidate_instances", "legacy_adapters"}
    assert [item["storage_dtype"] for item in registry["candidate_instances"]] == ["bf16", "fp16", "fp32"]
    assert [item["digest"] for item in registry["candidate_instances"]] == [
        digest for _, digest in CANDIDATE_POLICY_DIGESTS
    ]
    assert [item["selector"] for item in registry["candidate_instances"]] == [
        "candidate-bf16", "candidate-fp16", "candidate-fp32",
    ]
    assert [item["selector"] for item in registry["legacy_adapters"]] == [
        "legacy-bf16", "legacy-fp16", "legacy-fp32",
    ]


def test_policy_driver_stages_share_exact_attempt_and_frozen_authority_paths(tmp_path: Path) -> None:
    output = tmp_path / "gate"
    commands = policy_phase_commands("state-machine-cpu-v1", output)
    expected_outputs = {
        "policy-inputs": ("inputs", output / "policy-inputs"),
        "policy-cpu": ("cpu", output / "policy-cpu"),
        "policy-verify": ("verify", output / "policy-verified"),
    }
    for phase, (stage, destination) in expected_outputs.items():
        assert commands[phase] == [
            "python",
            "scripts/run_numerical_policy.py",
            "--stage",
            stage,
            "--protocol",
            PROTOCOL_PATH,
            "--approval",
            APPROVAL_PATH,
            "--attempt",
            str(output / "policy-attempt.json"),
            "--output",
            str(destination),
        ]


def test_raw_numerical_contract_is_complete_but_not_promoted(tmp_path: Path) -> None:
    output = tmp_path / "gate"
    parameters = policy_phase_parameters("state-machine-cpu-v1")
    commands = policy_phase_commands("state-machine-cpu-v1", output)
    raw = parameters["raw-numerical-build"]

    assert raw["case_count"] == 70
    assert raw["high_precision_case_count"] == 8
    assert raw["mutation_count"] == 9
    assert raw["reproduction_case_count"] == 20
    assert raw["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert raw["qualification_status"] == "unqualified"
    assert raw["global_result_axes"] == {axis: "incomplete" for axis in REQUIRED_RESULT_AXES}
    assert commands["raw-numerical-build"] == [
        "python", "scripts/validate_policy_cpu_raw.py", "--output", str(output / "raw-numerical"),
    ]
    assert commands["raw-numerical-verify"] == [
        "python", "scripts/validate_policy_cpu_raw.py", "--verify-existing", str(output / "raw-numerical"),
    ]


def test_demos_explicitly_select_one_candidate_for_b_and_c(tmp_path: Path) -> None:
    output = tmp_path / "gate"
    parameters = policy_phase_parameters("state-machine-cpu-v1")
    commands = policy_phase_commands("state-machine-cpu-v1", output)
    b, c = commands["demo-b"], commands["demo-c"]

    for argv in (b, c):
        assert argv[argv.index("--numerical-policy") + 1] == "candidate-fp32"
        assert argv[argv.index("--telemetry") + 1] == "request"
        assert argv[argv.index("--seed") + 1] == "7"
        assert "--capture" in argv
    assert b[b.index("--kv-mode") + 1] == parameters["demo-b"]["kv_mode"] == "none"
    assert c[c.index("--kv-mode") + 1] == parameters["demo-c"]["kv_mode"] == "contiguous"
    assert b[b.index("--capture") + 1] == str(output / "demo-b")
    assert c[c.index("--capture") + 1] == str(output / "demo-c")
    assert parameters["demo-b"]["policy_digest"] == parameters["demo-c"]["policy_digest"]
    assert parameters["demo-b"]["sampling"] == parameters["demo-c"]["sampling"] == "greedy"
    assert parameters["demo-b"]["seed"] == parameters["demo-c"]["seed"] == 7
    assert parameters["demo-b"]["capture_schema_version"] == 3
    assert parameters["demo-c"]["capture_catalog"] == CAPTURE_EVENT_CATALOG
    assert commands["replay-c"] == ["python", "-m", "illm", "replay", str(output / "demo-c"), "--mode", "trace"]
    assert commands["inspect-c"][-2:] == ["--request", "req-policy-c-0001"]


def test_lifecycle_portable_and_learning_paths_are_exact(tmp_path: Path) -> None:
    output = tmp_path / "gate"
    parameters = policy_phase_parameters("clean-room-release-v1")
    commands = policy_phase_commands("clean-room-release-v1", output)
    exercise = parameters["policy-exercise"]

    assert exercise["cycle_count"] == 100
    assert exercise["fault_count"] == 12
    assert exercise["required_capture_ids"] == list(REQUIRED_POLICY_CAPTURE_IDS)
    assert exercise["rollback_boundary_evaluated"] is False
    assert commands["policy-exercise"] == [
        "python", "scripts/exercise_policy_streaming.py", "--output", str(output / "exercise"),
    ]
    success = str(output / "exercise/captures/success")
    cancel = str(output / "exercise/captures/cancel-after-launch")
    assert commands["capture-portable-build"] == [
        "python", "scripts/check_policy_capture_portable.py", success, "--output", str(output / "portable"),
    ]
    assert commands["capture-portable-verify"][-1] == "--verify-existing"
    assert commands["learning-build"] == [
        "python", "scripts/render_streaming_learning.py", cancel, "--output", str(output / "learning"),
    ]
    assert commands["learning-verify"][-1] == "--verify-existing"


def test_factories_return_fresh_values() -> None:
    first = policy_phase_parameters("state-machine-cpu-v1")
    first["bootstrap"]["contract"]["result_axes"].append("invented")
    first["policy-exercise"]["required_capture_ids"].clear()
    second = policy_phase_parameters("state-machine-cpu-v1")

    assert second["bootstrap"]["contract"]["result_axes"] == list(REQUIRED_RESULT_AXES)
    assert second["policy-exercise"]["required_capture_ids"] == list(REQUIRED_POLICY_CAPTURE_IDS)


@pytest.mark.parametrize("factory", [policy_phase_parameters, policy_phase_commands])
def test_gpu_profile_fails_before_returning_any_runnable_argv(factory: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=POLICY_GPU_DEFERRED_REASON):
        if factory is policy_phase_parameters:
            policy_phase_parameters(POLICY_GPU_PROFILE_ID)
        else:
            policy_phase_commands(POLICY_GPU_PROFILE_ID, tmp_path / "gate")


def test_unknown_profiles_and_non_path_output_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        policy_phase_parameters("invented-profile")
    with pytest.raises(ValueError, match="unknown"):
        policy_phase_commands("invented-profile", tmp_path / "gate")
    with pytest.raises(TypeError, match="pathlib.Path"):
        policy_phase_commands("state-machine-cpu-v1", "out")  # type: ignore[arg-type]


def test_contract_module_imports_only_the_standard_library() -> None:
    source = Path("src/illm/stateful_policy_gate_contract.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".", 1)[0])
    assert imported <= {"pathlib", "typing", "__future__"}

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from illm.numerical_policy import candidate_policy
from illm.stateful_policy_cpu_component import (
    COMPONENT_KIND,
    PolicyCPUComponentError,
    verify_policy_cpu_development_component,
)
from illm.stateful_policy_gate_contract import (
    APPROVAL_SHA256,
    CANDIDATE_POLICY_DIGESTS,
    INPUT_BYTES_SHA256,
    INPUT_MANIFEST_DIGEST,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)


GATE_RELATIVE = "out/policy-component-unit"
APPROVED_COMMIT = "04144ab9ddb8d86c482ad6db8a618bded1f1e7aa"
PENDING = [
    "full raw-tensor numerical qualification and final capture3 lifecycle/portable profile integration",
    "formal portable release qualification",
    "all GPU pilot, correctness, benchmark, lifecycle, and rollback evidence",
]
SUITE_DIGESTS = [
    "57163407317f5e9f9f0cc26931c364718b299a79e3ec77cb1c7da4bac52a6f7a",
    "659337351c7fd93ce2ca101a76908bf537db586ded85e89ac41e45072e97f9cb",
]
STAGE_FILES = {
    "inputs": [
        "candidate-policies/bf16.json",
        "candidate-policies/fp16.json",
        "candidate-policies/fp32.json",
        "inputs-report.json",
        "policy-inputs.json",
    ],
    "cpu": [
        "cpu-report.json",
        "execution.json",
        "junit.xml",
        "logs/example.stderr.txt",
        "logs/example.stdout.txt",
        "logs/pytest.stderr.txt",
        "logs/pytest.stdout.txt",
        "logs/quality.stderr.txt",
        "logs/quality.stdout.txt",
        "quality.json",
    ],
    "verify": ["verification-report.json"],
}
STAGE_DIRECTORIES = {
    "inputs": "policy-inputs",
    "cpu": "policy-cpu",
    "verify": "policy-verified",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _toolchain() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "profile": "policy-component-unit",
        "runtime_profile_id": "py312-win-amd64-cpu",
        "project_root": "C:/source/illm",
        "environment_root": "C:/source/illm/.illm/envs/policy-component-unit",
        "python_executable": "C:/source/illm/.illm/envs/policy-component-unit/Scripts/python.exe",
        "artifact_cache": "C:/source/illm/.illm/artifacts",
        "finalized": True,
        "environment_id": str(uuid.UUID("11111111-1111-4111-8111-111111111111")),
        "python_version": "3.12.14",
        "os": "Windows",
        "arch": "AMD64",
        "lock_digest": "1" * 64,
        "runtime_lock_digest": "2" * 64,
        "bootstrap_digest": "3" * 64,
        "project_root_digest": "4" * 64,
        "entrypoints": {"python": "Scripts/python.exe"},
        "installed_packages": [{"name": "illm", "version": "0.2.0.dev0"}],
        "runtime_artifacts": [],
        "runtime_probe": {"cuda_available": False},
    }
    value["manifest_digest"] = _sha(_canonical(value))
    return value


def _portable(toolchain: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_id": toolchain["environment_id"],
        "lock_digest": toolchain["lock_digest"],
        "manifest_digest": toolchain["manifest_digest"],
        "project_root_digest": toolchain["project_root_digest"],
        "python_version": toolchain["python_version"],
        "runtime_profile_id": toolchain["runtime_profile_id"],
    }


def _protocol() -> dict[str, str]:
    return {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256}


def _policy_digests() -> dict[str, str]:
    return dict(CANDIDATE_POLICY_DIGESTS)


def _commands() -> list[dict[str, Any]]:
    cpu = f"{GATE_RELATIVE}/policy-cpu"
    argv = {
        "quality": ["{finalized_python}", "scripts/check_quality.py", "--output", f"{cpu}/quality.json"],
        "pytest": [
            "{finalized_python}",
            "-m",
            "pytest",
            "tests",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            f"{cpu}-pytest-scratch",
            "--junitxml",
            f"{cpu}/junit.xml",
            "--tb=short",
        ],
        "example": ["{finalized_python}", "docs/examples/v0_2_policy_cpu.py"],
    }
    records: list[dict[str, Any]] = []
    for phase in ("quality", "pytest", "example"):
        records.append(
            {
                "phase": phase,
                "argv": argv[phase],
                "returncode": 0,
                "stdout_file": f"logs/{phase}.stdout.txt",
                "stdout_sha256": "",
                "stderr_file": f"logs/{phase}.stderr.txt",
                "stderr_sha256": "",
            },
        )
    return records


def _result(kv_mode: str, portable: dict[str, Any]) -> dict[str, Any]:
    policy = candidate_policy("fp32")
    contiguous = kv_mode == "contiguous"
    environment = {
        "backend": "cpu",
        "python_version": portable["python_version"],
        "platform": "Windows-test",
        "machine": "AMD64",
        "torch_version": "2.8.0+cpu",
        "cuda_arithmetic": "not_applicable",
        "cuda_execution_authorized": False,
    }
    binding = {
        "schema_version": 1,
        "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": portable["project_root_digest"],
        "environment": environment,
        "environment_digest": _sha(_canonical(environment)),
        "toolchain": {
            "binding": "source-bound-manifest",
            "environment_id": portable["environment_id"],
            "lock_digest": portable["lock_digest"],
            "manifest_digest": portable["manifest_digest"],
        },
        "protocol_sha256": PROTOCOL_SHA256,
    }
    return {
        "schema_version": 3,
        "request_id": "req-example-c" if contiguous else "req-example-b",
        "input_token_ids": [1, 107, 104, 111, 111, 114],
        "output_token_ids": [114, 114, 114, 114],
        "text": "oooo",
        "outcome": "FINISHED",
        "finish_reason": "MAX_NEW_TOKENS",
        "error_code": None,
        "actual_seed": 7,
        "model_spec_id": "5" * 64,
        "sampling_spec_digest": "6" * 64,
        "kv_mode": kv_mode,
        "next_output_position": 4,
        "rng_draws_consumed": 0,
        "committed_draw_addresses": [],
        "query_tokens_per_step": [6, 1, 1, 1] if contiguous else [6, 7, 8, 9],
        "kv_valid_tokens_per_step": [6, 7, 8, 9] if contiguous else [0, 0, 0, 0],
        "resource_status": {
            "active_requests": 0,
            "active_tickets": 0,
            "input_leases": 0,
            "kv_reserved_bytes": 0,
            "kv_staged_tokens": 0,
            "kv_valid_tokens": 0,
            "kv_write_leases": 0,
            "output_leases": 0,
            "quarantined_requests": 0,
            "reservations": 0,
            "workspace_leases": 0,
        },
        "timings": {
            "accepted_ns": 1,
            "clock_domain": "host.perf_counter_ns",
            "critical_capture_event_drop_count": 0,
            "evidence_kind": "measured",
            "per_forward_wall_time_ns": [1, 1, 1, 1],
            "server_e2e_wall_time_ns": 4,
            "telemetry": "metrics",
            "terminal_ready_ns": 5,
            "token_ready_ns": [2, 3, 4, 5],
            "tokenization_wall_time_ns": 1,
        },
        "capture_manifest_digest": None,
        "capture_error": None,
        "event_count": 35 if contiguous else 25,
        "numerical_policy": policy.to_dict(),
        "numerical_policy_digest": policy.digest,
        "execution_binding": binding,
        "execution_binding_digest": _sha(_canonical(binding)),
        "memory_plan_digest": ("8" if contiguous else "9") * 64,
    }


def _seal(stage_root: Path, stage: str) -> dict[str, Any]:
    entries = [
        {
            "path": relative,
            "sha256": _sha((stage_root / Path(relative)).read_bytes()),
            "size_bytes": (stage_root / Path(relative)).stat().st_size,
        }
        for relative in STAGE_FILES[stage]
    ]
    value = {
        "schema_version": 1,
        "kind": "illm-policy-stage-checksums-v1",
        "stage": stage,
        "files": entries,
        "files_digest": _sha(_canonical(entries)),
    }
    _write_json(stage_root / "checksums.json", value)
    _write_json(
        stage_root / "COMPLETE",
        {"kind": "illm-policy-stage-complete-v1", "stage": stage, "files_digest": value["files_digest"]},
    )
    return value


def _build_component(
    tmp_path: Path,
    *,
    mutate_before_seal: Callable[[Path], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "component"
    inputs = root / "policy-inputs"
    cpu = root / "policy-cpu"
    verified = root / "policy-verified"
    (inputs / "candidate-policies").mkdir(parents=True)
    (cpu / "logs").mkdir(parents=True)
    verified.mkdir(parents=True)

    toolchain = _toolchain()
    portable = _portable(toolchain)
    _write_json(root / "toolchain.json", toolchain)
    input_bytes = Path("docs/implementation/v0.2-policy-inputs-draft.json").read_bytes()
    assert _sha(input_bytes) == INPUT_BYTES_SHA256
    (inputs / "policy-inputs.json").write_bytes(input_bytes)
    for dtype, digest in CANDIDATE_POLICY_DIGESTS:
        raw = _canonical(candidate_policy(dtype).to_dict())
        assert _sha(raw) == digest
        (inputs / "candidate-policies" / f"{dtype}.json").write_bytes(raw)
    _write_json(
        inputs / "inputs-report.json",
        {
            "schema_version": 1,
            "kind": "illm-policy-inputs-development-snapshot-v1",
            "status": "pass",
            "evidence_mode": "DEVELOPMENT_ONLY",
            "qualification_tier": "unqualified",
            "qualification_result": "not_evaluated",
            "approved_commit": APPROVED_COMMIT,
            "protocol": _protocol(),
            "input": {
                "raw_bytes_sha256": INPUT_BYTES_SHA256,
                "manifest_digest": INPUT_MANIFEST_DIGEST,
                "suite_digests": SUITE_DIGESTS,
                "case_counts": [20, 50],
                "snapshot": "policy-inputs.json",
            },
            "candidate_policy_digests": _policy_digests(),
            "toolchain": portable,
            "limitations": PENDING,
        },
    )

    quality = {
        "schema_version": 0,
        "checked_python_files": 17,
        "issues": [],
        "type_check": {
            "tool": "mypy",
            "version": "mypy 1.18.2",
            "returncode": 0,
            "status": "pass",
            "output": ["Success: no issues found in 17 source files"],
        },
        "status": "pass",
    }
    _write_json(cpu / "quality.json", quality)
    (cpu / "logs/quality.stdout.txt").write_bytes((cpu / "quality.json").read_bytes())
    (cpu / "logs/quality.stderr.txt").write_bytes(b"")
    (cpu / "logs/pytest.stdout.txt").write_bytes(b"1 passed in 0.01s\n")
    (cpu / "logs/pytest.stderr.txt").write_bytes(b"")
    junit = (
        b'<?xml version="1.0" encoding="utf-8"?>'
        b'<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1" time="0.01">'
        b'<testcase classname="tests.cpu.test_unit" name="test_one" time="0.01" />'
        b"</testsuite></testsuites>"
    )
    (cpu / "junit.xml").write_bytes(junit)
    (cpu / "logs/example.stdout.txt").write_bytes(
        _canonical(_result("none", portable))
        + b"\n"
        + _canonical(_result("contiguous", portable))
        + b"\n",
    )
    (cpu / "logs/example.stderr.txt").write_bytes(b"")

    commands = _commands()
    for record in commands:
        phase = cast(str, record["phase"])
        for stream in ("stdout", "stderr"):
            raw = (cpu / f"logs/{phase}.{stream}.txt").read_bytes()
            record[f"{stream}_sha256"] = _sha(raw)
    _write_json(
        cpu / "execution.json",
        {"schema_version": 1, "kind": "illm-policy-cpu-development-execution-v1", "commands": commands},
    )
    _write_json(
        cpu / "cpu-report.json",
        {
            "schema_version": 1,
            "kind": "illm-policy-cpu-development-gate-v1",
            "status": "pass",
            "evidence_mode": "DEVELOPMENT_ONLY",
            "qualification_tier": "unqualified",
            "qualification_result": "not_evaluated",
            "performance_outcome": "not_evaluated",
            "gpu_result": "not_run",
            "raw_tensor_evidence": "not_captured",
            "approved_commit": APPROVED_COMMIT,
            "protocol": _protocol(),
            "input_manifest_digest": INPUT_MANIFEST_DIGEST,
            "candidate_policy_digests": _policy_digests(),
            "toolchain": portable,
            "commands": commands,
            "required_artifacts": {
                "quality": {"path": "quality.json", "sha256": _sha((cpu / "quality.json").read_bytes())},
                "junit": {"path": "junit.xml", "sha256": _sha((cpu / "junit.xml").read_bytes())},
            },
            "pytest_scratch": {
                "path": f"{GATE_RELATIVE}/policy-cpu-pytest-scratch",
                "is_evidence": False,
                "retention": "retained",
            },
            "limitations": PENDING,
        },
    )

    if mutate_before_seal is not None:
        mutate_before_seal(root)

    stage_data = {
        "inputs": _seal(inputs, "inputs"),
        "cpu": _seal(cpu, "cpu"),
    }
    verified_stages = [
        {
            "stage": stage,
            "output_root": f"{GATE_RELATIVE}/{STAGE_DIRECTORIES[stage]}",
            "files_digest": stage_data[stage]["files_digest"],
            "file_count": len(STAGE_FILES[stage]),
        }
        for stage in ("inputs", "cpu")
    ]
    _write_json(
        verified / "verification-report.json",
        {
            "schema_version": 1,
            "kind": "illm-policy-cpu-development-verification-v1",
            "status": "pass",
            "evidence_mode": "DEVELOPMENT_ONLY",
            "qualification_tier": "unqualified",
            "qualification_result": "not_evaluated",
            "performance_outcome": "not_evaluated",
            "verified_stages": verified_stages,
            "approved_commit": APPROVED_COMMIT,
            "protocol": _protocol(),
            "input_manifest_digest": INPUT_MANIFEST_DIGEST,
            "candidate_policy_digests": _policy_digests(),
            "toolchain": portable,
            "limitations": PENDING,
        },
    )
    stage_data["verify"] = _seal(verified, "verify")
    journal: dict[str, Any] = {
        "schema_version": 1,
        "journal_kind": "illm-policy-development-attempt-v1",
        "status": "verified_development_only_unqualified",
        "evidence_mode": "DEVELOPMENT_ONLY",
        "qualification_tier": "unqualified",
        "source_identity": {
            "approved_commit": APPROVED_COMMIT,
            "project_root_digest": portable["project_root_digest"],
        },
        "protocol": _protocol(),
        "approval": {
            "approval_kind": "illm-human-policy-approval-v1",
            "sha256": APPROVAL_SHA256,
            "interpretation": "transcript_of_received_human_answer_not_self_authorization",
        },
        "toolchain": portable,
        "input_identity": {
            "raw_bytes_sha256": INPUT_BYTES_SHA256,
            "manifest_digest": INPUT_MANIFEST_DIGEST,
        },
        "candidate_policy_digests": _policy_digests(),
        "stages": [
            {
                "sequence": index + 1,
                "stage": stage,
                "depends_on": list(("inputs", "cpu", "verify")[:index]),
                "result": "pass",
                "evidence_mode": "DEVELOPMENT_ONLY",
                "qualification_tier": "unqualified",
                "output_root": f"{GATE_RELATIVE}/{STAGE_DIRECTORIES[stage]}",
                "files_digest": stage_data[stage]["files_digest"],
                "checksums_sha256": _sha(
                    (root / STAGE_DIRECTORIES[stage] / "checksums.json").read_bytes(),
                ),
            }
            for index, stage in enumerate(("inputs", "cpu", "verify"))
        ],
        "pending": PENDING,
    }
    journal["journal_digest"] = _sha(_canonical(journal))
    _write_json(root / "policy-attempt.json", journal)
    return root, toolchain


def test_valid_component_recomputes_one_full_cpu_suite_without_qualification(tmp_path: Path) -> None:
    root, toolchain = _build_component(tmp_path)
    first = verify_policy_cpu_development_component(
        root,
        expected_toolchain=toolchain,
        gate_relative=GATE_RELATIVE,
    )
    second = verify_policy_cpu_development_component(
        root,
        expected_toolchain=json.loads(json.dumps(toolchain)),
        gate_relative=GATE_RELATIVE,
    )

    assert first == second
    assert set(first) == {
        "schema_version",
        "component_kind",
        "status",
        "evidence_stage",
        "qualification_status",
        "qualification_result",
        "gpu_result",
        "raw_tensor_evidence",
        "inputs_files_digest",
        "cpu_files_digest",
        "verified_files_digest",
        "journal_digest",
        "cpu_report_sha256",
        "quality_sha256",
        "junit_sha256",
        "toolchain_manifest_digest",
        "implementation_source_digest",
        "environment_id",
        "cpu_full_suite",
        "toolchain_schema_version",
    }
    assert first["component_kind"] == COMPONENT_KIND
    assert first["status"] == "pass"
    assert first["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert first["qualification_status"] == "unqualified"
    assert first["qualification_result"] == "not_evaluated"
    assert first["gpu_result"] == "not_run"
    assert first["raw_tensor_evidence"] == "not_captured"
    assert first["cpu_full_suite"]["nested_execution"] == ["quality", "pytest", "example"]
    assert first["cpu_full_suite"]["junit"] == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }


def test_fully_resigned_cpu_claim_cannot_promote_raw_tensor_evidence(tmp_path: Path) -> None:
    def mutate(root: Path) -> None:
        report = _read_json(root / "policy-cpu/cpu-report.json")
        report["raw_tensor_evidence"] = "captured"
        _write_json(root / "policy-cpu/cpu-report.json", report)

    root, toolchain = _build_component(tmp_path, mutate_before_seal=mutate)
    with pytest.raises(PolicyCPUComponentError, match="result boundary"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


def test_fully_resigned_command_substitution_and_bool_returncode_fail_closed(tmp_path: Path) -> None:
    def mutate(root: Path) -> None:
        execution = _read_json(root / "policy-cpu/execution.json")
        commands = cast(list[dict[str, Any]], execution["commands"])
        commands[1]["argv"][3] = "tests/cpu"
        commands[2]["returncode"] = True
        _write_json(root / "policy-cpu/execution.json", execution)
        report = _read_json(root / "policy-cpu/cpu-report.json")
        report["commands"] = commands
        _write_json(root / "policy-cpu/cpu-report.json", report)

    root, toolchain = _build_component(tmp_path, mutate_before_seal=mutate)
    with pytest.raises(PolicyCPUComponentError, match="argv/order/returncode"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


@pytest.mark.parametrize(
    "junit",
    [
        b'<!DOCTYPE x [<!ENTITY y "boom">]><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase classname="x" name="y" /></testsuite>',
        b'<testsuite tests="2" failures="0" errors="0" skipped="0"><testcase classname="x" name="y" /></testsuite>',
    ],
)
def test_fully_resigned_junit_entity_or_counter_mismatch_is_rejected(tmp_path: Path, junit: bytes) -> None:
    def mutate(root: Path) -> None:
        path = root / "policy-cpu/junit.xml"
        path.write_bytes(junit)
        report = _read_json(root / "policy-cpu/cpu-report.json")
        report["required_artifacts"]["junit"]["sha256"] = _sha(junit)
        _write_json(root / "policy-cpu/cpu-report.json", report)

    root, toolchain = _build_component(tmp_path, mutate_before_seal=mutate)
    with pytest.raises(PolicyCPUComponentError, match="JUnit"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


def test_registered_policy_bytes_cannot_be_replaced_and_resigned(tmp_path: Path) -> None:
    def mutate(root: Path) -> None:
        path = root / "policy-inputs/candidate-policies/fp32.json"
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        descriptor["invented"] = "field"
        path.write_bytes(_canonical(descriptor))

    root, toolchain = _build_component(tmp_path, mutate_before_seal=mutate)
    with pytest.raises(PolicyCPUComponentError, match="registered digest"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


def test_duplicate_and_nonfinite_resigned_json_are_rejected_before_semantic_use(tmp_path: Path) -> None:
    def mutate(root: Path) -> None:
        path = root / "policy-cpu/cpu-report.json"
        raw = path.read_bytes().rstrip(b"\n")
        assert raw.startswith(b"{")
        path.write_bytes(b'{"schema_version":1,"schema_version":1,"x":NaN}\n')

    root, toolchain = _build_component(tmp_path, mutate_before_seal=mutate)
    with pytest.raises(PolicyCPUComponentError, match="duplicate JSON|non-finite"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


def test_toolchain_and_portable_gate_identity_are_caller_bound(tmp_path: Path) -> None:
    root, toolchain = _build_component(tmp_path)
    wrong = json.loads(json.dumps(toolchain))
    wrong["project_root_digest"] = "f" * 64
    projection = {key: value for key, value in wrong.items() if key != "manifest_digest"}
    wrong["manifest_digest"] = _sha(_canonical(projection))

    with pytest.raises(PolicyCPUComponentError, match="caller-verified"):
        verify_policy_cpu_development_component(root, expected_toolchain=wrong, gate_relative=GATE_RELATIVE)
    with pytest.raises(PolicyCPUComponentError, match="output_root|identity|differs"):
        verify_policy_cpu_development_component(
            root,
            expected_toolchain=toolchain,
            gate_relative="out/other-component",
        )
    with pytest.raises(PolicyCPUComponentError, match="below out"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative="../escape")


def test_component_tree_rejects_linked_evidence_when_host_allows_it(tmp_path: Path) -> None:
    root, toolchain = _build_component(tmp_path)
    target = root / "policy-cpu/logs/pytest.stdout.txt"
    external = tmp_path / "external.txt"
    external.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("host does not permit symlinks")
    with pytest.raises(PolicyCPUComponentError, match="links|ordinary"):
        verify_policy_cpu_development_component(root, expected_toolchain=toolchain, gate_relative=GATE_RELATIVE)


def test_component_reader_import_is_tensor_free_in_isolated_python() -> None:
    source_root = Path("src").resolve()
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import illm.stateful_policy_cpu_component;"
        "blocked={'torch','transformers','tokenizers','safetensors'};"
        "loaded=blocked.intersection({name.split('.')[0] for name in sys.modules});"
        "assert not loaded, sorted(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr

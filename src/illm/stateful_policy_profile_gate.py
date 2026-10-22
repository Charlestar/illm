"""Portable transcript checks for the policy-aware CPU profile, revision 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.policy_cpu_numerical_contract import NumericalEvidenceError, plain_path, read_json
from illm.policy_cpu_raw_bundle import inventory
from illm.reference_qualification import _relative
from illm.stateful_policy_gate_contract import (
    APPROVAL_SHA256, POLICY_CPU_PROFILE_IDS, POLICY_PROFILE_RUNTIME_IDS,
    PROTOCOL_SHA256, REQUIRED_RESULT_AXES, policy_phase_commands, policy_phase_parameters,
)
from illm.stateful_qualification import _absolute_argv_path, _verify_bootstrap_argv


GATE_KIND = "illm-v02-policy-profile-gate-v1"
GATE_FIELDS = frozenset({
    "schema_version", "report_kind", "contract_revision", "runtime_profile_id", "profile_id", "status",
    "fresh_environment", "offline_bootstrap", "network_used", "network_blocking", "network_policy",
    "toolchain_manifest_digest", "source_identity", "protocol_sha256", "approval_sha256", "commands",
    "qualification_status", "global_qualification_granted", "gpu_qualification_granted",
    *REQUIRED_RESULT_AXES, "report_digest",
})
COMMAND_FIELDS = frozenset({
    "phase", "argv", "parameters", "returncode", "stdout_file", "stderr_file", "stdout_sha256", "stderr_sha256",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericalEvidenceError("policy CPU gate: " + message)


def _same(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _argv(value: Any) -> list[str]:
    _require(isinstance(value, list) and bool(value)
             and all(type(arg) is str and bool(arg) and "\0" not in arg for arg in value), "invalid argv")
    return list(value)


def _exact_argv(actual: list[str], expected: list[str]) -> None:
    _require(len(actual) == len(expected), "extra, omitted, or reordered argv")
    for observed, wanted in zip(actual, expected, strict=True):
        # Only declared absolute artifact/interpreter paths receive separator
        # normalization.  No author path is resolved or opened on this host.
        if wanted.startswith("/") or (len(wanted) > 2 and wanted[1:3] == ":/"):
            observed = _absolute_argv_path(observed, "declared command path")
        _require(observed == wanted, "command differs from its closed phase contract")


def verify_policy_profile_gate(
    root: Path, *, profile_id: str, toolchain: Mapping[str, Any], source_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Check all fifteen archived commands and exact stdout/stderr file links.

    The return value includes a portable historical gate-relative path used by
    the embedded development-component reader. It grants no release fitness.
    """
    _require(profile_id in POLICY_CPU_PROFILE_IDS, "only authorized CPU profiles have an executable contract")
    root = plain_path(root)
    report = read_json(root / "gate-report.json")
    _require(isinstance(report, dict) and set(report) == GATE_FIELDS, "closed report fields differ")
    runtime = POLICY_PROFILE_RUNTIME_IDS[profile_id]
    _require(type(report["schema_version"]) is int and report["schema_version"] == 3
             and type(report["contract_revision"]) is int and report["contract_revision"] == 2
             and report["report_kind"] == GATE_KIND and report["profile_id"] == profile_id
             and report["runtime_profile_id"] == runtime and toolchain.get("runtime_profile_id") == runtime
             and report["status"] == "pass" and report["fresh_environment"] is True
             and report["offline_bootstrap"] is True and report["network_used"] == "not_observed"
             and report["network_blocking"] == "not_enforced"
             and report["network_policy"] == "offline_resolution_and_local_model_only"
             and report["protocol_sha256"] == PROTOCOL_SHA256 and report["approval_sha256"] == APPROVAL_SHA256
             and report["qualification_status"] == "unqualified"
             and report["global_qualification_granted"] is False and report["gpu_qualification_granted"] is False
             and all(report[axis] == "incomplete" for axis in REQUIRED_RESULT_AXES), "report status/scope differs")
    _require(_same(report["source_identity"], dict(source_identity))
             and report["toolchain_manifest_digest"] == toolchain.get("manifest_digest")
             and report["report_digest"] == sha256_json({
                 name: value for name, value in report.items() if name != "report_digest"
             }), "report source/toolchain/digest differs")
    commands = report["commands"]
    parameters = policy_phase_parameters(profile_id)
    _require(isinstance(commands, list) and len(commands) == len(parameters), "phase inventory differs")
    _require(_same(read_json(root / "execution.json"), {
        "schema_version": 3, "contract_revision": 2, "profile_id": profile_id, "commands": commands,
    }), "execution journal differs from the gate")
    bootstrap = commands[0]
    _require(isinstance(bootstrap, dict) and set(bootstrap) == COMMAND_FIELDS, "bootstrap fields differ")
    bootstrap_argv = _argv(bootstrap["argv"])
    interpreter, project, gate = _verify_bootstrap_argv(bootstrap_argv, runtime)
    _require(bootstrap_argv[3] == toolchain.get("profile")
             and project == _absolute_argv_path(str(toolchain.get("project_root", "")), "toolchain project")
             and gate.startswith(project + "/out/"), "bootstrap profile/project/output linkage differs")
    relative = _relative(gate[len(project) + 1:])
    _require(relative.startswith("out/"), "gate is not a project-local output")
    expected_commands = policy_phase_commands(profile_id, Path(gate))
    prefix = [
        interpreter, project + "/scripts/in_env.py", "--manifest", gate + "/toolchain.json",
        "--expected-manifest-digest-file", gate + "/toolchain.sha256", "--",
    ]
    expected_logs: set[str] = set()
    for command, phase in zip(commands, parameters, strict=True):
        _require(isinstance(command, dict) and set(command) == COMMAND_FIELDS and command["phase"] == phase,
                 "command fields or phase order differ")
        _require(_same(command["parameters"], parameters[phase]) and type(command["returncode"]) is int
                 and command["returncode"] == 0, "phase parameters or exit status differ")
        argv = _argv(command["argv"])
        if phase != "bootstrap":
            expected = [*prefix, *expected_commands[phase]]
            expected = [arg.replace("\\", "/") if arg.replace("\\", "/").startswith(gate + "/") else arg
                        for arg in expected]
            _exact_argv(argv, expected)
        for stream in ("stdout", "stderr"):
            path = f"logs/{phase}.{stream}.txt"
            _require(command[stream + "_file"] == path
                     and command[stream + "_sha256"] == sha256_bytes(plain_path(root / path).read_bytes()),
                     "phase log path or bytes differ")
            expected_logs.add(path)
    actual_files = {record["path"] for record in inventory(root)}
    _require(actual_files == expected_logs | {"gate-report.json", "execution.json"},
             "extra, missing, aliased, or empty-directory gate evidence")
    return {
        "report_digest": report["report_digest"], "command_count": len(commands), "gate_relative": relative,
        "runtime_profile_id": runtime, "toolchain_manifest_digest": toolchain["manifest_digest"],
        "protocol_sha256": PROTOCOL_SHA256, "approval_sha256": APPROVAL_SHA256,
        "qualification_status": "unqualified", "global_qualification_granted": False,
        "gpu_qualification_granted": False,
    }


__all__ = ["GATE_KIND", "verify_policy_profile_gate"]

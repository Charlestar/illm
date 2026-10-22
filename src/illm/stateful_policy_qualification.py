"""Version-3 CPU profile evidence; never a release or GPU qualification.

Revision 2 embeds the original development components unchanged, then derives
its CPU-only pass from their bytes. All three global result axes stay incomplete.
Portable readers do not execute archived code or follow author-machine paths.
Checksums establish internal consistency, not independent author authenticity.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.metrics import metric_registry, registry_conformance, registry_digest
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.policy_capture_contract import validate_execution_binding
from illm.policy_capture_proof import verify_policy_capture_proof
from illm.policy_cpu_numerical_contract import (
    NumericalEvidenceError, plain_path, read_json, strict_json, write_json_exclusive,
)
from illm.policy_cpu_raw_bundle import inventory, verify_bundle
from illm.reference_qualification import (
    BASE_LOCKS, _assert_live_environment, _relative, _verify_toolchain,
)
from illm.stateful_policy_gate_contract import (
    APPROVAL_PATH, APPROVAL_SHA256, INPUT_BYTES_SHA256, INPUT_MANIFEST_DIGEST, INPUT_PATH,
    METRIC_REGISTRY_DIGEST, POLICY_CPU_PROFILE_IDS, POLICY_PROFILE_RUNTIME_IDS,
    PROTOCOL_PATH, PROTOCOL_SHA256, REQUIRED_RESULT_AXES,
)
from illm.stateful_policy_profile_gate import verify_policy_profile_gate
from illm.stateful_qualification import CRITICAL_TESTS, _absolute_argv_path, _verify_v02_source
from illm.streaming_learning import verify_streaming_learning_assets


PROFILE_KIND = "illm-v02-policy-profile-qualification-v1"
INDEX_KIND = "illm-v02-policy-profile-evidence-index-v1"
SCHEMA_VERSION = 3
CONTRACT_REVISION = 2
_SEAL_FILES = frozenset({"profile-result.json", "evidence-index.json", "SHA256SUMS.json", "COMPLETE"})
_FLAT_FILES = frozenset({
    "toolchain.json", "toolchain.sha256", "source-provenance.json", "policy-attempt.json",
    *("locks/" + name for name in BASE_LOCKS), "locks/pyproject.toml", "locks/runtime-lock.json",
    "policy/protocol.md", "policy/approval.json", "policy/inputs.json", "policy/registry.json",
    "metrics/metric-registry-v1.json", "metrics/metric-registry-conformance.json",
    "provenance/spec.md", "provenance/check_policy_capture_portable.py",
})
_COMPONENT_DIRS = (
    "gate", "policy-inputs", "policy-cpu", "policy-verified", "raw-numerical",
    "demo-b", "demo-c", "exercise", "portable", "learning",
)
PROFILE_FIELDS = frozenset({
    "schema_version", "report_kind", "contract_revision", "profile_id", "simulated_profile_id",
    "evidence_mode", "result", "required_for_functional", "scope", "qualification_tier",
    "global_qualification_granted", "gpu_qualification_granted", *REQUIRED_RESULT_AXES,
    "cpu_results", "environment_manifest_digest", "source_identity", "execution_binding",
    "execution_binding_digest", "protocol_sha256", "approval_sha256", "input_manifest_digest",
    "policy_registry_digest", "evidence_digest", "components", "reason", "profile_digest",
})
INDEX_FIELDS = frozenset({
    "schema_version", "report_kind", "contract_revision", "profile_id", "evidence_mode", "scope",
    "os", "arch", "python_version", "source_identity", "runtime_profile_id", "runtime_lock_digest",
    "protocol_sha256", "policy_registry_digest", "files", "components", "evidence_digest",
    "profile_digest", "index_digest",
})
_RESOURCE_FIELDS = frozenset({
    "active_requests", "reservations", "quarantined_requests", "active_tickets", "input_leases",
    "output_leases", "workspace_leases", "kv_write_leases", "kv_reserved_bytes", "kv_valid_tokens", "kv_staged_tokens",
})
_RESULT_FIELDS = frozenset({
    "schema_version", "request_id", "input_token_ids", "output_token_ids", "text", "outcome", "finish_reason",
    "error_code", "actual_seed", "model_spec_id", "sampling_spec_digest", "kv_mode", "next_output_position",
    "rng_draws_consumed", "committed_draw_addresses", "query_tokens_per_step", "kv_valid_tokens_per_step",
    "resource_status", "timings", "capture_manifest_digest", "capture_error", "event_count",
    "numerical_policy", "numerical_policy_digest", "execution_binding", "execution_binding_digest", "memory_plan_digest",
})
_METADATA_FIELDS = frozenset({
    "kind", "schema_version", "request_id", "model_spec_id", "sampling_spec_digest", "memory_plan_digest",
    "input_tokens", "kv_mode", "accepted_ns", "numerical_policy", "numerical_policy_digest",
    "execution_binding", "execution_binding_digest",
})
_TOKEN_FIELDS = frozenset({
    "kind", "request_id", "output_index", "token_id", "text_delta", "server_ready_ns",
    "kv_valid_token_count", "committed_output_count",
})
_TERMINAL_FIELDS = frozenset({
    "kind", "request_id", "outcome", "finish_reason", "error_code", "final_text_delta", "server_ready_ns", "result",
})
_TIMING_FIELDS = frozenset({
    "accepted_ns", "clock_domain", "critical_capture_event_drop_count", "evidence_kind", "per_forward_wall_time_ns",
    "server_e2e_wall_time_ns", "telemetry", "terminal_ready_ns", "token_ready_ns", "tokenization_wall_time_ns",
})
REQUIRED_EXECUTED_REGRESSIONS = CRITICAL_TESTS | frozenset({
    "test_actual_candidate_b_c_cli_match_fixed_public_sequences_and_capture_story",
    "test_actual_demo_uses_literal_finish_reason_not_a_normalized_alias",
    "test_actual_candidate_policy_lifecycle_has_complete_closed_coverage",
    "test_every_actual_cycle_is_schema3_candidate_and_returns_all_resources",
    "test_fault_table_preserves_real_quarantine_and_cpu_simulation_boundary",
    "test_actual_candidate_c_reaches_fenced_zero_resource_close",
    "test_actual_explicit_legacy_none_is_a_new_full_context_request",
    "test_archived_proof_is_recomputed_without_upgrading_qualification",
    "test_proof_reader_import_is_stdlib_only_in_isolated_python",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericalEvidenceError("policy CPU profile: " + message)


def _same(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def policy_registry_snapshot() -> dict[str, Any]:
    """Derive all six descriptors from the closed registry, not report labels."""
    entries = []
    for family, factory in (("candidate", candidate_policy), ("legacy", legacy_policy)):
        for dtype in ("bf16", "fp16", "fp32"):
            policy = factory(dtype)
            entries.append({"selector": f"{family}-{dtype}", "policy": policy.to_dict(), "policy_digest": policy.digest})
    value: dict[str, Any] = {"schema_version": 1, "registry_kind": "illm-v02-numerical-policy-registry-v1",
                             "instances": entries}
    value["registry_digest"] = sha256_json(value)
    return value


def _file_map(root: Path, *, exclude: frozenset[str] = frozenset()) -> dict[str, str]:
    records = inventory(root, exclude=exclude)
    names = [record["path"] for record in records]
    _require(len({name.casefold() for name in names}) == len(names), "case-colliding evidence paths")
    for name in names:
        _require(_relative(name) == name, "noncanonical evidence path")
    return {record["path"]: record["sha256"] for record in records}


def _closed_inventory(root: Path, *, sealed: bool) -> dict[str, str]:
    files = _file_map(root, exclude=frozenset({"INCOMPLETE"}) if not sealed else frozenset())
    facts = {name: digest for name, digest in files.items() if name not in _SEAL_FILES}
    _require(_FLAT_FILES <= set(facts), "required evidence files missing")
    for directory in _COMPONENT_DIRS:
        _require(any(name.startswith(directory + "/") for name in facts), "required component missing: " + directory)
    _require(all(name in _FLAT_FILES or name.split("/")[0] in _COMPONENT_DIRS for name in facts),
             "unlisted profile evidence member")
    _require(set(files) - set(facts) == (_SEAL_FILES if sealed else frozenset()), "profile seal inventory differs")
    return facts


def _binding(value: Any, toolchain: Mapping[str, Any], source: Mapping[str, str]) -> dict[str, Any]:
    _require(isinstance(value, dict), "missing execution binding")
    binding = validate_execution_binding(value)
    _require(binding["implementation_source_digest"] == source["project_root_digest"]
             and binding["protocol_sha256"] == PROTOCOL_SHA256
             and _same(binding["toolchain"], {
                 "binding": "source-bound-manifest", "manifest_digest": toolchain["manifest_digest"],
                 "environment_id": toolchain["environment_id"], "lock_digest": toolchain["lock_digest"],
             }), "execution source/environment/toolchain binding differs")
    environment = binding["environment"]
    _require(environment["python_version"] == toolchain["python_version"]
             and environment["machine"] == toolchain["arch"]
             and environment["torch_version"] == toolchain["installed_packages"]["torch"],
             "execution binding software differs from the finalized toolchain")
    return binding


def _demo(root: Path, name: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    bundle = PolicyStreamingCaptureBundle.open(root / name)
    manifest = bundle.manifest
    expected_mode = "none" if name == "demo-b" else "contiguous"
    expected_id = "req-policy-b-0001" if name == "demo-b" else "req-policy-c-0001"
    policy = candidate_policy("fp32")
    _require(_same(manifest["execution_binding"], binding)
             and manifest["numerical_policy_digest"] == policy.digest
             and _same(manifest["numerical_policy"], policy.to_dict())
             and manifest["memory_plan"]["kv_mode"] == expected_mode
             and manifest["request_id"] == expected_id and manifest["terminal_outcome"] == "FINISHED"
             and manifest["capabilities"] == ["R0"] and manifest["observation_profile"] == "request",
             "demo capture policy/KV/request/binding differs")
    raw = plain_path(root / "gate/logs" / f"{name}.stdout.txt").read_bytes()
    chunks = [strict_json(line, label="policy demo stdout") for line in raw.splitlines()]
    _require(len(chunks) == 10 and all(isinstance(chunk, dict) for chunk in chunks), "demo chunk sequence differs")
    metadata, *tokens, terminal = chunks
    result = terminal.get("result")
    _require(set(metadata) == _METADATA_FIELDS and set(terminal) == _TERMINAL_FIELDS
             and metadata.get("kind") == "metadata" and metadata.get("schema_version") == 3
             and metadata.get("request_id") == expected_id
             and metadata.get("memory_plan_digest") == manifest["memory_plan_digest"]
             and _same(metadata.get("execution_binding"), binding)
             and metadata.get("execution_binding_digest") == sha256_json(binding)
             and _same(metadata.get("numerical_policy"), policy.to_dict())
             and metadata.get("numerical_policy_digest") == policy.digest
             and metadata.get("model_spec_id") == manifest["identity"]["model_spec_id"]
             and metadata.get("sampling_spec_digest") == manifest["sampling_spec_digest"]
             and _same(metadata.get("input_tokens"), 6) and metadata.get("kv_mode") == expected_mode
             and terminal.get("kind") == "terminal" and terminal.get("request_id") == expected_id
             and terminal.get("outcome") == "FINISHED" and terminal.get("finish_reason") == "MAX_NEW_TOKENS"
             and terminal.get("error_code") is None and isinstance(result, dict)
             and set(result) == _RESULT_FIELDS, "demo metadata/terminal/result envelope differs")
    _require(_same(result["schema_version"], 3) and result["request_id"] == expected_id
             and _same(result["input_token_ids"], [1, 107, 104, 111, 111, 114])
             and _same(result["output_token_ids"], [114] * 8) and result["text"] == "oooooooo"
             and result["outcome"] == "FINISHED" and result["finish_reason"] == "MAX_NEW_TOKENS"
             and result["error_code"] is None and result["capture_error"] is None
             and _same(result["actual_seed"], 7) and result["kv_mode"] == expected_mode
             and _same(result["next_output_position"], 8) and _same(result["rng_draws_consumed"], 0)
             and result["committed_draw_addresses"] == []
             and _same(result["query_tokens_per_step"], list(range(6, 14)) if expected_mode == "none" else [6] + [1] * 7)
             and _same(result["kv_valid_tokens_per_step"], [0] * 8 if expected_mode == "none" else list(range(6, 14)))
             and _same(result["numerical_policy"], policy.to_dict()) and result["numerical_policy_digest"] == policy.digest
             and _same(result["execution_binding"], binding) and result["execution_binding_digest"] == sha256_json(binding)
             and result["capture_manifest_digest"] == manifest["manifest_digest"]
             and result["memory_plan_digest"] == manifest["memory_plan_digest"]
             and result["sampling_spec_digest"] == manifest["sampling_spec_digest"]
             and result["model_spec_id"] == manifest["identity"]["model_spec_id"]
             and _same(result["event_count"], manifest["event_count"]), "demo full result differs from frozen public authority")
    resources = result["resource_status"]
    _require(isinstance(resources, dict) and set(resources) == _RESOURCE_FIELDS
             and all(type(value) is int and value == 0 for value in resources.values()), "demo resources not reclaimed")
    timings = result["timings"]
    _require(isinstance(timings, dict) and set(timings) == _TIMING_FIELDS
             and timings["clock_domain"] == "host.perf_counter_ns" and timings["evidence_kind"] == "measured"
             and timings["telemetry"] == "request" and _same(timings["critical_capture_event_drop_count"], 0),
             "demo timing envelope differs")
    for timing_name in ("accepted_ns", "terminal_ready_ns", "server_e2e_wall_time_ns", "tokenization_wall_time_ns"):
        _require(type(timings[timing_name]) is int and timings[timing_name] >= 0,
                 "demo timing value must be a nonnegative integer")
    for timing_name in ("per_forward_wall_time_ns", "token_ready_ns"):
        _require(isinstance(timings[timing_name], list) and len(timings[timing_name]) == 8
                 and all(type(value) is int and value >= 0 for value in timings[timing_name]), "demo timing vector differs")
    _require(_same(metadata["accepted_ns"], timings["accepted_ns"])
             and _same(terminal["server_ready_ns"], timings["terminal_ready_ns"])
             and timings["server_e2e_wall_time_ns"] == timings["terminal_ready_ns"] - timings["accepted_ns"]
             and timings["accepted_ns"] <= timings["token_ready_ns"][0]
             and timings["token_ready_ns"] == sorted(timings["token_ready_ns"])
             and timings["token_ready_ns"][-1] <= timings["terminal_ready_ns"]
             and terminal["final_text_delta"] == "", "demo stream/timing ordering differs")
    for index, chunk in enumerate(tokens):
        _require(set(chunk) == _TOKEN_FIELDS and chunk.get("kind") == "token" and chunk.get("request_id") == expected_id
                 and _same(chunk.get("output_index"), index) and _same(chunk.get("token_id"), 114)
                 and chunk.get("text_delta") == "o" and _same(chunk.get("server_ready_ns"), timings["token_ready_ns"][index])
                 and _same(chunk.get("committed_output_count"), index + 1)
                 and _same(chunk.get("kv_valid_token_count"), 0 if expected_mode == "none" else 6 + index),
                 "demo token stream differs from its complete result")
    if name == "demo-c":
        story = bundle.request_story(expected_id)
        for phase in ("replay-c", "inspect-c"):
            saved = strict_json(plain_path(root / "gate/logs" / f"{phase}.stdout.txt").read_bytes(), label=phase)
            _require(_same(saved, story), "R0 replay/inspection differs from capture-derived story")
    return {"manifest_digest": manifest["manifest_digest"], "memory_plan_digest": manifest["memory_plan_digest"],
            "result_digest": sha256_json(result), "event_count": manifest["event_count"]}


def _facts(
    root: Path, profile_id: str, expected: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Imports are lazy to keep the shared schema dispatcher usable while the
    # sibling CPU lifecycle and development-component modules evolve separately.
    from illm.stateful_policy_cpu_component import verify_policy_cpu_development_component
    from illm.stateful_policy_exercise import verify_policy_exercise_output

    _require(profile_id in POLICY_CPU_PROFILE_IDS, "unknown, synthetic, or GPU profile is not authorized")
    identity, versions = _verify_v02_source(root, expected)
    toolchain = _verify_toolchain(root, identity, versions)
    _require(toolchain["runtime_profile_id"] == POLICY_PROFILE_RUNTIME_IDS[profile_id], "profile/runtime differs")
    _require(plain_path(root / "toolchain.sha256").read_bytes() == (toolchain["manifest_digest"] + "\n").encode("ascii"),
             "toolchain digest sidecar differs")
    provenance = read_json(root / "source-provenance.json")
    files = provenance["project_files"]
    for relative, authority, digest in (
        ("policy/protocol.md", PROTOCOL_PATH, PROTOCOL_SHA256),
        ("policy/approval.json", APPROVAL_PATH, APPROVAL_SHA256),
        ("policy/inputs.json", INPUT_PATH, INPUT_BYTES_SHA256),
    ):
        _require(sha256_bytes(plain_path(root / relative).read_bytes()) == digest == files.get(authority),
                 "frozen protocol/approval/input bytes differ")
    spec = plain_path(root / "provenance/spec.md").read_bytes()
    _require(sha256_bytes(spec) == files.get("docs/specs/versions/v0.2-stateful-streaming.md")
             and b"spec_version: 3" in spec, "embedded specification is not source-bound SPEC-v3")
    registry = policy_registry_snapshot()
    _require(_same(read_json(root / "policy/registry.json"), registry), "policy descriptor registry differs")
    metrics = metric_registry()
    _require(registry_digest() == METRIC_REGISTRY_DIGEST
             and _same(read_json(root / "metrics/metric-registry-v1.json"), metrics)
             and _same(read_json(root / "metrics/metric-registry-conformance.json"), registry_conformance(metrics)),
             "metric registry/conformance differs")
    gate = verify_policy_profile_gate(root / "gate", profile_id=profile_id, toolchain=toolchain, source_identity=identity)
    development = verify_policy_cpu_development_component(
        root, expected_toolchain=toolchain, gate_relative=gate["gate_relative"],
    )
    regressions = _required_regressions(root / "policy-cpu/junit.xml")
    numerical = verify_bundle(root / "raw-numerical")
    cpu_results = numerical["cpu_results"]
    _require(all(cpu_results[axis] == "pass" for axis in REQUIRED_RESULT_AXES),
             "tiny-FP32 CPU hard, legacy elementwise, and full sequence gates must all pass")
    raw_manifest = read_json(root / "raw-numerical/bundle-manifest.json")
    binding = _binding(raw_manifest["binding"], toolchain, identity)
    # Bind copied raw-reader sources to the same profile source, not merely to
    # an internally re-signed standalone bundle manifest.
    for name, digest in _file_map(root / "raw-numerical/reader").items():
        _require(files.get("src/" + name) == digest, "raw reader source differs from profile provenance")
    exercise = verify_policy_exercise_output(root / "exercise", require_formal=True)
    _require(_same(exercise["execution_binding"], binding)
             and exercise["protocol_sha256"] == PROTOCOL_SHA256
             and exercise["producer"]["path"] == "scripts/exercise_policy_streaming.py"
             and exercise["producer"]["sha256"] == files.get("scripts/exercise_policy_streaming.py")
             and _same(read_json(root / "exercise/provenance/toolchain-manifest.json"), toolchain)
             and exercise["qualification_status"] == "unqualified"
             and exercise["global_qualification_granted"] is False and exercise["gpu_qualification_granted"] is False,
             "CPU lifecycle binding or qualification boundary differs")
    demos = {name: _demo(root, name, binding) for name in ("demo-b", "demo-c")}
    portable = verify_policy_capture_proof(
        root / "portable", expected_capture=root / "exercise/captures/success",
        producer_source=root / "provenance/check_policy_capture_portable.py", source_files=files,
    )
    learning = verify_streaming_learning_assets(root / "exercise/captures/cancel-after-launch", root / "learning")
    _phase_outputs(root, toolchain=toolchain, gate=gate, numerical=numerical, raw_manifest=raw_manifest,
                   exercise=exercise, portable=portable, learning=learning)
    components = {
        "gate": sha256_json(gate), "policy_development": sha256_json(development),
        "executed_regressions": sha256_json(regressions),
        "raw_numerical": sha256_json(numerical), "lifecycle": sha256_json(exercise),
        "demos": sha256_json(demos), "portable_capture": sha256_json(portable), "learning": sha256_json(learning),
    }
    return toolchain, identity, components, binding, cpu_results


def _required_regressions(path: Path) -> dict[str, Any]:
    raw = plain_path(path).read_bytes()
    _require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(), "JUnit cannot contain XML declarations")
    document = ET.fromstring(raw)
    passed = {
        case.attrib.get("name", "").split("[")[0] for case in document.iter("testcase")
        if all(case.find(outcome) is None for outcome in ("skipped", "failure", "error"))
    }
    _require(REQUIRED_EXECUTED_REGRESSIONS <= passed, "JUnit omits required actually executed policy/legacy regressions")
    return {"required_executed_regressions": sorted(REQUIRED_EXECUTED_REGRESSIONS), "status": "pass"}


def _phase_outputs(
    root: Path, *, toolchain: Mapping[str, Any], gate: Mapping[str, Any], numerical: Mapping[str, Any],
    raw_manifest: Mapping[str, Any], exercise: Mapping[str, Any], portable: Mapping[str, Any], learning: Mapping[str, Any],
) -> None:
    """Bind archived command output to the actual embedded component facts."""
    def stdout(phase: str) -> Any:
        return strict_json(plain_path(root / "gate/logs" / (phase + ".stdout.txt")).read_bytes(), label=phase)

    # Bootstrap deliberately streams offline pip installation text before its
    # final one-line manifest receipt. Preserve those original log bytes.
    bootstrap_lines = [line for line in plain_path(root / "gate/logs/bootstrap.stdout.txt").read_bytes().splitlines()
                       if line.strip()]
    _require(bool(bootstrap_lines) and _same(strict_json(bootstrap_lines[-1], label="bootstrap manifest receipt"),
                                            dict(toolchain)), "bootstrap stdout differs from its finalized manifest")
    for phase in ("policy-inputs", "policy-cpu", "policy-verify"):
        _require(plain_path(root / "gate/logs" / (phase + ".stdout.txt")).read_bytes() == b"",
                 "development stage has unexpected success stdout")
    _require(_same(stdout("raw-numerical-build"), {
        "status": numerical["status"], "cpu_results": numerical["cpu_results"],
        "manifest_digest": raw_manifest["manifest_digest"], "report_kind": numerical["report_kind"],
        "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified",
    }), "raw collector stdout differs from the complete raw component")
    _require(_same(stdout("raw-numerical-verify"), read_json(root / "raw-numerical/logs/reader-copy.stdout.log")),
             "full raw verification stdout differs from its isolated recomputation")
    anchor = _absolute_argv_path(str(toolchain["project_root"]), "original project") + "/" + str(gate["gate_relative"])
    exercise_stdout = stdout("policy-exercise")
    _require(isinstance(exercise_stdout, dict) and type(exercise_stdout.get("output")) is str
             and _absolute_argv_path(exercise_stdout["output"], "lifecycle output") == anchor + "/exercise"
             and _same(exercise_stdout, {
                 "status": exercise["status"], "report_digest": exercise["report_digest"],
                 "evidence_stage": exercise["evidence_stage"], "qualification_status": "unqualified",
                 "output": exercise_stdout["output"],
             }), "lifecycle stdout differs from the archived exercise report")
    for phase in ("capture-portable-build", "capture-portable-verify"):
        saved = stdout(phase)
        _require(isinstance(saved, dict), "portable proof stdout is not an object")
        output = saved.get("output")
        _require(type(output) is str and _absolute_argv_path(output, "portable proof output") == anchor + "/portable",
                 "portable proof stdout output anchor differs")
        expected = {
            "report_kind": "illm-policy-capture-portable-development-v1", "status": "pass",
            "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified",
            "report_sha256": portable["report_sha256"], "manifest_digest": portable["capture_manifest_digest"],
            "output": output,
        }
        if phase == "capture-portable-verify":
            expected["source_capture_argument_used"] = False
        _require(_same(saved, expected), "portable proof stdout differs from archived facts")
    for phase in ("learning-build", "learning-verify"):
        _require(_same(stdout(phase), dict(learning)), "learning stdout differs from exact capture-derived regeneration")


def _profile(
    profile_id: str, toolchain: Mapping[str, Any], identity: Mapping[str, str], components: Mapping[str, Any],
    binding: Mapping[str, Any], cpu_results: Mapping[str, Any], evidence_digest: str,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "report_kind": PROFILE_KIND, "contract_revision": CONTRACT_REVISION,
        "profile_id": profile_id, "simulated_profile_id": None, "evidence_mode": "formal", "result": "pass",
        "required_for_functional": True, "scope": "cpu-profile-only", "qualification_tier": "unqualified",
        "global_qualification_granted": False, "gpu_qualification_granted": False,
        **{axis: "incomplete" for axis in REQUIRED_RESULT_AXES}, "cpu_results": dict(cpu_results),
        "environment_manifest_digest": toolchain["manifest_digest"], "source_identity": dict(identity),
        "execution_binding": dict(binding), "execution_binding_digest": sha256_json(binding),
        "protocol_sha256": PROTOCOL_SHA256, "approval_sha256": APPROVAL_SHA256,
        "input_manifest_digest": INPUT_MANIFEST_DIGEST,
        "policy_registry_digest": policy_registry_snapshot()["registry_digest"],
        "evidence_digest": evidence_digest, "components": dict(components), "reason": None,
    }
    profile["profile_digest"] = sha256_json(profile)
    return profile


def _index(profile: Mapping[str, Any], toolchain: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    value = {name: profile[name] for name in (
        "schema_version", "contract_revision", "profile_id", "evidence_mode", "scope", "source_identity",
        "protocol_sha256", "policy_registry_digest", "components", "evidence_digest", "profile_digest",
    )}
    value.update({name: toolchain[name] for name in ("os", "arch", "python_version", "runtime_profile_id", "runtime_lock_digest")})
    value.update(report_kind=INDEX_KIND, files=dict(files))
    value["index_digest"] = sha256_json(value)
    return value


def _evidence_digest(profile_id: str, identity: Mapping[str, str], components: Mapping[str, Any], files: Mapping[str, str]) -> str:
    return sha256_json({"schema_version": SCHEMA_VERSION, "contract_revision": CONTRACT_REVISION,
                        "profile_id": profile_id, "source_identity": dict(identity),
                        "components": dict(components), "files": dict(files)})


def verify_policy_stateful_profile_evidence(
    root: Path, *, expected_source_identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read all CPU facts from a copied bundle; never promote or execute code."""
    root = plain_path(root)
    _require(root.is_dir() and not (root / "INCOMPLETE").exists(), "profile is incomplete")
    facts = _closed_inventory(root, sealed=True)
    _require(_same(read_json(root / "SHA256SUMS.json"), _file_map(root, exclude=frozenset({"SHA256SUMS.json"}))),
             "profile checksum coverage differs")
    profile, index = read_json(root / "profile-result.json"), read_json(root / "evidence-index.json")
    _require(isinstance(profile, dict) and set(profile) == PROFILE_FIELDS and isinstance(index, dict)
             and set(index) == INDEX_FIELDS, "unsupported or mixed profile/index fields")
    _require(type(profile["schema_version"]) is int and profile["schema_version"] == SCHEMA_VERSION
             and type(profile["contract_revision"]) is int and profile["contract_revision"] == CONTRACT_REVISION
             and profile["report_kind"] == PROFILE_KIND, "unsupported profile contract")
    profile_id = profile["profile_id"]
    _require(type(profile_id) is str and profile_id in POLICY_CPU_PROFILE_IDS, "profile is not an authorized CPU profile")
    toolchain, identity, components, binding, cpu_results = _facts(root, profile_id, expected_source_identity)
    expected_profile = _profile(profile_id, toolchain, identity, components, binding, cpu_results,
                                _evidence_digest(profile_id, identity, components, facts))
    _require(_same(profile, expected_profile) and _same(index, _index(expected_profile, toolchain, facts)),
             "profile/index differs from fully re-derived CPU facts")
    _require(plain_path(root / "COMPLETE").read_bytes() == (profile["profile_digest"] + "\n").encode("ascii"),
             "profile completion marker differs")
    return profile, index


def _copy_file(source: Path, destination: Path) -> None:
    source, destination = plain_path(source), plain_path(destination)
    _require(source.is_file(), "copy source is not an ordinary file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)


def _copy_tree(source: Path, destination: Path) -> None:
    # Inventory first rejects hardlinks/reparse points/empty dirs/case aliases;
    # exact per-file reads avoid shutil following links encountered at copy time.
    for name in _file_map(source):
        _copy_file(source / name, destination / name)


def create_policy_stateful_profile_evidence(
    *, profile_id: str, gate_root: Path, output: Path,
    expected_source_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal only a complete fresh CPU gate, leaving failed output diagnosable.

    An existing output is never replaced. The writer requires the finalized
    source-bound interpreter; portable reading does not require that interpreter.
    """
    _require(profile_id in POLICY_CPU_PROFILE_IDS, "GPU execution/qualification is deferred")
    gate, output = plain_path(gate_root), plain_path(output)
    _require(not output.exists() and not gate.is_relative_to(output), "output exists or contains the gate")
    _require(_same(read_json(gate / "INCOMPLETE"), {"reason": "gate_in_progress", "profile_id": profile_id}),
             "gate is not at its original complete-phase pre-seal boundary")
    toolchain = read_json(gate / "toolchain.json")
    project = plain_path(Path(toolchain["project_root"]))
    provenance = _assert_live_environment(toolchain, project)
    allowed = {
        "INCOMPLETE", "logs", "toolchain.json", "toolchain.sha256", "execution.json", "gate-report.json",
        "policy-attempt.json", "policy-cpu-pytest-scratch", *(_COMPONENT_DIRS[1:]),
    }
    _require({path.name for path in gate.iterdir()} <= allowed, "unlisted gate output")
    scratch = gate / "policy-cpu-pytest-scratch"
    _require(plain_path(scratch).is_dir(), "declared retained pytest scratch is missing or aliased")
    output.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output / "INCOMPLETE", {"reason": "policy_cpu_profile_sealing"})
    try:
        for name in ("toolchain.json", "toolchain.sha256", "policy-attempt.json"):
            _copy_file(gate / name, output / name)
        write_json_exclusive(output / "source-provenance.json", provenance)
        for name in BASE_LOCKS:
            _copy_file(project / name, output / "locks" / name)
        _copy_file(project / "pyproject.toml", output / "locks/pyproject.toml")
        runtime = POLICY_PROFILE_RUNTIME_IDS[profile_id]
        _copy_file(project / "runtime/artifacts" / (runtime + ".json"), output / "locks/runtime-lock.json")
        for source, destination in (
            (PROTOCOL_PATH, "policy/protocol.md"), (APPROVAL_PATH, "policy/approval.json"),
            (INPUT_PATH, "policy/inputs.json"), ("docs/specs/versions/v0.2-stateful-streaming.md", "provenance/spec.md"),
            ("scripts/check_policy_capture_portable.py", "provenance/check_policy_capture_portable.py"),
        ):
            _copy_file(project / source, output / destination)
        write_json_exclusive(output / "policy/registry.json", policy_registry_snapshot())
        write_json_exclusive(output / "metrics/metric-registry-v1.json", metric_registry())
        write_json_exclusive(output / "metrics/metric-registry-conformance.json", registry_conformance(metric_registry()))
        _copy_file(gate / "gate-report.json", output / "gate/gate-report.json")
        _copy_file(gate / "execution.json", output / "gate/execution.json")
        _copy_tree(gate / "logs", output / "gate/logs")
        for directory in _COMPONENT_DIRS[1:]:
            _copy_tree(gate / directory, output / directory)
        files = _closed_inventory(output, sealed=False)
        verified, identity, components, binding, cpu_results = _facts(output, profile_id, expected_source_identity)
        _require(_same(_assert_live_environment(toolchain, project), provenance),
                 "live source or finalized environment changed during profile sealing")
        profile = _profile(profile_id, verified, identity, components, binding, cpu_results,
                           _evidence_digest(profile_id, identity, components, files))
        write_json_exclusive(output / "profile-result.json", profile)
        write_json_exclusive(output / "evidence-index.json", _index(profile, verified, files))
        # This is our newly created in-progress marker, not historical evidence.
        (output / "INCOMPLETE").unlink()
        with (output / "COMPLETE").open("xb") as marker:
            marker.write((profile["profile_digest"] + "\n").encode("ascii"))
        write_json_exclusive(output / "SHA256SUMS.json", _file_map(output))
        verify_policy_stateful_profile_evidence(output, expected_source_identity=expected_source_identity)
        return profile
    except BaseException:
        if not (output / "INCOMPLETE").exists():
            write_json_exclusive(output / "INCOMPLETE", {"reason": "policy_cpu_profile_sealing_failed"})
        raise


__all__ = [
    "CONTRACT_REVISION", "INDEX_KIND", "PROFILE_KIND", "SCHEMA_VERSION", "policy_registry_snapshot",
    "create_policy_stateful_profile_evidence", "verify_policy_stateful_profile_evidence",
]

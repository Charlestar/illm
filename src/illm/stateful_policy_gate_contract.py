"""Pure-stdlib execution contract for V0.2 policy-aware CPU profiles.

The contract only describes the ordered child processes and their semantic
parameters.  It does not execute a profile, seal evidence, qualify a release,
or evaluate the independent rollback-v2 release component.

The GPU profile name remains registered so callers cannot accidentally invent
a fourth profile, but human approval explicitly deferred GPU execution.  Both
factories therefore fail before returning any runnable GPU command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


POLICY_QUALIFICATION_SCHEMA_VERSION = 3
POLICY_QUALIFICATION_CONTRACT_REVISION = 2

POLICY_PROFILE_RUNTIME_IDS = {
    "state-machine-cpu-v1": "py312-win-amd64-cpu",
    "stateful-streaming-single-gpu-v1": "py312-linux-x86_64-cu126",
    "clean-room-release-v1": "py312-linux-x86_64-cpu",
}
POLICY_CPU_PROFILE_IDS = frozenset({"state-machine-cpu-v1", "clean-room-release-v1"})
POLICY_GPU_PROFILE_ID = "stateful-streaming-single-gpu-v1"
POLICY_GPU_DEFERRED_REASON = "gpu_execution_deferred_by_human_approval"

PROTOCOL_ID = "v02-numerical-policy-protocol-draft-1"
PROTOCOL_PATH = "docs/implementation/v0.2-numerical-policy-protocol.md"
PROTOCOL_SHA256 = "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"
APPROVAL_PATH = "docs/implementation/v0.2-policy-approval.json"
APPROVAL_SHA256 = "2046ec2425b57d050ff61d7ad636e747b42eb6b08bd88a6883ae2188fae2ab7f"
INPUT_PATH = "docs/implementation/v0.2-policy-inputs-draft.json"
INPUT_BYTES_SHA256 = "93d0de1c5c606a58830eeef2d6b95f4a2b389b4defd17735507d2e95b9adb754"
INPUT_MANIFEST_DIGEST = "a5362433ff4f870cd4c0f1df1b5330ee42ace6be9e934ec159749e642b5bb36e"

CANDIDATE_POLICY_FAMILY = "illm-rowwise-fp32-contraction"
LEGACY_POLICY_FAMILY = "illm-v01-full-context"
CANDIDATE_POLICY_DIGESTS = (
    ("bf16", "c90519d8e1de749be6727cbb898e42b78a189d2f4c813e224dc9b343ce762497"),
    ("fp16", "b3a7f22a649b70d24e1014089ac5d5f8856cb5cb4fb7bdac9f469685cb171cb4"),
    ("fp32", "4e15f0d894637ef4a9533ee4c31adcc18de0b7fddefde98efe2798b9a2a29553"),
)
LEGACY_POLICY_DIGESTS = (
    ("bf16", "da4f28136447df33f87ed5cb6494212b027b83e681141b96bdeced59888b095f"),
    ("fp16", "dd24dfd7f032105cf80bc7927dfd64d2812c75d461426e736782c65166d56c6b"),
    ("fp32", "a1cc7a0e346d39907488e2f0fb15441f2a9eda56a33794a8dcd2568e1a772e4d"),
)
METRIC_REGISTRY_DIGEST = "18618f35bd4a9141de6d6191619150d0ce8121d31f37c7a7f1547a0a0136e3ce"

CAPTURE_MANIFEST_SCHEMA_VERSION = 3
CAPTURE_EVENT_ENVELOPE_VERSION = 1
CAPTURE_EVENT_CATALOG = "illm-stateful-v02-policy-v1"
CAPTURE_CAPABILITIES = ("R0",)
MEMORY_PLAN_SCHEMA_VERSION = 2
STREAM_API_CONTRACT_REVISION = 2
GENERATION_RESULT_SCHEMA_VERSION = 3
NUMERICAL_REPORT_KIND = "illm-stateful-numerical-policy-validation-v1"
NUMERICAL_REPORT_SCHEMA_VERSION = 1
POLICY_BENCHMARK_REPORT_KIND = "illm-stateful-policy-benchmark-v1"
POLICY_BENCHMARK_SCHEMA_VERSION = 1
REQUIRED_RESULT_AXES = (
    "hard_gate_result",
    "legacy_elementwise_compatibility",
    "sequence_compatibility",
)
RESULT_AXIS_VALUES = ("pass", "fail", "incomplete", "not_run")

REQUIRED_POLICY_CAPTURE_IDS = (
    "success",
    "length",
    "eos",
    "cancel-after-launch",
    "no-kv",
)
POLICY_EXERCISE_CYCLE_COUNT = 100
POLICY_EXERCISE_FAULT_COUNT = 12
RAW_CASE_COUNT = 70
RAW_HIGH_PRECISION_CASE_COUNT = 8
RAW_MUTATION_COUNT = 9
RAW_REPRODUCTION_CASE_COUNT = 20

POLICY_ROLLBACK_COMPONENT_KIND = "illm-stateful-policy-rollback-v2"
POLICY_ROLLBACK_COMPONENT_SCHEMA_VERSION = 2

_PHASE_ORDER = (
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


def _registered_policy_instances(
    family: str, digests: tuple[tuple[str, str], ...], prefix: str,
) -> list[dict[str, str]]:
    return [
        {
            "selector": f"{prefix}-{dtype}",
            "family": family,
            "storage_dtype": dtype,
            "digest": digest,
        }
        for dtype, digest in digests
    ]


def _contract_identity() -> dict[str, Any]:
    """Return a fresh JSON-shaped copy of the immutable revision-2 identity."""
    return {
        "qualification_schema_version": POLICY_QUALIFICATION_SCHEMA_VERSION,
        "contract_revision": POLICY_QUALIFICATION_CONTRACT_REVISION,
        "protocol": {"protocol_id": PROTOCOL_ID, "path": PROTOCOL_PATH, "sha256": PROTOCOL_SHA256},
        "approval": {"path": APPROVAL_PATH, "sha256": APPROVAL_SHA256},
        "input_manifest": {
            "path": INPUT_PATH,
            "raw_sha256": INPUT_BYTES_SHA256,
            "manifest_digest": INPUT_MANIFEST_DIGEST,
        },
        "policy_registry": {
            "schema_version": 1,
            "candidate_instances": _registered_policy_instances(
                CANDIDATE_POLICY_FAMILY, CANDIDATE_POLICY_DIGESTS, "candidate",
            ),
            "legacy_adapters": _registered_policy_instances(
                LEGACY_POLICY_FAMILY, LEGACY_POLICY_DIGESTS, "legacy",
            ),
        },
        "metric_registry": {"schema_version": 1, "registry_digest": METRIC_REGISTRY_DIGEST},
        "capture": {
            "manifest_schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
            "event_envelope_version": CAPTURE_EVENT_ENVELOPE_VERSION,
            "event_catalog": CAPTURE_EVENT_CATALOG,
            "capabilities": list(CAPTURE_CAPABILITIES),
        },
        "memory_plan_schema_version": MEMORY_PLAN_SCHEMA_VERSION,
        "stream_api_contract_revision": STREAM_API_CONTRACT_REVISION,
        "generation_result_schema_version": GENERATION_RESULT_SCHEMA_VERSION,
        "result_axes": list(REQUIRED_RESULT_AXES),
        "result_axis_values": list(RESULT_AXIS_VALUES),
        "rollback_component": {
            "report_kind": POLICY_ROLLBACK_COMPONENT_KIND,
            "schema_version": POLICY_ROLLBACK_COMPONENT_SCHEMA_VERSION,
            "required_by_release": True,
            "evaluated_by_profile_gate": False,
        },
    }


def _cpu_profile(profile_id: str) -> str:
    if profile_id not in POLICY_PROFILE_RUNTIME_IDS:
        raise ValueError("unknown V0.2 policy qualification profile")
    if profile_id == POLICY_GPU_PROFILE_ID:
        raise ValueError(
            "V0.2 policy GPU profile has no executable contract: " + POLICY_GPU_DEFERRED_REASON,
        )
    if profile_id not in POLICY_CPU_PROFILE_IDS:  # pragma: no cover - closed registry defence
        raise ValueError("V0.2 policy profile is not a registered CPU profile")
    return POLICY_PROFILE_RUNTIME_IDS[profile_id]


def policy_phase_parameters(profile_id: str) -> dict[str, dict[str, Any]]:
    """Return the ordered, closed semantic phases for one CPU profile.

    ``policy-cpu`` owns the one full quality/pytest/example execution.  There
    are deliberately no duplicate top-level quality or pytest phases.
    """
    runtime = _cpu_profile(profile_id)
    contract = _contract_identity()
    candidate_fp32 = dict(_registered_policy_instances(
        CANDIDATE_POLICY_FAMILY, (CANDIDATE_POLICY_DIGESTS[-1],), "candidate",
    )[0])
    phases: dict[str, dict[str, Any]] = {
        "bootstrap": {
            "runtime_profile_id": runtime,
            "locked": True,
            "offline": True,
            "fresh_environment": True,
            "profile_id": profile_id,
            "contract": contract,
        },
        "policy-inputs": {
            "stage": "inputs",
            "protocol_sha256": PROTOCOL_SHA256,
            "approval_sha256": APPROVAL_SHA256,
            "input_raw_sha256": INPUT_BYTES_SHA256,
            "input_manifest_digest": INPUT_MANIFEST_DIGEST,
        },
        "policy-cpu": {
            "stage": "cpu",
            "nested_execution": ["quality", "pytest", "example"],
            "test_scope": "all",
            "cuda_execution_authorized": False,
        },
        "policy-verify": {
            "stage": "verify",
            "verified_stage_order": ["inputs", "cpu"],
            "evidence_mode": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
        },
        "raw-numerical-build": {
            "report_kind": NUMERICAL_REPORT_KIND,
            "schema_version": NUMERICAL_REPORT_SCHEMA_VERSION,
            "scope": "approved-tiny-fp32-cpu-only",
            "case_count": RAW_CASE_COUNT,
            "high_precision_case_count": RAW_HIGH_PRECISION_CASE_COUNT,
            "mutation_count": RAW_MUTATION_COUNT,
            "reproduction_case_count": RAW_REPRODUCTION_CASE_COUNT,
            "expected_policy": candidate_fp32,
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
            "global_result_axes": {axis: "incomplete" for axis in REQUIRED_RESULT_AXES},
        },
        "raw-numerical-verify": {
            "mode": "read_only_full_raw_recomputation",
            "source_phase": "raw-numerical-build",
            "forbidden_tensor_packages": ["torch", "transformers", "tokenizers", "safetensors"],
        },
        "demo-b": {
            "model": "fixture:tiny-decoder",
            "device": "cpu",
            "request_id": "req-policy-b-0001",
            "prompt": "hello",
            "max_new_tokens": 8,
            "sampling": "greedy",
            "seed": 7,
            "numerical_policy": "candidate-fp32",
            "policy_digest": candidate_fp32["digest"],
            "kv_mode": "none",
            "capture_schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
            "capture_catalog": CAPTURE_EVENT_CATALOG,
            "capabilities": list(CAPTURE_CAPABILITIES),
        },
        "demo-c": {
            "model": "fixture:tiny-decoder",
            "device": "cpu",
            "request_id": "req-policy-c-0001",
            "prompt": "hello",
            "max_new_tokens": 8,
            "sampling": "greedy",
            "seed": 7,
            "numerical_policy": "candidate-fp32",
            "policy_digest": candidate_fp32["digest"],
            "kv_mode": "contiguous",
            "capture_schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
            "capture_catalog": CAPTURE_EVENT_CATALOG,
            "capabilities": list(CAPTURE_CAPABILITIES),
        },
        "replay-c": {"capture": "demo-c", "mode": "R0", "request_id": "req-policy-c-0001"},
        "inspect-c": {"capture": "demo-c", "mode": "R0", "request_id": "req-policy-c-0001"},
        "policy-exercise": {
            "scope": "tiny-policy-state-machine-cpu-v2",
            "cycle_count": POLICY_EXERCISE_CYCLE_COUNT,
            "fault_count": POLICY_EXERCISE_FAULT_COUNT,
            "required_capture_ids": list(REQUIRED_POLICY_CAPTURE_IDS),
            "policy": candidate_fp32,
            "capture_schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
            "capture_catalog": CAPTURE_EVENT_CATALOG,
            "capabilities": list(CAPTURE_CAPABILITIES),
            "rollback_boundary_evaluated": False,
        },
        "capture-portable-build": {
            "capture": "exercise/captures/success",
            "scope": "copied-minimal-reader-and-copied-schema3-r0-capture-internal-consistency",
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
        },
        "capture-portable-verify": {
            "source_phase": "capture-portable-build",
            "mode": "copied_reader_fresh_isolated_process",
            "forbidden_tensor_packages": ["torch", "transformers", "tokenizers", "safetensors"],
        },
        "learning-build": {
            "capture": "exercise/captures/cancel-after-launch",
            "capability": "R0",
        },
        "learning-verify": {
            "capture": "exercise/captures/cancel-after-launch",
            "mode": "exact_regeneration",
        },
    }
    if tuple(phases) != _PHASE_ORDER:  # pragma: no cover - local construction invariant
        raise RuntimeError("V0.2 policy phase order drifted")
    return phases


def policy_phase_commands(profile_id: str, output: Path) -> dict[str, list[str]]:
    """Return exact post-bootstrap argv for a policy-aware CPU profile."""
    parameters = policy_phase_parameters(profile_id)
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")

    attempt = output / "policy-attempt.json"
    inputs = output / "policy-inputs"
    cpu = output / "policy-cpu"
    verified = output / "policy-verified"
    raw = output / "raw-numerical"
    demo_b = output / "demo-b"
    demo_c = output / "demo-c"
    exercise = output / "exercise"
    portable = output / "portable"
    learning = output / "learning"

    def policy_stage(stage: str, destination: Path) -> list[str]:
        return [
            "python",
            "scripts/run_numerical_policy.py",
            "--stage",
            stage,
            "--protocol",
            PROTOCOL_PATH,
            "--approval",
            APPROVAL_PATH,
            "--attempt",
            str(attempt),
            "--output",
            str(destination),
        ]

    commands = {
        "policy-inputs": policy_stage("inputs", inputs),
        "policy-cpu": policy_stage("cpu", cpu),
        "policy-verify": policy_stage("verify", verified),
        "raw-numerical-build": [
            "python", "scripts/validate_policy_cpu_raw.py", "--output", str(raw),
        ],
        "raw-numerical-verify": [
            "python", "scripts/validate_policy_cpu_raw.py", "--verify-existing", str(raw),
        ],
        "demo-b": [
            "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--prompt", "hello", "--max-new-tokens", "8", "--stream", "--request-id", "req-policy-b-0001",
            "--seed", "7",
            "--numerical-policy", "candidate-fp32", "--kv-mode", "none", "--telemetry", "request",
            "--capture", str(demo_b),
        ],
        "demo-c": [
            "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--prompt", "hello", "--max-new-tokens", "8", "--stream", "--request-id", "req-policy-c-0001",
            "--seed", "7",
            "--numerical-policy", "candidate-fp32", "--kv-mode", "contiguous", "--telemetry", "request",
            "--capture", str(demo_c),
        ],
        "replay-c": ["python", "-m", "illm", "replay", str(demo_c), "--mode", "trace"],
        "inspect-c": [
            "python", "-m", "illm", "inspect", "request", str(demo_c), "--request", "req-policy-c-0001",
        ],
        "policy-exercise": [
            "python", "scripts/exercise_policy_streaming.py", "--output", str(exercise),
        ],
        "capture-portable-build": [
            "python", "scripts/check_policy_capture_portable.py", str(exercise / "captures/success"),
            "--output", str(portable),
        ],
        "capture-portable-verify": [
            "python", "scripts/check_policy_capture_portable.py", str(exercise / "captures/success"),
            "--output", str(portable), "--verify-existing",
        ],
        "learning-build": [
            "python", "scripts/render_streaming_learning.py", str(exercise / "captures/cancel-after-launch"),
            "--output", str(learning),
        ],
        "learning-verify": [
            "python", "scripts/render_streaming_learning.py", str(exercise / "captures/cancel-after-launch"),
            "--output", str(learning), "--verify-existing",
        ],
    }
    if tuple(commands) != tuple(parameters)[1:]:  # pragma: no cover - local construction invariant
        raise RuntimeError("V0.2 policy phase argv and semantic phase order drifted")
    return commands


__all__ = [
    "APPROVAL_PATH",
    "APPROVAL_SHA256",
    "CANDIDATE_POLICY_DIGESTS",
    "CANDIDATE_POLICY_FAMILY",
    "CAPTURE_EVENT_CATALOG",
    "CAPTURE_MANIFEST_SCHEMA_VERSION",
    "INPUT_BYTES_SHA256",
    "INPUT_MANIFEST_DIGEST",
    "INPUT_PATH",
    "LEGACY_POLICY_DIGESTS",
    "LEGACY_POLICY_FAMILY",
    "METRIC_REGISTRY_DIGEST",
    "POLICY_CPU_PROFILE_IDS",
    "POLICY_GPU_DEFERRED_REASON",
    "POLICY_GPU_PROFILE_ID",
    "POLICY_PROFILE_RUNTIME_IDS",
    "POLICY_QUALIFICATION_CONTRACT_REVISION",
    "POLICY_QUALIFICATION_SCHEMA_VERSION",
    "PROTOCOL_ID",
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256",
    "REQUIRED_POLICY_CAPTURE_IDS",
    "REQUIRED_RESULT_AXES",
    "policy_phase_commands",
    "policy_phase_parameters",
]

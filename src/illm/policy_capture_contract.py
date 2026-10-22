"""Pure-metadata validation for the V0.2 policy-aware capture contract.

This module does not construct runtime objects, inspect the workspace, or read
the current process environment.  It validates only supplied facts and returns
detached, canonical JSON-compatible mappings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, NoReturn, cast

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.numerical_policy import NumericalPolicyError, parse_policy
from illm.sampling.reference import SamplingError, SamplingSpec


PROTOCOL_SHA256 = "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"

_DIGEST = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_PROFILES = frozenset({"off", "metrics", "request", "ops"})
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
_CONTIGUOUS_LAYOUT_ID = "qwen2-contiguous-layer-major-v1"
_NONE_LAYOUT_ID = "none-reference-full-context-v1"

_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "implementation_source_scope",
        "implementation_source_digest",
        "environment",
        "environment_digest",
        "toolchain",
        "protocol_sha256",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "backend",
        "python_version",
        "platform",
        "machine",
        "torch_version",
        "cuda_arithmetic",
        "cuda_execution_authorized",
    }
)
_UNBOUND_TOOLCHAIN_FIELDS = frozenset({"binding", "manifest_digest"})
_BOUND_TOOLCHAIN_FIELDS = frozenset({"binding", "manifest_digest", "environment_id", "lock_digest"})
_MEMORY_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "model_spec_id",
        "weights_digest",
        "weight_manifest_digest",
        "runtime_dtype",
        "device",
        "kv_mode",
        "kv_layout_id",
        "capacity_tokens",
        "memory_budget_bytes",
        "weights_bytes",
        "kv_bytes",
        "activation_bytes",
        "workspace_bytes",
        "io_bytes",
        "telemetry_bytes",
        "headroom_bytes",
        "required_bytes",
        "numerical_policy",
        "numerical_policy_digest",
        "compute_dtype",
        "sidecar_bytes",
        "prefix_scratch_bytes",
        "execution_binding",
        "execution_binding_digest",
        "evidence_kind",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "engine_version",
        "git_commit",
        "git_dirty",
        "toolchain_digest",
        "toolchain_bound",
        "device",
        "dtype",
        "model_spec_id",
        "weights_digest",
        "load_manifest_digest",
        "tokenizer_digest",
        "kv_mode",
        "memory_plan_evidence_kind",
    }
)
_SAMPLING_FIELDS = frozenset(SamplingSpec().to_dict())


def _fail(message: str, *, field: str | None = None) -> NoReturn:
    details = None if field is None else {"field": field}
    raise IllmError(ErrorCode.CAPTURE_CORRUPT, message, details)


def _unsupported(message: str, *, field: str | None = None) -> NoReturn:
    details = None if field is None else {"field": field}
    raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, message, details)


def _clone_json(value: Any, *, path: str) -> Any:
    """Detach finite JSON data while rejecting unsupported Python values."""

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail("policy capture metadata contains a non-finite number", field=path)
        return value
    if isinstance(value, Mapping):
        names = list(value)
        if any(type(name) is not str or not name for name in names):
            _fail("policy capture metadata keys must be nonempty strings", field=path)
        if len(names) != len(set(names)):
            _fail("policy capture metadata contains a duplicate field", field=path)
        return {name: _clone_json(value[name], path=f"{path}.{name}") for name in cast(list[str], names)}
    if type(value) in {list, tuple}:
        return [_clone_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    _fail("policy capture metadata is not finite JSON data", field=path)


def _object(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object", field=name)
    result = _clone_json(value, path=name)
    if not isinstance(result, dict):  # pragma: no cover - guarded above
        _fail(f"{name} must be an object", field=name)
    return cast(dict[str, Any], result)


def _closed(value: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise IllmError(
            ErrorCode.CAPTURE_CORRUPT,
            f"{name} fields do not match the closed contract",
            {"missing": missing, "unknown": unknown},
        )


def _digest(value: Any, *, field: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _fail("policy capture digest is invalid", field=field)
    return value


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value:
        _fail("policy capture string is invalid", field=field)
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail("policy capture integer is invalid", field=field)
    return value


def _schema(value: Any, expected: int, *, field: str) -> None:
    if type(value) is not int:
        _fail("schema version must be an integer", field=field)
    if value != expected:
        _unsupported(f"unsupported schema version {value!r}", field=field)


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def validate_execution_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a closed CPU ``ExecutionBindingV1`` without observing this host."""

    result = _object(value, name="execution_binding")
    _closed(result, _BINDING_FIELDS, name="execution_binding")
    _schema(result["schema_version"], 1, field="execution_binding.schema_version")
    if result["implementation_source_scope"] != "project-root-v1":
        _unsupported(
            "execution binding source scope is unsupported",
            field="execution_binding.implementation_source_scope",
        )
    _digest(result["implementation_source_digest"], field="execution_binding.implementation_source_digest")

    environment = result["environment"]
    if not isinstance(environment, dict):
        _fail("execution binding environment must be an object", field="execution_binding.environment")
    _closed(environment, _ENVIRONMENT_FIELDS, name="execution_binding.environment")
    if environment["backend"] != "cpu":
        _unsupported("execution binding backend is unsupported", field="execution_binding.environment.backend")
    for name in ("python_version", "platform", "machine", "torch_version"):
        _text(environment[name], field=f"execution_binding.environment.{name}")
    if environment["cuda_arithmetic"] != "not_applicable":
        _fail(
            "CPU execution binding must declare CUDA arithmetic not applicable",
            field="execution_binding.environment.cuda_arithmetic",
        )
    if environment["cuda_execution_authorized"] is not False:
        _fail(
            "CPU execution binding cannot authorize CUDA",
            field="execution_binding.environment.cuda_execution_authorized",
        )
    _digest(result["environment_digest"], field="execution_binding.environment_digest")
    if result["environment_digest"] != sha256_json(environment):
        _fail("execution binding environment digest mismatch", field="execution_binding.environment_digest")

    toolchain = result["toolchain"]
    if not isinstance(toolchain, dict):
        _fail("execution binding toolchain must be an object", field="execution_binding.toolchain")
    binding_kind = toolchain.get("binding")
    if binding_kind == "unbound-development":
        _closed(toolchain, _UNBOUND_TOOLCHAIN_FIELDS, name="execution_binding.toolchain")
        if toolchain["manifest_digest"] is not None:
            _fail(
                "unbound execution binding cannot name a manifest",
                field="execution_binding.toolchain.manifest_digest",
            )
    elif binding_kind == "source-bound-manifest":
        _closed(toolchain, _BOUND_TOOLCHAIN_FIELDS, name="execution_binding.toolchain")
        _digest(toolchain["manifest_digest"], field="execution_binding.toolchain.manifest_digest")
        _text(toolchain["environment_id"], field="execution_binding.toolchain.environment_id")
        _digest(toolchain["lock_digest"], field="execution_binding.toolchain.lock_digest")
    else:
        _unsupported("execution binding toolchain mode is unsupported", field="execution_binding.toolchain.binding")

    if result["protocol_sha256"] != PROTOCOL_SHA256:
        _unsupported("execution binding protocol identity is unsupported", field="execution_binding.protocol_sha256")
    return result


def _parse_registered_policy(value: Any, *, field: str) -> tuple[dict[str, Any], str, str, bool]:
    if not isinstance(value, Mapping):
        _fail("numerical policy must be an object", field=field)
    policy_value = _object(cast(Mapping[str, Any], value), name=field)
    try:
        policy = parse_policy(policy_value)
    except NumericalPolicyError as exc:
        if exc.code.startswith("UNSUPPORTED_") or exc.code == "INTERNAL_POLICY_REGISTRY_ERROR":
            _unsupported(str(exc), field=field)
        _fail(str(exc), field=field)
    return cast(dict[str, Any], policy.to_dict()), policy.digest, policy.storage_dtype, policy.is_candidate


def validate_policy_memory_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact emitted ``PolicyMemoryPlan`` schema2 mapping."""

    result = _object(value, name="memory_plan")
    _closed(result, _MEMORY_PLAN_FIELDS, name="memory_plan")
    _schema(result["schema_version"], 2, field="memory_plan.schema_version")

    for name in ("model_spec_id", "weights_digest", "weight_manifest_digest"):
        _digest(result[name], field=f"memory_plan.{name}")
    runtime_dtype = result["runtime_dtype"]
    if type(runtime_dtype) is not str or runtime_dtype not in _DTYPE_BYTES:
        _unsupported("MemoryPlan runtime dtype is unsupported", field="memory_plan.runtime_dtype")
    if result["device"] != "cpu":
        _unsupported("policy MemoryPlan only supports the CPU implementation", field="memory_plan.device")

    kv_mode = result["kv_mode"]
    if type(kv_mode) is not str or kv_mode not in {"none", "contiguous"}:
        _unsupported("MemoryPlan KV mode is unsupported", field="memory_plan.kv_mode")
    if type(result["kv_layout_id"]) is not str:
        _fail("MemoryPlan KV layout identity is invalid", field="memory_plan.kv_layout_id")

    _integer(result["capacity_tokens"], field="memory_plan.capacity_tokens", minimum=1)
    budget = _integer(result["memory_budget_bytes"], field="memory_plan.memory_budget_bytes", minimum=1)
    components: dict[str, int] = {}
    for name in (
        "weights_bytes",
        "kv_bytes",
        "activation_bytes",
        "workspace_bytes",
        "io_bytes",
        "telemetry_bytes",
        "headroom_bytes",
        "required_bytes",
        "sidecar_bytes",
        "prefix_scratch_bytes",
    ):
        minimum = 1 if name in {"weights_bytes", "required_bytes"} else 0
        components[name] = _integer(result[name], field=f"memory_plan.{name}", minimum=minimum)

    if kv_mode == "none":
        if result["kv_layout_id"] != _NONE_LAYOUT_ID or components["kv_bytes"] != 0:
            _fail("no-KV MemoryPlan layout/bytes disagree", field="memory_plan.kv_layout_id")
    elif result["kv_layout_id"] != _CONTIGUOUS_LAYOUT_ID or components["kv_bytes"] <= 0:
        _fail("contiguous MemoryPlan layout/bytes disagree", field="memory_plan.kv_layout_id")

    policy_value, policy_digest, storage_dtype, is_candidate = _parse_registered_policy(
        result["numerical_policy"], field="memory_plan.numerical_policy"
    )
    _digest(result["numerical_policy_digest"], field="memory_plan.numerical_policy_digest")
    if result["numerical_policy_digest"] != policy_digest:
        _fail("MemoryPlan numerical policy digest mismatch", field="memory_plan.numerical_policy_digest")
    if runtime_dtype != storage_dtype:
        _fail("MemoryPlan dtype differs from numerical policy", field="memory_plan.runtime_dtype")
    try:
        parse_policy(policy_value).validate_kv_mode(kv_mode)
    except NumericalPolicyError as exc:
        _unsupported(str(exc), field="memory_plan.kv_mode")

    expected_compute_dtype = "fp32" if is_candidate else storage_dtype
    if result["compute_dtype"] != expected_compute_dtype:
        _fail("MemoryPlan compute dtype differs from numerical policy", field="memory_plan.compute_dtype")

    if is_candidate:
        dtype_bytes = _DTYPE_BYTES[runtime_dtype]
        if components["weights_bytes"] % dtype_bytes != 0:
            _fail("MemoryPlan weight bytes cannot define complete FP32 sidecars", field="memory_plan.weights_bytes")
        expected_sidecars = components["weights_bytes"] // dtype_bytes * 4
        if components["sidecar_bytes"] != expected_sidecars:
            _fail("MemoryPlan FP32 sidecar byte count is invalid", field="memory_plan.sidecar_bytes")
        if components["prefix_scratch_bytes"] <= 0 or components["prefix_scratch_bytes"] % 256 != 0:
            _fail("candidate MemoryPlan prefix scratch is invalid", field="memory_plan.prefix_scratch_bytes")
    elif components["sidecar_bytes"] != 0 or components["prefix_scratch_bytes"] != 0:
        _fail("legacy MemoryPlan cannot reserve candidate sidecars", field="memory_plan.sidecar_bytes")

    subtotal = sum(
        components[name]
        for name in (
            "weights_bytes",
            "kv_bytes",
            "activation_bytes",
            "workspace_bytes",
            "io_bytes",
            "telemetry_bytes",
            "sidecar_bytes",
            "prefix_scratch_bytes",
        )
    )
    expected_headroom = _align_up((subtotal + 9) // 10)
    if components["headroom_bytes"] != expected_headroom:
        _fail("MemoryPlan headroom differs from the fixed conservative formula", field="memory_plan.headroom_bytes")
    if components["required_bytes"] != subtotal + components["headroom_bytes"]:
        _fail("MemoryPlan total differs from component sum", field="memory_plan.required_bytes")
    if components["required_bytes"] > budget:
        _fail("accepted policy capture MemoryPlan exceeds its fixed budget", field="memory_plan.memory_budget_bytes")

    binding = validate_execution_binding(cast(Mapping[str, Any], result["execution_binding"]))
    binding_digest = _digest(result["execution_binding_digest"], field="memory_plan.execution_binding_digest")
    if binding_digest != sha256_json(binding):
        _fail("MemoryPlan execution binding digest mismatch", field="memory_plan.execution_binding_digest")
    if result["evidence_kind"] != "static_conservative_estimate":
        _unsupported("MemoryPlan evidence kind is unsupported", field="memory_plan.evidence_kind")

    result["numerical_policy"] = policy_value
    result["execution_binding"] = binding
    return result


def _validate_sampling_spec(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    result = _object(value, name="sampling_spec")
    _closed(result, _SAMPLING_FIELDS, name="sampling_spec")
    try:
        sampling = SamplingSpec.from_dict(result)
    except SamplingError as exc:
        if exc.code == "UNSUPPORTED_SAMPLING_PARAMETER":
            _unsupported(str(exc), field="sampling_spec")
        _fail(str(exc), field="sampling_spec")
    except TypeError as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "SamplingSpec fields have invalid types") from exc
    normalized = sampling.to_dict()
    return normalized, sampling.digest


def _validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _object(value, name="identity")
    _closed(result, _IDENTITY_FIELDS, name="identity")
    for name in ("engine_version", "git_commit"):
        _text(result[name], field=f"identity.{name}")
    if type(result["git_dirty"]) is not bool:
        _fail("identity git_dirty must be boolean", field="identity.git_dirty")
    _digest(result["toolchain_digest"], field="identity.toolchain_digest")
    if type(result["toolchain_bound"]) is not bool:
        _fail("identity toolchain_bound must be boolean", field="identity.toolchain_bound")
    if result["device"] != "cpu":
        _unsupported("policy capture identity only supports CPU", field="identity.device")
    if type(result["dtype"]) is not str or result["dtype"] not in _DTYPE_BYTES:
        _unsupported("policy capture identity dtype is unsupported", field="identity.dtype")
    for name in ("model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest"):
        _digest(result[name], field=f"identity.{name}")
    if type(result["kv_mode"]) is not str or result["kv_mode"] not in {"none", "contiguous"}:
        _unsupported("policy capture identity KV mode is unsupported", field="identity.kv_mode")
    if result["memory_plan_evidence_kind"] != "static_conservative_estimate":
        _unsupported(
            "policy capture identity evidence kind is unsupported",
            field="identity.memory_plan_evidence_kind",
        )
    return result


def validate_policy_capture_metadata(
    *,
    identity: Mapping[str, Any],
    sampling_spec: Mapping[str, Any],
    memory_plan: Mapping[str, Any],
    numerical_policy: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
    observation_profile: str,
) -> dict[str, Any]:
    """Normalize and cross-bind all schema3 policy capture metadata facts."""

    normalized_identity = _validate_identity(identity)
    normalized_sampling, sampling_digest = _validate_sampling_spec(sampling_spec)
    normalized_plan = validate_policy_memory_plan(memory_plan)
    normalized_policy, policy_digest, _, _ = _parse_registered_policy(
        numerical_policy, field="numerical_policy"
    )
    normalized_binding = validate_execution_binding(execution_binding)

    if type(observation_profile) is not str:
        _fail("observation profile must be a string", field="observation_profile")
    if observation_profile not in _OBSERVATION_PROFILES:
        _unsupported("observation profile is unsupported", field="observation_profile")

    if (
        normalized_plan["numerical_policy"] != normalized_policy
        or normalized_plan["numerical_policy_digest"] != policy_digest
    ):
        _fail("MemoryPlan and capture numerical policy bindings disagree", field="numerical_policy")
    binding_digest = sha256_json(normalized_binding)
    if (
        normalized_plan["execution_binding"] != normalized_binding
        or normalized_plan["execution_binding_digest"] != binding_digest
    ):
        _fail("MemoryPlan and capture execution bindings disagree", field="execution_binding")

    cross_fields = {
        "device": "device",
        "dtype": "runtime_dtype",
        "model_spec_id": "model_spec_id",
        "weights_digest": "weights_digest",
        "load_manifest_digest": "weight_manifest_digest",
        "kv_mode": "kv_mode",
        "memory_plan_evidence_kind": "evidence_kind",
    }
    for identity_name, plan_name in cross_fields.items():
        if normalized_identity[identity_name] != normalized_plan[plan_name]:
            _fail("identity and MemoryPlan bindings disagree", field=f"identity.{identity_name}")

    toolchain = normalized_binding["toolchain"]
    if toolchain["binding"] == "unbound-development":
        expected_toolchain_digest = sha256_json({"binding": "unbound-development"})
        expected_bound = False
    else:
        expected_toolchain_digest = cast(str, toolchain["manifest_digest"])
        expected_bound = True
    if (
        normalized_identity["toolchain_digest"] != expected_toolchain_digest
        or normalized_identity["toolchain_bound"] is not expected_bound
    ):
        _fail("identity and execution binding toolchain facts disagree", field="identity.toolchain_digest")

    return {
        "identity": normalized_identity,
        "sampling_spec": normalized_sampling,
        "sampling_spec_digest": sampling_digest,
        "memory_plan": normalized_plan,
        "memory_plan_digest": sha256_json(normalized_plan),
        "numerical_policy": normalized_policy,
        "numerical_policy_digest": policy_digest,
        "execution_binding": normalized_binding,
        "execution_binding_digest": binding_digest,
        "protocol_sha256": PROTOCOL_SHA256,
        "observation_profile": observation_profile,
    }


__all__ = [
    "PROTOCOL_SHA256",
    "validate_execution_binding",
    "validate_policy_capture_metadata",
    "validate_policy_memory_plan",
]

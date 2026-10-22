"""Pure-stdlib contract checks for policy-aware schema3 capture metadata."""

from __future__ import annotations

import copy
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.policy_capture_contract import (
    PROTOCOL_SHA256,
    validate_execution_binding,
    validate_policy_capture_metadata,
    validate_policy_memory_plan,
)
from illm.sampling.reference import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]


def _binding(*, bound: bool = False) -> dict[str, Any]:
    environment = {
        "backend": "cpu",
        "python_version": "3.12.7",
        "platform": "synthetic-test-platform",
        "machine": "x86_64",
        "torch_version": "2.7.1",
        "cuda_arithmetic": "not_applicable",
        "cuda_execution_authorized": False,
    }
    toolchain: dict[str, Any]
    if bound:
        toolchain = {
            "binding": "source-bound-manifest",
            "manifest_digest": "9" * 64,
            "environment_id": "cpu-test-environment",
            "lock_digest": "8" * 64,
        }
    else:
        toolchain = {"binding": "unbound-development", "manifest_digest": None}
    return {
        "schema_version": 1,
        "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": "7" * 64,
        "environment": environment,
        "environment_digest": sha256_json(environment),
        "toolchain": toolchain,
        "protocol_sha256": PROTOCOL_SHA256,
    }


def _plan(
    *,
    dtype: str = "fp32",
    kv_mode: str = "contiguous",
    legacy: bool = False,
    bound: bool = False,
) -> dict[str, Any]:
    policy = legacy_policy(dtype) if legacy else candidate_policy(dtype)
    binding = _binding(bound=bound)
    weights = 4096
    kv_bytes = 0 if kv_mode == "none" else 8192
    sidecars = 0 if legacy else weights // {"fp32": 4, "fp16": 2, "bf16": 2}[dtype] * 4
    prefix = 0 if legacy else 256
    components = {
        "weights_bytes": weights,
        "kv_bytes": kv_bytes,
        "activation_bytes": 16384,
        "workspace_bytes": 32768,
        "io_bytes": 4096,
        "telemetry_bytes": 1024,
        "sidecar_bytes": sidecars,
        "prefix_scratch_bytes": prefix,
    }
    subtotal = sum(components.values())
    headroom = ((subtotal + 9) // 10 + 255) // 256 * 256
    required = subtotal + headroom
    return {
        "schema_version": 2,
        "model_spec_id": "1" * 64,
        "weights_digest": "2" * 64,
        "weight_manifest_digest": "3" * 64,
        "runtime_dtype": dtype,
        "device": "cpu",
        "kv_mode": kv_mode,
        "kv_layout_id": (
            "none-reference-full-context-v1" if kv_mode == "none" else "qwen2-contiguous-layer-major-v1"
        ),
        "capacity_tokens": 16,
        "memory_budget_bytes": required + 4096,
        **components,
        "headroom_bytes": headroom,
        "required_bytes": required,
        "numerical_policy": policy.to_dict(),
        "numerical_policy_digest": policy.digest,
        "compute_dtype": dtype if legacy else "fp32",
        "execution_binding": binding,
        "execution_binding_digest": sha256_json(binding),
        "evidence_kind": "static_conservative_estimate",
    }


def _identity(plan: Mapping[str, Any], *, bound: bool = False) -> dict[str, Any]:
    return {
        "engine_version": "0.2.0.dev0",
        "git_commit": "unversioned",
        "git_dirty": True,
        "toolchain_digest": (
            "9" * 64 if bound else sha256_json({"binding": "unbound-development"})
        ),
        "toolchain_bound": bound,
        "device": plan["device"],
        "dtype": plan["runtime_dtype"],
        "model_spec_id": plan["model_spec_id"],
        "weights_digest": plan["weights_digest"],
        "load_manifest_digest": plan["weight_manifest_digest"],
        "tokenizer_digest": "4" * 64,
        "kv_mode": plan["kv_mode"],
        "memory_plan_evidence_kind": plan["evidence_kind"],
    }


def _inputs(
    *,
    dtype: str = "fp32",
    kv_mode: str = "contiguous",
    legacy: bool = False,
    bound: bool = False,
) -> dict[str, Any]:
    plan = _plan(dtype=dtype, kv_mode=kv_mode, legacy=legacy, bound=bound)
    sampling = SamplingSpec(
        mode="sample",
        temperature=0.75,
        top_k=7,
        top_p=0.9,
        actual_seed=11,
    ).to_dict()
    return {
        "identity": _identity(plan, bound=bound),
        "sampling_spec": sampling,
        "memory_plan": plan,
        "numerical_policy": copy.deepcopy(plan["numerical_policy"]),
        "execution_binding": copy.deepcopy(plan["execution_binding"]),
        "observation_profile": "request",
    }


def _assert_error(code: ErrorCode, call: Any) -> None:
    with pytest.raises(IllmError) as caught:
        call()
    assert caught.value.code is code


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("kv_mode", ["none", "contiguous"])
@pytest.mark.parametrize("profile", ["off", "metrics", "request", "ops"])
def test_candidate_metadata_normalizes_all_registered_cpu_shapes(
    dtype: str,
    kv_mode: str,
    profile: str,
) -> None:
    values = _inputs(dtype=dtype, kv_mode=kv_mode)
    values["observation_profile"] = profile

    result = validate_policy_capture_metadata(**values)

    assert set(result) == {
        "identity",
        "sampling_spec",
        "sampling_spec_digest",
        "memory_plan",
        "memory_plan_digest",
        "numerical_policy",
        "numerical_policy_digest",
        "execution_binding",
        "execution_binding_digest",
        "protocol_sha256",
        "observation_profile",
    }
    assert result["sampling_spec_digest"] == sha256_json(result["sampling_spec"])
    assert result["memory_plan_digest"] == sha256_json(result["memory_plan"])
    assert result["numerical_policy_digest"] == sha256_json(result["numerical_policy"])
    assert result["execution_binding_digest"] == sha256_json(result["execution_binding"])
    assert result["protocol_sha256"] == PROTOCOL_SHA256
    assert result["observation_profile"] == profile


def test_bound_toolchain_and_explicit_legacy_none_are_valid() -> None:
    bound = _inputs(bound=True)
    assert validate_policy_capture_metadata(**bound)["identity"]["toolchain_bound"] is True

    legacy = _inputs(kv_mode="none", legacy=True)
    result = validate_policy_capture_metadata(**legacy)
    assert result["memory_plan"]["sidecar_bytes"] == 0
    assert result["memory_plan"]["prefix_scratch_bytes"] == 0
    assert result["memory_plan"]["compute_dtype"] == "fp32"


def test_normalized_metadata_does_not_alias_caller_mappings() -> None:
    values = _inputs()
    result = validate_policy_capture_metadata(**values)

    values["identity"]["engine_version"] = "mutated"
    values["sampling_spec"]["actual_seed"] = 99
    values["memory_plan"]["capacity_tokens"] = 99
    values["numerical_policy"]["flags"]["cpu_threads"] = 99
    values["execution_binding"]["environment"]["machine"] = "mutated"

    assert result["identity"]["engine_version"] == "0.2.0.dev0"
    assert result["sampling_spec"]["actual_seed"] == 11
    assert result["memory_plan"]["capacity_tokens"] == 16
    assert result["numerical_policy"]["flags"]["cpu_threads"] == 1
    assert result["execution_binding"]["environment"]["machine"] == "x86_64"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update(schema_version=True), ErrorCode.CAPTURE_CORRUPT),
        (lambda value: value.update(schema_version=2), ErrorCode.UNSUPPORTED_SCHEMA),
        (lambda value: value.update(protocol_sha256="0" * 64), ErrorCode.UNSUPPORTED_SCHEMA),
        (lambda value: value.update(extra="forbidden"), ErrorCode.CAPTURE_CORRUPT),
        (
            lambda value: value["environment"].update(cuda_execution_authorized=True),
            ErrorCode.CAPTURE_CORRUPT,
        ),
        (
            lambda value: value.update(environment_digest="0" * 64),
            ErrorCode.CAPTURE_CORRUPT,
        ),
        (
            lambda value: value["toolchain"].update(binding="future-binding"),
            ErrorCode.UNSUPPORTED_SCHEMA,
        ),
    ],
)
def test_execution_binding_fails_closed(mutate: Any, code: ErrorCode) -> None:
    value = _binding()
    mutate(value)
    _assert_error(code, lambda: validate_execution_binding(value))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(capacity_tokens=True),
        lambda value: value.update(required_bytes=value["required_bytes"] + 1),
        lambda value: value.update(headroom_bytes=value["headroom_bytes"] + 256),
        lambda value: value.update(memory_budget_bytes=value["required_bytes"] - 1),
        lambda value: value.update(sidecar_bytes=value["sidecar_bytes"] + 4),
        lambda value: value.update(prefix_scratch_bytes=1),
        lambda value: value.update(kv_bytes=0),
        lambda value: value.update(execution_binding_digest="0" * 64),
        lambda value: value.update(numerical_policy_digest="0" * 64),
        lambda value: value.update(extra="forbidden"),
    ],
)
def test_policy_memory_plan_rejects_false_byte_or_identity_facts(mutation: Any) -> None:
    plan = _plan()
    mutation(plan)
    _assert_error(ErrorCode.CAPTURE_CORRUPT, lambda: validate_policy_memory_plan(plan))


def test_policy_memory_plan_rejects_unsupported_schema_and_legacy_contiguous() -> None:
    plan = _plan()
    plan["schema_version"] = 1
    _assert_error(ErrorCode.UNSUPPORTED_SCHEMA, lambda: validate_policy_memory_plan(plan))

    legacy = _plan(kv_mode="contiguous", legacy=True)
    _assert_error(ErrorCode.UNSUPPORTED_SCHEMA, lambda: validate_policy_memory_plan(legacy))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("rng_algorithm", "global-default", ErrorCode.UNSUPPORTED_SCHEMA),
        ("rng_contract_version", 2, ErrorCode.UNSUPPORTED_SCHEMA),
        ("actual_seed", True, ErrorCode.CAPTURE_CORRUPT),
        ("temperature", float("nan"), ErrorCode.CAPTURE_CORRUPT),
    ],
)
def test_sampling_contract_rejects_rng_seed_and_nonfinite_drift(
    field: str,
    value: Any,
    code: ErrorCode,
) -> None:
    values = _inputs()
    values["sampling_spec"][field] = value
    _assert_error(code, lambda: validate_policy_capture_metadata(**values))


def test_sampling_contract_is_closed_but_does_not_apply_generic_privacy_key_bans() -> None:
    values = _inputs()
    values["sampling_spec"]["unknown_token_ids"] = []
    _assert_error(ErrorCode.CAPTURE_CORRUPT, lambda: validate_policy_capture_metadata(**values))

    values = _inputs()
    values["identity"]["prompt"] = "private"
    _assert_error(ErrorCode.CAPTURE_CORRUPT, lambda: validate_policy_capture_metadata(**values))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "bf16"),
        ("device", "cuda:0"),
        ("model_spec_id", "a" * 64),
        ("weights_digest", "b" * 64),
        ("load_manifest_digest", "c" * 64),
        ("kv_mode", "none"),
        ("memory_plan_evidence_kind", "measured"),
        ("toolchain_bound", True),
    ],
)
def test_identity_and_plan_or_toolchain_cross_bindings_are_exact(field: str, value: Any) -> None:
    values = _inputs()
    values["identity"][field] = value
    code = (
        ErrorCode.UNSUPPORTED_SCHEMA
        if field in {"device", "memory_plan_evidence_kind"}
        else ErrorCode.CAPTURE_CORRUPT
    )
    _assert_error(code, lambda: validate_policy_capture_metadata(**values))


def test_top_level_policy_and_binding_must_equal_memory_plan() -> None:
    values = _inputs()
    values["numerical_policy"] = candidate_policy("bf16").to_dict()
    _assert_error(ErrorCode.CAPTURE_CORRUPT, lambda: validate_policy_capture_metadata(**values))

    values = _inputs()
    values["execution_binding"]["implementation_source_digest"] = "5" * 64
    _assert_error(ErrorCode.CAPTURE_CORRUPT, lambda: validate_policy_capture_metadata(**values))


@pytest.mark.parametrize("profile", ["", "trace", 1, True, None])
def test_observation_profile_is_closed(profile: Any) -> None:
    values = _inputs()
    values["observation_profile"] = profile
    expected = ErrorCode.CAPTURE_CORRUPT if type(profile) is not str else ErrorCode.UNSUPPORTED_SCHEMA
    _assert_error(expected, lambda: validate_policy_capture_metadata(**values))


class _DuplicateBinding(Mapping[str, Any]):
    def __init__(self, value: Mapping[str, Any]) -> None:
        self._value = value

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        yield "schema_version"
        yield "schema_version"
        yield from (key for key in self._value if key != "schema_version")

    def __len__(self) -> int:
        return len(self._value) + 1


def test_mapping_duplicate_iteration_is_rejected() -> None:
    _assert_error(
        ErrorCode.CAPTURE_CORRUPT,
        lambda: validate_execution_binding(_DuplicateBinding(_binding())),
    )


def test_contract_import_is_portable_and_does_not_import_tensor_libraries(tmp_path: Path) -> None:
    source = ROOT / "src"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "import illm.policy_capture_contract as contract\n"
        "assert contract.PROTOCOL_SHA256\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'safetensors' not in sys.modules\n"
        "assert 'tokenizers' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

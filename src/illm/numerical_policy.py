"""Tensor-free numerical execution policy registry for the V0.2 protocol.

The registry is deliberately closed: parsing validates an exact, registered
descriptor rather than accepting a configuration that merely looks similar.
Policy identity is the SHA-256 digest of canonical JSON bytes and is separate
from the runtime KV mode selected for a request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_bytes


CANDIDATE_FAMILY = "illm-rowwise-fp32-contraction"
LEGACY_FAMILY = "illm-v01-full-context"
STORAGE_DTYPES = ("bf16", "fp16", "fp32")

CANDIDATE_POLICY_DIGESTS: Mapping[str, str] = {
    "bf16": "c90519d8e1de749be6727cbb898e42b78a189d2f4c813e224dc9b343ce762497",
    "fp16": "b3a7f22a649b70d24e1014089ac5d5f8856cb5cb4fb7bdac9f469685cb171cb4",
    "fp32": "4e15f0d894637ef4a9533ee4c31adcc18de0b7fddefde98efe2798b9a2a29553",
}

_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "family",
        "version",
        "storage_dtype",
        "weights",
        "linear",
        "rms_norm",
        "rope",
        "gqa",
        "qk",
        "softmax",
        "pv",
        "pointwise",
        "layout",
        "full_schedule",
        "cached_schedule",
        "flags",
    }
)
_FLAG_FIELDS = frozenset(
    {
        "autocast",
        "tf32_matmul",
        "tf32_cudnn",
        "float32_matmul_precision",
        "fp16_reduced_precision_reduction",
        "bf16_reduced_precision_reduction",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cublas_workspace_config",
        "cpu_threads",
        "compile",
    }
)
_LEGACY_FIELDS = frozenset(
    {
        "schema_version",
        "family",
        "version",
        "storage_dtype",
        "authority_tag",
        "authority_commit",
        "reference_path",
        "reference_sha256",
        "execution",
        "legal_kv_modes",
    }
)


class NumericalPolicyError(ValueError):
    """A policy descriptor is malformed, unregistered, or used illegally."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


def _fail(code: str, message: str, *, field: str | None = None) -> NoReturn:
    raise NumericalPolicyError(code, message, field=field)


def _candidate_payload(storage_dtype: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": CANDIDATE_FAMILY,
        "version": 1,
        "storage_dtype": storage_dtype,
        "weights": "verified-source-to-storage-once;fp32-compute-sidecar-once;ties-shared",
        "linear": "position-ascending;F.linear-fp32-[1,K]-[N,K]-no-bias;fp32-bias-add;cast-storage",
        "rms_norm": "fp32-square-mean-last-axis-add-model-eps-rsqrt-multiply;cast-storage;storage-weight-multiply",
        "rope": "per-position-fp32-inv-freq-and-trig;cast-storage;rotate-halves;storage-mul-mul-add",
        "gqa": "query-head-h-uses-kv-head-floor(h/groups)",
        "qk": "per-position-per-head-fp32-[1,D]-[D,prefix];fp32-scale;cast-storage",
        "softmax": "per-position-per-head-legal-prefix-only;torch-fp32-last-axis;cast-storage",
        "pv": "per-position-per-head-fp32-[1,prefix]-[prefix,D];cast-storage",
        "pointwise": "rowwise-unfused-storage-residual-silu-product;no-inplace",
        "layout": "contiguous-row-inputs-and-weight-sidecars;contiguous-prefix-materialization;heads-ascending",
        "full_schedule": "layer-major;positions-ascending;recompute-all-prefix;all-position-lm-head",
        "cached_schedule": "layer-major;new-positions-ascending;reuse-committed-post-rope-K-and-V;stage-before-commit",
        "flags": {
            "autocast": False,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "float32_matmul_precision": "highest",
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cublas_workspace_config": ":16:8",
            "cpu_threads": 1,
            "compile": False,
        },
    }


def _legacy_payload(storage_dtype: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "family": LEGACY_FAMILY,
        "version": 1,
        "storage_dtype": storage_dtype,
        "authority_tag": "v0.1.0",
        "authority_commit": "8df86f2ab0b0f3e0c67bde68f65a9aa7f4adfa02",
        "reference_path": "src/illm/model/reference.py",
        "reference_sha256": "19fb396156e5739a3ddd6783a2fbbc610528fff8ed099abe6e0a17a5c63b44bf",
        "execution": "unmodified-reference-forward;adr005-flags;eager-full-context",
        "legal_kv_modes": ["none"],
    }


def _validate_dtype(value: object) -> str:
    if type(value) is not str or value not in STORAGE_DTYPES:
        _fail(
            "UNSUPPORTED_NUMERICAL_POLICY",
            "storage_dtype must be one of bf16, fp16, or fp32",
            field="storage_dtype",
        )
    return value


def _field_names(value: Mapping[object, object]) -> list[str]:
    keys = list(value.keys())
    if any(type(key) is not str for key in keys):
        _fail("INVALID_NUMERICAL_POLICY", "policy field names must be strings")
    names = cast(list[str], keys)
    if len(names) != len(set(names)):
        _fail("INVALID_NUMERICAL_POLICY", "duplicate policy field")
    return names


def _closed_fields(actual: list[str], expected: frozenset[str], family: str) -> None:
    names = set(actual)
    missing = sorted(expected - names)
    unknown = sorted(names - expected)
    if missing or unknown:
        _fail(
            "UNSUPPORTED_NUMERICAL_POLICY",
            f"{family} descriptor has missing or unknown fields: missing={missing}, unknown={unknown}",
        )


@dataclass(frozen=True, slots=True)
class NumericalExecutionPolicyV1:
    """Immutable, registered numerical execution policy descriptor."""

    _canonical_bytes: bytes
    family: str
    storage_dtype: str
    digest: str
    is_candidate: bool
    _legal_kv_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        if sha256_bytes(self._canonical_bytes) != self.digest:
            _fail("INVALID_NUMERICAL_POLICY", "policy digest does not match canonical bytes")
        try:
            value = json.loads(self._canonical_bytes.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:  # pragma: no cover - factories make this unreachable
            raise NumericalPolicyError("INVALID_NUMERICAL_POLICY", "invalid canonical policy bytes") from exc
        if (
            not isinstance(value, dict)
            or value.get("family") != self.family
            or value.get("storage_dtype") != self.storage_dtype
            or self.is_candidate != (self.family == CANDIDATE_FAMILY)
        ):
            _fail("INVALID_NUMERICAL_POLICY", "policy metadata disagrees with canonical bytes")
        storage_dtype = _validate_dtype(self.storage_dtype)
        expected_modes: tuple[str, ...]
        if self.family == CANDIDATE_FAMILY:
            expected = _candidate_payload(storage_dtype)
            expected_digest = CANDIDATE_POLICY_DIGESTS[storage_dtype]
            expected_modes = ("none", "contiguous")
        elif self.family == LEGACY_FAMILY:
            expected = _legacy_payload(storage_dtype)
            expected_digest = sha256_bytes(canonical_json_bytes(expected))
            expected_modes = ("none",)
        else:
            _fail("UNSUPPORTED_NUMERICAL_POLICY", "policy family is not registered")
        if (
            self._canonical_bytes != canonical_json_bytes(expected)
            or self.digest != expected_digest
            or self._legal_kv_modes != expected_modes
        ):
            _fail("UNSUPPORTED_NUMERICAL_POLICY", "policy instance is not registered")

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-compatible copy of the descriptor."""

        value = json.loads(self._canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - guarded in __post_init__
            _fail("INVALID_NUMERICAL_POLICY", "canonical policy is not an object")
        return cast(dict[str, object], value)

    def validate_kv_mode(self, mode: str) -> None:
        """Reject a KV mode that is not legal for this policy instance."""

        if type(mode) is not str or mode not in self._legal_kv_modes:
            _fail(
                "UNSUPPORTED_NUMERICAL_POLICY",
                f"KV mode {mode!r} is not legal for policy family {self.family}",
                field="kv_mode",
            )


def _make_policy(payload: dict[str, object], *, is_candidate: bool) -> NumericalExecutionPolicyV1:
    encoded = canonical_json_bytes(payload)
    digest = sha256_bytes(encoded)
    family = cast(str, payload["family"])
    storage_dtype = cast(str, payload["storage_dtype"])
    legal_modes: tuple[str, ...] = ("none", "contiguous") if is_candidate else ("none",)
    return NumericalExecutionPolicyV1(
        _canonical_bytes=encoded,
        family=family,
        storage_dtype=storage_dtype,
        digest=digest,
        is_candidate=is_candidate,
        _legal_kv_modes=legal_modes,
    )


def parse_policy(mapping: Mapping[str, object]) -> NumericalExecutionPolicyV1:
    """Parse an exact registered descriptor, failing closed on every mutation."""

    if not isinstance(mapping, Mapping):
        _fail("INVALID_NUMERICAL_POLICY", "policy must be a mapping")
    names = _field_names(cast(Mapping[object, object], mapping))
    family = mapping.get("family")
    if type(family) is not str:
        _fail("INVALID_NUMERICAL_POLICY", "policy family must be a string", field="family")
    storage_dtype = _validate_dtype(mapping.get("storage_dtype"))

    if family == CANDIDATE_FAMILY:
        _closed_fields(names, _CANDIDATE_FIELDS, family)
        flags = mapping.get("flags")
        if not isinstance(flags, Mapping):
            _fail("INVALID_NUMERICAL_POLICY", "candidate flags must be an object", field="flags")
        flag_names = _field_names(cast(Mapping[object, object], flags))
        _closed_fields(flag_names, _FLAG_FIELDS, f"{family}.flags")
        payload = dict(mapping)
        payload["flags"] = dict(cast(Mapping[str, object], flags))
        expected = _candidate_payload(storage_dtype)
        expected_digest = CANDIDATE_POLICY_DIGESTS[storage_dtype]
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            actual_digest = sha256_bytes(canonical_json_bytes(payload))
            _fail(
                "UNSUPPORTED_NUMERICAL_POLICY",
                f"candidate descriptor digest is not registered: {actual_digest}",
            )
        policy = _make_policy(expected, is_candidate=True)
        if policy.digest != expected_digest:
            _fail("INTERNAL_POLICY_REGISTRY_ERROR", "candidate policy registry digest drifted")
        return policy

    if family == LEGACY_FAMILY:
        _closed_fields(names, _LEGACY_FIELDS, family)
        legal_modes = mapping.get("legal_kv_modes")
        if type(legal_modes) is not list or legal_modes != ["none"]:
            _fail(
                "UNSUPPORTED_NUMERICAL_POLICY",
                "legacy policy legal_kv_modes must be exactly ['none']",
                field="legal_kv_modes",
            )
        payload = dict(mapping)
        expected = _legacy_payload(storage_dtype)
        if canonical_json_bytes(payload) != canonical_json_bytes(expected):
            actual_digest = sha256_bytes(canonical_json_bytes(payload))
            _fail(
                "UNSUPPORTED_NUMERICAL_POLICY",
                f"legacy descriptor digest is not registered: {actual_digest}",
            )
        return _make_policy(expected, is_candidate=False)

    _fail("UNSUPPORTED_NUMERICAL_POLICY", f"unknown numerical policy family: {family!r}", field="family")


def parse_policy_json(value: str | bytes) -> NumericalExecutionPolicyV1:
    """Parse raw JSON while preserving duplicate-field and non-finite rejection."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeError as exc:
            raise NumericalPolicyError("INVALID_NUMERICAL_POLICY", "policy JSON is not UTF-8") from exc
    elif isinstance(value, str):
        text = value
    else:
        _fail("INVALID_NUMERICAL_POLICY", "policy JSON must be str or bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail("INVALID_NUMERICAL_POLICY", f"duplicate policy field: {key}", field=key)
            result[key] = item
        return result

    def constant(name: str) -> NoReturn:
        _fail("INVALID_NUMERICAL_POLICY", f"non-finite JSON number: {name}")

    try:
        parsed = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    except NumericalPolicyError:
        raise
    except (TypeError, ValueError) as exc:
        raise NumericalPolicyError("INVALID_NUMERICAL_POLICY", "policy JSON is invalid") from exc
    if not isinstance(parsed, dict):
        _fail("INVALID_NUMERICAL_POLICY", "policy JSON must contain an object")
    return parse_policy(cast(dict[str, object], parsed))


def candidate_policy(dtype: str) -> NumericalExecutionPolicyV1:
    """Return the registered rowwise FP32-contraction candidate for a dtype."""

    storage_dtype = _validate_dtype(dtype)
    return parse_policy(_candidate_payload(storage_dtype))


def legacy_policy(dtype: str) -> NumericalExecutionPolicyV1:
    """Return the explicit frozen V0.1 full-context adapter descriptor."""

    storage_dtype = _validate_dtype(dtype)
    return parse_policy(_legacy_payload(storage_dtype))


__all__ = [
    "CANDIDATE_FAMILY",
    "CANDIDATE_POLICY_DIGESTS",
    "LEGACY_FAMILY",
    "NumericalExecutionPolicyV1",
    "NumericalPolicyError",
    "STORAGE_DTYPES",
    "candidate_policy",
    "legacy_policy",
    "parse_policy",
    "parse_policy_json",
]

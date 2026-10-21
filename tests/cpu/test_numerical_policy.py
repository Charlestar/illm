from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from illm.numerical_policy import (
    CANDIDATE_FAMILY,
    CANDIDATE_POLICY_DIGESTS,
    LEGACY_FAMILY,
    NumericalExecutionPolicyV1,
    NumericalPolicyError,
    candidate_policy,
    legacy_policy,
    parse_policy,
    parse_policy_json,
)


EXPECTED_CANDIDATE_DIGESTS = {
    "bf16": "c90519d8e1de749be6727cbb898e42b78a189d2f4c813e224dc9b343ce762497",
    "fp16": "b3a7f22a649b70d24e1014089ac5d5f8856cb5cb4fb7bdac9f469685cb171cb4",
    "fp32": "4e15f0d894637ef4a9533ee4c31adcc18de0b7fddefde98efe2798b9a2a29553",
}
EXPECTED_LEGACY_DIGESTS = {
    "bf16": "da4f28136447df33f87ed5cb6494212b027b83e681141b96bdeced59888b095f",
    "fp16": "dd24dfd7f032105cf80bc7927dfd64d2812c75d461426e736782c65166d56c6b",
    "fp32": "a1cc7a0e346d39907488e2f0fb15441f2a9eda56a33794a8dcd2568e1a772e4d",
}


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_candidate_policy_matches_frozen_protocol_digest(dtype: str) -> None:
    policy = candidate_policy(dtype)

    assert isinstance(policy, NumericalExecutionPolicyV1)
    assert policy.family == CANDIDATE_FAMILY
    assert policy.storage_dtype == dtype
    assert policy.is_candidate is True
    assert policy.digest == EXPECTED_CANDIDATE_DIGESTS[dtype]
    assert CANDIDATE_POLICY_DIGESTS[dtype] == policy.digest
    assert parse_policy(policy.to_dict()) == policy


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_legacy_policy_is_explicit_frozen_v01_descriptor(dtype: str) -> None:
    policy = legacy_policy(dtype)
    descriptor = policy.to_dict()

    assert policy.family == LEGACY_FAMILY
    assert policy.storage_dtype == dtype
    assert policy.is_candidate is False
    assert policy.digest == EXPECTED_LEGACY_DIGESTS[dtype]
    assert descriptor["authority_tag"] == "v0.1.0"
    assert descriptor["authority_commit"] == "8df86f2ab0b0f3e0c67bde68f65a9aa7f4adfa02"
    assert descriptor["reference_sha256"] == "19fb396156e5739a3ddd6783a2fbbc610528fff8ed099abe6e0a17a5c63b44bf"
    assert descriptor["legal_kv_modes"] == ["none"]
    assert parse_policy(descriptor) == policy


def test_policy_is_frozen_and_to_dict_is_detached() -> None:
    policy = candidate_policy("bf16")
    first = policy.to_dict()
    flags = cast(dict[str, Any], first["flags"])
    flags["autocast"] = True

    assert cast(dict[str, Any], policy.to_dict()["flags"])["autocast"] is False
    with pytest.raises(FrozenInstanceError):
        policy.family = "forged"  # type: ignore[misc]


def test_candidate_and_legacy_kv_modes_are_separate() -> None:
    candidate = candidate_policy("bf16")
    candidate.validate_kv_mode("none")
    candidate.validate_kv_mode("contiguous")

    legacy = legacy_policy("bf16")
    legacy.validate_kv_mode("none")
    with pytest.raises(NumericalPolicyError, match="not legal"):
        legacy.validate_kv_mode("contiguous")
    with pytest.raises(NumericalPolicyError, match="not legal"):
        candidate.validate_kv_mode("paged")
    with pytest.raises(NumericalPolicyError, match="not legal"):
        candidate.validate_kv_mode(cast(Any, True))


@pytest.mark.parametrize("dtype", ["BF16", "float32", "", True, 1, None])
def test_factories_reject_dtype_aliases_and_wrong_types(dtype: object) -> None:
    with pytest.raises(NumericalPolicyError, match="storage_dtype"):
        candidate_policy(cast(Any, dtype))
    with pytest.raises(NumericalPolicyError, match="storage_dtype"):
        legacy_policy(cast(Any, dtype))


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "schema-bool", "flag-missing", "flag-value", "arithmetic-value"],
)
def test_candidate_parser_is_closed_and_digest_registered(mutation: str) -> None:
    descriptor = candidate_policy("bf16").to_dict()
    if mutation == "missing":
        descriptor.pop("linear")
    elif mutation == "extra":
        descriptor["invented"] = "value"
    elif mutation == "schema-bool":
        descriptor["schema_version"] = True
    elif mutation == "flag-missing":
        cast(dict[str, Any], descriptor["flags"]).pop("compile")
    elif mutation == "flag-value":
        cast(dict[str, Any], descriptor["flags"])["autocast"] = True
    else:
        descriptor["linear"] = "opaque-backend-default"

    with pytest.raises(NumericalPolicyError):
        parse_policy(descriptor)


@pytest.mark.parametrize("mutation", ["contiguous", "wrong-authority", "missing", "extra"])
def test_legacy_parser_is_closed_and_cannot_register_c0(mutation: str) -> None:
    descriptor = legacy_policy("fp16").to_dict()
    if mutation == "contiguous":
        descriptor["legal_kv_modes"] = ["none", "contiguous"]
    elif mutation == "wrong-authority":
        descriptor["authority_tag"] = "main"
    elif mutation == "missing":
        descriptor.pop("reference_sha256")
    else:
        descriptor["cached_schedule"] = "historical-c0"

    with pytest.raises(NumericalPolicyError):
        parse_policy(descriptor)


def test_unknown_family_and_non_mapping_fail_closed() -> None:
    descriptor = candidate_policy("fp32").to_dict()
    descriptor["family"] = "third-party-numerical-policy"
    with pytest.raises(NumericalPolicyError, match="unknown numerical policy family"):
        parse_policy(descriptor)
    with pytest.raises(NumericalPolicyError, match="mapping"):
        parse_policy(cast(Any, []))


def test_raw_policy_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    raw = json.dumps(candidate_policy("bf16").to_dict(), ensure_ascii=False)
    duplicate = raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
    nonfinite = raw.replace('"cpu_threads": 1', '"cpu_threads": NaN', 1)

    with pytest.raises(NumericalPolicyError, match="duplicate"):
        parse_policy_json(duplicate)
    with pytest.raises(NumericalPolicyError, match="non-finite"):
        parse_policy_json(nonfinite)
    with pytest.raises(NumericalPolicyError, match="object"):
        parse_policy_json("[]")


def test_raw_policy_json_round_trip_preserves_registered_identity() -> None:
    policy = candidate_policy("fp16")
    raw = json.dumps(policy.to_dict(), ensure_ascii=False, sort_keys=False).encode("utf-8")

    assert parse_policy_json(raw) == policy

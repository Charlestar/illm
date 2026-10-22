"""Strict V0.2 profile schema dispatch; invalid policy evidence never falls back."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from illm.domain.errors import ErrorCode, IllmError
from illm.policy_cpu_numerical_contract import read_json


def verify_v02_profile_evidence(
    root: Path, *, expected_source_identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(root / "profile-result.json")
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "V0.2 profile requires an integer schema discriminator")
    schema = value["schema_version"]
    if schema == 2:
        # The old reader's exact closed field set rejects policy shadow fields.
        from illm.stateful_qualification import verify_stateful_profile_evidence

        return verify_stateful_profile_evidence(root, expected_source_identity=expected_source_identity)
    if schema == 3:
        from illm.stateful_policy_qualification import verify_policy_stateful_profile_evidence

        return verify_policy_stateful_profile_evidence(root, expected_source_identity=expected_source_identity)
    raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported V0.2 profile schema")


__all__ = ["verify_v02_profile_evidence"]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from illm.canonical import canonical_json_bytes, sha256_json
from illm.domain.errors import ErrorCode, IllmError


PROFILE_RESULTS = {
    "not_run",
    "pass",
    "fail",
    "invalid",
    "not_run_resource_unavailable",
    "not_applicable",
}
QUALIFICATION_TIERS = {
    "unqualified": 0,
    "learning_complete": 1,
    "functionally_qualified": 2,
    "stable_released": 3,
}
SPEC_STATUSES = {"draft", "accepted", "implemented", "released", "superseded"}
RELEASE_STAGES = {"dev_snapshot", "release_candidate", "released", "withdrawn"}
PERFORMANCE_OUTCOMES = {"not_applicable", "pending", "NoCandidate", "Adopt", "No-go"}
LTS_STATES = {"not_applicable", "candidate", "promoted", "eol"}
V0_REQUIRED_PROFILES = {
    "cpu-clean-room-v1",
    "windows-bootstrap-v1",
    "linux-bootstrap-v1",
}
RELEASE_FIELDS = {
    "schema_version",
    "spec_status",
    "release_stage",
    "qualification_tier",
    "qualification_profiles",
    "performance_outcome",
    "lts_state",
    "toolchain_manifest_digest",
    "capture_manifest_digest",
    "test_report_digest",
    "event_stall_digest",
    "manifest_digest",
}
PROFILE_FIELDS = {
    "profile_id",
    "result",
    "required_for_functional",
    "environment_manifest_digest",
    "evidence_digest",
    "reason",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def recompute_qualification_tier(profiles: Sequence[Mapping[str, Any]]) -> str:
    by_id = {str(profile["profile_id"]): profile for profile in profiles}
    cpu = by_id.get("cpu-clean-room-v1")
    if not cpu or cpu.get("result") != "pass":
        return "unqualified"
    fixed_required = [by_id.get(profile_id) for profile_id in sorted(V0_REQUIRED_PROFILES)]
    additional_required = [
        profile
        for profile in profiles
        if profile.get("required_for_functional", True)
        and profile.get("profile_id") not in V0_REQUIRED_PROFILES
    ]
    required = [*fixed_required, *additional_required]
    if all(profile is not None and profile.get("result") == "pass" for profile in required):
        return "functionally_qualified"
    return "learning_complete"


def verify_release_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release manifest is unreadable") from exc
    if manifest.get("schema_version") != 0:
        raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "unsupported ReleaseManifest schema")
    fields = set(manifest)
    if fields != RELEASE_FIELDS:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "ReleaseManifest fields do not match schema v0",
            {"missing": sorted(RELEASE_FIELDS - fields), "extra": sorted(fields - RELEASE_FIELDS)},
        )
    stored_digest = manifest.get("manifest_digest")
    digest_projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if not _is_sha256(stored_digest) or stored_digest != sha256_json(digest_projection):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release manifest digest mismatch")
    for field in (
        "toolchain_manifest_digest",
        "capture_manifest_digest",
        "test_report_digest",
        "event_stall_digest",
    ):
        if not _is_sha256(manifest.get(field)):
            raise IllmError(ErrorCode.INVALID_ARGUMENT, f"invalid {field}")
    enum_fields = {
        "spec_status": SPEC_STATUSES,
        "release_stage": RELEASE_STAGES,
        "performance_outcome": PERFORMANCE_OUTCOMES,
        "lts_state": LTS_STATES,
    }
    for field, values in enum_fields.items():
        if manifest.get(field) not in values:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                f"invalid {field}",
                {"value": manifest.get(field)},
            )
    profiles = manifest.get("qualification_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification_profiles must be non-empty")
    profile_ids = [str(profile.get("profile_id", "")) for profile in profiles]
    if "" in profile_ids or len(set(profile_ids)) != len(profile_ids):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profile IDs must be unique and non-empty")
    missing_profiles = V0_REQUIRED_PROFILES - set(profile_ids)
    if missing_profiles:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "required V0 qualification profiles are missing",
            {"missing": sorted(missing_profiles)},
        )
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != PROFILE_FIELDS:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profile fields do not match schema v0")
        if profile["profile_id"] in V0_REQUIRED_PROFILES and profile.get("required_for_functional") is not True:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                "required V0 profile cannot be made optional",
                {"profile_id": profile["profile_id"]},
            )
        if profile.get("result") not in PROFILE_RESULTS:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                "invalid qualification profile result",
                {"profile_id": profile.get("profile_id"), "result": profile.get("result")},
            )
        if profile.get("result") == "pass" and not (
            _is_sha256(profile.get("environment_manifest_digest"))
            and _is_sha256(profile.get("evidence_digest"))
            and profile.get("reason") is None
        ):
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                "passing profile requires exact digests and no failure reason",
                {"profile_id": profile.get("profile_id")},
            )
        if profile.get("result") != "pass" and (
            not isinstance(profile.get("reason"), str) or not profile.get("reason")
        ):
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                "non-passing profile requires a reason",
                {"profile_id": profile.get("profile_id")},
            )
    recomputed = recompute_qualification_tier(profiles)
    claimed = str(manifest.get("qualification_tier", "unqualified"))
    if claimed not in QUALIFICATION_TIERS:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "unknown qualification tier")
    if QUALIFICATION_TIERS[claimed] > QUALIFICATION_TIERS[recomputed]:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "qualification tier is higher than evidence permits",
            {"claimed": claimed, "recomputed": recomputed},
        )
    if manifest["release_stage"] == "released" and (
        manifest["spec_status"] != "released"
        or QUALIFICATION_TIERS[recomputed] < QUALIFICATION_TIERS["functionally_qualified"]
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "released stage is not supported by evidence/status")
    if manifest["spec_status"] == "released" and manifest["release_stage"] != "released":
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "released spec must use released stage")
    result = dict(manifest)
    result["recomputed_qualification_tier"] = recomputed
    result["verification_digest"] = sha256_json(result)
    return result


def write_verified_release_state(result: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(result)) + b"\n")
    temporary.replace(output)

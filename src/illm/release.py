from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
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
    """Validate state/schema only; publication must use verify_release_bundle."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 0:
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
        if not isinstance(manifest.get(field), str) or manifest[field] not in values:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                f"invalid {field}",
                {"value": manifest.get(field)},
            )
    profiles = manifest.get("qualification_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification_profiles must be non-empty")
    if any(not isinstance(profile, dict) for profile in profiles):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profiles must be objects")
    profile_ids = [str(profile.get("profile_id", "")) for profile in profiles]
    if any(not isinstance(profile.get("profile_id"), str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", profile["profile_id"]) is None for profile in profiles) or len(set(profile_ids)) != len(profile_ids):
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
        if not isinstance(profile.get("result"), str) or profile["result"] not in PROFILE_RESULTS:
            raise IllmError(
                ErrorCode.INVALID_ARGUMENT,
                "invalid qualification profile result",
                {"profile_id": profile.get("profile_id"), "result": profile.get("result")},
            )
        if type(profile.get("required_for_functional")) is not bool:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "profile required_for_functional must be boolean")
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


def _artifact_json(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate release JSON key")
            result[key] = item
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    except (OSError, ValueError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release artifact is unreadable", {"file": path.name}) from exc
    if not isinstance(value, dict):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release artifact must be an object", {"file": path.name})
    return value


def _inventory_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "release artifact cannot be a symlink")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def _verify_checksums(root: Path, checksums: Mapping[str, Any], files: Mapping[str, Path]) -> None:
    if set(checksums) != set(files):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release checksum coverage differs from artifact files")
    for name, digest in checksums.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts or "\\" in name:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "release checksum path is not contained")
        path = root / relative
        if not _is_sha256(digest) or path.resolve() != files[name].resolve() or sha256_file(path) != digest:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "release artifact checksum mismatch", {"file": name})


def _verify_stall_report(report: Mapping[str, Any]) -> None:
    positive = (
        "completed_scheduler_steps", "offered_events", "required_delivered_events",
        "exporter_queue_capacity", "exporter_in_flight_limit", "memory_growth_limit_bytes",
    )
    nonnegative = ("exporter_queue_high_watermark", "traced_current_growth_bytes", "traced_peak_bytes")
    if any(type(report.get(field)) is not int or report[field] < 1 for field in positive) or any(
        type(report.get(field)) is not int or report[field] < 0 for field in nonnegative
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "event-stall report has invalid measurements")
    booleans = (
        "exporter_was_blocked", "producer_completed_while_exporter_blocked", "forward_progress",
        "memory_bound_pass", "shutdown_after_release",
    )
    duration = report.get("duration_s")
    if (
        report.get("schema_version") != 0
        or any(report.get(field) is not True for field in booleans)
        or report.get("completed_scheduler_steps", 0) < 10_000
        or type(report.get("critical_event_drop_count")) is not int
        or report["critical_event_drop_count"] != 0
        or report["required_delivered_events"] != report["offered_events"]
        or report["exporter_queue_high_watermark"] > report["exporter_queue_capacity"]
        or report["exporter_in_flight_limit"] != 1
        or type(report.get("exporter_drop_count")) is not int
        or report["exporter_drop_count"] < 1
        or report["traced_current_growth_bytes"] > report["memory_growth_limit_bytes"]
        or report["memory_growth_limit_bytes"] > 1_048_576
        or report.get("close_while_blocked_returned") is not False
        or isinstance(duration, bool) or not isinstance(duration, (int, float))
        or not math.isfinite(duration) or not 0 < duration <= 300
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "event-stall report does not prove bounded forward progress")


def _verify_rollback(root: Path, source_identity: Mapping[str, str]) -> None:
    files = {
        name: root / name for name in (
            "manifest.json", "toolchain.json", "commands.txt", "uninstall.stdout.txt",
            "uninstall.stderr.txt", "rollback-report.md",
        )
    }
    if any(not path.is_file() for path in files.values()):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback evidence is incomplete")
    if set(_inventory_files(root)) != {*files, "checksums.json"}:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback artifact file set is invalid")
    _verify_checksums(root, _artifact_json(root / "checksums.json"), files)
    report = _artifact_json(root / "manifest.json")
    toolchain = _artifact_json(root / "toolchain.json")
    toolchain_digest = sha256_json({key: value for key, value in toolchain.items() if key != "manifest_digest"})
    if (
        toolchain.get("manifest_digest") != toolchain_digest
        or toolchain.get("finalized") is not True
        or toolchain.get("schema_version") != 0
        or any(toolchain.get(field) != value for field, value in source_identity.items())
        or report.get("toolchain_manifest_digest") != toolchain_digest
        or report.get("environment_id") != toolchain.get("environment_id")
        or report.get("python_executable") != toolchain.get("python_executable")
        or report.get("schema_version") != 0
        or report.get("source_release") != "v0.0.0"
        or report.get("target_state") != "pre-v0.0-empty-runtime"
        or report.get("installed_before") is not True
        or report.get("installed_after") is not False
        or report.get("success") is not True
        or type(report.get("pip_returncode")) is not int or report["pip_returncode"] != 0
        or type(report.get("background_processes_started")) is not int or report["background_processes_started"] != 0
        or type(report.get("ports_opened")) is not int or report["ports_opened"] != 0
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback evidence does not prove the qualified environment was removed")


def verify_release_bundle(
    path: Path,
    *,
    expected_source_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify recorded facts, without requiring their original host paths to exist.

    An optional source identity additionally binds the evidence to a checkout.
    Platform manifests retain historical absolute paths for provenance only.
    """
    from illm.capture import CaptureBundle
    from illm.qualification import SOURCE_IDENTITY_FIELDS, parse_junit, verify_profile_evidence
    from illm.replay import replay_events

    if path.is_symlink() or path.parent.is_symlink():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release root cannot be a symlink")
    header = _artifact_json(path)
    if type(header.get("schema_version")) is int and header["schema_version"] == 1:
        from illm.reference_release import verify_reference_release_bundle

        return verify_reference_release_bundle(path, expected_source_identity=expected_source_identity)
    result = verify_release_manifest(path)
    root = path.resolve().parent
    files = _inventory_files(root)
    sums_path = root / "SHA256SUMS.json"
    if sums_path.exists():
        _verify_checksums(root, _artifact_json(sums_path), {name: item for name, item in files.items() if name != "SHA256SUMS.json"})
    elif result["release_stage"] == "released":
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "released bundle requires SHA256SUMS.json")
    for name in ("commands.txt", "teach-back.md", "known-limitations.md", "release-notes.md"):
        if name not in files or not files[name].read_text(encoding="utf-8").strip():
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "required learning artifact is missing", {"file": name})
    if not any(name.startswith("screenshots/terminal/") and item.stat().st_size > 0 for name, item in files.items()):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "terminal demonstration evidence is missing")
    source_identity: dict[str, str] | None = None
    qualified: dict[str, tuple[dict[str, Any], Path]] = {}
    for profile in result["qualification_profiles"]:
        profile_id = str(profile["profile_id"])
        profile_dir = root / "qualification" / "profiles" / profile_id
        if _artifact_json(profile_dir / "profile-result.json") != profile:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "release profile result disagrees with manifest")
        if profile_id == "cpu-clean-room-v1" or profile["result"] != "pass":
            continue
        evidence_root = profile_dir / "evidence" / str(profile["evidence_digest"])
        verified_profile, index = verify_profile_evidence(evidence_root, expected_source_identity=expected_source_identity)
        if verified_profile != profile:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "sealed profile differs from release manifest")
        identity = {field: str(index["source_identity"][field]) for field in SOURCE_IDENTITY_FIELDS}
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "release platform evidence belongs to different sources")
        qualified[profile_id] = (verified_profile, evidence_root)
    if source_identity is None:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release has no sealed platform evidence")
    base = next((item for item in qualified.values() if item[0]["environment_manifest_digest"] == result["toolchain_manifest_digest"]), None)
    if base is None:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "release toolchain is not backed by sealed evidence")
    test_report = _artifact_json(root / "test-report.json")
    actual_tests = parse_junit(base[1] / "junit.xml")
    expected_test_report = {"schema_version": 0, **actual_tests, "junit_sha256": sha256_file(base[1] / "junit.xml")}
    if test_report != expected_test_report or sha256_json(test_report) != result["test_report_digest"]:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release test report disagrees with qualified JUnit facts")
    cpu = next(profile for profile in result["qualification_profiles"] if profile["profile_id"] == "cpu-clean-room-v1")
    if cpu["result"] != "pass" or cpu["evidence_digest"] != result["test_report_digest"] or cpu["environment_manifest_digest"] != result["toolchain_manifest_digest"]:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "CPU profile is not backed by the qualified test report")
    capture = CaptureBundle.open(root / "capture" / "toy-online-replay")
    state = replay_events(capture.initial_state, capture.events, capture.expected_step_digests)
    if capture.manifest["manifest_digest"] != result["capture_manifest_digest"] or state.digest != capture.manifest["final_state_digest"]:
        raise IllmError(ErrorCode.REPLAY_DIVERGED, "release capture digest/replay mismatch")
    base_capture = _artifact_json(base[1] / "capture" / "capture-manifest.json")
    if capture.manifest != base_capture:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release capture is not the qualified capture")
    stall_report = _artifact_json(root / "benchmark" / "event-stall" / "report.json")
    if sha256_json(stall_report) != result["event_stall_digest"]:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "release event-stall digest mismatch")
    _verify_stall_report(stall_report)
    _verify_rollback(root / "rollback", source_identity)
    recorded_state = _artifact_json(root / "qualification" / "release-state.json")
    if recorded_state != result:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "recorded release state is stale")
    return result


def write_verified_release_state(result: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(result)) + b"\n")
    temporary.replace(output)

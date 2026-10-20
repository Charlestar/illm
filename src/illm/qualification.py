from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture import CaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.release import recompute_qualification_tier, verify_release_manifest, write_verified_release_state
from illm.replay import replay_events


PROFILE_BY_OS = {
    "Windows": "windows-bootstrap-v1",
    "Linux": "linux-bootstrap-v1",
}
CAPTURE_FILES = (
    "capture-manifest.json",
    "checkpoint.json",
    "events.jsonl",
    "state-digests.jsonl",
    "checksums.json",
    "COMPLETE",
)
EVIDENCE_FILES = {
    "toolchain.json",
    "junit.xml",
    "quality.json",
    *(f"capture/{name}" for name in CAPTURE_FILES),
}
SOURCE_IDENTITY_FIELDS = {"project_root_digest", "lock_digest", "bootstrap_digest"}
PROFILE_FIELDS = {
    "profile_id",
    "result",
    "required_for_functional",
    "environment_manifest_digest",
    "evidence_digest",
    "reason",
}
INDEX_FIELDS = {
    "schema_version",
    "profile_id",
    "os",
    "arch",
    "python_version",
    "source_identity",
    "files",
    "evidence_digest",
    "index_digest",
}


def _filesystem_path(path: Path) -> Path:
    """Use the Windows long-path API without rewriting provenance identities."""
    resolved = path.resolve()
    value = str(resolved)
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return resolved
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, f"{label} must be an object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(dict(value)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})


def parse_junit(path: Path) -> dict[str, int | float]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "JUnit report is unreadable") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite")) if root.tag == "testsuites" else []
    if not suites:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "JUnit report contains no test suites")
    result: dict[str, int | float] = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "duration_s": 0.0}
    try:
        for suite in suites:
            counts = {field: int(suite.attrib[field]) for field in ("tests", "failures", "errors", "skipped")}
            duration = float(suite.attrib.get("time", "0"))
            if any(value < 0 for value in counts.values()) or not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid JUnit counts or duration")
            cases = list(suite.findall("testcase"))
            actual = {
                "tests": len(cases),
                "failures": sum(case.find("failure") is not None for case in cases),
                "errors": sum(case.find("error") is not None for case in cases),
                "skipped": sum(case.find("skipped") is not None for case in cases),
            }
            if counts != actual:
                raise ValueError("JUnit summary disagrees with testcase facts")
            for field, count in counts.items():
                result[field] += count
            result["duration_s"] += duration
    except (KeyError, ValueError, OverflowError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "JUnit report has invalid testcase accounting") from exc
    if result["tests"] <= result["skipped"] or result["failures"] or result["errors"] or not math.isfinite(result["duration_s"]):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "JUnit report is not passing", result)
    return result


def _verify_quality(path: Path) -> dict[str, Any]:
    quality = _read_json(path, "quality report")
    type_check = quality.get("type_check")
    if (
        quality.get("schema_version") != 0
        or quality.get("status") != "pass"
        or quality.get("issues") != []
        or type(quality.get("checked_python_files")) is not int
        or quality.get("checked_python_files", 0) < 1
        or not isinstance(type_check, dict)
        or type_check.get("tool") != "mypy"
        or not isinstance(type_check.get("version"), str)
        or type_check.get("status") != "pass"
        or type_check.get("returncode") != 0
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "quality report is not passing")
    return quality


def _verify_capture(path: Path) -> CaptureBundle:
    bundle = CaptureBundle.open(path)
    state = replay_events(bundle.initial_state, bundle.events, bundle.expected_step_digests)
    if state.digest != bundle.manifest.get("final_state_digest"):
        raise IllmError(ErrorCode.REPLAY_DIVERGED, "qualification capture final digest mismatch")
    return bundle


def _profile_evidence_digest(
    profile_id: str,
    source_identity: Mapping[str, str],
    files: Mapping[str, str],
) -> str:
    return sha256_json(
        {
            "profile_id": profile_id,
            "source_identity": dict(source_identity),
            "files": dict(files),
        }
    )


def create_profile_evidence(
    *,
    toolchain_manifest: Path,
    junit: Path,
    quality: Path,
    capture: Path,
    output: Path,
) -> dict[str, Any]:
    toolchain_manifest = _filesystem_path(toolchain_manifest)
    junit = _filesystem_path(junit)
    quality = _filesystem_path(quality)
    capture = _filesystem_path(capture)
    system = platform.system()
    profile_id = PROFILE_BY_OS.get(system)
    if profile_id is None:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "V0.0 qualification supports only Windows or Linux")
    toolchain = _read_json(toolchain_manifest, "toolchain manifest")
    stored_manifest_digest = toolchain.get("manifest_digest")
    if not _is_sha256(stored_manifest_digest) or stored_manifest_digest != _manifest_digest(toolchain):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest digest mismatch")
    if toolchain.get("finalized") is not True or toolchain.get("os") != system:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain platform/finalization mismatch")
    if Path(str(toolchain.get("python_executable", ""))).resolve() != Path(sys.executable).resolve():
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualification is not running in manifest Python")
    if os.environ.get("ILLM_ENVIRONMENT_ID") != toolchain.get("environment_id"):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualification environment ID mismatch")
    if os.environ.get("ILLM_LOCK_DIGEST") != toolchain.get("lock_digest"):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualification lock digest mismatch")
    parse_junit(junit)
    _verify_quality(quality)
    _verify_capture(capture)

    output = _filesystem_path(output)
    if output.exists():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence output already exists")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification temporary output already exists")
    temporary.mkdir(parents=True)
    (temporary / "INCOMPLETE").write_text("qualification_not_finalized\n", encoding="ascii")
    try:
        shutil.copy2(toolchain_manifest, temporary / "toolchain.json")
        shutil.copy2(junit, temporary / "junit.xml")
        shutil.copy2(quality, temporary / "quality.json")
        capture_output = temporary / "capture"
        capture_output.mkdir()
        for name in CAPTURE_FILES:
            source = capture / name
            if not source.is_file():
                raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "qualification capture fact is missing", {"file": name})
            shutil.copy2(source, capture_output / name)
        files = {name: sha256_file(temporary / name) for name in sorted(EVIDENCE_FILES)}
        source_identity = {
            field: str(toolchain.get(field, "")) for field in sorted(SOURCE_IDENTITY_FIELDS)
        }
        if not all(_is_sha256(value) for value in source_identity.values()):
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain source identity is invalid")
        evidence_digest = _profile_evidence_digest(profile_id, source_identity, files)
        profile = {
            "profile_id": profile_id,
            "result": "pass",
            "required_for_functional": True,
            "environment_manifest_digest": stored_manifest_digest,
            "evidence_digest": evidence_digest,
            "reason": None,
        }
        index: dict[str, Any] = {
            "schema_version": 0,
            "profile_id": profile_id,
            "os": system,
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "source_identity": source_identity,
            "files": files,
            "evidence_digest": evidence_digest,
        }
        index["index_digest"] = sha256_json(index)
        _atomic_write_json(temporary / "profile-result.json", profile)
        _atomic_write_json(temporary / "evidence-index.json", index)
        (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
        (temporary / "INCOMPLETE").unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return profile
    except Exception:
        if temporary.exists():
            (temporary / "INCOMPLETE").write_text("qualification_failed\n", encoding="ascii")
        raise


def verify_profile_evidence(
    root: Path,
    *,
    expected_source_identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _filesystem_path(root)
    if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
        raise IllmError(ErrorCode.CAPTURE_INCOMPLETE, "qualification evidence is not finalized")
    profile = _read_json(root / "profile-result.json", "profile result")
    index = _read_json(root / "evidence-index.json", "evidence index")
    if set(profile) != PROFILE_FIELDS or set(index) != INDEX_FIELDS:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence fields do not match schema v0")
    stored_index_digest = index.get("index_digest")
    if not _is_sha256(stored_index_digest) or stored_index_digest != sha256_json(
        {key: value for key, value in index.items() if key != "index_digest"}
    ):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence index digest mismatch")
    profile_id = str(profile.get("profile_id", ""))
    if profile_id != index.get("profile_id") or PROFILE_BY_OS.get(str(index.get("os"))) != profile_id:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profile/platform identity mismatch")
    if profile.get("result") != "pass" or profile.get("required_for_functional") is not True:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profile is not a required pass")
    files = index.get("files")
    source_identity = index.get("source_identity")
    if not isinstance(files, dict) or set(files) != EVIDENCE_FILES:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence file set is invalid")
    if not isinstance(source_identity, dict) or set(source_identity) != SOURCE_IDENTITY_FIELDS:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification source identity is invalid")
    if not all(_is_sha256(value) for value in source_identity.values()):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification source identity digests are invalid")
    for name, expected_digest in files.items():
        path = root / name
        if not _is_sha256(expected_digest) or not path.is_file() or sha256_file(path) != expected_digest:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence checksum mismatch", {"file": name})
    evidence_digest = _profile_evidence_digest(profile_id, source_identity, files)
    if profile.get("evidence_digest") != evidence_digest or index.get("evidence_digest") != evidence_digest:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification evidence digest mismatch")
    toolchain = _read_json(root / "toolchain.json", "qualified toolchain manifest")
    if toolchain.get("manifest_digest") != _manifest_digest(toolchain):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified toolchain manifest digest mismatch")
    if (
        toolchain.get("schema_version") != 0
        or toolchain.get("finalized") is not True
        or any(toolchain.get(field) != index.get(field) for field in ("os", "arch", "python_version"))
        or profile.get("reason") is not None
    ):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified toolchain platform/finalization mismatch")
    if profile.get("environment_manifest_digest") != toolchain.get("manifest_digest"):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualification environment digest mismatch")
    if any(toolchain.get(field) != source_identity[field] for field in SOURCE_IDENTITY_FIELDS):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified toolchain source identity mismatch")
    if expected_source_identity is not None and dict(source_identity) != dict(expected_source_identity):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualification evidence belongs to different source")
    parse_junit(root / "junit.xml")
    quality = _verify_quality(root / "quality.json")
    packages = toolchain.get("installed_packages")
    if not isinstance(packages, dict) or quality["type_check"]["version"] != packages.get("mypy"):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified type checker differs from locked toolchain")
    capture = _verify_capture(root / "capture")
    run = capture.manifest.get("run")
    if not isinstance(run, dict) or any(
        run.get(run_field) != toolchain.get(toolchain_field)
        for run_field, toolchain_field in (("os", "os"), ("arch", "arch"), ("python", "python_version"))
    ):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "qualified capture platform differs from toolchain")
    return profile, index


def merge_profile_evidence(
    *,
    release_dir: Path,
    base_toolchain_manifest: Path,
    evidence_roots: Iterable[Path],
) -> dict[str, Any]:
    release_dir = _filesystem_path(release_dir)
    base_toolchain_manifest = _filesystem_path(base_toolchain_manifest)
    manifest_path = release_dir / "manifest.json"
    manifest = _read_json(manifest_path, "release manifest")
    verify_release_manifest(manifest_path)
    base_toolchain = _read_json(base_toolchain_manifest, "base toolchain manifest")
    if base_toolchain.get("manifest_digest") != _manifest_digest(base_toolchain):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "base toolchain manifest digest mismatch")
    if manifest.get("toolchain_manifest_digest") != base_toolchain.get("manifest_digest"):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "release and base toolchain digests differ")
    source_identity = {
        field: str(base_toolchain.get(field, "")) for field in sorted(SOURCE_IDENTITY_FIELDS)
    }
    profiles = {str(item["profile_id"]): dict(item) for item in manifest["qualification_profiles"]}
    for evidence_root in evidence_roots:
        evidence_root = _filesystem_path(evidence_root)
        profile, index = verify_profile_evidence(
            evidence_root,
            expected_source_identity=source_identity,
        )
        profile_id = str(profile["profile_id"])
        if profile_id not in profiles:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification profile is not declared by release")
        destination = (
            release_dir
            / "qualification"
            / "profiles"
            / profile_id
            / "evidence"
            / str(index["evidence_digest"])
        )
        if destination.exists():
            verify_profile_evidence(destination, expected_source_identity=source_identity)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            if staging.exists():
                raise IllmError(ErrorCode.INVALID_ARGUMENT, "qualification import staging path exists")
            shutil.copytree(evidence_root, staging)
            verify_profile_evidence(staging, expected_source_identity=source_identity)
            os.replace(staging, destination)
        profiles[profile_id] = profile
        profile_path = release_dir / "qualification" / "profiles" / profile_id / "profile-result.json"
        _atomic_write_json(profile_path, profile)
    manifest["qualification_profiles"] = [profiles[key] for key in sorted(profiles)]
    manifest["qualification_tier"] = recompute_qualification_tier(manifest["qualification_profiles"])
    manifest["manifest_digest"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    _atomic_write_json(manifest_path, manifest)
    verified = verify_release_manifest(manifest_path)
    write_verified_release_state(verified, release_dir / "qualification" / "release-state.json")
    return verified

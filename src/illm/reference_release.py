"""V0.1 release assembly and portable semantic verification, never Git mutation."""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from illm.canonical import sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.qualification import parse_junit
from illm.reference_qualification import (
    PROFILES, PROFILE_FIELDS, SOURCE_FIELDS, _digest, _fail, _fs, _hash_map, _inventory,
    _object, _relative, _signed, _write, verify_reference_profile_evidence,
)
from illm.reference_rollback import portable_rollback_inventory, verify_reference_rollback
from illm.reference_learning import LEARNING_DIRECTORY, LEARNING_FILES, derive_reference_learning_assets


VERSION = "0.1.0"
TAG = "v0.1.0"
PRIMARY = "tiny-reference-cpu-v1"
DOCUMENTS = {"commands.txt", "teach-back.md", "known-limitations.md", "release-notes.md", "spec.md"}
DOCUMENT_SOURCES = {
    "commands.txt": "evidence/v0.1/commands.txt", "teach-back.md": "docs/teach-back-v0.1.md",
    "known-limitations.md": "evidence/v0.1/known-limitations.md", "release-notes.md": "evidence/v0.1/release-notes.md",
    "spec.md": "docs/specs/versions/v0.1-reference-runner.md",
}
MAX_RELEASE_FILE_BYTES = 64 * 1024 * 1024
MAX_RELEASE_BYTES = 256 * 1024 * 1024
MANIFEST_FIELDS = {
    "schema_version", "release_version", "release_tag", "spec_status", "release_stage", "qualification_tier",
    "qualification_profiles", "performance_outcome", "lts_state", "source_identity", "primary_profile_id",
    "toolchain_manifest_digest", "capture_manifest_digest", "test_report_digest", "benchmark_report_digest",
    "rollback_report_digest", "documentation_digest", "manifest_digest",
}


def _identity(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != SOURCE_FIELDS or any(not _digest(item) for item in value.values()):
        _fail("V0.1 release requires the exact common source identity")
    return {str(name): str(item) for name, item in value.items()}


def _bounded(root: Path) -> None:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            _fail("release cannot contain symbolic links")
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > MAX_RELEASE_FILE_BYTES or total > MAX_RELEASE_BYTES:
                _fail("release exceeds portable evidence bounds; external weights/environments must not be embedded")


def _state(path: Path) -> dict[str, Any]:
    manifest = _object(path)
    if set(manifest) != MANIFEST_FIELDS or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        _fail("unsupported V0.1 ReleaseManifest fields/schema", ErrorCode.UNSUPPORTED_SCHEMA)
    _signed(manifest, "manifest_digest")
    if manifest["release_version"] != VERSION or manifest["release_tag"] != TAG or manifest["primary_profile_id"] != PRIMARY:
        _fail("V0.1 release version/tag/primary profile identity is invalid")
    _identity(manifest["source_identity"])
    if any(not _digest(manifest[name]) for name in ("toolchain_manifest_digest", "capture_manifest_digest", "test_report_digest", "benchmark_report_digest", "rollback_report_digest", "documentation_digest")):
        _fail("V0.1 release artifact digest is invalid")
    profiles = manifest["qualification_profiles"]
    if not isinstance(profiles, list) or len(profiles) != 3 or any(not isinstance(item, dict) or set(item) != PROFILE_FIELDS for item in profiles):
        _fail("V0.1 release requires exactly three fully specified profiles")
    if {item["profile_id"] for item in profiles} != PROFILES:
        _fail("V0.1 release omits or duplicates a required profile")
    for profile in profiles:
        if profile["result"] != "pass" or profile["required_for_functional"] is not True or profile["reason"] is not None or not _digest(profile["environment_manifest_digest"]) or not _digest(profile["evidence_digest"]):
            _fail("V0.1 full release is not backed by every required passing profile")
    stage = manifest["release_stage"]
    if stage not in {"release_candidate", "released"} or manifest["spec_status"] != ("released" if stage == "released" else "implemented"):
        _fail("V0.1 spec/release axes do not describe a complete candidate or release")
    if manifest["qualification_tier"] != "functionally_qualified" or manifest["performance_outcome"] != "not_applicable" or manifest["lts_state"] != "not_applicable":
        _fail("V0.1 cannot claim stable/LTS promotion or performance improvement")
    result = dict(manifest)
    result["recomputed_qualification_tier"] = "functionally_qualified"
    result["verification_digest"] = sha256_json(result)
    return result


def _profile_root(root: Path, profile: Mapping[str, Any]) -> Path:
    return root / "qualification" / "profiles" / str(profile["profile_id"]) / "evidence" / str(profile["evidence_digest"])


def _test_report(profiles: Mapping[str, Path]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for profile_id, directory in sorted(profiles.items()):
        toolchain = _object(directory / "toolchain.json")
        reports[profile_id] = {**parse_junit(directory / "junit.xml"),
                               "junit_sha256": sha256_file(directory / "junit.xml"),
                               "quality_sha256": sha256_file(directory / "quality.json"),
                               "toolchain_manifest_digest": toolchain["manifest_digest"]}
    report: dict[str, Any] = {"schema_version": 1, "primary_profile_id": PRIMARY, "profiles": reports}
    report["report_digest"] = sha256_json(report)
    return report


def _verify_documentation(root: Path, primary: Path, state: Mapping[str, Any]) -> None:
    index = _object(root / "documentation-index.json")
    _signed(index, "documentation_digest")
    if set(index) != {"schema_version", "documents", "documentation_digest"} or type(index["schema_version"]) is not int or index["schema_version"] != 1 or index["documentation_digest"] != state["documentation_digest"] or not isinstance(index["documents"], dict) or set(index["documents"]) != DOCUMENTS:
        _fail("release learning documentation inventory is incomplete")
    source_files = _object(primary / "source-provenance.json")["project_files"]
    for name, record in index["documents"].items():
        expected = DOCUMENT_SOURCES[name]
        path = root / name
        if record != {"source_path": expected, "sha256": sha256_file(path)} or source_files.get(expected) != record["sha256"]:
            _fail("learning document bytes do not belong to qualified source")
        if path.stat().st_size > 512 * 1024:
            _fail("release learning document exceeds its text artifact bound")
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 40 or "\0" in text:
            _fail("required release learning document is empty or not text")
    spec = (root / "spec.md").read_text(encoding="utf-8")
    if "target_release: v0.1.0" not in spec or "## 12. Definition of Done" not in spec:
        _fail("release SPEC is not the complete V0.1 contract")
    if state["release_stage"] == "released" and ("spec_status: released" not in spec or "release_stage: released" not in spec):
        _fail("published V0.1 SPEC is not in the frozen released state")


def _verify_rollback(root: Path, source: Mapping[str, str], primary: Path, state: Mapping[str, Any]) -> None:
    directory = root / "rollback"
    if any(path.name == "local-only" for path in directory.rglob("*")):
        _fail("release must not publish local-only rollback environments/weight copies")
    report = verify_reference_rollback(directory)
    primary_toolchain = _object(primary / "toolchain.json")
    primary_capture = _object(primary / "capture" / "capture-manifest.json")
    if report.get("evidence_mode") != "release" or report.get("source_identity") != dict(source) or report.get("live_v01_source_verified") is not True or report.get("primary_v01_toolchain_manifest_digest") != primary_toolchain["manifest_digest"] or report.get("capture_manifest_digest") != primary_capture["manifest_digest"] or report["report_digest"] != state["rollback_report_digest"]:
        _fail("rollback is development-only or not bound to primary release source/capture")


def _learning_assets(primary: Path) -> dict[str, bytes]:
    directories = sorted((primary / "validation").iterdir())
    if len(directories) != 1 or not directories[0].is_dir() or directories[0].is_symlink():
        _fail("release learning diagrams require exactly the primary tiny validation")
    return derive_reference_learning_assets(directories[0])


def _verify_learning(root: Path, primary: Path) -> None:
    directory = root / LEARNING_DIRECTORY
    if (not directory.is_dir() or {path.name for path in directory.iterdir()} != LEARNING_FILES
            or any(not path.is_file() or path.is_symlink() for path in directory.iterdir())
            or set(_inventory(directory)) != LEARNING_FILES):
        _fail("release learning diagram inventory is incomplete or contains extra content")
    for name, expected in _learning_assets(primary).items():
        if (directory / name).read_bytes() != expected:
            _fail("release learning diagram bytes do not rebuild from qualified model/fault facts")


def _verify_bundle(path: Path, expected: Mapping[str, str] | None) -> dict[str, Any]:
    path = _fs(path)
    if path.is_symlink() or path.parent.is_symlink() or path.name != "manifest.json":
        _fail("release verification requires an ordinary manifest.json artifact")
    root = path.parent.resolve()
    if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
        _fail("V0.1 release evidence bundle is incomplete", ErrorCode.CAPTURE_INCOMPLETE)
    _bounded(root)
    files = _inventory(root)
    checksums = _hash_map(_object(root / "SHA256SUMS.json"))
    if checksums != {name: digest for name, digest in files.items() if name != "SHA256SUMS.json"}:
        _fail("release checksum coverage or artifact bytes do not match")
    result = _state(path)
    if (root / "COMPLETE").read_text(encoding="ascii") != result["manifest_digest"] + "\n":
        _fail("release completion marker is stale")
    source = _identity(result["source_identity"])
    if expected is not None and source != dict(expected):
        _fail("release evidence belongs to different source", ErrorCode.TOOLCHAIN_MISMATCH)
    qualified: dict[str, Path] = {}
    for profile in result["qualification_profiles"]:
        directory = _profile_root(root, profile)
        verified, index = verify_reference_profile_evidence(directory, expected_source_identity=source)
        if verified != profile or index["source_identity"] != source or _object(directory.parents[1] / "profile-result.json") != profile:
            _fail("release profile/index/manifest identities disagree")
        toolchain = _object(directory / "toolchain.json")
        project = tomllib.loads((directory / "locks" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        if toolchain.get("installed_packages", {}).get("illm") != VERSION or project.get("version") != VERSION:
            _fail("a development or forged package version cannot be published as v0.1.0")
        qualified[profile["profile_id"]] = directory
    primary = qualified[PRIMARY]
    if sha256_file(root / "toolchain.json") != sha256_file(primary / "toolchain.json") or _object(root / "toolchain.json")["manifest_digest"] != result["toolchain_manifest_digest"]:
        _fail("release primary toolchain differs from qualified tiny CPU environment")
    tests = _test_report(qualified)
    if _object(root / "test-report.json") != tests or tests["report_digest"] != result["test_report_digest"]:
        _fail("release test report does not recompute from every qualified JUnit")
    capture = root / "capture" / "tiny-generate"
    if _inventory(capture) != _inventory(primary / "capture"):
        _fail("public release capture differs from qualified capture bytes")
    bundle = ReferenceCaptureBundle.open(capture)
    if bundle.manifest["manifest_digest"] != result["capture_manifest_digest"] or bundle.final_state["status"] != "FINISHED":
        _fail("release capture does not prove finished reference generation/replay")
    benchmark = root / "benchmark" / "full-context-baseline"
    if _inventory(benchmark) != _inventory(primary / "benchmark") or _object(benchmark / "benchmark-report.json")["report_digest"] != result["benchmark_report_digest"]:
        _fail("public release baseline differs from qualified measured baseline")
    _verify_documentation(root, primary, result)
    _verify_rollback(root, source, primary, result)
    _verify_learning(root, primary)
    if _object(root / "qualification" / "release-state.json") != result:
        _fail("release-state record is stale or overclaims qualification")
    required = {"manifest.json", "toolchain.json", "test-report.json", "documentation-index.json", "qualification/release-state.json", "COMPLETE", "SHA256SUMS.json", *DOCUMENTS}
    required.update(f"{LEARNING_DIRECTORY}/{name}" for name in LEARNING_FILES)
    for profile in result["qualification_profiles"]:
        directory = qualified[profile["profile_id"]]
        prefix = directory.relative_to(root).as_posix()
        required.update(f"{prefix}/{name}" for name in _inventory(directory))
        required.add(f"qualification/profiles/{profile['profile_id']}/profile-result.json")
    for directory in (capture, benchmark, root / "rollback"):
        prefix = directory.relative_to(root).as_posix()
        required.update(f"{prefix}/{name}" for name in _inventory(directory))
    if set(files) != required:
        _fail("release contains missing or unreferenced evidence artifacts")
    return result


def verify_reference_release_bundle(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Audit every embedded required fact without using author environment paths."""
    try:
        return _verify_bundle(path, expected_source_identity)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "V0.1 release artifacts are malformed", {"exception_type": type(exc).__name__}) from exc


def build_reference_release_bundle(*, evidence_roots: Sequence[Path], rollback: Path, documentation_root: Path,
                                    output: Path, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Create a new complete release candidate; no source edits, tags or promotion."""
    output = _fs(output)
    evidence_roots = [_fs(path) for path in evidence_roots]
    rollback = _fs(rollback)
    documentation_root = _fs(documentation_root)
    if len(evidence_roots) != 3 or output.exists() or output.is_symlink():
        _fail("release assembly requires three sealed profiles and a new output")
    source: dict[str, str] | None = None
    profiles: dict[str, tuple[dict[str, Any], Path]] = {}
    for directory in evidence_roots:
        _bounded(directory)
        profile, index = verify_reference_profile_evidence(directory, expected_source_identity=expected_source_identity)
        identity = _identity(index["source_identity"])
        if source is not None and source != identity:
            _fail("release profiles come from mixed source snapshots", ErrorCode.TOOLCHAIN_MISMATCH)
        source = identity
        if profile["profile_id"] in profiles:
            _fail("release profile evidence is duplicated")
        profiles[profile["profile_id"]] = (profile, directory)
    if set(profiles) != PROFILES or source is None:
        _fail("release assembly omits a required qualified profile")
    primary = profiles[PRIMARY][1]
    for _, directory in profiles.values():
        project = tomllib.loads((directory / "locks" / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        if _object(directory / "toolchain.json").get("installed_packages", {}).get("illm") != VERSION or project.get("version") != VERSION:
            _fail("release assembly requires final package version0.1.0, not a development snapshot")
    rollback_report = verify_reference_rollback(rollback)
    if rollback_report.get("evidence_mode") != "release" or rollback_report.get("source_identity") != source:
        _fail("release assembly requires source-bound release-mode rollback evidence")
    if set(_inventory(documentation_root)) != DOCUMENTS:
        _fail("release documentation input must contain exactly five learning documents")
    temporary = output.resolve().with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        _fail("release staging output already exists")
    temporary.mkdir(parents=True)
    (temporary / "INCOMPLETE").write_text("release_assembly_in_progress\n", encoding="ascii")
    try:
        copied: dict[str, Path] = {}
        for profile_id, (profile, directory) in sorted(profiles.items()):
            destination = _profile_root(temporary, profile)
            shutil.copytree(directory, destination)
            _write(destination.parents[1] / "profile-result.json", profile)
            copied[profile_id] = destination
        primary_copy = copied[PRIMARY]
        shutil.copy2(primary_copy / "toolchain.json", temporary / "toolchain.json")
        shutil.copytree(primary_copy / "capture", temporary / "capture" / "tiny-generate")
        shutil.copytree(primary_copy / "benchmark", temporary / "benchmark" / "full-context-baseline")
        learning = temporary / LEARNING_DIRECTORY
        learning.mkdir(parents=True)
        for name, data in _learning_assets(primary_copy).items():
            (learning / name).write_bytes(data)
        rollback_files = {**portable_rollback_inventory(rollback), "checksums.json": sha256_file(rollback / "checksums.json"), "COMPLETE": sha256_file(rollback / "COMPLETE")}
        for name, digest in rollback_files.items():
            target = temporary / "rollback" / _relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rollback / name, target)
            if sha256_file(target) != digest:
                _fail("rollback evidence changed during portable release copy")
        docs: dict[str, Any] = {}
        for name in sorted(DOCUMENTS):
            shutil.copy2(documentation_root / name, temporary / name)
            docs[name] = {"source_path": DOCUMENT_SOURCES[name], "sha256": sha256_file(temporary / name)}
        document_index: dict[str, Any] = {"schema_version": 1, "documents": docs}
        document_index["documentation_digest"] = sha256_json(document_index)
        _write(temporary / "documentation-index.json", document_index)
        tests = _test_report(copied)
        _write(temporary / "test-report.json", tests)
        toolchain = _object(primary_copy / "toolchain.json")
        capture = _object(primary_copy / "capture" / "capture-manifest.json")
        benchmark = _object(primary_copy / "benchmark" / "benchmark-report.json")
        manifest: dict[str, Any] = {"schema_version": 1, "release_version": VERSION, "release_tag": TAG,
                                    "spec_status": "implemented", "release_stage": "release_candidate",
                                    "qualification_tier": "functionally_qualified", "qualification_profiles": [value[0] for _, value in sorted(profiles.items())],
                                    "performance_outcome": "not_applicable", "lts_state": "not_applicable", "source_identity": source,
                                    "primary_profile_id": PRIMARY, "toolchain_manifest_digest": toolchain["manifest_digest"],
                                    "capture_manifest_digest": capture["manifest_digest"], "test_report_digest": tests["report_digest"],
                                    "benchmark_report_digest": benchmark["report_digest"], "rollback_report_digest": rollback_report["report_digest"],
                                    "documentation_digest": document_index["documentation_digest"]}
        manifest["manifest_digest"] = sha256_json(manifest)
        _write(temporary / "manifest.json", manifest)
        _write(temporary / "qualification" / "release-state.json", _state(temporary / "manifest.json"))
        (temporary / "INCOMPLETE").unlink()
        (temporary / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
        _write(temporary / "SHA256SUMS.json", _inventory(temporary))
        result = verify_reference_release_bundle(temporary / "manifest.json", expected_source_identity=expected_source_identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return result
    except Exception:
        (temporary / "INCOMPLETE").write_text("release_assembly_failed\n", encoding="ascii")
        raise

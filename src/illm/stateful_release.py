"""V0.2 release assembly from complete profiles, not a qualification shortcut.

All three required profiles and a real V0.1 rollback must already be sealed.
This module never edits sources, changes a release stage, runs Git, or creates
a tag. Portable verification never resolves recorded author-machine paths.
"""

from __future__ import annotations

import os
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import registry_digest
from illm.qualification import parse_junit
from illm.reference_qualification import _fs, _hash_map, _inventory, _object, _relative, _signed, _write
from illm.stateful_benchmark import verify_completed_output
from illm.stateful_qualification import FORMAL_PROFILES, SOURCE_FIELDS, _check_tree, _digest, verify_stateful_profile_evidence
from illm.streaming_learning import verify_streaming_learning_assets


VERSION = "0.2.0"
TAG = "v0.2.0"
PRIMARY = "state-machine-cpu-v1"
GPU = "stateful-streaming-single-gpu-v1"
REPORT_KIND = "illm-v02-release-v1"
DOCUMENT_SOURCES = {
    "commands.txt": "evidence/v0.2/commands.txt",
    "teach-back.md": "docs/teach-back-v0.2.md",
    "known-limitations.md": "docs/known-limitations-v0.2.md",
    "release-notes.md": "docs/release-notes-v0.2.md",
    "spec.md": "docs/specs/versions/v0.2-stateful-streaming.md",
}
CAPTURE_SOURCES = {
    "success": "success", "length": "length", "eos": "eos",
    "cancel-after-launch": "cancel-after-launch", "no-kv": "no-kv",
    "exporter-stall": "exporter-stall", "incomplete-exporter": "required-write-failure",
}
BENCHMARKS = frozenset(f"{dtype}-{suite}" for dtype in ("bf16", "fp16") for suite in ("tuning", "heldout"))
MANIFEST_FIELDS = frozenset({
    "schema_version", "report_kind", "release_version", "release_tag", "spec_status", "release_stage",
    "qualification_tier", "qualification_profiles", "performance_outcome", "lts_state", "source_identity",
    "primary_profile_id", "toolchain_manifest_digest", "capture_manifest_digest", "capture_manifests",
    "execution_identities", "metric_registry_digest", "test_report_digest", "benchmark_report_digests",
    "rollback_report_digest", "documentation_digest", "manifest_digest",
})
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024


def _fail(message: str, code: ErrorCode = ErrorCode.INVALID_ARGUMENT) -> NoReturn:
    raise IllmError(code, message)


def _same(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _identity(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != SOURCE_FIELDS or any(not _digest(item) for item in value.values()):
        _fail("V0.2 release requires an exact source identity")
    return cast(dict[str, str], value)


def _bounded(root: Path) -> None:
    _check_tree(root)
    total = 0
    for path in root.rglob("*"):
        if not path.is_dir() and not path.is_file():
            _fail("V0.2 release forbids nonregular artifacts")
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > MAX_FILE_BYTES or total > MAX_BUNDLE_BYTES:
                _fail("V0.2 release exceeds portable artifact bounds; do not embed weights or environments")


def _state(path: Path) -> dict[str, Any]:
    value = _object(path)
    if set(value) != MANIFEST_FIELDS or type(value.get("schema_version")) is not int or value["schema_version"] != 2:
        _fail("unsupported V0.2 release manifest fields/schema", ErrorCode.UNSUPPORTED_SCHEMA)
    _signed(value, "manifest_digest")
    if (value["report_kind"] != REPORT_KIND or value["release_version"] != VERSION
            or value["release_tag"] != TAG or value["primary_profile_id"] != PRIMARY):
        _fail("V0.2 release identity is invalid")
    _identity(value["source_identity"])
    profiles = value.get("qualification_profiles")
    if not isinstance(profiles, list) or len(profiles) != len(FORMAL_PROFILES) or any(not isinstance(item, dict) for item in profiles):
        _fail("V0.2 release requires every complete CPU/GPU/clean-room profile")
    if {item.get("profile_id") for item in profiles} != FORMAL_PROFILES:
        _fail("V0.2 release profiles are missing, duplicated, or unknown")
    for profile in profiles:
        if (profile.get("evidence_mode") != "formal" or profile.get("simulated_profile_id") is not None
                or profile.get("result") != "pass" or profile.get("required_for_functional") is not True
                or profile.get("reason") is not None or not _digest(profile.get("evidence_digest"))
                or not _digest(profile.get("environment_manifest_digest"))):
            _fail("synthetic, partial, or failed evidence cannot qualify V0.2")
    stage = value["release_stage"]
    if (stage not in {"release_candidate", "released"}
            or value["spec_status"] != ("released" if stage == "released" else "implemented")
            or value["qualification_tier"] != "functionally_qualified"
            or value["performance_outcome"] != "not_applicable" or value["lts_state"] != "not_applicable"):
        _fail("V0.2 release axes cannot claim an incomplete, performance, or stable/LTS release")
    for name in ("toolchain_manifest_digest", "capture_manifest_digest", "metric_registry_digest", "test_report_digest",
                 "rollback_report_digest", "documentation_digest"):
        if not _digest(value[name]):
            _fail("V0.2 release artifact digest is invalid")
    if (not isinstance(value["benchmark_report_digests"], dict)
            or set(value["benchmark_report_digests"]) != BENCHMARKS
            or any(not _digest(item) for item in value["benchmark_report_digests"].values())):
        _fail("V0.2 release needs both dtypes and both benchmark suites")
    result = dict(value)
    result["recomputed_qualification_tier"] = "functionally_qualified"
    result["verification_digest"] = sha256_json(result)
    return result


def _profile_root(root: Path, profile: Mapping[str, Any]) -> Path:
    return root / "qualification/profiles" / str(profile["profile_id"]) / "evidence" / str(profile["evidence_digest"])


def _final_package(directory: Path) -> None:
    project = tomllib.loads((directory / "locks/pyproject.toml").read_text(encoding="utf-8"))["project"]
    packages = _object(directory / "toolchain.json").get("installed_packages", {})
    if project.get("version") != VERSION or packages.get("illm") != VERSION:
        _fail("a development package cannot be assembled or published as v0.2.0")


def _test_report(profiles: Mapping[str, Path]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 2, "primary_profile_id": PRIMARY,
        "profiles": {name: {**parse_junit(path / "junit.xml"), "junit_sha256": sha256_file(path / "junit.xml"),
                            "quality_sha256": sha256_file(path / "quality.json"),
                            "toolchain_manifest_digest": _object(path / "toolchain.json")["manifest_digest"]}
                     for name, path in sorted(profiles.items())},
    }
    report["report_digest"] = sha256_json(report)
    return report


def _capture_facts(primary: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for name, source_name in CAPTURE_SOURCES.items():
        directory = primary / "exercise/captures" / source_name
        if name == "incomplete-exporter":
            if not (directory / "INCOMPLETE").is_file() or (directory / "COMPLETE").exists():
                _fail("required capture write failure cannot be relabelled a complete capture")
            manifests[name] = {"complete": False, "source_capture": source_name, "files_digest": sha256_json(_inventory(directory))}
            continue
        capture = StreamingCaptureBundle.open(directory)
        manifest = capture.manifest
        manifests[name] = {"complete": True, "source_capture": source_name, "manifest_digest": manifest["manifest_digest"]}
        identities[name] = {key: manifest[key] for key in (
            "identity", "memory_plan", "memory_plan_digest", "sampling_spec", "sampling_spec_digest",
            "event_catalog", "event_envelope_version", "capabilities",
        )}
    return manifests, identities


def _benchmark_summaries(gpu: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    reports = {name: verify_completed_output(gpu / "benchmark" / name) for name in sorted(BENCHMARKS)}
    digests = {name: str(report["report_digest"]) for name, report in reports.items()}
    evidence_digest = _object(gpu / "profile-result.json")["evidence_digest"]
    common = {"schema_version": 2, "evidence_kind": "derived_from_qualified_raw_benchmarks",
              "performance_claim": "none", "report_digests": digests}
    # Both documents link to complete raw groups, not just selected wins.
    values = {name: {**common, "question": question,
                     "benchmarks": {key: {"raw_path": f"qualification/profiles/{GPU}/evidence/{evidence_digest}/benchmark/{key}",
                                          "measurement_contract": report["measurement_contract"],
                                          "groups": report["groups"], "correctness": report["correctness"]}
                                    for key, report in reports.items()}}
              for name, question in (("no-kv-vs-contiguous.json", "cache mode, query work, latency and memory"),
                                     ("telemetry-overhead.json", "off versus request capture within each cache mode"))}
    for value in values.values():
        value["report_digest"] = sha256_json(value)
    return digests, values


def _execution_identities(profiles: Mapping[str, Path], captures: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cpu_captures": dict(captures),
        "real_gpu_models": {dtype: _object(profiles[GPU] / "stateful" / dtype / "report/stateful-validation-report.json")["identity"]
                            for dtype in ("bf16", "fp16")},
        "runtime_profiles": {name: {key: _object(directory / "toolchain.json")[key]
                                    for key in ("manifest_digest", "runtime_profile_id", "runtime_lock_digest", "runtime_probe")}
                             for name, directory in sorted(profiles.items())},
    }


def verify_frozen_v02_spec(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^```yaml[ \t]*\r?\n(.*?)^```[ \t]*\r?$", text)
    if match is None:
        _fail("V0.2 release SPEC metadata is missing")
    axes: dict[str, str] = {}
    for line in match[1].splitlines():
        entry = re.fullmatch(r"([a-z_]+):[ \t]*([^#]*?)[ \t]*(?:#.*)?", line)
        if entry:
            if entry[1] in axes:
                _fail("V0.2 release SPEC metadata is duplicated")
            axes[entry[1]] = entry[2].strip()
    expected = {"status": "implemented", "spec_status": "implemented", "release_stage": "dev_snapshot",
                "qualification_tier": "unqualified", "target_release": TAG,
                "performance_outcome": "not_applicable", "lts_state": "not_applicable", "spec_version": "2"}
    if any(axes.get(key) != value for key, value in expected.items()):
        _fail("V0.2 release SPEC must be an implemented, unqualified source-frozen snapshot")
    profiles: dict[str, dict[str, str]] = {}
    lines = match[1].splitlines()
    for index, line in enumerate(lines):
        entry = re.fullmatch(r"  - profile_id:[ \t]*([a-z0-9-]+)[ \t]*", line)
        if entry is None:
            continue
        name = entry[1]
        if name in profiles:
            _fail("V0.2 release SPEC repeats a profile")
        fields: dict[str, str] = {}
        for field_line in lines[index + 1:]:
            field = re.fullmatch(r"    ([a-z_]+):[ \t]*([^#]*?)[ \t]*(?:#.*)?", field_line)
            if field is None:
                break
            if field[1] in fields:
                _fail("V0.2 release SPEC repeats a profile field")
            fields[field[1]] = field[2].strip()
        profiles[name] = fields
    if profiles != {name: {"result": "not_run", "reason": "final_source_bound_gate_pending"} for name in FORMAL_PROFILES}:
        _fail("V0.2 source SPEC profiles must remain explicit pre-gate facts")
    if (any(heading not in text for heading in ("## 12. Definition of Done", "### 12.1 Must", "### 12.2 Conditional"))
            or re.search(r"(?mi)^- \[x\]", text.split("## 12. Definition of Done", 1)[1])):
        _fail("V0.2 release SPEC cannot pre-claim final gate completion")


def _verify_documents(root: Path, primary: Path, state: Mapping[str, Any]) -> None:
    index = _object(root / "documentation-index.json")
    _signed(index, "documentation_digest")
    if (set(index) != {"schema_version", "documents", "documentation_digest"} or type(index["schema_version"]) is not int
            or index["schema_version"] != 2 or index["documentation_digest"] != state["documentation_digest"]
            or not isinstance(index["documents"], dict) or set(index["documents"]) != set(DOCUMENT_SOURCES)):
        _fail("V0.2 learning documentation index is incomplete")
    files = _object(primary / "source-provenance.json")["project_files"]
    for name, source in DOCUMENT_SOURCES.items():
        path = root / name
        digest = sha256_file(path)
        text = path.read_text(encoding="utf-8")
        if (index["documents"][name] != {"source_path": source, "sha256": digest} or files.get(source) != digest
                or not 40 <= len(text.strip()) or path.stat().st_size > 512 * 1024 or "\0" in text):
            _fail("V0.2 learning document is malformed or not bound to qualified source")
    verify_frozen_v02_spec(root / "spec.md")


def _verify_bundle(path: Path, expected: Mapping[str, str] | None) -> dict[str, Any]:
    from illm.stateful_rollback import verify_stateful_rollback

    path = _fs(path)
    if path.is_symlink() or path.parent.is_symlink() or path.name != "manifest.json":
        _fail("V0.2 release requires an ordinary manifest.json")
    root = path.parent.resolve()
    if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
        _fail("V0.2 release evidence bundle is incomplete", ErrorCode.CAPTURE_INCOMPLETE)
    _bounded(root)
    files = _inventory(root)
    if _hash_map(_object(root / "SHA256SUMS.json")) != {key: value for key, value in files.items() if key != "SHA256SUMS.json"}:
        _fail("V0.2 release checksum coverage or bytes differ")
    state = _state(path)
    if (root / "COMPLETE").read_text(encoding="ascii") != state["manifest_digest"] + "\n":
        _fail("V0.2 release completion marker is stale")
    source = _identity(state["source_identity"])
    if expected is not None and not _same(source, dict(expected)):
        _fail("V0.2 release belongs to different source", ErrorCode.TOOLCHAIN_MISMATCH)
    profiles: dict[str, Path] = {}
    for record in state["qualification_profiles"]:
        directory = _profile_root(root, record)
        verified, index = verify_stateful_profile_evidence(directory, expected_source_identity=source)
        if not _same(verified, record) or index["source_identity"] != source or not _same(_object(directory.parents[1] / "profile-result.json"), record):
            _fail("V0.2 release profile and manifest disagree")
        _final_package(directory)
        profiles[record["profile_id"]] = directory
    primary = profiles[PRIMARY]
    toolchain = _object(primary / "toolchain.json")
    if (sha256_file(root / "toolchain.json") != sha256_file(primary / "toolchain.json")
            or toolchain["manifest_digest"] != state["toolchain_manifest_digest"]):
        _fail("V0.2 release toolchain differs from qualified CPU profile")
    tests = _test_report(profiles)
    if not _same(_object(root / "test-report.json"), tests) or state["test_report_digest"] != tests["report_digest"]:
        _fail("V0.2 release tests do not recompute from qualified JUnit")
    capture_facts, capture_identities = _capture_facts(primary)
    identities = _execution_identities(profiles, capture_identities)
    if (not _same(state["capture_manifests"], capture_facts) or not _same(state["execution_identities"], identities)
            or state["metric_registry_digest"] != registry_digest()):
        _fail("V0.2 release execution/MemoryPlan/sampling/event identities differ")
    for name, source_name in CAPTURE_SOURCES.items():
        if _inventory(root / "capture" / name) != _inventory(primary / "exercise/captures" / source_name):
            _fail("V0.2 public capture differs from qualified lifecycle evidence")
    demo = StreamingCaptureBundle.open(primary / "demo-capture")
    if demo.manifest["manifest_digest"] != state["capture_manifest_digest"]:
        _fail("V0.2 primary demo capture identity is stale")
    digests, summaries = _benchmark_summaries(profiles[GPU])
    if state["benchmark_report_digests"] != digests:
        _fail("V0.2 release benchmark digests differ")
    for name, report in summaries.items():
        if not _same(_object(root / "benchmark" / name), report):
            _fail("V0.2 derived benchmark summary differs from complete raw groups")
    cancel = primary / "exercise/captures/cancel-after-launch"
    verify_streaming_learning_assets(cancel, root / "screenshots")
    if _inventory(root / "screenshots") != _inventory(primary / "learning"):
        _fail("V0.2 learning views differ from qualified deterministic assets")
    _verify_documents(root, primary, state)
    rollback = verify_stateful_rollback(root / "rollback")
    if (rollback.get("evidence_mode") != "release" or rollback.get("source_identity") != source
            or rollback.get("primary_v02_toolchain_manifest_digest") != toolchain["manifest_digest"]
            or rollback.get("capture_manifest_digest") != demo.manifest["manifest_digest"]
            or rollback.get("report_digest") != state["rollback_report_digest"]):
        _fail("V0.2 rollback is not bound to primary source/environment/capture")
    if not _same(_object(root / "qualification/release-state.json"), state):
        _fail("V0.2 release-state record is stale")
    required = {"manifest.json", "toolchain.json", "test-report.json", "documentation-index.json",
                "qualification/release-state.json", "COMPLETE", "SHA256SUMS.json", *DOCUMENT_SOURCES}
    for record in state["qualification_profiles"]:
        directory = profiles[record["profile_id"]]
        prefix = directory.relative_to(root).as_posix()
        required.update(f"{prefix}/{name}" for name in _inventory(directory))
        required.add(f"qualification/profiles/{record['profile_id']}/profile-result.json")
    for name in CAPTURE_SOURCES:
        required.update(f"capture/{name}/{link}" for link in _inventory(root / "capture" / name))
    required.update(f"benchmark/{name}" for name in summaries)
    for name in ("screenshots", "rollback"):
        required.update(f"{name}/{link}" for link in _inventory(root / name))
    if set(files) != required:
        _fail("V0.2 release contains missing or unreferenced artifacts")
    return state


def verify_stateful_release_bundle(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Verify a copied complete bundle without source checkout or tensor libraries."""
    try:
        return _verify_bundle(path, expected_source_identity)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "V0.2 release artifacts are malformed",
                        {"exception_type": type(exc).__name__}) from exc


def build_stateful_release_bundle(*, evidence_roots: Sequence[Path], rollback: Path, documentation_root: Path,
                                  output: Path, expected_source_identity: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Create a new verified candidate; never relax missing evidence or tag it."""
    from illm.stateful_rollback import portable_stateful_rollback_inventory, verify_stateful_rollback

    output = _fs(output)
    if output.exists() or output.is_symlink() or len(evidence_roots) != len(FORMAL_PROFILES):
        _fail("V0.2 assembly requires three sealed profiles and a new output")
    profiles: dict[str, tuple[dict[str, Any], Path]] = {}
    source: dict[str, str] | None = None
    for value in evidence_roots:
        directory = _fs(value)
        _bounded(directory)
        profile, index = verify_stateful_profile_evidence(directory, expected_source_identity=expected_source_identity)
        identity = _identity(index["source_identity"])
        if source is not None and source != identity:
            _fail("V0.2 profiles belong to mixed source snapshots", ErrorCode.TOOLCHAIN_MISMATCH)
        source = identity
        if profile["profile_id"] in profiles:
            _fail("V0.2 profile is duplicated")
        _final_package(directory)
        profiles[profile["profile_id"]] = (profile, directory)
    if set(profiles) != FORMAL_PROFILES or source is None:
        _fail("V0.2 assembly omits a required profile")
    rollback = _fs(rollback)
    rollback_report = verify_stateful_rollback(rollback)
    if rollback_report.get("evidence_mode") != "release" or rollback_report.get("source_identity") != source:
        _fail("V0.2 assembly requires source-bound real rollback evidence")
    documentation_root = _fs(documentation_root)
    _bounded(documentation_root)
    if set(_inventory(documentation_root)) != set(DOCUMENT_SOURCES):
        _fail("V0.2 documentation input must contain exactly five learning documents")
    temporary = output.resolve().with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        _fail("V0.2 release staging output already exists")
    temporary.mkdir(parents=True)
    (temporary / "INCOMPLETE").write_text("stateful_release_assembly_in_progress\n", encoding="ascii")
    try:
        copied: dict[str, Path] = {}
        for name, (record, directory) in sorted(profiles.items()):
            destination = _profile_root(temporary, record)
            shutil.copytree(directory, destination)
            _write(destination.parents[1] / "profile-result.json", record)
            copied[name] = destination
        primary = copied[PRIMARY]
        shutil.copy2(primary / "toolchain.json", temporary / "toolchain.json")
        for name, source_name in CAPTURE_SOURCES.items():
            shutil.copytree(primary / "exercise/captures" / source_name, temporary / "capture" / name)
        shutil.copytree(primary / "learning", temporary / "screenshots")
        digests, summaries = _benchmark_summaries(copied[GPU])
        for name, report in summaries.items():
            _write(temporary / "benchmark" / name, report)
        rollback_files = {
            **portable_stateful_rollback_inventory(rollback),
            "checksums.json": sha256_file(rollback / "checksums.json"),
            "COMPLETE": sha256_file(rollback / "COMPLETE"),
        }
        for name, digest in rollback_files.items():
            relative = _relative(name)
            target = temporary / "rollback" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rollback / relative, target)
            if sha256_file(target) != digest:
                _fail("V0.2 rollback evidence changed while copying")
        docs: dict[str, Any] = {}
        for name, original in DOCUMENT_SOURCES.items():
            shutil.copy2(documentation_root / name, temporary / name)
            docs[name] = {"source_path": original, "sha256": sha256_file(temporary / name)}
        documentation: dict[str, Any] = {"schema_version": 2, "documents": docs}
        documentation["documentation_digest"] = sha256_json(documentation)
        _write(temporary / "documentation-index.json", documentation)
        tests = _test_report(copied)
        _write(temporary / "test-report.json", tests)
        capture_facts, capture_identities = _capture_facts(primary)
        identities = _execution_identities(copied, capture_identities)
        manifest: dict[str, Any] = {
            "schema_version": 2, "report_kind": REPORT_KIND, "release_version": VERSION, "release_tag": TAG,
            "spec_status": "implemented", "release_stage": "release_candidate", "qualification_tier": "functionally_qualified",
            "qualification_profiles": [record for record, _ in (profiles[name] for name in sorted(profiles))],
            "performance_outcome": "not_applicable", "lts_state": "not_applicable", "source_identity": source,
            "primary_profile_id": PRIMARY, "toolchain_manifest_digest": _object(primary / "toolchain.json")["manifest_digest"],
            "capture_manifest_digest": _object(primary / "demo-capture/capture-manifest.json")["manifest_digest"],
            "capture_manifests": capture_facts, "execution_identities": identities, "metric_registry_digest": registry_digest(),
            "test_report_digest": tests["report_digest"], "benchmark_report_digests": digests,
            "rollback_report_digest": rollback_report["report_digest"], "documentation_digest": documentation["documentation_digest"],
        }
        manifest["manifest_digest"] = sha256_json(manifest)
        _write(temporary / "manifest.json", manifest)
        _write(temporary / "qualification/release-state.json", _state(temporary / "manifest.json"))
        (temporary / "INCOMPLETE").unlink()
        (temporary / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
        _write(temporary / "SHA256SUMS.json", _inventory(temporary))
        verified = verify_stateful_release_bundle(temporary / "manifest.json", expected_source_identity=source)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return verified
    except BaseException:
        (temporary / "INCOMPLETE").write_text("stateful_release_assembly_failed\n", encoding="ascii")
        raise

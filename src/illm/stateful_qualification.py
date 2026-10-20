"""Source-bound V0.2 profile sealing and tensor-free portable verification.

This module only seals or reads one profile.  It does not run the gate, build a
release bundle, promote a release, tag Git, or claim rollback completion.
Portable verification uses embedded relative links and never follows paths
recorded from the author machine.  Checksums provide integrity, not an
independent signature against an author who fabricates every input.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import metric_registry, registry_conformance, registry_digest, validate_metric_registry
from illm.qualification import parse_junit
from illm.reference_qualification import (
    BASE_LOCKS,
    BOOTSTRAP_FILES,
    PROJECT_FILES,
    RUNTIME_LOCKS,
    _assert_live_environment as _assert_reference_live_environment,
    _entries_digest,
    _fs,
    _hash_map,
    _inventory,
    _object,
    _relative,
    _signed,
    _source_provenance,
    _verify_pytest_transcript,
    _verify_toolchain,
    _write,
)
from illm.stateful_exercise import CYCLE_COUNT, SCOPE as EXERCISE_SCOPE, verify_exercise_output
from illm.stateful_gate_contract import (
    BENCHMARK_GROUPS,
    MODEL_PROFILES,
    STATEFUL_PROFILE_RUNTIME_IDS,
    phase_parameters,
)
from illm.stateful_validation import verify_stateful_output
from illm.streaming_learning import verify_streaming_learning_assets


FORMAL_PROFILE_RUNTIMES = dict(STATEFUL_PROFILE_RUNTIME_IDS)
FORMAL_PROFILES = frozenset(FORMAL_PROFILE_RUNTIMES)
CPU_PROFILES = frozenset({"state-machine-cpu-v1", "clean-room-release-v1"})
SYNTHETIC_PROFILE_ID = "synthetic-test-fixture-v02"
SOURCE_FIELDS = frozenset({"project_root_digest", "bootstrap_digest", "base_lock_digest"})
GROUPS = tuple(BENCHMARK_GROUPS)
REPORT_KIND = "illm-v02-profile-gate-v1"
PROFILE_FIELDS = frozenset({
    "schema_version", "profile_id", "simulated_profile_id", "evidence_mode", "result",
    "required_for_functional", "environment_manifest_digest", "evidence_digest", "components",
    "reason", "profile_digest",
})
INDEX_FIELDS = frozenset({
    "schema_version", "profile_id", "simulated_profile_id", "evidence_mode", "os", "arch",
    "python_version", "source_identity", "runtime_profile_id", "runtime_lock_digest", "files",
    "components", "evidence_digest", "index_digest",
})
_DIGEST = re.compile(r"[0-9a-f]{64}")
_V02_VERSION = re.compile(r"0\.2\.0(?:\.dev[0-9]+)?")
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")

CRITICAL_TESTS = frozenset({
    "test_real_tiny_full_scope_runs_50_new_cases_and_20_frozen_prompts",
    "test_real_tiny_lifecycle_runs_hundred_cycles_and_complete_fault_matrix",
    "test_fifty_cache_modes_match_sampling_and_committed_state",
    "test_twenty_v01_golden_sequences_remain_exact_at_engine_boundary",
    "test_cancel_after_launch_waits_for_fence_then_aborts_and_reclaims",
    "test_unknown_fence_fail_stops_and_quarantines_without_fake_terminal",
    "test_registry_fixture_digest_schema_and_conformance_are_exact",
    "test_png_svg_trace_are_exactly_derived_and_portable",
})


def _fail(message: str, code: ErrorCode = ErrorCode.INVALID_ARGUMENT) -> NoReturn:
    raise IllmError(code, message)


def _digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _same(left: object, right: object) -> bool:
    """Compare JSON values without treating bool as an integer."""
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _safe_relative(value: object) -> str:
    """Validate a portable evidence link without touching the linked path."""
    name = _relative(value)
    path = PurePosixPath(name)
    if path.as_posix() != name:
        _fail("V0.2 qualification link is not canonical")
    return name


def _check_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        _fail("V0.2 qualification root must be an ordinary directory")
    resolved = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(resolved):
            _fail("V0.2 qualification forbids links and escaped paths")
        if path.is_dir() and not any(path.iterdir()):
            _fail("V0.2 qualification forbids unreferenced empty directories")


def _check_gate_tree(root: Path) -> None:
    """Validate evidence entries while treating pytest basetemp as scratch.

    Pytest legitimately leaves empty directories and may create links while
    exercising rejection paths.  Its one predeclared top-level scratch root is
    therefore validated but never traversed.  Every evidence entry retains the
    stricter qualification-tree rules and is copied later from an allowlist.
    """

    if root.is_symlink() or not root.is_dir():
        _fail("V0.2 qualification root must be an ordinary directory")
    resolved = root.resolve()
    for path in root.iterdir():
        if path.name == "pytest-temp":
            if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(resolved):
                _fail("V0.2 gate pytest scratch must be an ordinary in-root directory")
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(resolved):
            _fail("V0.2 qualification forbids links and escaped paths")
        if path.is_dir():
            if not any(path.iterdir()):
                _fail("V0.2 qualification forbids unreferenced empty directories")
            _check_tree(path)


def stateful_source_identity(project_root: Path) -> dict[str, str]:
    """Recompute the source identity used by every V0.2 profile."""
    return cast(dict[str, str], _source_provenance(project_root)["source_identity"])


def _verify_v02_source(
    root: Path, expected: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read schema1 provenance with a V0.2-only package-version guard."""
    provenance = _object(root / "source-provenance.json")
    expected_fields = {
        "schema_version", "project_files", "base_lock_files", "project_name",
        "project_version", "source_identity",
    }
    if (
        set(provenance) != expected_fields
        or type(provenance.get("schema_version")) is not int
        or provenance.get("schema_version") != 1
    ):
        _fail("unsupported V0.2 source provenance schema", ErrorCode.UNSUPPORTED_SCHEMA)
    files = _hash_map(provenance["project_files"])
    base = _hash_map(provenance["base_lock_files"])
    if not set(PROJECT_FILES) <= set(files) or not set(BOOTSTRAP_FILES) <= set(files) or not set(BASE_LOCKS) < set(base):
        _fail("V0.2 source provenance omits required authority files")
    if any(name not in BASE_LOCKS and not name.startswith("vendor/wheels/") for name in base):
        _fail("V0.2 base lock inventory includes a non-authority artifact")
    identity = {
        "project_root_digest": _entries_digest(files),
        "bootstrap_digest": _entries_digest({name: files[name] for name in BOOTSTRAP_FILES}),
        "base_lock_digest": _entries_digest(base),
    }
    if provenance["source_identity"] != identity or (expected is not None and dict(expected) != identity):
        _fail("V0.2 qualification belongs to different source", ErrorCode.TOOLCHAIN_MISMATCH)
    versions: dict[str, str] = {}
    for name in BASE_LOCKS:
        path = root / "locks" / name
        if sha256_file(path) != base[name] or files[name] != base[name]:
            _fail("embedded V0.2 base lock differs from source provenance")
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line.strip())
            if match:
                package = re.sub(r"[-_.]+", "-", match[1]).lower()
                if package in versions and versions[package] != match[2]:
                    _fail("embedded V0.2 base lock pins conflict")
                versions[package] = match[2]
    version = provenance.get("project_version")
    if provenance.get("project_name") != "illm" or not isinstance(version, str) or _V02_VERSION.fullmatch(version) is None:
        _fail("qualification project package identity is not V0.2")
    project_path = root / "locks" / "pyproject.toml"
    if sha256_file(project_path) != files["pyproject.toml"]:
        _fail("embedded V0.2 project metadata differs from source provenance")
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))["project"]
    if project.get("name") != "illm" or project.get("version") != version:
        _fail("qualified V0.2 name/version differs from package metadata")
    versions["illm"] = version
    return identity, versions


def _verify_tests(root: Path, toolchain: Mapping[str, Any]) -> None:
    aggregate = parse_junit(root / "junit.xml")
    if (
        type(aggregate.get("tests")) is not int
        or cast(int, aggregate["tests"]) < len(CRITICAL_TESTS)
        or aggregate.get("failures") != 0
        or aggregate.get("errors") != 0
    ):
        _fail("V0.2 JUnit is not a passing full-suite report")
    suites = ET.parse(root / "junit.xml").getroot()
    passed = {
        case.attrib.get("name", "").split("[")[0]
        for case in suites.iter("testcase")
        if case.find("skipped") is None and case.find("failure") is None and case.find("error") is None
    }
    if not CRITICAL_TESTS <= passed:
        _fail("V0.2 JUnit omits required executed stateful regressions")
    quality = _object(root / "quality.json")
    checking = quality.get("type_check")
    provenance = _object(root / "source-provenance.json")
    source_files = cast(dict[str, Any], provenance["project_files"])
    count = sum(name.startswith("src/illm/") and name.endswith(".py") for name in source_files)
    packages = toolchain.get("installed_packages")
    if (
        type(quality.get("schema_version")) is not int
        or quality.get("schema_version") != 0
        or quality.get("status") != "pass"
        or quality.get("issues") != []
        or type(quality.get("checked_python_files")) is not int
        or quality.get("checked_python_files") != count
        or not isinstance(checking, dict)
        or checking.get("tool") != "mypy"
        or checking.get("status") != "pass"
        or type(checking.get("returncode")) is not int
        or checking.get("returncode") != 0
        or not isinstance(packages, dict)
        or checking.get("version") != packages.get("mypy")
    ):
        _fail("V0.2 static quality evidence is not exact and passing")


def _expected_gate_parameters(profile_id: str, runtime_id: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    if FORMAL_PROFILE_RUNTIMES.get(profile_id) != runtime_id:
        _fail("V0.2 gate profile/runtime mapping drifted", ErrorCode.TOOLCHAIN_MISMATCH)
    return tuple(phase_parameters(profile_id).items())


def _absolute_argv_path(value: str, label: str) -> str:
    """Normalize separators in an absolute recorded path without accessing it."""
    if not value or "\x00" in value or value != value.strip():
        _fail(f"V0.2 gate {label} path is not lexical and absolute")
    normalized = value.replace("\\", "/")
    if normalized.endswith("/"):
        _fail(f"V0.2 gate {label} path is not lexical and absolute")
    if normalized.startswith("//?/UNC/"):
        remainder = normalized.removeprefix("//?/UNC/")
    elif normalized.startswith("//?/") and re.match(r"^[A-Za-z]:/", normalized[4:]):
        remainder = normalized[7:]
    elif normalized.startswith("//"):
        remainder = normalized[2:]
    elif normalized.startswith("/"):
        remainder = normalized[1:]
    elif re.match(r"^[A-Za-z]:/", normalized):
        remainder = normalized[3:]
    else:
        _fail(f"V0.2 gate {label} path is not lexical and absolute")
    if not remainder or any(part in {"", ".", ".."} for part in remainder.split("/")):
        _fail(f"V0.2 gate {label} path aliases or escapes its root")
    return normalized


def _path_root_for_script(value: str, script: str) -> str:
    normalized = _absolute_argv_path(value, script)
    suffix = f"/scripts/{script}"
    if not normalized.endswith(suffix) or len(normalized) == len(suffix):
        _fail(f"V0.2 gate must execute the exact scripts/{script} entrypoint")
    return normalized[:-len(suffix)]


def _gate_artifact(root: str, relative: str) -> str:
    return f"{root}/{relative}"


def _exact_child_argv(actual: Sequence[str], expected: Sequence[str], path_indices: frozenset[int]) -> None:
    if len(actual) != len(expected):
        _fail("V0.2 gate child argv has extra, omitted, or reordered arguments")
    for index, (observed, wanted) in enumerate(zip(actual, expected, strict=True)):
        candidate = _absolute_argv_path(observed, "artifact") if index in path_indices else observed
        if candidate != wanted:
            _fail("V0.2 gate child argv does not exactly bind its declared phase and artifacts")


def _verify_bootstrap_argv(argv: Sequence[str], runtime_id: str) -> tuple[str, str, str]:
    """Return interpreter, project-root and gate-root lexical anchors."""
    if len(argv) not in {13, 15}:
        _fail("V0.2 bootstrap argv has extra or omitted arguments")
    interpreter = _absolute_argv_path(argv[0], "bootstrap interpreter")
    project_root = _path_root_for_script(argv[1], "bootstrap.py")
    fixed = (
        "--profile", argv[3], "--runtime-profile", runtime_id, "--locked", "--offline",
        "--require-new-environment", "--write-manifest",
    )
    if tuple(argv[2:10]) != fixed or _PROFILE_NAME.fullmatch(argv[3]) is None:
        _fail("V0.2 bootstrap argv is not the exact locked/offline/new-environment command")
    manifest = _absolute_argv_path(argv[10], "toolchain manifest")
    if argv[11] != "--write-manifest-digest":
        _fail("V0.2 bootstrap argv omits the exact manifest digest flag")
    sidecar = _absolute_argv_path(argv[12], "toolchain digest")
    suffix = "/toolchain.json"
    if not manifest.endswith(suffix) or len(manifest) == len(suffix):
        _fail("V0.2 bootstrap manifest is not a gate-root artifact")
    gate_root = manifest[:-len(suffix)]
    if sidecar != _gate_artifact(gate_root, "toolchain.sha256"):
        _fail("V0.2 bootstrap manifest and sidecar do not share one gate root")
    if len(argv) == 15:
        if argv[13] != "--artifact-cache":
            _fail("V0.2 bootstrap optional argv is not the artifact-cache pair")
        _absolute_argv_path(argv[14], "artifact cache")
    return interpreter, project_root, gate_root


def _verify_wrapped_argv(
    phase: str,
    argv: Sequence[str],
    parameters: Mapping[str, Any],
    *,
    interpreter: str,
    project_root: str,
    gate_root: str,
) -> None:
    if len(argv) < 8 or argv.count("--") != 1 or argv[6] != "--":
        _fail("V0.2 gate wrapper requires one exact in_env.py -- boundary")
    if _absolute_argv_path(argv[0], "phase interpreter") != interpreter:
        _fail("V0.2 gate phases do not use the bootstrap interpreter")
    if _path_root_for_script(argv[1], "in_env.py") != project_root:
        _fail("V0.2 gate wrapper is not the bootstrap source's in_env.py")
    if argv[2] != "--manifest" or argv[4] != "--expected-manifest-digest-file":
        _fail("V0.2 gate wrapper flags are not exact")
    if (
        _absolute_argv_path(argv[3], "toolchain manifest") != _gate_artifact(gate_root, "toolchain.json")
        or _absolute_argv_path(argv[5], "toolchain digest") != _gate_artifact(gate_root, "toolchain.sha256")
    ):
        _fail("V0.2 gate wrapper does not bind the bootstrap toolchain files")
    child = argv[7:]

    def artifact(relative: str) -> str:
        return _gate_artifact(gate_root, relative)

    expected: list[str]
    paths: frozenset[int]
    if phase == "quality":
        expected = ["python", "scripts/check_quality.py", "--output", artifact("quality.json")]
        paths = frozenset({3})
    elif phase == "pytest":
        expected = [
            "python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--basetemp",
            artifact("pytest-temp"), "--junitxml", artifact("junit.xml"), "--tb=short",
        ]
        paths = frozenset({7, 9})
    elif phase == "verify":
        expected = ["python", "-m", "illm", "verify", "--fixture", "tiny-decoder"]
        paths = frozenset()
    elif phase == "run":
        expected = [
            "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--prompt", "hello", "--max-new-tokens", "8", "--stream", "--request-id", "req-0001",
            "--kv-mode", "contiguous", "--telemetry", "request", "--capture", artifact("demo-capture"),
        ]
        paths = frozenset({20})
    elif phase == "replay-trace":
        expected = ["python", "-m", "illm", "replay", artifact("demo-capture"), "--mode", "trace"]
        paths = frozenset({4})
    elif phase == "inspect-request":
        expected = [
            "python", "-m", "illm", "inspect", "request", artifact("demo-capture"), "--request", "req-0001",
        ]
        paths = frozenset({5})
    elif phase == "stateful-full":
        expected = [
            "python", "scripts/validate_stateful.py", "--model", "fixture:tiny-decoder", "--device", "cpu",
            "--scope", "full", "--output", artifact("stateful-full"),
        ]
        paths = frozenset({9})
    elif phase in {"stateful-bf16", "stateful-fp16"}:
        dtype = phase.removeprefix("stateful-")
        expected = [
            "python", "scripts/validate_stateful.py", "--model", str(parameters["model_profile"]),
            "--device", "cuda:0", "--scope", "real-smoke", "--output", artifact(f"stateful-{dtype}"),
        ]
        paths = frozenset({9})
    elif phase == "exercise":
        expected = ["python", "scripts/exercise_streaming.py", "--output", artifact("exercise")]
        paths = frozenset({3})
    elif phase in {"learning-build", "learning-verify"}:
        expected = [
            "python", "scripts/render_streaming_learning.py", artifact("exercise/captures/cancel-after-launch"),
            "--output", artifact("learning"),
        ]
        if phase == "learning-verify":
            expected.append("--verify-existing")
        paths = frozenset({2, 4})
    elif phase.startswith("benchmark-"):
        name = phase.removeprefix("benchmark-")
        dtype, suite = name.split("-", 1)
        expected = [
            "python", "scripts/benchmark_streaming.py", "--model", MODEL_PROFILES[dtype],
            "--device", "cuda:0", "--target", "real", "--suite", suite, "--warmups", "3",
            "--repetitions", "7", "--output", artifact(f"benchmark-{dtype}-{suite}"),
        ]
        paths = frozenset({15})
    else:
        _fail("unknown V0.2 gate phase")
    _exact_child_argv(child, expected, paths)


def _json_objects(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            values.append(cast(dict[str, Any], value))
    return values


def _cancel_capture(root: Path) -> Path:
    captures = _object(root / "exercise" / "raw" / "captures.json")
    records = captures.get("records")
    if not isinstance(records, list):
        _fail("lifecycle capture index is missing")
    matches = [item for item in records if isinstance(item, dict) and item.get("capture_id") == "cancel-after-launch"]
    if len(matches) != 1:
        _fail("lifecycle evidence lacks one cancel-after-launch capture")
    return root / "exercise" / _safe_relative(matches[0].get("path"))


def _verify_run_transcript(objects: Sequence[Mapping[str, Any]], root: Path) -> None:
    bundle = StreamingCaptureBundle.open(root / "demo-capture")
    if not objects or objects[0].get("kind") != "metadata" or objects[-1].get("kind") != "terminal":
        _fail("V0.2 run transcript is not an ordered JSONL stream")
    if any(item.get("request_id") != "req-0001" for item in objects):
        _fail("V0.2 run transcript request identity drifted")
    tokens = [item for item in objects if item.get("kind") == "token"]
    if [item.get("output_index") for item in tokens] != list(range(len(tokens))):
        _fail("V0.2 run transcript token indices are not contiguous")
    terminal = objects[-1]
    result = terminal.get("result")
    if (
        not isinstance(result, dict)
        or result.get("capture_manifest_digest") != bundle.manifest["manifest_digest"]
        or result.get("outcome") != terminal.get("outcome")
        or result.get("output_token_ids") != [item.get("token_id") for item in tokens]
        or bundle.manifest.get("request_id") != "req-0001"
    ):
        _fail("V0.2 run transcript does not bind the complete demo capture")


def _verify_gate_phase_output(
    phase: str,
    stdout: Path,
    root: Path,
    toolchain: Mapping[str, Any],
    components: Mapping[str, Any],
) -> None:
    try:
        text = stdout.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "V0.2 gate stdout is unreadable") from exc
    if phase == "pytest":
        _verify_pytest_transcript(text, root / "junit.xml")
        return
    objects = _json_objects(text)
    if not objects:
        _fail("V0.2 gate phase lacks its structured command result")
    result = objects[-1]
    if phase == "bootstrap":
        valid = result == toolchain
    elif phase == "quality":
        valid = result == _object(root / "quality.json")
    elif phase == "verify":
        valid = result.get("status") == "pass" and result.get("device") == "cpu"
    elif phase == "run":
        _verify_run_transcript(objects, root)
        valid = True
    elif phase in {"replay-trace", "inspect-request"}:
        story = StreamingCaptureBundle.open(root / "demo-capture").request_story("req-0001")
        valid = result == story
    elif phase == "stateful-full":
        valid = result == components["stateful"]["tiny-full"]
    elif phase in {"stateful-bf16", "stateful-fp16"}:
        valid = result == components["stateful"][phase.removeprefix("stateful-")]
    elif phase == "exercise":
        valid = result == components["exercise"]
    elif phase in {"learning-build", "learning-verify"}:
        valid = result == components["learning"]
    elif phase.startswith("benchmark-"):
        name = phase.removeprefix("benchmark-")
        valid = result == components["benchmarks"][name]
    else:
        valid = False
    if not valid:
        _fail("V0.2 gate transcript does not bind its verified artifact")


def _verify_gate(
    directory: Path,
    profile_id: str,
    toolchain: Mapping[str, Any],
    identity: Mapping[str, str],
    components: Mapping[str, Any],
) -> None:
    report = _object(directory / "gate-report.json")
    _signed(report, "report_digest")
    expected_fields = {
        "schema_version", "report_kind", "profile_id", "runtime_profile_id", "status",
        "toolchain_manifest_digest", "source_identity", "fresh_environment", "offline_bootstrap",
        "network_used", "network_blocking", "network_policy", "commands", "report_digest",
    }
    if (
        set(report) != expected_fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 2
        or report.get("report_kind") != REPORT_KIND
        or report.get("profile_id") != profile_id
        or report.get("runtime_profile_id") != FORMAL_PROFILE_RUNTIMES[profile_id]
        or report.get("status") != "pass"
        or report.get("toolchain_manifest_digest") != toolchain.get("manifest_digest")
        or report.get("source_identity") != dict(identity)
        or report.get("fresh_environment") is not True
        or report.get("offline_bootstrap") is not True
        or report.get("network_used") != "not_observed"
        or report.get("network_blocking") != "not_enforced"
        or report.get("network_policy") != "offline_resolution_and_local_model_only"
    ):
        _fail("V0.2 gate report identity or fresh offline boundary is invalid")
    expected = _expected_gate_parameters(profile_id, FORMAL_PROFILE_RUNTIMES[profile_id])
    commands = report.get("commands")
    if not isinstance(commands, list) or len(commands) != len(expected):
        _fail("V0.2 gate command count is not the predeclared profile set")
    inventory = {"gate-report.json"}
    command_fields = {
        "phase", "argv", "parameters", "returncode", "stdout_file", "stderr_file",
        "stdout_sha256", "stderr_sha256",
    }
    interpreter: str | None = None
    project_root: str | None = None
    gate_root: str | None = None
    for command, (expected_phase, expected_parameters) in zip(commands, expected, strict=True):
        if (
            not isinstance(command, dict)
            or set(command) != command_fields
            or command.get("phase") != expected_phase
            or not _same(command.get("parameters"), expected_parameters)
            or type(command.get("returncode")) is not int
            or command.get("returncode") != 0
        ):
            _fail("V0.2 gate phase schema/order/parameters are invalid")
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
            _fail("V0.2 gate argv is invalid")
        string_argv = cast(list[str], argv)
        if expected_phase == "bootstrap":
            interpreter, project_root, gate_root = _verify_bootstrap_argv(
                string_argv, FORMAL_PROFILE_RUNTIMES[profile_id],
            )
        else:
            if interpreter is None or project_root is None or gate_root is None:
                _fail("V0.2 gate commands appear before their bootstrap authority")
            _verify_wrapped_argv(
                expected_phase,
                string_argv,
                expected_parameters,
                interpreter=interpreter,
                project_root=project_root,
                gate_root=gate_root,
            )
        for stream in ("stdout", "stderr"):
            link = _safe_relative(command.get(f"{stream}_file"))
            digest = command.get(f"{stream}_sha256")
            if link in inventory or not _digest(digest) or sha256_file(directory / link) != digest:
                _fail("V0.2 gate transcript link/checksum is invalid")
            inventory.add(link)
        _verify_gate_phase_output(expected_phase, directory / cast(str, command["stdout_file"]), root=directory.parent,
                                  toolchain=toolchain, components=components)
    if set(_inventory(directory)) != inventory:
        _fail("V0.2 gate transcript inventory is not exact")


def _bind_component(report: Mapping[str, Any], toolchain: Mapping[str, Any], identity: Mapping[str, str]) -> None:
    component_identity = report.get("identity")
    if not isinstance(component_identity, dict):
        _fail("V0.2 component identity is absent")
    source = component_identity.get("source")
    binding = component_identity.get("toolchain")
    if (
        not isinstance(source, dict)
        or source.get("project_root_digest") != identity["project_root_digest"]
        or source.get("bootstrap_digest") != identity["bootstrap_digest"]
        or not isinstance(binding, dict)
        or binding.get("binding") != "finalized-source-bound-manifest"
        or binding.get("formal_evidence_eligible") is not True
        or binding.get("manifest_digest") != toolchain.get("manifest_digest")
        or binding.get("runtime_profile_id") != toolchain.get("runtime_profile_id")
        or binding.get("runtime_lock_digest") != toolchain.get("runtime_lock_digest")
        or binding.get("lock_digest") != toolchain.get("lock_digest")
    ):
        _fail("V0.2 component is not bound to this source/toolchain")


def _bind_exercise(report: Mapping[str, Any], toolchain: Mapping[str, Any], identity: Mapping[str, str]) -> None:
    source = report.get("source_identity")
    binding = report.get("toolchain")
    if (
        not isinstance(source, dict)
        or source.get("project_root_digest") != identity["project_root_digest"]
        or source.get("bootstrap_digest") != identity["bootstrap_digest"]
        or not isinstance(binding, dict)
        or binding.get("binding") != "finalized-source-bound-manifest"
        or binding.get("formal_evidence_eligible") is not True
        or binding.get("manifest_digest") != toolchain.get("manifest_digest")
        or binding.get("runtime_profile_id") != toolchain.get("runtime_profile_id")
        or binding.get("runtime_lock_digest") != toolchain.get("runtime_lock_digest")
        or binding.get("lock_digest") != toolchain.get("lock_digest")
    ):
        _fail("V0.2 lifecycle evidence is not bound to this source/toolchain")


def _benchmark_reader(output: Path) -> dict[str, Any]:
    """Load the pure reader lazily; importing this module must never import torch."""
    try:
        module = importlib.import_module("illm.stateful_benchmark")
    except ImportError as exc:
        raise IllmError(ErrorCode.UNSUPPORTED_SCHEMA, "portable V0.2 benchmark reader is unavailable") from exc
    reader = getattr(module, "verify_completed_output", None)
    if not callable(reader):
        _fail("portable V0.2 benchmark reader has no verify_completed_output", ErrorCode.UNSUPPORTED_SCHEMA)
    value = reader(output)
    if not isinstance(value, dict):
        _fail("portable V0.2 benchmark reader returned a non-object")
    return cast(dict[str, Any], value)


def _verify_metrics(root: Path) -> dict[str, Any]:
    registry = _object(root / "metrics" / "metric-registry-v1.json")
    conformance = _object(root / "metrics" / "metric-registry-conformance.json")
    try:
        validate_metric_registry(registry)
    except ValueError as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "embedded MetricRegistryV1 differs from the frozen contract") from exc
    expected = registry_conformance(registry)
    if conformance != expected or conformance.get("registry_digest") != registry_digest():
        _fail("embedded MetricRegistryV1 conformance is invalid")
    return conformance


def _verify_components(
    root: Path, profile_id: str, toolchain: Mapping[str, Any], identity: Mapping[str, str],
) -> dict[str, Any]:
    conformance = _verify_metrics(root)
    components: dict[str, Any] = {"metric_registry": conformance}
    stateful_root = root / "stateful"
    if profile_id in CPU_PROFILES:
        if set(path.name for path in stateful_root.iterdir()) != {"tiny-full"}:
            _fail("CPU V0.2 profile requires exactly one tiny-full differential artifact")
        report = verify_stateful_output(stateful_root / "tiny-full", require_formal=True)
        _bind_component(report, toolchain, identity)
        hardware = report.get("hardware")
        if (
            report.get("target") != "tiny"
            or report.get("scope") != "full"
            or report.get("qualification_role") != "tiny-full-authority"
            or report.get("expected_counts") != {"differential": 50, "fixed_greedy": 20}
            or report.get("actual_counts") != {"differential": 50, "fixed_greedy": 20}
            or not isinstance(hardware, dict)
            or hardware.get("actual_device") != "cpu"
        ):
            _fail("CPU V0.2 profile cannot substitute smoke, real-model, or GPU evidence")
        exercise = verify_exercise_output(root / "exercise", require_formal=True)
        _bind_exercise(exercise, toolchain, identity)
        if (
            exercise.get("actual_cycle_count") != CYCLE_COUNT
            or exercise.get("required_matrix_complete") is not True
            or exercise.get("required_fault_pass_count") != len(exercise.get("required_fault_scenario_ids", []))
        ):
            _fail("CPU V0.2 profile lacks the full 100-cycle/fault matrix")
        demo = StreamingCaptureBundle.open(root / "demo-capture")
        if demo.manifest.get("schema_version") != 2 or demo.manifest.get("request_id") != "req-0001" or demo.manifest.get("capabilities") != ["R0"]:
            _fail("CPU V0.2 demo capture identity/capability is invalid")
        learning = verify_streaming_learning_assets(_cancel_capture(root), root / "learning")
        components.update({"stateful": {"tiny-full": report}, "exercise": exercise, "learning": learning,
                           "demo_capture_manifest_digest": demo.manifest["manifest_digest"]})
    else:
        expected_names = {"bf16", "fp16"}
        if set(path.name for path in stateful_root.iterdir()) != expected_names:
            _fail("GPU V0.2 profile requires exactly BF16 and FP16 real-smoke artifacts")
        reports: dict[str, Any] = {}
        for dtype in sorted(expected_names):
            report = verify_stateful_output(stateful_root / dtype, require_formal=True)
            _bind_component(report, toolchain, identity)
            component_identity = cast(dict[str, Any], report.get("identity", {}))
            hardware = report.get("hardware")
            if (
                report.get("target") != "real"
                or report.get("scope") != "real-smoke"
                or report.get("qualification_role") != "real-model-gpu-smoke"
                or report.get("expected_counts") != {"differential": 0, "fixed_greedy": 20}
                or report.get("actual_counts") != {"differential": 0, "fixed_greedy": 20}
                or component_identity.get("runtime_dtype") != dtype
                or component_identity.get("fixture_id") is not None
                or not isinstance(hardware, dict)
                or not str(hardware.get("actual_device", "")).startswith("cuda")
            ):
                _fail("GPU V0.2 profile requires actual real-model CUDA BF16 and FP16 smoke")
            reports[dtype] = report
        benchmark_root = root / "benchmark"
        expected_benchmarks = {f"{dtype}-{suite}" for dtype in expected_names for suite in ("tuning", "heldout")}
        if set(path.name for path in benchmark_root.iterdir()) != expected_benchmarks:
            _fail("GPU V0.2 profile requires four predeclared real benchmark artifacts")
        benchmarks: dict[str, Any] = {}
        for name in sorted(expected_benchmarks):
            dtype, suite = name.split("-", 1)
            report = _benchmark_reader(benchmark_root / name)
            _bind_component(report, toolchain, identity)
            component_identity = cast(dict[str, Any], report.get("identity", {}))
            model_identity = component_identity.get("model")
            hardware = report.get("hardware")
            if (
                report.get("target") != "real"
                or report.get("suite") != suite
                or report.get("evidence_scope") != "real-single-gpu-256x64"
                or not isinstance(model_identity, dict)
                or model_identity.get("runtime_dtype") != dtype
                or not isinstance(hardware, dict)
                or not str(hardware.get("device", "")).startswith("cuda")
                or set(cast(dict[str, Any], report.get("groups", {}))) != set(GROUPS)
            ):
                _fail("GPU V0.2 benchmark scope/dtype/groups cannot be substituted")
            contract = report.get("measurement_contract")
            correctness = report.get("correctness")
            if (
                not isinstance(contract, dict)
                or contract.get("groups") != list(GROUPS)
                or not isinstance(correctness, dict)
                or correctness.get("real_workload_completed_256x64") is not True
            ):
                _fail("GPU V0.2 benchmark is not the predeclared 256/64 workload")
            benchmarks[name] = report
        components.update({"stateful": reports, "benchmarks": benchmarks})
    return components


def _expected_top_level(profile_id: str, final: bool) -> set[str]:
    common = {"source-provenance.json", "toolchain.json", "junit.xml", "quality.json", "locks", "metrics", "stateful", "gate"}
    if profile_id in CPU_PROFILES:
        common |= {"exercise", "demo-capture", "learning"}
    else:
        common.add("benchmark")
    if final:
        common |= {"profile-result.json", "evidence-index.json", "SHA256SUMS.json", "COMPLETE"}
    else:
        common.add("INCOMPLETE")
    return common


def _verify_facts(
    root: Path,
    profile_id: str,
    expected_source_identity: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    if profile_id not in FORMAL_PROFILES:
        _fail("unknown V0.2 qualification profile")
    top = {path.name for path in root.iterdir()}
    final = "COMPLETE" in top
    if top != _expected_top_level(profile_id, final):
        _fail("V0.2 qualification contains missing or unreferenced top-level artifacts")
    if set(_inventory(root / "locks")) != {*BASE_LOCKS, "runtime-lock.json", "pyproject.toml"}:
        _fail("V0.2 qualification lock inventory is not exact")
    if set(_inventory(root / "metrics")) != {"metric-registry-v1.json", "metric-registry-conformance.json"}:
        _fail("V0.2 qualification metric inventory is not exact")
    identity, versions = _verify_v02_source(root, expected_source_identity)
    toolchain = _verify_toolchain(root, identity, versions)
    runtime_id = FORMAL_PROFILE_RUNTIMES[profile_id]
    if toolchain.get("runtime_profile_id") != runtime_id:
        _fail(f"{profile_id} requires exact runtime {runtime_id}", ErrorCode.TOOLCHAIN_MISMATCH)
    _verify_tests(root, toolchain)
    components = _verify_components(root, profile_id, toolchain, identity)
    _verify_gate(root / "gate", profile_id, toolchain, identity, components)
    return toolchain, identity, components


def _copy_tree(source: Path, destination: Path) -> None:
    _check_tree(source)
    shutil.copytree(source, destination)


def _copy_gate(source_report: Path, destination: Path) -> None:
    report = _object(source_report)
    destination.mkdir(parents=True)
    _write(destination / "gate-report.json", report)
    commands = report.get("commands")
    if not isinstance(commands, list):
        _fail("V0.2 gate report commands are absent")
    source_root = source_report.parent.resolve()
    copied = {"gate-report.json"}
    for command in commands:
        if not isinstance(command, dict):
            _fail("V0.2 gate command is not an object")
        for stream in ("stdout", "stderr"):
            link = _safe_relative(command.get(f"{stream}_file"))
            if link in copied:
                _fail("V0.2 gate transcript link is duplicated")
            copied.add(link)
            source = source_root / link
            if source.is_symlink() or not source.resolve().is_relative_to(source_root) or not source.is_file():
                _fail("V0.2 gate transcript escapes or is missing")
            target = destination / link
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _stateful_dtype(path: Path) -> str:
    report = verify_stateful_output(path, require_formal=True)
    identity = report.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("runtime_dtype"), str):
        _fail("stateful validation runtime dtype is absent")
    return cast(str, identity["runtime_dtype"])


def _benchmark_name(path: Path) -> str:
    report = _benchmark_reader(path)
    identity = report.get("identity")
    model = identity.get("model") if isinstance(identity, dict) else None
    dtype = model.get("runtime_dtype") if isinstance(model, dict) else None
    suite = report.get("suite")
    if dtype not in {"bf16", "fp16"} or suite not in {"tuning", "heldout"}:
        _fail("benchmark dtype/suite identity is invalid")
    return f"{dtype}-{suite}"


def create_stateful_profile_evidence(
    *,
    profile_id: str,
    gate_root: Path,
    output: Path,
    expected_source_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate and atomically seal one real V0.2 profile; never promote it."""
    if profile_id not in FORMAL_PROFILES:
        _fail("unknown V0.2 qualification profile")
    gate_root = _fs(gate_root)
    _check_gate_tree(gate_root)
    gate_root = gate_root.resolve()
    cpu = profile_id in CPU_PROFILES
    expected_gate_entries = {
        "toolchain.json", "toolchain.sha256", "gate-report.json", "quality.json", "junit.xml", "logs",
        "INCOMPLETE", "execution.json",
    }
    expected_gate_entries |= ({"demo-capture", "stateful-full", "exercise", "learning"} if cpu else {
        "stateful-bf16", "stateful-fp16", "benchmark-bf16-tuning", "benchmark-bf16-heldout",
        "benchmark-fp16-tuning", "benchmark-fp16-heldout",
    })
    actual_gate_entries = {path.name for path in gate_root.iterdir()}
    if actual_gate_entries != expected_gate_entries | {"pytest-temp"}:
        _fail("V0.2 gate root inventory is not the exact predeclared profile output")
    gate_report_value = _object(gate_root / "gate-report.json")
    execution = _object(gate_root / "execution.json")
    incomplete = _object(gate_root / "INCOMPLETE")
    if (
        execution != {
            "schema_version": 2,
            "profile_id": profile_id,
            "commands": gate_report_value.get("commands"),
        }
        or incomplete != {"reason": "gate_in_progress", "profile_id": profile_id}
    ):
        _fail("V0.2 gate control records do not bind the completed phase set")
    toolchain_manifest = gate_root / "toolchain.json"
    try:
        manifest_digest_file = (gate_root / "toolchain.sha256").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "V0.2 gate toolchain digest file is unreadable") from exc
    gate_toolchain = _object(toolchain_manifest)
    if (
        not _digest(manifest_digest_file)
        or gate_toolchain.get("manifest_digest") != manifest_digest_file
        or manifest_digest_file != sha256_json({key: value for key, value in gate_toolchain.items() if key != "manifest_digest"})
    ):
        _fail("V0.2 gate toolchain file digest does not match", ErrorCode.TOOLCHAIN_MISMATCH)
    output = _fs(output)
    if output.exists() or output.is_symlink():
        _fail("V0.2 qualification output already exists and cannot be overwritten")
    output = output.resolve()
    toolchain = gate_toolchain
    project_root = Path(str(toolchain.get("project_root", ""))).resolve()
    provenance = _assert_reference_live_environment(toolchain, project_root)
    version = provenance.get("project_version")
    if not isinstance(version, str) or _V02_VERSION.fullmatch(version) is None:
        _fail("live qualification source is not V0.2", ErrorCode.TOOLCHAIN_MISMATCH)
    runtime_id = FORMAL_PROFILE_RUNTIMES[profile_id]
    if toolchain.get("runtime_profile_id") != runtime_id or toolchain.get("runtime_lock_digest") != RUNTIME_LOCKS[runtime_id]:
        _fail("live qualification runtime does not match the exact V0.2 profile", ErrorCode.TOOLCHAIN_MISMATCH)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        _fail("V0.2 qualification staging output already exists")
    temporary.mkdir(parents=True)
    (temporary / "INCOMPLETE").write_text("stateful_qualification_in_progress\n", encoding="ascii")
    try:
        for source, relative in (
            (toolchain_manifest, "toolchain.json"),
            (gate_root / "junit.xml", "junit.xml"),
            (gate_root / "quality.json", "quality.json"),
        ):
            target = temporary / relative
            shutil.copy2(source, target)
        _write(temporary / "source-provenance.json", provenance)
        locks = temporary / "locks"
        locks.mkdir()
        for name in BASE_LOCKS:
            shutil.copy2(project_root / name, locks / name)
        shutil.copy2(project_root / "pyproject.toml", locks / "pyproject.toml")
        shutil.copy2(project_root / "runtime" / "artifacts" / f"{runtime_id}.json", locks / "runtime-lock.json")
        registry = _object(project_root / "fixtures" / "metrics" / "metric-registry-v1.json")
        try:
            validate_metric_registry(registry)
        except ValueError as exc:
            raise IllmError(ErrorCode.INVALID_ARGUMENT, "metric registry input is not frozen V1") from exc
        _write(temporary / "metrics" / "metric-registry-v1.json", registry)
        _write(temporary / "metrics" / "metric-registry-conformance.json", registry_conformance(registry))
        if cpu:
            _copy_tree(gate_root / "stateful-full", temporary / "stateful" / "tiny-full")
            _copy_tree(gate_root / "exercise", temporary / "exercise")
            _copy_tree(gate_root / "demo-capture", temporary / "demo-capture")
            _copy_tree(gate_root / "learning", temporary / "learning")
        else:
            for dtype in ("bf16", "fp16"):
                source = gate_root / f"stateful-{dtype}"
                if _stateful_dtype(source) != dtype:
                    _fail("GPU stateful gate artifact dtype is mislabeled")
                _copy_tree(source, temporary / "stateful" / dtype)
            for dtype in ("bf16", "fp16"):
                for suite in ("tuning", "heldout"):
                    name = f"{dtype}-{suite}"
                    source = gate_root / f"benchmark-{name}"
                    if _benchmark_name(source) != name:
                        _fail("GPU benchmark gate artifact dtype/suite is mislabeled")
                    _copy_tree(source, temporary / "benchmark" / name)
        _copy_gate(gate_root / "gate-report.json", temporary / "gate")
        toolchain_verified, identity, components = _verify_facts(temporary, profile_id, expected_source_identity)
        (temporary / "INCOMPLETE").unlink()
        facts = _inventory(temporary)
        component_digests = _component_digests(components)
        evidence_digest = sha256_json({
            "schema_version": 2, "profile_id": profile_id, "source_identity": identity,
            "components": component_digests, "files": facts,
        })
        profile: dict[str, Any] = {
            "schema_version": 2, "profile_id": profile_id, "simulated_profile_id": None,
            "evidence_mode": "formal", "result": "pass", "required_for_functional": True,
            "environment_manifest_digest": toolchain_verified["manifest_digest"],
            "evidence_digest": evidence_digest, "components": component_digests, "reason": None,
        }
        profile["profile_digest"] = sha256_json(profile)
        index: dict[str, Any] = {
            "schema_version": 2, "profile_id": profile_id, "simulated_profile_id": None,
            "evidence_mode": "formal", "os": toolchain_verified["os"], "arch": toolchain_verified["arch"],
            "python_version": toolchain_verified["python_version"], "source_identity": identity,
            "runtime_profile_id": toolchain_verified["runtime_profile_id"],
            "runtime_lock_digest": toolchain_verified["runtime_lock_digest"], "files": facts,
            "components": component_digests, "evidence_digest": evidence_digest,
        }
        index["index_digest"] = sha256_json(index)
        _write(temporary / "profile-result.json", profile)
        _write(temporary / "evidence-index.json", index)
        (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
        _write(temporary / "SHA256SUMS.json", _inventory(temporary))
        verify_stateful_profile_evidence(temporary, expected_source_identity=expected_source_identity)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return profile
    except Exception:
        (temporary / "INCOMPLETE").write_text("stateful_qualification_failed\n", encoding="ascii")
        raise


def _component_digests(components: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in sorted(components.items()):
        if isinstance(value, dict) and value and all(isinstance(item, dict) for item in value.values()):
            result[name] = {key: sha256_json(item) for key, item in sorted(cast(dict[str, Any], value).items())}
        elif isinstance(value, dict):
            result[name] = sha256_json(value)
        elif isinstance(value, str) and _digest(value):
            result[name] = value
        else:
            _fail("V0.2 component digest projection is invalid")
    return result


def _verify_stateful_profile_evidence(
    root: Path,
    expected_source_identity: Mapping[str, str] | None,
    *,
    allow_synthetic_test_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Internal reader; the synthetic switch exists only for unit fixtures."""
    root = _fs(root)
    if root.is_symlink():
        _fail("V0.2 qualification root cannot be a symbolic link")
    root = root.resolve()
    if (root / "INCOMPLETE").exists() or not (root / "COMPLETE").is_file():
        _fail("V0.2 qualification evidence is incomplete", ErrorCode.CAPTURE_INCOMPLETE)
    _check_tree(root)
    files = _inventory(root)
    checksums = _hash_map(_object(root / "SHA256SUMS.json"))
    if checksums != {name: digest for name, digest in files.items() if name != "SHA256SUMS.json"}:
        _fail("V0.2 qualification checksum coverage/match is invalid")
    profile = _object(root / "profile-result.json")
    index = _object(root / "evidence-index.json")
    if (
        set(profile) != PROFILE_FIELDS
        or set(index) != INDEX_FIELDS
        or type(profile.get("schema_version")) is not int
        or type(index.get("schema_version")) is not int
        or profile.get("schema_version") != 2
        or index.get("schema_version") != 2
    ):
        _fail("unsupported V0.2 qualification schema", ErrorCode.UNSUPPORTED_SCHEMA)
    _signed(profile, "profile_digest")
    _signed(index, "index_digest")
    profile_id = profile.get("profile_id")
    evidence_mode = profile.get("evidence_mode")
    simulated = profile.get("simulated_profile_id")
    if profile_id in FORMAL_PROFILES:
        if (
            evidence_mode != "formal"
            or simulated is not None
            or profile.get("result") != "pass"
            or profile.get("required_for_functional") is not True
            or profile.get("reason") is not None
        ):
            _fail("V0.2 formal qualification is not a required complete pass")
        effective_profile = cast(str, profile_id)
    elif (
        allow_synthetic_test_fixture
        and profile_id == SYNTHETIC_PROFILE_ID
        and evidence_mode == "synthetic_test_fixture"
        and simulated in FORMAL_PROFILES
        and profile.get("result") == "synthetic_pass"
        and profile.get("required_for_functional") is False
        and profile.get("reason") == "unit_test_only_never_release_eligible"
    ):
        effective_profile = cast(str, simulated)
    else:
        _fail("unknown or non-release V0.2 qualification profile")
    if (
        index.get("profile_id") != profile_id
        or index.get("simulated_profile_id") != simulated
        or index.get("evidence_mode") != evidence_mode
    ):
        _fail("V0.2 profile and index identity differ")
    facts = {
        name: digest for name, digest in files.items()
        if name not in {"SHA256SUMS.json", "COMPLETE", "profile-result.json", "evidence-index.json"}
    }
    if _hash_map(index.get("files")) != facts:
        _fail("V0.2 qualification index does not cover exact evidence facts")
    toolchain, identity, components = _verify_facts(root, effective_profile, expected_source_identity)
    component_digests = _component_digests(components)
    evidence_digest = sha256_json({
        "schema_version": 2, "profile_id": profile_id, "source_identity": identity,
        "components": component_digests, "files": facts,
    })
    if (
        index.get("source_identity") != identity
        or index.get("components") != component_digests
        or profile.get("components") != component_digests
        or index.get("evidence_digest") != evidence_digest
        or profile.get("evidence_digest") != evidence_digest
        or profile.get("environment_manifest_digest") != toolchain.get("manifest_digest")
        or any(index.get(name) != toolchain.get(name) for name in (
            "os", "arch", "python_version", "runtime_profile_id", "runtime_lock_digest",
        ))
    ):
        _fail("V0.2 qualification index/profile does not bind verified provenance")
    return profile, index


def verify_stateful_profile_evidence(
    root: Path,
    *,
    expected_source_identity: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Portable fail-closed reader for one formal V0.2 profile."""
    try:
        return _verify_stateful_profile_evidence(root, expected_source_identity)
    except IllmError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, StopIteration,
            OverflowError, RecursionError, ET.ParseError) as exc:
        raise IllmError(
            ErrorCode.INVALID_ARGUMENT,
            "V0.2 qualification artifacts are malformed",
            {"exception_type": type(exc).__name__},
        ) from exc


__all__ = [
    "FORMAL_PROFILE_RUNTIMES",
    "FORMAL_PROFILES",
    "create_stateful_profile_evidence",
    "stateful_source_identity",
    "verify_stateful_profile_evidence",
]

"""Portable verification of the approved CPU development-driver component.

This reader is intentionally limited: a valid component proves that the
source-bound ``inputs -> cpu -> verify`` development driver ran its one full
CPU suite and that its sealed bytes remain internally consistent.  It does not
turn ``DEVELOPMENT_ONLY`` evidence into numerical, GPU, clean-room, rollback,
release, or LTS qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.stateful_policy_gate_contract import (
    APPROVAL_SHA256,
    CANDIDATE_POLICY_DIGESTS,
    CANDIDATE_POLICY_FAMILY,
    INPUT_BYTES_SHA256,
    INPUT_MANIFEST_DIGEST,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)


COMPONENT_KIND = "illm-policy-cpu-development-component-v1"
APPROVED_COMMIT = "04144ab9ddb8d86c482ad6db8a618bded1f1e7aa"
CPU_STAGES = ("inputs", "cpu", "verify")
STAGE_DIRECTORIES = {
    "inputs": "policy-inputs",
    "cpu": "policy-cpu",
    "verify": "policy-verified",
}
PENDING_WORK = (
    "full raw-tensor numerical qualification and final capture3 lifecycle/portable profile integration",
    "formal portable release qualification",
    "all GPU pilot, correctness, benchmark, lifecycle, and rollback evidence",
)
INPUT_SUITE_DIGESTS = (
    "57163407317f5e9f9f0cc26931c364718b299a79e3ec77cb1c7da4bac52a6f7a",
    "659337351c7fd93ce2ca101a76908bf537db586ded85e89ac41e45072e97f9cb",
)
INPUT_CASE_COUNTS = (20, 50)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOOLCHAIN_FIELDS = frozenset({
    "arch", "artifact_cache", "bootstrap_digest", "entrypoints", "environment_id",
    "environment_root", "finalized", "installed_packages", "lock_digest", "manifest_digest",
    "os", "profile", "project_root", "project_root_digest", "python_executable",
    "python_version", "runtime_artifacts", "runtime_lock_digest", "runtime_probe",
    "runtime_profile_id", "schema_version",
})
_CANDIDATE_FIELDS = frozenset({
    "schema_version", "family", "version", "storage_dtype", "weights", "linear",
    "rms_norm", "rope", "pointwise", "qk", "softmax", "pv", "gqa", "layout",
    "full_schedule", "cached_schedule", "flags",
})
_RESULT_FIELDS = frozenset({
    "schema_version", "request_id", "input_token_ids", "output_token_ids", "text",
    "outcome", "finish_reason", "error_code", "actual_seed", "model_spec_id",
    "sampling_spec_digest", "kv_mode", "next_output_position", "rng_draws_consumed",
    "committed_draw_addresses", "query_tokens_per_step", "kv_valid_tokens_per_step",
    "resource_status", "timings", "capture_manifest_digest", "capture_error", "event_count",
    "numerical_policy", "numerical_policy_digest", "execution_binding",
    "execution_binding_digest", "memory_plan_digest",
})
_RESOURCE_FIELDS = frozenset({
    "active_requests", "active_tickets", "input_leases", "kv_reserved_bytes",
    "kv_staged_tokens", "kv_valid_tokens", "kv_write_leases", "output_leases",
    "quarantined_requests", "reservations", "workspace_leases",
})
_TIMING_FIELDS = frozenset({
    "accepted_ns", "clock_domain", "critical_capture_event_drop_count", "evidence_kind",
    "per_forward_wall_time_ns", "server_e2e_wall_time_ns", "telemetry", "terminal_ready_ns",
    "token_ready_ns", "tokenization_wall_time_ns",
})
_BINDING_FIELDS = frozenset({
    "schema_version", "implementation_source_scope", "implementation_source_digest",
    "environment", "environment_digest", "toolchain", "protocol_sha256",
})
_BINDING_ENVIRONMENT_FIELDS = frozenset({
    "backend", "python_version", "platform", "machine", "torch_version",
    "cuda_arithmetic", "cuda_execution_authorized",
})
_STAGE_FILES = {
    "inputs": frozenset({
        "candidate-policies/bf16.json", "candidate-policies/fp16.json",
        "candidate-policies/fp32.json", "inputs-report.json", "policy-inputs.json",
    }),
    "cpu": frozenset({
        "cpu-report.json", "execution.json", "junit.xml", "quality.json",
        "logs/quality.stdout.txt", "logs/quality.stderr.txt",
        "logs/pytest.stdout.txt", "logs/pytest.stderr.txt",
        "logs/example.stdout.txt", "logs/example.stderr.txt",
    }),
    "verify": frozenset({"verification-report.json"}),
}
_STAGE_DIRECTORIES = {
    "inputs": frozenset({"candidate-policies"}),
    "cpu": frozenset({"logs"}),
    "verify": frozenset(),
}


class PolicyCPUComponentError(ValueError):
    """The embedded development component is corrupt or semantically invalid."""


def _fail(message: str) -> NoReturn:
    raise PolicyCPUComponentError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyCPUComponentError("value is not canonical finite JSON") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PolicyCPUComponentError(f"could not read evidence file: {path.name}") from exc
    return digest.hexdigest()


def _is_linklike(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PolicyCPUComponentError(f"could not inspect evidence path: {path.name}") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _plain_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if _is_linklike(absolute) or not absolute.is_dir():
        _fail(f"{label} must be an ordinary directory")
    return absolute


def _plain_file(path: Path, *, label: str) -> Path:
    if _is_linklike(path) or not path.is_file():
        _fail(f"{label} must be an ordinary file")
    try:
        if path.stat().st_nlink != 1:
            _fail(f"{label} cannot be hard-linked")
    except OSError as exc:
        raise PolicyCPUComponentError(f"could not inspect {label}") from exc
    return path


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON constant: {value}")


def _decode_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyCPUComponentError(f"{label} is not strict UTF-8 JSON") from exc


def _read_json(path: Path, *, label: str, canonical_newline: bool = True) -> dict[str, Any]:
    checked = _plain_file(path, label=label)
    try:
        raw = checked.read_bytes()
    except OSError as exc:
        raise PolicyCPUComponentError(f"could not read {label}") from exc
    value = _decode_json(raw, label=label)
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    expected = _canonical_bytes(value) + (b"\n" if canonical_newline else b"")
    if raw != expected:
        _fail(f"{label} is not in its canonical byte encoding")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be one lowercase SHA-256 digest")
    return value


def _uuid4(value: object, *, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be a canonical UUID4 string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise PolicyCPUComponentError(f"{label} must be a canonical UUID4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        _fail(f"{label} must be a canonical lowercase UUID4 string")
    return value


def _safe_gate_relative(value: str) -> str:
    if type(value) is not str or not value.startswith("out/") or "\\" in value or ":" in value:
        _fail("gate_relative must be a canonical path below out/")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        _fail("gate_relative must be a canonical path below out/")
    if len(path.parts) < 2:
        _fail("gate_relative cannot name the out root itself")
    return value


def _portable_toolchain(toolchain: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "environment_id": toolchain["environment_id"],
        "lock_digest": toolchain["lock_digest"],
        "manifest_digest": toolchain["manifest_digest"],
        "project_root_digest": toolchain["project_root_digest"],
        "python_version": toolchain["python_version"],
        "runtime_profile_id": toolchain["runtime_profile_id"],
    }


def _validate_toolchain(root: Path, expected: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    embedded = _read_json(root / "toolchain.json", label="embedded toolchain manifest")
    try:
        expected_copy = cast(dict[str, Any], _decode_json(_canonical_bytes(expected), label="expected toolchain"))
    except PolicyCPUComponentError:
        raise
    if set(embedded) != _TOOLCHAIN_FIELDS or set(expected_copy) != _TOOLCHAIN_FIELDS:
        _fail("toolchain manifest fields differ from the closed schema")
    if _canonical_bytes(embedded) != _canonical_bytes(expected_copy):
        _fail("embedded toolchain differs from the caller-verified toolchain")
    projection = {key: value for key, value in embedded.items() if key != "manifest_digest"}
    manifest_digest = _digest(embedded.get("manifest_digest"), label="toolchain manifest_digest")
    if manifest_digest != _sha_bytes(_canonical_bytes(projection)):
        _fail("toolchain manifest digest does not bind its closed fields")
    if type(embedded.get("schema_version")) is not int or embedded["schema_version"] != 1:
        _fail("toolchain manifest schema version differs")
    if embedded.get("finalized") is not True:
        _fail("toolchain manifest is not finalized")
    _uuid4(embedded.get("environment_id"), label="toolchain environment_id")
    for name in ("bootstrap_digest", "lock_digest", "project_root_digest", "runtime_lock_digest"):
        _digest(embedded.get(name), label=f"toolchain {name}")
    for name in ("python_version", "runtime_profile_id", "os", "arch"):
        if type(embedded.get(name)) is not str or not embedded[name]:
            _fail(f"toolchain {name} must be a nonempty string")
    portable = _portable_toolchain(embedded)
    if set(portable) != {
        "environment_id", "lock_digest", "manifest_digest", "project_root_digest",
        "python_version", "runtime_profile_id",
    }:
        _fail("portable toolchain projection differs")
    return embedded, portable


def _walk(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise PolicyCPUComponentError("evidence directory is unreadable") from exc
        for child in children:
            if _is_linklike(child):
                _fail("evidence tree cannot contain links or reparse points")
            relative = prefix / child.name
            relative_name = relative.as_posix()
            if child.is_dir():
                directories.add(relative_name)
                visit(child, relative)
            elif child.is_file():
                _plain_file(child, label=relative_name)
                files.add(relative_name)
            else:
                _fail("evidence tree contains a non-file member")

    visit(root, PurePosixPath())
    return files, directories


def _stage_inventory(stage_root: Path, stage: str) -> dict[str, Any]:
    checked = _plain_directory(stage_root, label=f"{stage} stage")
    files, directories = _walk(checked)
    expected_files = set(_STAGE_FILES[stage]) | {"checksums.json", "COMPLETE"}
    if files != expected_files or directories != set(_STAGE_DIRECTORIES[stage]):
        _fail(f"{stage} stage inventory differs from the closed component schema")
    if (checked / "INCOMPLETE").exists():
        _fail(f"{stage} stage is incomplete")

    entries = [
        {
            "path": relative,
            "sha256": _sha_file(checked / Path(relative)),
            "size_bytes": (checked / Path(relative)).stat().st_size,
        }
        for relative in sorted(_STAGE_FILES[stage])
    ]
    expected = {
        "schema_version": 1,
        "kind": "illm-policy-stage-checksums-v1",
        "stage": stage,
        "files": entries,
        "files_digest": _sha_bytes(_canonical_bytes(entries)),
    }
    reported = _read_json(checked / "checksums.json", label=f"{stage} checksums")
    if _canonical_bytes(reported) != _canonical_bytes(expected):
        _fail(f"{stage} checksums do not cover the exact bytes")
    marker = _read_json(checked / "COMPLETE", label=f"{stage} completion marker")
    if marker != {
        "kind": "illm-policy-stage-complete-v1",
        "stage": stage,
        "files_digest": expected["files_digest"],
    }:
        _fail(f"{stage} completion marker differs")
    return expected


def _fixed_policy_digests() -> dict[str, str]:
    return dict(CANDIDATE_POLICY_DIGESTS)


def _fixed_protocol() -> dict[str, str]:
    return {"protocol_id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256}


def _verify_candidate_descriptor(path: Path, dtype: str, expected_digest: str) -> None:
    checked = _plain_file(path, label=f"{dtype} candidate descriptor")
    try:
        raw = checked.read_bytes()
    except OSError as exc:
        raise PolicyCPUComponentError("candidate descriptor is unreadable") from exc
    if _sha_bytes(raw) != expected_digest:
        _fail(f"{dtype} candidate descriptor bytes differ from the registered digest")
    value = _decode_json(raw, label=f"{dtype} candidate descriptor")
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        _fail(f"{dtype} candidate descriptor is not canonical JSON")
    if (
        set(value) != _CANDIDATE_FIELDS
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or type(value.get("version")) is not int
        or value.get("version") != 1
        or value.get("family") != CANDIDATE_POLICY_FAMILY
        or value.get("storage_dtype") != dtype
        or not isinstance(value.get("flags"), dict)
    ):
        _fail(f"{dtype} candidate descriptor fields differ from the closed policy schema")


def _verify_inputs(root: Path, portable: Mapping[str, Any]) -> dict[str, Any]:
    if _sha_file(root / "policy-inputs.json") != INPUT_BYTES_SHA256:
        _fail("policy input snapshot is not byte-identical to the approved authority")
    for dtype, digest in CANDIDATE_POLICY_DIGESTS:
        _verify_candidate_descriptor(root / "candidate-policies" / f"{dtype}.json", dtype, digest)

    report = _read_json(root / "inputs-report.json", label="inputs report")
    expected_fields = {
        "schema_version", "kind", "status", "evidence_mode", "qualification_tier",
        "qualification_result", "approved_commit", "protocol", "input",
        "candidate_policy_digests", "toolchain", "limitations",
    }
    input_value = report.get("input")
    if (
        set(report) != expected_fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "illm-policy-inputs-development-snapshot-v1"
        or report.get("status") != "pass"
        or report.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or report.get("qualification_tier") != "unqualified"
        or report.get("qualification_result") != "not_evaluated"
        or report.get("approved_commit") != APPROVED_COMMIT
        or report.get("protocol") != _fixed_protocol()
        or report.get("candidate_policy_digests") != _fixed_policy_digests()
        or report.get("toolchain") != portable
        or report.get("limitations") != list(PENDING_WORK)
        or not isinstance(input_value, dict)
        or set(input_value) != {
            "raw_bytes_sha256", "manifest_digest", "suite_digests", "case_counts", "snapshot",
        }
        or input_value.get("raw_bytes_sha256") != INPUT_BYTES_SHA256
        or input_value.get("manifest_digest") != INPUT_MANIFEST_DIGEST
        or input_value.get("suite_digests") != list(INPUT_SUITE_DIGESTS)
        or input_value.get("case_counts") != list(INPUT_CASE_COUNTS)
        or input_value.get("snapshot") != "policy-inputs.json"
    ):
        _fail("inputs report differs from the approved closed development snapshot")
    return report


def _verify_quality(path: Path) -> dict[str, Any]:
    quality = _read_json(path, label="quality report")
    if set(quality) != {"schema_version", "checked_python_files", "issues", "type_check", "status"}:
        _fail("quality report fields differ")
    checked = quality.get("checked_python_files")
    type_check = quality.get("type_check")
    if (
        type(quality.get("schema_version")) is not int
        or quality.get("schema_version") != 0
        or type(checked) is not int
        or checked <= 0
        or quality.get("issues") != []
        or quality.get("status") != "pass"
        or not isinstance(type_check, dict)
        or set(type_check) != {"tool", "version", "returncode", "status", "output"}
        or type_check.get("tool") != "mypy"
        or type(type_check.get("version")) is not str
        or not type_check.get("version")
        or type(type_check.get("returncode")) is not int
        or type_check.get("returncode") != 0
        or type_check.get("status") != "pass"
        or type(type_check.get("output")) is not list
        or any(type(line) is not str for line in type_check.get("output", []))
    ):
        _fail("quality report does not prove a passing nonempty mypy run")
    return quality


def _junit_integer(suite: ET.Element, field: str) -> int:
    raw = suite.attrib.get(field)
    if raw is None or not raw.isdecimal():
        _fail(f"JUnit {field} counter is absent or non-decimal")
    return int(raw)


def _verify_junit(path: Path) -> dict[str, int]:
    checked = _plain_file(path, label="JUnit report")
    try:
        raw = checked.read_bytes()
    except OSError as exc:
        raise PolicyCPUComponentError("JUnit report is unreadable") from exc
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        _fail("JUnit report cannot declare a document type or entity")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise PolicyCPUComponentError("JUnit report is not well-formed XML") from exc
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = [child for child in list(root) if child.tag == "testsuite"]
        if len(suites) != len(list(root)):
            _fail("JUnit testsuites contains an unexpected child")
    else:
        _fail("JUnit root must be testsuite or testsuites")
    if not suites:
        _fail("JUnit contains no suites")
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        cases = [child for child in list(suite) if child.tag == "testcase"]
        counters = {field: _junit_integer(suite, field) for field in totals}
        try:
            duration = float(suite.attrib.get("time", "0"))
        except ValueError as exc:
            raise PolicyCPUComponentError("JUnit duration is invalid") from exc
        if not math.isfinite(duration) or duration < 0 or counters["tests"] != len(cases):
            _fail("JUnit suite duration/count differs")
        observed = {"failures": 0, "errors": 0, "skipped": 0}
        for case in cases:
            if not case.attrib.get("name") or not case.attrib.get("classname"):
                _fail("JUnit testcase lacks name or classname")
            for child in list(case):
                if child.tag in observed:
                    observed[child.tag] += 1
        if any(counters[field] != observed[field] for field in observed):
            _fail("JUnit outcome counters differ from testcase elements")
        for field in totals:
            totals[field] += counters[field]
    if (
        totals["tests"] <= 0
        or totals["tests"] <= totals["skipped"]
        or totals["failures"] != 0
        or totals["errors"] != 0
    ):
        _fail("JUnit does not prove a passing nonempty CPU suite")
    return totals


def _expected_commands(gate_relative: str) -> list[tuple[str, list[str]]]:
    cpu = f"{gate_relative}/policy-cpu"
    return [
        ("quality", ["{finalized_python}", "scripts/check_quality.py", "--output", f"{cpu}/quality.json"]),
        (
            "pytest",
            [
                "{finalized_python}", "-m", "pytest", "tests", "-p", "no:cacheprovider",
                "--basetemp", f"{cpu}-pytest-scratch", "--junitxml", f"{cpu}/junit.xml", "--tb=short",
            ],
        ),
        ("example", ["{finalized_python}", "docs/examples/v0_2_policy_cpu.py"]),
    ]


def _nonnegative_integers(value: object, *, length: int | None = None) -> bool:
    return (
        type(value) is list
        and (length is None or len(value) == length)
        and all(type(item) is int and item >= 0 for item in value)
    )


def _verify_binding(value: object, digest: object, portable: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
        _fail("CPU example execution binding fields differ")
    environment = value.get("environment")
    toolchain = value.get("toolchain")
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("implementation_source_scope") != "project-root-v1"
        or value.get("implementation_source_digest") != portable["project_root_digest"]
        or value.get("protocol_sha256") != PROTOCOL_SHA256
        or not isinstance(environment, dict)
        or set(environment) != _BINDING_ENVIRONMENT_FIELDS
        or environment.get("backend") != "cpu"
        or environment.get("python_version") != portable["python_version"]
        or type(environment.get("platform")) is not str
        or not environment.get("platform")
        or type(environment.get("machine")) is not str
        or not environment.get("machine")
        or type(environment.get("torch_version")) is not str
        or not environment.get("torch_version")
        or environment.get("cuda_arithmetic") != "not_applicable"
        or environment.get("cuda_execution_authorized") is not False
        or value.get("environment_digest") != _sha_bytes(_canonical_bytes(environment))
        or toolchain != {
            "binding": "source-bound-manifest",
            "environment_id": portable["environment_id"],
            "lock_digest": portable["lock_digest"],
            "manifest_digest": portable["manifest_digest"],
        }
        or digest != _sha_bytes(_canonical_bytes(value))
    ):
        _fail("CPU example execution binding is not source/toolchain/protocol bound")


def _verify_timings(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _TIMING_FIELDS:
        _fail("CPU example timing fields differ")
    accepted = value.get("accepted_ns")
    terminal = value.get("terminal_ready_ns")
    server = value.get("server_e2e_wall_time_ns")
    token_ready = value.get("token_ready_ns")
    forward = value.get("per_forward_wall_time_ns")
    if (
        type(accepted) is not int
        or accepted < 0
        or type(terminal) is not int
        or terminal < accepted
        or type(server) is not int
        or server != terminal - accepted
        or not _nonnegative_integers(token_ready, length=4)
        or cast(list[int], token_ready) != sorted(cast(list[int], token_ready))
        or any(item < accepted or item > terminal for item in cast(list[int], token_ready))
        or not _nonnegative_integers(forward, length=4)
        or type(value.get("tokenization_wall_time_ns")) is not int
        or value["tokenization_wall_time_ns"] < 0
        or value.get("critical_capture_event_drop_count") != 0
        or value.get("clock_domain") != "host.perf_counter_ns"
        or value.get("evidence_kind") != "measured"
        or value.get("telemetry") != "metrics"
    ):
        _fail("CPU example timings are malformed or internally inconsistent")


def _verify_example_stdout(path: Path, portable: Mapping[str, Any]) -> None:
    checked = _plain_file(path, label="example stdout")
    try:
        lines = checked.read_bytes().splitlines()
    except OSError as exc:
        raise PolicyCPUComponentError("example stdout is unreadable") from exc
    if len(lines) != 2:
        _fail("CPU example must emit exactly B and C result records")
    records = [_decode_json(line, label="example result") for line in lines]
    if any(not isinstance(record, dict) for record in records):
        _fail("CPU example result must be an object")
    b, c = cast(list[dict[str, Any]], records)
    zero_resources = b.get("resource_status")
    candidate = b.get("numerical_policy")
    if not isinstance(candidate, dict):
        _fail("CPU example candidate policy must be an object")
    candidate_bytes = _canonical_bytes(candidate)
    output_ids = b.get("output_token_ids")
    if (
        set(b) != _RESULT_FIELDS
        or set(c) != _RESULT_FIELDS
        or type(b.get("schema_version")) is not int
        or b.get("schema_version") != 3
        or type(c.get("schema_version")) is not int
        or c.get("schema_version") != 3
        or type(b.get("request_id")) is not str
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", b["request_id"])
        or type(c.get("request_id")) is not str
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", c["request_id"])
        or b.get("request_id") == c.get("request_id")
        or b.get("input_token_ids") != [1, 107, 104, 111, 111, 114]
        or c.get("input_token_ids") != b.get("input_token_ids")
        or not _nonnegative_integers(output_ids, length=4)
        or c.get("output_token_ids") != output_ids
        or type(b.get("text")) is not str
        or c.get("text") != b.get("text")
        or b.get("kv_mode") != "none"
        or c.get("kv_mode") != "contiguous"
        or b.get("numerical_policy_digest") != dict(CANDIDATE_POLICY_DIGESTS)["fp32"]
        or c.get("numerical_policy_digest") != dict(CANDIDATE_POLICY_DIGESTS)["fp32"]
        or _sha_bytes(candidate_bytes) != dict(CANDIDATE_POLICY_DIGESTS)["fp32"]
        or c.get("numerical_policy") != candidate
        or b.get("outcome") != "FINISHED"
        or c.get("outcome") != "FINISHED"
        or b.get("finish_reason") != "MAX_NEW_TOKENS"
        or c.get("finish_reason") != "MAX_NEW_TOKENS"
        or b.get("error_code") is not None
        or c.get("error_code") is not None
        or b.get("actual_seed") != 7
        or c.get("actual_seed") != 7
        or type(b.get("model_spec_id")) is not str
        or _SHA256.fullmatch(b["model_spec_id"]) is None
        or c.get("model_spec_id") != b.get("model_spec_id")
        or type(b.get("sampling_spec_digest")) is not str
        or _SHA256.fullmatch(b["sampling_spec_digest"]) is None
        or c.get("sampling_spec_digest") != b.get("sampling_spec_digest")
        or b.get("next_output_position") != 4
        or c.get("next_output_position") != 4
        or b.get("rng_draws_consumed") != 0
        or c.get("rng_draws_consumed") != 0
        or b.get("committed_draw_addresses") != []
        or c.get("committed_draw_addresses") != []
        or b.get("query_tokens_per_step") != [6, 7, 8, 9]
        or c.get("query_tokens_per_step") != [6, 1, 1, 1]
        or b.get("kv_valid_tokens_per_step") != [0, 0, 0, 0]
        or c.get("kv_valid_tokens_per_step") != [6, 7, 8, 9]
        or b.get("capture_manifest_digest") is not None
        or c.get("capture_manifest_digest") is not None
        or b.get("capture_error") is not None
        or c.get("capture_error") is not None
        or type(b.get("event_count")) is not int
        or b["event_count"] <= 0
        or type(c.get("event_count")) is not int
        or c["event_count"] <= 0
        or not isinstance(zero_resources, dict)
        or set(zero_resources) != _RESOURCE_FIELDS
        or zero_resources != c.get("resource_status")
        or any(type(value) is not int or value != 0 for value in zero_resources.values())
        or type(b.get("memory_plan_digest")) is not str
        or _SHA256.fullmatch(b["memory_plan_digest"]) is None
        or type(c.get("memory_plan_digest")) is not str
        or _SHA256.fullmatch(c["memory_plan_digest"]) is None
        or b.get("memory_plan_digest") == c.get("memory_plan_digest")
    ):
        _fail("CPU example stdout does not prove the fixed B/C development example")
    _verify_binding(b.get("execution_binding"), b.get("execution_binding_digest"), portable)
    _verify_binding(c.get("execution_binding"), c.get("execution_binding_digest"), portable)
    if b.get("execution_binding") != c.get("execution_binding"):
        _fail("CPU example B/C executions use different source/environment bindings")
    _verify_timings(b.get("timings"))
    _verify_timings(c.get("timings"))


def _verify_execution(
    root: Path,
    gate_relative: str,
    portable: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    execution = _read_json(root / "execution.json", label="CPU execution report")
    if (
        set(execution) != {"schema_version", "kind", "commands"}
        or type(execution.get("schema_version")) is not int
        or execution.get("schema_version") != 1
        or execution.get("kind") != "illm-policy-cpu-development-execution-v1"
        or type(execution.get("commands")) is not list
    ):
        _fail("CPU execution report fields differ")
    commands = cast(list[Any], execution["commands"])
    expected = _expected_commands(gate_relative)
    if len(commands) != len(expected) or any(not isinstance(item, dict) for item in commands):
        _fail("CPU execution must contain exactly three ordered command objects")
    fields = {
        "phase", "argv", "returncode", "stdout_file", "stdout_sha256", "stderr_file", "stderr_sha256",
    }
    log_summary: dict[str, dict[str, Any]] = {}
    normalized = cast(list[dict[str, Any]], commands)
    for record, (phase, argv) in zip(normalized, expected, strict=True):
        for stream in ("stdout", "stderr"):
            relative = f"logs/{phase}.{stream}.txt"
            log = _plain_file(root / relative, label=relative)
            digest = _sha_file(log)
            try:
                size = log.stat().st_size
            except OSError as exc:
                raise PolicyCPUComponentError("CPU command log is unreadable") from exc
            log_summary[f"{phase}.{stream}"] = {"sha256": digest, "size_bytes": size}
            if record.get(f"{stream}_file") != relative or record.get(f"{stream}_sha256") != digest:
                _fail("CPU command log path or digest differs")
        if (
            set(record) != fields
            or record.get("phase") != phase
            or record.get("argv") != argv
            or type(record.get("returncode")) is not int
            or record.get("returncode") != 0
        ):
            _fail("CPU command argv/order/returncode differs")
    _verify_example_stdout(root / "logs/example.stdout.txt", portable)
    return normalized, log_summary


def _verify_cpu(
    root: Path,
    portable: Mapping[str, Any],
    gate_relative: str,
) -> tuple[dict[str, Any], dict[str, int], dict[str, dict[str, Any]]]:
    quality = _verify_quality(root / "quality.json")
    junit = _verify_junit(root / "junit.xml")
    commands, logs = _verify_execution(root, gate_relative, portable)
    if (root / "logs/quality.stdout.txt").read_bytes() != (root / "quality.json").read_bytes():
        _fail("quality stdout is not byte-identical to the quality artifact")
    report = _read_json(root / "cpu-report.json", label="CPU report")
    fields = {
        "schema_version", "kind", "status", "evidence_mode", "qualification_tier",
        "qualification_result", "performance_outcome", "gpu_result", "raw_tensor_evidence",
        "approved_commit", "protocol", "input_manifest_digest", "candidate_policy_digests",
        "toolchain", "commands", "required_artifacts", "pytest_scratch", "limitations",
    }
    required = {
        "quality": {"path": "quality.json", "sha256": _sha_file(root / "quality.json")},
        "junit": {"path": "junit.xml", "sha256": _sha_file(root / "junit.xml")},
    }
    if (
        set(report) != fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "illm-policy-cpu-development-gate-v1"
        or report.get("status") != "pass"
        or report.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or report.get("qualification_tier") != "unqualified"
        or report.get("qualification_result") != "not_evaluated"
        or report.get("performance_outcome") != "not_evaluated"
        or report.get("gpu_result") != "not_run"
        or report.get("raw_tensor_evidence") != "not_captured"
        or report.get("approved_commit") != APPROVED_COMMIT
        or report.get("protocol") != _fixed_protocol()
        or report.get("input_manifest_digest") != INPUT_MANIFEST_DIGEST
        or report.get("candidate_policy_digests") != _fixed_policy_digests()
        or report.get("toolchain") != portable
        or report.get("commands") != commands
        or report.get("required_artifacts") != required
        or report.get("pytest_scratch") != {
            "path": f"{gate_relative}/policy-cpu-pytest-scratch",
            "is_evidence": False,
            "retention": "retained",
        }
        or report.get("limitations") != list(PENDING_WORK)
    ):
        _fail("CPU report identity, result boundary, or linked evidence differs")
    return quality, junit, logs


def _verify_verified(
    root: Path,
    portable: Mapping[str, Any],
    gate_relative: str,
    stages: Mapping[str, Mapping[str, Any]],
) -> None:
    report = _read_json(root / "verification-report.json", label="verification report")
    fields = {
        "schema_version", "kind", "status", "evidence_mode", "qualification_tier",
        "qualification_result", "performance_outcome", "verified_stages", "approved_commit",
        "protocol", "input_manifest_digest", "candidate_policy_digests", "toolchain", "limitations",
    }
    verified = [
        {
            "stage": stage,
            "output_root": f"{gate_relative}/{STAGE_DIRECTORIES[stage]}",
            "files_digest": stages[stage]["files_digest"],
            "file_count": len(cast(list[Any], stages[stage]["files"])),
        }
        for stage in ("inputs", "cpu")
    ]
    if (
        set(report) != fields
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("kind") != "illm-policy-cpu-development-verification-v1"
        or report.get("status") != "pass"
        or report.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or report.get("qualification_tier") != "unqualified"
        or report.get("qualification_result") != "not_evaluated"
        or report.get("performance_outcome") != "not_evaluated"
        or report.get("verified_stages") != verified
        or report.get("approved_commit") != APPROVED_COMMIT
        or report.get("protocol") != _fixed_protocol()
        or report.get("input_manifest_digest") != INPUT_MANIFEST_DIGEST
        or report.get("candidate_policy_digests") != _fixed_policy_digests()
        or report.get("toolchain") != portable
        or report.get("limitations") != list(PENDING_WORK)
    ):
        _fail("verification report differs from independently verified stages")


def _verify_journal(
    root: Path,
    portable: Mapping[str, Any],
    gate_relative: str,
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    journal = _read_json(root / "policy-attempt.json", label="policy attempt journal")
    fields = {
        "schema_version", "journal_kind", "status", "evidence_mode", "qualification_tier",
        "source_identity", "protocol", "approval", "toolchain", "input_identity",
        "candidate_policy_digests", "stages", "pending", "journal_digest",
    }
    records = [
        {
            "sequence": index + 1,
            "stage": stage,
            "depends_on": list(CPU_STAGES[:index]),
            "result": "pass",
            "evidence_mode": "DEVELOPMENT_ONLY",
            "qualification_tier": "unqualified",
            "output_root": f"{gate_relative}/{STAGE_DIRECTORIES[stage]}",
            "files_digest": stages[stage]["files_digest"],
            "checksums_sha256": _sha_file(root / STAGE_DIRECTORIES[stage] / "checksums.json"),
        }
        for index, stage in enumerate(CPU_STAGES)
    ]
    projection = {key: value for key, value in journal.items() if key != "journal_digest"}
    if (
        set(journal) != fields
        or type(journal.get("schema_version")) is not int
        or journal.get("schema_version") != 1
        or journal.get("journal_kind") != "illm-policy-development-attempt-v1"
        or journal.get("status") != "verified_development_only_unqualified"
        or journal.get("evidence_mode") != "DEVELOPMENT_ONLY"
        or journal.get("qualification_tier") != "unqualified"
        or journal.get("source_identity") != {
            "approved_commit": APPROVED_COMMIT,
            "project_root_digest": portable["project_root_digest"],
        }
        or journal.get("protocol") != _fixed_protocol()
        or journal.get("approval") != {
            "approval_kind": "illm-human-policy-approval-v1",
            "sha256": APPROVAL_SHA256,
            "interpretation": "transcript_of_received_human_answer_not_self_authorization",
        }
        or journal.get("toolchain") != portable
        or journal.get("input_identity") != {
            "raw_bytes_sha256": INPUT_BYTES_SHA256,
            "manifest_digest": INPUT_MANIFEST_DIGEST,
        }
        or journal.get("candidate_policy_digests") != _fixed_policy_digests()
        or journal.get("stages") != records
        or journal.get("pending") != list(PENDING_WORK)
        or journal.get("journal_digest") != _sha_bytes(_canonical_bytes(projection))
    ):
        _fail("policy attempt journal identity, stage order, or digest differs")
    return journal


def verify_policy_cpu_development_component(
    root: Path,
    *,
    expected_toolchain: Mapping[str, Any],
    gate_relative: str,
) -> dict[str, Any]:
    """Verify and summarize one copied CPU development-driver component.

    The function never reads paths recorded by the author.  ``gate_relative``
    is a caller-controlled, validated portable identity used only to recompute
    the argv and scratch metadata stored inside the copied evidence.
    """
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not isinstance(expected_toolchain, Mapping):
        raise TypeError("expected_toolchain must be a mapping")
    relative = _safe_gate_relative(gate_relative)
    checked = _plain_directory(root, label="policy profile root")
    toolchain, portable = _validate_toolchain(checked, expected_toolchain)

    for name in ("policy-attempt.json", "toolchain.json"):
        _plain_file(checked / name, label=name)
    stages = {
        stage: _stage_inventory(checked / STAGE_DIRECTORIES[stage], stage)
        for stage in CPU_STAGES
    }
    _verify_inputs(checked / STAGE_DIRECTORIES["inputs"], portable)
    quality, junit, logs = _verify_cpu(checked / STAGE_DIRECTORIES["cpu"], portable, relative)
    _verify_verified(checked / STAGE_DIRECTORIES["verify"], portable, relative, stages)
    journal = _verify_journal(checked, portable, relative, stages)

    cpu_root = checked / STAGE_DIRECTORIES["cpu"]
    return {
        "schema_version": 1,
        "component_kind": COMPONENT_KIND,
        "status": "pass",
        "evidence_stage": "DEVELOPMENT_ONLY",
        "qualification_status": "unqualified",
        "qualification_result": "not_evaluated",
        "gpu_result": "not_run",
        "raw_tensor_evidence": "not_captured",
        "inputs_files_digest": stages["inputs"]["files_digest"],
        "cpu_files_digest": stages["cpu"]["files_digest"],
        "verified_files_digest": stages["verify"]["files_digest"],
        "journal_digest": journal["journal_digest"],
        "cpu_report_sha256": _sha_file(cpu_root / "cpu-report.json"),
        "quality_sha256": _sha_file(cpu_root / "quality.json"),
        "junit_sha256": _sha_file(cpu_root / "junit.xml"),
        "toolchain_manifest_digest": portable["manifest_digest"],
        "implementation_source_digest": portable["project_root_digest"],
        "environment_id": portable["environment_id"],
        "cpu_full_suite": {
            "nested_execution": ["quality", "pytest", "example"],
            "quality_status": "pass",
            "checked_python_files": quality["checked_python_files"],
            "junit": junit,
            "command_logs": logs,
            "pytest_scratch": {
                "path": f"{relative}/policy-cpu-pytest-scratch",
                "is_evidence": False,
                "retention": "retained",
            },
        },
        "toolchain_schema_version": toolchain["schema_version"],
    }


__all__ = [
    "COMPONENT_KIND",
    "PolicyCPUComponentError",
    "verify_policy_cpu_development_component",
]

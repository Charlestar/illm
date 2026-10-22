"""Read-only, stdlib-only integrity and recomputation of CPU raw evidence.

The seal protects internal consistency, not authenticity or release fitness.
GPU evidence is deliberately absent. No local checkout, journal, environment
manifest, tensor package, or network is needed to inspect a copied bundle.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.policy_cpu_numerical_contract import (
    NUMERICAL_REPORT_KIND,
    PROTOCOL_SHA256,
    NumericalEvidenceError,
    plain_path,
    read_json,
)


BUNDLE_KIND = "illm-policy-cpu-raw-bundle-v1"
APPROVAL_SHA256 = "2046ec2425b57d050ff61d7ad636e747b42eb6b08bd88a6883ae2188fae2ab7f"
READER_PROGRAM_SHA256 = "0a79eeab1dd405e7332429c4a3f820c06c2800b6bd13d3abead66acdfdef9b13"
WORKER_NAMES = ("native", "candidate", "formula", "mutations", "reproduction")
COMMAND_NAMES = (*WORKER_NAMES, "reader-source", "reader-copy")
AUTHORITY_NAMES = (
    "inputs.json", "oracle-manifest.json", "hello.safetensors",
    "vocab-min-position-zero.safetensors", "vocab-max-position-last.safetensors",
    "config.json", "model.safetensors", "supported-model-spec.json",
    "weight-load-manifest.json", "tokenizer.json", "fixture-manifest.json",
    "protocol.md", "approval.json",
)


def file_identity(path: Path, relative: str) -> dict[str, Any]:
    requested = plain_path(path)
    if not requested.is_file():
        raise NumericalEvidenceError("bundle member must be a plain file")
    digest = hashlib.sha256()
    size = 0
    with requested.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"path": relative, "size_bytes": size, "sha256": digest.hexdigest()}


def safe_relative(value: Any) -> str:
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise NumericalEvidenceError("invalid bundle member path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise NumericalEvidenceError("bundle member path cannot escape its root")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise NumericalEvidenceError("noncanonical bundle member path")
    return value


def inventory(root: Path, *, exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    requested = plain_path(root)
    if not requested.is_dir():
        raise NumericalEvidenceError("bundle root must be a plain directory")
    files: list[dict[str, Any]] = []
    directories: set[str] = set()
    for parent, dirnames, filenames in os.walk(requested, followlinks=False):
        directory = plain_path(Path(parent))
        for name in dirnames:
            member = plain_path(directory / name)
            if not member.is_dir():
                raise NumericalEvidenceError("bundle contains a non-directory member")
            directories.add(member.relative_to(requested).as_posix())
        for name in filenames:
            member = plain_path(directory / name)
            relative = safe_relative(member.relative_to(requested).as_posix())
            if relative not in exclude:
                files.append(file_identity(member, relative))
    allowed_directories: set[str] = set()
    for record in files:
        allowed_directories.update(
            parent.as_posix() for parent in PurePosixPath(record["path"]).parents
            if parent.as_posix() != "."
        )
    if directories != allowed_directories:
        raise NumericalEvidenceError("bundle contains an empty or unlisted directory")
    return sorted(files, key=lambda record: record["path"])


def _authority(root: Path) -> None:
    authority = plain_path(root / "authorities")
    if not authority.is_dir():
        raise NumericalEvidenceError("CPU authorities must be a plain directory")
    if sorted(path.name for path in authority.iterdir()) != sorted(AUTHORITY_NAMES):
        raise NumericalEvidenceError("CPU authority file inventory differs")
    for filename, digest in (("protocol.md", PROTOCOL_SHA256), ("approval.json", APPROVAL_SHA256)):
        if file_identity(authority / filename, filename)["sha256"] != digest:
            raise NumericalEvidenceError("human approval or immutable protocol bytes differ")
    approval = read_json(authority / "approval.json")
    if (approval["authorized_stages"] != ["inputs", "cpu", "verify"]
            or approval["gpu_authorized"] is not False or approval["gpu_budget_status"] != "unreconciled"):
        raise NumericalEvidenceError("CPU bundle has no matching human authorization")


def verify_payload(root: Path) -> dict[str, Any]:
    """Recompute every numerical record without trusting saved pass booleans."""
    from illm.policy_cpu_raw_validation import derive_cpu_numerical

    requested = plain_path(root)
    _authority(requested)
    report = read_json(requested / "numerical-report.json")
    if (not isinstance(report, dict) or report.get("report_kind") != NUMERICAL_REPORT_KIND
            or report.get("evidence_stage") != "DEVELOPMENT_ONLY"
            or report.get("qualification_status") != "unqualified"
            or report.get("status") not in {"incomplete", "fail"}
            or any(report.get(axis) != "incomplete" for axis in (
                "hard_gate_result", "legacy_elementwise_compatibility", "sequence_compatibility"))):
        raise NumericalEvidenceError("CPU-only report cannot assert full protocol qualification")
    derived = derive_cpu_numerical(requested, verify_records_path=requested / "comparisons.jsonl")
    expected_bytes = canonical_json_bytes(derived) + b"\n"
    if report != derived or (requested / "numerical-report.json").read_bytes() != expected_bytes:
        raise NumericalEvidenceError("saved numerical report differs from raw recomputation")
    return derived


def worker_program(module: str) -> str:
    if module not in {"runtime", "formula", "mutation"}:
        raise NumericalEvidenceError("unknown CPU worker module")
    return ("import sys;sys.dont_write_bytecode=True;sys.path.insert(0,sys.argv.pop(1));"
            f"from illm.policy_cpu_{module}_worker import main;raise SystemExit(main())")


def _command_shapes(commands: list[dict[str, Any]]) -> None:
    # Original author paths are historical strings, never opened on the reader's
    # host. Slash-only normalization permits Windows evidence to be read on Unix.
    def lexical(value: str) -> str:
        return value.replace("\\", "/")

    project = lexical(commands[0]["cwd"])
    first = commands[0]["argv"]
    if len(first) != 13 or not lexical(first[10]).endswith("/workers/native"):
        raise NumericalEvidenceError("native command has an invalid output binding")
    artifact = lexical(first[10])[:-len("/workers/native")]
    python = first[0]
    roles = {"native": ("runtime", "native"), "candidate": ("runtime", "candidate"),
             "formula": ("formula", None), "mutations": ("mutation", None),
             "reproduction": ("runtime", "candidate-reproduction")}
    for command in commands:
        label, argv, cwd = command["label"], command["argv"], lexical(command["cwd"])
        if argv[0] != python:
            raise NumericalEvidenceError("CPU child interpreters differ")
        if label in roles:
            module, role = roles[label]
            expected = [python, "-I", "-X", "utf8", "-c", worker_program(module), project + "/src",
                        "--project-root", project, "--output", artifact + "/workers/" + label]
            if role is not None:
                expected.extend(("--role", role))
            if len(argv) != len(expected) or [lexical(arg) for arg in argv] != [lexical(arg) for arg in expected] or cwd != project:
                raise NumericalEvidenceError("CPU worker command program, role, or path binding differs")
        else:
            reader = project + "/src" if label == "reader-source" else artifact + "/reader"
            expected_cwd = artifact + ("/authorities" if label == "reader-source" else "/reader")
            if (len(argv) != 10 or argv[:6] != [python, "-I", "-S", "-X", "utf8", "-c"]
                    or sha256_bytes(argv[6].encode("utf-8")) != READER_PROGRAM_SHA256
                    or [lexical(arg) for arg in argv[7:]] != [reader, artifact, "payload"]
                    or cwd != expected_cwd):
                raise NumericalEvidenceError("portable reader command program or path binding differs")


def _commands(root: Path) -> None:
    commands = read_json(root / "commands.json")
    if not isinstance(commands, list) or len(commands) != len(COMMAND_NAMES):
        raise NumericalEvidenceError("CPU command transcript is incomplete")
    for command, label in zip(commands, COMMAND_NAMES, strict=True):
        if (not isinstance(command, dict)
                or set(command) != {"label", "argv", "cwd", "returncode", "stdout", "stderr"}
                or command["label"] != label or type(command["returncode"]) is not int
                or command["returncode"] != 0 or type(command["cwd"]) is not str
                or not isinstance(command["argv"], list) or not command["argv"]
                or any(type(arg) is not str or not arg or "\0" in arg for arg in command["argv"])
                or "-I" not in command["argv"]
                or (label.startswith("reader-") and "-S" not in command["argv"])):
            raise NumericalEvidenceError("CPU command transcript has invalid identity or exit status")
        for stream in ("stdout", "stderr"):
            expected_path = f"logs/{label}.{stream}.log"
            if command[stream] != file_identity(root / expected_path, expected_path):
                raise NumericalEvidenceError("CPU command transcript log bytes differ")
        if read_json(root / "logs" / f"{label}.command.json") != command:
            raise NumericalEvidenceError("individual CPU command transcript differs")
    _command_shapes(commands)
    interpreter = commands[0]["argv"][0].replace("\\", "/")
    for name in WORKER_NAMES:
        worker = read_json(root / "workers" / name / "worker.json")
        process = worker.get("process")
        if (not isinstance(process, dict) or type(process.get("python_executable")) is not str
                or process["python_executable"].replace("\\", "/") != interpreter):
            raise NumericalEvidenceError("CPU command interpreter differs from actual worker process identity")
    source = read_json(root / "logs/reader-source.stdout.log")
    copied = read_json(root / "logs/reader-copy.stdout.log")
    if source != copied:
        raise NumericalEvidenceError("source and copied reader recomputations differ")
    for result in (source, copied):
        if (not isinstance(result, dict) or set(result) != {
            "schema_version", "report_digest", "comparison_records", "module_files",
            "forbidden_import_attempts", "forbidden_loaded_modules", "network_events",
        } or type(result["schema_version"]) is not int or result["schema_version"] != 1
                or result["forbidden_import_attempts"] != [] or result["forbidden_loaded_modules"] != []
                or result["network_events"] != []):
            raise NumericalEvidenceError("portable reader isolation transcript differs")
        modules = result["module_files"]
        if (not isinstance(modules, dict) or not modules or any(
            type(name) is not str or (name != "illm" and not name.startswith("illm."))
            or not safe_relative(relative).endswith(".py") for name, relative in modules.items()
        )):
            raise NumericalEvidenceError("portable reader module inventory differs")
    report = read_json(root / "numerical-report.json")
    if source["report_digest"] != sha256_json(report) or source["comparison_records"] != report["comparison_records"]:
        raise NumericalEvidenceError("portable transcript does not bind the saved numerical report")


def verify_seal(root: Path) -> dict[str, Any]:
    """Check the closed file/transcript seal before executing a copied reader."""
    requested = plain_path(root)
    if (requested / "INCOMPLETE").exists():
        raise NumericalEvidenceError("CPU bundle is still incomplete")
    manifest = read_json(requested / "bundle-manifest.json")
    if (not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "bundle_kind", "evidence_stage", "qualification_status",
        "binding", "binding_digest", "files", "manifest_digest",
    } or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["bundle_kind"] != BUNDLE_KIND
            or manifest["evidence_stage"] != "DEVELOPMENT_ONLY"
            or manifest["qualification_status"] != "unqualified"
            or manifest["binding_digest"] != sha256_json(manifest["binding"])
            or manifest["manifest_digest"] != sha256_json({
                key: value for key, value in manifest.items() if key != "manifest_digest"})):
        raise NumericalEvidenceError("CPU bundle manifest identity differs")
    complete = plain_path(requested / "COMPLETE")
    if not complete.is_file() or complete.read_bytes() != (manifest["manifest_digest"] + "\n").encode("ascii"):
        raise NumericalEvidenceError("CPU bundle completion marker differs")
    actual = inventory(requested, exclude=frozenset({"bundle-manifest.json", "COMPLETE"}))
    if manifest["files"] != actual:
        raise NumericalEvidenceError("CPU bundle has missing, modified, or extra files")
    fixed_files = {
        *(f"authorities/{name}" for name in AUTHORITY_NAMES),
        *(f"logs/{label}.{stream}.log" for label in COMMAND_NAMES for stream in ("stdout", "stderr")),
        *(f"logs/{label}.command.json" for label in COMMAND_NAMES),
        "commands.json", "numerical-report.json", "comparisons.jsonl",
        *(f"workers/{name}/{leaf}" for name in WORKER_NAMES for leaf in (
            "worker.json", "COMPLETE", "pack/data.bin", "pack/index.jsonl", "pack/manifest.json", "pack/COMPLETE")),
    }
    reader_files = {record["path"] for record in actual if record["path"].startswith("reader/")}
    if not reader_files or any(not path.endswith(".py") for path in reader_files):
        raise NumericalEvidenceError("CPU bundle reader closure must contain only plain Python source")
    if {record["path"] for record in actual} != fixed_files | reader_files:
        raise NumericalEvidenceError("CPU bundle member catalog differs")
    for name in WORKER_NAMES:
        worker = read_json(requested / "workers" / name / "worker.json")
        if (worker.get("binding") != manifest["binding"]
                or worker.get("binding_digest") != manifest["binding_digest"]):
            raise NumericalEvidenceError("worker execution binding differs from outer bundle")
    _commands(requested)
    source = read_json(requested / "logs/reader-source.stdout.log")
    if reader_files != {"reader/" + safe_relative(value) for value in source["module_files"].values()}:
        raise NumericalEvidenceError("CPU reader closure differs from the isolated dependency transcript")
    return manifest


def verify_bundle(root: Path) -> dict[str, Any]:
    """Verify the closed seal, transcripts and all raw-derived metrics read-only."""
    verify_seal(root)
    return verify_payload(root)


__all__ = ["APPROVAL_SHA256", "AUTHORITY_NAMES", "BUNDLE_KIND", "COMMAND_NAMES", "WORKER_NAMES",
           "READER_PROGRAM_SHA256", "file_identity", "inventory", "safe_relative", "verify_bundle", "verify_payload",
           "verify_seal", "worker_program"]

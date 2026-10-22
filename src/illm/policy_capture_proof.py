"""Read archived schema-3 portable-reader proofs without executing their code.

The standalone producer still labels its proof DEVELOPMENT_ONLY.  Embedding
it in a CPU profile does not change those bytes or grant release qualification.
Author paths are compared lexically and are never opened on the reader's host.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.policy_cpu_numerical_contract import NumericalEvidenceError, plain_path, read_json, strict_json
from illm.policy_cpu_raw_bundle import inventory
from illm.reference_qualification import _relative
from illm.stateful_qualification import _absolute_argv_path


REPORT_FIELDS = frozenset({
    "schema_version", "report_kind", "status", "evidence_stage", "qualification_status",
    "evidence_scope", "limitations", "input_capture_path", "input_capture_manifest",
    "source_dependency_snapshot", "reader_modules", "reader_files", "capture_files", "commands",
})
IDENTITY_FIELDS = frozenset({
    "schema_version", "manifest_digest", "event_catalog", "capabilities", "protocol_sha256",
    "numerical_policy_digest", "memory_plan_digest", "execution_binding_digest",
    "implementation_source_digest", "environment_digest", "toolchain", "observation_profile",
    "request_id", "event_count", "logical_terminal_outcome", "terminal_outcome",
})
CHILD_FIELDS = IDENTITY_FIELDS | {
    "story_capability", "module_files", "forbidden_import_attempts", "forbidden_loaded_modules", "network_events",
}
COMMAND_FIELDS = frozenset({
    "argv", "cwd", "returncode", "stdout", "stderr", "stdout_sha256", "stderr_sha256", "result",
})
LIMITATIONS = [
    "DEVELOPMENT_ONLY; this report is not qualification or release evidence",
    "copied source hashes prove internal byte consistency, not independent source truth",
    "R0 metadata replay only; no model execution, R1 state, or numerical recomputation",
]
CAPTURE_FILES = frozenset({"events.jsonl", "checksums.json", "capture-manifest.json", "COMPLETE"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericalEvidenceError("policy portable capture proof: " + message)


def _same(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _hashes(value: Any) -> dict[str, str]:
    _require(isinstance(value, dict) and bool(value), "empty or invalid digest map")
    result: dict[str, str] = {}
    for name, digest in value.items():
        _require(type(name) is str and _relative(name) == name, "noncanonical member path")
        _require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
                 "invalid member digest")
        result[name] = digest
    _require(len({name.casefold() for name in result}) == len(result), "case-colliding member paths")
    return result


def _tree_hashes(root: Path) -> dict[str, str]:
    records = inventory(root)
    result = {record["path"]: record["sha256"] for record in records}
    return _hashes(result)


def _child_program(path: Path) -> str:
    """Extract a source-bound literal; never import or execute archived code."""
    tree = ast.parse(plain_path(path).read_text(encoding="utf-8"))
    values = [
        node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_CHILD_PROGRAM" for target in node.targets)
    ]
    _require(len(values) == 1, "producer must contain exactly one child-program literal")
    value = ast.literal_eval(values[0])
    _require(type(value) is str and bool(value), "producer child-program literal is invalid")
    return str(value)


def _identity(bundle: PolicyStreamingCaptureBundle) -> dict[str, Any]:
    manifest = bundle.manifest
    binding = manifest["execution_binding"]
    return {
        **{name: manifest[name] for name in IDENTITY_FIELDS - {
            "implementation_source_digest", "environment_digest", "toolchain",
        }},
        "implementation_source_digest": binding["implementation_source_digest"],
        "environment_digest": binding["environment_digest"], "toolchain": binding["toolchain"],
    }


def verify_policy_capture_proof(
    root: Path, *, expected_capture: Path, producer_source: Path,
    source_files: Mapping[str, str],
) -> dict[str, Any]:
    """Fully check saved transcripts and copied bytes, with no subprocesses.

    ``producer_source`` is an embedded file whose SHA must match the profile's
    source inventory.  It supplies only the exact isolated child program literal.
    This proves consistency of archived execution facts, not author authenticity.
    """
    root = plain_path(root)
    _require(root.is_dir() and not (root / "INCOMPLETE").exists(), "proof is incomplete")
    _require({item.name for item in root.iterdir()} == {"reader", "capture", "report.json", "COMPLETE"},
             "unexpected proof root members")
    producer = plain_path(producer_source)
    _require(sha256_bytes(producer.read_bytes()) == source_files.get("scripts/check_policy_capture_portable.py"),
             "producer source differs from embedded provenance")
    program = _child_program(producer)
    report = read_json(root / "report.json")
    _require(isinstance(report, dict) and set(report) == REPORT_FIELDS, "report fields differ")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1
             and report["report_kind"] == "illm-policy-capture-portable-development-v1"
             and report["status"] == "pass" and report["evidence_stage"] == "DEVELOPMENT_ONLY"
             and report["qualification_status"] == "unqualified"
             and report["evidence_scope"] == "copied-minimal-reader-and-copied-schema3-r0-capture-internal-consistency"
             and report["limitations"] == LIMITATIONS, "report changes its development-only scope")
    report_sha = sha256_bytes(plain_path(root / "report.json").read_bytes())
    _require(plain_path(root / "COMPLETE").read_bytes() == (report_sha + "\n").encode("ascii"),
             "completion marker differs")
    copied = PolicyStreamingCaptureBundle.open(root / "capture")
    expected = PolicyStreamingCaptureBundle.open(expected_capture)
    identity = _identity(copied)
    _require(_same(identity, _identity(expected)) and _same(report["input_capture_manifest"], identity),
             "copied capture differs from the profile capture")
    captured_files = _hashes(report["capture_files"])
    _require(set(captured_files) == CAPTURE_FILES and captured_files == _tree_hashes(root / "capture")
             and captured_files == _tree_hashes(expected_capture), "copied capture file bytes differ")
    files = _hashes(report["reader_files"])
    snapshot = report["source_dependency_snapshot"]
    _require(isinstance(snapshot, dict) and set(snapshot) == {"scope", "files"}
             and snapshot["scope"] == "integrity-only-copied-current-reader-source;not-independent-source-truth"
             and _same(snapshot["files"], files), "reader dependency snapshot differs")
    _require(files == _tree_hashes(root / "reader") and all(
        name.endswith(".py") and source_files.get("src/" + name) == digest for name, digest in files.items()
    ), "copied reader files differ from profile source")
    modules = report["reader_modules"]
    _require(isinstance(modules, dict) and bool(modules) and all(
        type(name) is str and (name == "illm" or name.startswith("illm."))
        and type(relative) is str and relative in files for name, relative in modules.items()
    ) and len(set(modules.values())) == len(modules) and set(modules.values()) == set(files),
             "observed module closure differs")
    commands = report["commands"]
    _require(isinstance(commands, dict) and set(commands) == {"source_validation", "copied_validation"},
             "isolated execution pair is absent")
    source_command = commands["source_validation"]
    copied_command = commands["copied_validation"]
    for command in (source_command, copied_command):
        _require(isinstance(command, dict) and set(command) == COMMAND_FIELDS, "command fields differ")
        argv = command["argv"]
        _require(isinstance(argv, list) and len(argv) == 7
                 and all(type(arg) is str and bool(arg) and "\0" not in arg for arg in argv)
                 and argv[1:5] == ["-I", "-S", "-c", program]
                 and type(command["returncode"]) is int and command["returncode"] == 0
                 and type(command["stdout"]) is str and command["stderr"] == "", "command did not pass in isolation")
        for index in (0, 5, 6):
            _absolute_argv_path(argv[index], "portable-reader command")
        _absolute_argv_path(command["cwd"], "portable-reader cwd")
        for stream in ("stdout", "stderr"):
            _require(command[stream + "_sha256"] == sha256_bytes(command[stream].encode("utf-8")),
                     "command stream digest differs")
        result = strict_json(command["stdout"], label="portable reader stdout")
        _require(isinstance(result, dict) and set(result) == CHILD_FIELDS
                 and _same(result, command["result"]), "command stdout/result differs")
        _require(_same({name: result[name] for name in IDENTITY_FIELDS}, identity)
                 and result["story_capability"] == "R0" and _same(result["module_files"], modules)
                 and all(result[name] == [] for name in (
                     "forbidden_import_attempts", "forbidden_loaded_modules", "network_events",
                 )), "child capture/import/network facts differ")
    source_argv, copied_argv = source_command["argv"], copied_command["argv"]
    def lexical(value: str) -> str:
        return _absolute_argv_path(value, "portable-reader anchor")
    original = lexical(report["input_capture_path"])
    copied_capture = lexical(copied_argv[6])
    _require(copied_capture.endswith("/capture"), "copied capture anchor differs")
    proof_anchor = copied_capture[:-len("/capture")]
    _require(lexical(source_argv[6]) == original
             and lexical(copied_argv[5]) == proof_anchor + "/reader"
             and lexical(copied_command["cwd"]) == proof_anchor
             and lexical(source_argv[5]) == lexical(source_command["cwd"]) + "/src"
             and lexical(source_argv[0]) == lexical(copied_argv[0]), "command path/interpreter bindings differ")
    _require(_same(source_command["result"], copied_command["result"]), "source and copied reader results differ")
    return {
        "report_sha256": report_sha, "capture_manifest_digest": copied.manifest["manifest_digest"],
        "reader_files_digest": sha256_json(files), "reader_module_count": len(modules),
        "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified",
        "archived_transcripts_only": True, "reader_executed_by_verifier": False,
    }


__all__ = ["verify_policy_capture_proof"]

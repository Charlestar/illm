"""Portable V0.2 -> immutable V0.1 rollback facts; never execute a binary.

Recorded absolute author paths are compared lexically, never opened. Checksums
are integrity receipts, not signatures against an author fabricating all facts.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, NoReturn, Sequence
import uuid

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import (
    _fs, _object, _signed, _verify_source, _verify_toolchain,
)
from illm.reference_release import verify_reference_release_bundle
from illm.stateful_qualification import _verify_v02_source


OLD_TAG = "v0.1.0"
OLD_COMMIT = "8df86f2ab0b0f3e0c67bde68f65a9aa7f4adfa02"
OLD_TAG_OBJECT = "0cebfd3ea1730db6f89ee3bc68579064a0534f80"
OLD_BUNDLE = "1e87e9cdf10d66a55a4a3231e87b96e97d0dfe0d7ed6a138fc0fc265ec9cdbac"
OLD_SOURCE = {
    "project_root_digest": "3b0de700bfa4e7e43e9aaed9765dc1ad1274d1f0c33f86015105d00629488c09",
    "bootstrap_digest": "1a090333ad37390fbd9dce5691d0176996eec8b53b61a5b3b116d6ce34fdfdd9",
    "base_lock_digest": "19489d7dd52ec0f54bbd45b784add2453453523008ab46550dea8bf989cef274",
}
REPORT_KIND = "illm-v02-to-v01-rollback-v1"
LAST_SAFE_REQUEST = "rollback-last-safe-request"
OLD_REQUEST = "rollback-old-reference-request"
CONFIG = {
    "model": "fixture:tiny-decoder", "device": "cpu", "prompt": "hello",
    "max_new_tokens": 8, "sampling_mode": "greedy", "seed": 7,
    "request_id": LAST_SAFE_REQUEST, "kv_mode": "contiguous", "telemetry": "request",
}
RESOURCE_FIELDS = frozenset({
    "active_requests", "reservations", "quarantined_requests", "active_tickets",
    "input_leases", "output_leases", "workspace_leases", "kv_write_leases",
    "kv_reserved_bytes", "kv_valid_tokens", "kv_staged_tokens",
})
PHASES = (
    "resolve-tag", "archive", "new-run", "new-replay-primary", "new-inspect-primary",
    "new-replay", "new-inspect", "bootstrap-old", "old-environment", "old-generate",
    "old-replay-trace", "old-replay-state", "old-bundle-verify", "old-reject-v02",
    "new-read-old",
)
ENVIRONMENT_PROBE = (
    "import importlib.metadata as m,importlib.util as u,json,os,pathlib;"
    "print(json.dumps({'illm_version':m.version('illm'),"
    "'stateful_engine_present':u.find_spec('illm.engine.streaming') is not None,"
    "'environment_id':os.environ.get('ILLM_ENVIRONMENT_ID'),"
    "'toolchain_digest':json.loads(pathlib.Path(os.environ['ILLM_TOOLCHAIN_MANIFEST']).read_text(encoding='utf-8'))['manifest_digest']},sort_keys=True))"
)
MAX_BYTES = 256 * 1024 * 1024


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _same(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        _fail("stateful rollback link is not portable")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {".", ".."} for part in path.parts):
        _fail("stateful rollback link escapes its sealed root")
    return value


def _author_path(value: object) -> str:
    """Normalize a recorded absolute path without accessing its machine."""
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail("stateful rollback author path is invalid")
    normalized = value.replace("\\", "/")
    if normalized.startswith("//?/"):
        normalized = normalized[4:]
    path = PureWindowsPath(normalized) if re.match(r"^[A-Za-z]:/", normalized) else PurePosixPath(normalized)
    if (not path.is_absolute() or any(part in {".", ".."} for part in path.parts)
            or path.as_posix() != normalized or normalized.endswith("/")):
        _fail("stateful rollback author path must be absolute and canonical")
    return normalized


def portable_stateful_rollback_inventory(root: Path) -> dict[str, str]:
    """Inventory portable facts; omit only explicitly local extracted source/env."""
    if root.is_symlink() or not root.is_dir():
        _fail("stateful rollback root must be an ordinary directory")
    scan = _fs(root)
    resolved = scan.resolve()
    result: dict[str, str] = {}
    total = 0
    for top in scan.iterdir():
        if top.is_symlink():
            _fail("stateful rollback forbids linked artifacts")
        if top.name == "local-only":
            if not top.is_dir():
                _fail("stateful rollback local-only work must be a directory")
            continue
        paths = [top, *top.rglob("*")] if top.is_dir() else [top]
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(resolved):
                _fail("stateful rollback forbids links and escaped artifacts")
            if path.is_dir():
                if not any(path.iterdir()):
                    _fail("stateful rollback contains an unreferenced empty directory")
                continue
            if not path.is_file():
                _fail("stateful rollback contains a nonregular artifact")
            total += path.stat().st_size
            if total > MAX_BYTES:
                _fail("stateful rollback exceeds portable byte budget")
            name = _relative(path.relative_to(scan).as_posix())
            if name not in {"checksums.json", "COMPLETE", "INCOMPLETE"}:
                result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _last_json(path: Path) -> dict[str, Any]:
    lines = [line for line in _fs(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        _fail("stateful rollback command has no structured output")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in stateful rollback transcript")
            result[key] = value
        return result
    value = json.loads(lines[-1], object_pairs_hook=unique,
                       parse_constant=lambda _: _fail("nonfinite rollback transcript"))
    if not isinstance(value, dict):
        _fail("stateful rollback transcript must end in an object")
    return value


def _verify_commands(commands: Sequence[Mapping[str, Any]], new: Mapping[str, Any], old: Mapping[str, Any],
                     primary: StreamingCaptureBundle) -> None:
    if tuple(item.get("phase") for item in commands) != PHASES:
        _fail("stateful rollback lacks the exact executed phase order")
    argv: dict[str, list[str]] = {}
    current_python = _author_path(new.get("python_executable"))
    for item in commands:
        tokens = item.get("argv")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) and token for token in tokens):
            _fail("stateful rollback lacks actual command argv")
        argv[str(item["phase"])] = [token.replace("\\", "/") for token in tokens]
        if item["phase"] not in {"resolve-tag", "archive"} and _author_path(tokens[0]) != current_python:
            _fail("stateful rollback wrapper did not use the current bound Python interpreter")
    if argv["resolve-tag"][0] != argv["archive"][0] or PurePosixPath(argv["archive"][0]).name.lower() not in {"git", "git.exe"}:
        _fail("stateful rollback old source was not resolved and archived by the same Git executable")
    if argv["resolve-tag"][1:] != ["rev-parse", f"refs/tags/{OLD_TAG}", f"refs/tags/{OLD_TAG}^{{commit}}"]:
        _fail("stateful rollback did not resolve the immutable annotated tag")
    archive = argv["archive"]
    if len(archive) != 6 or archive[1:4] != ["archive", "--format=zip", "--output"] or archive[-1] != OLD_COMMIT:
        _fail("stateful rollback archive is not the immutable old source")
    zip_path = _author_path(archive[4])
    suffix = "/local-only/v0.1.0.zip"
    if not zip_path.endswith(suffix):
        _fail("stateful rollback archive is not isolated local work")
    output = zip_path[:-len(suffix)]
    checkout = output + "/local-only/v01"
    old_manifest = checkout + "/out/toolchain.json"
    old_manifest_digest = checkout + "/out/toolchain.sha256"
    new_root = _author_path(new.get("project_root"))
    if _author_path(old.get("project_root")) != checkout or new_root == checkout:
        _fail("stateful rollback did not install a separate old checkout")
    old_runtime = old.get("runtime_profile_id")
    if old_runtime not in {"py312-win-amd64-cpu", "py312-linux-x86_64-cpu"}:
        _fail("stateful rollback old rehearsal must use the pinned host CPU runtime")
    old_environment = checkout + "/.illm/envs/rb01"
    python_suffix = "/Scripts/python.exe" if old_runtime == "py312-win-amd64-cpu" else "/bin/python"
    if (old.get("profile") != "rb01" or _author_path(old.get("environment_root")) != old_environment
            or _author_path(old.get("python_executable")) != old_environment + python_suffix
            or _author_path(old.get("python_executable")) == current_python):
        _fail("stateful rollback old interpreter is not in its newly extracted source environment")
    bootstrap = argv["bootstrap-old"]
    expected_bootstrap = [checkout + "/scripts/bootstrap.py", "--profile", "rb01", "--runtime-profile", "v01-cpu",
                          "--locked", "--offline", "--artifact-cache"]
    if (len(bootstrap) != 14 or bootstrap[1:9] != expected_bootstrap
            or bootstrap[10:] != ["--write-manifest", old_manifest, "--write-manifest-digest", old_manifest_digest]):
        _fail("stateful rollback old bootstrap command bypasses source-bound offline installation")
    _author_path(bootstrap[9])
    new_manifest: str | None = None
    old_capture, new_capture = checkout + "/out/reference-capture", output + "/new-capture"
    expected = {
        "new-run": ["python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
                    "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                    "--request-id", LAST_SAFE_REQUEST, "--kv-mode", "contiguous", "--telemetry", "request", "--capture", new_capture],
        "new-replay-primary": ["python", "-m", "illm", "replay", output + "/quarantine/capture", "--mode", "trace"],
        "new-inspect-primary": ["python", "-m", "illm", "inspect", "request", output + "/quarantine/capture", "--request", primary.manifest["request_id"]],
        "new-replay": ["python", "-m", "illm", "replay", new_capture, "--mode", "trace"],
        "new-inspect": ["python", "-m", "illm", "inspect", "request", new_capture, "--request", LAST_SAFE_REQUEST],
        "old-environment": ["python", "-I", "-c", ENVIRONMENT_PROBE],
        "old-generate": ["python", "-m", "illm", "generate", "--model", "fixture:tiny-decoder", "--device", "cpu",
                         "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                         "--request-id", OLD_REQUEST, "--telemetry", "request", "--capture", old_capture],
        "old-replay-trace": ["python", "-m", "illm", "replay", old_capture, "--mode", "trace"],
        "old-replay-state": ["python", "-m", "illm", "replay", old_capture, "--mode", "state", "--assert-state-digest"],
        "old-bundle-verify": ["python", "-m", "illm", "release", "verify", checkout + "/release/v0.1.0/manifest.json"],
        "old-reject-v02": ["python", "-m", "illm", "replay", output + "/quarantine/capture", "--mode", "trace"],
        "new-read-old": ["python", "-m", "illm", "replay", output + "/old-capture", "--mode", "state", "--assert-state-digest"],
    }
    for name, child in expected.items():
        tokens = argv[name]
        is_old = name.startswith("old-")
        source_root = checkout if is_old else new_root
        digest_file = old_manifest_digest if is_old else output + "/new-toolchain.sha256"
        if len(tokens) < 9 or tokens[1:3] != [source_root + "/scripts/in_env.py", "--manifest"] or tokens[4:7] != ["--expected-manifest-digest-file", digest_file, "--"]:
            _fail("stateful rollback command bypasses its manifest/digest wrapper")
        manifest = _author_path(tokens[3])
        if is_old:
            if manifest != old_manifest:
                _fail("stateful rollback old command entered a different toolchain")
        elif new_manifest is None:
            new_manifest = manifest
        elif new_manifest != manifest:
            _fail("stateful rollback new commands entered different toolchains")
        if tokens[7:] != child:
            _fail("stateful rollback executed a different command than its claimed phase")


def _bound_capture(bundle: StreamingCaptureBundle, toolchain: Mapping[str, Any]) -> None:
    identity = bundle.manifest["identity"]
    if (identity.get("toolchain_bound") is not True or identity.get("toolchain_digest") != toolchain["manifest_digest"]
            or identity.get("engine_version") != toolchain["installed_packages"]["illm"]):
        _fail("stateful rollback capture is not bound to the new finalized toolchain")
    story = bundle.request_story(str(bundle.manifest["request_id"]))
    if (story["terminal_outcome"] not in {"FINISHED", "CANCELLED"}
            or story["kv"]["final_active_allocation_bytes"] != 0 or story["kv"]["final_open_write_count"] != 0):
        _fail("stateful rollback capture has no safe terminal/fence reclamation")


def _verify(root: Path, *, synthetic_test_fixture: bool = False) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir() or not _fs(root / "COMPLETE").is_file() or _fs(root / "INCOMPLETE").exists():
        _fail("stateful rollback evidence is incomplete")
    root = _fs(root).resolve()
    # Reject FIFOs/devices/links and enforce the byte budget before opening an
    # attacker-controlled JSON entry point. Reuse the exact checked inventory.
    inventory = portable_stateful_rollback_inventory(root)
    report = _object(root / "rollback-report.json")
    _signed(report, "report_digest")
    expected_mode, expected_status = ("synthetic_test_fixture", "synthetic_pass") if synthetic_test_fixture else ("release", "pass")
    if (type(report.get("schema_version")) is not int or report.get("schema_version") != 1
            or report.get("report_kind") != REPORT_KIND or report.get("status") != expected_status
            or report.get("evidence_mode") != expected_mode or report.get("old_release_tag") != OLD_TAG
            or report.get("old_release_commit") != OLD_COMMIT or report.get("old_release_tag_object") != OLD_TAG_OBJECT
            or report.get("rollback_kind") != "isolated-single-request-artifact-rehearsal"):
        _fail("stateful rollback is not a release rehearsal of the immutable V0.1 binary")
    if (root / "COMPLETE").read_text(encoding="ascii") != report["report_digest"] + "\n":
        _fail("stateful rollback completion marker is stale")
    if _object(root / "checksums.json") != inventory or report.get("files") != {key: value for key, value in inventory.items() if key != "rollback-report.json"}:
        _fail("stateful rollback sealed inventory is not exact")
    source, versions = _verify_v02_source(root / "new-source", None)
    new = _verify_toolchain(root / "new-source", source, versions)
    old_source, old_versions = _verify_source(root / "old-source", OLD_SOURCE)
    old = _verify_toolchain(root / "old-source", old_source, old_versions)
    if not synthetic_test_fixture:
        for toolchain in (new, old):
            environment_id = toolchain.get("environment_id")
            try:
                identifier = uuid.UUID(environment_id) if isinstance(environment_id, str) else None
            except ValueError:
                identifier = None
            if identifier is None or str(identifier) != environment_id or identifier.version != 4:
                _fail("stateful rollback release runtime lacks a fresh UUID4 environment identity")
    if (old["installed_packages"].get("illm") != "0.1.0" or report.get("source_identity") != source
            or report.get("primary_v02_toolchain_manifest_digest") != new["manifest_digest"]
            or report.get("live_v02_source_verified") is not True or new["environment_id"] == old["environment_id"]):
        _fail("stateful rollback source/toolchain identities disagree")
    primary = StreamingCaptureBundle.open(root / "quarantine/capture")
    rehearsal = StreamingCaptureBundle.open(root / "new-capture")
    for capture in (primary, rehearsal):
        _bound_capture(capture, new)
    if report.get("capture_manifest_digest") != primary.manifest["manifest_digest"] or report.get("rehearsal_capture_manifest_digest") != rehearsal.manifest["manifest_digest"]:
        _fail("stateful rollback captures differ from the bound report")
    commands = _object(root / "execution.json").get("commands")
    if not isinstance(commands, list) or not all(isinstance(item, dict) for item in commands):
        _fail("stateful rollback lacks actual execution records")
    _verify_commands(commands, new, old, primary)
    outputs: dict[str, dict[str, Any]] = {}
    for item in commands:
        name = item["phase"]
        expected = 78 if name == "old-reject-v02" else 0
        if type(item.get("returncode")) is not int or item["returncode"] != expected:
            _fail("stateful rollback command failed or schema2 rejection did not occur")
        for stream in ("stdout", "stderr"):
            link = _relative(item.get(stream + "_file"))
            if link != f"logs/{name}.{stream}.txt" or item.get(stream + "_sha256") != sha256_file(root / link):
                _fail("stateful rollback transcript checksum/link mismatch")
        if name not in {"resolve-tag", "archive"}:
            outputs[name] = _last_json(root / item["stderr_file" if expected else "stdout_file"])
    if (root / "logs/resolve-tag.stdout.txt").read_text(encoding="ascii").splitlines() != [OLD_TAG_OBJECT, OLD_COMMIT]:
        _fail("stateful rollback did not resolve the fixed annotated old tag")
    if outputs["bootstrap-old"] != old:
        _fail("stateful rollback old bootstrap stdout differs from installed manifest")
    probe = outputs["old-environment"]
    if (probe.get("illm_version") != "0.1.0" or probe.get("stateful_engine_present") is not False
            or probe.get("environment_id") != old["environment_id"] or probe.get("toolchain_digest") != old["manifest_digest"]):
        _fail("stateful rollback old binary still contains V0.2 implementation")
    for name in ("new-replay-primary", "new-inspect-primary"):
        if not _same(outputs[name], primary.request_story(str(primary.manifest["request_id"]))):
            _fail("stateful rollback new reader did not read the preserved primary capture")
    story = rehearsal.request_story(LAST_SAFE_REQUEST)
    for name in ("new-replay", "new-inspect"):
        if not _same(outputs[name], story):
            _fail("stateful rollback new request story does not match canonical capture")
    result = outputs["new-run"]
    resources = result.get("resource_status")
    if (type(result.get("schema_version")) is not int or result.get("schema_version") != 2
            or result.get("request_id") != LAST_SAFE_REQUEST or result.get("outcome") != "FINISHED"
            or result.get("finish_reason") != "MAX_NEW_TOKENS" or result.get("error_code") is not None
            or result.get("capture_error") is not None or result.get("capture_manifest_digest") != rehearsal.manifest["manifest_digest"]
            or not isinstance(resources, dict) or set(resources) != RESOURCE_FIELDS
            or any(type(value) is not int or value != 0 for value in resources.values())
            or report.get("resource_status") != resources or report.get("last_safe_request_id") != LAST_SAFE_REQUEST):
        _fail("stateful rollback did not prove terminal completion and zero runtime resources")
    if (result.get("kv_mode") != "contiguous" or result.get("query_tokens_per_step") != [6, 1, 1, 1, 1, 1, 1, 1]
            or result.get("kv_valid_tokens_per_step") != list(range(6, 14))
            or len(story["tokens"]) != 8 or len(story["kv"]["releases"]) != 1):
        _fail("stateful rollback did not actually allocate/use/reclaim contiguous KV")
    old_capture = ReferenceCaptureBundle.open(root / "old-capture")
    old_result = outputs["old-generate"]
    for field in ("input_token_ids", "output_token_ids", "text", "finish_reason", "actual_seed", "model_spec_id",
                  "sampling_spec_digest", "next_output_position", "rng_draws_consumed", "committed_draw_addresses"):
        if not _same(result.get(field), old_result.get(field)):
            _fail("stateful rollback old reference output differs from the new tiny safe request")
    if (len(result.get("output_token_ids", [])) != 8 or result.get("rng_draws_consumed") != 0
            or result.get("next_output_position") != 8 or result.get("committed_draw_addresses") != []
            or old_result.get("request_id") != OLD_REQUEST or old_result.get("capture_manifest_digest") != old_capture.manifest["manifest_digest"]
            or old_capture.final_state.get("status") != "FINISHED"):
        _fail("stateful rollback old capture/result identity differs from the actual execution")
    for field in ("model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest"):
        if old_capture.manifest["identity"].get(field) != rehearsal.manifest["identity"].get(field):
            _fail("stateful rollback old and new captures use different model/weight/tokenizer identities")
    trace = outputs["old-replay-trace"]
    expected_state = {"mode": "R1_REFERENCE", "asserted": True, "numerical_reexecution": False,
                      "final_state_digest": old_capture.manifest["final_state_digest"], "status": "FINISHED",
                      "output_tokens": old_capture.final_state["output_tokens"], "rng_draws_consumed": 0}
    if (trace.get("mode") != "R0" or type(trace.get("total_events")) is not int
            or trace.get("total_events") != len(old_capture.events) or trace.get("displayed_events") != len(old_capture.events)
            or not _same(outputs["old-replay-state"], expected_state) or not _same(outputs["new-read-old"], expected_state)):
        _fail("stateful rollback old/new readers did not replay the same old reference facts")
    bundle = verify_reference_release_bundle(root / "old-release-bundle/manifest.json", expected_source_identity=OLD_SOURCE)
    if bundle["manifest_digest"] != OLD_BUNDLE or outputs["old-bundle-verify"] != bundle or report.get("old_bundle_manifest_digest") != OLD_BUNDLE:
        _fail("stateful rollback old release bundle is not the immutable published artifact")
    if outputs["old-reject-v02"].get("code") != "UNSUPPORTED_SCHEMA":
        _fail("stateful rollback old reader did not fail closed on schema2")
    quarantine = _object(root / "quarantine-inventory.json")
    actual = {name: digest for name, digest in inventory.items() if name.startswith("quarantine/")}
    primary_files = {"quarantine/capture/" + path.name for path in primary.root.iterdir()}
    if set(quarantine) != set(actual) or set(actual) != primary_files:
        _fail("stateful rollback quarantine inventory omits preserved facts")
    for name, receipt in quarantine.items():
        if (not isinstance(receipt, dict) or receipt != {"sha256": actual[name], "source_sha256_before": actual[name],
                "source_sha256_after": actual[name], "read_only_at_creation": True}):
            _fail("stateful rollback modified or failed to preserve its primary capture")
    archive = report.get("archive")
    if (not isinstance(archive, dict) or archive.get("git_archive_comment") != OLD_COMMIT or archive.get("embedded") is not False
            or archive.get("local_relative_path") != "local-only/v0.1.0.zip"
            or not isinstance(archive.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]) is None
            or type(archive.get("bytes")) is not int or not 0 < archive["bytes"] <= MAX_BYTES
            or archive.get("source_identity") != OLD_SOURCE):
        _fail("stateful rollback archive does not bind the immutable old source")
    if (not isinstance(report.get("trigger_reason"), str) or not report["trigger_reason"].strip()
            or report.get("config") != CONFIG or report.get("config_digest") != sha256_json(CONFIG)
            or report.get("memory_plan_digest") != rehearsal.manifest["memory_plan_digest"]
            or report.get("original_sources_unchanged") is not True
            or type(report.get("new_requests_after_safe_terminal")) is not int or report.get("new_requests_after_safe_terminal") != 0
            or type(report.get("unrelated_processes_interrupted")) is not int or report.get("unrelated_processes_interrupted") != 0
            or report.get("capabilities_lost") != ["stateful_streaming", "contiguous_kv", "schema2_R0", "request_story", "streaming_metrics"]):
        _fail("stateful rollback omits trigger/config/safe boundary or overclaims service mutation")
    expected_roots = {"new-source", "old-source", "quarantine", "new-capture", "old-capture", "old-release-bundle", "logs"}
    expected_files = {"rollback-report.json", "execution.json", "commands.txt", "rollback-report.md", "manifest.json",
                      "quarantine-inventory.json", "new-toolchain.sha256"}
    actual_roots = {name.split("/", 1)[0] for name in inventory if "/" in name}
    if actual_roots != expected_roots or {name for name in inventory if "/" not in name} != expected_files:
        _fail("stateful rollback contains unreferenced portable artifacts")
    authority_files = {"toolchain.json", "source-provenance.json", "locks/requirements-bootstrap.lock",
                       "locks/requirements-dev.lock", "locks/runtime-lock.json", "locks/pyproject.toml"}
    for name in ("new-source", "old-source"):
        if {path[len(name) + 1:] for path in inventory if path.startswith(name + "/")} != authority_files:
            _fail("stateful rollback source authority inventory is not exact")
    if {name for name in inventory if name.startswith("logs/")} != {
            f"logs/{phase}.{stream}.txt" for phase in PHASES for stream in ("stdout", "stderr")}:
        _fail("stateful rollback transcript inventory is not exact")
    manifest = _object(root / "manifest.json")
    if manifest != {"schema_version": 1, "report_kind": REPORT_KIND, "old_release_tag": OLD_TAG,
                    "old_release_commit": OLD_COMMIT, "source_identity": source,
                    "primary_v02_toolchain_manifest_digest": new["manifest_digest"],
                    "capture_manifest_digest": primary.manifest["manifest_digest"], "config_digest": sha256_json(CONFIG)}:
        _fail("stateful rollback manifest differs from verified facts")
    if (root / "new-toolchain.sha256").read_text(encoding="ascii") != new["manifest_digest"] + "\n":
        _fail("stateful rollback new wrapper digest authority is stale")
    return report


def verify_stateful_rollback(root: Path) -> dict[str, Any]:
    """Audit portable, actual rollback receipts without author paths or torch."""
    try:
        return _verify(root)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "stateful rollback evidence has invalid structure",
                        {"exception_type": type(exc).__name__}) from exc


__all__ = ["verify_stateful_rollback", "portable_stateful_rollback_inventory"]

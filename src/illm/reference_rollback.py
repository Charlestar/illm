"""Read portable rollback facts without executing or locating the old environment."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, NoReturn, cast

from illm.canonical import sha256_file, sha256_json
from illm.capture import CaptureBundle
from illm.capture.reference import ReferenceCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.release import verify_release_bundle
from illm.replay import replay_events
from illm.sampling import SamplingSpec


PREVIOUS_TAG = "v0.0.0"
PREVIOUS_COMMIT = "02f7068be26ed921b56857b563312177b7c2582d"
PREVIOUS_TAG_OBJECT = "0304a400834648bc2d948cf81b501b2e46a133e5"
V00_SOURCE = "1ed0f463a78ce502efc3d2d25ecd2c5d79e697bb6672ace851ffa4ff76459b29"
V00_BOOTSTRAP = "bdd072cd00b8149d942fed2a7310fa717bb755b2b18570c066925202e8e4a399"
V00_LOCK = "19489d7dd52ec0f54bbd45b784add2453453523008ab46550dea8bf989cef274"
V00_RELEASE = "b68f1e18036c4807ad3353d8ee83f6c38502365f5b1ef9a9c602b5bcad066fb5"
V00_STATE = "8678d4d2a8442c7b56830d326fc7874f5473e92357883e7402e38a2545420a55"
PHASES = ("resolve-tag", "archive", "bootstrap", "environment", "toy", "replay-trace", "replay-state",
          "bundle-verify", "reject-v01-capture", "reject-generate", "reject-model")
TINY_WEIGHT_SHA256 = "8eeaeeb38565ca8ac40177f239a8405396a2b6642aef5223b5bf09205ffb056a"


def _fs(path: Path) -> Path:
    raw = str(path.absolute())
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)
    return path


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate JSON field in rollback evidence")
            result[key] = value
        return result

    try:
        value = json.loads(_fs(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda item: _fail("nonfinite rollback evidence"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback JSON evidence is unreadable", {"file": path.name}) from exc
    if not isinstance(value, dict):
        _fail("rollback JSON evidence must be an object")
    return cast(dict[str, Any], value)


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        _fail("rollback evidence path is not portable")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(item in {".", ".."} for item in path.parts):
        _fail("rollback evidence path escapes its sealed root")
    return value


def _signed(value: Mapping[str, Any], field: str) -> None:
    claimed = value.get(field)
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None or claimed != sha256_json({key: item for key, item in value.items() if key != field}):
        _fail(f"rollback {field} mismatch")


def portable_rollback_inventory(root: Path) -> dict[str, str]:
    """Inventory published facts; explicitly omit regenerable local-only work."""
    if root.is_symlink() or not root.is_dir():
        _fail("rollback root must be an ordinary directory")
    result: dict[str, str] = {}
    scan = _fs(root)
    for directory in scan.iterdir():
        if directory.name == "local-only":
            continue
        if directory.is_symlink():
            _fail("symlinks are forbidden in portable rollback facts")
        paths = [directory] if directory.is_file() else list(directory.rglob("*"))
        for path in paths:
            if path.is_symlink():
                _fail("symlinks are forbidden in portable rollback facts")
            if path.is_file():
                name = _relative(path.relative_to(scan).as_posix())
                if not path.resolve().is_relative_to(scan.resolve()):
                    _fail("rollback fact escapes its sealed root")
                if name not in {"checksums.json", "COMPLETE", "INCOMPLETE"}:
                    result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _last_json(path: Path) -> dict[str, Any]:
    try:
        lines = [line for line in _fs(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        value = json.loads(lines[-1])
    except (OSError, ValueError, UnicodeError, IndexError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback phase lacks actual structured output") from exc
    if not isinstance(value, dict):
        _fail("rollback phase output must end in a JSON object")
    return cast(dict[str, Any], value)


def _verify_commands(phases: list[dict[str, Any]], old: Mapping[str, Any]) -> None:
    commands = {item["phase"]: [str(token).replace("\\", "/") for token in item["argv"]] for item in phases}
    if commands["resolve-tag"][1:] != ["rev-parse", f"refs/tags/{PREVIOUS_TAG}", f"refs/tags/{PREVIOUS_TAG}^{{commit}}"]:
        _fail("rollback tag phase did not resolve the immutable annotated tag")
    archive = commands["archive"]
    if len(archive) != 6 or archive[1:4] != ["archive", "--format=zip", "--output"] or archive[-1] != PREVIOUS_COMMIT or not archive[4].endswith("/local-only/v0.0.0.zip"):
        _fail("rollback archive command has unexpected semantics")
    output = archive[4][:-len("/local-only/v0.0.0.zip")]
    checkout = output + "/local-only/v00"
    manifest = checkout + "/out/toolchain.json"
    if str(old.get("project_root", "")).replace("\\", "/") != checkout:
        _fail("rollback old environment was not created inside isolated extracted source")
    if commands["bootstrap"][1:] != [checkout + "/scripts/bootstrap.py", "--profile", "rb00", "--locked", "--write-manifest", manifest]:
        _fail("rollback bootstrap command did not install only the minimal old source")
    toy = checkout + "/out/toy"
    expected = {
        "toy": ["python", "-m", "illm", "demo", "toy-scheduler", "--seed", "7", "--steps", "20", "--capture", toy],
        "replay-trace": ["python", "-m", "illm", "replay", toy, "--mode", "trace"],
        "replay-state": ["python", "-m", "illm", "replay", toy, "--mode", "state", "--assert-state-digest"],
        "bundle-verify": ["python", "-m", "illm", "release", "verify", checkout + "/release/v0.0.0/manifest.json"],
        "reject-v01-capture": ["python", "-m", "illm", "replay", output + "/quarantine/capture", "--mode", "state", "--assert-state-digest"],
        "reject-generate": ["python", "-m", "illm", "generate"],
        "reject-model": ["python", "-m", "illm", "model", "inspect", "--model", "fixture:tiny-decoder"],
    }
    for name in ("environment", *expected):
        argv = commands[name]
        if len(argv) < 6 or argv[1:5] != [checkout + "/scripts/in_env.py", "--manifest", manifest, "--"]:
            _fail("rollback phase bypassed the isolated old manifest runner")
        child = argv[5:]
        if name == "environment":
            if len(child) != 4 or child[:3] != ["python", "-I", "-c"] or any(text not in child[3] for text in ("find_spec('torch')", "find_spec('transformers')", "find_spec('illm.engine')")):
                _fail("rollback environment phase did not inspect lost runtime capabilities")
        elif child != expected[name]:
            _fail("rollback executed a different command than its claimed phase")


def verify_quarantined_validation(directory: Path) -> dict[str, Any]:
    """Verify preserved report links/identity, not rerun or qualify its numerics."""
    report = _object(directory / "validation-report.json")
    _signed(report, "report_digest")
    if type(report.get("schema_version")) is not int or report.get("schema_version") != 1 or report.get("status") != "pass" or report.get("fixed_input_count") != 20:
        _fail("quarantined validation is not a completed twenty-input report")
    model = _object(directory / "model-spec.json")
    spec_id = model.get("spec_id")
    if spec_id != sha256_json({key: item for key, item in model.items() if key != "spec_id"}):
        _fail("quarantined normalized model identity mismatch")
    weights = _object(directory / "weight-load-manifest.json")
    _signed(weights, "manifest_digest")
    identity = report.get("identity")
    if (not isinstance(identity, dict) or identity.get("spec_id") != spec_id
            or identity.get("weights_digest") != weights.get("weights_digest")
            or identity.get("load_manifest_digest") != weights.get("manifest_digest")
            or weights.get("spec_id") != spec_id or weights.get("finalized") is not True):
        _fail("quarantined model/weight/validation identities disagree")
    artifacts = report.get("model_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"model-spec.json", "weight-load-manifest.json", "tokenizer-spec.json"}:
        _fail("quarantined validation omits model manifests")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or record.get("path") != name or record.get("sha256") != sha256_file(directory / name):
            _fail("quarantined model artifact checksum mismatch")
    greedy = report.get("greedy_reports")
    ops = report.get("per_op_reports")
    if (not isinstance(greedy, list) or len(greedy) != 20 or not isinstance(ops, list) or len(ops) != 20
            or {item.get("public_input_fixture_id") for item in greedy if isinstance(item, dict)} != {f"fixed-greedy-{index:02d}" for index in range(20)}):
        _fail("quarantined validation cases are incomplete")
    for item in greedy:
        if (not isinstance(item, dict) or item.get("status") != "pass" or not item.get("reference_tokens")
                or item.get("reference_tokens") != item.get("oracle_tokens")
                or item.get("generated_count") != len(item["reference_tokens"])):
            _fail("quarantined greedy evidence is not an exact token comparison")
    for item in ops:
        if not isinstance(item, dict) or item.get("status") != "pass" or not item.get("ops"):
            _fail("quarantined op evidence is incomplete")
        link = _relative(item.get("report_link"))
        if _object(directory / link) != item:
            _fail("quarantined op report link differs from its summary")
    for name in ("first_divergence_diagnostic", "token_mismatch_diagnostic"):
        diagnostic = report.get(name)
        if not isinstance(diagnostic, dict) or diagnostic.get("status") != "pass":
            _fail("quarantined diagnostic evidence is incomplete")
        if _object(directory / _relative(diagnostic.get("report_link"))) != diagnostic.get("report"):
            _fail("quarantined diagnostic report link mismatch")
    return report


def _verify_reference_rollback(root: Path) -> dict[str, Any]:
    if root.is_symlink():
        _fail("rollback root must not be a symlink")
    root = root.resolve()
    if not (root / "COMPLETE").is_file() or (root / "INCOMPLETE").exists():
        _fail("rollback evidence is incomplete")
    report = _object(root / "rollback-report.json")
    _signed(report, "report_digest")
    if ((root / "COMPLETE").read_text(encoding="ascii").strip() != report["report_digest"]
            or report.get("schema_version") != 1 or report.get("status") != "pass"
            or report.get("previous_tag") != PREVIOUS_TAG or report.get("previous_commit") != PREVIOUS_COMMIT
            or report.get("previous_tag_object") != PREVIOUS_TAG_OBJECT
            or report.get("evidence_mode") not in {"development", "release"}
            or report.get("rollback_kind") != "isolated-artifact-rehearsal"):
        _fail("rollback target/state is not the immutable V0.0 rehearsal contract")
    inventory = portable_rollback_inventory(root)
    if _object(root / "checksums.json") != inventory:
        _fail("rollback checksum inventory is not exact")
    if report.get("files") != {key: value for key, value in inventory.items() if key != "rollback-report.json"}:
        _fail("rollback report inventory differs from sealed facts")
    phases = _object(root / "execution.json").get("commands")
    if not isinstance(phases, list) or tuple(item.get("phase") for item in phases if isinstance(item, dict)) != PHASES:
        _fail("rollback does not contain every executed phase exactly once")
    outputs: dict[str, dict[str, Any]] = {}
    for phase in phases:
        name = phase["phase"]
        expected_exit = 78 if name == "reject-v01-capture" else 64 if name in {"reject-generate", "reject-model"} else 0
        if type(phase.get("returncode")) is not int or phase["returncode"] != expected_exit:
            _fail("rollback phase failed or expected rejection was not observed")
        argv = phase.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            _fail("rollback phase command is missing")
        for stream in ("stdout", "stderr"):
            link = _relative(phase.get(f"{stream}_file"))
            if not link.startswith("logs/") or phase.get(f"{stream}_sha256") != sha256_file(root / link):
                _fail("rollback phase transcript checksum mismatch")
        if name not in {"resolve-tag", "archive"}:
            outputs[name] = _last_json(root / phase["stderr_file" if expected_exit else "stdout_file"])
    if (root / phases[0]["stdout_file"]).read_text(encoding="utf-8").splitlines() != [PREVIOUS_TAG_OBJECT, PREVIOUS_COMMIT]:
        _fail("local tag resolution did not produce the fixed annotated tag and commit")
    if PREVIOUS_COMMIT not in phases[1]["argv"] or "archive" not in phases[1]["argv"] or "--format=zip" not in phases[1]["argv"]:
        _fail("rollback archive was not built from the fixed commit")
    old = _object(root / "v00-toolchain.json")
    _verify_commands(phases, old)
    _signed(old, "manifest_digest")
    if (old.get("schema_version") != 0 or old.get("finalized") is not True
            or old.get("project_root_digest") != V00_SOURCE or old.get("bootstrap_digest") != V00_BOOTSTRAP
            or old.get("lock_digest") != V00_LOCK or old.get("installed_packages", {}).get("illm") != "0.0.0"
            or "torch" in old["installed_packages"] or "transformers" in old["installed_packages"]
            or outputs["bootstrap"] != old):
        _fail("rollback did not bootstrap the exact minimal V0.0 source and toolchain")
    environment = outputs["environment"]
    if (environment.get("illm_version") != "0.0.0" or environment.get("torch_present") is not False
            or environment.get("transformers_present") is not False or environment.get("engine_present") is not False
            or environment.get("environment_id") != old["environment_id"]):
        _fail("rollback environment retained V0.1 model/runtime capabilities")
    toy = CaptureBundle.open(root / "v00-capture")
    replayed = replay_events(toy.initial_state, toy.events, toy.expected_step_digests)
    if (len(toy.events) != 79 or toy.manifest.get("seed") != 7 or toy.manifest.get("steps") != 20
            or replayed.digest != V00_STATE
            or toy.manifest.get("final_state_digest") != V00_STATE
            or outputs["toy"].get("final_state_digest") != V00_STATE
            or outputs["replay-trace"].get("mode") != "R0" or outputs["replay-trace"].get("total_events") != 79
            or outputs["replay-state"].get("final_state_digest") != V00_STATE
            or outputs["replay-state"].get("asserted") is not True or outputs["replay-state"].get("steps") != 20
            or outputs["replay-state"].get("mode") != "R1_TOY"):
        _fail("rollback toy/R0/R1 facts do not match the executed V0.0 baseline")
    state = verify_release_bundle(root / "v00-release-bundle" / "manifest.json")
    if state.get("qualification_tier") != "functionally_qualified" or outputs["bundle-verify"] != state:
        _fail("rollback did not actually verify the complete old release bundle")
    if _object(root / "v00-release-bundle" / "manifest.json").get("manifest_digest") != V00_RELEASE:
        _fail("rollback old bundle is not the tagged V0.0 release")
    if outputs["reject-v01-capture"].get("code") != "UNSUPPORTED_SCHEMA":
        _fail("V0.0 reader did not fail closed on a V0.1 capture")
    for name, feature in (("reject-generate", "generate"), ("reject-model", "model")):
        if outputs[name].get("code") != "INVALID_ARGUMENT" or feature not in str(outputs[name].get("message", "")):
            _fail("V0.0 CLI did not demonstrate loss of the V0.1 capability")
    capture = ReferenceCaptureBundle.open(root / "quarantine" / "capture")
    if sha256_file(root / "quarantine" / "tiny-model.safetensors") != TINY_WEIGHT_SHA256:
        _fail("quarantine omits the immutable tiny model weights")
    sampling = _object(root / "quarantine" / "sampling-spec.json")
    if sampling != capture.manifest["sampling_spec"] or SamplingSpec.from_dict(sampling).to_dict() != sampling:
        _fail("quarantined sampling manifest is not the V0.1 capture's spec")
    v01 = _object(root / "quarantine" / "toolchain.json")
    _signed(v01, "manifest_digest")
    if (v01.get("schema_version") != 1 or v01.get("finalized") is not True
            or report.get("primary_v01_toolchain_manifest_digest") != v01["manifest_digest"]
            or report.get("capture_manifest_digest") != capture.manifest["manifest_digest"]):
        _fail("quarantined V0.1 toolchain identity mismatch")
    source_identity = report.get("source_identity")
    if (not isinstance(source_identity, dict) or set(source_identity) != {"project_root_digest", "bootstrap_digest", "base_lock_digest"}
            or any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in source_identity.values())
            or source_identity["project_root_digest"] != v01.get("project_root_digest")
            or source_identity["bootstrap_digest"] != v01.get("bootstrap_digest")):
        _fail("rollback V0.1 source identity mismatch")
    if v01.get("lock_digest") != sha256_json({"base_lock_digest": source_identity["base_lock_digest"], "runtime_lock_digest": v01.get("runtime_lock_digest")}):
        _fail("rollback V0.1 base/runtime lock identity mismatch")
    if report["evidence_mode"] == "release" and report.get("live_v01_source_verified") is not True:
        _fail("release rollback lacks live source/manifest verification")
    validations = [verify_quarantined_validation(path) for path in sorted((root / "quarantine" / "validation").iterdir())]
    if not any(item["identity"]["spec_id"] == capture.manifest["identity"]["model_spec_id"]
               and item["identity"]["load_manifest_digest"] == capture.manifest["identity"]["load_manifest_digest"] for item in validations):
        _fail("quarantined capture does not bind a preserved model/weight validation")
    quarantine = _object(root / "quarantine-inventory.json")
    external_weights = quarantine.get("external_weights")
    if not isinstance(external_weights, dict) or external_weights != report.get("external_weights") or external_weights.get("embedded") is not False:
        _fail("rollback real-weight external artifact policy mismatch")
    if external_weights.get("provided") is True:
        if (external_weights.get("sha256") != "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
                or external_weights.get("bytes") != 988_097_824 or external_weights.get("source_unchanged") is not True
                or external_weights.get("read_only_at_creation") is not True
                or external_weights.get("local_relative_path") != "local-only/quarantine-weights/model.safetensors"):
            _fail("rollback external real-weight receipt is not the fixed model")
    elif external_weights.get("provided") is not False:
        _fail("rollback external real-weight receipt has no explicit disposition")
    files = quarantine.get("files")
    actual = {key: value for key, value in inventory.items() if key.startswith("quarantine/")}
    if not isinstance(files, dict) or set(files) != set(actual):
        _fail("quarantine source inventory does not cover exact preserved facts")
    for name, item in files.items():
        if (not isinstance(item, dict) or item.get("sha256") != actual[name] or item.get("source_sha256_before") != actual[name]
                or item.get("source_sha256_after") != actual[name] or item.get("read_only_at_creation") is not True):
            _fail("quarantine source changed or its read-only copy was not verified")
    if report.get("original_sources_unchanged") is not True or report.get("unrelated_processes_interrupted") != 0:
        _fail("rollback modified original V0.1 inputs or interrupted unrelated work")
    if report.get("capabilities_lost") != ["real_model", "generate", "sampling", "model_inspect", "model_validation", "R1_REFERENCE"]:
        _fail("rollback does not explicitly enumerate lost V0.1 capabilities")
    archive = report.get("archive")
    if (not isinstance(archive, dict) or archive.get("git_archive_comment") != PREVIOUS_COMMIT
            or archive.get("embedded") is not False or archive.get("local_relative_path") != "local-only/v0.0.0.zip"
            or not isinstance(archive.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]) is None
            or type(archive.get("bytes")) is not int or not 0 < archive["bytes"] <= 256 * 1024 * 1024):
        _fail("rollback archive receipt does not bind the immutable old artifact")
    return report


def verify_reference_rollback(root: Path) -> dict[str, Any]:
    """Verify embedded rollback facts; never access author environments or run a process."""
    try:
        return _verify_reference_rollback(root)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "rollback evidence has invalid structure", {"exception_type": type(exc).__name__}) from exc

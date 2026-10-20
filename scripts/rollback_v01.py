"""Rehearse an isolated, offline rollback to the immutable local V0.0 tag."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from typing import Any, NoReturn, Sequence
import zipfile

from _runtime_artifacts import isolated_environment
from _toolchain_common import atomic_write_json, atomic_write_text, bootstrap_digest, is_within, lock_digest, manifest_digest, project_digest, sha256_file
from illm.canonical import sha256_json
from illm.capture.reference import ReferenceCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_rollback import (
    PREVIOUS_COMMIT, PREVIOUS_TAG, PREVIOUS_TAG_OBJECT, portable_rollback_inventory,
    verify_quarantined_validation, verify_reference_rollback,
)


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_FILES = 4096
MAX_EVIDENCE_BYTES = 256 * 1024 * 1024
REAL_WEIGHT_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TINY_WEIGHT_SHA256 = "8eeaeeb38565ca8ac40177f239a8405396a2b6642aef5223b5bf09205ffb056a"


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _fs(path: Path) -> Path:
    raw = str(path.absolute())
    if os.name == "nt" and not raw.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + raw[2:] if raw.startswith("\\\\") else "\\\\?\\" + raw)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_fs(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail("rollback input must be a JSON object")
    return dict(value)


def _source_inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        _fail("rollback input directory must not be a symlink")
    scan = _fs(root)
    result: dict[str, str] = {}
    size = 0
    for path in scan.rglob("*"):
        if path.is_symlink():
            _fail("rollback input contains a symlink")
        if path.is_file():
            size += path.stat().st_size
            if size > MAX_EVIDENCE_BYTES:
                _fail("rollback evidence input exceeds the bounded copy budget")
            result[path.relative_to(scan).as_posix()] = sha256_file(path)
    return result


def extract_archive_safely(archive: Path, target: Path) -> dict[str, Any]:
    """Preflight all ZIP names/types/sizes before extracting into a new directory."""
    if target.exists() or target.is_symlink() or archive.is_symlink() or not archive.is_file():
        _fail("rollback extraction requires an ordinary archive and a new target")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        _fail("rollback archive exceeds the compressed byte budget")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members or len(members) > MAX_ARCHIVE_FILES or sum(item.file_size for item in members) > MAX_ARCHIVE_BYTES:
            _fail("rollback archive exceeds file count/uncompressed byte budget")
        names: set[str] = set()
        files: set[str] = set()
        for member in members:
            name = member.orig_filename.rstrip("/")
            parts = name.split("/")
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (member.orig_filename != member.filename or not name or "\\" in name or ":" in name or PurePosixPath(name).is_absolute()
                    or any(item in {"", ".", ".."} or item.rstrip(" .") != item for item in parts)
                    or any(item.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]} for item in parts)
                    or kind not in {0, stat.S_IFREG, stat.S_IFDIR} or member.flag_bits & 1
                    or name.casefold() in names):
                _fail("rollback archive has unsafe, duplicate, or nonregular members")
            names.add(name.casefold())
            if not member.is_dir():
                files.add(name.casefold())
        if any(any("/".join(name.split("/")[:index]) in files for index in range(1, len(name.split("/")))) for name in names):
            _fail("rollback archive aliases a file as a directory")
        target.mkdir(parents=True)
        for member in members:
            destination = target / member.filename.rstrip("/")
            if not is_within(destination, target):
                _fail("rollback archive member escapes extraction target")
            if member.is_dir():
                _fs(destination).mkdir(parents=True, exist_ok=True)
                continue
            _fs(destination.parent).mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, _fs(destination).open("xb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if _fs(destination).stat().st_size != member.file_size:
                _fail("rollback extracted member byte size mismatch")
    return {"sha256": sha256_file(archive), "bytes": archive.stat().st_size, "file_count": len(files),
            "uncompressed_bytes": sum(item.file_size for item in members), "embedded": False,
            "local_relative_path": "local-only/v0.0.0.zip"}


def _copy_tree(source: Path, destination: Path) -> dict[str, str]:
    before = _source_inventory(source)
    for relative in before:
        target = destination / relative
        _fs(target.parent).mkdir(parents=True, exist_ok=True)
        with _fs(source / relative).open("rb") as input_stream, _fs(target).open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
        if sha256_file(_fs(target)) != before[relative]:
            _fail("rollback copy checksum mismatch")
    if _source_inventory(source) != before:
        _fail("rollback input changed during its preservation copy")
    return before


def _readonly(path: Path) -> bool:
    _fs(path).chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return not bool(_fs(path).stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def run_rollback(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v01-capture", type=Path, required=True)
    parser.add_argument("--v01-toolchain-manifest", type=Path, required=True)
    parser.add_argument("--validation-directory", type=Path, action="append", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--evidence-mode", choices=("development", "release"), default="development")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not is_within(output, project / "out") or output == (project / "out").resolve():
        _fail("rollback output must be a new directory strictly below workspace out/")
    if any(path.is_symlink() for path in [args.v01_capture, args.v01_toolchain_manifest, *args.validation_directory]):
        _fail("rollback inputs must not be symlink aliases")
    capture_path = args.v01_capture.resolve()
    toolchain_path = args.v01_toolchain_manifest.resolve()
    validations = [path.resolve() for path in args.validation_directory]
    for path in [capture_path, toolchain_path, *validations]:
        if not is_within(path, project) or path == output or is_within(output, path):
            _fail("rollback inputs must be existing workspace facts separate from output")
    capture = ReferenceCaptureBundle.open(capture_path)
    reports = [verify_quarantined_validation(path) for path in validations]
    if not any(item["identity"]["spec_id"] == capture.manifest["identity"]["model_spec_id"]
               and item["identity"]["load_manifest_digest"] == capture.manifest["identity"]["load_manifest_digest"] for item in reports):
        _fail("V0.1 capture does not bind supplied model/weight numerical evidence")
    toolchain = _read_json(toolchain_path)
    if toolchain.get("schema_version") != 1 or toolchain.get("finalized") is not True or toolchain.get("manifest_digest") != manifest_digest(toolchain):
        _fail("rollback requires a finalized V0.1 runtime manifest")
    live_source = toolchain.get("project_root_digest") == project_digest(project) and toolchain.get("bootstrap_digest") == bootstrap_digest(project)
    if args.evidence_mode == "release":
        if not live_source or os.environ.get("ILLM_ENVIRONMENT_ID") != toolchain.get("environment_id") or Path(sys.executable).resolve() != Path(toolchain["python_executable"]).resolve():
            _fail("release rollback must enter the current source-bound V0.1 finalized environment")
    base_lock = lock_digest(project)
    source_identity = {"project_root_digest": toolchain["project_root_digest"], "bootstrap_digest": toolchain["bootstrap_digest"], "base_lock_digest": base_lock}
    if toolchain.get("lock_digest") != sha256_json({"base_lock_digest": base_lock, "runtime_lock_digest": toolchain.get("runtime_lock_digest")}):
        _fail("V0.1 toolchain does not bind the current base/runtime lock authority")
    original_toolchain_sha = sha256_file(toolchain_path)
    sources = [(capture_path, _source_inventory(capture_path)), *[(path, _source_inventory(path)) for path in validations]]
    if args.weights and args.weights.is_symlink():
        _fail("rollback weights must not be a symlink alias")
    weight_source = args.weights.resolve() if args.weights else None
    if weight_source and (not is_within(weight_source, project) or weight_source.is_symlink() or not weight_source.is_file()
                          or weight_source.stat().st_size != 988_097_824 or sha256_file(weight_source) != REAL_WEIGHT_SHA256):
        _fail("--weights must name the unchanged fixed real-model safetensors cache")
    output.mkdir(parents=True)
    atomic_write_text(output / "INCOMPLETE", "rollback_in_progress\n")
    (output / "logs").mkdir()
    local = output / "local-only"
    local.mkdir()
    quarantine = output / "quarantine"
    _copy_tree(capture_path, quarantine / "capture")
    (quarantine / "toolchain.json").write_bytes(toolchain_path.read_bytes())
    for index, path in enumerate(validations):
        _copy_tree(path, quarantine / "validation" / f"model-{index}")
    atomic_write_json(quarantine / "sampling-spec.json", capture.manifest["sampling_spec"])
    tiny = project / "fixtures" / "tiny-decoder" / "model.safetensors"
    if tiny.is_file() and sha256_file(tiny) == TINY_WEIGHT_SHA256:
        (quarantine / "tiny-model.safetensors").write_bytes(tiny.read_bytes())
    external_weights: dict[str, Any] = {"provided": False, "embedded": False, "policy": "original real-model cache is never mounted as a V0.0 model"}
    if weight_source:
        target = local / "quarantine-weights" / "model.safetensors"
        target.parent.mkdir()
        shutil.copyfile(weight_source, target)
        if sha256_file(target) != REAL_WEIGHT_SHA256 or sha256_file(weight_source) != REAL_WEIGHT_SHA256 or not _readonly(target):
            _fail("real-weight quarantine copy failed integrity/read-only verification")
        external_weights = {"provided": True, "embedded": False, "sha256": REAL_WEIGHT_SHA256, "bytes": 988_097_824,
                            "local_relative_path": target.relative_to(output).as_posix(), "source_unchanged": True, "read_only_at_creation": True}
    quarantine_files: dict[str, Any] = {}
    for path in quarantine.rglob("*"):
        if path.is_file():
            checksum = sha256_file(path)
            readonly = _readonly(path)
            if not readonly:
                _fail("V0.1 quarantine copy did not become read-only")
            quarantine_files[path.relative_to(output).as_posix()] = {"sha256": checksum, "source_sha256_before": checksum,
                    "source_sha256_after": checksum, "read_only_at_creation": readonly,
                    "origin_kind": "capture_sampling_projection" if path.name == "sampling-spec.json" else "byte_preserving_copy"}
    atomic_write_json(output / "quarantine-inventory.json", {"schema_version": 1, "files": quarantine_files, "external_weights": external_weights,
                      "numerical_policy": "preserve_only_no_V0.0_numerical_interpretation"})
    commands: list[dict[str, Any]] = []
    environment = isolated_environment()
    for key in list(environment):
        if key.startswith("ILLM_") or key.startswith("GIT_") or key == "VIRTUAL_ENV":
            environment.pop(key, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", GIT_CEILING_DIRECTORIES=str(local))
    base_python = str(getattr(sys, "_base_executable", sys.executable))

    def phase(name: str, command: list[str], *, cwd: Path, expected: int = 0) -> subprocess.CompletedProcess[bytes]:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, check=False)
        stdout, stderr = output / "logs" / f"{name}.stdout.txt", output / "logs" / f"{name}.stderr.txt"
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        commands.append({"phase": name, "argv": command, "returncode": process.returncode,
                         "stdout_file": stdout.relative_to(output).as_posix(), "stderr_file": stderr.relative_to(output).as_posix(),
                         "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr)})
        atomic_write_json(output / "execution.json", {"schema_version": 1, "commands": commands})
        print(json.dumps({"phase": name, "returncode": process.returncode, "expected_returncode": expected}), flush=True)
        if process.returncode != expected:
            _fail(f"rollback phase {name} produced an unexpected exit status; inspect preserved transcripts")
        return process

    git = shutil.which("git")
    if git is None:
        _fail("local Git executable is unavailable; rollback never fetches a replacement")
    identity = phase("resolve-tag", [git, "rev-parse", f"refs/tags/{PREVIOUS_TAG}", f"refs/tags/{PREVIOUS_TAG}^{{commit}}"], cwd=project)
    if identity.stdout.decode("ascii").splitlines() != [PREVIOUS_TAG_OBJECT, PREVIOUS_COMMIT]:
        _fail("local rollback tag is missing or no longer names the immutable release")
    archive = local / "v0.0.0.zip"
    phase("archive", [git, "archive", "--format=zip", "--output", str(archive), PREVIOUS_COMMIT], cwd=project)
    with zipfile.ZipFile(archive) as generated:
        if generated.comment.decode("ascii") != PREVIOUS_COMMIT:
            _fail("Git archive ZIP comment does not bind the fixed commit")
    checkout = local / "v00"
    archive_record = extract_archive_safely(archive, checkout)
    archive_record["git_archive_comment"] = PREVIOUS_COMMIT
    old_manifest = checkout / "out" / "toolchain.json"
    phase("bootstrap", [base_python, str(checkout / "scripts" / "bootstrap.py"), "--profile", "rb00", "--locked", "--write-manifest", str(old_manifest)], cwd=checkout)
    shutil.copyfile(old_manifest, output / "v00-toolchain.json")
    runner = [base_python, str(checkout / "scripts" / "in_env.py"), "--manifest", str(old_manifest), "--"]
    probe = ("import importlib.metadata as m,importlib.util as u,json,os;"
             "print(json.dumps({'illm_version':m.version('illm'),'torch_present':u.find_spec('torch') is not None,"
             "'transformers_present':u.find_spec('transformers') is not None,'engine_present':u.find_spec('illm.engine') is not None,"
             "'environment_id':os.environ.get('ILLM_ENVIRONMENT_ID')},sort_keys=True))")
    phase("environment", [*runner, "python", "-I", "-c", probe], cwd=checkout)
    old_capture = checkout / "out" / "toy"
    phase("toy", [*runner, "python", "-m", "illm", "demo", "toy-scheduler", "--seed", "7", "--steps", "20", "--capture", str(old_capture)], cwd=checkout)
    phase("replay-trace", [*runner, "python", "-m", "illm", "replay", str(old_capture), "--mode", "trace"], cwd=checkout)
    phase("replay-state", [*runner, "python", "-m", "illm", "replay", str(old_capture), "--mode", "state", "--assert-state-digest"], cwd=checkout)
    old_release = checkout / "release" / "v0.0.0"
    phase("bundle-verify", [*runner, "python", "-m", "illm", "release", "verify", str(old_release / "manifest.json")], cwd=checkout)
    phase("reject-v01-capture", [*runner, "python", "-m", "illm", "replay", str(quarantine / "capture"), "--mode", "state", "--assert-state-digest"], cwd=checkout, expected=78)
    phase("reject-generate", [*runner, "python", "-m", "illm", "generate"], cwd=checkout, expected=64)
    phase("reject-model", [*runner, "python", "-m", "illm", "model", "inspect", "--model", "fixture:tiny-decoder"], cwd=checkout, expected=64)
    _copy_tree(old_capture, output / "v00-capture")
    _copy_tree(old_release, output / "v00-release-bundle")
    if sha256_file(toolchain_path) != original_toolchain_sha or any(_source_inventory(path) != before for path, before in sources):
        _fail("original V0.1 evidence changed during the isolated rehearsal")
    if weight_source and sha256_file(weight_source) != REAL_WEIGHT_SHA256:
        _fail("original model weight cache changed during rehearsal")
    atomic_write_text(output / "commands.txt", "\n".join(subprocess.list2cmdline(item["argv"]) for item in commands) + "\n")
    atomic_write_text(output / "rollback-report.md", "# V0.1 rollback rehearsal\n\n"
                      "Fixed V0.0 source bootstrapped in a separate environment. Toy scheduler, R0/R1 and the old release bundle were actually executed.\n\n"
                      "V0.0 rejected the preserved V0.1 capture and has no generate/model commands. V0.1 evidence was copied read-only; original inputs and unrelated processes were not modified.\n\n"
                      f"Evidence mode: {args.evidence_mode}. This isolated artifact rehearsal does not stop or replace an independently running V0.1 service.\n")
    report: dict[str, Any] = {"schema_version": 1, "status": "pass", "previous_tag": PREVIOUS_TAG, "previous_commit": PREVIOUS_COMMIT,
        "previous_tag_object": PREVIOUS_TAG_OBJECT, "rollback_kind": "isolated-artifact-rehearsal", "evidence_mode": args.evidence_mode,
        "source_identity": source_identity, "primary_v01_toolchain_manifest_digest": toolchain["manifest_digest"],
        "capture_manifest_digest": capture.manifest["manifest_digest"], "live_v01_source_verified": bool(live_source and args.evidence_mode == "release"),
        "original_sources_unchanged": True, "unrelated_processes_interrupted": 0,
        "capabilities_lost": ["real_model", "generate", "sampling", "model_inspect", "model_validation", "R1_REFERENCE"],
        "archive": archive_record, "external_weights": external_weights,
        "files": portable_rollback_inventory(output)}
    report["report_digest"] = sha256_json(report)
    atomic_write_json(output / "rollback-report.json", report)
    atomic_write_json(output / "checksums.json", portable_rollback_inventory(output))
    atomic_write_text(output / "COMPLETE", report["report_digest"] + "\n")
    (output / "INCOMPLETE").unlink()
    try:
        return verify_reference_rollback(output)
    except Exception:
        (output / "COMPLETE").unlink(missing_ok=True)
        atomic_write_text(output / "INCOMPLETE", "portable_verification_failed\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_rollback(argv)
        print(json.dumps({"status": report["status"], "evidence_mode": report["evidence_mode"], "report_digest": report["report_digest"]}, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"code": "INVALID_ARGUMENT", "message": "rollback stopped without sealing success", "exception_type": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())

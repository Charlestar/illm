"""Actually rehearse V0.2 -> immutable V0.1 in a new, offline source checkout.

Run through the current source-bound scripts/in_env.py. This does not stop an
unrelated live service, switch an active request's backend, tag, or promote.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, NoReturn, Sequence
import zipfile

from _runtime_artifacts import isolated_environment
from _toolchain_common import atomic_write_json, atomic_write_text, is_within
from rollback_v01 import _copy_tree, _readonly, _source_inventory, extract_archive_safely
from illm.canonical import sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.reference_qualification import BASE_LOCKS, _assert_live_environment, _object, _source_provenance
from illm.stateful_rollback import (
    CONFIG, ENVIRONMENT_PROBE, LAST_SAFE_REQUEST, OLD_BUNDLE, OLD_COMMIT, OLD_REQUEST,
    OLD_SOURCE, OLD_TAG, OLD_TAG_OBJECT, REPORT_KIND, RESOURCE_FIELDS,
    _bound_capture, _last_json, portable_stateful_rollback_inventory, verify_stateful_rollback,
)


PROJECT = Path(__file__).resolve().parents[1]


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _preserve_authority(source: Path, toolchain: Mapping[str, Any], target: Path) -> dict[str, Any]:
    provenance = _source_provenance(source)
    target.mkdir(parents=True)
    atomic_write_json(target / "source-provenance.json", provenance)
    atomic_write_json(target / "toolchain.json", toolchain)
    locks = target / "locks"
    locks.mkdir()
    for name in (*BASE_LOCKS, "pyproject.toml"):
        shutil.copyfile(source / name, locks / name)
    shutil.copyfile(source / "runtime/artifacts" / f"{toolchain['runtime_profile_id']}.json", locks / "runtime-lock.json")
    return provenance


def run_rollback(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v02-primary-capture", type=Path, required=True)
    parser.add_argument("--v02-toolchain-manifest", type=Path, required=True)
    parser.add_argument("--trusted-git-repository", type=Path, required=True,
                        help="explicitly trusted local repository containing the immutable annotated v0.1.0 tag; never fetched")
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--trigger-reason", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or not is_within(output, PROJECT / "out") or output == (PROJECT / "out").resolve():
        _fail("rollback output must be a new directory strictly below workspace out/")
    if not args.trigger_reason.strip() or len(args.trigger_reason) > 1000:
        _fail("rollback trigger reason must be nonempty and at most1000 characters")
    for path in (args.v02_primary_capture, args.v02_toolchain_manifest, args.trusted_git_repository, args.artifact_cache):
        if path.is_symlink() or not path.exists():
            _fail("rollback inputs must be existing ordinary paths, never symlink aliases")
    primary_path = args.v02_primary_capture.resolve()
    manifest_path = args.v02_toolchain_manifest.resolve()
    if (not is_within(primary_path, PROJECT) or not is_within(manifest_path, PROJECT)
            or is_within(output, primary_path) or primary_path == output):
        _fail("rollback primary capture/toolchain must be independent existing workspace facts")
    toolchain = _object(manifest_path)
    if os.environ.get("ILLM_TOOLCHAIN_MANIFEST") is None or Path(os.environ["ILLM_TOOLCHAIN_MANIFEST"]).resolve() != manifest_path:
        _fail("rollback must run through the supplied current source-bound manifest")
    provenance = _assert_live_environment(toolchain, PROJECT)
    if not str(provenance.get("project_version", "")).startswith("0.2.0"):
        _fail("rollback producer requires current V0.2 source, not the main branch's old API")
    primary = StreamingCaptureBundle.open(primary_path)
    _bound_capture(primary, toolchain)
    original = _source_inventory(primary_path)
    original_manifest = sha256_file(manifest_path)
    git = shutil.which("git")
    if git is None:
        _fail("rollback requires local Git; it never fetches an old binary")
    output.mkdir(parents=True)
    atomic_write_text(output / "INCOMPLETE", "rollback_in_progress\n")
    logs = output / "logs"
    logs.mkdir()
    local = output / "local-only"
    local.mkdir()
    commands: list[dict[str, Any]] = []
    environment = isolated_environment()
    for key in list(environment):
        if key.startswith(("ILLM_", "GIT_")) or key in {"VIRTUAL_ENV", "PYTHONPATH"}:
            environment.pop(key, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The wrapper/bootstrap interpreter is itself bound by the new manifest;
    # in_env resolves its child to the independently installed old interpreter.
    base_python = str(Path(sys.executable).resolve())

    def phase(name: str, command: list[str], cwd: Path, expected: int = 0) -> None:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, check=False)
        stdout, stderr = logs / f"{name}.stdout.txt", logs / f"{name}.stderr.txt"
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        commands.append({"phase": name, "argv": command, "returncode": process.returncode,
                         "stdout_file": stdout.relative_to(output).as_posix(), "stderr_file": stderr.relative_to(output).as_posix(),
                         "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr)})
        atomic_write_json(output / "execution.json", {"schema_version": 1, "commands": commands})
        print(json.dumps({"phase": name, "returncode": process.returncode, "expected_returncode": expected}), flush=True)
        if process.returncode != expected:
            _fail(f"rollback phase {name} failed; raw transcripts and INCOMPLETE are preserved")

    repository = args.trusted_git_repository.resolve()
    phase("resolve-tag", [git, "rev-parse", f"refs/tags/{OLD_TAG}", f"refs/tags/{OLD_TAG}^{{commit}}"], repository)
    if (logs / "resolve-tag.stdout.txt").read_text(encoding="ascii").splitlines() != [OLD_TAG_OBJECT, OLD_COMMIT]:
        _fail("trusted local tag is absent or no longer names the immutable V0.1 release")
    archive = local / "v0.1.0.zip"
    phase("archive", [git, "archive", "--format=zip", "--output", str(archive), OLD_COMMIT], repository)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.comment.decode("ascii") != OLD_COMMIT:
            _fail("rollback Git archive ZIP comment does not bind the immutable commit")
    checkout = local / "v01"
    archive_record = extract_archive_safely(archive, checkout)
    # An archive has no .git: never let old metadata probes walk upward into
    # the enclosing V0.2 checkout and misattribute its revision to old source.
    # PROJECT is above, not below, this ceiling, so current-source commands
    # retain their own Git discovery boundary.
    environment["GIT_CEILING_DIRECTORIES"] = str(checkout.parent)
    old_provenance = _source_provenance(checkout)
    if old_provenance["source_identity"] != OLD_SOURCE or old_provenance["project_version"] != "0.1.0":
        _fail("immutable archive source/locks do not match the independently pinned V0.1 authority")
    archive_record.update(git_archive_comment=OLD_COMMIT, local_relative_path="local-only/v0.1.0.zip", source_identity=OLD_SOURCE)
    _preserve_authority(PROJECT, toolchain, output / "new-source")
    _copy_tree(primary_path, output / "quarantine/capture")
    quarantine: dict[str, Any] = {}
    for name, digest in original.items():
        target = output / "quarantine/capture" / name
        if not _readonly(target):
            _fail("rollback primary capture preservation did not become read-only")
        quarantine["quarantine/capture/" + name] = {"sha256": digest, "source_sha256_before": digest,
                "source_sha256_after": digest, "read_only_at_creation": True}
    atomic_write_json(output / "quarantine-inventory.json", quarantine)
    atomic_write_text(output / "new-toolchain.sha256", str(toolchain["manifest_digest"]) + "\n")
    current = [base_python, str(PROJECT / "scripts/in_env.py"), "--manifest", str(manifest_path),
               "--expected-manifest-digest-file", str(output / "new-toolchain.sha256"), "--"]
    new_capture = output / "new-capture"
    phase("new-run", [*current, "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
          "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
          "--request-id", LAST_SAFE_REQUEST, "--kv-mode", "contiguous", "--telemetry", "request", "--capture", str(new_capture)], PROJECT)
    result = _last_json(logs / "new-run.stdout.txt")
    resources = result.get("resource_status")
    if (not isinstance(resources, dict) or set(resources) != RESOURCE_FIELDS
            or any(type(value) is not int or value != 0 for value in resources.values()) or result.get("outcome") != "FINISHED"):
        _fail("rollback cannot launch the old binary before known terminal/fence/resource-baseline proof")
    for name, capture, request_id in (("primary", output / "quarantine/capture", str(primary.manifest["request_id"])),
                                      ("", new_capture, LAST_SAFE_REQUEST)):
        suffix = "-primary" if name else ""
        phase("new-replay" + suffix, [*current, "python", "-m", "illm", "replay", str(capture), "--mode", "trace"], PROJECT)
        phase("new-inspect" + suffix, [*current, "python", "-m", "illm", "inspect", "request", str(capture), "--request", request_id], PROJECT)
    # The extracted source directory did not exist before this rehearsal; the
    # old bootstrap may never replace an environment from another run.
    if (checkout / ".illm/envs/rb01").exists():
        _fail("rollback old environment unexpectedly already exists")
    old_manifest, old_digest = checkout / "out/toolchain.json", checkout / "out/toolchain.sha256"
    phase("bootstrap-old", [base_python, str(checkout / "scripts/bootstrap.py"), "--profile", "rb01", "--runtime-profile", "v01-cpu",
          "--locked", "--offline", "--artifact-cache", str(args.artifact_cache.resolve()), "--write-manifest", str(old_manifest),
          "--write-manifest-digest", str(old_digest)], checkout)
    old_toolchain = _object(old_manifest)
    _preserve_authority(checkout, old_toolchain, output / "old-source")
    old = [base_python, str(checkout / "scripts/in_env.py"), "--manifest", str(old_manifest),
           "--expected-manifest-digest-file", str(old_digest), "--"]
    phase("old-environment", [*old, "python", "-I", "-c", ENVIRONMENT_PROBE], checkout)
    old_capture = checkout / "out/reference-capture"
    phase("old-generate", [*old, "python", "-m", "illm", "generate", "--model", "fixture:tiny-decoder", "--device", "cpu",
          "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
          "--request-id", OLD_REQUEST, "--telemetry", "request", "--capture", str(old_capture)], checkout)
    phase("old-replay-trace", [*old, "python", "-m", "illm", "replay", str(old_capture), "--mode", "trace"], checkout)
    phase("old-replay-state", [*old, "python", "-m", "illm", "replay", str(old_capture), "--mode", "state", "--assert-state-digest"], checkout)
    old_release = checkout / "release/v0.1.0"
    phase("old-bundle-verify", [*old, "python", "-m", "illm", "release", "verify", str(old_release / "manifest.json")], checkout)
    phase("old-reject-v02", [*old, "python", "-m", "illm", "replay", str(output / "quarantine/capture"), "--mode", "trace"], checkout, 78)
    _copy_tree(old_capture, output / "old-capture")
    _copy_tree(old_release, output / "old-release-bundle")
    phase("new-read-old", [*current, "python", "-m", "illm", "replay", str(output / "old-capture"), "--mode", "state", "--assert-state-digest"], PROJECT)
    if (_source_inventory(primary_path) != original or sha256_file(manifest_path) != original_manifest
            or _source_provenance(PROJECT) != provenance or _source_provenance(checkout) != old_provenance):
        _fail("rollback source/capture/toolchain changed while the isolated rehearsal ran")
    rehearsal = StreamingCaptureBundle.open(new_capture)
    source = provenance["source_identity"]
    atomic_write_json(output / "manifest.json", {"schema_version": 1, "report_kind": REPORT_KIND,
          "old_release_tag": OLD_TAG, "old_release_commit": OLD_COMMIT, "source_identity": source,
          "primary_v02_toolchain_manifest_digest": toolchain["manifest_digest"], "capture_manifest_digest": primary.manifest["manifest_digest"],
          "config_digest": sha256_json(CONFIG)})
    atomic_write_text(output / "commands.txt", "\n".join(subprocess.list2cmdline(item["argv"]) for item in commands) + "\n")
    atomic_write_text(output / "rollback-report.md", "# V0.2 to V0.1 rollback rehearsal\n\n"
          "Isolated single-request artifact rehearsal, not a live-service deployment rollback.\n\n"
          "The current source-bound tiny request reached a safe terminal and zero runtime resources before the separately installed immutable V0.1 binary ran. "
          "Both readers replayed old schema1 facts; the old reader rejected schema2. No unrelated process was stopped, and no persistent KV was reused.\n\n"
          f"Trigger: {args.trigger_reason.strip()}\n")
    report: dict[str, Any] = {
        "schema_version": 1, "report_kind": REPORT_KIND, "status": "pass", "evidence_mode": "release",
        "rollback_kind": "isolated-single-request-artifact-rehearsal", "source_identity": source,
        "old_release_tag": OLD_TAG, "old_release_commit": OLD_COMMIT, "old_release_tag_object": OLD_TAG_OBJECT,
        "old_bundle_manifest_digest": OLD_BUNDLE, "primary_v02_toolchain_manifest_digest": toolchain["manifest_digest"],
        "capture_manifest_digest": primary.manifest["manifest_digest"], "rehearsal_capture_manifest_digest": rehearsal.manifest["manifest_digest"],
        "live_v02_source_verified": True, "trigger_reason": args.trigger_reason.strip(), "last_safe_request_id": LAST_SAFE_REQUEST,
        "resource_status": resources, "config": CONFIG, "config_digest": sha256_json(CONFIG), "memory_plan_digest": rehearsal.manifest["memory_plan_digest"],
        "original_sources_unchanged": True, "new_requests_after_safe_terminal": 0, "unrelated_processes_interrupted": 0,
        "capabilities_lost": ["stateful_streaming", "contiguous_kv", "schema2_R0", "request_story", "streaming_metrics"],
        "archive": archive_record, "files": portable_stateful_rollback_inventory(output),
    }
    report["report_digest"] = sha256_json(report)
    atomic_write_json(output / "rollback-report.json", report)
    atomic_write_json(output / "checksums.json", portable_stateful_rollback_inventory(output))
    atomic_write_text(output / "COMPLETE", report["report_digest"] + "\n")
    (output / "INCOMPLETE").unlink()
    try:
        return verify_stateful_rollback(output)
    except BaseException:
        (output / "COMPLETE").unlink(missing_ok=True)
        atomic_write_text(output / "INCOMPLETE", "portable_verification_failed\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_rollback(argv)
        print(json.dumps({"status": report["status"], "report_digest": report["report_digest"], "evidence_mode": report["evidence_mode"]}, sort_keys=True))
        return 0
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"code": "INVALID_ARGUMENT", "message": "rollback stopped without sealing success",
                          "exception_type": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())

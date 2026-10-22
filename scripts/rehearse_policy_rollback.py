"""Rehearse the CPU policy rollback chain in fresh, source-bound runtimes.

This is an isolated release-component rehearsal.  It does not stop an
unrelated service, initialize CUDA, grant qualification, tag Git, or promote a
release.  The immutable V0.1 runtime is rebuilt offline from its annotated tag
because the historical release contains no prebuilt illm binary artifact.
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
from rollback_v01 import _copy_tree, extract_archive_safely
from illm.canonical import canonical_json_text, sha256_file, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.reference_qualification import BASE_LOCKS, _assert_live_environment, _object, _source_provenance
from illm.stateful_policy_rollback import (
    CANDIDATE_REJECTED_REQUEST,
    CANDIDATE_REQUEST,
    CAPTURE_PATHS,
    COMPATIBILITY,
    CONFIGS,
    EXPECTED_INPUT_TOKEN_IDS,
    EXPECTED_OUTPUT_TOKEN_IDS,
    EXPECTED_TEXT,
    LEGACY_REJECTED_REQUEST,
    LEGACY_REQUEST,
    OLD_BUNDLE,
    OLD_COMMIT,
    OLD_ENVIRONMENT_PROBE,
    OLD_REQUEST,
    OLD_SOURCE,
    OLD_TAG,
    OLD_TAG_OBJECT,
    PHASES,
    REPORT_KIND,
    SCHEMA2_REQUEST,
    _capture_receipt,
    _fence_facts,
    _last_json,
    _old_runtime_identity,
    _validate_policy_result,
    _zero_resources,
    portable_stateful_policy_rollback_inventory,
    verify_stateful_policy_rollback,
)
from illm.numerical_policy import candidate_policy, legacy_policy


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


def _request(engine: Any, capture: Path, request_id: str) -> Any:
    from illm.engine.streaming_contracts import GenerationRequest
    from illm.sampling import SamplingSpec

    return GenerationRequest(
        model_spec_id=engine.model_spec_id,
        prompt="hello",
        max_new_tokens=8,
        sampling=SamplingSpec(mode="greedy", actual_seed=7),
        request_id=request_id,
        capture=capture,
        telemetry="request",
    )


def _run_one_policy(*, output: Path, candidate: bool) -> dict[str, Any]:
    from illm.engine.streaming import Engine

    policy = candidate_policy("fp32") if candidate else legacy_policy("fp32")
    mode = "contiguous" if candidate else "none"
    request_id = CANDIDATE_REQUEST if candidate else LEGACY_REQUEST
    rejected_id = CANDIDATE_REJECTED_REQUEST if candidate else LEGACY_REJECTED_REQUEST
    engine = Engine(
        model="fixture:tiny-decoder",
        device="cpu",
        kv_mode=mode,
        numerical_policy=policy,
        project_root=PROJECT,
    )
    closed = False
    try:
        result = engine.generate(_request(engine, output, request_id)).to_dict()
        before_close = engine.resource_status
        close_result = engine.close(timeout=2.0)
        closed = True
        after_close = engine.resource_status
        rejection: dict[str, Any]
        try:
            engine.generate(_request(engine, output.parent / (output.name + "-must-not-exist"), rejected_id))
        except IllmError as exc:
            rejection = {
                "request_id": rejected_id,
                "accepted": False,
                "error_code": exc.code.value,
                "reason": exc.message,
            }
        else:  # pragma: no cover - a serious contract violation
            _fail("closed engine accepted a rollback-boundary request")
        expected_query = [6, 1, 1, 1, 1, 1, 1, 1] if candidate else list(range(6, 14))
        expected_valid = list(range(6, 14)) if candidate else [0] * 8
        if (result.get("input_token_ids") != EXPECTED_INPUT_TOKEN_IDS
                or result.get("output_token_ids") != EXPECTED_OUTPUT_TOKEN_IDS
                or result.get("text") != EXPECTED_TEXT
                or result.get("query_tokens_per_step") != expected_query
                or result.get("kv_valid_tokens_per_step") != expected_valid):
            _fail("current transition differs from the frozen full token/query contract")
        for resources in (result.get("resource_status"), before_close, after_close):
            _zero_resources(resources)
        if close_result is not True or rejection != {
            "request_id": rejected_id,
            "accepted": False,
            "error_code": "INVALID_ARGUMENT",
            "reason": "engine is closed",
        }:
            _fail("current transition did not close admission after safe cleanup")
        return {
            "result": result,
            "engine_status_before_close": before_close,
            "engine_close_returned": close_result,
            "engine_status_after_close": after_close,
            "post_close_admission": rejection,
        }
    finally:
        if not closed:
            engine.close(timeout=2.0)


def _run_current_transition(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-current-transition", action="store_true", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-policy", required=True)
    parser.add_argument("--candidate-kv-mode", required=True)
    parser.add_argument("--legacy-policy", required=True)
    parser.add_argument("--legacy-kv-mode", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if (args.candidate_policy, args.candidate_kv_mode, args.legacy_policy, args.legacy_kv_mode,
            args.prompt, args.max_new_tokens, args.seed) != (
            "candidate-fp32", "contiguous", "legacy-fp32", "none", "hello", 8, 7):
        _fail("internal current transition only accepts the frozen CPU rollback contract")
    output = args.output_root.resolve()
    if not is_within(output, PROJECT / "out") or output == (PROJECT / "out").resolve() or not output.is_dir():
        _fail("internal transition output must be the existing rehearsal root below workspace out/")
    captures = output / "captures"
    captures.mkdir(exist_ok=True)
    candidate = _run_one_policy(output=captures / "candidate-c", candidate=True)
    legacy = _run_one_policy(output=captures / "legacy-a", candidate=False)
    print(canonical_json_text({
        "schema_version": 1,
        "report_kind": "illm-policy-rollback-current-transition-v1",
        "candidate_c": candidate,
        "legacy_a": legacy,
    }))
    return 0


def _preflight_before_old(output: Path, transition: Mapping[str, Any], toolchain: Mapping[str, Any]) -> None:
    if (set(transition) != {"schema_version", "report_kind", "candidate_c", "legacy_a"}
            or type(transition.get("schema_version")) is not int or transition.get("schema_version") != 1
            or transition.get("report_kind") != "illm-policy-rollback-current-transition-v1"):
        _fail("current transition receipt is not the closed rollback-v2 contract")
    for name, candidate in (("candidate_c", True), ("legacy_a", False)):
        segment = transition.get(name)
        if (not isinstance(segment, dict)
                or set(segment) != {"result", "engine_status_before_close", "engine_close_returned",
                                    "engine_status_after_close", "post_close_admission"}):
            _fail("current transition omitted a policy segment")
        capture = PolicyStreamingCaptureBundle.open(output / CAPTURE_PATHS[name])
        result = segment.get("result")
        if not isinstance(result, dict):
            _fail("current transition omitted its result")
        _validate_policy_result(result, capture, candidate=candidate)
        for key in ("engine_status_before_close", "engine_status_after_close"):
            _zero_resources(segment.get(key))
        if segment.get("engine_close_returned") is not True:
            _fail("current transition engine did not close safely")
        expected_id = CANDIDATE_REJECTED_REQUEST if candidate else LEGACY_REJECTED_REQUEST
        if segment.get("post_close_admission") != {
            "request_id": expected_id,
            "accepted": False,
            "error_code": "INVALID_ARGUMENT",
            "reason": "engine is closed",
        }:
            _fail("current transition did not close admission")
        _fence_facts(capture, candidate=candidate)
        binding = capture.manifest["execution_binding"]
        if binding["toolchain"] != {
            "binding": "source-bound-manifest",
            "manifest_digest": toolchain["manifest_digest"],
            "environment_id": toolchain["environment_id"],
            "lock_digest": toolchain["lock_digest"],
        }:
            _fail("current transition capture is not bound to the supplied live toolchain")


def run_policy_rollback(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v02-toolchain-manifest", type=Path, required=True)
    parser.add_argument("--trusted-git-repository", type=Path, required=True,
                        help="trusted local repository containing immutable annotated v0.1.0; never fetched")
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--trigger-reason", required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.output.is_symlink() or output.exists() or not is_within(output, PROJECT / "out") or output == (PROJECT / "out").resolve():
        _fail("policy rollback output must be a new directory strictly below workspace out/")
    if not args.trigger_reason.strip() or len(args.trigger_reason) > 1000:
        _fail("policy rollback trigger reason must be nonempty and at most1000 characters")
    if args.v02_toolchain_manifest.is_symlink() or not args.v02_toolchain_manifest.is_file():
        _fail("policy rollback current manifest must be an existing ordinary file")
    for path in (args.trusted_git_repository, args.artifact_cache):
        if path.is_symlink() or not path.is_dir():
            _fail("policy rollback repository/cache inputs must be existing ordinary directories")
    manifest_path = args.v02_toolchain_manifest.resolve()
    if not is_within(manifest_path, PROJECT):
        _fail("policy rollback current toolchain must belong to this workspace")
    toolchain = _object(manifest_path)
    if os.environ.get("ILLM_TOOLCHAIN_MANIFEST") is None or Path(os.environ["ILLM_TOOLCHAIN_MANIFEST"]).resolve() != manifest_path:
        _fail("policy rollback producer must run through its supplied current manifest")
    provenance = _assert_live_environment(toolchain, PROJECT)
    if not str(provenance.get("project_version", "")).startswith("0.2.0"):
        _fail("policy rollback producer requires V0.2 source")
    git = shutil.which("git")
    if git is None:
        _fail("policy rollback requires local Git and never fetches old source")

    output.mkdir(parents=True)
    atomic_write_text(output / "INCOMPLETE", "policy_rollback_in_progress\n")
    logs = output / "logs"
    logs.mkdir()
    local = output / "local-only"
    local.mkdir()
    commands: list[dict[str, Any]] = []
    environment = isolated_environment()
    for key in list(environment):
        if key.startswith(("ILLM_", "GIT_")) or key in {"VIRTUAL_ENV", "PYTHONPATH"}:
            environment.pop(key, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES="")
    base_python = str(Path(sys.executable).resolve())

    def phase(name: str, command: list[str], cwd: Path, expected: int = 0) -> None:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, check=False)
        stdout, stderr = logs / f"{name}.stdout.txt", logs / f"{name}.stderr.txt"
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        commands.append({
            "phase": name,
            "argv": command,
            "returncode": process.returncode,
            "stdout_file": stdout.relative_to(output).as_posix(),
            "stderr_file": stderr.relative_to(output).as_posix(),
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        })
        atomic_write_json(output / "execution.json", {"schema_version": 2, "report_kind": REPORT_KIND, "commands": commands})
        print(json.dumps({"phase": name, "returncode": process.returncode, "expected_returncode": expected}), flush=True)
        if process.returncode != expected:
            _fail(f"policy rollback phase {name} failed; INCOMPLETE and raw transcripts are preserved")

    repository = args.trusted_git_repository.resolve()
    phase("resolve-tag", [git, "rev-parse", f"refs/tags/{OLD_TAG}", f"refs/tags/{OLD_TAG}^{{commit}}"], repository)
    if (logs / "resolve-tag.stdout.txt").read_text(encoding="ascii").splitlines() != [OLD_TAG_OBJECT, OLD_COMMIT]:
        _fail("trusted local tag is absent or no longer names immutable V0.1")
    archive = local / "v0.1.0.zip"
    phase("archive", [git, "archive", "--format=zip", "--output", str(archive), OLD_COMMIT], repository)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.comment.decode("ascii") != OLD_COMMIT:
            _fail("policy rollback Git archive comment does not bind the old commit")
    checkout = local / "v01"
    extract_archive_safely(archive, checkout)
    environment["GIT_CEILING_DIRECTORIES"] = str(checkout.parent)
    old_provenance = _source_provenance(checkout)
    if old_provenance["source_identity"] != OLD_SOURCE or old_provenance["project_version"] != "0.1.0":
        _fail("immutable archive source/locks differ from pinned V0.1")

    _preserve_authority(PROJECT, toolchain, output / "new-source")
    atomic_write_text(output / "current-toolchain.sha256", str(toolchain["manifest_digest"]) + "\n")
    current = [base_python, str(PROJECT / "scripts/in_env.py"), "--manifest", str(manifest_path),
               "--expected-manifest-digest-file", str(output / "current-toolchain.sha256"), "--"]
    phase("current-transition", [*current, "python", str(PROJECT / "scripts/rehearse_policy_rollback.py"),
          "--internal-current-transition", "--output-root", str(output),
          "--candidate-policy", "candidate-fp32", "--candidate-kv-mode", "contiguous",
          "--legacy-policy", "legacy-fp32", "--legacy-kv-mode", "none",
          "--prompt", "hello", "--max-new-tokens", "8", "--seed", "7"], PROJECT)
    transition = _last_json(logs / "current-transition.stdout.txt")
    _preflight_before_old(output, transition, toolchain)

    schema0 = output / CAPTURE_PATHS["toy_schema0"]
    phase("new-create-schema0", [*current, "python", "-m", "illm", "demo", "toy-scheduler", "--seed", "7", "--steps", "4", "--capture", str(schema0)], PROJECT)
    phase("new-read-schema0", [*current, "python", "-m", "illm", "replay", str(schema0), "--mode", "trace"], PROJECT)
    schema2 = output / CAPTURE_PATHS["historical_schema2"]
    phase("new-run-schema2", [*current, "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
          "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
          "--request-id", SCHEMA2_REQUEST, "--kv-mode", "contiguous", "--telemetry", "request", "--capture", str(schema2)], PROJECT)
    phase("new-read-schema2", [*current, "python", "-m", "illm", "replay", str(schema2), "--mode", "trace"], PROJECT)
    candidate_capture = output / CAPTURE_PATHS["candidate_c"]
    legacy_capture = output / CAPTURE_PATHS["legacy_a"]
    phase("new-read-schema3-candidate", [*current, "python", "-m", "illm", "replay", str(candidate_capture), "--mode", "trace"], PROJECT)
    phase("new-read-schema3-legacy", [*current, "python", "-m", "illm", "replay", str(legacy_capture), "--mode", "trace"], PROJECT)

    if (checkout / ".illm/envs/rb01-policy").exists():
        _fail("policy rollback old environment unexpectedly already exists")
    old_manifest, old_digest = checkout / "out/toolchain.json", checkout / "out/toolchain.sha256"
    phase("bootstrap-old", [base_python, str(checkout / "scripts/bootstrap.py"), "--profile", "rb01-policy", "--runtime-profile", "v01-cpu",
          "--locked", "--offline", "--artifact-cache", str(args.artifact_cache.resolve()), "--write-manifest", str(old_manifest),
          "--write-manifest-digest", str(old_digest)], checkout)
    old_toolchain = _object(old_manifest)
    _preserve_authority(checkout, old_toolchain, output / "old-source")
    old = [base_python, str(checkout / "scripts/in_env.py"), "--manifest", str(old_manifest),
           "--expected-manifest-digest-file", str(old_digest), "--"]
    phase("old-environment", [*old, "python", "-I", "-c", OLD_ENVIRONMENT_PROBE], checkout)
    old_capture = checkout / "out/policy-rollback-schema1"
    phase("old-generate-schema1", [*old, "python", "-m", "illm", "generate", "--model", "fixture:tiny-decoder", "--device", "cpu",
          "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
          "--request-id", OLD_REQUEST, "--telemetry", "request", "--capture", str(old_capture)], checkout)
    phase("old-replay-schema1-trace", [*old, "python", "-m", "illm", "replay", str(old_capture), "--mode", "trace"], checkout)
    phase("old-replay-schema1-state", [*old, "python", "-m", "illm", "replay", str(old_capture), "--mode", "state", "--assert-state-digest"], checkout)
    old_release = checkout / "release/v0.1.0"
    phase("old-bundle-verify", [*old, "python", "-m", "illm", "release", "verify", str(old_release / "manifest.json")], checkout)
    phase("old-reject-schema2", [*old, "python", "-m", "illm", "replay", str(schema2), "--mode", "trace"], checkout, 78)
    phase("old-reject-schema3", [*old, "python", "-m", "illm", "replay", str(candidate_capture), "--mode", "trace"], checkout, 78)
    copied_old_capture = output / CAPTURE_PATHS["old_schema1"]
    _copy_tree(old_capture, copied_old_capture)
    _copy_tree(old_release, output / "old-release-bundle")
    phase("new-read-schema1-state", [*current, "python", "-m", "illm", "replay", str(copied_old_capture), "--mode", "state", "--assert-state-digest"], PROJECT)
    if (_source_provenance(PROJECT) != provenance or _source_provenance(checkout) != old_provenance
            or sha256_file(manifest_path) != sha256_file(output / "new-source/toolchain.json")):
        _fail("policy rollback source/toolchain changed while the rehearsal ran")

    candidate_bundle = PolicyStreamingCaptureBundle.open(candidate_capture)
    legacy_bundle = PolicyStreamingCaptureBundle.open(legacy_capture)
    opened: dict[str, Any] = {
        "candidate_c": candidate_bundle,
        "legacy_a": legacy_bundle,
    }
    from illm.capture.dispatch import open_capture
    for name in ("historical_schema2", "toy_schema0", "old_schema1"):
        opened[name] = open_capture(output / CAPTURE_PATHS[name])
    captures = {name: _capture_receipt(name, opened[name]) for name in CAPTURE_PATHS}
    candidate_segment = transition["candidate_c"]
    legacy_segment = transition["legacy_a"]
    safe_boundary = {
        "candidate_c": {
            "result_resource_status": candidate_segment["result"]["resource_status"],
            "engine_status_before_close": candidate_segment["engine_status_before_close"],
            "engine_close_returned": candidate_segment["engine_close_returned"],
            "engine_status_after_close": candidate_segment["engine_status_after_close"],
            "terminal_outcome": candidate_segment["result"]["outcome"],
            "fence_facts": _fence_facts(candidate_bundle, candidate=True),
            "post_close_admission": candidate_segment["post_close_admission"],
        },
        "legacy_a": {
            "result_resource_status": legacy_segment["result"]["resource_status"],
            "engine_status_before_close": legacy_segment["engine_status_before_close"],
            "engine_close_returned": legacy_segment["engine_close_returned"],
            "engine_status_after_close": legacy_segment["engine_status_after_close"],
            "terminal_outcome": legacy_segment["result"]["outcome"],
            "fence_facts": _fence_facts(legacy_bundle, candidate=False),
            "post_close_admission": legacy_segment["post_close_admission"],
        },
        "new_requests_after_last_safe_terminal": 0,
    }
    source = provenance["source_identity"]
    old_probe = _last_json(logs / "old-environment.stdout.txt")
    manifest = {
        "schema_version": 2,
        "report_kind": REPORT_KIND,
        "source_identity": source,
        "protocol_sha256": PROTOCOL_SHA256,
        "current_toolchain_manifest_digest": toolchain["manifest_digest"],
        "current_environment_id": toolchain["environment_id"],
        "from_policy_digest": candidate_policy("fp32").digest,
        "target_policy_digest": legacy_policy("fp32").digest,
        "configs_digest": sha256_json(CONFIGS),
        "primary_capture_manifest_digest": candidate_bundle.manifest["manifest_digest"],
        "old_release_tag": OLD_TAG,
        "old_release_commit": OLD_COMMIT,
        "old_release_tag_object": OLD_TAG_OBJECT,
        "old_bundle_manifest_digest": OLD_BUNDLE,
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_text(output / "commands.txt", "\n".join(subprocess.list2cmdline(item["argv"]) for item in commands) + "\n")
    atomic_write_text(output / "rollback-report.md", "# V0.2 policy rollback rehearsal\n\n"
          "Actual isolated CPU release-component rehearsal; qualification remains unqualified and GPU was not used.\n\n"
          "Candidate C reached a fenced zero-resource boundary, admission closed, a new explicit legacy+none request ran as A, "
          "and only after its own safe close did the independently source-built V0.1 runtime execute.\n\n"
          f"Trigger: {args.trigger_reason.strip()}\n")
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_kind": REPORT_KIND,
        "status": "pass",
        "evidence_mode": "release",
        "rollback_kind": "isolated-cpu-policy-rollback-rehearsal",
        "qualification_status": "unqualified",
        "global_qualification_granted": False,
        "gpu_qualification_granted": False,
        "source_identity": source,
        "protocol_sha256": PROTOCOL_SHA256,
        "current_toolchain_manifest_digest": toolchain["manifest_digest"],
        "current_environment_id": toolchain["environment_id"],
        "execution_binding": candidate_bundle.manifest["execution_binding"],
        "execution_binding_digest": candidate_bundle.manifest["execution_binding_digest"],
        "trigger_reason": args.trigger_reason.strip(),
        "last_safe_request_id": LEGACY_REQUEST,
        "from_policy": candidate_policy("fp32").to_dict(),
        "from_policy_digest": candidate_policy("fp32").digest,
        "target_policy": legacy_policy("fp32").to_dict(),
        "target_policy_digest": legacy_policy("fp32").digest,
        "configs": CONFIGS,
        "configs_digest": sha256_json(CONFIGS),
        "captures": captures,
        "safe_boundary": safe_boundary,
        "compatibility": COMPATIBILITY,
        "old_runtime_identity": _old_runtime_identity(old_probe, old_toolchain),
        "original_sources_unchanged": True,
        "unrelated_processes_interrupted": 0,
        "files": portable_stateful_policy_rollback_inventory(output),
    }
    report["report_digest"] = sha256_json(report)
    atomic_write_json(output / "rollback-report.json", report)
    atomic_write_json(output / "checksums.json", portable_stateful_policy_rollback_inventory(output))
    atomic_write_text(output / "COMPLETE", report["report_digest"] + "\n")
    (output / "INCOMPLETE").unlink()
    try:
        return verify_stateful_policy_rollback(output)
    except BaseException:
        (output / "COMPLETE").unlink(missing_ok=True)
        atomic_write_text(output / "INCOMPLETE", "portable_verification_failed\n")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--internal-current-transition" in args:
            return _run_current_transition(args)
        report = run_policy_rollback(args)
        print(canonical_json_text({
            "report_kind": report["report_kind"],
            "status": report["status"],
            "qualification_status": report["qualification_status"],
            "report_digest": report["report_digest"],
        }))
        return 0
    except IllmError as exc:
        print(canonical_json_text(exc.to_dict()), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

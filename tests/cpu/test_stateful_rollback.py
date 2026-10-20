"""Rollback contracts; fabricated command records are never release evidence.

The full semantic fixture is explicitly synthetic_test_fixture/synthetic_pass.
Only the private test-only reader can inspect it; the public reader rejects it.
Actual old checkout/bootstrap/execution remains an external release rehearsal.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, cast

import pytest

from illm import __version__
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import IllmError
from illm.reference_qualification import BASE_LOCKS, _source_provenance
from illm.reference_release import _state as _old_bundle_state
import illm.stateful_rollback as rollback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
producer = importlib.import_module("rehearse_stateful_rollback")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _seal(root: Path, report: dict[str, Any]) -> None:
    (root / "rollback-report.json").unlink(missing_ok=True)
    report.pop("report_digest", None)
    report["files"] = rollback.portable_stateful_rollback_inventory(root)
    report["report_digest"] = sha256_json(report)
    _write(root / "rollback-report.json", report)
    _write(root / "checksums.json", rollback.portable_stateful_rollback_inventory(root))
    (root / "COMPLETE").write_text(report["report_digest"] + "\n", encoding="ascii")


def _authority(source: Path, toolchain: dict[str, Any], target: Path, provenance: Mapping[str, Any]) -> None:
    target.mkdir(parents=True)
    _write(target / "source-provenance.json", provenance)
    _write(target / "toolchain.json", toolchain)
    locks = target / "locks"
    locks.mkdir()
    for name in (*BASE_LOCKS, "pyproject.toml"):
        shutil.copyfile(source / name, locks / name)
    shutil.copyfile(source / "runtime/artifacts" / f"{toolchain['runtime_profile_id']}.json", locks / "runtime-lock.json")


@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from illm.engine import Engine as OldAPI, GenerationRequest as OldRequest
    from illm.engine.streaming import Engine
    from illm.engine.streaming_contracts import GenerationRequest
    from illm.sampling import SamplingSpec

    root = tmp_path_factory.mktemp("explicitly-synthetic-rollback")
    primary = next((ROOT / "release/v0.1.0/qualification/profiles/tiny-reference-cpu-v1/evidence").iterdir())
    old = json.loads((primary / "toolchain.json").read_text(encoding="utf-8"))
    author_output, author_source = "/unavailable/synthetic-rollback", "/unavailable/synthetic-current-source"
    old_root = author_output + "/local-only/v01"
    old.update(project_root=old_root, environment_id="synthetic-old-not-executed", profile="rb01",
               environment_root=old_root + "/.illm/envs/rb01", python_executable=old_root + "/.illm/envs/rb01/Scripts/python.exe")
    old.pop("manifest_digest")
    old["manifest_digest"] = sha256_json(old)
    new = copy.deepcopy(old)
    provenance = _source_provenance(ROOT)
    new.update(project_root=author_source, environment_id="synthetic-new-not-installed", profile="synthetic-new-not-installed",
               environment_root=author_source + "/.illm/envs/current", python_executable=author_source + "/.illm/envs/current/Scripts/python.exe",
               project_root_digest=provenance["source_identity"]["project_root_digest"],
               bootstrap_digest=provenance["source_identity"]["bootstrap_digest"])
    new["installed_packages"]["illm"] = __version__
    new["lock_digest"] = sha256_json({"base_lock_digest": provenance["source_identity"]["base_lock_digest"], "runtime_lock_digest": new["runtime_lock_digest"]})
    new.pop("manifest_digest")
    new["manifest_digest"] = sha256_json(new)
    _authority(ROOT, new, root / "new-source", provenance)
    shutil.copytree(primary / "locks", root / "old-source/locks")
    shutil.copyfile(primary / "source-provenance.json", root / "old-source/source-provenance.json")
    _write(root / "old-source/toolchain.json", old)
    (root / "new-toolchain.sha256").write_text(new["manifest_digest"] + "\n", encoding="ascii")
    # Real tiny operations exercise capture/result relations, not historical
    # binary provenance. The fabricated commands below stay explicitly synthetic.
    bound = root / "new-source/toolchain.json"
    previous = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    os.environ["ILLM_TOOLCHAIN_MANIFEST"] = str(bound)
    engine = Engine(project_root=ROOT)
    try:
        result = engine.generate(GenerationRequest(engine.model_spec_id, 8, SamplingSpec(actual_seed=7), prompt="hello",
                                request_id=rollback.LAST_SAFE_REQUEST, capture=root / "new-capture", telemetry="request"))
    finally:
        engine.close()
        if previous is None:
            os.environ.pop("ILLM_TOOLCHAIN_MANIFEST", None)
        else:
            os.environ["ILLM_TOOLCHAIN_MANIFEST"] = previous
    shutil.copytree(root / "new-capture", root / "quarantine/capture")
    old_api = OldAPI(project_root=ROOT)
    try:
        old_result = old_api.generate(OldRequest(prompt="hello", max_new_tokens=8, sampling=SamplingSpec(actual_seed=7),
                                     request_id=rollback.OLD_REQUEST, capture=root / "old-capture"))
    finally:
        old_api.close()
    capture = StreamingCaptureBundle.open(root / "new-capture")
    old_capture = ReferenceCaptureBundle.open(root / "old-capture")
    producer._copy_tree(ROOT / "release/v0.1.0", root / "old-release-bundle")
    quarantine = {"quarantine/capture/" + path.name: {"sha256": sha256_file(path), "source_sha256_before": sha256_file(path),
            "source_sha256_after": sha256_file(path), "read_only_at_creation": True} for path in (root / "quarantine/capture").iterdir()}
    _write(root / "quarantine-inventory.json", quarantine)
    base, new_manifest = str(new["python_executable"]), author_source + "/out/toolchain.json"
    new_runner = [base, author_source + "/scripts/in_env.py", "--manifest", new_manifest,
                  "--expected-manifest-digest-file", author_output + "/new-toolchain.sha256", "--"]
    old_runner = [base, old_root + "/scripts/in_env.py", "--manifest", old_root + "/out/toolchain.json",
                  "--expected-manifest-digest-file", old_root + "/out/toolchain.sha256", "--"]
    new_path, old_path, preserved = author_output + "/new-capture", old_root + "/out/reference-capture", author_output + "/quarantine/capture"
    commands: dict[str, list[str]] = {
        "resolve-tag": ["git", "rev-parse", f"refs/tags/{rollback.OLD_TAG}", f"refs/tags/{rollback.OLD_TAG}^{{commit}}"],
        "archive": ["git", "archive", "--format=zip", "--output", author_output + "/local-only/v0.1.0.zip", rollback.OLD_COMMIT],
        "new-run": [*new_runner, "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
                    "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                    "--request-id", rollback.LAST_SAFE_REQUEST, "--kv-mode", "contiguous", "--telemetry", "request", "--capture", new_path],
        "new-replay-primary": [*new_runner, "python", "-m", "illm", "replay", preserved, "--mode", "trace"],
        "new-inspect-primary": [*new_runner, "python", "-m", "illm", "inspect", "request", preserved, "--request", rollback.LAST_SAFE_REQUEST],
        "new-replay": [*new_runner, "python", "-m", "illm", "replay", new_path, "--mode", "trace"],
        "new-inspect": [*new_runner, "python", "-m", "illm", "inspect", "request", new_path, "--request", rollback.LAST_SAFE_REQUEST],
        "bootstrap-old": [base, old_root + "/scripts/bootstrap.py", "--profile", "rb01", "--runtime-profile", "v01-cpu", "--locked", "--offline",
                          "--artifact-cache", "/unavailable/cache", "--write-manifest", old_root + "/out/toolchain.json",
                          "--write-manifest-digest", old_root + "/out/toolchain.sha256"],
        "old-environment": [*old_runner, "python", "-I", "-c", rollback.ENVIRONMENT_PROBE],
        "old-generate": [*old_runner, "python", "-m", "illm", "generate", "--model", "fixture:tiny-decoder", "--device", "cpu",
                         "--prompt", "hello", "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                         "--request-id", rollback.OLD_REQUEST, "--telemetry", "request", "--capture", old_path],
        "old-replay-trace": [*old_runner, "python", "-m", "illm", "replay", old_path, "--mode", "trace"],
        "old-replay-state": [*old_runner, "python", "-m", "illm", "replay", old_path, "--mode", "state", "--assert-state-digest"],
        "old-bundle-verify": [*old_runner, "python", "-m", "illm", "release", "verify", old_root + "/release/v0.1.0/manifest.json"],
        "old-reject-v02": [*old_runner, "python", "-m", "illm", "replay", preserved, "--mode", "trace"],
        "new-read-old": [*new_runner, "python", "-m", "illm", "replay", author_output + "/old-capture", "--mode", "state", "--assert-state-digest"],
    }
    old_state = {"mode": "R1_REFERENCE", "asserted": True, "numerical_reexecution": False,
                 "final_state_digest": old_capture.manifest["final_state_digest"], "status": "FINISHED",
                 "output_tokens": old_capture.final_state["output_tokens"], "rng_draws_consumed": 0}
    outputs: dict[str, Any] = {"new-run": result.to_dict(), "bootstrap-old": old,
        "old-environment": {"illm_version": "0.1.0", "stateful_engine_present": False,
                            "environment_id": old["environment_id"], "toolchain_digest": old["manifest_digest"]},
        "old-generate": old_result.to_dict(), "old-replay-trace": {"mode": "R0", "displayed_events": len(old_capture.events), "total_events": len(old_capture.events)},
        "old-replay-state": old_state, "new-read-old": old_state,
        "old-bundle-verify": _old_bundle_state(ROOT / "release/v0.1.0/manifest.json"),
        "old-reject-v02": {"code": "UNSUPPORTED_SCHEMA", "message": "synthetic rejection receipt, not actual old execution"}}
    for name in ("new-replay-primary", "new-inspect-primary", "new-replay", "new-inspect"):
        outputs[name] = capture.request_story(rollback.LAST_SAFE_REQUEST)
    records = []
    for name in rollback.PHASES:
        stdout = rollback.OLD_TAG_OBJECT + "\n" + rollback.OLD_COMMIT + "\n" if name == "resolve-tag" else "" if name in {"archive", "old-reject-v02"} else json.dumps(outputs[name]) + "\n"
        stderr = json.dumps(outputs[name]) + "\n" if name == "old-reject-v02" else ""
        for stream, value in (("stdout", stdout), ("stderr", stderr)):
            path = root / f"logs/{name}.{stream}.txt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(value, encoding="utf-8")
        records.append({"phase": name, "argv": commands[name], "returncode": 78 if name == "old-reject-v02" else 0,
                        **{key: value for stream in ("stdout", "stderr") for key, value in (
                            (stream + "_file", f"logs/{name}.{stream}.txt"), (stream + "_sha256", sha256_file(root / f"logs/{name}.{stream}.txt")))}})
    _write(root / "execution.json", {"schema_version": 1, "commands": records})
    manifest = {"schema_version": 1, "report_kind": rollback.REPORT_KIND, "old_release_tag": rollback.OLD_TAG,
                "old_release_commit": rollback.OLD_COMMIT, "source_identity": provenance["source_identity"],
                "primary_v02_toolchain_manifest_digest": new["manifest_digest"], "capture_manifest_digest": capture.manifest["manifest_digest"],
                "config_digest": sha256_json(rollback.CONFIG)}
    _write(root / "manifest.json", manifest)
    for name in ("commands.txt", "rollback-report.md"):
        (root / name).write_text("SYNTHETIC TEST FIXTURE - NOT RELEASE EVIDENCE\n", encoding="ascii")
    report = {**manifest, "status": "synthetic_pass", "evidence_mode": "synthetic_test_fixture",
        "rollback_kind": "isolated-single-request-artifact-rehearsal", "old_release_tag_object": rollback.OLD_TAG_OBJECT,
        "old_bundle_manifest_digest": rollback.OLD_BUNDLE, "rehearsal_capture_manifest_digest": capture.manifest["manifest_digest"],
        "live_v02_source_verified": True, "trigger_reason": "unit fixture only - no old binary executed",
        "last_safe_request_id": rollback.LAST_SAFE_REQUEST, "resource_status": result.resource_status,
        "config": rollback.CONFIG, "memory_plan_digest": capture.manifest["memory_plan_digest"],
        "original_sources_unchanged": True, "new_requests_after_safe_terminal": 0, "unrelated_processes_interrupted": 0,
        "capabilities_lost": ["stateful_streaming", "contiguous_kv", "schema2_R0", "request_story", "streaming_metrics"],
        "archive": {"sha256": "a" * 64, "bytes": 123, "git_archive_comment": rollback.OLD_COMMIT,
                    "embedded": False, "local_relative_path": "local-only/v0.1.0.zip", "source_identity": rollback.OLD_SOURCE}}
    _seal(root, report)
    return root


def test_full_semantic_fixture_is_explicitly_synthetic_and_public_reader_rejects(synthetic_fixture: Path) -> None:
    report = rollback._verify(synthetic_fixture, synthetic_test_fixture=True)
    assert report["status"] == "synthetic_pass"
    with pytest.raises(IllmError, match="not a release rehearsal"):
        rollback.verify_stateful_rollback(synthetic_fixture)


def test_semantic_reader_rejects_resigned_arbitrary_command(synthetic_fixture: Path, tmp_path: Path) -> None:
    root = tmp_path / "bad-command"
    producer._copy_tree(synthetic_fixture, root)
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    execution["commands"][7]["argv"] = [execution["commands"][7]["argv"][0], "-c", "print('invented successful bootstrap')"]
    _write(root / "execution.json", execution)
    report = json.loads((root / "rollback-report.json").read_text(encoding="utf-8"))
    _seal(root, report)
    with pytest.raises(IllmError, match="bootstrap command"):
        rollback._verify(root, synthetic_test_fixture=True)


def test_semantic_reader_rejects_resigned_unreclaimed_resources(synthetic_fixture: Path, tmp_path: Path) -> None:
    root = tmp_path / "unsafe-resources"
    producer._copy_tree(synthetic_fixture, root)
    output = json.loads((root / "logs/new-run.stdout.txt").read_text(encoding="utf-8"))
    output["resource_status"]["kv_reserved_bytes"] = 8192
    _write(root / "logs/new-run.stdout.txt", output)
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    execution["commands"][2]["stdout_sha256"] = sha256_file(root / "logs/new-run.stdout.txt")
    _write(root / "execution.json", execution)
    report = json.loads((root / "rollback-report.json").read_text(encoding="utf-8"))
    report["resource_status"] = output["resource_status"]
    _seal(root, report)
    with pytest.raises(IllmError, match="zero runtime resources"):
        rollback._verify(root, synthetic_test_fixture=True)


def test_semantic_reader_rejects_resigned_wrong_old_commit(synthetic_fixture: Path, tmp_path: Path) -> None:
    root = tmp_path / "wrong-old-source"
    producer._copy_tree(synthetic_fixture, root)
    report = json.loads((root / "rollback-report.json").read_text(encoding="utf-8"))
    report["old_release_commit"] = "0" * 40
    _seal(root, report)
    with pytest.raises(IllmError, match="immutable V0.1"):
        rollback._verify(root, synthetic_test_fixture=True)


def test_relabeling_synthetic_report_does_not_create_a_release_runtime(synthetic_fixture: Path, tmp_path: Path) -> None:
    root = tmp_path / "relabeled-synthetic"
    for name in ("new-source", "old-source"):
        producer._copy_tree(synthetic_fixture / name, root / name)
    report = json.loads((synthetic_fixture / "rollback-report.json").read_text(encoding="utf-8"))
    report.update(evidence_mode="release", status="pass")
    _seal(root, report)
    with pytest.raises(IllmError, match="UUID4"):
        rollback.verify_stateful_rollback(root)


def test_reader_import_and_full_read_have_no_tensor_runtime_or_scripts_dependency(synthetic_fixture: Path, tmp_path: Path) -> None:
    package = tmp_path / "portable/illm"
    shutil.copytree(ROOT / "src/illm", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    script = ("import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import illm.stateful_rollback as r;"
              "report=r._verify(Path(sys.argv[2]),synthetic_test_fixture=True);"
              "assert not ({'torch','transformers','tokenizers','safetensors'} & set(sys.modules));print(report['evidence_mode'])")
    process = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(package.parent), str(synthetic_fixture)],
                             cwd=tmp_path, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "synthetic_test_fixture"


@pytest.mark.parametrize("mutation", ["echo-interpreter", "shared-old-env", "wrong-order", "wrapper-bypass"])
def test_command_contract_rejects_fake_interpreter_reuse_and_execution_order(synthetic_fixture: Path, mutation: str) -> None:
    records = json.loads((synthetic_fixture / "execution.json").read_text(encoding="utf-8"))["commands"]
    new = json.loads((synthetic_fixture / "new-source/toolchain.json").read_text(encoding="utf-8"))
    old = json.loads((synthetic_fixture / "old-source/toolchain.json").read_text(encoding="utf-8"))
    if mutation == "echo-interpreter":
        records[2]["argv"][0] = "/bin/echo"
    elif mutation == "shared-old-env":
        old["environment_root"] = new["environment_root"]
    elif mutation == "wrong-order":
        records[2], records[7] = records[7], records[2]
    else:
        records[2]["argv"][6] = "--different-wrapper-boundary"
    primary = StreamingCaptureBundle.open(synthetic_fixture / "quarantine/capture")
    with pytest.raises(IllmError):
        rollback._verify_commands(records, new, old, primary)


def test_inventory_omits_only_local_work_and_rejects_extra_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "local-only").mkdir()
    (tmp_path / "local-only/private.bin").write_bytes(b"not portable")
    (tmp_path / "fact.json").write_bytes(b"{}")
    assert rollback.portable_stateful_rollback_inventory(tmp_path) == {"fact.json": sha256_file(tmp_path / "fact.json")}
    (tmp_path / "unused").mkdir()
    with pytest.raises(IllmError, match="empty directory"):
        rollback.portable_stateful_rollback_inventory(tmp_path)


@pytest.mark.parametrize("value", ["../escape", "a/../b", "C:/absolute", "a//b", "a\\b", "./same"])
def test_portable_links_reject_noncanonical_or_escaped_values(value: str) -> None:
    with pytest.raises(IllmError):
        rollback._relative(value)


def test_author_paths_are_lexical_and_cannot_be_relative_or_traversing() -> None:
    assert rollback._author_path("C:\\absent\\author\\source") == "C:/absent/author/source"
    assert rollback._author_path("/unavailable/author/source") == "/unavailable/author/source"
    for value in ("relative/source", "C:/source/../escape", "/source/../escape"):
        with pytest.raises(IllmError):
            rollback._author_path(value)


def test_producer_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve", encoding="ascii")
    with pytest.raises(IllmError, match="new directory"):
        producer.run_rollback(["--output", str(tmp_path), "--v02-primary-capture", str(tmp_path),
            "--v02-toolchain-manifest", str(tmp_path / "missing.json"), "--trusted-git-repository", str(ROOT),
            "--artifact-cache", str(ROOT / ".illm/cache/runtime"), "--trigger-reason", "test"])
    assert sentinel.read_text(encoding="ascii") == "preserve"


def test_reader_never_treats_missing_execution_as_a_pass(tmp_path: Path) -> None:
    with pytest.raises(IllmError, match="incomplete"):
        rollback.verify_stateful_rollback(tmp_path)


@pytest.mark.parametrize("poison_pythonpath", [False, True])
def test_environment_probe_really_reads_the_wrapper_manifest_without_an_invented_digest_variable(
    tmp_path: Path, poison_pythonpath: bool,
) -> None:
    digest = sha256_json({"actual_subprocess_contract": "manifest-file-not-environment-digest"})
    manifest = tmp_path / "toolchain.json"
    _write(manifest, {"manifest_digest": digest})
    # Deliberately no ILLM_TOOLCHAIN_DIGEST: neither immutable V0.1 nor the
    # current in_env wrapper creates such a variable.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("ILLM_", "PYTHON"))}
    environment.update(ILLM_ENVIRONMENT_ID="probe-contract-environment", ILLM_LOCK_DIGEST="b" * 64,
                       ILLM_TOOLCHAIN_MANIFEST=str(manifest))
    if poison_pythonpath:
        shadow = tmp_path / "untrusted-current-source/illm"
        shadow.mkdir(parents=True)
        (shadow / "__init__.py").write_text("raise RuntimeError('isolated probe imported untrusted source')\n", encoding="ascii")
        environment["PYTHONPATH"] = str(shadow.parent)
    assert "ILLM_TOOLCHAIN_DIGEST" not in environment
    process = subprocess.run([sys.executable, "-I", "-c", rollback.ENVIRONMENT_PROBE], env=environment,
                             cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30)
    assert process.returncode == 0, process.stderr
    # The caller may deliberately use PYTHONPATH=src with a newer egg-info than
    # the installed development wheel. Compare with an independent -I metadata
    # query, not the caller's source-influenced distribution lookup.
    installed = subprocess.run([sys.executable, "-I", "-c", "import importlib.metadata as m;print(m.version('illm'))"],
                               env=environment, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30)
    assert installed.returncode == 0, installed.stderr
    result = json.loads(process.stdout)
    assert result == {"illm_version": installed.stdout.strip(),
                      "stateful_engine_present": result["stateful_engine_present"],
                      "environment_id": "probe-contract-environment", "toolchain_digest": digest}
    assert type(result["stateful_engine_present"]) is bool


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="actual FIFO is POSIX-only")
def test_public_reader_preflights_fifo_before_opening_the_json_entry_point(tmp_path: Path) -> None:
    (tmp_path / "COMPLETE").write_text("ordinary marker\n", encoding="ascii")
    make_fifo = cast(Callable[[Path], None], getattr(os, "mkfifo"))
    make_fifo(tmp_path / "rollback-report.json")
    # An isolated subprocess timeout is a backstop: the actual FIFO must be
    # rejected before any JSON open, not merely represented by a mock path.
    code = ("import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "from illm.stateful_rollback import verify_stateful_rollback;from illm.domain.errors import IllmError;"
            "\ntry: verify_stateful_rollback(Path(sys.argv[2]))"
            "\nexcept IllmError as exc: print(exc.message);sys.exit(0)"
            "\nraise SystemExit('FIFO unexpectedly accepted')")
    process = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(ROOT / "src"), str(tmp_path)],
                             cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30)
    assert process.returncode == 0, process.stderr
    assert "nonregular artifact" in process.stdout


def test_archive_git_ceiling_blocks_parent_revision_without_hiding_current_checkout(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("local Git executable is unavailable")
    repository = tmp_path / "current-repository"
    repository.mkdir()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    initialized = subprocess.run([git, "-c", "init.templateDir=", "init", "--quiet", str(repository)],
                                 env=environment, capture_output=True, text=True, check=False, timeout=30)
    assert initialized.returncode == 0, initialized.stderr
    checkout = repository / "out/local-only/v01"
    checkout.mkdir(parents=True)
    command = [git, "rev-parse", "--show-toplevel"]
    unbounded = subprocess.run(command, cwd=checkout, env=environment, capture_output=True, text=True, check=False, timeout=30)
    assert unbounded.returncode == 0
    environment["GIT_CEILING_DIRECTORIES"] = str(checkout.parent)
    old = subprocess.run(command, cwd=checkout, env=environment, capture_output=True, text=True, check=False, timeout=30)
    current = subprocess.run(command, cwd=repository, env=environment, capture_output=True, text=True, check=False, timeout=30)
    assert old.returncode == 128
    assert current.returncode == 0, current.stderr
    assert Path(current.stdout.strip()).resolve() == repository.resolve()

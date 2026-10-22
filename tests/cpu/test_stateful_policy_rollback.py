"""Portable policy rollback reader tests; fabricated phases stay synthetic."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, cast

import pytest

from illm import __version__
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.bundle import CaptureBundle
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.capture.reference import ReferenceCaptureBundle
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import IllmError
from illm.engine.contracts import GenerationRequest as OldRequest
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.reference_qualification import BASE_LOCKS, _source_provenance
from illm.reference_release import _state as old_bundle_state
from illm.sampling import SamplingSpec
import illm.stateful_policy_rollback as rollback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
producer = importlib.import_module("rehearse_policy_rollback")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _seal(root: Path, report: dict[str, Any]) -> None:
    for name in ("rollback-report.json", "checksums.json", "COMPLETE"):
        (root / name).unlink(missing_ok=True)
    report.pop("report_digest", None)
    report["files"] = rollback.portable_stateful_policy_rollback_inventory(root)
    report["report_digest"] = sha256_json(report)
    _write(root / "rollback-report.json", report)
    _write(root / "checksums.json", rollback.portable_stateful_policy_rollback_inventory(root))
    (root / "COMPLETE").write_text(report["report_digest"] + "\n", encoding="ascii")


def _authority(source: Path, toolchain: Mapping[str, Any], target: Path,
               provenance: Mapping[str, Any]) -> None:
    target.mkdir(parents=True)
    _write(target / "source-provenance.json", provenance)
    _write(target / "toolchain.json", toolchain)
    locks = target / "locks"
    locks.mkdir()
    for name in (*BASE_LOCKS, "pyproject.toml"):
        shutil.copyfile(source / name, locks / name)
    shutil.copyfile(source / "runtime/artifacts" / f"{toolchain['runtime_profile_id']}.json",
                    locks / "runtime-lock.json")


def _current_toolchain(primary: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    toolchain = cast(dict[str, Any], json.loads(
        (primary / "toolchain.json").read_text(encoding="utf-8")
    ))
    source = provenance["source_identity"]
    environment_root = Path(sys.executable).resolve().parent.parent
    toolchain.update({
        "project_root": str(ROOT.resolve()),
        "project_root_digest": source["project_root_digest"],
        "bootstrap_digest": source["bootstrap_digest"],
        "lock_digest": sha256_json({
            "base_lock_digest": source["base_lock_digest"],
            "runtime_lock_digest": toolchain["runtime_lock_digest"],
        }),
        "environment_id": "synthetic-current-runtime-not-formal",
        "profile": "synthetic-policy-rollback",
        "environment_root": str(environment_root),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "arch": platform.machine(),
        "entrypoints": {"illm": str(environment_root / "Scripts/illm.exe")},
    })
    toolchain["installed_packages"]["illm"] = __version__
    toolchain.pop("manifest_digest", None)
    toolchain["manifest_digest"] = sha256_json(toolchain)
    return toolchain


def _old_toolchain(primary: Path, output: Path) -> dict[str, Any]:
    toolchain = cast(dict[str, Any], json.loads(
        (primary / "toolchain.json").read_text(encoding="utf-8")
    ))
    checkout = output / "local-only/v01"
    environment = checkout / ".illm/envs/rb01-policy"
    toolchain.update({
        "project_root": str(checkout),
        "environment_id": "synthetic-old-runtime-not-executed",
        "profile": "rb01-policy",
        "environment_root": str(environment),
        "python_executable": str(environment / "Scripts/python.exe"),
        "entrypoints": {"illm": str(environment / "Scripts/illm.exe")},
    })
    toolchain.pop("manifest_digest", None)
    toolchain["manifest_digest"] = sha256_json(toolchain)
    return toolchain


def _safe_segment(segment: Mapping[str, Any], bundle: PolicyStreamingCaptureBundle,
                  *, candidate: bool) -> dict[str, Any]:
    result = segment["result"]
    return {
        "result_resource_status": result["resource_status"],
        "engine_status_before_close": segment["engine_status_before_close"],
        "engine_close_returned": segment["engine_close_returned"],
        "engine_status_after_close": segment["engine_status_after_close"],
        "terminal_outcome": result["outcome"],
        "fence_facts": rollback._fence_facts(bundle, candidate=candidate),
        "post_close_admission": segment["post_close_admission"],
    }


@pytest.fixture(scope="module")
def synthetic_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from illm.engine import Engine as OldEngine
    from illm.engine.streaming import Engine
    from illm.engine.streaming_contracts import GenerationRequest
    from illm.toy import run_toy_scheduler

    root = tmp_path_factory.mktemp("explicitly-synthetic-policy-rollback")
    primary = next((ROOT / "release/v0.1.0/qualification/profiles/tiny-reference-cpu-v1/evidence").iterdir())
    provenance = _source_provenance(ROOT)
    current = _current_toolchain(primary, provenance)
    old = _old_toolchain(primary, root)
    _authority(ROOT, current, root / "new-source", provenance)
    shutil.copytree(primary / "locks", root / "old-source/locks")
    shutil.copyfile(primary / "source-provenance.json", root / "old-source/source-provenance.json")
    _write(root / "old-source/toolchain.json", old)
    (root / "current-toolchain.sha256").write_text(current["manifest_digest"] + "\n", encoding="ascii")

    saved = {name: os.environ.get(name) for name in (
        "ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST",
    )}
    os.environ.update({
        "ILLM_TOOLCHAIN_MANIFEST": str(root / "new-source/toolchain.json"),
        "ILLM_ENVIRONMENT_ID": current["environment_id"],
        "ILLM_LOCK_DIGEST": current["lock_digest"],
    })
    captures = root / "captures"
    captures.mkdir()
    try:
        candidate = producer._run_one_policy(output=captures / "candidate-c", candidate=True)
        legacy = producer._run_one_policy(output=captures / "legacy-a", candidate=False)
        schema2_engine = Engine(model="fixture:tiny-decoder", device="cpu", kv_mode="contiguous",
                                project_root=ROOT)
        try:
            schema2_result = schema2_engine.generate(GenerationRequest(
                model_spec_id=schema2_engine.model_spec_id,
                prompt="hello",
                max_new_tokens=8,
                sampling=SamplingSpec(mode="greedy", actual_seed=7),
                request_id=rollback.SCHEMA2_REQUEST,
                capture=captures / "historical-schema2",
                telemetry="request",
            )).to_dict()
        finally:
            schema2_engine.close()
        run_toy_scheduler(seed=7, steps=4, capture_root=captures / "toy-schema0", project_root=ROOT)
        old_engine = OldEngine(project_root=ROOT)
        try:
            old_result = old_engine.generate(OldRequest(
                prompt="hello",
                max_new_tokens=8,
                sampling=SamplingSpec(mode="greedy", actual_seed=7),
                request_id=rollback.OLD_REQUEST,
                capture=captures / "old-schema1",
            )).to_dict()
        finally:
            old_engine.close()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    candidate_capture = PolicyStreamingCaptureBundle.open(captures / "candidate-c")
    legacy_capture = PolicyStreamingCaptureBundle.open(captures / "legacy-a")
    schema2_capture = StreamingCaptureBundle.open(captures / "historical-schema2")
    toy_capture = CaptureBundle.open(captures / "toy-schema0")
    old_capture = ReferenceCaptureBundle.open(captures / "old-schema1")
    producer._copy_tree(ROOT / "release/v0.1.0", root / "old-release-bundle")

    transition = {
        "schema_version": 1,
        "report_kind": "illm-policy-rollback-current-transition-v1",
        "candidate_c": candidate,
        "legacy_a": legacy,
    }
    old_state = {
        "mode": "R1_REFERENCE",
        "asserted": True,
        "numerical_reexecution": False,
        "final_state_digest": old_capture.manifest["final_state_digest"],
        "status": "FINISHED",
        "output_tokens": old_capture.final_state["output_tokens"],
        "rng_draws_consumed": 0,
    }
    old_probe = {
        "illm_version": "0.1.0",
        "stateful_engine_present": False,
        "numerical_policy_module_present": False,
        "reported_numerical_policy": None,
        "environment_id": old["environment_id"],
        "toolchain_digest": old["manifest_digest"],
        "runtime_profile_id": old["runtime_profile_id"],
        "python_version": old["python_version"],
    }
    outputs: dict[str, dict[str, Any]] = {
        "current-transition": transition,
        "new-create-schema0": {"status": "created", "schema_version": 0},
        "new-read-schema0": {"mode": "R0", "displayed_events": len(toy_capture.events),
                             "total_events": len(toy_capture.events)},
        "new-run-schema2": schema2_result,
        "new-read-schema2": schema2_capture.request_story(rollback.SCHEMA2_REQUEST),
        "new-read-schema3-candidate": candidate_capture.request_story(rollback.CANDIDATE_REQUEST),
        "new-read-schema3-legacy": legacy_capture.request_story(rollback.LEGACY_REQUEST),
        "bootstrap-old": old,
        "old-environment": old_probe,
        "old-generate-schema1": old_result,
        "old-replay-schema1-trace": {"mode": "R0", "displayed_events": len(old_capture.events),
                                      "total_events": len(old_capture.events)},
        "old-replay-schema1-state": old_state,
        "old-bundle-verify": old_bundle_state(ROOT / "release/v0.1.0/manifest.json"),
        "old-reject-schema2": {"code": "UNSUPPORTED_SCHEMA", "message": "synthetic receipt"},
        "old-reject-schema3": {"code": "UNSUPPORTED_SCHEMA", "message": "synthetic receipt"},
        "new-read-schema1-state": old_state,
    }

    output_path = str(root.resolve())
    old_environment = str((root / "local-only/v01/.illm/envs/rb01-policy").resolve())
    old_manifest = str((root / "local-only/v01/out/toolchain.json").resolve())
    old_digest = str((root / "local-only/v01/out/toolchain.sha256").resolve())
    current_digest = str((root / "current-toolchain.sha256").resolve())
    base = str(Path(sys.executable).resolve())
    current_runner = [base, str((ROOT / "scripts/in_env.py").resolve()), "--manifest",
                      str((root / "new-source/toolchain.json").resolve()),
                      "--expected-manifest-digest-file", current_digest, "--"]
    old_runner = [base, str((root / "local-only/v01/scripts/in_env.py").resolve()), "--manifest",
                  old_manifest, "--expected-manifest-digest-file", old_digest, "--"]
    def capture_path(name: str) -> str:
        return str((root / rollback.CAPTURE_PATHS[name]).resolve())
    commands: dict[str, list[str]] = {
        "resolve-tag": ["git", "rev-parse", f"refs/tags/{rollback.OLD_TAG}",
                        f"refs/tags/{rollback.OLD_TAG}^{{commit}}"],
        "archive": ["git", "archive", "--format=zip", "--output",
                    str((root / "local-only/v0.1.0.zip").resolve()), rollback.OLD_COMMIT],
        "current-transition": [*current_runner, "python", str((ROOT / "scripts/rehearse_policy_rollback.py").resolve()),
                               "--internal-current-transition", "--output-root", output_path,
                               "--candidate-policy", "candidate-fp32", "--candidate-kv-mode", "contiguous",
                               "--legacy-policy", "legacy-fp32", "--legacy-kv-mode", "none",
                               "--prompt", "hello", "--max-new-tokens", "8", "--seed", "7"],
        "new-create-schema0": [*current_runner, "python", "-m", "illm", "demo", "toy-scheduler",
                               "--seed", "7", "--steps", "4", "--capture", capture_path("toy_schema0")],
        "new-read-schema0": [*current_runner, "python", "-m", "illm", "replay",
                             capture_path("toy_schema0"), "--mode", "trace"],
        "new-run-schema2": [*current_runner, "python", "-m", "illm", "run", "--model", "fixture:tiny-decoder",
                            "--device", "cpu", "--prompt", "hello", "--max-new-tokens", "8",
                            "--sampling-mode", "greedy", "--seed", "7", "--request-id", rollback.SCHEMA2_REQUEST,
                            "--kv-mode", "contiguous", "--telemetry", "request", "--capture",
                            capture_path("historical_schema2")],
        "new-read-schema2": [*current_runner, "python", "-m", "illm", "replay",
                             capture_path("historical_schema2"), "--mode", "trace"],
        "new-read-schema3-candidate": [*current_runner, "python", "-m", "illm", "replay",
                                       capture_path("candidate_c"), "--mode", "trace"],
        "new-read-schema3-legacy": [*current_runner, "python", "-m", "illm", "replay",
                                    capture_path("legacy_a"), "--mode", "trace"],
        "bootstrap-old": [base, str((root / "local-only/v01/scripts/bootstrap.py").resolve()), "--profile",
                          "rb01-policy", "--runtime-profile", "v01-cpu", "--locked", "--offline",
                          "--artifact-cache", str((ROOT / ".illm/cache/runtime").resolve()),
                          "--write-manifest", old_manifest, "--write-manifest-digest", old_digest],
        "old-environment": [*old_runner, "python", "-I", "-c", rollback.OLD_ENVIRONMENT_PROBE],
        "old-generate-schema1": [*old_runner, "python", "-m", "illm", "generate", "--model",
                                 "fixture:tiny-decoder", "--device", "cpu", "--prompt", "hello",
                                 "--max-new-tokens", "8", "--sampling-mode", "greedy", "--seed", "7",
                                 "--request-id", rollback.OLD_REQUEST, "--telemetry", "request", "--capture",
                                 str((root / "local-only/v01/out/policy-rollback-schema1").resolve())],
        "old-replay-schema1-trace": [*old_runner, "python", "-m", "illm", "replay",
                                     str((root / "local-only/v01/out/policy-rollback-schema1").resolve()),
                                     "--mode", "trace"],
        "old-replay-schema1-state": [*old_runner, "python", "-m", "illm", "replay",
                                     str((root / "local-only/v01/out/policy-rollback-schema1").resolve()),
                                     "--mode", "state", "--assert-state-digest"],
        "old-bundle-verify": [*old_runner, "python", "-m", "illm", "release", "verify",
                              str((root / "local-only/v01/release/v0.1.0/manifest.json").resolve())],
        "old-reject-schema2": [*old_runner, "python", "-m", "illm", "replay",
                               capture_path("historical_schema2"), "--mode", "trace"],
        "old-reject-schema3": [*old_runner, "python", "-m", "illm", "replay",
                               capture_path("candidate_c"), "--mode", "trace"],
        "new-read-schema1-state": [*current_runner, "python", "-m", "illm", "replay",
                                   capture_path("old_schema1"), "--mode", "state", "--assert-state-digest"],
    }
    assert old_environment.replace("\\", "/") in old["python_executable"].replace("\\", "/")
    logs = root / "logs"
    logs.mkdir()
    records: list[dict[str, Any]] = []
    for phase in rollback.PHASES:
        if phase == "resolve-tag":
            stdout = rollback.OLD_TAG_OBJECT + "\n" + rollback.OLD_COMMIT + "\n"
            stderr = ""
        elif phase == "archive":
            stdout = stderr = ""
        elif phase in {"old-reject-schema2", "old-reject-schema3"}:
            stdout = ""
            stderr = json.dumps(outputs[phase], sort_keys=True) + "\n"
        else:
            stdout = json.dumps(outputs[phase], sort_keys=True) + "\n"
            stderr = ""
        for stream, text in (("stdout", stdout), ("stderr", stderr)):
            (logs / f"{phase}.{stream}.txt").write_text(text, encoding="utf-8")
        records.append({
            "phase": phase,
            "argv": commands[phase],
            "returncode": 78 if phase in {"old-reject-schema2", "old-reject-schema3"} else 0,
            "stdout_file": f"logs/{phase}.stdout.txt",
            "stderr_file": f"logs/{phase}.stderr.txt",
            "stdout_sha256": sha256_file(logs / f"{phase}.stdout.txt"),
            "stderr_sha256": sha256_file(logs / f"{phase}.stderr.txt"),
        })
    _write(root / "execution.json", {"schema_version": 2, "report_kind": rollback.REPORT_KIND,
                                      "commands": records})

    receipt_bundles: dict[str, Any] = {
        "candidate_c": candidate_capture,
        "legacy_a": legacy_capture,
        "historical_schema2": schema2_capture,
        "toy_schema0": toy_capture,
        "old_schema1": old_capture,
    }
    capture_receipts = {name: rollback._capture_receipt(name, bundle)
                        for name, bundle in receipt_bundles.items()}
    safe_boundary = {
        "candidate_c": _safe_segment(candidate, candidate_capture, candidate=True),
        "legacy_a": _safe_segment(legacy, legacy_capture, candidate=False),
        "new_requests_after_last_safe_terminal": 0,
    }
    binding = candidate_capture.manifest["execution_binding"]
    manifest = {
        "schema_version": 2,
        "report_kind": rollback.REPORT_KIND,
        "source_identity": provenance["source_identity"],
        "protocol_sha256": PROTOCOL_SHA256,
        "current_toolchain_manifest_digest": current["manifest_digest"],
        "current_environment_id": current["environment_id"],
        "from_policy_digest": candidate_policy("fp32").digest,
        "target_policy_digest": legacy_policy("fp32").digest,
        "configs_digest": sha256_json(rollback.CONFIGS),
        "primary_capture_manifest_digest": candidate_capture.manifest["manifest_digest"],
        "old_release_tag": rollback.OLD_TAG,
        "old_release_commit": rollback.OLD_COMMIT,
        "old_release_tag_object": rollback.OLD_TAG_OBJECT,
        "old_bundle_manifest_digest": rollback.OLD_BUNDLE,
    }
    _write(root / "manifest.json", manifest)
    (root / "commands.txt").write_text("SYNTHETIC COMMAND TRANSCRIPTS - NOT EXECUTED\n", encoding="ascii")
    (root / "rollback-report.md").write_text(
        "# Synthetic policy rollback fixture\n\nNot release or qualification evidence.\n", encoding="ascii")
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_kind": rollback.REPORT_KIND,
        "status": "synthetic_pass",
        "evidence_mode": "synthetic_test_fixture",
        "rollback_kind": "isolated-cpu-policy-rollback-rehearsal",
        "qualification_status": "unqualified",
        "global_qualification_granted": False,
        "gpu_qualification_granted": False,
        "source_identity": provenance["source_identity"],
        "protocol_sha256": PROTOCOL_SHA256,
        "current_toolchain_manifest_digest": current["manifest_digest"],
        "current_environment_id": current["environment_id"],
        "execution_binding": binding,
        "execution_binding_digest": sha256_json(binding),
        "trigger_reason": "unit fixture only; historical runtime was not executed",
        "last_safe_request_id": rollback.LEGACY_REQUEST,
        "from_policy": candidate_policy("fp32").to_dict(),
        "from_policy_digest": candidate_policy("fp32").digest,
        "target_policy": legacy_policy("fp32").to_dict(),
        "target_policy_digest": legacy_policy("fp32").digest,
        "configs": rollback.CONFIGS,
        "configs_digest": sha256_json(rollback.CONFIGS),
        "captures": capture_receipts,
        "safe_boundary": safe_boundary,
        "compatibility": rollback.COMPATIBILITY,
        "old_runtime_identity": rollback._old_runtime_identity(old_probe, old),
        "original_sources_unchanged": True,
        "unrelated_processes_interrupted": 0,
    }
    _seal(root, report)
    return root


def test_synthetic_fixture_proves_semantics_but_public_reader_rejects_release_claim(
    synthetic_fixture: Path,
) -> None:
    report = rollback._verify(synthetic_fixture, synthetic_test_fixture=True)
    assert report["status"] == "synthetic_pass"
    assert report["qualification_status"] == "unqualified"
    assert report["global_qualification_granted"] is False
    assert report["gpu_qualification_granted"] is False
    with pytest.raises(IllmError, match="wrong version/mode"):
        rollback.verify_stateful_policy_rollback(synthetic_fixture)


@pytest.mark.parametrize("field,value", [
    ("qualification_status", "qualified"),
    ("global_qualification_granted", True),
    ("gpu_qualification_granted", True),
])
def test_resigned_fixture_cannot_claim_qualification(
    synthetic_fixture: Path, tmp_path: Path, field: str, value: object,
) -> None:
    root = tmp_path / field
    producer._copy_tree(synthetic_fixture, root)
    report = json.loads((root / "rollback-report.json").read_text(encoding="utf-8"))
    report[field] = value
    _seal(root, report)
    with pytest.raises(IllmError, match="overclaims"):
        rollback._verify(root, synthetic_test_fixture=True)


def test_old_runtime_cannot_be_given_an_invented_policy_identity(
    synthetic_fixture: Path, tmp_path: Path,
) -> None:
    root = tmp_path / "invented-old-policy"
    producer._copy_tree(synthetic_fixture, root)
    output = json.loads((root / "logs/old-generate-schema1.stdout.txt").read_text(encoding="utf-8"))
    output["numerical_policy"] = legacy_policy("fp32").to_dict()
    _write(root / "logs/old-generate-schema1.stdout.txt", output)
    execution = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    record = next(item for item in execution["commands"] if item["phase"] == "old-generate-schema1")
    record["stdout_sha256"] = sha256_file(root / record["stdout_file"])
    _write(root / "execution.json", execution)
    report = json.loads((root / "rollback-report.json").read_text(encoding="utf-8"))
    _seal(root, report)
    with pytest.raises(IllmError, match="invented a numerical policy"):
        rollback._verify(root, synthetic_test_fixture=True)


@pytest.mark.parametrize("name", sorted(rollback.RESOURCE_FIELDS))
def test_every_resource_counter_must_be_exactly_zero(name: str) -> None:
    values = {field: 0 for field in rollback.RESOURCE_FIELDS}
    values[name] = 1
    with pytest.raises(IllmError, match="zero baseline"):
        rollback._zero_resources(values)


def test_bool_is_not_a_zero_resource_counter() -> None:
    values = {field: 0 for field in rollback.RESOURCE_FIELDS}
    values["active_requests"] = False
    with pytest.raises(IllmError, match="zero baseline"):
        rollback._zero_resources(values)


@pytest.mark.parametrize("mutation", ["wrong-order", "wrapper-bypass", "shared-runtime", "extra-record-field"])
def test_command_contract_rejects_reordered_or_shared_runtime(
    synthetic_fixture: Path, mutation: str,
) -> None:
    records = json.loads((synthetic_fixture / "execution.json").read_text(encoding="utf-8"))["commands"]
    current = json.loads((synthetic_fixture / "new-source/toolchain.json").read_text(encoding="utf-8"))
    old = json.loads((synthetic_fixture / "old-source/toolchain.json").read_text(encoding="utf-8"))
    if mutation == "wrong-order":
        records[2], records[3] = records[3], records[2]
    elif mutation == "wrapper-bypass":
        records[2]["argv"][6] = "--not-wrapper-boundary"
    elif mutation == "extra-record-field":
        records[2]["claimed_success"] = True
    else:
        old["python_executable"] = current["python_executable"]
    with pytest.raises(IllmError):
        rollback._verify_commands(records, current, old)


def test_reader_import_and_full_synthetic_read_need_no_tensor_or_scripts(
    synthetic_fixture: Path, tmp_path: Path,
) -> None:
    package = tmp_path / "portable/illm"
    shutil.copytree(ROOT / "src/illm", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "import illm.stateful_policy_rollback as r;"
        "x=r._verify(Path(sys.argv[2]),synthetic_test_fixture=True);"
        "assert not ({'torch','transformers','tokenizers','safetensors'} & set(sys.modules));"
        "print(x['qualification_status'])"
    )
    process = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(package.parent), str(synthetic_fixture)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "unqualified"


def test_inventory_skips_only_local_work_and_rejects_empty_evidence_directory(tmp_path: Path) -> None:
    (tmp_path / "local-only").mkdir()
    (tmp_path / "local-only/private.bin").write_bytes(b"not portable")
    (tmp_path / "fact.json").write_bytes(b"{}")
    assert rollback.portable_stateful_policy_rollback_inventory(tmp_path) == {
        "fact.json": sha256_file(tmp_path / "fact.json"),
    }
    (tmp_path / "empty-evidence").mkdir()
    with pytest.raises(IllmError, match="empty directory"):
        rollback.portable_stateful_policy_rollback_inventory(tmp_path)


def test_inventory_rejects_hardlinked_artifacts(tmp_path: Path) -> None:
    original = tmp_path / "fact.json"
    alias = tmp_path / "alias.json"
    original.write_bytes(b"{}")
    try:
        os.link(original, alias)
    except OSError as exc:
        pytest.skip(f"host does not permit hardlinks: {exc}")
    with pytest.raises(IllmError, match="hard-linked"):
        rollback.portable_stateful_policy_rollback_inventory(tmp_path)


def test_local_only_must_itself_be_plain_not_a_reparse_or_symlink(tmp_path: Path) -> None:
    target = tmp_path / "scratch-target"
    target.mkdir()
    (target / "private.bin").write_bytes(b"scratch")
    link = tmp_path / "local-only"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host does not permit directory symlinks: {exc}")
    with pytest.raises(IllmError, match="links, reparse"):
        rollback.portable_stateful_policy_rollback_inventory(tmp_path)


def test_casefold_alias_and_boolean_integer_claims_fail_closed() -> None:
    folded: set[str] = set()
    rollback._claim_casefold("Facts/Entry.json", folded)
    with pytest.raises(IllmError, match="case-colliding"):
        rollback._claim_casefold("facts/entry.JSON", folded)
    with pytest.raises(IllmError, match="exact integer"):
        rollback._exact_int(True, expected=1, field="transition.schema_version")
    for field in ("new_requests_after_last_safe_terminal", "unrelated_processes_interrupted"):
        with pytest.raises(IllmError, match="exact integer"):
            rollback._exact_int(False, expected=0, field=field)


def test_producer_refuses_an_existing_output_without_bootstrapping_old_runtime(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve", encoding="ascii")
    with pytest.raises(IllmError, match="new directory"):
        producer.run_policy_rollback([
            "--output", str(tmp_path),
            "--v02-toolchain-manifest", str(tmp_path / "missing.json"),
            "--trusted-git-repository", str(ROOT),
            "--artifact-cache", str(ROOT / ".illm/cache/runtime"),
            "--trigger-reason", "test",
        ])
    assert sentinel.read_text(encoding="ascii") == "preserve"

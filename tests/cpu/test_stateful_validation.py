"""V0.2 stateful differential producer tests; outputs are development-only."""

from __future__ import annotations

import json
import platform
from dataclasses import replace
from pathlib import Path
import shutil
import sys
from typing import Any, Generator

import pytest
import torch

from illm.canonical import sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.executor import EagerExecutor, ExecutionTicket
from illm.model import LoadedModel, ModelError, load_model
from illm.model.validation import FIXED_PROMPTS
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_stateful as validator


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    value = load_model(project_root=ROOT)
    yield value
    torch.set_num_threads(original)


@pytest.fixture(scope="module")
def full_development_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    output = tmp_path_factory.mktemp("stateful-full") / "evidence"
    before = (
        torch.get_num_threads(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    report = validator.run_validation(
        "fixture:tiny-decoder",
        device="cpu",
        output=output,
        scope="full",
        development_no_manifest=True,
    )
    after = (
        torch.get_num_threads(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    assert after == before
    return output, report


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_real_tiny_full_scope_runs_50_new_cases_and_20_frozen_prompts(
    full_development_output: tuple[Path, dict[str, Any]],
) -> None:
    output, report = full_development_output
    assert report["status"] == "pass"
    assert report["scope"] == "full" and report["target"] == "tiny"
    assert report["qualification_role"] == "tiny-full-authority"
    assert report["expected_counts"] == report["actual_counts"] == {
        "differential": 50,
        "fixed_greedy": 20,
    }
    assert report["formal_evidence_eligible"] is False
    assert report["qualification_tier"] == "unqualified"
    assert report["first_divergence"] is None
    assert report["all_fenced_closed_resources_zero"] is True
    assert report["hardware"]["cpu_threads"] == 1
    assert report["hardware"]["tf32_matmul_enabled"] is False
    assert report["hardware"]["tf32_cudnn_enabled"] is False
    assert report["contract"]["transformers_model_loaded"] is False
    assert report["coverage"]["sampled_case_count"] == 25
    assert report["coverage"]["tiny_natural_eos_first"] is True
    assert (output / "DEVELOPMENT_ONLY").is_file()
    assert not (output / "COMPLETE").exists() and not (output / "INCOMPLETE").exists()
    assert validator.verify_stateful_output(output, require_formal=False) == report

    differential = _read(output / "raw" / "differential-cases.json")
    cases = differential["cases"]
    assert len(cases) == 50
    assert len({case["case_id"] for case in cases}) == 50
    assert len({tuple(case["input_token_ids"]) for case in cases}) == 50
    assert sum(case["sampling_spec"]["mode"] == "sample" for case in cases) == 25
    assert len(cases[0]["input_token_ids"]) == 1 and cases[0]["max_new_tokens"] == 1
    assert len(cases[1]["input_token_ids"]) == 2
    assert len(cases[2]["input_token_ids"]) == 255 and cases[2]["max_new_tokens"] == 1
    assert cases[3]["input_token_ids"] == [2]
    assert cases[3]["generated_tokens"] == [2] and cases[3]["finish_reason"] == "EOS"
    for case in cases:
        assert case["status"] == "pass" and case["resource_baseline"] is True
        assert all(value == 0 for status in case["resource_status_after_close"].values() for value in status.values())
        initial = len(case["input_token_ids"])
        for position, step in enumerate(case["steps"]):
            assert step["position"] == position
            assert step["query_tokens"]["none"] == initial + position
            assert step["query_tokens"]["contiguous"] == (initial if position == 0 else 1)
            assert step["kv_valid_after"]["none"] == 0
            assert step["kv_valid_after"]["contiguous"] == initial + position
            assert step["logits"]["mismatch_count"] == 0
            assert step["logits"]["max_absolute_error"] >= 0.0
            assert step["logits"]["max_relative_error"] >= 0.0
            assert step["chosen_token"]["none"] == step["chosen_token"]["contiguous"]
            assert step["finish"]["none"] == step["finish"]["contiguous"]
        if case["sampling_spec"]["mode"] == "sample":
            assert case["rng_draws_consumed"] == len(case["generated_tokens"])
            assert case["committed_draw_addresses"] == [
                [position, 0, 0] for position in range(len(case["generated_tokens"]))
            ]
        else:
            assert case["rng_draws_consumed"] == 0 and case["committed_draw_addresses"] == []

    fixed = _read(output / "raw" / "fixed-prompts.json")
    assert len(fixed["cases"]) == 20
    assert fixed["tiny_oracle_manifest_digest"] is not None
    assert [case["public_prompt"] for case in fixed["cases"]] == list(FIXED_PROMPTS)
    assert all(case["status"] == "pass" and len(case["generated_tokens"]) == 8 for case in fixed["cases"])
    assert fixed["cases"][0]["generated_tokens"] == [114] * 8


def test_real_smoke_scope_is_strictly_20_fixed_greedy_cases(tmp_path: Path) -> None:
    output = tmp_path / "real-smoke-shape"
    report = validator.run_validation(
        "fixture:tiny-decoder",
        device="cpu",
        output=output,
        scope="real-smoke",
        development_no_manifest=True,
    )
    assert report["status"] == "pass"
    assert report["scope"] == "real-smoke"
    assert report["qualification_role"] == "tiny-smoke-diagnostic-not-full"
    assert report["expected_counts"] == report["actual_counts"] == {
        "differential": 0,
        "fixed_greedy": 20,
    }
    assert report["coverage"]["sampled_case_count"] == 0
    assert _read(output / "raw" / "differential-cases.json")["cases"] == []
    assert len(_read(output / "raw" / "fixed-prompts.json")["cases"]) == 20
    validator.verify_stateful_output(output, require_formal=False)


def test_portable_reader_rejects_tampered_raw_step(
    full_development_output: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, _ = full_development_output
    tampered = tmp_path / "tampered"
    shutil.copytree(original, tampered)
    path = tampered / "raw" / "differential-cases.json"
    value = _read(path)
    value["cases"][0]["steps"][0]["chosen_token"]["contiguous"] += 1
    value.pop("suite_digest")
    validator._atomic_json(path, validator._signed(value, "suite_digest"))
    report_path = tampered / "report" / "stateful-validation-report.json"
    report = _read(report_path)
    report["raw_digests"]["differential"] = value["suite_digest"]
    report.pop("report_digest")
    validator._atomic_json(report_path, validator._signed(report, "report_digest"))
    marker_path = tampered / "DEVELOPMENT_ONLY"
    marker = _read(marker_path)
    marker["report_digest"] = report["report_digest"]
    validator._atomic_json(marker_path, marker)
    validator._write_checksums(tampered)
    with pytest.raises(IllmError) as raised:
        validator.verify_stateful_output(tampered, require_formal=False)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_numerical_fault_records_first_divergence_and_never_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolve = EagerExecutor.resolve
    faulted = False

    def faulty_resolve(executor: EagerExecutor, ticket: ExecutionTicket) -> tuple[float, ...]:
        nonlocal faulted
        values = original_resolve(executor, ticket)
        if executor.kv_mode == "contiguous" and not faulted:
            faulted = True
            values = (values[0] + 1.0, *values[1:])
        return values

    monkeypatch.setattr(EagerExecutor, "resolve", faulty_resolve)
    output = tmp_path / "failed"
    report = validator.run_validation(
        "fixture:tiny-decoder",
        device="cpu",
        output=output,
        scope="full",
        development_no_manifest=True,
    )
    assert faulted is True and report["status"] == "fail"
    assert report["formal_evidence_eligible"] is False
    assert report["first_divergence"]["suite"] == "differential"
    assert report["first_divergence"]["case_id"] == "differential-00"
    assert "last_row_logits" in report["first_divergence"]["reasons"]
    assert report["first_divergence"]["step"]["logits"]["mismatch_count"] > 0
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists() and not (output / "DEVELOPMENT_ONLY").exists()
    with pytest.raises(IllmError) as raised:
        validator.verify_stateful_output(output)
    assert raised.value.code is ErrorCode.CAPTURE_INCOMPLETE


def test_case_validation_rejects_context_and_sampling_contract_drift(
    loaded: LoadedModel,
) -> None:
    cases = list(validator.build_differential_cases(loaded))
    cases[0] = replace(cases[0], input_token_ids=(3,) * loaded.max_context_tokens, max_new_tokens=1)
    with pytest.raises(IllmError) as context:
        validator._validate_differential_cases(cases, loaded)
    assert context.value.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED

    cases = list(validator.build_differential_cases(loaded))
    cases[25] = replace(cases[25], sampling=SamplingSpec(mode="greedy", actual_seed=7))
    with pytest.raises(IllmError, match="at least half") as sampling:
        validator._validate_differential_cases(cases, loaded)
    assert sampling.value.code is ErrorCode.INVALID_ARGUMENT


def test_parameters_and_existing_output_fail_before_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_identity = validator._source_identity
    monkeypatch.setattr(validator, "_source_identity", lambda: pytest.fail("must reject before source/model work"))
    with pytest.raises(IllmError) as scope:
        validator.run_validation(
            "fixture:tiny-decoder", device="cpu", output=tmp_path / "scope", scope="adaptive",
            development_no_manifest=True,
        )
    assert scope.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(IllmError) as device:
        validator.run_validation(
            "fixture:tiny-decoder", device="gpu", output=tmp_path / "device",
            development_no_manifest=True,
        )
    assert device.value.code is ErrorCode.INVALID_ARGUMENT

    monkeypatch.setattr(validator, "_source_identity", source_identity)
    existing = tmp_path / "existing"
    existing.mkdir()
    preserved = existing / "preserve.txt"
    preserved.write_text("preserve", encoding="utf-8")
    with pytest.raises(IllmError) as output:
        validator.run_validation(
            "fixture:tiny-decoder", device="cpu", output=existing, development_no_manifest=True,
        )
    assert output.value.code is ErrorCode.INVALID_ARGUMENT
    assert preserved.read_text(encoding="utf-8") == "preserve"


def test_load_failure_replaces_in_progress_marker_with_exception_fact(tmp_path: Path) -> None:
    output = tmp_path / "load-failure"
    with pytest.raises(ModelError):
        validator.run_validation(
            "fixture:missing-stateful-validation-fixture",
            device="cpu",
            output=output,
            development_no_manifest=True,
        )
    marker = _read(output / "INCOMPLETE")
    assert marker == {
        "reason": "stateful_validation_exception",
        "scope": "full",
        "exception_type": "ModelError",
    }
    assert not (output / "COMPLETE").exists() and not (output / "DEVELOPMENT_ONLY").exists()


def _formal_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    source = validator._source_identity()
    environment_id = "stateful-validator-unit"
    lock_digest = "a" * 64
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "finalized": True,
        "environment_id": environment_id,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "project_root": str(ROOT),
        "project_root_digest": source["project_root_digest"],
        "bootstrap_digest": source["bootstrap_digest"],
        "lock_digest": lock_digest,
        "runtime_profile_id": "synthetic-stateful-unit",
        "runtime_lock_digest": "b" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    path = tmp_path / "toolchain.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("ILLM_TOOLCHAIN_MANIFEST", str(path))
    monkeypatch.setenv("ILLM_ENVIRONMENT_ID", environment_id)
    monkeypatch.setenv("ILLM_LOCK_DIGEST", lock_digest)
    return path, source


def test_formal_mode_requires_exact_source_bound_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    monkeypatch.delenv("ILLM_LOCK_DIGEST", raising=False)
    source = validator._source_identity()
    with pytest.raises(IllmError) as missing:
        validator._toolchain_binding(False, source)
    assert missing.value.code is ErrorCode.TOOLCHAIN_MISMATCH

    path, source = _formal_manifest(tmp_path, monkeypatch)
    binding = validator._toolchain_binding(False, source)
    assert binding["binding"] == "finalized-source-bound-manifest"
    assert binding["formal_evidence_eligible"] is True

    manifest = _read(path)
    manifest["project_root_digest"] = "f" * 64
    manifest["manifest_digest"] = sha256_json({
        key: value for key, value in manifest.items() if key != "manifest_digest"
    })
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IllmError) as stale:
        validator._toolchain_binding(False, source)
    assert stale.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_formal_manifest_bound_run_self_verifies_complete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _formal_manifest(tmp_path, monkeypatch)
    output = tmp_path / "formal-full"
    report = validator.run_validation(
        "fixture:tiny-decoder",
        device="cpu",
        output=output,
        scope="full",
    )
    assert report["status"] == "pass"
    assert report["scope"] == "full" and report["qualification_role"] == "tiny-full-authority"
    assert report["formal_evidence_eligible"] is True
    assert (output / "COMPLETE").read_text(encoding="ascii") == report["report_digest"] + "\n"
    assert not (output / "DEVELOPMENT_ONLY").exists() and not (output / "INCOMPLETE").exists()
    assert validator.verify_stateful_output(output, require_formal=True) == report
    with pytest.raises(IllmError) as development_only:
        validator.verify_stateful_output(output, require_formal=False)
    assert development_only.value.code is ErrorCode.CAPTURE_CORRUPT


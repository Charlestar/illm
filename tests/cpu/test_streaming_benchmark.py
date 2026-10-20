"""V0.2 source-bound tiny baseline and explicitly unqualified development tests."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.metrics import METRIC_IDS


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_streaming as benchmark


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _formal_manifest(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = benchmark._source_identity()
    environment_id = "streaming-benchmark-tiny-cpu-unit"
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
        "runtime_profile_id": "synthetic-tiny-cpu-learning-baseline",
        "runtime_lock_digest": "b" * 64,
    }
    manifest["manifest_digest"] = sha256_json(manifest)
    path = root / "toolchain-manifest.json"
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    monkeypatch.setenv("ILLM_TOOLCHAIN_MANIFEST", str(path))
    monkeypatch.setenv("ILLM_ENVIRONMENT_ID", environment_id)
    monkeypatch.setenv("ILLM_LOCK_DIGEST", lock_digest)
    return path


def _reseal(output: Path) -> None:
    checksums = benchmark._checksum_inventory(output)
    benchmark._atomic_json(output / "checksums.json", checksums)
    (output / "COMPLETE").write_text(
        f"{sha256_file(output / 'checksums.json')}\n",
        encoding="ascii",
    )


@pytest.fixture(scope="module")
def tiny_development_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    output = tmp_path_factory.mktemp("streaming-benchmark") / "result"
    report = benchmark.run_benchmark(
        "fixture:tiny-decoder",
        device="cpu",
        output=output,
        suite="tuning",
        target="tiny",
        warmups=1,
        repetitions=3,
        cpu_threads=1,
        development_no_manifest=True,
    )
    return output, report


@pytest.fixture(scope="module")
def tiny_formal_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("streaming-benchmark-formal")
    monkeypatch = pytest.MonkeyPatch()
    try:
        _formal_manifest(root, monkeypatch)
        output = root / "result"
        report = benchmark.run_benchmark(
            "fixture:tiny-decoder",
            device="cpu",
            output=output,
            suite="tuning",
            target="tiny",
            warmups=1,
            repetitions=3,
            cpu_threads=1,
        )
    finally:
        monkeypatch.undo()
    return output, report


@pytest.mark.parametrize("suite", ["tuning", "heldout"])
def test_frozen_workload_suites_have_external_digest_and_capacity_boundaries(suite: str) -> None:
    path = benchmark.SUITE_PATHS[suite]
    value = benchmark.load_workload_suite(path, suite=suite, target="tiny")
    assert value["suite_digest"] == benchmark.EXPECTED_SUITE_DIGESTS[suite]
    tiny = value["targets"]["tiny"]
    real = value["targets"]["real"]
    assert tiny["required_capacity_tokens"] == 256
    assert all(case["prompt_tokens"] + case["max_new_tokens"] <= 256 for case in tiny["cases"])
    assert real["required_capacity_tokens"] == 512
    assert all(
        case["prompt_tokens"] == 256 and case["max_new_tokens"] == 64
        for case in real["cases"]
    )
    benchmark.load_workload_suite(path, suite=suite, target="real")


def test_workload_resign_and_duplicate_key_tampering_fail_closed(tmp_path: Path) -> None:
    original = _json(benchmark.SUITE_PATHS["tuning"])
    changed = json.loads(json.dumps(original))
    changed["targets"]["tiny"]["cases"][0]["max_new_tokens"] = 5
    changed["suite_digest"] = sha256_json({key: value for key, value in changed.items() if key != "suite_digest"})
    resigned = tmp_path / "resigned.json"
    resigned.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(IllmError) as raised:
        benchmark.load_workload_suite(resigned, suite="tuning", target="tiny")
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT

    duplicated = tmp_path / "duplicate.json"
    text = benchmark.SUITE_PATHS["tuning"].read_text(encoding="utf-8")
    duplicated.write_text(text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1), encoding="utf-8")
    with pytest.raises(IllmError) as duplicate:
        benchmark.load_workload_suite(duplicated, suite="tuning", target="tiny")
    assert duplicate.value.code is ErrorCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    "model,target,inspected,required",
    [
        (
            "fixture:tiny-decoder",
            "tiny",
            {"fixture_id": "tiny-decoder", "max_context_tokens": 255, "spec": {}},
            256,
        ),
        (
            ROOT / "configs" / "models" / "v0_2.yaml",
            "real",
            {
                "fixture_id": None,
                "max_context_tokens": 511,
                "spec": {
                    "model_revision": benchmark.REAL_MODEL_REVISION,
                    "tokenizer_revision": benchmark.REAL_MODEL_REVISION,
                    "runtime_dtype": "bf16",
                },
            },
            512,
        ),
    ],
)
def test_predeclared_context_capacity_fails_closed(
    model: str | Path, target: str, inspected: dict[str, Any], required: int,
) -> None:
    with pytest.raises(IllmError) as raised:
        benchmark.validate_target_capacity(model, target, inspected, required)
    assert raised.value.code is ErrorCode.CONTEXT_LIMIT_EXCEEDED


@pytest.mark.parametrize(
    "warmups,repetitions,cpu_threads",
    [(0, 3, 1), (1, 2, 1), (1, 3, 2)],
)
def test_measurement_count_boundaries_reject_before_output(
    tmp_path: Path, warmups: int, repetitions: int, cpu_threads: int,
) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(IllmError) as raised:
        benchmark.run_benchmark(
            "fixture:tiny-decoder",
            device="cpu",
            output=output,
            suite="tuning",
            target="tiny",
            warmups=warmups,
            repetitions=repetitions,
            cpu_threads=cpu_threads,
            development_no_manifest=True,
        )
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert not output.exists()


def test_formal_mode_requires_manifest_before_model_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    output = tmp_path / "not-created"
    with pytest.raises(IllmError) as raised:
        benchmark.run_benchmark(
            "fixture:tiny-decoder",
            device="cpu",
            output=output,
            suite="tuning",
            target="tiny",
            warmups=1,
            repetitions=3,
        )
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH
    assert not output.exists()


def test_real_target_cannot_be_relabelled_from_cpu_before_output(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(IllmError) as raised:
        benchmark.run_benchmark(
            ROOT / "configs" / "models" / "v0_2.yaml",
            device="cpu",
            output=output,
            suite="heldout",
            target="real",
            warmups=1,
            repetitions=3,
        )
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert not output.exists()


def test_tiny_real_execution_records_all_cells_without_formal_claim(
    tiny_development_run: tuple[Path, dict[str, Any]],
) -> None:
    output, report = tiny_development_run
    assert report["status"] == "pass"
    assert report["evidence_stage"] == "development"
    assert report["qualification_tier"] == "unqualified"
    assert report["formal_evidence_eligible"] is False
    assert report["speedup_claim"] is None and report["external_engine_comparison"] is None
    assert (output / "DEVELOPMENT_ONLY").is_file()
    assert (output / "report" / "dev-report.json").is_file()
    assert not (output / "report" / "benchmark-report.json").exists()
    assert not (output / "COMPLETE").exists() and not (output / "checksums.json").exists()
    assert not (output / "INCOMPLETE").exists()

    raw = _json(output / "raw" / "observations.json")
    records = raw["records"]
    assert len(records) == 4 * (1 + 3)
    assert len({record["request_id"] for record in records}) == len(records)
    assert {record["group_id"] for record in records} == {group[0] for group in benchmark.GROUPS}
    measured_outputs: dict[int, tuple[int, ...]] = {}
    for record in records:
        assert not any(record["resource_status"].values())
        assert record["correctness"]["rng_draws_consumed"] == 0
        assert record["correctness"]["committed_draw_addresses"] == []
        assert record["correctness"]["next_output_position"] == record["committed_output_tokens"]
        prompt = record["prompt_tokens"]
        count = record["committed_output_tokens"]
        expected_query = list(range(prompt, prompt + count)) if record["group_id"].startswith("none-") else [prompt, *([1] * (count - 1))]
        assert record["query_tokens_per_step"] == expected_query
        if record["group_id"].endswith("-off"):
            assert record["capture"] is None and record["capture_manifest_digest"] is None
        else:
            assert record["capture"] is not None
            bundle = StreamingCaptureBundle.open(output / record["capture"])
            assert bundle.manifest["manifest_digest"] == record["capture_manifest_digest"]
        if record["phase"] == "measured":
            output_tokens = tuple(record["correctness"]["output_token_ids"])
            previous = measured_outputs.setdefault(record["iteration"], output_tokens)
            assert output_tokens == previous

    for group in report["groups"].values():
        assert group["warmup_count"] == 1
        assert group["measurement_count"] == 3
        metrics = group["metrics"]
        assert metrics["request_totals"] == {"offered": 3, "accepted": 3, "preaccept": 0}
        assert metrics["metrics"][METRIC_IDS[4]]["sample_count"] == 3
    assert report["correctness"]["exact_across_all_four_groups"] is True
    assert report["correctness"]["globally_unique_request_ids"] == 16
    assert report["correctness"]["accepted_request_failure_rate"] == {
        "numerator": 0, "denominator": 16, "value": 0.0,
    }


def test_source_bound_formal_tiny_cpu_run_is_complete_but_not_gpu_qualified(
    tiny_formal_run: tuple[Path, dict[str, Any]],
) -> None:
    output, report = tiny_formal_run
    assert report["status"] == "pass"
    assert report["evidence_scope"] == "tiny-cpu-learning-baseline"
    assert report["target"] == "tiny" and report["hardware"]["device"] == "cpu"
    assert report["formal_evidence_eligible"] is True
    assert report["evidence_stage"] == "qualification_candidate"
    assert report["qualification_tier"] == "unqualified"
    assert report["qualification_claim"] is None
    assert report["correctness"]["real_workload_completed_256x64"] is False
    assert report["speedup_claim"] is None and report["external_engine_comparison"] is None
    assert report["identity"]["toolchain"]["binding"] == "finalized-source-bound-manifest"
    authority = report["identity"]["authority"]
    authority_digest = report["identity"]["authority_digest"]
    assert authority_digest == sha256_json(authority)
    assert authority["suite"]["suite_digest"] == report["suite_digest"]
    assert authority["model_authority"]["fixture_id"] == "tiny-decoder"
    assert _json(output / "raw" / "observations.json")["authority_digest"] == authority_digest
    assert _json(output / "raw" / "materialized-workload.json")["authority_digest"] == authority_digest
    assert (output / "COMPLETE").is_file() and (output / "checksums.json").is_file()
    assert not (output / "DEVELOPMENT_ONLY").exists() and not (output / "INCOMPLETE").exists()
    assert benchmark.verify_completed_output(output) == report


def test_development_output_is_rejected_by_formal_reader(
    tiny_development_run: tuple[Path, dict[str, Any]],
) -> None:
    output, _ = tiny_development_run
    with pytest.raises(IllmError) as raised:
        benchmark.verify_completed_output(output)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_development_output_cannot_be_finalized(
    tmp_path: Path,
    tiny_development_run: tuple[Path, dict[str, Any]],
) -> None:
    source, _ = tiny_development_run
    output = tmp_path / "development-finalize-attempt"
    shutil.copytree(source, output)
    with pytest.raises(IllmError) as raised:
        benchmark._finalize_formal_output(output)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH
    assert not (output / "COMPLETE").exists()
    assert not (output / "checksums.json").exists()


def test_self_signed_pass_labels_cannot_create_formal_evidence(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-integrity-unit"
    report = benchmark._signed_document(
        {
            "schema_version": 1,
            "status": "pass",
            "all_semantic_checks_passed": True,
            "development_no_manifest": False,
            "formal_evidence_eligible": True,
            "evidence_stage": "qualification_candidate",
            "qualification_tier": "unqualified",
            "target": "tiny",
            "suite": "tuning",
            "suite_digest": benchmark.EXPECTED_SUITE_DIGESTS["tuning"],
            "speedup_claim": None,
            "external_engine_comparison": None,
            "unit_fixture": True,
        },
        "report_digest",
    )
    benchmark._atomic_json(output / "report" / "benchmark-report.json", report)
    benchmark._atomic_json(output / "raw" / "unit-fixture.json", {"synthetic_unit_fixture": True})
    benchmark._atomic_json(output / "INCOMPLETE", {"synthetic_unit_fixture": True})
    with pytest.raises(IllmError) as raised:
        benchmark._finalize_formal_output(output)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT
    assert not (output / "COMPLETE").exists()
    assert _json(output / "INCOMPLETE") == {
        "reason": "formal_benchmark_semantic_verification_failed",
        "formal": True,
    }


@pytest.mark.parametrize(
    "tamper",
    ["raw_schema_bool", "query_bool", "output_token", "resource_bool", "metric", "authority_fixture"],
)
def test_resigned_semantic_tampering_is_rejected(
    tmp_path: Path,
    tiny_formal_run: tuple[Path, dict[str, Any]],
    tamper: str,
) -> None:
    source, _ = tiny_formal_run
    output = tmp_path / tamper
    shutil.copytree(source, output)
    if tamper == "authority_fixture":
        report_path = output / "report" / "benchmark-report.json"
        report = _json(report_path)
        authority = report["identity"]["authority"]
        authority["model_authority"]["fixture_id"] = "forged-tiny-decoder"
        authority_digest = sha256_json(authority)
        report["identity"]["authority_digest"] = authority_digest
        report["report_digest"] = sha256_json({
            key: value for key, value in report.items() if key != "report_digest"
        })
        benchmark._atomic_json(report_path, report)
        raw_path = output / "raw" / "observations.json"
        raw = _json(raw_path)
        raw["authority_digest"] = authority_digest
        raw["raw_digest"] = sha256_json({key: value for key, value in raw.items() if key != "raw_digest"})
        benchmark._atomic_json(raw_path, raw)
        workload_path = output / "raw" / "materialized-workload.json"
        workload = _json(workload_path)
        workload["authority_digest"] = authority_digest
        workload["materialized_workload_digest"] = sha256_json({
            key: value for key, value in workload.items() if key != "materialized_workload_digest"
        })
        benchmark._atomic_json(workload_path, workload)
    elif tamper == "metric":
        report_path = output / "report" / "benchmark-report.json"
        report = _json(report_path)
        metric = report["groups"]["none-off"]["metrics"]["metrics"][METRIC_IDS[0]]
        assert metric["p99"] is not None
        metric["p99"] += 1.0
        report["report_digest"] = sha256_json({
            key: value for key, value in report.items() if key != "report_digest"
        })
        benchmark._atomic_json(report_path, report)
    else:
        raw_path = output / "raw" / "observations.json"
        raw = _json(raw_path)
        records = raw["records"]
        record: dict[str, Any] | None = None
        if tamper == "raw_schema_bool":
            raw["schema_version"] = True
        elif tamper == "query_bool":
            record = next(
                item for item in records
                if item["group_id"] == "contiguous-off" and len(item["query_tokens_per_step"]) > 1
            )
            assert record["query_tokens_per_step"][1] == 1
            record["query_tokens_per_step"][1] = True
        elif tamper == "output_token":
            record = records[0]
            record["correctness"]["output_token_ids"][0] += 1
            record["output_token_ids_digest"] = sha256_json({
                "token_ids": record["correctness"]["output_token_ids"]
            })
        else:
            record = records[0]
            assert record["resource_status"]["active_requests"] == 0
            record["resource_status"]["active_requests"] = False
        if record is not None:
            record["record_digest"] = sha256_json({
                key: value for key, value in record.items() if key != "record_digest"
            })
        raw["raw_digest"] = sha256_json({
            key: value for key, value in raw.items() if key != "raw_digest"
        })
        benchmark._atomic_json(raw_path, raw)
    _reseal(output)
    with pytest.raises(IllmError) as raised:
        benchmark.verify_completed_output(output)
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_portable_reader_works_under_isolated_no_site_runtime(
    tmp_path: Path,
    tiny_formal_run: tuple[Path, dict[str, Any]],
) -> None:
    positive, report = tiny_formal_run
    portable_source = tmp_path / "portable-src"
    shutil.copytree(ROOT / "src" / "illm", portable_source / "illm")
    positive_copy = tmp_path / "portable-positive"
    shutil.copytree(positive, positive_copy)
    negative = tmp_path / "portable-negative"
    shutil.copytree(positive, negative)
    isolated_cwd = tmp_path / "isolated-cwd"
    isolated_cwd.mkdir()
    raw_path = negative / "raw" / "observations.json"
    raw = _json(raw_path)
    raw["schema_version"] = True
    raw["raw_digest"] = sha256_json({
        key: value for key, value in raw.items() if key != "raw_digest"
    })
    benchmark._atomic_json(raw_path, raw)
    _reseal(negative)

    code = """
import json
import sys
sys.path.insert(0, sys.argv[1])
from illm.domain.errors import IllmError
from illm.stateful_benchmark import PROJECT_ROOT, verify_completed_output
positive = verify_completed_output(__import__('pathlib').Path(sys.argv[2]))
try:
    verify_completed_output(__import__('pathlib').Path(sys.argv[3]))
except IllmError as exc:
    negative_code = exc.code.value
else:
    negative_code = None
banned = sorted(name for name in ('torch', 'transformers', 'safetensors', 'tokenizers') if name in sys.modules)
print(json.dumps({'digest': positive['report_digest'], 'scope': positive['evidence_scope'],
                  'negative_code': negative_code, 'banned_imports': banned,
                  'reader_project_root': str(PROJECT_ROOT)}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            code,
            str(portable_source),
            str(positive_copy),
            str(negative),
        ],
        cwd=isolated_cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "digest": report["report_digest"],
        "scope": "tiny-cpu-learning-baseline",
        "negative_code": ErrorCode.CAPTURE_CORRUPT.value,
        "banned_imports": [],
        "reader_project_root": str(tmp_path),
    }

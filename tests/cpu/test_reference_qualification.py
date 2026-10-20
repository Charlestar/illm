"""Adversarial verifier unit fixtures, not claimed model/runtime qualification.

Numerical/timing records here are deliberately synthetic. Only unit tests patch
the live-environment assertion; production callers have no bypass parameter.
"""

from __future__ import annotations

import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEventKind as K
from illm.model.spec import SupportedModelSpec
from illm.reference_qualification import (
    CRITICAL_TESTS, RUNTIME_LOCKS, TINY_FILES, _canonical_shape, _compound_fault, _inventory, _op_order, _oracle_contract,
    _source_provenance, _tiny_fault, _tolerances, create_reference_profile_evidence,
    verify_reference_profile_evidence,
)
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _signed(path: Path, value: dict[str, Any], field: str) -> None:
    value[field] = sha256_json({key: item for key, item in value.items() if key != field})
    _write(path, value)


def _read(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return value


def _capture(path: Path, identity: dict[str, Any], request_id: str, length: int, *, sampling: SamplingSpec | None = None,
             ops: bool = False) -> dict[str, Any]:
    spec = sampling or SamplingSpec(actual_seed=7)
    session = ReferenceCaptureSession(path, identity, request_id, spec.to_dict())
    session.emit(K.REQUEST_ACCEPTED, {"max_new_tokens": 1})
    session.emit(K.REQUEST_TOKENIZED, {"input_tokens": length, "input_digest": "a" * 64})
    session.emit(K.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": length})
    if ops:
        raw = _read(ROOT / "fixtures" / "tiny-decoder" / "supported-model-spec.json")
        model_spec = SupportedModelSpec(**{name: value for name, value in raw.items() if name != "spec_id"})
        for name in _op_order(model_spec.num_layers):
            session.emit(K.MODEL_OP, {"output_position": 0, "op_name": name,
                                     "shape": _canonical_shape(name, model_spec, length), "dtype": "torch.float32"})
    session.emit(K.MODEL_FORWARD_COMPLETED, {"output_position": 0, "logits_digest": "b" * 64})
    session.emit(K.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": spec.digest})
    session.emit(K.SAMPLING_SELECTED, {"output_position": 0, "sampling_spec_digest": spec.digest,
                                     "logits_digest": "b" * 64, "draw_address": [0, 0, 0] if spec.mode == "sample" else None,
                                     "selected_digest": "c" * 64})
    session.emit(K.REQUEST_FINISHED, {"output_tokens": 1, "finish_reason": "MAX_NEW_TOKENS", "output_digest": "d" * 64})
    return session.finalize()


def _validation(path: Path, toolchain: dict[str, Any], fault_identity: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for destination, original in (("model-spec.json", "supported-model-spec.json"),
                                  ("weight-load-manifest.json", "weight-load-manifest.json"),
                                  ("tokenizer-spec.json", "tokenizer.json")):
        value = _read(ROOT / "fixtures" / "tiny-decoder" / original)
        _write(path / destination, value)
        metadata[destination] = {"path": destination, "sha256": sha256_file(path / destination)}
    model = _read(path / "model-spec.json")
    model_spec = SupportedModelSpec(**{name: value for name, value in model.items() if name != "spec_id"})
    weights = _read(path / "weight-load-manifest.json")
    identity = {"spec_id": model["spec_id"], "weights_digest": weights["weights_digest"],
                "load_manifest_digest": weights["manifest_digest"], "tokenizer_revision": model["tokenizer_revision"],
                "fixture_id": "tiny-decoder", "capture_identity": "public-fixed-model-validation-v1"}
    tolerance = _tolerances("fp32")[0]
    order = _op_order(2)

    def op_report(link: str, op_identity: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": 1, "status": "pass", "identity": op_identity, "first_divergence": None,
                "op_count": len(order), "unexpected_ops": [], "report_link": link, "tolerance_profile": tolerance,
                "ops": [{"op": name, "status": "pass", "evidence_kind": "measured", "tolerance": tolerance,
                         "report_link": link, "expected_shape": _canonical_shape(name, model_spec, 1), "actual_shape": _canonical_shape(name, model_spec, 1),
                         "expected_dtype": "torch.float32", "actual_dtype": "torch.float32",
                         "max_absolute_error": 0.0, "max_relative_error": 0.0, "mismatch_count": 0,
                         "masked_negative_infinity_count": 0} for name in order]}

    ops: list[dict[str, Any]] = []
    greedy: list[dict[str, Any]] = []
    for index in range(20):
        fixture = f"fixed-greedy-{index:02d}"
        input_identity = sha256_json(fixture)
        link = f"ops/input-{index:02d}.json"
        op = op_report(link, {**identity, "input_identity": input_identity, "public_input_fixture_id": fixture})
        _write(path / link, op)
        ops.append(op)
        greedy.append({"public_input_fixture_id": fixture, "input_identity": input_identity,
                       "status": "pass", "reference_tokens": [75] * 8, "oracle_tokens": [75] * 8, "generated_count": 8})
    fault_identity = {**fault_identity, "fault_execution_identity": sha256_json(fault_identity)}
    fault = op_report("fault-transposed-q.json", fault_identity)
    fault["status"] = "fail"
    fault["ops"][2].update(status="fail", reason="outside_tolerance", mismatch_count=1,
                            expected_shape=[1, 4, 6, 8], actual_shape=[1, 4, 6, 8],
                            first_flat_index=0, expected_value=1.0, actual_value=2.0, max_absolute_error=1.0, max_relative_error=1.0)
    fault["first_divergence"] = dict(fault["ops"][2])
    fault["output_comparison"] = {"evidence_kind": "measured", "final_logits_changed_count": 1,
                                  "final_logits_max_absolute_error": 0.001, "expected_next_token": 114,
                                  "actual_next_token": 114, "token_mismatch": False}
    _write(path / "fault-transposed-q.json", fault)
    compound_identity = _compound_fault(path.parent, {key: value for key, value in fault_identity.items() if key != "fault_execution_identity"})
    compound_identity["fault_execution_identity"] = sha256_json(compound_identity)
    compound = copy.deepcopy(fault)
    compound["identity"] = compound_identity
    compound["report_link"] = "fault-q-transpose-and-zero-final-norm.json"
    for op in compound["ops"]:
        op["report_link"] = compound["report_link"]
    compound["first_divergence"]["report_link"] = compound["report_link"]
    compound["output_comparison"].update(actual_next_token=0, token_mismatch=True)
    _write(path / compound["report_link"], compound)
    report = {"schema_version": 1, "status": "pass", "identity": identity, "device": "cpu", "runtime_dtype": "fp32",
              "torch_version": toolchain["installed_packages"]["torch"], "transformers_version": "4.57.1", "tf32_enabled": False,
              "oracle_contract": _oracle_contract("fp32"),
              "nonfinite_policy": "only-paired-strict-causal-future-negative-infinity-in-attention-scores-v1",
              "model_artifacts": metadata, "tolerance_profile": tolerance, "attention_probability_tolerance": tolerance,
              "fixed_input_count": 20, "per_op_reports": ops, "greedy_reports": greedy,
              "first_divergence_diagnostic": {"status": "pass", "report": fault, "report_link": "fault-transposed-q.json"},
              "token_mismatch_diagnostic": {"status": "pass", "report": compound, "report_link": compound["report_link"]}}
    _signed(path / "validation-report.json", report, "report_digest")
    return report


def _benchmark(path: Path, toolchain: dict[str, Any], identity: dict[str, Any]) -> None:
    path.mkdir()
    buckets: list[dict[str, Any]] = []
    invariance_sampling = SamplingSpec(mode="sample", temperature=0.8, top_k=50, top_p=0.9, actual_seed=7)
    result_fields = ["input_token_ids", "output_token_ids", "finish_reason", "actual_seed", "sampling_spec_digest",
                     "next_output_position", "rng_draws_consumed", "committed_draw_addresses"]
    invariance: dict[str, Any] = {"schema_version": 1, "status": "pass", "evidence_kind": "measured",
                                  "scope": "fixed-public-synthetic-context-fixtures-only",
                                  "token_id_disclosure": "public-synthetic-fixture-results-only; captures-remain-metadata-only",
                                  "modes": ["off", "request", "ops"], "comparison_fields": result_fields,
                                  "sampling_spec": invariance_sampling.to_dict(), "sampling_spec_digest": invariance_sampling.digest,
                                  "context_buckets": []}
    for length in (8, 32, 128):
        observations: list[dict[str, Any]] = []
        for repetition in range(3):
            link = f"captures/context-{length}/repeat-{repetition}"
            manifest = _capture(path / link, identity, f"measured-{length}-{repetition}", length)
            observations.append({"repetition": repetition, "context_tokens": length, "output_tokens": 1,
                                 "server_full_context_forward_wall_time_ns": 1000 + repetition,
                                 "logits_host_export_wall_time_ns": 100 + repetition, "sampling_wall_time_ns": 200 + repetition,
                                 "total_request_wall_time_ns": 2000 + repetition, "peak_device_allocated_bytes": None,
                                 "peak_device_reserved_bytes": None, "capture": link, "capture_manifest_digest": manifest["manifest_digest"]})
        ids = tuple(3 + index % 256 for index in range(length))
        fixture_id = sha256_json({"schema_version": 1, "kind": "ascending-vocab-v1", "input_ids": ids})
        outcomes: list[dict[str, Any]] = []
        for mode in ("off", "request", "ops"):
            link = f"telemetry-invariance/context-{length}/{mode}"
            manifest = _capture(path / link, identity, f"telemetry-{length}", length, sampling=invariance_sampling, ops=mode == "ops")
            result = {"input_token_ids": list(ids), "output_token_ids": [75], "finish_reason": "MAX_NEW_TOKENS",
                      "actual_seed": 7, "sampling_spec_digest": invariance_sampling.digest, "next_output_position": 1,
                      "rng_draws_consumed": 1, "committed_draw_addresses": [[0, 0, 0]]}
            outcomes.append({"telemetry": mode, "request_id": f"telemetry-{length}", "generation_result": result,
                             "generation_result_digest": sha256_json(result), "capture": link,
                             "capture_manifest_digest": manifest["manifest_digest"]})
        invariance["context_buckets"].append({"context_tokens": length, "public_input_fixture_id": fixture_id,
                                             "status": "pass", "outcomes": outcomes})
        buckets.append({"context_tokens": length, "public_input_fixture_id": fixture_id,
                        "observations": observations, "median_forward_wall_time_ns": 1001.0, "min_forward_wall_time_ns": 1000,
                        "max_forward_wall_time_ns": 1002, "relative_median_absolute_deviation": 1 / 1001, "stability": "pass"})
    report = {"schema_version": 1, "status": "pass", "experiment_id": "reference-full-context-v1", "performance_outcome": "baseline_only",
              "evidence_kind": "measured", "speedup_claim": None, "hardware": {"device": "cpu", "cpu_threads": 1, "torch_version": toolchain["installed_packages"]["torch"]},
              "identity": {"model_spec_id": identity["model_spec_id"], "weights_digest": identity["weights_digest"],
                           "toolchain_manifest_digest": toolchain["manifest_digest"], "project_root_digest": toolchain["project_root_digest"]},
              "measurement_contract": {"warmups_per_bucket": 1, "repetitions_per_bucket": 3, "output_tokens_per_repetition": 1,
                                       "clock": "perf_counter_ns", "tf32_enabled": False, "telemetry": "request", "synchronization": "synchronous_cpu",
                                       "stability_rule": "relative_median_absolute_deviation<=0.20; no observations discarded",
                                       "partial_timing_spans_are_total_overhead": False, "telemetry_invariance_is_timing_comparison": False,
                                       "cpu_peak_memory_measured": False, "load_time_excluded": True},
              "telemetry_invariance": invariance, "context_buckets": buckets}
    _signed(path / "benchmark-report.json", report, "report_digest")
    _write(path / "checksums.json", _inventory(path))
    (path / "COMPLETE").touch()


@pytest.fixture(scope="module")
def base_sealed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("reference-qualification-unit-inputs")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    provenance = _source_provenance(ROOT)
    # Unit fixture uses the real pinned closure metadata, not an installed-env claim.
    runtime_id = "py312-win-amd64-cpu"
    runtime = _read(ROOT / "runtime" / "artifacts" / f"{runtime_id}.json")
    versions: dict[str, str] = {}
    import re

    for lock in ("requirements-bootstrap.lock", "requirements-dev.lock"):
        for line in (ROOT / lock).read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line.strip())
            if match:
                versions[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    versions.update({item["name"]: item["version"] for item in runtime["artifacts"]})
    versions["illm"] = provenance["project_version"]
    source = provenance["source_identity"]
    toolchain = {"schema_version": 1, "finalized": True, "profile": "unit-fixture-not-real-qualification",
                 "os": "Windows", "arch": "AMD64", "python_version": "3.12.14", "project_root": str(ROOT),
                 "environment_root": "/absent/unit-only/environment", "python_executable": "/absent/unit-only/python",
                 "entrypoints": {"illm": "/absent/unit-only/illm"}, "environment_id": "unit-only-not-installed",
                 "project_root_digest": source["project_root_digest"], "bootstrap_digest": source["bootstrap_digest"],
                 "lock_digest": sha256_json({"base_lock_digest": source["base_lock_digest"], "runtime_lock_digest": RUNTIME_LOCKS[runtime_id]}),
                 "runtime_profile_id": runtime_id, "runtime_lock_digest": RUNTIME_LOCKS[runtime_id], "artifact_cache": "/absent/unit-only/cache",
                 "installed_packages": dict(sorted(versions.items())),
                 "runtime_artifacts": [{key: item[key] for key in ("name", "version", "filename", "sha256", "size_bytes")} for item in runtime["artifacts"]],
                 "runtime_probe": {"torch": "2.8.0+cpu", "transformers": "4.57.1", "safetensors": "0.6.2", "tokenizers": "0.22.1",
                                   "device": "cpu", "cuda_build": None, "gpu_name": None, "device_capability": None, "compiled_arches": [], "smoke_sum": 3.0}}
    _signed(inputs / "toolchain.json", toolchain, "manifest_digest")
    suite = ET.Element("testsuite", tests=str(len(CRITICAL_TESTS)), failures="0", errors="0", skipped="0", time="1.0")
    for name in sorted(CRITICAL_TESTS):
        ET.SubElement(suite, "testcase", name=name, classname="unit.synthetic.only")
    ET.ElementTree(suite).write(inputs / "junit.xml", encoding="utf-8")
    _write(inputs / "quality.json", {"schema_version": 0, "status": "pass", "issues": [],
                                    "checked_python_files": sum(name.startswith("src/illm/") and name.endswith(".py") for name in provenance["project_files"]),
                                    "type_check": {"tool": "mypy", "version": "1.18.2", "status": "pass", "returncode": 0,
                                                   "output": ["Success: no issues found in 100 source files"]}})
    _write(inputs / "source-provenance.json", provenance)
    tiny = inputs / "tiny-fixture"
    tiny.mkdir()
    for name in TINY_FILES:
        shutil.copy2(ROOT / "fixtures" / "tiny-decoder" / name, tiny / name)
    fault = _tiny_fault(inputs)
    validation = _validation(inputs / "validation", toolchain, fault)
    identity = {"model_spec_id": validation["identity"]["spec_id"], "weights_digest": validation["identity"]["weights_digest"],
                "load_manifest_digest": validation["identity"]["load_manifest_digest"],
                "tokenizer_digest": sha256_json(_read(inputs / "validation" / "tokenizer-spec.json")),
                "input_spec_digest": "e" * 64, "runtime_dtype": "fp32", "device": "cpu", "backend": "illm_eager"}
    _capture(inputs / "capture", identity, "unit-synthetic-capture", 8)
    _benchmark(inputs / "benchmark", toolchain, identity)
    output = tmp_path / "sealed-unit-fixture"
    with patch("illm.reference_qualification._assert_live_environment", return_value=provenance):
        create_reference_profile_evidence(profile_id="tiny-reference-cpu-v1", toolchain_manifest=inputs / "toolchain.json",
                                          junit=inputs / "junit.xml", quality=inputs / "quality.json", capture=inputs / "capture",
                                          validation_directories=[inputs / "validation"], benchmark=inputs / "benchmark",
                                          sampler_fixture=ROOT / "fixtures" / "sampling" / "sampler-known-answer-v1.json", output=output,
                                          expected_source_identity=source)
    return output


@pytest.fixture
def sealed(base_sealed: Path, tmp_path: Path) -> Path:
    result = tmp_path / "sealed-unit-fixture"
    shutil.copytree(base_sealed, result)
    return result


def _resign(root: Path) -> None:
    index = _read(root / "evidence-index.json")
    files = {name: digest for name, digest in _inventory(root).items() if name not in {"profile-result.json", "evidence-index.json", "SHA256SUMS.json", "COMPLETE"}}
    index["files"] = files
    index["evidence_digest"] = sha256_json({"schema_version": 1, "profile_id": index["profile_id"], "source_identity": index["source_identity"], "files": files})
    _signed(root / "evidence-index.json", index, "index_digest")
    profile = _read(root / "profile-result.json")
    profile["evidence_digest"] = index["evidence_digest"]
    _write(root / "profile-result.json", profile)
    _write(root / "SHA256SUMS.json", {name: digest for name, digest in _inventory(root).items() if name != "SHA256SUMS.json"})


def test_reference_profile_roundtrip_is_portable_and_source_bound(sealed: Path, tmp_path: Path) -> None:
    profile, index = verify_reference_profile_evidence(sealed)
    assert profile["profile_id"] == "tiny-reference-cpu-v1"
    moved = tmp_path / "relocated"
    shutil.copytree(sealed, moved)
    assert verify_reference_profile_evidence(moved) == (profile, index)
    assert verify_reference_profile_evidence(moved, expected_source_identity=index["source_identity"])[0] == profile
    wrong = {**index["source_identity"], "project_root_digest": "f" * 64}
    with pytest.raises(IllmError) as raised:
        verify_reference_profile_evidence(moved, expected_source_identity=wrong)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_ci_only_runtime_cannot_grant_formal_release_qualification(sealed: Path) -> None:
    assert set(RUNTIME_LOCKS) == {"py312-win-amd64-cpu", "py312-linux-x86_64-cpu", "py312-linux-x86_64-cu126"}
    ci = _read(ROOT / "runtime" / "artifacts" / "py312-win-amd64-cpu-ci.json")
    path = sealed / "toolchain.json"
    toolchain = _read(path)
    toolchain.update(runtime_profile_id=ci["profile_id"], runtime_lock_digest=ci["lock_digest"], python_version="3.12.10")
    _signed(path, toolchain, "manifest_digest")
    _resign(sealed)
    with pytest.raises(IllmError, match="runtime profile/lock is not pinned by V0.1") as raised:
        verify_reference_profile_evidence(sealed)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_formal_qualification_rejects_resigned_wrong_python_patch(sealed: Path) -> None:
    path = sealed / "toolchain.json"
    toolchain = _read(path)
    toolchain["python_version"] = "3.12.10"
    _signed(path, toolchain, "manifest_digest")
    _resign(sealed)
    with pytest.raises(IllmError, match="platform differs from pinned CPython3.12"):
        verify_reference_profile_evidence(sealed)


@pytest.mark.parametrize("mutation", ["partial", "checksum", "extra", "missing", "path-traversal", "status-only", "gpu-label", "runtime", "mypy", "skipped-critical", "no-greedy", "token-mismatch", "loose-tolerance", "missing-op", "bad-fault", "sampler", "median", "capture-reuse", "bool-schema", "shape", "oracle-contract", "fake-gpu-oom", "unreferenced-resigned", "fake-final-package", "masked-nonfinite"])
def test_resigned_incomplete_or_semantically_invalid_profile_is_rejected(sealed: Path, mutation: str) -> None:
    if mutation == "partial":
        (sealed / "INCOMPLETE").touch()
    elif mutation == "checksum":
        (sealed / "quality.json").write_text("{}", encoding="utf-8")
    elif mutation == "extra":
        (sealed / "unindexed.json").write_text("{}", encoding="utf-8")
    elif mutation == "missing":
        (sealed / "capture" / "events.jsonl").unlink()
    elif mutation == "path-traversal":
        sums = _read(sealed / "SHA256SUMS.json")
        sums["../escape"] = "0" * 64
        _write(sealed / "SHA256SUMS.json", sums)
    elif mutation == "status-only":
        _write(sealed / "quality.json", {"status": "pass"})
        _resign(sealed)
    elif mutation == "bool-schema":
        value = _read(sealed / "quality.json")
        value["schema_version"] = False
        _write(sealed / "quality.json", value)
        _resign(sealed)
    elif mutation == "unreferenced-resigned":
        _write(sealed / "fake-extra.json", {"status": "pass"})
        _resign(sealed)
    elif mutation == "fake-gpu-oom":
        directory = sealed / "gpu-failure"
        shutil.copytree(sealed / "capture", directory / "capture")
        _signed(directory / "oom-report.json", {"status": "pass", "genuine_torch_oom": True}, "report_digest")
        _resign(sealed)
    elif mutation == "fake-final-package":
        value = _read(sealed / "source-provenance.json")
        value["project_version"] = "0.1.0"
        _write(sealed / "source-provenance.json", value)
        _resign(sealed)
    elif mutation == "gpu-label":
        profile, index = _read(sealed / "profile-result.json"), _read(sealed / "evidence-index.json")
        profile["profile_id"] = index["profile_id"] = "real-model-single-gpu-v1"
        _write(sealed / "profile-result.json", profile)
        _write(sealed / "evidence-index.json", index)
        _resign(sealed)
    elif mutation in {"runtime", "mypy"}:
        if mutation == "runtime":
            path = sealed / "toolchain.json"
            value = _read(path)
            value["installed_packages"]["torch"] = "0.0"
            _signed(path, value, "manifest_digest")
        else:
            value = _read(sealed / "quality.json")
            value["type_check"]["version"] = "0.0"
            _write(sealed / "quality.json", value)
        _resign(sealed)
    elif mutation == "skipped-critical":
        path = sealed / "junit.xml"
        tree = ET.parse(path)
        suite = tree.getroot()
        suite.set("skipped", "1")
        ET.SubElement(suite.findall("testcase")[0], "skipped")
        tree.write(path, encoding="utf-8")
        _resign(sealed)
    elif mutation in {"no-greedy", "token-mismatch", "loose-tolerance", "missing-op", "bad-fault", "shape", "oracle-contract", "masked-nonfinite"}:
        path = sealed / "validation" / "model-0" / "validation-report.json"
        value = _read(path)
        if mutation == "no-greedy":
            value["greedy_reports"] = []
        elif mutation == "token-mismatch":
            value["greedy_reports"][0]["oracle_tokens"][0] += 1
        elif mutation == "loose-tolerance":
            value["tolerance_profile"]["atol"] = 1
        elif mutation == "missing-op":
            value["per_op_reports"][0]["ops"].pop()
            _write(path.parent / value["per_op_reports"][0]["report_link"], value["per_op_reports"][0])
        elif mutation == "shape":
            op = value["per_op_reports"][0]["ops"][0]
            op["expected_shape"] = op["actual_shape"] = [1, 1, 1]
            _write(path.parent / value["per_op_reports"][0]["report_link"], value["per_op_reports"][0])
        elif mutation == "oracle-contract":
            value["oracle_contract"]["rope_inv_freq_dtype"] = "torch.bfloat16"
        elif mutation == "masked-nonfinite":
            value["per_op_reports"][0]["ops"][0]["masked_negative_infinity_count"] = 1
            _write(path.parent / value["per_op_reports"][0]["report_link"], value["per_op_reports"][0])
        else:
            value["first_divergence_diagnostic"]["report"]["identity"]["fault_transform"]["mutated_tensor_checksum"] = "0" * 64
            _write(path.parent / "fault-transposed-q.json", value["first_divergence_diagnostic"]["report"])
        _signed(path, value, "report_digest")
        _resign(sealed)
    elif mutation == "sampler":
        (sealed / "sampling" / "known-answer.json").write_text("{}", encoding="utf-8")
        _resign(sealed)
    elif mutation in {"median", "capture-reuse"}:
        path = sealed / "benchmark" / "benchmark-report.json"
        value = _read(path)
        if mutation == "median":
            value["context_buckets"][0]["median_forward_wall_time_ns"] += 10
        else:
            observations = value["context_buckets"][0]["observations"]
            observations[1]["capture"] = observations[0]["capture"]
            observations[1]["capture_manifest_digest"] = observations[0]["capture_manifest_digest"]
        _signed(path, value, "report_digest")
        checksums = _read(path.parent / "checksums.json")
        checksums["benchmark-report.json"] = sha256_file(path)
        _write(path.parent / "checksums.json", checksums)
        _resign(sealed)
    with pytest.raises(IllmError):
        verify_reference_profile_evidence(sealed)


def test_sealer_cannot_run_outside_finalized_environment(sealed: Path, tmp_path: Path) -> None:
    with pytest.raises(IllmError) as raised:
        create_reference_profile_evidence(profile_id="tiny-reference-cpu-v1", toolchain_manifest=sealed / "toolchain.json",
                                          junit=sealed / "junit.xml", quality=sealed / "quality.json", capture=sealed / "capture",
                                          validation_directories=[sealed / "validation" / "model-0"], benchmark=sealed / "benchmark",
                                          sampler_fixture=sealed / "sampling" / "known-answer.json", output=tmp_path / "forbidden")
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("mutation", ["token", "finish", "address", "seed", "missing-mode", "capture-reuse", "no-ops",
                                      "negative-host-export", "float-host-export", "boolean-sampling", "missing-sampling",
                                      "partial-spans-exceed-total", "overhead-claim"])
def test_benchmark_telemetry_guardrails_reject_fully_resigned_semantic_changes(sealed: Path, mutation: str) -> None:
    directory = sealed / "benchmark"
    path = directory / "benchmark-report.json"
    value = _read(path)
    outcomes = value["telemetry_invariance"]["context_buckets"][0]["outcomes"]
    result = outcomes[2]["generation_result"]
    if mutation == "token":
        result["output_token_ids"][0] += 1
    elif mutation == "finish":
        result["finish_reason"] = "EOS"
    elif mutation == "address":
        result["committed_draw_addresses"][0][2] = 1
    elif mutation == "seed":
        result["actual_seed"] += 1
    elif mutation == "missing-mode":
        outcomes.pop()
    elif mutation == "capture-reuse":
        outcomes[2]["capture"] = outcomes[0]["capture"]
        outcomes[2]["capture_manifest_digest"] = outcomes[0]["capture_manifest_digest"]
    elif mutation == "no-ops":
        target = directory / outcomes[2]["capture"]
        shutil.rmtree(target)
        shutil.copytree(directory / outcomes[0]["capture"], target)
        outcomes[2]["capture_manifest_digest"] = _read(target / "capture-manifest.json")["manifest_digest"]
    elif mutation == "negative-host-export":
        value["context_buckets"][0]["observations"][0]["logits_host_export_wall_time_ns"] = -1
    elif mutation == "float-host-export":
        value["context_buckets"][0]["observations"][0]["logits_host_export_wall_time_ns"] = 1.5
    elif mutation == "boolean-sampling":
        value["context_buckets"][0]["observations"][0]["sampling_wall_time_ns"] = True
    elif mutation == "missing-sampling":
        value["context_buckets"][0]["observations"][0].pop("sampling_wall_time_ns")
    elif mutation == "partial-spans-exceed-total":
        value["context_buckets"][0]["observations"][0]["sampling_wall_time_ns"] = 1500
    else:
        value["measurement_contract"]["partial_timing_spans_are_total_overhead"] = True
    for outcome in outcomes:
        outcome["generation_result_digest"] = sha256_json(outcome["generation_result"])
    _signed(path, value, "report_digest")
    _write(directory / "checksums.json", {name: digest for name, digest in _inventory(directory).items() if name not in {"checksums.json", "COMPLETE"}})
    _resign(sealed)
    with pytest.raises(IllmError):
        verify_reference_profile_evidence(sealed)


def test_cleanroom_profile_cannot_be_claimed_without_phase_transcripts(sealed: Path) -> None:
    profile, index = _read(sealed / "profile-result.json"), _read(sealed / "evidence-index.json")
    profile["profile_id"] = index["profile_id"] = "clean-room-release-v1"
    _write(sealed / "profile-result.json", profile)
    _write(sealed / "evidence-index.json", index)
    _resign(sealed)
    with pytest.raises(IllmError):
        verify_reference_profile_evidence(sealed)


def _add_gate(root: Path) -> None:
    """Synthetic transcript fixture for validating cross-artifact semantics only."""
    gate = root / "gate"
    toolchain = _read(root / "toolchain.json")
    validation = _read(root / "validation" / "model-0" / "validation-report.json")
    capture = _read(root / "capture" / "capture-manifest.json")
    state = _read(root / "capture" / "checkpoint.json")
    spec = _read(root / "validation" / "model-0" / "model-spec.json")
    weights = _read(root / "validation" / "model-0" / "weight-load-manifest.json")
    benchmark = _read(root / "benchmark" / "benchmark-report.json")
    cases = len(CRITICAL_TESTS)
    results: dict[str, Any] = {
        "bootstrap": toolchain, "quality": _read(root / "quality.json"), "pytest": f"{cases} passed in 1.0s",
        "inspect": {"spec": spec, "weight_manifest": weights},
        "generate": {"capture_manifest_digest": capture["manifest_digest"], "model_spec_id": spec["spec_id"],
                     "sampling_spec_digest": state["sampling_spec_digest"], "actual_seed": 7,
                     "input_token_ids": [75] * 8, "output_token_ids": [75], "next_output_position": 1, "rng_draws_consumed": 0},
        "validate": validation,
        "replay-trace": {"mode": "R0", "displayed_events": 1, "total_events": capture["event_count"], "capture_schema_version": 1, "privacy_mode": "metadata_only"},
        "replay-steps": {"mode": "R1_REFERENCE", "asserted": True, "numerical_reexecution": False,
                         "final_state_digest": capture["final_state_digest"], "status": "FINISHED", "output_tokens": 1, "rng_draws_consumed": 0},
        "benchmark": {"status": "pass", "report_digest": benchmark["report_digest"]},
    }
    children = {"quality": ["scripts/check_quality.py"], "pytest": ["-m", "pytest", "--junitxml", "unit.xml"],
                "inspect": ["-m", "illm", "model", "inspect"], "generate": ["-m", "illm", "generate", "--capture", "capture"],
                "validate": ["-m", "illm", "model", "validate"], "replay-trace": ["-m", "illm", "replay", "capture", "--mode", "trace"],
                "replay-steps": ["-m", "illm", "replay", "capture", "--mode", "state", "--assert-state-digest"],
                "benchmark": ["scripts/benchmark_reference.py"]}
    commands: list[dict[str, Any]] = []
    for phase, result in results.items():
        stdout = gate / "logs" / f"{phase}.stdout.txt"
        stderr = gate / "logs" / f"{phase}.stderr.txt"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(result if isinstance(result, str) else json.dumps(result), encoding="utf-8")
        stderr.touch()
        argv = ["python", "scripts/bootstrap.py", "--locked", "--offline", "--runtime-profile", "v01-cpu"] if phase == "bootstrap" else ["python", "scripts/in_env.py", "--", "python", *children[phase]]
        commands.append({"phase": phase, "argv": argv, "returncode": 0,
                         "stdout_file": stdout.relative_to(gate).as_posix(), "stderr_file": stderr.relative_to(gate).as_posix(),
                         "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr)})
    report = {"schema_version": 1, "status": "pass", "toolchain_manifest_digest": toolchain["manifest_digest"],
              "source_identity": _read(root / "evidence-index.json")["source_identity"], "fresh_environment": True,
              "offline_bootstrap": True, "network_used": "not_observed", "network_blocking": "not_enforced",
              "network_policy": "offline_resolution_and_local_model_only", "commands": commands}
    _signed(gate / "gate-report.json", report, "report_digest")
    profile, index = _read(root / "profile-result.json"), _read(root / "evidence-index.json")
    profile["profile_id"] = index["profile_id"] = "clean-room-release-v1"
    _write(root / "profile-result.json", profile)
    _write(root / "evidence-index.json", index)
    _resign(root)


def test_cleanroom_transcripts_must_bind_actual_artifact_results(sealed: Path) -> None:
    _add_gate(sealed)
    assert verify_reference_profile_evidence(sealed)[0]["profile_id"] == "clean-room-release-v1"
    report_path = sealed / "gate" / "gate-report.json"
    report = _read(report_path)
    command = next(item for item in report["commands"] if item["phase"] == "validate")
    transcript = report_path.parent / command["stdout_file"]
    transcript.write_text('{"status":"pass"}', encoding="utf-8")
    command["stdout_sha256"] = sha256_file(transcript)
    _signed(report_path, report, "report_digest")
    _resign(sealed)
    with pytest.raises(IllmError, match="transcript does not bind"):
        verify_reference_profile_evidence(sealed)

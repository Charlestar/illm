"""Driver unit tests use synthetic process/timing results, not release evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from illm.canonical import sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.engine import GenerationRequest, GenerationResult


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_reference as benchmark
import run_v01_gate as gate


@pytest.mark.parametrize("profile_id,device,fp16", [
    ("tiny-reference-cpu-v1", "cpu", None),
    ("clean-room-release-v1", "cpu", None),
    ("real-model-single-gpu-v1", "cuda:1", "configs/models/v0_1_fp16.yaml"),
])
def test_gate_executes_manifest_runner_phases_and_forwards_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_id: str, device: str, fp16: str | None,
) -> None:
    project = tmp_path / "project"
    output = project / "out" / "gate"
    monkeypatch.setattr(gate, "__file__", str(project / "scripts" / "run_v01_gate.py"))
    monkeypatch.setattr(gate, "lock_digest", lambda root: "b" * 64)
    invoked: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        invoked.append(command)
        environments.append(kwargs["env"])
        if "--write-manifest" in command:
            manifest = {"manifest_digest": "a" * 64, "project_root_digest": "c" * 64, "bootstrap_digest": "d" * 64}
            path = Path(command[command.index("--write-manifest") + 1])
            path.write_text(json.dumps(manifest), encoding="utf-8")
            Path(command[command.index("--write-manifest-digest") + 1]).write_text("a" * 64, encoding="ascii")
        return subprocess.CompletedProcess(command, 0, b'{"synthetic_unit_result":true}\n', b"")

    monkeypatch.setattr(subprocess, "run", process)
    arguments = ["--profile", "driver-unit", "--runtime-profile", "v01-cpu", "--profile-id", profile_id,
                 "--output", str(output), "--device", device]
    if fp16:
        arguments.extend(["--model", "configs/models/v0_1.yaml", "--fp16-model", fp16])
    assert gate.run_gate(arguments) == 0
    report = json.loads((output / "gate-report.json").read_text(encoding="utf-8"))
    assert report["report_digest"] == sha256_json({key: value for key, value in report.items() if key != "report_digest"})
    expected = ["bootstrap", "quality", "pytest", "inspect", "generate", "validate"]
    if fp16:
        expected.append("validate-fp16")
    expected.extend(["replay-trace", "replay-steps", "benchmark"])
    if fp16:
        expected.append("gpu-oom")
    assert [record["phase"] for record in report["commands"]] == expected
    assert "--locked" in invoked[0] and "--offline" in invoked[0]
    for command in invoked[1:]:
        assert Path(command[1]).name == "in_env.py"
        assert "--expected-manifest-digest-file" in command and "--" in command
    for environment in environments:
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
    for record in report["commands"]:
        for stream in ("stdout", "stderr"):
            assert record[f"{stream}_sha256"] == sha256_file(output / record[f"{stream}_file"])
    qualifier = invoked[-1]
    assert "qualify_v01.py" in " ".join(qualifier)
    assert ("--clean-room-report" in qualifier) == (profile_id == "clean-room-release-v1")
    assert ("--gpu-failure-evidence" in qualifier) == bool(fp16)
    if fp16:
        oom = next(command for command in invoked if "scripts/probe_gpu_oom.py" in command)
        assert oom[oom.index("--device") + 1] == device
        assert qualifier.count("--validation-directory") == 2
    assert (output / "COMPLETE").is_file() and not (output / "INCOMPLETE").exists()


@pytest.mark.parametrize("profile_id,arguments", [
    ("real-model-single-gpu-v1", []),
    ("real-model-single-gpu-v1", ["--device", "cuda:0"]),
    ("clean-room-release-v1", ["--device", "cuda:0"]),
    ("tiny-reference-cpu-v1", ["--model", "configs/models/v0_1.yaml"]),
])
def test_gate_rejects_incomplete_profile_before_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_id: str, arguments: list[str],
) -> None:
    executed: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: executed.append(command))
    with pytest.raises(SystemExit):
        gate.run_gate(["--profile", "unit", "--runtime-profile", "v01-cpu", "--profile-id", profile_id,
                       "--output", str(tmp_path / "not-created"), *arguments])
    assert not executed and not (tmp_path / "not-created").exists()


def _manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toolchain = {"schema_version": 1, "finalized": True, "environment_id": "synthetic-driver-unit",
                 "python_executable": str(Path(sys.executable).resolve()), "project_root_digest": "a" * 64}
    toolchain["manifest_digest"] = sha256_json(toolchain)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(toolchain), encoding="utf-8")
    monkeypatch.setenv("ILLM_TOOLCHAIN_MANIFEST", str(path))
    monkeypatch.setenv("ILLM_ENVIRONMENT_ID", "synthetic-driver-unit")


def _synthetic_result(request: GenerationRequest, duration: int = 100) -> GenerationResult:
    assert request.input_token_ids is not None
    sampled = request.sampling.mode == "sample"
    return GenerationResult(request.request_id, request.input_token_ids, (3,), "synthetic unit fixture", "MAX_NEW_TOKENS",
                            request.sampling.actual_seed, "synthetic-driver-unit", request.sampling.digest, 1, int(sampled),
                            ((0, 0, 0),) if sampled else (),
                            {"server_full_context_forward_wall_time_ns": duration,
                             "logits_host_export_wall_time_ns": 3, "sampling_wall_time_ns": 5,
                             "total_request_wall_time_ns": duration + 50}, "c" * 64, 8)


def test_benchmark_keeps_all_observations_and_restores_runtime_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest(tmp_path, monkeypatch)
    inspected = {"max_context_tokens": 128, "spec": {"vocab_size": 259}, "weight_manifest": {"weights_digest": "b" * 64}}
    monkeypatch.setattr(benchmark, "inspect_model", lambda model: inspected)
    closed: list[bool] = []
    requests: list[Any] = []

    class FakeEngine:
        device = "cpu"
        model_spec_id = "synthetic-driver-unit"

        def __init__(self, model: str, *, device: str) -> None:
            pass

        def generate(self, request: GenerationRequest) -> GenerationResult:
            requests.append(request)
            # A large outlier remains in the report; MAD is recomputed without
            # filtering or changing this predeclared stability rule.
            duration = 1_000_000 if request.request_id.endswith("-6") else 100
            return _synthetic_result(request, duration)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(benchmark, "Engine", FakeEngine)
    before = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    report = benchmark.run_benchmark("fixture:tiny-decoder", device="cpu", output=tmp_path / "benchmark")
    assert report["status"] == "pass" and report["speedup_claim"] is None
    assert len(requests) == 3 * (3 + 3 + 7) and closed == [True]
    assert report["measurement_contract"]["cpu_peak_memory_measured"] is False
    for bucket in report["context_buckets"]:
        assert len(bucket["observations"]) == 7
        assert bucket["median_forward_wall_time_ns"] == 100.0
        assert bucket["min_forward_wall_time_ns"] == 100
        assert bucket["max_forward_wall_time_ns"] == 1_000_000
        assert bucket["relative_median_absolute_deviation"] == 0.0
        assert all(observation["peak_device_allocated_bytes"] is None for observation in bucket["observations"])
        assert all(observation["logits_host_export_wall_time_ns"] == 3 and observation["sampling_wall_time_ns"] == 5
                   for observation in bucket["observations"])
    invariance = report["telemetry_invariance"]
    assert invariance["status"] == "pass" and invariance["modes"] == ["off", "request", "ops"]
    assert [request.telemetry for request in requests[:9]] == ["off", "request", "ops"] * 3
    assert all(request.request_id.startswith("telemetry-") for request in requests[:9])
    for bucket in invariance["context_buckets"]:
        assert [outcome["telemetry"] for outcome in bucket["outcomes"]] == ["off", "request", "ops"]
        assert len({outcome["generation_result_digest"] for outcome in bucket["outcomes"]}) == 1
        result = bucket["outcomes"][0]["generation_result"]
        assert result["rng_draws_consumed"] == 1 and result["committed_draw_addresses"] == [[0, 0, 0]]
    assert before == (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    assert all(request.max_new_tokens == 1 for request in requests)
    assert not (tmp_path / "benchmark" / "INCOMPLETE").exists()


@pytest.mark.parametrize("field,value", [
    ("input_token_ids", (5,)), ("output_token_ids", (4,)), ("finish_reason", "EOS"), ("actual_seed", 8),
    ("sampling_spec_digest", "d" * 64), ("next_output_position", 2), ("rng_draws_consumed", 2),
    ("committed_draw_addresses", ((0, 0, 1),)),
])
def test_telemetry_mismatch_stops_before_timing_and_leaves_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark, "inspect_model", lambda model: {
        "max_context_tokens": 128, "spec": {"vocab_size": 259}, "weight_manifest": {"weights_digest": "b" * 64}})
    requests: list[GenerationRequest] = []
    closed: list[bool] = []

    class FaultyEngine:
        device = "cpu"
        model_spec_id = "synthetic-driver-unit"

        def __init__(self, model: str, *, device: str) -> None:
            pass

        def generate(self, request: GenerationRequest) -> GenerationResult:
            requests.append(request)
            result = _synthetic_result(request)
            return replace(result, **{field: value}) if request.telemetry == "ops" else result

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(benchmark, "Engine", FaultyEngine)
    output = tmp_path / "failed-baseline"
    with pytest.raises(IllmError) as raised:
        benchmark.run_benchmark("fixture:tiny-decoder", device="cpu", output=output)
    assert raised.value.code is ErrorCode.NUMERICAL_ERROR
    assert len(requests) == 3 and closed == [True]
    assert all(request.request_id == "telemetry-8" for request in requests)
    assert (output / "INCOMPLETE").exists() and not (output / "COMPLETE").exists()
    assert not (output / "benchmark-report.json").exists()
    failure = json.loads((output / "telemetry-invariance-failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "fail" and failure["performance_outcome"] == "not_evaluated"
    assert failure["speedup_claim"] is None and failure["bottleneck_claim"] is None
    assert failure["telemetry_invariance"]["status"] == "fail"


def test_benchmark_rejects_output_file_before_model_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark, "inspect_model", lambda model: {"max_context_tokens": 128})
    path = tmp_path / "existing"
    path.write_bytes(b"preserve")
    with pytest.raises(IllmError) as raised:
        benchmark.run_benchmark("fixture:tiny-decoder", device="cpu", output=path)
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT and path.read_bytes() == b"preserve"

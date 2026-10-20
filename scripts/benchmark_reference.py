"""Measure the full-context correctness baseline, never a speedup claim.

Execute through in_env.py after source freeze. Every measured repetition has a
metadata-only capture, and CPU/GPU memory evidence remains explicitly distinct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.engine import Engine, GenerationRequest
from illm.model import inspect_model
from illm.sampling import SamplingSpec


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def run_benchmark(model: str, *, device: str, output: Path, contexts: tuple[int, ...] = (8, 32, 128),
                  warmups: int = 3, repetitions: int = 7, cpu_threads: int = 1) -> dict[str, Any]:
    import torch

    if len(contexts) < 3 or len(set(contexts)) != len(contexts) or any(type(n) is not int or n < 1 for n in contexts):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "baseline requires at least three distinct positive context lengths")
    if type(warmups) is not int or warmups < 1 or type(repetitions) is not int or repetitions < 3:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "baseline requires warmups>=1 and repetitions>=3")
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "cpu_threads must be a positive integer")
    manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not manifest_name:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "run the baseline through the finalized manifest runner")
    manifest_path = Path(manifest_name)
    toolchain = json.loads(manifest_path.read_text(encoding="utf-8"))
    if toolchain.get("schema_version") != 1 or toolchain.get("finalized") is not True:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "baseline requires a finalized selected runtime profile")
    if toolchain.get("manifest_digest") != sha256_json({key: value for key, value in toolchain.items() if key != "manifest_digest"}):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "baseline toolchain manifest checksum mismatch")
    if os.environ.get("ILLM_ENVIRONMENT_ID") != toolchain.get("environment_id") or Path(sys.executable).resolve() != Path(toolchain["python_executable"]).resolve():
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "baseline interpreter differs from the finalized environment")
    inspected = inspect_model(model)
    if max(contexts) > inspected["max_context_tokens"]:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "baseline context exceeds the fixed model profile limit")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "baseline output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "INCOMPLETE", {"reason": "benchmark_in_progress"})
    original_threads = torch.get_num_threads()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine: Engine | None = None
    try:
        engine = Engine(model, device=device)
        gpu = engine.device.startswith("cuda")
        hardware: dict[str, Any] = {"device": engine.device, "cpu_threads": cpu_threads, "torch_version": str(torch.__version__)}
        if gpu:
            properties = torch.cuda.get_device_properties(engine.device)
            hardware.update(name=properties.name, total_device_memory_bytes=properties.total_memory,
                            compute_capability=[properties.major, properties.minor], cuda_runtime=torch.version.cuda)
        buckets: list[dict[str, Any]] = []
        for length in contexts:
            # Public synthetic fixture, no user's prompt. One committed output
            # guarantees equal forward count even if the chosen token is EOS.
            ids = tuple(3 + (index % (inspected["spec"]["vocab_size"] - 3)) for index in range(length))
            fixture_id = sha256_json({"schema_version": 1, "kind": "ascending-vocab-v1", "input_ids": ids})
            for index in range(warmups):
                engine.generate(GenerationRequest(model=model, input_token_ids=ids, max_new_tokens=1,
                                                   sampling=SamplingSpec(actual_seed=7), telemetry="off",
                                                   request_id=f"warmup-{length}-{index}"))
            observations: list[dict[str, Any]] = []
            for index in range(repetitions):
                if gpu:
                    torch.cuda.synchronize(engine.device)
                    torch.cuda.reset_peak_memory_stats(engine.device)
                capture = output / "captures" / f"context-{length}" / f"repeat-{index}"
                result = engine.generate(GenerationRequest(model=model, input_token_ids=ids, max_new_tokens=1,
                                                           sampling=SamplingSpec(actual_seed=7), telemetry="request",
                                                           request_id=f"measured-{length}-{index}", capture=capture))
                if gpu:
                    torch.cuda.synchronize(engine.device)
                observations.append({
                    "repetition": index, "context_tokens": length, "output_tokens": len(result.output_token_ids),
                    "server_full_context_forward_wall_time_ns": result.timings["server_full_context_forward_wall_time_ns"],
                    "total_request_wall_time_ns": result.timings["total_request_wall_time_ns"],
                    "peak_device_allocated_bytes": torch.cuda.max_memory_allocated(engine.device) if gpu else None,
                    "peak_device_reserved_bytes": torch.cuda.max_memory_reserved(engine.device) if gpu else None,
                    "capture": capture.relative_to(output).as_posix(), "capture_manifest_digest": result.capture_manifest_digest,
                })
            values = [int(item["server_full_context_forward_wall_time_ns"]) for item in observations]
            median = float(statistics.median(values))
            relative_mad = float(statistics.median([abs(value - median) for value in values])) / median if median > 0 else math.inf
            buckets.append({"context_tokens": length, "public_input_fixture_id": fixture_id, "observations": observations,
                            "median_forward_wall_time_ns": median, "min_forward_wall_time_ns": min(values),
                            "max_forward_wall_time_ns": max(values), "relative_median_absolute_deviation": relative_mad,
                            "stability": "pass" if math.isfinite(relative_mad) and relative_mad <= 0.20 else "unstable"})
        report: dict[str, Any] = {
            "schema_version": 1, "status": "pass" if all(bucket["stability"] == "pass" for bucket in buckets) else "unstable",
            "experiment_id": "reference-full-context-v1", "performance_outcome": "baseline_only",
            "evidence_kind": "measured", "speedup_claim": None, "hardware": hardware,
            "identity": {"model_spec_id": engine.model_spec_id, "weights_digest": inspected["weight_manifest"]["weights_digest"],
                         "toolchain_manifest_digest": toolchain["manifest_digest"], "project_root_digest": toolchain["project_root_digest"]},
            "measurement_contract": {"warmups_per_bucket": warmups, "repetitions_per_bucket": repetitions,
                                     "output_tokens_per_repetition": 1, "clock": "perf_counter_ns", "tf32_enabled": False,
                                     "telemetry": "request", "synchronization": "cuda_device_boundary" if gpu else "synchronous_cpu",
                                     "stability_rule": "relative_median_absolute_deviation<=0.20; no observations discarded",
                                     "cpu_peak_memory_measured": False, "load_time_excluded": True},
            "context_buckets": buckets,
        }
        report["report_digest"] = sha256_json(report)
        _atomic_json(output / "benchmark-report.json", report)
        files = {path.relative_to(output).as_posix(): sha256_file(path)
                 for path in output.rglob("*") if path.is_file() and path.name != "INCOMPLETE"}
        _atomic_json(output / "checksums.json", files)
        (output / "COMPLETE").write_text("complete\n", encoding="ascii")
        (output / "INCOMPLETE").unlink()
        return report
    finally:
        try:
            if engine is not None:
                engine.close()
        finally:
            torch.set_num_threads(original_threads)
            torch.backends.cuda.matmul.allow_tf32 = original_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.model, device=args.device, output=args.output, contexts=tuple(args.contexts),
                               warmups=args.warmups, repetitions=args.repetitions, cpu_threads=args.cpu_threads)
        print(json.dumps({"status": report["status"], "report_digest": report["report_digest"], "output": str(args.output)}, sort_keys=True))
        return 0 if report["status"] == "pass" else 1
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

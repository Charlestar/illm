"""Isolated allocator-budget CUDA OOM probe; never exhaust the whole device.

Invoke this script as its own process through the finalized runtime runner.
The process loads weights normally, then reduces only its PyTorch allocator
allowance. An actual torch.OutOfMemoryError must survive the Engine adapter's
cause chain, and the accepted request must leave one complete FAILED capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureBundle, replay_reference_events
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEventKind as K
from illm.engine import Engine, GenerationRequest


def validate_arguments(output: Path, device: str, fraction: float) -> None:
    if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "OOM probe requires an explicit cuda:<index> device")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1e-4:
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "OOM probe fraction must be finite and in (0, 0.0001]")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "OOM probe output must be a new or empty directory")


def _exception_chain(exc: BaseException, oom_class: type[BaseException]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(records) < 8:
        seen.add(id(current))
        message = str(current)
        records.append({"exception_type": f"{type(current).__module__}.{type(current).__qualname__}",
                        "is_torch_oom": isinstance(current, oom_class),
                        "message_sha256": hashlib.sha256(message.encode("utf-8", errors="replace")).hexdigest(),
                        "message_excerpt": message[:512]})
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return records


def probe_gpu_oom(model: str | Path, output: Path, *, device: str = "cuda:0",
                  fraction: float = 1e-6, project_root: Path | None = None) -> dict[str, Any]:
    validate_arguments(output, device, fraction)
    import torch

    if not torch.cuda.is_available():
        raise IllmError(ErrorCode.UNSUPPORTED_MODEL, "OOM probe requires actual CUDA; no CPU/mock qualification")
    index = int(device.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise IllmError(ErrorCode.INVALID_ARGUMENT, "OOM probe CUDA index does not exist")
    output.mkdir(parents=True, exist_ok=True)
    properties = torch.cuda.get_device_properties(index)
    report: dict[str, Any] = {
        "schema_version": 1, "profile_id": "real-cuda-allocator-oom-v1", "status": "fail",
        "evidence_kind": "measured", "isolation": "separate-process-pytorch-allocator-budget",
        "process_id": os.getpid(), "allocation_budget_scope": "this_process_only",
        "injection": "allocator_budget_only_no_mock_no_tensor_corruption", "retry_count": 0,
        "device": device, "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_device_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
        "model": str(model), "allocator_budget_fraction": fraction,
        "allocator_budget_bytes": int(properties.total_memory * fraction),
        "allocator_restore": {"status": "not_attempted"}, "engine_close": {"status": "not_attempted"},
        "exception_chain": [], "capture": None, "probe_error": None,
    }
    engine: Engine | None = None
    original_fraction: float | None = None
    changed_budget = False
    caught_code: str | None = None
    genuine_oom = False
    request_started = False
    before_oom_count = 0
    try:
        engine = Engine(model, device=device, project_root=project_root)
        loaded = engine._loaded  # Probe-only inspection; no mutation of model weights.
        report["identity"] = {"model_spec_id": loaded.spec.spec_id,
                              "weights_digest": loaded.weight_manifest["weights_digest"],
                              "load_manifest_digest": loaded.weight_manifest["manifest_digest"],
                              "runtime_dtype": loaded.spec.runtime_dtype,
                              "tokenizer_revision": loaded.tokenizer.revision}
        torch.cuda.synchronize(index)
        original_fraction = torch.cuda.get_per_process_memory_fraction(index)
        torch.cuda.empty_cache()
        report["allocator_before"] = {"fraction": original_fraction,
                                      "allocated_bytes": torch.cuda.memory_allocated(index),
                                      "reserved_bytes": torch.cuda.memory_reserved(index)}
        before_oom_count = int(torch.cuda.memory_stats(index).get("num_ooms", 0))
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        changed_budget = True
        report["allocator_applied_fraction"] = torch.cuda.get_per_process_memory_fraction(index)
        context = min(256, loaded.max_context_tokens)
        report["context_tokens"] = context
        report["max_new_tokens"] = 1
        request = GenerationRequest(model=model, input_token_ids=tuple((position * 17 + 3) % loaded.spec.vocab_size
                                                                       for position in range(context)),
                                    max_new_tokens=1, request_id="actual-cuda-allocator-oom-v1",
                                    capture=output / "capture", telemetry="ops")
        request_started = True
        try:
            engine.generate(request)
        except IllmError as exc:
            caught_code = exc.code.value
            report["exception_chain"] = _exception_chain(exc, torch.OutOfMemoryError)
            genuine_oom = any(item["is_torch_oom"] for item in report["exception_chain"])
        report["allocator_after_request"] = {
            "allocated_bytes": torch.cuda.memory_allocated(index),
            "reserved_bytes": torch.cuda.memory_reserved(index),
            "oom_count_delta": int(torch.cuda.memory_stats(index).get("num_ooms", 0)) - before_oom_count,
        }
    except Exception as exc:
        report["probe_error"] = {"exception_type": type(exc).__name__,
                                 "error_code": exc.code.value if isinstance(exc, IllmError) else None,
                                 "message_sha256": hashlib.sha256(str(exc).encode("utf-8", errors="replace")).hexdigest()}
    finally:
        if changed_budget and original_fraction is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(original_fraction, index)
                restored = torch.cuda.get_per_process_memory_fraction(index)
                report["allocator_restore"] = {"status": "pass" if restored == original_fraction else "fail",
                                               "original_fraction": original_fraction, "restored_fraction": restored}
            except Exception as exc:
                report["allocator_restore"] = {"status": "fail", "exception_type": type(exc).__name__}
        if engine is not None:
            try:
                closed = engine.close(timeout=1.0)
                report["engine_close"] = {"status": "pass" if closed else "fail"}
            except Exception as exc:
                report["engine_close"] = {"status": "fail", "exception_type": type(exc).__name__}
    capture_pass = False
    if request_started and (output / "capture").is_dir():
        try:
            bundle = ReferenceCaptureBundle.open(output / "capture")
            accepted = sum(event.kind is K.REQUEST_ACCEPTED for event in bundle.events)
            failed = [event for event in bundle.events if event.kind is K.REQUEST_FAILED]
            finished = sum(event.kind is K.REQUEST_FINISHED for event in bundle.events)
            replayed = replay_reference_events(bundle.events, bundle.initial_state)
            capture_pass = (accepted == 1 and len(failed) == 1 and finished == 0
                            and failed[0].payload["error_code"] == ErrorCode.RESOURCE_EXHAUSTED.value
                            and bundle.final_state["status"] == "FAILED" and replayed == bundle.final_state)
            report["capture"] = {"path": "capture", "manifest_digest": bundle.manifest["manifest_digest"],
                                 "manifest_sha256": sha256_file(output / "capture" / "capture-manifest.json"),
                                 "event_count": len(bundle.events), "accepted_count": accepted,
                                 "failed_count": len(failed), "finished_count": finished,
                                 "run_status": bundle.final_state["status"],
                                 "r1_status": "pass" if replayed == bundle.final_state else "fail",
                                 "final_state_digest": sha256_json(bundle.final_state),
                                 "privacy_mode": bundle.manifest["privacy_mode"],
                                 "critical_event_drop_count": bundle.manifest["critical_event_drop_count"]}
        except Exception as exc:
            report["capture_error"] = {"exception_type": type(exc).__name__}
    report["caught_error_code"] = caught_code
    report["genuine_torch_oom"] = genuine_oom
    counter_incremented = report.get("allocator_after_request", {}).get("oom_count_delta", 0) > 0
    report["status"] = "pass" if (caught_code == ErrorCode.RESOURCE_EXHAUSTED.value and genuine_oom and counter_incremented
                                  and capture_pass and report["allocator_restore"]["status"] == "pass"
                                  and report["engine_close"]["status"] == "pass" and report["probe_error"] is None) else "fail"
    report["report_digest"] = sha256_json(report)
    temporary = output / "oom-report.json.tmp"
    temporary.write_bytes(canonical_json_bytes(report) + b"\n")
    temporary.replace(output / "oom-report.json")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-fraction", type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        report = probe_gpu_oom(args.model, args.output, device=args.device, fraction=args.budget_fraction)
    except IllmError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True))
        return exc.exit_code
    print(json.dumps({"status": report["status"], "report_digest": report["report_digest"],
                      "report": str(args.output / "oom-report.json")}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Actual CPU B/C/cancel captures and derived views; development evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from illm.canonical import canonical_json_bytes
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest
from illm.numerical_policy import candidate_policy
from illm.sampling import SamplingSpec
from illm.streaming_learning import derive_streaming_learning_assets, verify_streaming_learning_assets


def run_case(root: Path, name: str, mode: str, *, cancel: bool = False) -> dict[str, Any]:
    policy = candidate_policy("fp32")
    engine = Engine(device="cpu", kv_mode=mode, numerical_policy=policy)
    capture, views = root / name / "capture", root / name / "views"
    try:
        request = GenerationRequest(model_spec_id=engine.model_spec_id, request_id=name,
                                    prompt="hello", max_new_tokens=4, capture=capture,
                                    sampling=SamplingSpec(mode="greedy", actual_seed=7))
        stream = engine.stream(request)
        try:
            if cancel:
                next(stream)  # metadata
                next(stream)  # first committed output
                engine.cancel(name, "learning-cancel")
            list(stream)
        finally:
            stream.close()
        result = engine.last_result
        if result is None or any(result.resource_status.values()) or result.capture_error is not None:
            raise RuntimeError("capture example did not prove completion and safe cleanup")
        expected = "CANCELLED" if cancel else "FINISHED"
        bundle = PolicyStreamingCaptureBundle.open(capture)
        if result.outcome != expected or bundle.manifest["terminal_outcome"] != expected:
            raise RuntimeError("capture terminal facts differ from the actual request")
        views.mkdir(parents=True, exist_ok=False)
        for filename, data in derive_streaming_learning_assets(capture).items():
            with (views / filename).open("xb") as output:
                output.write(data)
        verified_views = verify_streaming_learning_assets(capture, views)
        return {"case": name, "kv_mode": mode, "outcome": result.outcome,
                "query_tokens_per_step": list(result.query_tokens_per_step),
                "output_token_ids": list(result.output_token_ids),
                "resources_zero": True, "numerical_policy_digest": policy.digest,
                "capture_manifest_digest": result.capture_manifest_digest,
                "capture": capture.relative_to(root).as_posix(),
                "views": views.relative_to(root).as_posix(), "derived_verification": verified_views}
    finally:
        if not engine.close(timeout=2.0):
            raise RuntimeError("capture example engine cleanup failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    requested = args.output.absolute()
    for path in (requested, *requested.parents):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if path.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("example output cannot follow a link or reparse point")
    root = requested.resolve()
    root.mkdir(parents=True, exist_ok=False)
    marker = root / "INCOMPLETE"
    marker.write_text("policy capture example not finalized\n", encoding="ascii")
    cases = [run_case(root, "full-recompute", "none"), run_case(root, "cached", "contiguous"),
             run_case(root, "cancelled", "contiguous", cancel=True)]
    if cases[0]["output_token_ids"] != cases[1]["output_token_ids"]:
        raise RuntimeError("B/C complete outputs differ")
    if cases[0]["query_tokens_per_step"] != [6, 7, 8, 9] or cases[1]["query_tokens_per_step"] != [6, 1, 1, 1]:
        raise RuntimeError("B/C logical work does not match the requested KV mode")
    report = {"schema_version": 1, "kind": "illm-policy-capture-cpu-demo-v1", "status": "pass",
              "evidence_mode": "DEVELOPMENT_ONLY", "qualification_tier": "unqualified",
              "gpu_result": "not_run", "performance_outcome": "not_evaluated", "cases": cases,
              "limitations": ["R0 only", "not full raw-tensor qualification", "GPU deferred",
                              "final source-matched profiles/release/rollback remain required"]}
    try:
        with (root / "demo-report.json").open("xb") as output:
            output.write(canonical_json_bytes(report) + b"\n")
    except FileExistsError as exc:
        raise RuntimeError("capture example refuses to overwrite an existing report") from exc
    marker.unlink()
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

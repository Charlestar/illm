"""Development-only first-op diagnosis of full-context versus cached decode.

This executes the real model. It never grants qualification or changes the
predeclared tolerance. Traces compare the last query row; the separate cached
prefix comparison detects shape-dependent changes in already computed K/V.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

from illm.canonical import canonical_json_bytes, sha256_json
from illm.executor.fence import record_completion_fence
from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
from illm.model import load_model
from illm.model.incremental import IncrementalDecoder
from illm.model.reference import OpRecord, ReferenceDecoder, canonical_op_names
from illm.model.validation import _tolerances


def _hook(values: dict[str, Tensor]) -> Callable[[OpRecord, Tensor], None]:
    def capture(record: OpRecord, value: Tensor) -> None:
        values[record.name] = value.detach().float().cpu().clone()
    return capture


def _last_query(value: Tensor) -> Tensor:
    return value[:, :, -1:, :] if value.ndim == 4 else value[:, -1:, :]


@contextmanager
def _cached_contractions(policy: str) -> Iterator[None]:
    """Diagnostic instrumentation only; the original oracle is outside it."""
    if policy == "original":
        yield
        return
    native_linear = F.linear
    native_matmul = Tensor.__matmul__

    def linear(inputs: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        if inputs.dtype in {torch.bfloat16, torch.float16}:
            return native_linear(inputs.float(), weight.float(), None if bias is None else bias.float()).to(inputs.dtype)
        return native_linear(inputs, weight, bias)

    def matmul(left: Tensor, right: Tensor) -> Tensor:
        if left.dtype in {torch.bfloat16, torch.float16} and right.dtype == left.dtype:
            return torch.matmul(left.float(), right.float()).to(left.dtype)
        return native_matmul(left, right)

    attention = patch.object(Tensor, "__matmul__", matmul) if policy == "fp32-contractions" else nullcontext()
    with patch.object(F, "linear", linear), attention:
        yield


def _diagnose(model: str, device: str, prompt: str, output: Path, *, fp32_shadow: bool = False,
              cached_policy: str = "original") -> dict[str, Any]:
    """Persist measured per-op differences without modifying execution math."""
    if output.exists():
        raise ValueError("diagnostic output must be a new path")
    root = Path(__file__).resolve().parents[1]
    loaded = load_model(model, device=device, project_root=root)
    source_spec_id, source_dtype = loaded.spec.spec_id, loaded.spec.runtime_dtype
    if fp32_shadow:
        shadow_spec = replace(loaded.spec, runtime_dtype="fp32")
        shadow = ReferenceDecoder(shadow_spec, {
            name: loaded.reference.canonical_weight(name) for name in shadow_spec.tensor_shapes()
        }, device=device)
        loaded = replace(loaded, spec=shadow_spec, reference=shadow)
    ids = loaded.tokenizer.encode(prompt)
    embedding = loaded.reference.canonical_weight("model.embed_tokens.weight")
    initial = torch.tensor([ids], dtype=torch.int64, device=embedding.device)
    layout = KVLayoutSpec.for_model(loaded.spec, device=str(embedding.device), capacity=loaded.max_context_tokens)
    store = ContiguousKVStore(layout)
    cached = IncrementalDecoder(loaded.reference, layout)
    prefix_trace: dict[str, Tensor] = {}
    reference_trace: dict[str, Tensor] = {}
    cached_trace: dict[str, Tensor] = {}
    with torch.inference_mode():
        lease = store.begin_append(ids)
        with _cached_contractions(cached_policy):
            first = cached.forward(initial, lease, op_hook=_hook(prefix_trace))
        fence = record_completion_fence(embedding.device)
        fence.wait()
        lease.commit(fence)
        token = int(first[0, -1].float().argmax().item())
        full = torch.tensor([[*ids, token]], dtype=torch.int64, device=embedding.device)
        loaded.reference.forward(full, op_hook=_hook(reference_trace))
        lease = store.begin_append((token,))
        with _cached_contractions(cached_policy):
            cached.forward(full[:, -1:], lease, op_hook=_hook(cached_trace))
        fence = record_completion_fence(embedding.device)
        fence.wait()
        lease.commit(fence)
    tolerance, probability_tolerance = _tolerances(loaded.spec.runtime_dtype)
    records: list[dict[str, Any]] = []
    prefix_records: list[dict[str, Any]] = []

    def compare(name: str, expected: Tensor, actual: Tensor) -> dict[str, Any]:
        active = probability_tolerance if name.endswith("attention_probabilities") else tolerance
        difference = (expected - actual).abs()
        legal_mask = torch.isneginf(expected) & torch.isneginf(actual)
        finite = torch.isfinite(expected) & torch.isfinite(actual)
        matches = legal_mask | (finite & (difference <= active.atol + active.rtol * expected.abs()))
        finite_difference = difference[finite]
        return {
            "op": name, "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
            "mismatch_count": int((~matches).sum().item()),
            "exact_difference_count": int((expected != actual).sum().item()),
            "max_absolute_error": float(finite_difference.max().item()) if finite_difference.numel() else None,
            "tolerance": asdict(active), "evidence_kind": "measured-development",
        }

    for name in canonical_op_names(loaded.spec):
        records.append(compare(name, _last_query(reference_trace[name]), _last_query(cached_trace[name])))
        if name.endswith(("k_projection", "v_projection", "rope_k")):
            prefix_records.append(compare(name, reference_trace[name][:, :, :len(ids), :], prefix_trace[name]))
    # Keep this a diagnostic only: probe identical query data under different
    # GEMM row counts without altering either of the actual inference paths.
    linear_probes: list[dict[str, Any]] = []
    with torch.inference_mode():
        norm = reference_trace["layer.0.input_norm"][:, -1:, :].to(device=embedding.device, dtype=embedding.dtype)
        weight = loaded.reference.canonical_weight("model.layers.0.self_attn.q_proj.weight")
        bias = loaded.reference.canonical_weight("model.layers.0.self_attn.q_proj.bias")
        expected = reference_trace["layer.0.q_projection"][:, :, -1:, :].transpose(1, 2).reshape(1, 1, -1)
        for rows in (1, 2, 4, 8, 16, 32):
            padded = torch.zeros((1, rows, norm.shape[-1]), device=norm.device, dtype=norm.dtype)
            padded[:, :1, :] = norm
            measured = F.linear(padded, weight, bias)[:, :1, :].float().cpu()
            linear_probes.append({"padded_rows": rows, **compare("layer.0.q_projection", expected, measured)})
        accurate = F.linear(norm.float(), weight.float(), bias.float()).to(norm.dtype).float().cpu()
        linear_probes.append({"method": "fp32-multiply-accumulate-bias-then-cast", **compare("layer.0.q_projection", expected, accurate)})
        separate_bias = (F.linear(norm, weight).float() + bias.float()).to(norm.dtype).float().cpu()
        linear_probes.append({"method": "bf16-dot-then-fp32-bias-then-cast", **compare("layer.0.q_projection", expected, separate_bias)})
    store.close()
    report: dict[str, Any] = {
        "schema_version": 1, "qualification_tier": "unqualified", "formal_evidence_eligible": False,
        "model_spec_id": loaded.spec.spec_id, "dtype": loaded.spec.runtime_dtype, "device": str(embedding.device),
        "source_model_spec_id": source_spec_id, "source_runtime_dtype": source_dtype,
        "source_weights_digest": loaded.weight_manifest["weights_digest"],
        "diagnostic_mode": "fp32-shadow-not-a-low-precision-pass" if fp32_shadow else "declared-runtime",
        "cached_contraction_policy": cached_policy,
        "oracle_policy": "fp32-shadow-not-v01-runtime" if fp32_shadow else "v01-full-context-unchanged-v1",
        "input_token_ids": list(ids), "prefill_selected_token": token,
        "torch_version": str(torch.__version__),
        "cuda_matmul_reduced_precision": {
            "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        },
        "process_numerical_controls": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "disable_addmm_cuda_lt": os.environ.get("DISABLE_ADDMM_CUDA_LT"),
        },
        "last_query_ops": records, "cached_prefix_ops": prefix_records,
        "linear_padding_probes": linear_probes,
        "first_last_query_divergence": next((item for item in records if item["mismatch_count"]), None),
        "first_cached_prefix_divergence": next((item for item in prefix_records if item["mismatch_count"]), None),
    }
    report["report_digest"] = sha256_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def diagnose(model: str, device: str, prompt: str, output: Path, *, full_precision_reduction: bool = False,
             fp32_shadow: bool = False, cached_policy: str = "original", deterministic_math: bool = False) -> dict[str, Any]:
    """Run an isolated diagnostic policy, restoring caller precision settings."""
    if cached_policy not in {"original", "fp32-linear", "fp32-contractions"} or (fp32_shadow and cached_policy != "original"):
        raise ValueError("cached policy must be a declared independent experiment, not combined with a shadow oracle")
    previous = (
        torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if full_precision_reduction:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        if deterministic_math:
            torch.use_deterministic_algorithms(True)
        return _diagnose(model, device, prompt, output, fp32_shadow=fp32_shadow, cached_policy=cached_policy)
    finally:
        torch.set_num_threads(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous[3]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous[4]
        torch.use_deterministic_algorithms(previous[5], warn_only=previous[6])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-precision-reduction", action="store_true")
    parser.add_argument("--fp32-shadow", action="store_true", help="separate FP32 diagnostic, never a BF16/FP16 pass")
    parser.add_argument("--cached-policy", choices=("original", "fp32-linear", "fp32-contractions"), default="original")
    parser.add_argument("--deterministic-math", action="store_true", help="diagnose the explicit process-level deterministic policy")
    args = parser.parse_args(argv)
    report = diagnose(args.model, args.device, args.prompt, args.output,
                      full_precision_reduction=args.full_precision_reduction, fp32_shadow=args.fp32_shadow,
                      cached_policy=args.cached_policy, deterministic_math=args.deterministic_math)
    print(json.dumps({key: report[key] for key in (
        "report_digest", "first_last_query_divergence", "first_cached_prefix_divergence",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Testing-only oracle adapters and deterministic first-op divergence reports."""

from __future__ import annotations

import json
import hashlib
import importlib
import math
from pathlib import Path
from dataclasses import asdict
from typing import Any, Callable, Mapping, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from illm.model.loader import LoadedModel
from illm.model.reference import OpRecord, ReferenceDecoder, canonical_op_names
from illm.model.spec import ModelError
from illm.numerical_contracts import FIXED_PROMPTS as FIXED_PROMPTS
from illm.numerical_contracts import TensorTolerance as TensorTolerance
from illm.numerical_contracts import tensor_tolerances
from illm.model.spec import canonical_bytes, digest


class _OracleConfig(Protocol):
    _attn_implementation: str
    use_cache: bool


class _OracleOutput(Protocol):
    @property
    def logits(self) -> Tensor: ...


class TransformersOracle(Protocol):
    """Narrow boundary for pinned Transformers' untyped model implementation."""

    model: Any
    lm_head: Callable[[Tensor], Tensor]

    def load_state_dict(self, state_dict: Mapping[str, Tensor], *, strict: bool) -> object: ...

    def to(self, *, device: torch.device) -> TransformersOracle: ...

    def eval(self) -> TransformersOracle: ...

    def __call__(self, *, input_ids: Tensor, position_ids: Tensor | None = None,
                 use_cache: bool = False) -> _OracleOutput: ...


def _tolerances(dtype: str) -> tuple[TensorTolerance, TensorTolerance]:
    return tensor_tolerances(dtype)


def collect_reference_ops(reference: ReferenceDecoder, input_ids: Tensor, *, position_ids: Tensor | None = None) -> dict[str, Tensor]:
    trace: dict[str, Tensor] = {}

    def capture(record: OpRecord, value: Tensor) -> None:
        if record.name in trace:
            raise ModelError("INVALID_ARGUMENT", "duplicate canonical op in one forward", field=record.name)
        trace[record.name] = value.detach().clone()

    reference.forward(input_ids, position_ids=position_ids, op_hook=capture)
    return trace


def _causally_masked_negative_infinity(name: str, tensor: Tensor) -> Tensor:
    if name.endswith(".attention_scores") and tensor.ndim == 4 and tensor.shape[-2] == tensor.shape[-1]:
        causal_future = torch.triu(torch.ones_like(tensor, dtype=torch.bool), diagonal=1)
        return causal_future & torch.isneginf(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def compare_op_traces(expected: Mapping[str, Tensor], actual: Mapping[str, Tensor], *,
                      op_order: tuple[str, ...], identity: Mapping[str, Any],
                      tolerance: TensorTolerance = TensorTolerance(), report_link: str | None = None,
                      probability_tolerance: TensorTolerance | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for name in op_order:
        active_tolerance = probability_tolerance if name.endswith("attention_probabilities") and probability_tolerance else tolerance
        record: dict[str, Any] = {"op": name, "status": "pass", "evidence_kind": "measured",
                                  "tolerance": asdict(active_tolerance), "report_link": report_link}
        if name not in expected or name not in actual:
            record.update(status="fail", reason="missing_op", expected_present=name in expected, actual_present=name in actual)
        else:
            left, right = expected[name].detach(), actual[name].detach()
            record.update(expected_shape=list(left.shape), actual_shape=list(right.shape),
                          expected_dtype=str(left.dtype), actual_dtype=str(right.dtype))
            if left.shape != right.shape:
                record.update(status="fail", reason="shape_mismatch")
            elif left.dtype != right.dtype:
                record.update(status="fail", reason="dtype_mismatch")
            else:
                allowed_left = _causally_masked_negative_infinity(name, left)
                allowed_right = _causally_masked_negative_infinity(name, right)
                if (not bool((torch.isfinite(left) | allowed_left).all().item())
                        or not bool((torch.isfinite(right) | allowed_right).all().item())):
                    record.update(status="fail", reason="non_finite")
                elif not bool((allowed_left == allowed_right).all().item()):
                    record.update(status="fail", reason="masked_negative_infinity_mismatch")
                else:
                    # Matching legal causal -Inf entries are the same sentinel,
                    # not a numerical subtraction (-Inf - -Inf would be NaN).
                    record["masked_negative_infinity_count"] = int(allowed_left.sum().item())
                    left = torch.where(allowed_left, torch.zeros_like(left), left)
                    right = torch.where(allowed_right, torch.zeros_like(right), right)
            if record["status"] == "pass" and "expected_shape" in record:
                a, b = left.double().cpu(), right.double().cpu()
                absolute = (a - b).abs()
                relative = absolute / a.abs().clamp_min(1e-30)
                matches = absolute <= active_tolerance.atol + active_tolerance.rtol * a.abs()
                record.update(max_absolute_error=float(absolute.max().item()),
                              max_relative_error=float(relative.max().item()),
                              mismatch_count=int((~matches).sum().item()))
                if not bool(matches.all().item()):
                    first_index = int((~matches).flatten().nonzero()[0].item())
                    record.update(status="fail", reason="outside_tolerance", first_flat_index=first_index,
                                  expected_value=float(a.flatten()[first_index].item()), actual_value=float(b.flatten()[first_index].item()))
        if record["status"] == "fail" and first is None:
            first = dict(record)
        records.append(record)
    unexpected = sorted((set(expected) | set(actual)) - set(op_order))
    if unexpected and first is None:
        first = {"op": unexpected[0], "status": "fail", "reason": "unexpected_op", "report_link": report_link}
    return {"schema_version": 1, "status": "pass" if first is None else "fail", "identity": dict(identity),
            "first_divergence": first, "op_count": len(op_order), "ops": records,
            "unexpected_ops": unexpected, "report_link": report_link, "tolerance_profile": asdict(tolerance)}


def load_transformers_oracle(loaded: LoadedModel) -> TransformersOracle:
    """Construct pinned local Transformers oracle without runtime Hub access."""
    transformers = importlib.import_module("transformers")
    from safetensors.torch import load_file

    if transformers.__version__ != "4.57.1":
        raise ModelError("UNSUPPORTED_MODEL", "oracle contract pins Transformers 4.57.1")
    config_data = json.loads((loaded.assets_directory / "config.json").read_text(encoding="utf-8"))
    config_factory = cast(Callable[..., _OracleConfig], transformers.Qwen2Config)
    model_factory = cast(Callable[..., TransformersOracle], transformers.Qwen2ForCausalLM._from_config)
    config = config_factory(**config_data)
    config._attn_implementation = "eager"
    config.use_cache = False
    weight = loaded.reference.canonical_weight("model.embed_tokens.weight")
    # Preserve process global RNG despite the oracle's temporary random init.
    # Use Transformers' own dtype-aware construction: casting the entire model
    # afterward would wrongly round its explicitly FP32 RoPE inv_freq buffer.
    with torch.random.fork_rng(devices=[]):
        oracle = model_factory(config, dtype=weight.dtype, attn_implementation="eager")
    # The oracle loads original verified storage independently. Reading weights
    # back from our ReferenceDecoder would hide a constructor/copy/mapping bug
    # by giving the same corrupted tensors to both paths.
    tensors: dict[str, Tensor] = {}
    for shard in loaded.weight_manifest["source_shards"]:
        name = shard["shard"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ModelError("INVALID_WEIGHTS", "oracle source shard must be one local basename")
        path = loaded.assets_directory / name
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != shard["sha256"]:
            raise ModelError("INVALID_WEIGHTS", "oracle source shard differs from verified weight identity", field=name)
        values = load_file(str(path), device="cpu")
        if set(tensors) & set(values):
            raise ModelError("DUPLICATE_WEIGHT", "oracle source shards repeat tensor storage")
        tensors.update(values)
    if set(tensors) != set(loaded.spec.tensor_shapes()):
        raise ModelError("INVALID_WEIGHTS", "oracle source tensor coverage differs from model spec")
    if loaded.spec.tie_word_embeddings:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    oracle.load_state_dict(tensors, strict=True)
    oracle = oracle.to(device=weight.device)
    if oracle.model.rotary_emb.inv_freq.dtype != torch.float32:
        raise ModelError("ORACLE_DIVERGED", "oracle RoPE inverse frequencies must remain FP32 under low-precision weights")
    oracle.eval()
    return oracle


def oracle_forward_logits(oracle: TransformersOracle, input_ids: Tensor, *, vocab_size: int,
                          position_ids: Tensor | None = None) -> Tensor:
    """Check the untyped oracle's output before treating it as tensor evidence."""
    output = oracle(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits = getattr(output, "logits", None)
    if (not isinstance(logits, Tensor) or tuple(logits.shape) != (*input_ids.shape, vocab_size)
            or not bool(torch.isfinite(logits).all().item())):
        raise ModelError("ORACLE_DIVERGED", "ordinary Transformers forward returned invalid/nonfinite logits")
    return logits


@torch.inference_mode()
def collect_transformers_ops(oracle: TransformersOracle, input_ids: Tensor, loaded: LoadedModel, *, position_ids: Tensor | None = None) -> dict[str, Tensor]:
    """Use oracle-owned modules/RoPE/GQA, and cross-check its complete forward.

    Exposed canonical intermediates are evaluated from the oracle's modules.
    This adapter then verifies its logits against ordinary Transformers forward,
    so an erroneous trace adapter cannot silently define the numerical oracle.
    """
    modeling_qwen2 = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    apply_rope = cast(Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]], modeling_qwen2.apply_rotary_pos_emb)
    repeat_kv = cast(Callable[[Tensor, int], Tensor], modeling_qwen2.repeat_kv)

    trace: dict[str, Tensor] = {}

    def record(name: str, tensor: Tensor) -> Tensor:
        trace[name] = tensor.detach().clone()
        return tensor

    length = input_ids.shape[1]
    if position_ids is None:
        position_ids = torch.arange(length, device=input_ids.device, dtype=torch.int64).unsqueeze(0)
    hidden = record("embedding", oracle.model.embed_tokens(input_ids))
    cos, sin = oracle.model.rotary_emb(hidden, position_ids)
    mask = torch.triu(torch.full((length, length), torch.finfo(hidden.dtype).min,
                                device=hidden.device, dtype=hidden.dtype), diagonal=1)
    for index, layer in enumerate(oracle.model.layers):
        prefix = f"layer.{index}"
        residual = hidden
        norm = record(f"{prefix}.input_norm", layer.input_layernorm(hidden))
        attn = layer.self_attn
        head_dim = loaded.spec.head_dim
        q = record(f"{prefix}.q_projection", attn.q_proj(norm).view(1, length, loaded.spec.num_attention_heads, head_dim).transpose(1, 2))
        k = record(f"{prefix}.k_projection", attn.k_proj(norm).view(1, length, loaded.spec.num_kv_heads, head_dim).transpose(1, 2))
        v = record(f"{prefix}.v_projection", attn.v_proj(norm).view(1, length, loaded.spec.num_kv_heads, head_dim).transpose(1, 2))
        q, k = apply_rope(q, k, cos, sin)
        record(f"{prefix}.rope_q", q)
        record(f"{prefix}.rope_k", k)
        k = repeat_kv(k, attn.num_key_value_groups)
        v = repeat_kv(v, attn.num_key_value_groups)
        scores = record(f"{prefix}.attention_scores", (q @ k.transpose(-2, -1)) * attn.scaling + mask[None, None, :, :])
        probabilities = record(f"{prefix}.attention_probabilities", F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype))
        context = record(f"{prefix}.attention_context", probabilities @ v)
        attention = record(f"{prefix}.attention_output", attn.o_proj(context.transpose(1, 2).contiguous().view(1, length, -1)))
        hidden = record(f"{prefix}.attention_residual", residual + attention)
        residual = hidden
        norm = record(f"{prefix}.post_attention_norm", layer.post_attention_layernorm(hidden))
        gate = record(f"{prefix}.mlp_gate", layer.mlp.gate_proj(norm))
        up = record(f"{prefix}.mlp_up", layer.mlp.up_proj(norm))
        activated = record(f"{prefix}.mlp_silu", layer.mlp.act_fn(gate))
        product = record(f"{prefix}.mlp_product", activated * up)
        mlp = record(f"{prefix}.mlp_output", layer.mlp.down_proj(product))
        hidden = record(f"{prefix}.mlp_residual", residual + mlp)
    hidden = record("final_norm", oracle.model.norm(hidden))
    recorded_logits = record("logits", oracle.lm_head(hidden))
    regular_logits = oracle_forward_logits(oracle, input_ids, vocab_size=loaded.spec.vocab_size, position_ids=position_ids)
    if not torch.equal(recorded_logits, regular_logits):
        # Same oracle implementation must give identical operations, not merely
        # a loose user-selected reference tolerance.
        raise ModelError("ORACLE_DIVERGED", "oracle trace adapter differs from ordinary Transformers forward")
    expected_names = canonical_op_names(loaded.spec)
    if tuple(trace) != expected_names:
        raise ModelError("ORACLE_DIVERGED", "oracle trace coverage/order differs from canonical op contract")
    return trace



def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(report) + b"\n")
    temporary.replace(path)


def _tensor_checksum(tensor: Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _output_comparison(expected: Mapping[str, Tensor], actual: Mapping[str, Tensor]) -> dict[str, Any]:
    a, b = expected["logits"].float(), actual["logits"].float()
    expected_token = int(a[0, -1].argmax().item())
    actual_token = int(b[0, -1].argmax().item())
    return {"evidence_kind": "measured", "final_logits_changed_count": int((a != b).sum().item()),
            "final_logits_max_absolute_error": float((a - b).abs().max().item()),
            "expected_next_token": expected_token, "actual_next_token": actual_token,
            "token_mismatch": expected_token != actual_token}


@torch.inference_mode()
def validate_model(model: str | Path = "fixture:tiny-decoder", *, device: str = "cpu",
                   project_root: Path | None = None, output_directory: Path | None = None) -> dict[str, Any]:
    """Run 20 fixed greedy cases, per-op oracle and affected-projection fault test."""
    from illm.model.loader import load_model

    if output_directory is not None and output_directory.exists():
        if not output_directory.is_dir() or any(output_directory.iterdir()):
            raise ModelError("INVALID_ARGUMENT", "validation output must be a new or empty directory; existing evidence is immutable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = load_model(model, device=device, project_root=project_root)
    oracle = load_transformers_oracle(loaded)
    tolerance, probability_tolerance = _tolerances(loaded.spec.runtime_dtype)
    target_device = loaded.reference.canonical_weight("model.embed_tokens.weight").device
    base_identity = {"spec_id": loaded.spec.spec_id, "weights_digest": loaded.weight_manifest["weights_digest"],
                     "load_manifest_digest": loaded.weight_manifest["manifest_digest"],
                     "tokenizer_revision": loaded.tokenizer.revision,
                     "fixture_id": loaded.fixture_id, "capture_identity": "public-fixed-model-validation-v1"}
    artifacts: dict[str, dict[str, str]] = {}
    if output_directory:
        for name, value in (
            ("model-spec.json", {**loaded.spec.to_dict(), "spec_id": loaded.spec.spec_id}),
            ("weight-load-manifest.json", loaded.weight_manifest),
            ("tokenizer-spec.json", loaded.tokenizer.to_dict()),
        ):
            _write_report(output_directory / name, value)
            artifacts[name] = {"path": name, "sha256": hashlib.sha256((output_directory / name).read_bytes()).hexdigest()}
    op_reports: list[dict[str, Any]] = []
    token_reports: list[dict[str, Any]] = []
    for index, prompt in enumerate(FIXED_PROMPTS):
        initial_ids = loaded.tokenizer.encode(prompt)
        tensor = torch.tensor([initial_ids], dtype=torch.int64, device=target_device)
        expected = collect_transformers_ops(oracle, tensor, loaded)
        actual = collect_reference_ops(loaded.reference, tensor)
        identity = {**base_identity, "input_identity": loaded.tokenizer.input_identity(initial_ids, add_bos=loaded.fixture_id is not None),
                    "public_input_fixture_id": f"fixed-greedy-{index:02d}"}
        link = f"ops/input-{index:02d}.json"
        report = compare_op_traces(expected, actual, op_order=canonical_op_names(loaded.spec), identity=identity,
                                   tolerance=tolerance, probability_tolerance=probability_tolerance, report_link=link)
        op_reports.append(report)
        if output_directory:
            _write_report(output_directory / link, report)
        full = list(initial_ids)
        reference_tokens: list[int] = []
        oracle_tokens: list[int] = []
        for position in range(8):
            inputs = torch.tensor([full], dtype=torch.int64, device=target_device)
            reference_logits = loaded.reference.forward(inputs)
            oracle_logits = oracle_forward_logits(oracle, inputs, vocab_size=loaded.spec.vocab_size)
            reference_token = int(reference_logits[0, -1].float().argmax().item())
            oracle_token = int(oracle_logits[0, -1].float().argmax().item())
            reference_tokens.append(reference_token)
            oracle_tokens.append(oracle_token)
            if reference_token != oracle_token:
                divergence = compare_op_traces(collect_transformers_ops(oracle, inputs, loaded), collect_reference_ops(loaded.reference, inputs),
                                               op_order=canonical_op_names(loaded.spec), identity={**identity, "output_position": position},
                                               tolerance=tolerance, probability_tolerance=probability_tolerance,
                                               report_link=f"ops/token-mismatch-{index:02d}-{position}.json")
                op_reports.append(divergence)
                if output_directory:
                    _write_report(output_directory / f"ops/token-mismatch-{index:02d}-{position}.json", divergence)
                break
            full.append(reference_token)
            if reference_token == loaded.spec.eos_token_id:
                break
        token_reports.append({"public_input_fixture_id": f"fixed-greedy-{index:02d}", "input_identity": identity["input_identity"],
                              "status": "pass" if reference_tokens == oracle_tokens else "fail",
                              "reference_tokens": reference_tokens, "oracle_tokens": oracle_tokens,
                              "generated_count": len(reference_tokens)})
    # The fixed wrong-layout experiment uses tiny weights even during real-model
    # qualification, keeping the diagnostic failure independent of GPU memory.
    tiny = loaded if loaded.fixture_id else load_model("fixture:tiny-decoder", project_root=project_root)
    tiny_oracle = oracle if loaded.fixture_id else load_transformers_oracle(tiny)
    bad_weights = {name: tiny.reference.canonical_weight(name).detach().clone() for name in tiny.spec.tensor_shapes()}
    fault_name = "model.layers.0.self_attn.q_proj.weight"
    source_checksum = _tensor_checksum(bad_weights[fault_name])
    bad_weights[fault_name] = bad_weights[fault_name].T.contiguous()
    mutated_checksum = _tensor_checksum(bad_weights[fault_name])
    bad = ReferenceDecoder(tiny.spec, bad_weights, device=str(tiny.reference.canonical_weight("model.embed_tokens.weight").device))
    wrong_input = torch.tensor([tiny.tokenizer.encode("hello")], dtype=torch.int64,
                               device=tiny.reference.canonical_weight("model.embed_tokens.weight").device)
    expected_fault_ops = collect_transformers_ops(tiny_oracle, wrong_input, tiny)
    actual_fault_ops = collect_reference_ops(bad, wrong_input)
    fault_report = compare_op_traces(expected_fault_ops, actual_fault_ops,
                                    op_order=canonical_op_names(tiny.spec), identity={"spec_id": tiny.spec.spec_id,
                                    "weights_digest": tiny.weight_manifest["weights_digest"],
                                    "load_manifest_digest": tiny.weight_manifest["manifest_digest"],
                                    "weights_identity_role": "unmodified-source-before-explicit-fault",
                                    "fault_fixture_id": "transpose-layer0-q-weight-v1",
                                    "fault_transform": {"transform_id": "transpose-axes-0-1-contiguous-v1",
                                                        "tensor_name": fault_name, "shape": list(bad_weights[fault_name].shape),
                                                        "source_tensor_checksum": source_checksum,
                                                        "mutated_tensor_checksum": mutated_checksum},
                                    "capture_identity": "public-transpose-diagnostic-v1"}, report_link="fault-transposed-q.json")
    fault_report["identity"]["fault_execution_identity"] = digest(fault_report["identity"])
    fault_report["output_comparison"] = _output_comparison(expected_fault_ops, actual_fault_ops)
    diagnostic_pass = fault_report["first_divergence"] is not None and fault_report["first_divergence"]["op"] == "layer.0.q_projection"
    if output_directory:
        _write_report(output_directory / "fault-transposed-q.json", fault_report)
    # A Q transpose can produce a genuine op/logit divergence while leaving the
    # untrained tiny model's greedy token unchanged. Do not mislabel that case.
    # This second, explicitly two-fault fixture creates a real final-token
    # mismatch and checks that diagnosis still finds the earliest failing op.
    norm_name = "model.norm.weight"
    source_norm_checksum = _tensor_checksum(bad_weights[norm_name])
    bad_weights[norm_name] = torch.zeros_like(bad_weights[norm_name])
    token_bad = ReferenceDecoder(tiny.spec, bad_weights, device=str(wrong_input.device))
    token_fault_ops = collect_reference_ops(token_bad, wrong_input)
    token_fault_identity = {
        "spec_id": tiny.spec.spec_id, "weights_digest": tiny.weight_manifest["weights_digest"],
        "load_manifest_digest": tiny.weight_manifest["manifest_digest"],
        "weights_identity_role": "unmodified-source-before-explicit-fault",
        "fault_fixture_id": "q-transpose-plus-zero-final-norm-v1",
        "fault_transforms": [fault_report["identity"]["fault_transform"],
                             {"transform_id": "zero-tensor-v1", "tensor_name": norm_name,
                              "shape": list(bad_weights[norm_name].shape),
                              "source_tensor_checksum": source_norm_checksum,
                              "mutated_tensor_checksum": _tensor_checksum(bad_weights[norm_name])}],
        "capture_identity": "public-two-fault-token-diagnostic-v1",
    }
    token_fault_identity["fault_execution_identity"] = digest(token_fault_identity)
    token_fault_report = compare_op_traces(expected_fault_ops, token_fault_ops, op_order=canonical_op_names(tiny.spec),
                                           identity=token_fault_identity, report_link="fault-q-transpose-and-zero-final-norm.json")
    token_fault_report["output_comparison"] = _output_comparison(expected_fault_ops, token_fault_ops)
    token_diagnostic_pass = (token_fault_report["first_divergence"] is not None
                             and token_fault_report["first_divergence"]["op"] == "layer.0.q_projection"
                             and token_fault_report["output_comparison"]["token_mismatch"] is True)
    if output_directory:
        _write_report(output_directory / "fault-q-transpose-and-zero-final-norm.json", token_fault_report)
    summary: dict[str, Any] = {"schema_version": 1, "status": "pass" if all(item["status"] == "pass" for item in [*op_reports, *token_reports]) and diagnostic_pass and token_diagnostic_pass else "fail",
                               "identity": base_identity, "device": str(target_device), "runtime_dtype": loaded.spec.runtime_dtype,
                               "torch_version": torch.__version__, "transformers_version": "4.57.1", "tf32_enabled": False,
                               "oracle_contract": {"factory": "Qwen2ForCausalLM._from_config(dtype=runtime_dtype)",
                                                   "weight_source": "independent-local-verified-safetensors",
                                                   "attention_implementation": "eager",
                                                   "parameter_dtype": str(oracle.model.embed_tokens.weight.dtype),
                                                   "rope_inv_freq_dtype": str(oracle.model.rotary_emb.inv_freq.dtype),
                                                   "device_transfer": "device-only-no-global-buffer-dtype-cast",
                                                   "trace_matches_ordinary_forward": True},
                               "nonfinite_policy": "only-paired-strict-causal-future-negative-infinity-in-attention-scores-v1",
                               "model_artifacts": artifacts,
                               "tolerance_profile": asdict(tolerance), "attention_probability_tolerance": asdict(probability_tolerance),
                               "fixed_input_count": len(FIXED_PROMPTS), "per_op_reports": op_reports, "greedy_reports": token_reports,
                               "first_divergence_diagnostic": {"status": "pass" if diagnostic_pass else "fail", "report": fault_report,
                                                               "report_link": "fault-transposed-q.json"},
                               "token_mismatch_diagnostic": {"status": "pass" if token_diagnostic_pass else "fail", "report": token_fault_report,
                                                             "report_link": "fault-q-transpose-and-zero-final-norm.json"}}
    summary["report_digest"] = digest(summary)
    if output_directory:
        _write_report(output_directory / "validation-report.json", summary)
    return summary

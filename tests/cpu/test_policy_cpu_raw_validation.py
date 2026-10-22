from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
import illm.policy_cpu_raw_validation as validation
from illm.policy_cpu_numerical_contract import (
    PROTOCOL_SHA256,
    TINY_LOAD_MANIFEST_DIGEST,
    TINY_SPEC_ID,
    TINY_TOKENIZER_DIGEST,
    TINY_WEIGHTS_DIGEST,
    canonical_tiny_ops,
    tensor_id,
)
from illm.policy_tensor_evidence import TensorPack, TensorPackWriter
from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample


def _raw(values: Sequence[float], dtype: str = "fp32") -> bytes:
    code = "f" if dtype == "fp32" else "d"
    return struct.pack(f"<{len(values)}{code}", *values)


def _zeros(shape: Sequence[int], dtype: str = "fp32") -> bytes:
    count = 1
    for dimension in shape:
        count *= dimension
    return _raw([0.0] * count, dtype)


def _flags() -> dict[str, object]:
    return {
        "autocast_cpu": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
        "cpu_threads": 1,
        "cublas_workspace_config": ":16:8",
        "compile": False,
    }


def _resources() -> dict[str, object]:
    return {
        "closed": True,
        "active_leases": 0,
        "allocated_bytes": 0,
        "valid_tokens": 0,
        "staged_tokens": 0,
    }


def _worker_from_records(
    root: Path,
    *,
    directory_name: str,
    role: str,
    records: Sequence[tuple[str, str, Sequence[int], bytes]],
    report: Mapping[str, Any] | None = None,
) -> validation._Worker:
    root.mkdir()
    writer = TensorPackWriter(root / "pack")
    for identifier, dtype, shape, raw in records:
        writer.write_tensor(identifier, dtype, shape, raw)
    manifest = writer.finalize()
    pack = TensorPack.open(root / "pack")
    digest = cast(str, manifest["manifest_digest"])
    return validation._Worker(
        directory_name,
        role,
        root,
        dict(report or {}),
        "1" * 64,
        digest,
        pack,
    )


def _logits(token: int) -> list[float]:
    result = [-10.0] * 259
    result[token] = 10.0
    return result


def _state_record(state: SamplingState) -> dict[str, object]:
    return {"digest": state.digest, "value": state.to_dict()}


def test_candidate_shape_registry_preserves_explicit_query_axis() -> None:
    assert validation._candidate_shape("layer.0.q_projection", 4) == (1, 4, 1, 8)
    assert validation._candidate_shape("layer.0.k_projection", 4) == (1, 2, 1, 8)
    assert validation._candidate_shape("layer.0.rope_q", 4) == (1, 4, 1, 8)
    assert validation._candidate_shape("layer.0.attention_scores", 4) == (1, 4, 1, 5)
    assert validation._native_shape("layer.0.q_projection", 5) == (1, 4, 5, 8)


def _binding_worker(
    *,
    environment_id: str = "8e72088c-61cc-4dd9-846a-c1d43e107a3a",
    manifest_digest: str = "a" * 64,
    lock_digest: str = "b" * 64,
) -> validation._Worker:
    environment: dict[str, object] = {
        "backend": "cpu",
        "python_version": "3.12.14",
        "platform": "Windows-11-10.0.26300-SP0",
        "machine": "AMD64",
        "torch_version": "2.8.0+cpu",
        "cuda_arithmetic": "not_applicable",
        "cuda_execution_authorized": False,
    }
    binding: dict[str, object] = {
        "schema_version": 1,
        "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": "c" * 64,
        "environment": environment,
        "environment_digest": sha256_json(environment),
        "toolchain": {
            "binding": "source-bound-manifest",
            "manifest_digest": manifest_digest,
            "environment_id": environment_id,
            "lock_digest": lock_digest,
        },
        "protocol_sha256": PROTOCOL_SHA256,
    }
    return cast(
        validation._Worker,
        SimpleNamespace(
            role="candidate",
            report={"binding": binding, "binding_digest": sha256_json(binding)},
        ),
    )


def test_execution_binding_accepts_real_builder_uuid4_shape() -> None:
    worker = _binding_worker()
    assert validation._validate_binding(worker) == worker.report["binding"]


@pytest.mark.parametrize(
    "environment_id",
    (
        "8e72088c61cc4dd9846ac1d43e107a3a",
        "8E72088C-61CC-4DD9-846A-C1D43E107A3A",
        "{8e72088c-61cc-4dd9-846a-c1d43e107a3a}",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "0" * 64,
        "not-a-uuid",
    ),
)
def test_execution_binding_rejects_noncanonical_or_non_v4_environment_id(
    environment_id: str,
) -> None:
    with pytest.raises(validation.NumericalEvidenceError, match="canonical UUIDv4"):
        validation._validate_binding(_binding_worker(environment_id=environment_id))


@pytest.mark.parametrize("digest_field", ("manifest_digest", "lock_digest"))
def test_execution_binding_digests_cannot_be_replaced_by_uuid(digest_field: str) -> None:
    uuid_value = "8e72088c-61cc-4dd9-846a-c1d43e107a3a"
    worker = _binding_worker(
        manifest_digest=uuid_value if digest_field == "manifest_digest" else "a" * 64,
        lock_digest=uuid_value if digest_field == "lock_digest" else "b" * 64,
    )
    with pytest.raises(validation.NumericalEvidenceError, match=f"toolchain {digest_field}"):
        validation._validate_binding(worker)


def _model_validation_workers() -> dict[str, validation._Worker]:
    fixture_root = Path(__file__).resolve().parents[2] / "fixtures" / "tiny-decoder"
    spec = cast(dict[str, object], json.loads((fixture_root / "supported-model-spec.json").read_text("utf-8")))
    manifest = cast(dict[str, object], json.loads((fixture_root / "weight-load-manifest.json").read_text("utf-8")))
    tokenizer = cast(dict[str, object], json.loads((fixture_root / "tokenizer.json").read_text("utf-8")))
    authority_files = {
        name: sha256_bytes((fixture_root / name).read_bytes())
        for name in (
            "config.json", "model.safetensors", "supported-model-spec.json",
            "weight-load-manifest.json", "tokenizer.json", "fixture-manifest.json",
        )
    }
    base: dict[str, object] = {
        "model_spec": spec,
        "model_spec_id": TINY_SPEC_ID,
        "weight_manifest": manifest,
        "weight_manifest_digest": TINY_LOAD_MANIFEST_DIGEST,
        "weights_digest": TINY_WEIGHTS_DIGEST,
        "tokenizer": tokenizer,
        "tokenizer_digest": TINY_TOKENIZER_DIGEST,
        "asset_files": authority_files,
    }
    fixture = cast(dict[str, object], json.loads((fixture_root / "fixture-manifest.json").read_text("utf-8")))
    native_oracle: dict[str, object] = {
        "kind": "independent-pinned-transformers-oracle-v1",
        "transformers_version": "4.57.1",
        "model_class": "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM",
        "source": {"module": "transformers.models.qwen2.modeling_qwen2", "sha256": "a" * 64},
        "device": "cpu",
        "parameter_dtype": "torch.float32",
        "rope_inv_freq_dtype": "torch.float32",
        "attention_implementation": "eager",
        "use_cache": False,
        "asset_files": authority_files,
        "fixture_manifest_digest": fixture["manifest_digest"],
        "spec_id": TINY_SPEC_ID,
        "weight_manifest_digest": TINY_LOAD_MANIFEST_DIGEST,
        "weights_digest": TINY_WEIGHTS_DIGEST,
        "source_shards": manifest["source_shards"],
    }
    runtime_models = {
        "native": {**base, "native_oracle": native_oracle},
        "candidate": {**base, "native_oracle": None},
        "reproduction": {**base, "native_oracle": None},
    }
    formula_model = {
        "kind": "independent-binary64-tiny-formula",
        "spec_id": TINY_SPEC_ID,
        "oracle_identity": {
            "fixture_manifest_digest": fixture["manifest_digest"],
            "model_sha256": authority_files["model.safetensors"],
            "spec_id": TINY_SPEC_ID,
            "weight_manifest_digest": TINY_LOAD_MANIFEST_DIGEST,
            "weights_digest": TINY_WEIGHTS_DIGEST,
        },
        "source": {"path": "src/illm/policy_cpu_formula_worker.py", "sha256": "b" * 64},
        "candidate_loaded": False,
    }
    mutation_model = {
        "fixture_id": "tiny-decoder",
        "spec_id": TINY_SPEC_ID,
        "runtime_dtype": "fp32",
        "weight_manifest_digest": TINY_LOAD_MANIFEST_DIGEST,
        "source": {"path": "src/illm/policy_cpu_formula_worker.py", "sha256": "c" * 64},
    }
    models = {**runtime_models, "formula": formula_model, "mutations": mutation_model}
    return {
        name: cast(validation._Worker, SimpleNamespace(report={"model": model}))
        for name, model in models.items()
    }


def test_native_oracle_identity_is_closed_and_not_inherited_by_candidate() -> None:
    workers = _model_validation_workers()
    validation._validate_models(workers)

    candidate_model = cast(dict[str, object], workers["candidate"].report["model"])
    candidate_model["native_oracle"] = copy.deepcopy(
        cast(Mapping[str, object], workers["native"].report["model"])["native_oracle"]
    )
    with pytest.raises(validation.NumericalEvidenceError, match="candidate model identity"):
        validation._validate_models(workers)

    workers = _model_validation_workers()
    native_model = cast(dict[str, object], workers["native"].report["model"])
    native_oracle = cast(dict[str, object], native_model["native_oracle"])
    native_oracle["source_shards"] = []
    with pytest.raises(validation.NumericalEvidenceError, match="source shard identity"):
        validation._validate_models(workers)


def test_record_stream_is_exclusive_and_byte_verified(tmp_path: Path) -> None:
    output = tmp_path / "comparisons.jsonl"
    sink = validation._RecordSink(output=output, verify=None)
    sink.add({"schema_version": 1, "result": {"atol": 1.0e-4}})
    summary = sink.finish()
    assert summary["record_count"] == 1
    assert summary["sha256"] == sha256_bytes(output.read_bytes())

    verifier = validation._RecordSink(output=None, verify=output)
    verifier.add({"schema_version": 1, "result": {"atol": 1.0e-4}})
    assert verifier.finish() == summary

    derived = validation._RecordSink(output=None, verify=None)
    derived.add({"schema_version": 1, "result": {"atol": 1.0e-4}})
    assert derived.finish() == summary

    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(canonical_json_bytes({"schema_version": 1, "result": {"atol": 1.0e-3}}) + b"\n")
    verifier = validation._RecordSink(output=None, verify=changed)
    with pytest.raises(validation.NumericalEvidenceError, match="differs"):
        verifier.add({"schema_version": 1, "result": {"atol": 1.0e-4}})
    verifier.abort()

    trailing = tmp_path / "trailing.jsonl"
    trailing.write_bytes(output.read_bytes() + b"{}\n")
    verifier = validation._RecordSink(output=None, verify=trailing)
    verifier.add({"schema_version": 1, "result": {"atol": 1.0e-4}})
    with pytest.raises(validation.NumericalEvidenceError, match="trailing"):
        verifier.finish()


def test_comparison_is_derived_from_raw_not_a_worker_pass_label(tmp_path: Path) -> None:
    expected_id = "B/case/000/logits/000"
    actual_id = "C/case/000/logits/000"
    expected_worker = _worker_from_records(
        tmp_path / "expected", directory_name="candidate", role="candidate",
        records=[(expected_id, "fp32", [1], _raw([1.0]))],
    )
    actual_worker = _worker_from_records(
        tmp_path / "actual", directory_name="candidate", role="candidate",
        records=[(actual_id, "fp32", [1], _raw([1.1]))],
    )
    expected = expected_worker.tensor(expected_id)
    actual = actual_worker.tensor(actual_id)
    axis = validation._Axis()
    path = tmp_path / "records.jsonl"
    sink = validation._RecordSink(output=path, verify=None)
    result = validation._compare(
        sink,
        axis,
        axis_name="B-to-C",
        case_id="case",
        step=0,
        op="logits",
        position=0,
        expected_worker=expected_worker,
        expected_id=expected_id,
        expected_source=expected,
        expected=expected,
        actual_worker=actual_worker,
        actual_id=actual_id,
        actual_source=actual,
        actual=actual,
        expected_direction="B-expected",
    )
    sink.finish()
    assert result["status"] == "fail"
    assert axis.report()["status"] == "fail"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["expected_direction"] == "B-expected"
    assert record["expected"]["source_raw_sha256"] == expected.raw_sha256
    assert record["actual"]["source_raw_sha256"] == actual.raw_sha256
    assert record["result"]["atol"] == 1.0e-4
    assert record["result"]["rtol"] == 1.0e-3


def _stateful_fixture(tmp_path: Path, *, token: int = 5) -> tuple[validation._Worker, dict[str, Any], dict[str, Any]]:
    case_id = "case"
    identifier = tensor_id("B", case_id, 0, "full_logits", None)
    logits = _logits(5)
    worker = _worker_from_records(
        tmp_path, directory_name="candidate", role="candidate",
        records=[(identifier, "fp32", [1, 1, 259], _raw(logits))],
    )
    spec = SamplingSpec(mode="greedy", actual_seed=7)
    state = SamplingState.for_request(case_id, spec)
    decision = prepare_sample(logits, spec, state)
    committed = commit_sample(state, decision, spec=spec, logits=logits)
    case: dict[str, Any] = {
        "case_id": case_id,
        "input_token_ids": [3],
        "max_new_tokens": 1,
        "sampling_spec": spec.to_dict(),
        "input_digest": "2" * 64,
        "sampling_spec_digest": spec.digest,
        "suite_id": "synthetic",
        "expected_output_token_ids": None,
        "case_digest": "3" * 64,
    }
    path: dict[str, Any] = {
        "path_id": "B",
        "execution_flags": _flags(),
        "steps": [{
            "step": 0,
            "context": [3],
            "output_token_id": token,
            "logits_id": identifier,
            "state_before": _state_record(state),
            "state_after": _state_record(committed),
            "query_tokens": 1,
            "kv_before": 0,
            "kv_after": 0,
            "op_ids": {},
            "prefix_ids": {},
            "ordinary_logits_id": None,
            "fence_complete": None,
        }],
        "output_token_ids": [token],
        "finish_reason": "MAX_NEW_TOKENS",
        "resource_status": _resources(),
    }
    return worker, case, path


def test_sampling_token_and_state_are_recomputed_from_full_vocab_raw(tmp_path: Path) -> None:
    worker, case, path = _stateful_fixture(tmp_path / "valid")
    result = validation._validate_stateful_path(
        worker, path, path_id="B", case=case, candidate=True
    )
    assert result["output_token_ids"] == [5]

    worker, case, path = _stateful_fixture(tmp_path / "forged", token=6)
    with pytest.raises(validation.NumericalEvidenceError, match="token or committed sampling state"):
        validation._validate_stateful_path(worker, path, path_id="B", case=case, candidate=True)


def _candidate_op_fixture(
    root: Path,
    *,
    path_id: str = "B",
    alias_embedding: bool = False,
) -> tuple[validation._Worker, dict[str, Any]]:
    case_id = "case"
    records: list[tuple[str, str, Sequence[int], bytes]] = []
    op_ids: dict[str, dict[str, str]] = {}
    logits_raw = _raw(_logits(5))
    for op in canonical_tiny_ops():
        identifier = tensor_id(path_id, case_id, 0, op, 0)
        shape = validation._candidate_shape(op, 0)
        raw = logits_raw if op == "logits" else _zeros(shape)
        records.append((identifier, "fp32", shape, raw))
        op_ids[op] = {"0": identifier}
    full_id = tensor_id(path_id, case_id, 0, "full_logits", None)
    records.append((full_id, "fp32", [1, 1, 259], logits_raw))
    if alias_embedding:
        alias = tensor_id(path_id, case_id, 0, "alias", 0)
        records.append((alias, "fp32", validation._candidate_shape("embedding", 0),
                        _zeros(validation._candidate_shape("embedding", 0))))
        op_ids["embedding"] = {"0": alias}
    worker = _worker_from_records(
        root, directory_name="candidate", role="candidate", records=records
    )
    path: dict[str, Any] = {
        "steps": [{
            "step": 0,
            "context": [3],
            "op_ids": op_ids,
            "logits_id": full_id,
            "prefix_ids": {},
        }]
    }
    return worker, path


def test_candidate_raw_inventory_and_semantic_ids_are_closed(tmp_path: Path) -> None:
    worker, path = _candidate_op_fixture(tmp_path / "valid")
    validation._validate_candidate_ops(worker, path, path_id="B", case_id="case")
    worker.require_complete_inventory()

    worker, path = _candidate_op_fixture(tmp_path / "alias", alias_embedding=True)
    with pytest.raises(validation.NumericalEvidenceError, match="semantic identity"):
        validation._validate_candidate_ops(worker, path, path_id="B", case_id="case")

    worker, path = _candidate_op_fixture(tmp_path / "missing")
    del cast(dict[str, Any], cast(list[Any], path["steps"])[0])["op_ids"]["layer.0.input_norm"]
    with pytest.raises(validation.NumericalEvidenceError, match="fields differ"):
        validation._validate_candidate_ops(worker, path, path_id="B", case_id="case")


def _native_op_fixture(root: Path, *, future_probability: float) -> tuple[validation._Worker, dict[str, Any]]:
    case_id = "case"
    length = 2
    records: list[tuple[str, str, Sequence[int], bytes]] = []
    op_ids: dict[str, str] = {}
    for op in canonical_tiny_ops():
        identifier = tensor_id("A", case_id, 0, op, None)
        shape = validation._native_shape(op, length)
        if op.endswith("attention_probabilities"):
            values = [0.0] * 16
            for head in range(4):
                values[head * 4] = 1.0
                values[head * 4 + 1] = future_probability
                values[head * 4 + 3] = 1.0
            raw = _raw(values)
        elif op == "logits":
            raw = _raw(_logits(5) + _logits(6))
        else:
            raw = _zeros(shape)
        records.append((identifier, "fp32", shape, raw))
        op_ids[op] = identifier
    worker = _worker_from_records(root, directory_name="native", role="native", records=records)
    path: dict[str, Any] = {
        "steps": [{
            "step": 0,
            "context": [3, 4],
            "op_ids": op_ids,
            "logits_id": op_ids["logits"],
            "ordinary_logits_id": None,
        }]
    }
    return worker, path


def test_native_future_probability_support_is_recomputed(tmp_path: Path) -> None:
    worker, path = _native_op_fixture(tmp_path / "valid", future_probability=0.0)
    validation._validate_native_ops(worker, path, path_id="A", case_id="case")
    worker.require_complete_inventory()

    worker, path = _native_op_fixture(tmp_path / "bad", future_probability=0.25)
    with pytest.raises(validation.NumericalEvidenceError, match="future attention probability"):
        validation._validate_native_ops(worker, path, path_id="A", case_id="case")


def _identity(descriptor: object) -> dict[str, object]:
    return {"descriptor": descriptor, "sha256": sha256_bytes(canonical_json_bytes(descriptor))}


def _tensor_identity(record: Mapping[str, object], dtype: str = "torch.float32") -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": record["shape"],
        "raw_sha256": record["sha256"],
    }


def _no_fence() -> dict[str, object]:
    return {
        "required": False,
        "fence_kind": None,
        "fence_complete": None,
        "fence_succeeded": None,
        "lease_state": None,
        "active_leases_after_cleanup": None,
        "store_closed": None,
    }


def _required_fence(lease_state: object = "aborted") -> dict[str, object]:
    return {
        "required": True,
        "fence_kind": "ImmediateFence",
        "fence_complete": True,
        "fence_succeeded": True,
        "lease_state": lease_state,
        "active_leases_after_cleanup": 0,
        "store_closed": True,
    }


def _mutation_fixture(root: Path, *, false_transpose: bool) -> validation._Worker:
    root.mkdir()
    writer = TensorPackWriter(root / "pack")
    metadata: dict[str, dict[str, object]] = {}

    def put(mutation_id: str, label: str, values: Sequence[float], shape: Sequence[int] = (1,)) -> str:
        identifier = tensor_id("D-mutation", mutation_id, 0, label, None)
        metadata[identifier] = writer.write_tensor(identifier, "fp32", shape, _raw(values))
        return identifier

    def observed(identifier: str) -> dict[str, object]:
        return _tensor_identity(metadata[identifier])

    mutations: list[dict[str, object]] = []
    mutation_id = "transpose-square-q-weight"
    original = put(mutation_id, "original_q_weight", [0.0])
    modified = put(mutation_id, "modified_q_weight", [1.0])
    expected = put(mutation_id, "expected_first_tensor", [0.0])
    actual = put(mutation_id, "actual_first_tensor", [0.0 if false_transpose else 1.0])
    mutations.append({
        "mutation_id": mutation_id, "status": "detected",
        "input_identity": _identity({"original": metadata[original]}),
        "modified_input_identity": _identity({"modified": metadata[modified]}),
        "expected_detector": {"kind": "first_trace_divergence", "name": "layer.0.q_projection", "position": 0},
        "observation": {"kind": "first_trace_divergence", "name": "layer.0.q_projection", "position": 0,
                        "expected_tensor": observed(expected), "actual_tensor": observed(actual)},
        "tensor_paths": {"original_q_weight": original, "modified_q_weight": modified,
                         "expected_first_tensor": expected, "actual_first_tensor": actual},
        "resource_fence": _no_fence(),
    })

    mutation_id = "decode-position-plus-one"
    expected = put(mutation_id, "expected_first_tensor", [0.0])
    actual = put(mutation_id, "actual_first_tensor", [1.0])
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"position": 1}),
        "modified_input_identity": _identity({"position": 2}),
        "expected_detector": {"kind": "first_trace_divergence", "name": "layer.0.rope_q", "position": 2},
        "observation": {"kind": "first_trace_divergence", "name": "layer.0.rope_q", "position": 2,
                        "expected_tensor": observed(expected), "actual_tensor": observed(actual)},
        "tensor_paths": {"expected_first_tensor": expected, "actual_first_tensor": actual},
        "resource_fence": _required_fence(["aborted", "aborted"]),
    })

    mutation_id = "wrong-gqa-group"
    labels = {name: put(mutation_id, name, [0.0 if name.startswith("expected") else 1.0])
              for name in ("expected_scores", "actual_scores", "expected_context", "actual_context")}
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"gqa": "right"}),
        "modified_input_identity": _identity({"gqa": "wrong"}),
        "expected_detector": {"kind": "scores_and_context", "first_name": "layer.0.attention_scores", "position": 0},
        "observation": {"kind": "scores_and_context", "first_name": "layer.0.attention_scores", "position": 0,
                        "scores_changed": True, "context_changed": True},
        "tensor_paths": labels, "resource_fence": _no_fence(),
    })

    mutation_id = "future-key-visible"
    query = put(mutation_id, "query", [0.0] * 32, (1, 4, 8))
    key = put(mutation_id, "bad_key_prefix", [0.0] * 32, (1, 2, 2, 8))
    value = put(mutation_id, "bad_value_prefix", [0.0] * 32, (1, 2, 2, 8))
    error = {"type": "ModelError", "code": "INVALID_ARGUMENT", "field": None, "message": "prefix shape"}
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"prefix": 1}),
        "modified_input_identity": _identity({"prefix": 2}),
        "expected_detector": {"kind": "exception", "code": "INVALID_ARGUMENT", "phase": "support-shape"},
        "observation": {"kind": "exception", "phase": "support-shape", "error": error},
        "tensor_paths": {"query": query, "bad_key_prefix": key, "bad_value_prefix": value},
        "resource_fence": _no_fence(),
    })

    mutation_id = "corrupt-committed-prefix-k"
    labels = {
        "original_committed_prefix_k": put(mutation_id, "original_committed_prefix_k", [0.0]),
        "modified_committed_prefix_k": put(mutation_id, "modified_committed_prefix_k", [1.0]),
        "b_scores": put(mutation_id, "b_scores", [0.0]),
        "native_scores": put(mutation_id, "native_scores", [0.0]),
        "c_scores": put(mutation_id, "c_scores", [1.0]),
    }
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"prefix": "clean"}),
        "modified_input_identity": _identity({"prefix": "corrupt"}),
        "expected_detector": {"kind": "b_native_prefix_hard_gate", "name": "layer.0.attention_scores", "position": 2},
        "observation": {"kind": "b_native_prefix_hard_gate", "name": "layer.0.attention_scores", "position": 2,
                        "b_native_within_tolerance": True, "candidate_native_within_tolerance": False},
        "tensor_paths": labels, "resource_fence": _required_fence(),
    })

    for mutation_id, code in (
        ("wrong-policy-digest", "INVALID_NUMERICAL_POLICY"),
        ("unknown-policy-field", "UNSUPPORTED_NUMERICAL_POLICY"),
    ):
        error = {"type": "NumericalPolicyError", "code": code, "field": None, "message": "rejected"}
        mutations.append({
            "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"valid": True}),
            "modified_input_identity": _identity({"valid": False}),
            "expected_detector": {"kind": "exception", "code": code, "phase": "pre-acceptance"},
            "observation": {"kind": "exception", "phase": "pre-acceptance", "error": error},
            "tensor_paths": {}, "resource_fence": _no_fence(),
        })

    mutation_id = "fused-bias"
    fused_paths = {label: put(mutation_id, label, [float(index)])
                   for index, label in enumerate(("attempted_input", "attempted_weight", "attempted_bias"))}
    fused_error = {"type": "AssertionError", "code": None, "field": None, "message": "fused bias"}
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"bias": False}),
        "modified_input_identity": _identity({"bias": True}),
        "expected_detector": {"kind": "linear_conformance_guard", "reason": "fused_bias"},
        "observation": {"kind": "linear_conformance_guard", "phase": "first-real-F.linear-call",
                        "call_count_before_rejection": 1,
                        "attempted_input": observed(fused_paths["attempted_input"]),
                        "attempted_weight": observed(fused_paths["attempted_weight"]),
                        "attempted_bias": observed(fused_paths["attempted_bias"]),
                        "error": fused_error, "attempted_bf16_raw": None},
        "tensor_paths": fused_paths, "resource_fence": _no_fence(),
    })

    mutation_id = "early-storage-cast"
    early_paths = {label: put(mutation_id, label, [float(index)])
                   for index, label in enumerate(("pre_cast_input", "pre_cast_weight"))}
    input_raw = struct.pack("<H", 1)
    weight_raw = struct.pack("<H", 2)
    inline = {
        "input": {"dtype": "torch.bfloat16", "shape": [1], "nbytes": 2,
                  "sha256": sha256_bytes(input_raw), "raw_hex": input_raw.hex()},
        "weight": {"dtype": "torch.bfloat16", "shape": [1], "nbytes": 2,
                   "sha256": sha256_bytes(weight_raw), "raw_hex": weight_raw.hex()},
    }
    early_error = {"type": "AssertionError", "code": None, "field": None, "message": "non-FP32"}
    mutations.append({
        "mutation_id": mutation_id, "status": "detected", "input_identity": _identity({"dtype": "fp32"}),
        "modified_input_identity": _identity({"dtype": "bf16"}),
        "expected_detector": {"kind": "linear_conformance_guard", "reason": "non_fp32_contraction"},
        "observation": {"kind": "linear_conformance_guard", "phase": "first-real-F.linear-call",
                        "call_count_before_rejection": 1,
                        "attempted_input": {"dtype": "torch.bfloat16", "shape": [1],
                                            "raw_sha256": inline["input"]["sha256"]},
                        "attempted_weight": {"dtype": "torch.bfloat16", "shape": [1],
                                             "raw_sha256": inline["weight"]["sha256"]},
                        "attempted_bias": None, "error": early_error, "attempted_bf16_raw": inline},
        "tensor_paths": early_paths, "resource_fence": _no_fence(),
    })

    manifest = writer.finalize()
    pack = TensorPack.open(root / "pack")
    report = {"mutations": mutations}
    return validation._Worker(
        "mutations", "mutation", root, report, "4" * 64,
        cast(str, manifest["manifest_digest"]), pack,
    )


def test_mutation_status_is_not_trusted_when_raw_first_difference_is_absent(tmp_path: Path) -> None:
    valid = _mutation_fixture(tmp_path / "valid", false_transpose=False)
    axis = validation._Axis()
    validation._validate_mutations(valid, axis)
    valid.require_complete_inventory()
    assert axis.report()["status"] == "pass"

    false_claim = _mutation_fixture(tmp_path / "false-claim", false_transpose=True)
    axis = validation._Axis()
    validation._validate_mutations(false_claim, axis)
    false_claim.require_complete_inventory()
    assert axis.report()["status"] == "fail"
    assert axis.failures == 1


def _runtime_report(pack_digest: str, *, status: str = "complete") -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_kind": "illm-policy-cpu-raw-worker-v1",
        "status": status,
        "evidence_stage": "DEVELOPMENT_ONLY",
        "qualification_status": "unqualified",
        "role": "native",
        "coverage": {},
        "binding": {},
        "binding_digest": "0" * 64,
        "process": {},
        "model": {},
        "policy": None,
        "pack_manifest_digest": pack_digest,
        "cases": [],
        "high_cases": [],
        "op_authority_cases": [],
    }


def _write_worker_root(root: Path, *, status: str = "complete") -> None:
    worker_root = root / "workers" / "native"
    worker_root.mkdir(parents=True)
    writer = TensorPackWriter(worker_root / "pack")
    writer.write_tensor("A/case/000/logits/full", "fp32", [1], _raw([0.0]))
    manifest = writer.finalize()
    report = _runtime_report(cast(str, manifest["manifest_digest"]), status=status)
    raw = canonical_json_bytes(report)
    (worker_root / "worker.json").write_bytes(raw)
    (worker_root / "COMPLETE").write_bytes((sha256_bytes(raw) + "\n").encode("ascii"))


def test_worker_reader_rejects_resigned_false_lifecycle_claim(tmp_path: Path) -> None:
    _write_worker_root(tmp_path / "valid")
    worker = validation._read_worker(tmp_path / "valid", "native", "native")
    assert worker.report["status"] == "complete"

    _write_worker_root(tmp_path / "false", status="pass")
    with pytest.raises(validation.NumericalEvidenceError, match="lifecycle"):
        validation._read_worker(tmp_path / "false", "native", "native")


def test_validator_import_is_stdlib_only_under_isolated_python(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "import illm.policy_cpu_raw_validation as module\n"
        "for name in ('torch','transformers','tokenizers','safetensors'):\n"
        "    assert name not in sys.modules, name\n"
        "assert callable(module.derive_cpu_numerical)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_reparse_or_symlink_paths_fail_before_reading(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("test account cannot create symlinks")
    with pytest.raises(validation.NumericalEvidenceError, match="links or reparse"):
        validation._ordinary_directory(linked, label="linked evidence")

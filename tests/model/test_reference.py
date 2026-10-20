"""Numerical gates require the explicit locked torch/oracle runtime profile."""

from __future__ import annotations

import tempfile
import hashlib
import json
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import torch

from illm.model import load_model
from illm.model.loader import LoadedModel, _preflight_device, build_weight_manifest, resolve_model
from illm.model.reference import ReferenceDecoder, canonical_op_names, _validate_execution_mode
from illm.model.spec import ModelError, digest
from illm.model.validation import (
    collect_reference_ops, collect_transformers_ops, compare_op_traces,
    load_transformers_oracle, validate_model,
    TransformersOracle,
)


ROOT = Path(__file__).resolve().parents[2]


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    parent = ROOT / "out" / "model-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class ReferenceNumericsTests(unittest.TestCase):
    original_threads: ClassVar[int]
    loaded: ClassVar[LoadedModel]
    oracle: ClassVar[TransformersOracle]

    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.loaded = load_model(project_root=ROOT)
        cls.oracle = load_transformers_oracle(cls.loaded)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def test_actual_safetensors_forward_shape_finite_and_tied_storage(self) -> None:
        ids = self.loaded.tokenizer.encode("hello")
        logits = self.loaded.reference(torch.tensor([ids], dtype=torch.int64))
        self.assertEqual(tuple(logits.shape), (1, len(ids), 259))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(self.loaded.reference.canonical_weight("model.embed_tokens.weight").data_ptr(),
                         self.loaded.reference.canonical_weight("lm_head.weight").data_ptr())

    def test_canonical_per_op_matches_independent_transformers(self) -> None:
        for ids in [(0,), (1,), (258,), self.loaded.tokenizer.encode("hello"), self.loaded.tokenizer.encode("你好")]:
            with self.subTest(ids=ids):
                inputs = torch.tensor([ids], dtype=torch.int64)
                expected = collect_transformers_ops(self.oracle, inputs, self.loaded)
                actual = collect_reference_ops(self.loaded.reference, inputs)
                report = compare_op_traces(expected, actual, op_order=canonical_op_names(self.loaded.spec),
                                           identity={"spec_id": self.loaded.spec.spec_id})
                self.assertEqual(report["status"], "pass", report["first_divergence"])
                self.assertEqual(len(report["ops"]), 39)

    def test_committed_independent_oracle_goldens_and_twenty_greedy_cases(self) -> None:
        from safetensors.torch import load_file

        directory = ROOT / "fixtures" / "tiny-decoder" / "oracle-v1"
        manifest = json.loads((directory / "oracle-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["manifest_digest"], digest({key: value for key, value in manifest.items() if key != "manifest_digest"}))
        self.assertEqual(manifest["spec_id"], self.loaded.spec.spec_id)
        self.assertEqual(manifest["weights_digest"], self.loaded.weight_manifest["weights_digest"])
        self.assertEqual(manifest["load_manifest_digest"], self.loaded.weight_manifest["manifest_digest"])
        self.assertEqual(manifest["tokenizer_revision"], self.loaded.tokenizer.revision)
        self.assertEqual(manifest["transformers_version"], "4.57.1")
        self.assertTrue(manifest["trace_adapter_verified_against_ordinary_forward"])
        for case in manifest["op_cases"]:
            with self.subTest(fixture=case["fixture_id"]):
                artifact = directory / case["file"]
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), case["sha256"])
                expected = load_file(str(artifact))
                inputs = torch.tensor([case["input_ids"]], dtype=torch.int64)
                positions = torch.tensor([case["position_ids"]], dtype=torch.int64) if case["position_ids"] is not None else None
                actual = collect_reference_ops(self.loaded.reference, inputs, position_ids=positions)
                self.assertEqual(tuple(case["op_order"]), canonical_op_names(self.loaded.spec))
                report = compare_op_traces(expected, actual, op_order=tuple(case["op_order"]),
                                           identity={"oracle_manifest_digest": manifest["manifest_digest"]})
                self.assertEqual(report["status"], "pass", report["first_divergence"])
        self.assertEqual(len(manifest["greedy_cases"]), 20)
        for case in manifest["greedy_cases"]:
            with self.subTest(fixture=case["fixture_id"]):
                full = list(self.loaded.tokenizer.encode(case["public_prompt"]))
                self.assertEqual(full, case["input_ids"])
                for expected_token in case["expected_tokens"]:
                    logits = self.loaded.reference(torch.tensor([full], dtype=torch.int64))
                    selected = int(logits[0, -1].float().argmax().item())
                    self.assertEqual(selected, expected_token)
                    full.append(selected)

    def test_rope_absolute_position_boundaries(self) -> None:
        for position in [0, 1, self.loaded.spec.max_position_embeddings - 1]:
            with self.subTest(position=position):
                inputs, positions = torch.tensor([[258]], dtype=torch.int64), torch.tensor([[position]], dtype=torch.int64)
                expected = collect_transformers_ops(self.oracle, inputs, self.loaded, position_ids=positions)
                actual = collect_reference_ops(self.loaded.reference, inputs, position_ids=positions)
                report = compare_op_traces(expected, actual, op_order=canonical_op_names(self.loaded.spec), identity={"position": position})
                self.assertEqual(report["status"], "pass", report["first_divergence"])

    def test_bad_transpose_localizes_first_q_projection_not_just_logits(self) -> None:
        weights = {name: self.loaded.reference.canonical_weight(name).clone() for name in self.loaded.spec.tensor_shapes()}
        weights["model.layers.0.self_attn.q_proj.weight"] = weights["model.layers.0.self_attn.q_proj.weight"].T.contiguous()
        bad = ReferenceDecoder(self.loaded.spec, weights)
        inputs = torch.tensor([self.loaded.tokenizer.encode("hello")], dtype=torch.int64)
        report = compare_op_traces(collect_transformers_ops(self.oracle, inputs, self.loaded), collect_reference_ops(bad, inputs),
                                   op_order=canonical_op_names(self.loaded.spec),
                                   identity={"spec_id": self.loaded.spec.spec_id, "fault_fixture_id": "transpose-layer0-q-weight-v1"},
                                   report_link="fault-transposed-q.json")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["first_divergence"]["op"], "layer.0.q_projection")
        self.assertEqual(report["first_divergence"]["report_link"], "fault-transposed-q.json")

    def test_oracle_loads_source_not_potentially_corrupted_reference_buffers(self) -> None:
        mutated = load_model(project_root=ROOT)
        weight = mutated.reference.canonical_weight("model.layers.0.self_attn.q_proj.weight")
        with torch.no_grad():
            weight.copy_(weight.T.contiguous())
        independent = load_transformers_oracle(mutated)
        inputs = torch.tensor([mutated.tokenizer.encode("hello")], dtype=torch.int64)
        report = compare_op_traces(collect_transformers_ops(independent, inputs, mutated),
                                   collect_reference_ops(mutated.reference, inputs),
                                   op_order=canonical_op_names(mutated.spec),
                                   identity={"fault_fixture_id": "reference-buffer-transpose-before-oracle-load-v1"})
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["first_divergence"]["op"], "layer.0.q_projection")

    def test_oracle_rejects_source_artifact_changed_since_loader_preflight(self) -> None:
        with temporary_directory() as temporary:
            assets = Path(temporary)
            (assets / "config.json").write_bytes((self.loaded.assets_directory / "config.json").read_bytes())
            original = (self.loaded.assets_directory / "model.safetensors").read_bytes()
            (assets / "model.safetensors").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ModelError) as raised:
                load_transformers_oracle(replace(self.loaded, assets_directory=assets))
            self.assertEqual(raised.exception.code, "INVALID_WEIGHTS")

    def test_dtype_aware_oracle_retains_original_fp32_rope_and_global_state(self) -> None:
        weights = {name: self.loaded.reference.canonical_weight(name).detach() for name in self.loaded.spec.tensor_shapes()}
        original_dtype = torch.get_default_dtype()
        original_rng = torch.random.get_rng_state().clone()
        for dtype, expected_dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            with self.subTest(dtype=dtype):
                spec = replace(self.loaded.spec, runtime_dtype=dtype)
                reference = ReferenceDecoder(spec, weights)
                manifest = build_weight_manifest([self.loaded.assets_directory / "model.safetensors"], spec)
                low_precision = replace(self.loaded, spec=spec, reference=reference, weight_manifest=manifest)
                oracle = load_transformers_oracle(low_precision)
                self.assertEqual(oracle.model.embed_tokens.weight.dtype, expected_dtype)
                self.assertEqual(oracle.model.rotary_emb.inv_freq.dtype, torch.float32)
                self.assertTrue(torch.equal(oracle.model.rotary_emb.inv_freq, self.oracle.model.rotary_emb.inv_freq))
                self.assertEqual(torch.get_default_dtype(), original_dtype)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))

    def test_fp16_mask_overflow_is_legal_but_probabilities_and_logits_stay_finite(self) -> None:
        weights = {name: self.loaded.reference.canonical_weight(name).clone() for name in self.loaded.spec.tensor_shapes()}
        for layer in range(self.loaded.spec.num_layers):
            q = weights[f"model.layers.{layer}.self_attn.q_proj.weight"]
            k = weights[f"model.layers.{layer}.self_attn.k_proj.weight"]
            q.zero_().fill_diagonal_(40)
            k.zero_().fill_diagonal_(-40)
        reference = ReferenceDecoder(replace(self.loaded.spec, runtime_dtype="fp16"), weights)
        trace = collect_reference_ops(reference, torch.tensor([[11, 11]], dtype=torch.int64))
        count = sum(int(torch.isneginf(value).sum().item()) for name, value in trace.items() if name.endswith(".attention_scores"))
        self.assertGreater(count, 0)
        for name, value in trace.items():
            if not name.endswith(".attention_scores"):
                self.assertTrue(torch.isfinite(value).all(), name)
        report = compare_op_traces(trace, trace, op_order=canonical_op_names(reference.spec), identity={"fixture_id": "fp16-causal-mask-overflow-v1"})
        self.assertEqual(report["status"], "pass")
        self.assertEqual(sum(op["masked_negative_infinity_count"] for op in report["ops"]), count)
        self.assertEqual(max(op["max_absolute_error"] for op in report["ops"]), 0)

    def test_nonfinite_exceptions_are_restricted_to_paired_strict_future_mask(self) -> None:
        name = "layer.0.attention_scores"
        base = torch.zeros((1, 1, 2, 2), dtype=torch.float16)
        masked = base.clone()
        masked[0, 0, 0, 1] = float("-inf")
        self.loaded.reference._record(name, masked, 0, None)
        for category, position, value, op_name in (
            ("nan-future", (0, 1), float("nan"), name),
            ("positive-inf-future", (0, 1), float("inf"), name),
            ("negative-inf-past", (1, 0), float("-inf"), name),
            ("negative-inf-diagonal", (0, 0), float("-inf"), name),
            ("negative-inf-logits", (0, 1), float("-inf"), "logits"),
            ("negative-inf-probabilities", (0, 1), float("-inf"), "layer.0.attention_probabilities"),
        ):
            with self.subTest(category=category):
                invalid = base.clone()
                invalid[0, 0, position[0], position[1]] = value
                with self.assertRaises(ModelError) as raised:
                    self.loaded.reference._record(op_name, invalid, 0, None)
                self.assertEqual(raised.exception.code, "NUMERICAL_ERROR")
                report = compare_op_traces({op_name: invalid}, {op_name: invalid}, op_order=(op_name,), identity={})
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["first_divergence"]["reason"], "non_finite")
        mismatch = compare_op_traces({name: masked}, {name: base}, op_order=(name,), identity={})
        self.assertEqual(mismatch["status"], "fail")
        self.assertEqual(mismatch["first_divergence"]["reason"], "masked_negative_infinity_mismatch")
        for shape in ((2, 2), (1, 1, 2, 3)):
            with self.subTest(noncanonical_shape=shape):
                invalid = torch.zeros(shape, dtype=torch.float16)
                invalid.reshape(-1)[1] = float("-inf")
                with self.assertRaises(ModelError):
                    self.loaded.reference._record(name, invalid, 0, None)
                report = compare_op_traces({name: invalid}, {name: invalid}, op_order=(name,), identity={})
                self.assertEqual(report["status"], "fail")

    def test_invalid_inputs_and_positions_fail_before_forward(self) -> None:
        bad_inputs = [torch.tensor([[259]]), torch.tensor([[-1]]), torch.zeros((2, 1), dtype=torch.int64),
                      torch.zeros((1, 0), dtype=torch.int64), torch.zeros((1, 257), dtype=torch.int64),
                      torch.tensor([[1.0]])]
        for inputs in bad_inputs:
            with self.subTest(shape=tuple(inputs.shape)), self.assertRaises(ModelError):
                self.loaded.reference(inputs)
        with self.assertRaises(ModelError):
            self.loaded.reference(torch.tensor([[1]]), position_ids=torch.tensor([[256]]))
        with self.assertRaises(ModelError):
            self.loaded.reference(torch.tensor([[1, 2]]), position_ids=torch.tensor([[1, 3]]))

    def test_nonfinite_weights_and_autocast_fail_closed(self) -> None:
        weights = {name: self.loaded.reference.canonical_weight(name).clone() for name in self.loaded.spec.tensor_shapes()}
        weights["model.norm.weight"][0] = float("nan")
        with self.assertRaises(ModelError):
            ReferenceDecoder(self.loaded.spec, weights)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), self.assertRaises(ModelError):
            self.loaded.reference(torch.tensor([[1]]))

    def test_external_tf32_enabled_cuda_mode_fails_without_global_mutation(self) -> None:
        original = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            with patch("torch.is_autocast_enabled", return_value=False):
                with self.assertRaises(ModelError) as raised:
                    _validate_execution_mode("cuda")
                self.assertEqual(raised.exception.code, "UNSUPPORTED_MODEL")
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                _validate_execution_mode("cpu")
            torch.backends.cuda.matmul.allow_tf32 = False
            with patch("torch.is_autocast_enabled", return_value=False):
                _validate_execution_mode("cuda")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = original

    def test_cuda_hardware_guards_fail_before_any_weight_allocation(self) -> None:
        source = replace(resolve_model(project_root=ROOT), minimum_gpu_memory_bytes=6 * 1024 ** 3,
                         minimum_compute_capability=(8, 0))
        for invalid_device in ("mps", "cuda:-1", "cpu:0", "cuda:one"):
            with self.subTest(device=invalid_device), self.assertRaises(ModelError):
                _preflight_device(source, invalid_device)
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(ModelError) as raised:
            _preflight_device(source, "cuda:0")
        self.assertEqual(raised.exception.code, "UNSUPPORTED_MODEL")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaises(ModelError) as raised:
                _preflight_device(source, "cuda:1")
            self.assertEqual(raised.exception.code, "INVALID_ARGUMENT")
            with patch("torch.cuda.get_device_capability", return_value=(7, 5)), self.assertRaises(ModelError) as raised:
                _preflight_device(source, "cuda:0")
            self.assertEqual(raised.exception.field, "minimum_compute_capability")
            with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
                with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=4 * 1024 ** 3)):
                    with self.assertRaises(ModelError) as raised:
                        _preflight_device(source, "cuda:0")
                    self.assertEqual(raised.exception.code, "RESOURCE_EXHAUSTED")
                with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=8 * 1024 ** 3)):
                    _preflight_device(source, "cuda:0")

    def test_all_twenty_fixed_greedy_inputs_and_linked_reports(self) -> None:
        with temporary_directory() as temporary:
            directory = Path(temporary)
            report = validate_model(project_root=ROOT, output_directory=directory)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["fixed_input_count"], 20)
            self.assertTrue(all(case["reference_tokens"] == case["oracle_tokens"] for case in report["greedy_reports"]))
            self.assertTrue((directory / "validation-report.json").is_file())
            self.assertTrue((directory / report["first_divergence_diagnostic"]["report_link"]).is_file())
            self.assertTrue(all((directory / op_report["report_link"]).is_file() for op_report in report["per_op_reports"]))
            self.assertEqual(set(report["model_artifacts"]), {"model-spec.json", "weight-load-manifest.json", "tokenizer-spec.json"})
            for artifact in report["model_artifacts"].values():
                self.assertEqual(hashlib.sha256((directory / artifact["path"]).read_bytes()).hexdigest(), artifact["sha256"])
            fault_identity = report["first_divergence_diagnostic"]["report"]["identity"]
            self.assertEqual(fault_identity["weights_identity_role"], "unmodified-source-before-explicit-fault")
            self.assertNotEqual(fault_identity["fault_transform"]["source_tensor_checksum"],
                                fault_identity["fault_transform"]["mutated_tensor_checksum"])
            self.assertEqual(fault_identity["fault_execution_identity"],
                             digest({key: value for key, value in fault_identity.items() if key != "fault_execution_identity"}))
            pure_output = report["first_divergence_diagnostic"]["report"]["output_comparison"]
            self.assertGreater(pure_output["final_logits_changed_count"], 0)
            self.assertGreater(pure_output["final_logits_max_absolute_error"], 0)
            self.assertFalse(pure_output["token_mismatch"])
            compound = report["token_mismatch_diagnostic"]
            self.assertEqual(compound["status"], "pass")
            self.assertTrue((directory / compound["report_link"]).is_file())
            self.assertEqual(compound["report"]["first_divergence"]["op"], "layer.0.q_projection")
            self.assertEqual(len(compound["report"]["identity"]["fault_transforms"]), 2)
            self.assertTrue(compound["report"]["output_comparison"]["token_mismatch"])
            self.assertEqual(compound["report"]["output_comparison"]["expected_next_token"], 114)
            self.assertEqual(compound["report"]["output_comparison"]["actual_next_token"], 0)

    def test_validation_never_overwrites_existing_evidence_or_file(self) -> None:
        with temporary_directory() as temporary:
            directory = Path(temporary)
            marker = directory / "validation-report.json"
            marker.write_bytes(b"existing immutable evidence")
            for target in (directory, marker):
                with self.subTest(target=target), patch("illm.model.loader.load_model") as load:
                    with self.assertRaises(ModelError) as raised:
                        validate_model(project_root=ROOT, output_directory=target)
                    self.assertEqual(raised.exception.code, "INVALID_ARGUMENT")
                    load.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"existing immutable evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)

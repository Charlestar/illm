"""CPU-only gates for the independent D-high binary64 tiny oracle."""

from __future__ import annotations

import ast
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar
from unittest.mock import patch

import torch
from torch import Tensor

from illm.model.high_precision import HighPrecisionError, HighPrecisionOracle, OpRecord


ROOT = Path(__file__).resolve().parents[2]

_LAYER_OPS = (
    "input_norm", "q_projection", "k_projection", "v_projection", "rope_q", "rope_k",
    "attention_scores", "attention_probabilities", "attention_context", "attention_output",
    "attention_residual", "post_attention_norm", "mlp_gate", "mlp_up", "mlp_silu",
    "mlp_product", "mlp_output", "mlp_residual",
)
_CANONICAL_OPS = ("embedding",) + tuple(
    f"layer.{layer}.{name}" for layer in range(2) for name in _LAYER_OPS
) + ("final_norm", "logits")


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    parent = ROOT / "out" / "high-precision-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def collect_trace(oracle: HighPrecisionOracle, ids: list[int]) -> tuple[Tensor, list[tuple[OpRecord, Tensor]]]:
    trace: list[tuple[OpRecord, Tensor]] = []

    def hook(record: OpRecord, tensor: Tensor) -> None:
        trace.append((record, tensor))

    inputs = torch.tensor([ids], dtype=torch.int64)
    positions = torch.arange(len(ids), dtype=torch.int64).unsqueeze(0)
    return oracle.forward(inputs, positions, hook), trace


def trace_map(trace: list[tuple[OpRecord, Tensor]]) -> dict[tuple[str, int], tuple[OpRecord, Tensor]]:
    result: dict[tuple[str, int], tuple[OpRecord, Tensor]] = {}
    for record, tensor in trace:
        key = (record.name, record.position)
        if key in result:
            raise AssertionError(f"duplicate trace key {key}")
        result[key] = (record, tensor)
    return result


class HighPrecisionOracleTests(unittest.TestCase):
    oracle: ClassVar[HighPrecisionOracle]
    original_threads: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.oracle = HighPrecisionOracle.from_fixture(ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def test_original_fixture_is_bound_before_safetensors_load(self) -> None:
        identity = self.oracle.identity
        self.assertEqual(identity["model_sha256"], "8eeaeeb38565ca8ac40177f239a8405396a2b6642aef5223b5bf09205ffb056a")
        self.assertEqual(identity["spec_id"], "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873")
        self.assertEqual(identity["fixture_manifest_digest"], "4733493e2b57ce218545762f6ef1418dd6b47b5cd5c0cdc40bd1b17d033be38b")
        self.assertEqual(identity["weight_manifest_digest"], "dd0827b9ed7e4b1e2808e30552602a38cdda4666bf8b1417482b69ce1e5840bd")
        embedding = self.oracle.canonical_weight("model.embed_tokens.weight")
        tied_head = self.oracle.canonical_weight("lm_head.weight")
        self.assertEqual(embedding.dtype, torch.float64)
        self.assertTrue(torch.equal(embedding, tied_head))
        embedding.zero_()
        self.assertFalse(torch.equal(embedding, self.oracle.canonical_weight("model.embed_tokens.weight")))

        with temporary_directory() as temporary:
            project = Path(temporary) / "project"
            target = project / "fixtures" / "tiny-decoder"
            target.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "fixtures" / "tiny-decoder", target)
            path = target / "model.safetensors"
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with patch("safetensors.torch.load") as load, self.assertRaises(HighPrecisionError) as raised:
                HighPrecisionOracle.from_fixture(project)
            self.assertEqual(raised.exception.code, "INVALID_FIXTURE")
            load.assert_not_called()

    def test_seven_full_forward_cases_remain_cpu_binary64(self) -> None:
        cases = {
            "high-00": [3],
            "high-01": [4, 5],
            "high-02": [2],
            "high-03": [0, 0, 0, 0, 0, 0, 0],
            "high-04": [1, 2, 258, 3, 257, 4, 256],
            "high-06": [3 + ((43 * 6 + 17 * position) % 256) for position in range(31)],
            "high-07": [3 + ((43 * 7 + 17 * position) % 256) for position in range(255)],
        }
        self.assertEqual(set(cases), {f"high-{index:02d}" for index in (0, 1, 2, 3, 4, 6, 7)})
        for fixture_id, ids in cases.items():
            with self.subTest(fixture_id=fixture_id):
                inputs = torch.tensor([ids], dtype=torch.int64)
                positions = torch.arange(len(ids), dtype=torch.int64).unsqueeze(0)
                logits = self.oracle.forward(inputs, positions)
                self.assertEqual(tuple(logits.shape), (1, len(ids), 259))
                self.assertEqual(logits.dtype, torch.float64)
                self.assertEqual(logits.device.type, "cpu")
                self.assertTrue(bool(torch.isfinite(logits).all().item()))
                self.assertTrue(torch.equal(inputs, torch.tensor([ids], dtype=torch.int64)))
        first = self.oracle.forward(torch.tensor([[3]], dtype=torch.int64))
        second = self.oracle.forward(torch.tensor([[3]], dtype=torch.int64))
        self.assertTrue(torch.equal(first, second))

    def test_trace_has_unique_position_keys_and_ragged_canonical_shapes(self) -> None:
        logits, trace = collect_trace(self.oracle, [4, 5])
        indexed = trace_map(trace)
        expected = {(name, position) for name in _CANONICAL_OPS for position in (0, 1)}
        self.assertEqual(set(indexed), expected)
        self.assertEqual(len(trace), 78)
        for (name, position), (record, tensor) in indexed.items():
            with self.subTest(name=name, position=position):
                self.assertEqual(record.shape, tuple(tensor.shape))
                self.assertEqual(record.dtype, "torch.float64")
                self.assertEqual(record.device, "cpu")
                self.assertEqual(tensor.dtype, torch.float64)
                self.assertEqual(tensor.device.type, "cpu")
                self.assertTrue(bool(torch.isfinite(tensor).all().item()))
                if name.startswith("layer."):
                    self.assertEqual(record.layer, int(name.split(".")[1]))
                else:
                    self.assertIsNone(record.layer)
                if name.endswith("attention_scores") or name.endswith("attention_probabilities"):
                    self.assertEqual(tuple(tensor.shape), (1, 4, 1, position + 1))
                elif name.endswith("attention_context"):
                    self.assertEqual(tuple(tensor.shape), (1, 4, 1, 8))
                elif name.endswith("q_projection") or name.endswith("rope_q"):
                    self.assertEqual(tuple(tensor.shape), (1, 4, 1, 8))
                elif (name.endswith("k_projection") or name.endswith("v_projection")
                      or name.endswith("rope_k")):
                    self.assertEqual(tuple(tensor.shape), (1, 2, 1, 8))
                elif name.endswith(("mlp_gate", "mlp_up", "mlp_silu", "mlp_product")):
                    self.assertEqual(tuple(tensor.shape), (1, 64))
                elif name == "logits":
                    self.assertEqual(tuple(tensor.shape), (1, 259))
                else:
                    self.assertEqual(tuple(tensor.shape), (1, 32))
        self.assertTrue(torch.equal(logits[:, 0, :], indexed[("logits", 0)][1]))
        self.assertTrue(torch.equal(logits[:, 1, :], indexed[("logits", 1)][1]))
        for position in (0, 1):
            probabilities = indexed[("layer.0.attention_probabilities", position)][1]
            torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 4, 1), dtype=torch.float64),
                                       rtol=0.0, atol=2e-15)

    def test_gqa_mapping_and_legal_prefix_mask_are_formula_checked(self) -> None:
        _, trace = collect_trace(self.oracle, [4, 5])
        indexed = trace_map(trace)
        query = indexed[("layer.0.rope_q", 1)][1]
        keys = torch.cat([indexed[("layer.0.rope_k", position)][1] for position in (0, 1)], dim=2)
        values = torch.cat([indexed[("layer.0.v_projection", position)][1] for position in (0, 1)], dim=2)
        expected_scores: list[Tensor] = []
        expected_context: list[Tensor] = []
        wrong_context: list[Tensor] = []
        for head in range(4):
            kv_head = head // 2
            scores = torch.matmul(query[:, head, :, :], keys[:, kv_head, :, :].transpose(-2, -1)) * (8.0 ** -0.5)
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            probability = torch.exp(shifted) / torch.exp(shifted).sum(dim=-1, keepdim=True)
            expected_scores.append(scores)
            expected_context.append(torch.matmul(probability, values[:, kv_head, :, :]))

            wrong_kv = head % 2
            wrong_scores = torch.matmul(query[:, head, :, :], keys[:, wrong_kv, :, :].transpose(-2, -1)) * (8.0 ** -0.5)
            wrong_shifted = wrong_scores - wrong_scores.amax(dim=-1, keepdim=True)
            wrong_probability = torch.exp(wrong_shifted) / torch.exp(wrong_shifted).sum(dim=-1, keepdim=True)
            wrong_context.append(torch.matmul(wrong_probability, values[:, wrong_kv, :, :]))
        score_tensor = torch.cat(expected_scores, dim=1).unsqueeze(2)
        context_tensor = torch.cat(expected_context, dim=1).unsqueeze(2)
        wrong_gqa_tensor = torch.cat(wrong_context, dim=1).unsqueeze(2)
        torch.testing.assert_close(indexed[("layer.0.attention_scores", 1)][1], score_tensor,
                                   rtol=0.0, atol=2e-15)
        torch.testing.assert_close(indexed[("layer.0.attention_context", 1)][1], context_tensor,
                                   rtol=0.0, atol=2e-15)
        self.assertFalse(torch.allclose(context_tensor, wrong_gqa_tensor, rtol=1e-12, atol=1e-12))

        # A deliberate future-visible formula at p=0 must disagree with the legal prefix.
        q0 = indexed[("layer.0.rope_q", 0)][1]
        future_context: list[Tensor] = []
        for head in range(4):
            kv_head = head // 2
            scores = torch.matmul(q0[:, head, :, :], keys[:, kv_head, :, :].transpose(-2, -1)) * (8.0 ** -0.5)
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            probability = torch.exp(shifted) / torch.exp(shifted).sum(dim=-1, keepdim=True)
            future_context.append(torch.matmul(probability, values[:, kv_head, :, :]))
        future_visible = torch.cat(future_context, dim=1).unsqueeze(2)
        legal = indexed[("layer.0.attention_context", 0)][1]
        expected_legal = torch.cat([values[:, head // 2, 0:1, :] for head in range(4)], dim=1).unsqueeze(2)
        torch.testing.assert_close(legal, expected_legal, rtol=0.0, atol=0.0)
        self.assertFalse(torch.allclose(legal, future_visible, rtol=1e-12, atol=1e-12))

    def test_transposed_square_q_weight_first_diverges_at_q_projection(self) -> None:
        _, baseline = collect_trace(self.oracle, [3, 4])
        name = "model.layers.0.self_attn.q_proj.weight"
        mutated = dict(self.oracle._weights)
        original = mutated[name]
        self.assertFalse(torch.equal(original, original.transpose(0, 1)))
        mutated[name] = original.transpose(0, 1).contiguous()
        with patch.object(self.oracle, "_weights", MappingProxyType(mutated)):
            _, faulted = collect_trace(self.oracle, [3, 4])
        self.assertEqual([(record.name, record.position) for record, _ in baseline],
                         [(record.name, record.position) for record, _ in faulted])
        first = next(
            (left_record.name, left_record.position)
            for (left_record, left), (_, right) in zip(baseline, faulted, strict=True)
            if not torch.equal(left, right)
        )
        self.assertEqual(first, ("layer.0.q_projection", 0))

    def test_high_05_is_rope_only_with_fp64_expected_never_cast_down(self) -> None:
        q_head = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)
        k_head = torch.arange(2, dtype=torch.float32).reshape(1, 2, 1, 1)
        dimension = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        query = ((2.0 * q_head + dimension - 9.0) / 16.0).expand(1, 4, 2, 8).contiguous()
        key = ((3.0 * k_head - dimension + 5.0) / 32.0).expand(1, 2, 2, 8).contiguous()
        positions = torch.tensor([[31, 32]], dtype=torch.int64)
        actual_q, actual_k = self.oracle.rope(query, key, positions)
        self.assertEqual(actual_q.dtype, torch.float64)
        self.assertEqual(actual_k.dtype, torch.float64)

        def scalar_expected(source: Tensor) -> Tensor:
            expected = torch.empty(source.shape, dtype=torch.float64)
            for head in range(source.shape[1]):
                for row, position in enumerate((31, 32)):
                    for index in range(8):
                        frequency_index = index % 4
                        inverse = 1.0 / (10000.0 ** ((2 * frequency_index) / 8.0))
                        angle = position * inverse
                        original = float(source[0, head, row, index].item())
                        rotated = (-float(source[0, head, row, index + 4].item()) if index < 4
                                   else float(source[0, head, row, index - 4].item()))
                        expected[0, head, row, index] = original * math.cos(angle) + rotated * math.sin(angle)
            return expected

        torch.testing.assert_close(actual_q, scalar_expected(query), rtol=2e-15, atol=2e-15)
        torch.testing.assert_close(actual_k, scalar_expected(key), rtol=2e-15, atol=2e-15)
        with self.assertRaises(HighPrecisionError) as raised:
            self.oracle.forward(torch.tensor([[3, 4]], dtype=torch.int64), positions)
        self.assertEqual(raised.exception.code, "INVALID_ARGUMENT")

    def test_invalid_inputs_fail_closed_and_hook_cannot_mutate_forward(self) -> None:
        invalid = (
            torch.tensor([[259]], dtype=torch.int64), torch.tensor([[-1]], dtype=torch.int64),
            torch.zeros((2, 1), dtype=torch.int64), torch.zeros((1, 0), dtype=torch.int64),
            torch.zeros((1, 257), dtype=torch.int64), torch.tensor([[1.0]], dtype=torch.float64),
        )
        for value in invalid:
            with self.subTest(shape=tuple(value.shape), dtype=str(value.dtype)), self.assertRaises(HighPrecisionError):
                self.oracle.forward(value)
        with self.assertRaises(HighPrecisionError):
            self.oracle.forward(torch.tensor([[3, 4]], dtype=torch.int64), torch.tensor([[0, 2]], dtype=torch.int64))
        q = torch.zeros((1, 4, 2, 8), dtype=torch.float64)
        k = torch.zeros((1, 2, 2, 8), dtype=torch.float64)
        with self.assertRaises(HighPrecisionError):
            self.oracle.rope(q, k, torch.tensor([[31, 32]], dtype=torch.int64))

        expected = self.oracle.forward(torch.tensor([[3, 4]], dtype=torch.int64))

        def destructive_hook(_record: OpRecord, tensor: Tensor) -> None:
            tensor.zero_()

        actual = self.oracle.forward(torch.tensor([[3, 4]], dtype=torch.int64), op_hook=destructive_hook)
        self.assertTrue(torch.equal(expected, actual))

    def test_module_has_no_runtime_or_model_oracle_dependency(self) -> None:
        path = ROOT / "src" / "illm" / "model" / "high_precision.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.append(node.module)
        forbidden = ("illm.", "transformers", "torch.nn")
        self.assertFalse([module for module in imported if module.startswith(forbidden)])
        self.assertNotIn("torch.cuda", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

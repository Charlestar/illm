from __future__ import annotations

import ast
import copy
import hashlib
import runpy
import json
import struct
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable, cast

from illm.model.loader import build_weight_manifest, inspect_safetensors, inspect_model, resolve_model
from illm.model.spec import ModelError, SupportedModelSpec
from illm.tokenizer.reference import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = runpy.run_path(str(ROOT / "fixtures" / "tiny-decoder" / "generate.py"))
tiny_config = cast(Callable[[], dict[str, Any]], _GENERATOR["tiny_config"])
encode_safetensors = cast(Callable[..., bytes], _GENERATOR["encode_safetensors"])
generate_fixture = cast(Callable[[Path], dict[str, Any]], _GENERATOR["generate_fixture"])


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    parent = ROOT / "out" / "model-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def make_spec(config: dict[str, Any] | None = None) -> SupportedModelSpec:
    return SupportedModelSpec.from_hf_config(config if config is not None else tiny_config(),
                                           model_revision="a" * 40, tokenizer_revision=ByteTokenizer().revision,
                                           runtime_dtype="fp32")


class SpecTests(unittest.TestCase):
    def test_immutable_identity_and_expected_canonical_shapes(self) -> None:
        spec = make_spec()
        self.assertEqual(spec.tensor_shapes()["model.layers.1.self_attn.k_proj.weight"], (16, 32))
        self.assertNotIn("lm_head.weight", spec.tensor_shapes())
        self.assertEqual(len(spec.tensor_shapes()), 26)
        self.assertGreater(spec.parameter_count, 20000)
        self.assertEqual(make_spec().spec_id, spec.spec_id)
        with self.assertRaises(FrozenInstanceError):
            spec.hidden_size = 64  # type: ignore[misc]

    def test_unknown_semantic_fields_fail_closed(self) -> None:
        for field in ("quantization_config", "rope_extra", "unknown_bias", "num_experts"):
            with self.subTest(field=field):
                config = tiny_config()
                config[field] = 1
                with self.assertRaises(ModelError) as raised:
                    make_spec(config)
                self.assertEqual(raised.exception.code, "UNSUPPORTED_MODEL")
                self.assertEqual(raised.exception.field, field)

    def test_missing_and_illegal_fields_fail_before_forward(self) -> None:
        cases: list[dict[str, Any]] = [
            {"hidden_size": 31}, {"head_dim": 7}, {"num_key_value_heads": 3},
            {"num_hidden_layers": True}, {"hidden_act": "relu"}, {"rms_norm_eps": 0},
            {"rope_theta": float("nan")}, {"use_sliding_window": True},
            {"use_mrope": True}, {"attention_dropout": 0.1}, {"attention_bias": False},
            {"mlp_bias": True}, {"tie_word_embeddings": 1}, {"eos_token_id": 259},
            {"bos_token_id": -1}, {"rope_scaling": {"rope_type": "linear", "factor": 2}},
            {"torch_dtype": "int8"}, {"architectures": ["LlamaForCausalLM"]},
        ]
        for patch in cases:
            with self.subTest(patch=patch), self.assertRaises(ModelError):
                make_spec({**tiny_config(), **patch})
        config = tiny_config()
        del config["hidden_size"]
        with self.assertRaises(ModelError):
            make_spec(config)

    def test_floating_revision_rejected_and_runtime_dtype_in_identity(self) -> None:
        with self.assertRaises(ModelError):
            SupportedModelSpec.from_hf_config(tiny_config(), model_revision="main",
                                             tokenizer_revision="a" * 40, runtime_dtype="fp32")
        config = tiny_config()
        first = make_spec(config)
        other = SupportedModelSpec.from_hf_config(config, model_revision=first.model_revision,
                                                 tokenizer_revision=first.tokenizer_revision, runtime_dtype="bf16")
        self.assertNotEqual(first.spec_id, other.spec_id)

    def test_real_qwen_config_can_be_normalized_without_torch(self) -> None:
        config = {**tiny_config(), "hidden_size": 896, "intermediate_size": 4864,
                  "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2,
                  "vocab_size": 151936, "bos_token_id": 151643, "eos_token_id": 151643,
                  "max_position_embeddings": 32768, "rope_theta": 1000000.0,
                  "torch_dtype": "bfloat16", "max_window_layers": 24, "sliding_window": 32768,
                  "use_cache": True, "transformers_version": "4.40.1", "initializer_range": 0.02}
        spec = make_spec(config)
        self.assertEqual(spec.head_dim, 64)
        self.assertEqual(spec.num_attention_heads // spec.num_kv_heads, 7)
        self.assertEqual(spec.parameter_count, 494032768)

    def test_real_profile_rejects_invalid_hardware_and_forged_revision_assets(self) -> None:
        profile = json.loads((ROOT / "configs" / "models" / "v0_1.yaml").read_text(encoding="utf-8"))
        cases: list[tuple[str, object]] = [
            ("minimum_gpu_memory_bytes", True), ("minimum_gpu_memory_bytes", 0),
            ("minimum_gpu_memory_bytes", 6 * 1024 ** 3 - 1),
            ("minimum_compute_capability", [7, 5]), ("minimum_compute_capability", [8]),
            ("minimum_compute_capability", [8, -1]), ("minimum_compute_capability", [8, False]),
        ]
        with temporary_directory() as temporary:
            config_path = Path(temporary) / "model.json"
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    config_path.write_text(json.dumps({**profile, field: value}), encoding="utf-8")
                    with self.assertRaises(ModelError) as raised:
                        resolve_model(config_path, project_root=ROOT)
                    self.assertEqual(raised.exception.code, "INVALID_MODEL_CONFIG")
                    self.assertEqual(raised.exception.field, field)
            forged = copy.deepcopy(profile)
            forged["artifacts"][0]["git_blob_sha1"] = "a" * 40
            config_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(ModelError) as raised:
                resolve_model(config_path, project_root=ROOT)
            self.assertEqual(raised.exception.code, "INVALID_MODEL_CONFIG")
            self.assertEqual(raised.exception.field, "config.json")


class TokenizerTests(unittest.TestCase):
    def test_unicode_roundtrip_and_byte_boundaries(self) -> None:
        tokenizer = ByteTokenizer()
        text = "hello 你好 🌍\x00"
        tokens = tokenizer.encode(text)
        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokenizer.decode(tokens), text)
        self.assertEqual(tokenizer.validate_ids((3, 258)), (3, 258))
        self.assertEqual(tokenizer.decode((0, 1, 2), skip_special_tokens=False), "<pad><bos><eos>")

    def test_empty_input_policy_and_explicit_input_identity(self) -> None:
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.encode(""), (1,))
        with self.assertRaises(ModelError):
            tokenizer.encode("", add_bos=False)
        tokens = tokenizer.encode("x")
        self.assertNotEqual(tokenizer.input_identity(tokens, add_bos=True), tokenizer.input_identity(tokens, add_bos=False))

    def test_invalid_tokens_surrogates_and_contract_fail(self) -> None:
        tokenizer = ByteTokenizer()
        for tokens in ((-1,), (259,), (True,)):
            with self.subTest(tokens=tokens), self.assertRaises(ModelError):
                tokenizer.decode(tokens)
        with self.assertRaises(ModelError):
            tokenizer.encode("\ud800")
        with self.assertRaises(ModelError):
            ByteTokenizer(byte_offset=4)


class SafetensorsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = ROOT / "fixtures" / "tiny-decoder"
        self.source_asset_hashes = json.loads((directory / "fixture-manifest.json").read_text(encoding="utf-8"))["files"]
        self._assert_source_assets_unchanged()

    def tearDown(self) -> None:
        self._assert_source_assets_unchanged()

    def _assert_source_assets_unchanged(self) -> None:
        directory = ROOT / "fixtures" / "tiny-decoder"
        for name, expected in self.source_asset_hashes.items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected,
                             f"negative loader test changed immutable source fixture: {name}")

    def _fixture_tensors(self) -> dict[str, tuple[tuple[int, ...], bytes]]:
        source_path = ROOT / "fixtures" / "tiny-decoder" / "model.safetensors"
        raw = source_path.read_bytes()
        return {name: (record.shape, raw[record.source_byte_start:record.source_byte_end])
                for name, record in inspect_safetensors(source_path).items()}

    def test_fixture_generation_reproducible_and_tied_storage_covered_once(self) -> None:
        with temporary_directory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            manifest = generate_fixture(first)
            self.assertEqual(generate_fixture(second), manifest)
            self.assertTrue((first / "model.safetensors").is_file())
            self.assertGreater((first / "model.safetensors").stat().st_size, 100000)
            records = inspect_safetensors(first / "model.safetensors")
            self.assertEqual(len(records), 26)
            weights = json.loads((first / "weight-load-manifest.json").read_text())
            aliases = [item for item in weights["storages"] if item["tied_weight_group"]]
            self.assertEqual(len(aliases), 1)
            self.assertEqual(aliases[0]["parameter_aliases"], ["model.embed_tokens.weight", "lm_head.weight"])
            self.assertEqual(weights["missing_keys"], [])
            self.assertEqual(weights["unexpected_keys"], [])
            self.assertEqual(weights["duplicate_keys"], [])
            with self.assertRaises(ModelError):
                generate_fixture(first)

    def test_missing_unexpected_wrong_shape_fail(self) -> None:
        with temporary_directory() as temporary:
            base = Path(temporary)
            path = base / "bad.safetensors"
            path.write_bytes(encode_safetensors({"wrong": ((1,), struct.pack("<f", 0))}))
            with self.assertRaises(ModelError):
                build_weight_manifest([path], make_spec())
            path.write_bytes(encode_safetensors({"model.embed_tokens.weight": ((1,), struct.pack("<f", 0))}))
            with self.assertRaises(ModelError):
                build_weight_manifest([path], make_spec())

    def test_each_coverage_and_tied_alias_fault_is_rejected_individually(self) -> None:
        source = self._fixture_tensors()
        cases = {}
        missing = dict(source)
        del missing["model.layers.0.self_attn.q_proj.bias"]
        cases["missing_only"] = missing
        unexpected = dict(source)
        unexpected["unused.weight"] = ((1,), struct.pack("<f", 0))
        cases["unexpected_only"] = unexpected
        duplicate_alias = dict(source)
        duplicate_alias["lm_head.weight"] = source["model.embed_tokens.weight"]
        cases["copied_tied_alias"] = duplicate_alias
        wrong_shape = dict(source)
        shape, data = source["model.layers.0.self_attn.k_proj.weight"]
        wrong_shape["model.layers.0.self_attn.k_proj.weight"] = (tuple(reversed(shape)), data)
        cases["transposed_non_square_shape"] = wrong_shape
        with temporary_directory() as temporary:
            path = Path(temporary) / "bad.safetensors"
            for fault, tensors in cases.items():
                with self.subTest(fault=fault):
                    path.write_bytes(encode_safetensors(tensors))
                    with self.assertRaises(ModelError) as raised:
                        build_weight_manifest([path], make_spec())
                    self.assertEqual(raised.exception.code, "INVALID_WEIGHTS")

    def test_valid_distinct_shards_cover_each_storage_once(self) -> None:
        tensors = self._fixture_tensors()
        names = sorted(tensors)
        with temporary_directory() as temporary:
            first, second = Path(temporary) / "a.safetensors", Path(temporary) / "b.safetensors"
            first.write_bytes(encode_safetensors({name: tensors[name] for name in names[:13]}))
            second.write_bytes(encode_safetensors({name: tensors[name] for name in names[13:]}))
            manifest = build_weight_manifest([second, first], make_spec())
            self.assertEqual(manifest["canonical_storage_count"], 26)
            self.assertEqual(len({item["canonical_name"] for item in manifest["storages"]}), 26)
            self.assertEqual(manifest["missing_keys"], [])
            self.assertEqual(manifest["unexpected_keys"], [])
            self.assertEqual(manifest["duplicate_keys"], [])

    def test_wrong_dtype_rejected_even_with_correct_byte_coverage(self) -> None:
        source_path = ROOT / "fixtures" / "tiny-decoder" / "model.safetensors"
        raw = source_path.read_bytes()
        header_length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_length])
        # Half as many F32 elements -> twice as many F16 elements: byte ranges
        # still cover the exact file, so rejection must be model shape/dtype.
        header["model.norm.weight"]["dtype"] = "F16"
        header["model.norm.weight"]["shape"] = [64]
        rewritten = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        with temporary_directory() as temporary:
            wrong = Path(temporary) / "wrong-dtype.safetensors"
            wrong.write_bytes(struct.pack("<Q", len(rewritten)) + rewritten + raw[8 + header_length:])
            self.assertEqual(inspect_safetensors(wrong)["model.norm.weight"].dtype, "F16")
            with self.assertRaises(ModelError) as raised:
                build_weight_manifest([wrong], make_spec())
            self.assertEqual(raised.exception.field, "model.norm.weight")

    def test_duplicate_header_key_rejected_before_dictionary_can_erase_it(self) -> None:
        with temporary_directory() as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}'
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 8)
            with self.assertRaises(ModelError) as raised:
                inspect_safetensors(path)
            self.assertEqual(raised.exception.code, "DUPLICATE_WEIGHT")

    def test_overlap_hole_trailing_or_truncated_storage_rejected(self) -> None:
        headers = [
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}},
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            {"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}},
        ]
        with temporary_directory() as temporary:
            path = Path(temporary) / "bad.safetensors"
            for header in headers:
                raw = json.dumps(header).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 8)
                with self.subTest(header=header), self.assertRaises(ModelError):
                    inspect_safetensors(path)

    def test_duplicate_across_shards_rejected(self) -> None:
        with temporary_directory() as temporary:
            first, second = Path(temporary) / "first.safetensors", Path(temporary) / "second.safetensors"
            raw = encode_safetensors({"model.norm.weight": ((32,), struct.pack("<f", 1) * 32)})
            first.write_bytes(raw)
            second.write_bytes(raw)
            with self.assertRaises(ModelError) as raised:
                build_weight_manifest([first, second], make_spec())
            self.assertEqual(raised.exception.code, "DUPLICATE_WEIGHT")

    def test_eager_module_parses_without_claiming_dependency_runtime_test(self) -> None:
        module = ROOT / "src" / "illm" / "model" / "reference.py"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        self.assertIn("ReferenceDecoder", [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Regenerate the immutable tiny fixture without torch or random backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from illm.model.spec import ModelError, SupportedModelSpec, canonical_bytes, digest
from illm.model.loader import build_weight_manifest, _file_digest
from illm.tokenizer.reference import ByteTokenizer

GENERATOR = "illm-tiny-sha256-binary32-v1"


def tiny_config() -> dict[str, Any]:
    return {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "vocab_size": 259, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "tie_word_embeddings": True, "max_position_embeddings": 256,
        "torch_dtype": "float32", "bos_token_id": 1, "eos_token_id": 2,
        "attention_dropout": 0.0, "use_cache": False,
        "use_sliding_window": False, "use_mrope": False, "rope_scaling": None,
    }


def deterministic_tensor_bytes(name: str, shape: Sequence[int]) -> bytes:
    """All values are exactly representable binary32; no libm or RNG backend."""
    count = math.prod(shape)
    if name.endswith("layernorm.weight") or name == "model.norm.weight":
        return struct.pack("<f", 1.0) * count
    result = bytearray()
    for block in range((count + 7) // 8):
        words = struct.unpack("<8I", hashlib.sha256(f"{GENERATOR}:{name}:{block}".encode("ascii")).digest())
        for word in words:
            if len(result) == count * 4:
                break
            # Integer numerator + power-of-two denominator freezes every bit.
            denominator = 16777216 if name.endswith(".bias") else 1048576
            result.extend(struct.pack("<f", ((word & 65535) - 32768) / denominator))
    return bytes(result)


def encode_safetensors(tensors: Mapping[str, tuple[Sequence[int], bytes]]) -> bytes:
    header: dict[str, Any] = {"__metadata__": {"format": "pt", "generator": GENERATOR}}
    payload = bytearray()
    for name in sorted(tensors):
        if name == "__metadata__" or not name:
            raise ModelError("INVALID_WEIGHTS", "invalid tensor name")
        shape, data = tensors[name]
        if any(type(dimension) is not int or dimension <= 0 for dimension in shape) or len(data) != math.prod(shape) * 4:
            raise ModelError("INVALID_WEIGHTS", "fixture tensor shape/byte count disagree", field=name)
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [begin, len(payload)]}
    encoded_header = canonical_bytes(header)
    encoded_header += b" " * ((-len(encoded_header)) % 8)
    return struct.pack("<Q", len(encoded_header)) + encoded_header + payload


def generate_fixture(output: Path) -> dict[str, Any]:
    if output.exists() and any(path.name != "generate.py" for path in output.iterdir()):
        raise ModelError("INVALID_ARGUMENT", "fixture output must be empty; existing fixture is immutable")
    output.mkdir(parents=True, exist_ok=True)
    config, tokenizer = tiny_config(), ByteTokenizer()
    provisional = SupportedModelSpec.from_hf_config(config, model_revision="0" * 40,
                                                  tokenizer_revision=tokenizer.revision, runtime_dtype="fp32")
    tensors = {name: (shape, deterministic_tensor_bytes(name, shape)) for name, shape in provisional.tensor_shapes().items()}
    weights = encode_safetensors(tensors)
    model_revision = "sha256:" + digest({"config": config, "weights_sha256": hashlib.sha256(weights).hexdigest(), "generator": GENERATOR})
    spec = SupportedModelSpec.from_hf_config(config, model_revision=model_revision,
                                           tokenizer_revision=tokenizer.revision, runtime_dtype="fp32")
    weight_path = output / "model.safetensors"
    weight_path.write_bytes(weights)
    artifacts = {"config.json": config, "tokenizer.json": tokenizer.to_dict(),
                 "supported-model-spec.json": {**spec.to_dict(), "spec_id": spec.spec_id},
                 "weight-load-manifest.json": build_weight_manifest([weight_path], spec)}
    for name, value in artifacts.items():
        (output / name).write_bytes(canonical_bytes(value) + b"\n")
    index: dict[str, Any] = {"schema_version": 1, "generator": GENERATOR, "model_revision": model_revision,
                             "tokenizer_revision": tokenizer.revision, "spec_id": spec.spec_id,
                             "files": {path.name: _file_digest(path) for path in sorted(output.iterdir()) if path.name != "generate.py"}}
    index["manifest_digest"] = digest(index)
    (output / "fixture-manifest.json").write_bytes(canonical_bytes(index) + b"\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_fixture(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

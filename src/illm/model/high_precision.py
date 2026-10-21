"""Independent CPU binary64 oracle for the immutable tiny Qwen2 fixture.

This module deliberately does not import an iLLM model/runtime implementation.
It verifies the original fixture identities and safetensors layout before loading,
then evaluates the decoder equations directly with CPU ``torch.float64`` tensor
primitives.  It is a mathematical-semantics oracle, not a low-precision runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, NoReturn, Sequence, cast

import torch
from torch import Tensor


TINY_FIXTURE_MANIFEST_DIGEST = "4733493e2b57ce218545762f6ef1418dd6b47b5cd5c0cdc40bd1b17d033be38b"
TINY_MODEL_SHA256 = "8eeaeeb38565ca8ac40177f239a8405396a2b6642aef5223b5bf09205ffb056a"
TINY_SPEC_ID = "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873"
TINY_WEIGHT_MANIFEST_FILE_SHA256 = "0fe7168519a1b63ed67d59f1521882db599428a2a83ceba315b9621f96a6063d"
TINY_WEIGHT_MANIFEST_DIGEST = "dd0827b9ed7e4b1e2808e30552602a38cdda4666bf8b1417482b69ce1e5840bd"
TINY_WEIGHTS_DIGEST = "3868a9840d0b7e418e26c6a31d5022fd829e68090fe75d617964eee7ea09ac92"

_FIXTURE_FILES = {
    "config.json", "model.safetensors", "supported-model-spec.json",
    "tokenizer.json", "weight-load-manifest.json",
}
_STORAGE_FIELDS = {
    "canonical_name", "checksum", "final_storage_bytes", "final_storage_checksum",
    "parameter_aliases", "runtime_dtype", "shape", "source_byte_range",
    "source_dtype", "source_shard", "tied_weight_group", "transforms",
}


class HighPrecisionError(ValueError):
    """Fail-closed fixture, input, or numerical-contract error."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


@dataclass(frozen=True, slots=True)
class OpRecord:
    """One canonical operation at one absolute query position."""

    name: str
    position: int
    shape: tuple[int, ...]
    dtype: str
    layer: int | None
    device: str


OpHook = Callable[[OpRecord, Tensor], None]


@dataclass(frozen=True, slots=True)
class _TinySpec:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    spec_id: str


def _fail(code: str, message: str, *, field: str | None = None) -> NoReturn:
    raise HighPrecisionError(code, message, field=field)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("INVALID_FIXTURE", "duplicate JSON field", field=key)
        result[key] = value
    return result


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        _fail("INVALID_FIXTURE", f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        _fail("INVALID_FIXTURE", f"{label} must be a JSON array")
    return cast(list[object], value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        _fail("INVALID_FIXTURE", f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(cast(int | float, value))):
        _fail("INVALID_FIXTURE", f"{label} must be a finite number")
    return float(cast(int | float, value))


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        _fail("INVALID_FIXTURE", f"{label} must be a string")
    return value


def _loads_object(raw: str | bytes, label: str) -> dict[str, object]:
    try:
        parsed: object = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HighPrecisionError("INVALID_FIXTURE", f"{label} is not valid JSON") from exc
    return _object(parsed, label)


def _read_object(path: Path) -> dict[str, object]:
    try:
        return _loads_object(path.read_bytes(), path.name)
    except OSError as exc:
        raise HighPrecisionError("INVALID_FIXTURE", f"cannot read {path.name}") from exc


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HighPrecisionError("INVALID_FIXTURE", "value is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                result.update(block)
    except OSError as exc:
        raise HighPrecisionError("INVALID_FIXTURE", f"cannot hash {path.name}") from exc
    return result.hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _fail("INVALID_FIXTURE", f"{label} fields differ from the frozen contract")


def _parse_spec(value: dict[str, object], fixture: Mapping[str, object]) -> _TinySpec:
    keys = {
        "activation", "architecture", "attention_output_bias", "bos_token_id", "eos_token_id",
        "head_dim", "hidden_size", "intermediate_size", "kv_layout_id", "max_position_embeddings",
        "mlp_bias", "model_revision", "num_attention_heads", "num_kv_heads", "num_layers",
        "qkv_bias", "rms_norm_eps", "rope_scaling", "rope_theta", "runtime_dtype", "schema_version",
        "spec_id", "tie_word_embeddings", "tokenizer_revision", "vocab_size", "weight_dtype",
        "weight_layout_id",
    }
    _require_keys(value, keys, "supported model spec")
    spec_id = _string(value["spec_id"], "spec_id")
    unsigned = {key: item for key, item in value.items() if key != "spec_id"}
    if spec_id != TINY_SPEC_ID or spec_id != _digest(unsigned) or fixture.get("spec_id") != spec_id:
        _fail("INVALID_FIXTURE", "tiny spec identity differs from the immutable authority")
    literal = {
        "activation": "silu", "architecture": "Qwen2ForCausalLM", "attention_output_bias": False,
        "bos_token_id": 1, "eos_token_id": 2, "kv_layout_id": "none-reference-full-context-v1",
        "mlp_bias": False, "qkv_bias": True, "rope_scaling": None, "runtime_dtype": "fp32",
        "schema_version": 1, "tie_word_embeddings": True, "weight_dtype": "fp32",
        "weight_layout_id": "hf-qwen2-out-in-v1",
    }
    if any(value.get(key) != expected for key, expected in literal.items()):
        _fail("INVALID_FIXTURE", "tiny spec literals differ from the binary64 oracle contract")
    if value.get("model_revision") != fixture.get("model_revision") or value.get("tokenizer_revision") != fixture.get("tokenizer_revision"):
        _fail("INVALID_FIXTURE", "tiny spec revisions differ from the fixture manifest")
    spec = _TinySpec(
        vocab_size=_integer(value["vocab_size"], "vocab_size"),
        hidden_size=_integer(value["hidden_size"], "hidden_size"),
        intermediate_size=_integer(value["intermediate_size"], "intermediate_size"),
        num_layers=_integer(value["num_layers"], "num_layers"),
        num_attention_heads=_integer(value["num_attention_heads"], "num_attention_heads"),
        num_kv_heads=_integer(value["num_kv_heads"], "num_kv_heads"),
        head_dim=_integer(value["head_dim"], "head_dim"),
        max_position_embeddings=_integer(value["max_position_embeddings"], "max_position_embeddings"),
        rms_norm_eps=_number(value["rms_norm_eps"], "rms_norm_eps"),
        rope_theta=_number(value["rope_theta"], "rope_theta"),
        spec_id=spec_id,
    )
    if spec != _TinySpec(259, 32, 64, 2, 4, 2, 8, 256, 1e-6, 10000.0, TINY_SPEC_ID):
        _fail("INVALID_FIXTURE", "tiny dimensions or numerical constants changed")
    return spec


def _verify_config(config: dict[str, object]) -> None:
    expected: dict[str, object] = {
        "architectures": ["Qwen2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 1,
        "eos_token_id": 2, "hidden_act": "silu", "hidden_size": 32, "intermediate_size": 64,
        "max_position_embeddings": 256, "model_type": "qwen2", "num_attention_heads": 4,
        "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "rope_scaling": None, "rope_theta": 10000.0, "tie_word_embeddings": True,
        "torch_dtype": "float32", "use_cache": False, "use_mrope": False,
        "use_sliding_window": False, "vocab_size": 259,
    }
    if config != expected:
        _fail("INVALID_FIXTURE", "tiny config differs from the immutable authority")


def _expected_shapes(spec: _TinySpec) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (spec.vocab_size, spec.hidden_size)}
    q_width = spec.num_attention_heads * spec.head_dim
    kv_width = spec.num_kv_heads * spec.head_dim
    for layer in range(spec.num_layers):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.input_layernorm.weight": (spec.hidden_size,),
            f"{prefix}.self_attn.q_proj.weight": (q_width, spec.hidden_size),
            f"{prefix}.self_attn.q_proj.bias": (q_width,),
            f"{prefix}.self_attn.k_proj.weight": (kv_width, spec.hidden_size),
            f"{prefix}.self_attn.k_proj.bias": (kv_width,),
            f"{prefix}.self_attn.v_proj.weight": (kv_width, spec.hidden_size),
            f"{prefix}.self_attn.v_proj.bias": (kv_width,),
            f"{prefix}.self_attn.o_proj.weight": (spec.hidden_size, q_width),
            f"{prefix}.post_attention_layernorm.weight": (spec.hidden_size,),
            f"{prefix}.mlp.gate_proj.weight": (spec.intermediate_size, spec.hidden_size),
            f"{prefix}.mlp.up_proj.weight": (spec.intermediate_size, spec.hidden_size),
            f"{prefix}.mlp.down_proj.weight": (spec.hidden_size, spec.intermediate_size),
        })
    shapes["model.norm.weight"] = (spec.hidden_size,)
    return shapes


def _parse_weight_manifest(value: dict[str, object], spec: _TinySpec,
                           model_sha256: str) -> dict[str, dict[str, object]]:
    keys = {
        "canonical_storage_count", "duplicate_keys", "finalized", "manifest_digest", "missing_keys",
        "parameter_count", "runtime_dtype", "schema_version", "source_shards", "spec_id", "storages",
        "unexpected_keys", "weights_digest",
    }
    _require_keys(value, keys, "weight manifest")
    claimed = _string(value["manifest_digest"], "weight manifest digest")
    unsigned = {key: item for key, item in value.items() if key != "manifest_digest"}
    if claimed != TINY_WEIGHT_MANIFEST_DIGEST or claimed != _digest(unsigned):
        _fail("INVALID_FIXTURE", "weight manifest digest differs from the immutable authority")
    if (value["schema_version"] != 1 or value["spec_id"] != spec.spec_id
            or value["runtime_dtype"] != "fp32" or value["finalized"] is not True
            or value["missing_keys"] != [] or value["unexpected_keys"] != [] or value["duplicate_keys"] != []):
        _fail("INVALID_FIXTURE", "weight manifest status is not the finalized tiny FP32 contract")
    shards = _array(value["source_shards"], "source_shards")
    if shards != [{"shard": "model.safetensors", "sha256": model_sha256}]:
        _fail("INVALID_FIXTURE", "weight manifest source shard identity differs")

    shapes = _expected_shapes(spec)
    raw_storages = _array(value["storages"], "storages")
    if _integer(value["canonical_storage_count"], "canonical_storage_count") != len(shapes):
        _fail("INVALID_FIXTURE", "canonical storage count differs")
    storages: dict[str, dict[str, object]] = {}
    for raw in raw_storages:
        item = _object(raw, "storage")
        _require_keys(item, _STORAGE_FIELDS, "storage")
        name = _string(item["canonical_name"], "canonical_name")
        if name in storages or name not in shapes:
            _fail("INVALID_FIXTURE", "weight storage name is duplicate or unexpected", field=name)
        shape = tuple(_integer(dimension, f"{name}.shape") for dimension in _array(item["shape"], f"{name}.shape"))
        aliases = [alias for alias in _array(item["parameter_aliases"], f"{name}.parameter_aliases")
                   if isinstance(alias, str)]
        expected_aliases = [name, "lm_head.weight"] if name == "model.embed_tokens.weight" else [name]
        expected_group: str | None = "embed-lm-head-v1" if name == "model.embed_tokens.weight" else None
        byte_count = math.prod(shapes[name]) * 4
        if (shape != shapes[name] or item["source_dtype"] != "F32" or item["runtime_dtype"] != "fp32"
                or item["source_shard"] != "model.safetensors" or aliases != expected_aliases
                or len(aliases) != len(_array(item["parameter_aliases"], f"{name}.parameter_aliases"))
                or item["tied_weight_group"] != expected_group or item["transforms"] != []
                or item["final_storage_bytes"] != byte_count or item["final_storage_checksum"] != item["checksum"]
                or not _is_digest(item["checksum"])):
            _fail("INVALID_FIXTURE", "weight storage metadata differs", field=name)
        storages[name] = item
    if list(storages) != sorted(shapes) or set(storages) != set(shapes):
        _fail("INVALID_FIXTURE", "weight storage ordering or coverage differs")
    parameter_count = sum(math.prod(shape) for shape in shapes.values())
    if _integer(value["parameter_count"], "parameter_count") != parameter_count:
        _fail("INVALID_FIXTURE", "parameter count differs")
    weights_identity = [
        {"name": name, "shape": list(shapes[name]), "source_dtype": "F32", "checksum": storages[name]["checksum"]}
        for name in sorted(shapes)
    ]
    if value["weights_digest"] != TINY_WEIGHTS_DIGEST or value["weights_digest"] != _digest(weights_identity):
        _fail("INVALID_FIXTURE", "weights digest differs from the immutable authority")
    return storages


def _verify_safetensors(raw: bytes, shapes: Mapping[str, tuple[int, ...]],
                        storages: Mapping[str, Mapping[str, object]]) -> None:
    if len(raw) < 10:
        _fail("INVALID_FIXTURE", "safetensors file is truncated")
    header_size = struct.unpack("<Q", raw[:8])[0]
    payload_start = 8 + header_size
    if header_size < 2 or header_size > 16 * 1024 * 1024 or payload_start > len(raw):
        _fail("INVALID_FIXTURE", "safetensors header length is invalid")
    header = _loads_object(raw[8:payload_start], "safetensors header")
    metadata = _object(header.pop("__metadata__", {}), "safetensors metadata")
    if metadata != {"format": "pt", "generator": "illm-tiny-sha256-binary32-v1"} or set(header) != set(shapes):
        _fail("INVALID_FIXTURE", "safetensors metadata or tensor coverage differs")
    ranges: list[tuple[int, int]] = []
    for name, shape in shapes.items():
        tensor = _object(header[name], f"{name} header")
        _require_keys(tensor, {"dtype", "shape", "data_offsets"}, f"{name} header")
        recorded_shape = tuple(_integer(dimension, f"{name}.shape") for dimension in _array(tensor["shape"], f"{name}.shape"))
        offsets = [_integer(offset, f"{name}.data_offsets") for offset in _array(tensor["data_offsets"], f"{name}.data_offsets")]
        if len(offsets) != 2:
            _fail("INVALID_FIXTURE", "safetensors offsets must contain begin/end", field=name)
        begin, end = offsets
        source_range = _array(storages[name]["source_byte_range"], f"{name}.source_byte_range")
        absolute = [payload_start + begin, payload_start + end]
        if (tensor["dtype"] != "F32" or recorded_shape != shape or begin < 0 or end <= begin
                or end - begin != math.prod(shape) * 4 or absolute != source_range or absolute[1] > len(raw)
                or hashlib.sha256(raw[absolute[0]:absolute[1]]).hexdigest() != storages[name]["checksum"]):
            _fail("INVALID_FIXTURE", "safetensors tensor bytes disagree with the frozen manifest", field=name)
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            _fail("INVALID_FIXTURE", "safetensors payload has an overlap or hole")
        cursor = end
    if cursor != len(raw) - payload_start:
        _fail("INVALID_FIXTURE", "safetensors payload is not covered exactly")


class HighPrecisionOracle:
    """Formula-only binary64 decoder for the frozen tiny fixture."""

    _spec: _TinySpec
    _weights: Mapping[str, Tensor]
    _identity: Mapping[str, str]

    def __init__(self, spec: _TinySpec, weights: Mapping[str, Tensor], identity: Mapping[str, str]) -> None:
        self._spec = spec
        self._weights = MappingProxyType(dict(weights))
        self._identity = MappingProxyType(dict(identity))

    @classmethod
    def from_fixture(cls, project_root: Path) -> HighPrecisionOracle:
        """Verify the immutable tiny authority, then load and promote weights once."""
        try:
            root = Path(project_root).resolve(strict=True)
            directory = (root / "fixtures" / "tiny-decoder").resolve(strict=True)
        except OSError as exc:
            raise HighPrecisionError("INVALID_FIXTURE", "tiny fixture directory does not exist") from exc
        if not root.is_dir() or not directory.is_dir() or not directory.is_relative_to(root):
            _fail("INVALID_FIXTURE", "tiny fixture path escapes the project root")
        fixture = _read_object(directory / "fixture-manifest.json")
        _require_keys(fixture, {"files", "generator", "manifest_digest", "model_revision", "schema_version",
                                "spec_id", "tokenizer_revision"}, "fixture manifest")
        claimed = _string(fixture["manifest_digest"], "fixture manifest digest")
        unsigned = {key: item for key, item in fixture.items() if key != "manifest_digest"}
        if (claimed != TINY_FIXTURE_MANIFEST_DIGEST or claimed != _digest(unsigned)
                or fixture["schema_version"] != 1 or fixture["generator"] != "illm-tiny-sha256-binary32-v1"):
            _fail("INVALID_FIXTURE", "fixture manifest differs from the immutable authority")
        files = _object(fixture["files"], "fixture files")
        if set(files) != _FIXTURE_FILES:
            _fail("INVALID_FIXTURE", "fixture file inventory differs")
        for name, expected in files.items():
            path = directory / name
            if not path.is_file() or path.is_symlink() or not _is_digest(expected) or _file_digest(path) != expected:
                _fail("INVALID_FIXTURE", "fixture file checksum differs", field=name)
        if files["model.safetensors"] != TINY_MODEL_SHA256 or files["weight-load-manifest.json"] != TINY_WEIGHT_MANIFEST_FILE_SHA256:
            _fail("INVALID_FIXTURE", "tiny weight artifacts differ from the immutable authority")

        _verify_config(_read_object(directory / "config.json"))
        spec = _parse_spec(_read_object(directory / "supported-model-spec.json"), fixture)
        model_path = directory / "model.safetensors"
        model_sha256 = _file_digest(model_path)
        manifest = _read_object(directory / "weight-load-manifest.json")
        storages = _parse_weight_manifest(manifest, spec, model_sha256)
        try:
            raw_model = model_path.read_bytes()
        except OSError as exc:
            raise HighPrecisionError("INVALID_FIXTURE", "cannot read model.safetensors") from exc
        if hashlib.sha256(raw_model).hexdigest() != model_sha256:
            _fail("INVALID_FIXTURE", "model.safetensors changed after fixture verification")
        shapes = _expected_shapes(spec)
        _verify_safetensors(raw_model, shapes, storages)

        # Import and load the exact verified bytes only after all identities,
        # header fields, ranges and per-storage checksums pass.
        from safetensors.torch import load

        loaded = load(raw_model)
        if set(loaded) != set(shapes):
            _fail("INVALID_FIXTURE", "loaded safetensors coverage differs")
        weights: dict[str, Tensor] = {}
        for name, shape in shapes.items():
            tensor = loaded[name]
            if (tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or not tensor.is_contiguous()
                    or not bool(torch.isfinite(tensor).all().item())):
                _fail("INVALID_FIXTURE", "loaded tensor differs from the tiny FP32 contract", field=name)
            promoted = tensor.detach().to(dtype=torch.float64).contiguous().clone()
            if promoted.dtype != torch.float64 or not bool(torch.isfinite(promoted).all().item()):
                _fail("NUMERICAL_ERROR", "binary64 weight promotion failed", field=name)
            weights[name] = promoted
        identity = {
            "fixture_manifest_digest": claimed, "model_sha256": model_sha256, "spec_id": spec.spec_id,
            "weight_manifest_digest": _string(manifest["manifest_digest"], "weight manifest digest"),
            "weights_digest": _string(manifest["weights_digest"], "weights digest"),
        }
        return cls(spec, weights, identity)

    @property
    def identity(self) -> Mapping[str, str]:
        return self._identity

    @property
    def spec_id(self) -> str:
        return self._spec.spec_id

    def canonical_weight(self, name: str) -> Tensor:
        """Return an isolated FP64 copy for formula diagnostics and fault tests."""
        return self._weight(name).detach().clone()

    def _weight(self, name: str) -> Tensor:
        canonical = "model.embed_tokens.weight" if name == "lm_head.weight" else name
        try:
            tensor = self._weights[canonical]
        except KeyError as exc:
            raise HighPrecisionError("INVALID_FIXTURE", "unknown canonical weight", field=name) from exc
        if tensor.dtype != torch.float64 or tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all().item()):
            _fail("NUMERICAL_ERROR", "oracle weight is not finite CPU binary64", field=name)
        return tensor

    def _validate_value(self, name: str, tensor: Tensor) -> None:
        if tensor.dtype != torch.float64 or tensor.device.type != "cpu" or tensor.layout != torch.strided:
            _fail("NUMERICAL_ERROR", "oracle op left CPU binary64", field=name)
        if not bool(torch.isfinite(tensor).all().item()):
            _fail("NUMERICAL_ERROR", "oracle op produced a non-finite value", field=name)

    def _emit(self, name: str, position: int, tensor: Tensor, layer: int | None,
              hook: OpHook | None) -> Tensor:
        self._validate_value(name, tensor)
        if hook is not None:
            record = OpRecord(name, position, tuple(tensor.shape), str(tensor.dtype), layer, str(tensor.device))
            hook(record, tensor.detach().clone())
        return tensor

    def _emit_rows(self, name: str, tensor: Tensor, positions: Sequence[int], layer: int | None,
                   hook: OpHook | None) -> Tensor:
        self._validate_value(name, tensor)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != len(positions):
            _fail("NUMERICAL_ERROR", "canonical hidden op has a non-row shape", field=name)
        if hook is not None:
            for index, position in enumerate(positions):
                self._emit(name, position, tensor[:, index, :], layer, hook)
        return tensor

    def _emit_heads(self, name: str, tensor: Tensor, positions: Sequence[int], layer: int,
                    hook: OpHook | None) -> Tensor:
        self._validate_value(name, tensor)
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2] != len(positions):
            _fail("NUMERICAL_ERROR", "canonical head op has a non-row shape", field=name)
        if hook is not None:
            for index, position in enumerate(positions):
                self._emit(name, position, tensor[:, :, index:index + 1, :], layer, hook)
        return tensor

    def _linear(self, hidden: Tensor, name: str, *, bias: bool = False) -> Tensor:
        weight = self._weight(name + ".weight")
        result = torch.matmul(hidden, weight.transpose(0, 1))
        if bias:
            result = result + self._weight(name + ".bias")
        return result

    def _norm(self, hidden: Tensor, name: str) -> Tensor:
        variance = (hidden * hidden).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self._spec.rms_norm_eps) * self._weight(name)

    def _rope_fp64(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        indices = torch.arange(0, self._spec.head_dim, 2, dtype=torch.float64, device="cpu")
        inverse = 1.0 / torch.pow(torch.tensor(self._spec.rope_theta, dtype=torch.float64),
                                  indices / float(self._spec.head_dim))
        frequencies = positions.to(dtype=torch.float64).unsqueeze(-1) * inverse
        angles = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        half = self._spec.head_dim // 2
        q_rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        k_rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        return query * cosine + q_rotated * sine, key * cosine + k_rotated * sine

    @torch.inference_mode()
    def rope(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the high-05 formula primitive from fixed FP32 Q/K inputs."""
        length = query.shape[2] if query.ndim == 4 else -1
        if (query.dtype != torch.float32 or key.dtype != torch.float32
                or query.device.type != "cpu" or key.device.type != "cpu"
                or tuple(query.shape) != (1, self._spec.num_attention_heads, length, self._spec.head_dim)
                or tuple(key.shape) != (1, self._spec.num_kv_heads, length, self._spec.head_dim)
                or position_ids.dtype != torch.int64 or position_ids.device.type != "cpu"
                or tuple(position_ids.shape) != (1, length) or not 1 <= length <= self._spec.max_position_embeddings):
            _fail("INVALID_ARGUMENT", "RoPE formula requires CPU FP32 Q/K and int64 positions with frozen tiny shapes")
        if bool(((position_ids < 0) | (position_ids >= self._spec.max_position_embeddings)).any().item()):
            _fail("INVALID_ARGUMENT", "RoPE position is outside the tiny contract")
        if length > 1 and not bool((position_ids[:, 1:] == position_ids[:, :-1] + 1).all().item()):
            _fail("INVALID_ARGUMENT", "RoPE positions must be contiguous")
        result = self._rope_fp64(query.to(dtype=torch.float64), key.to(dtype=torch.float64), position_ids)
        self._validate_value("rope_q", result[0])
        self._validate_value("rope_k", result[1])
        return result

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, position_ids: Tensor | None = None,
                op_hook: OpHook | None = None) -> Tensor:
        """Evaluate a position-zero tiny sequence with independent FP64 formulas."""
        if (input_ids.dtype != torch.int64 or input_ids.device.type != "cpu"
                or input_ids.ndim != 2 or input_ids.shape[0] != 1):
            _fail("INVALID_ARGUMENT", "input_ids must be CPU int64 [1,T]")
        length = input_ids.shape[1]
        if not 1 <= length <= self._spec.max_position_embeddings:
            _fail("INVALID_ARGUMENT", "context length is outside the tiny contract")
        if bool(((input_ids < 0) | (input_ids >= self._spec.vocab_size)).any().item()):
            _fail("INVALID_ARGUMENT", "input token is outside the tiny vocabulary")
        canonical_positions = torch.arange(length, dtype=torch.int64, device="cpu").unsqueeze(0)
        if position_ids is None:
            position_ids = canonical_positions
        if (position_ids.dtype != torch.int64 or position_ids.device.type != "cpu"
                or tuple(position_ids.shape) != (1, length) or not torch.equal(position_ids, canonical_positions)):
            _fail("INVALID_ARGUMENT", "full-model D-high positions must start at zero; use rope() for high-05")
        positions = [int(value) for value in position_ids[0].tolist()]

        embedding = self._weight("model.embed_tokens.weight")
        hidden = self._emit_rows("embedding", embedding[input_ids], positions, None, op_hook)
        groups = self._spec.num_attention_heads // self._spec.num_kv_heads
        scale = float(self._spec.head_dim) ** -0.5
        for layer in range(self._spec.num_layers):
            prefix = f"model.layers.{layer}"
            op = f"layer.{layer}"
            residual = hidden
            norm = self._emit_rows(f"{op}.input_norm", self._norm(hidden, prefix + ".input_layernorm.weight"),
                                   positions, layer, op_hook)
            query = self._linear(norm, prefix + ".self_attn.q_proj", bias=True).reshape(
                1, length, self._spec.num_attention_heads, self._spec.head_dim).permute(0, 2, 1, 3).contiguous()
            key = self._linear(norm, prefix + ".self_attn.k_proj", bias=True).reshape(
                1, length, self._spec.num_kv_heads, self._spec.head_dim).permute(0, 2, 1, 3).contiguous()
            value = self._linear(norm, prefix + ".self_attn.v_proj", bias=True).reshape(
                1, length, self._spec.num_kv_heads, self._spec.head_dim).permute(0, 2, 1, 3).contiguous()
            self._emit_heads(f"{op}.q_projection", query, positions, layer, op_hook)
            self._emit_heads(f"{op}.k_projection", key, positions, layer, op_hook)
            self._emit_heads(f"{op}.v_projection", value, positions, layer, op_hook)
            query, key = self._rope_fp64(query, key, position_ids)
            self._emit_heads(f"{op}.rope_q", query, positions, layer, op_hook)
            self._emit_heads(f"{op}.rope_k", key, positions, layer, op_hook)

            context_rows: list[Tensor] = []
            for row, position in enumerate(positions):
                score_heads: list[Tensor] = []
                probability_heads: list[Tensor] = []
                context_heads: list[Tensor] = []
                for head in range(self._spec.num_attention_heads):
                    kv_head = head // groups
                    q_row = query[:, head, row, :].unsqueeze(1)
                    k_prefix = key[:, kv_head, :row + 1, :]
                    v_prefix = value[:, kv_head, :row + 1, :]
                    scores = torch.matmul(q_row, k_prefix.transpose(-2, -1)) * scale
                    shifted = scores - scores.amax(dim=-1, keepdim=True)
                    exponentials = torch.exp(shifted)
                    probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
                    context = torch.matmul(probabilities, v_prefix)
                    score_heads.append(scores)
                    probability_heads.append(probabilities)
                    context_heads.append(context)
                ragged_scores = torch.cat(score_heads, dim=1).unsqueeze(2)
                ragged_probabilities = torch.cat(probability_heads, dim=1).unsqueeze(2)
                ragged_context = torch.cat(context_heads, dim=1).unsqueeze(2)
                self._emit(f"{op}.attention_scores", position, ragged_scores, layer, op_hook)
                self._emit(f"{op}.attention_probabilities", position, ragged_probabilities, layer, op_hook)
                self._emit(f"{op}.attention_context", position, ragged_context, layer, op_hook)
                context_rows.append(ragged_context)
            context_all = torch.cat(context_rows, dim=2)
            merged = context_all.permute(0, 2, 1, 3).contiguous().reshape(1, length, self._spec.hidden_size)
            attention = self._emit_rows(f"{op}.attention_output",
                                        self._linear(merged, prefix + ".self_attn.o_proj"),
                                        positions, layer, op_hook)
            hidden = self._emit_rows(f"{op}.attention_residual", residual + attention,
                                     positions, layer, op_hook)
            residual = hidden
            norm = self._emit_rows(f"{op}.post_attention_norm",
                                   self._norm(hidden, prefix + ".post_attention_layernorm.weight"),
                                   positions, layer, op_hook)
            gate = self._emit_rows(f"{op}.mlp_gate", self._linear(norm, prefix + ".mlp.gate_proj"),
                                   positions, layer, op_hook)
            up = self._emit_rows(f"{op}.mlp_up", self._linear(norm, prefix + ".mlp.up_proj"),
                                 positions, layer, op_hook)
            activated = self._emit_rows(f"{op}.mlp_silu", gate * torch.sigmoid(gate),
                                        positions, layer, op_hook)
            product = self._emit_rows(f"{op}.mlp_product", activated * up, positions, layer, op_hook)
            mlp = self._emit_rows(f"{op}.mlp_output", self._linear(product, prefix + ".mlp.down_proj"),
                                  positions, layer, op_hook)
            hidden = self._emit_rows(f"{op}.mlp_residual", residual + mlp, positions, layer, op_hook)
        hidden = self._emit_rows("final_norm", self._norm(hidden, "model.norm.weight"),
                                 positions, None, op_hook)
        logits = self._linear(hidden, "lm_head")
        return self._emit_rows("logits", logits, positions, None, op_hook)


__all__ = ["HighPrecisionError", "HighPrecisionOracle", "OpHook", "OpRecord"]

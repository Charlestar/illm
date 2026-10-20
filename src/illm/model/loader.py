"""Fail-closed safetensors header/storage preflight, independent of torch."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from illm.model.spec import ModelError, SupportedModelSpec, digest

if TYPE_CHECKING:
    from illm.model.reference import ReferenceDecoder
    from illm.tokenizer.reference import TokenizerAdapter

DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2}
DTYPE_CONTRACT = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}
PINNED_REAL_ARTIFACTS: dict[str, dict[str, object]] = {
    "config.json": {"path": "config.json", "bytes": 681, "git_blob_sha1": "141506237b9022c4ab3e01734cfafb9310b57d2d"},
    "model.safetensors": {"path": "model.safetensors", "bytes": 988097824, "sha256": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"},
    "tokenizer.json": {"path": "tokenizer.json", "bytes": 7031645, "git_blob_sha1": "443909a61d429dff23010e5bddd28ff530edda00"},
    "tokenizer_config.json": {"path": "tokenizer_config.json", "bytes": 7228, "git_blob_sha1": "ba7e4c5637b9732dadcd66286ce48334e8b31e9e"},
}


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelError("DUPLICATE_WEIGHT", f"duplicate JSON field/tensor {key}", field=key)
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class TensorStorage:
    canonical_name: str
    source_shard: str
    source_byte_start: int
    source_byte_end: int
    shape: tuple[int, ...]
    dtype: str
    checksum: str
    storage_bytes: int
    parameter_aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name, "source_shard": self.source_shard,
            "source_byte_range": [self.source_byte_start, self.source_byte_end],
            "shape": list(self.shape), "source_dtype": self.dtype,
            "checksum": self.checksum, "final_storage_bytes": self.storage_bytes,
            "final_storage_checksum": self.checksum,
            "transforms": [], "parameter_aliases": list(self.parameter_aliases),
            "tied_weight_group": "embed-lm-head-v1" if len(self.parameter_aliases) > 1 else None,
        }


def inspect_safetensors(path: Path) -> dict[str, TensorStorage]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise ModelError("INVALID_WEIGHTS", "safetensors length prefix is truncated")
        header_size = struct.unpack("<Q", encoded_length)[0]
        if header_size < 2 or header_size > 16 * 1024 * 1024 or 8 + header_size > file_size:
            raise ModelError("INVALID_WEIGHTS", "safetensors header length is invalid")
        encoded_header = handle.read(header_size)
        if not encoded_header.startswith(b"{"):
            raise ModelError("INVALID_WEIGHTS", "safetensors header must begin with object")
        try:
            header = json.loads(encoded_header, object_pairs_hook=_unique_object)
        except (ValueError, UnicodeDecodeError) as exc:
            if isinstance(exc, ModelError):
                raise
            raise ModelError("INVALID_WEIGHTS", "safetensors header JSON is invalid") from exc
        if not isinstance(header, dict):
            raise ModelError("INVALID_WEIGHTS", "safetensors header must be an object")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values()):
            raise ModelError("INVALID_WEIGHTS", "safetensors metadata must be string-valued")
        payload_start = 8 + header_size
        records: dict[str, TensorStorage] = {}
        ranges: list[tuple[int, int, str]] = []
        for name, item in header.items():
            if not isinstance(name, str) or not name or not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
                raise ModelError("INVALID_WEIGHTS", "tensor header fields differ from safetensors contract", field=name)
            shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
            if not isinstance(shape, list) or not shape or any(type(value) is not int or value <= 0 for value in shape):
                raise ModelError("INVALID_WEIGHTS", "tensor shape is invalid", field=name)
            if not isinstance(dtype, str) or dtype not in DTYPE_SIZE:
                raise ModelError("INVALID_WEIGHTS", "unsupported tensor dtype", field=name)
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(value) is not int or value < 0 for value in offsets):
                raise ModelError("INVALID_WEIGHTS", "tensor offsets are invalid", field=name)
            begin, end = offsets
            if end <= begin or end - begin != math.prod(shape) * DTYPE_SIZE[dtype] or payload_start + end > file_size:
                raise ModelError("INVALID_WEIGHTS", "tensor byte range disagrees with shape/dtype/file", field=name)
            handle.seek(payload_start + begin)
            remaining = end - begin
            checksum = hashlib.sha256()
            while remaining:
                data = handle.read(min(remaining, 1 << 20))
                if not data:
                    raise ModelError("INVALID_WEIGHTS", "tensor data is truncated", field=name)
                checksum.update(data)
                remaining -= len(data)
            ranges.append((begin, end, name))
            records[name] = TensorStorage(name, path.name, payload_start + begin, payload_start + end,
                                          tuple(shape), dtype, checksum.hexdigest(), end - begin, (name,))
        position = 0
        for begin, end, name in sorted(ranges):
            if begin != position:
                raise ModelError("INVALID_WEIGHTS", "tensor storage has overlap or uncovered hole", field=name)
            position = end
        if position != file_size - payload_start:
            raise ModelError("INVALID_WEIGHTS", "tensor storage does not cover file payload exactly")
        return records


def build_weight_manifest(paths: Sequence[Path], spec: SupportedModelSpec) -> dict[str, Any]:
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ModelError("INVALID_WEIGHTS", "shard list is empty or contains duplicate names")
    expected = spec.tensor_shapes()
    combined: dict[str, TensorStorage] = {}
    sources: list[dict[str, str]] = []
    for path in sorted(paths):
        records = inspect_safetensors(path)
        for name, record in records.items():
            if name in combined:
                raise ModelError("DUPLICATE_WEIGHT", "tensor appears in more than one shard", field=name)
            combined[name] = record
        sources.append({"shard": path.name, "sha256": _file_digest(path)})
    missing, unexpected = set(expected) - set(combined), set(combined) - set(expected)
    if missing or unexpected:
        raise ModelError("INVALID_WEIGHTS", f"tensor coverage differs: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    storages = []
    for name in sorted(expected):
        record = combined[name]
        if record.shape != expected[name] or DTYPE_CONTRACT[record.dtype] != spec.weight_dtype:
            raise ModelError("INVALID_WEIGHTS", "tensor shape/dtype disagrees with SupportedModelSpec", field=name)
        item = record.to_dict()
        if name == "model.embed_tokens.weight" and spec.tie_word_embeddings:
            item["parameter_aliases"] = [name, "lm_head.weight"]
            item["tied_weight_group"] = "embed-lm-head-v1"
        item["runtime_dtype"] = spec.runtime_dtype
        item["final_storage_bytes"] = math.prod(record.shape) * ({"fp32": 4, "fp16": 2, "bf16": 2}[spec.runtime_dtype])
        item["transforms"] = [] if spec.weight_dtype == spec.runtime_dtype else [{"op": "cast", "from": spec.weight_dtype, "to": spec.runtime_dtype}]
        if spec.weight_dtype != spec.runtime_dtype:
            # Never claim an unexecuted conversion's source digest is its output.
            item["final_storage_checksum"] = None
        storages.append(item)
    weights_digest = digest([{"name": record["canonical_name"], "shape": record["shape"],
                              "source_dtype": record["source_dtype"], "checksum": record["checksum"]}
                             for record in storages])
    result: dict[str, Any] = {
        "schema_version": 1, "spec_id": spec.spec_id, "weights_digest": weights_digest,
        "finalized": spec.weight_dtype == spec.runtime_dtype,
        "runtime_dtype": spec.runtime_dtype, "source_shards": sources, "storages": storages,
        "missing_keys": [], "unexpected_keys": [], "duplicate_keys": [],
        "canonical_storage_count": len(storages),
        "parameter_count": spec.parameter_count,
    }
    result["manifest_digest"] = digest(result)
    return result


@dataclass(frozen=True, slots=True)
class ModelSource:
    spec: SupportedModelSpec
    assets_directory: Path
    weight_paths: tuple[Path, ...]
    tokenizer_kind: str
    fixture_id: str | None
    max_context_tokens: int
    minimum_gpu_memory_bytes: int = 0
    minimum_compute_capability: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    spec: SupportedModelSpec
    weight_manifest: dict[str, Any]
    reference: ReferenceDecoder
    tokenizer: TokenizerAdapter
    assets_directory: Path
    fixture_id: str | None
    max_context_tokens: int


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError("INVALID_MODEL_CONFIG", f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ModelError("INVALID_MODEL_CONFIG", f"JSON artifact must be an object: {path}")
    return value


def _asset_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ModelError("INVALID_MODEL_CONFIG", "asset name must be one local file basename")
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ModelError("MODEL_ARTIFACT_MISSING", f"required local model artifact is missing: {path}", field=name)
    return path


def _verify_fixture(root: Path) -> ModelSource:
    index = _json_file(_asset_path(root, "fixture-manifest.json"))
    claimed = index.get("manifest_digest")
    if claimed != digest({key: value for key, value in index.items() if key != "manifest_digest"}):
        raise ModelError("INVALID_WEIGHTS", "fixture manifest digest mismatch")
    files = index.get("files")
    required = {"config.json", "tokenizer.json", "supported-model-spec.json", "weight-load-manifest.json", "model.safetensors"}
    if not isinstance(files, dict) or set(files) != required:
        raise ModelError("INVALID_WEIGHTS", "fixture manifest does not cover exact required assets")
    for name, expected in files.items():
        if _file_digest(_asset_path(root, name)) != expected:
            raise ModelError("INVALID_WEIGHTS", "fixture artifact checksum mismatch", field=name)
    spec = SupportedModelSpec.from_hf_config(_json_file(root / "config.json"),
                                           model_revision=index["model_revision"],
                                           tokenizer_revision=index["tokenizer_revision"], runtime_dtype="fp32")
    if spec.spec_id != index.get("spec_id"):
        raise ModelError("INVALID_MODEL_CONFIG", "fixture spec identity mismatch")
    stored_spec = _json_file(root / "supported-model-spec.json")
    if stored_spec != {**spec.to_dict(), "spec_id": spec.spec_id}:
        raise ModelError("INVALID_MODEL_CONFIG", "fixture normalized spec disagrees with raw config")
    from illm.tokenizer.reference import ByteTokenizer

    tokenizer = ByteTokenizer(**_json_file(root / "tokenizer.json"))
    if tokenizer.revision != spec.tokenizer_revision:
        raise ModelError("INVALID_MODEL_CONFIG", "fixture tokenizer revision mismatch")
    return ModelSource(spec, root, (root / "model.safetensors",), "byte-v1", "tiny-decoder", spec.max_position_embeddings)


def _project_root(project_root: Path | None) -> Path:
    return (project_root if project_root is not None else Path.cwd()).resolve()


def resolve_model(model: str | Path = "fixture:tiny-decoder", *, project_root: Path | None = None) -> ModelSource:
    project = _project_root(project_root)
    if str(model).startswith("fixture:"):
        if str(model) != "fixture:tiny-decoder":
            raise ModelError("UNSUPPORTED_MODEL", "only fixture:tiny-decoder is supported")
        candidates = [project / "fixtures" / "tiny-decoder",
                      Path(__file__).resolve().parents[3] / "fixtures" / "tiny-decoder",
                      Path(sys.prefix) / "share" / "illm" / "fixtures" / "tiny-decoder"]
        root = next((candidate for candidate in candidates if (candidate / "fixture-manifest.json").is_file()), None)
        if root is None:
            raise ModelError("MODEL_ARTIFACT_MISSING", "packaged tiny-decoder fixture is missing")
        return _verify_fixture(root.resolve())
    path = Path(model)
    path = path.resolve() if path.is_absolute() else (project / path).resolve()
    profile = _json_file(path)
    fields = {"schema_version", "model_id", "model_revision", "tokenizer_revision", "assets_directory",
              "runtime_dtype", "max_context_tokens", "minimum_gpu_memory_bytes", "minimum_compute_capability",
              "tokenizer_kind", "artifacts"}
    if set(profile) != fields or type(profile.get("schema_version")) is not int or profile.get("schema_version") != 1:
        raise ModelError("INVALID_MODEL_CONFIG", "model profile fields/schema differ from v1 contract")
    revision = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    if profile["model_id"] != "Qwen/Qwen2.5-0.5B" or profile["model_revision"] != revision or profile["tokenizer_revision"] != revision:
        raise ModelError("UNSUPPORTED_MODEL", "only the pinned Qwen2.5-0.5B real-model revision is supported")
    if profile["tokenizer_kind"] != "hf-tokenizers-json":
        raise ModelError("UNSUPPORTED_MODEL", "unsupported tokenizer adapter")
    memory = profile["minimum_gpu_memory_bytes"]
    if type(memory) is not int or memory < 6 * 1024 ** 3:
        raise ModelError("INVALID_MODEL_CONFIG", "real model requires an integer minimum GPU memory of at least 6 GiB",
                         field="minimum_gpu_memory_bytes")
    capability = profile["minimum_compute_capability"]
    if (not isinstance(capability, list) or len(capability) != 2
            or any(type(value) is not int or value < 0 for value in capability)
            or tuple(capability) < (8, 0)):
        raise ModelError("INVALID_MODEL_CONFIG", "real model requires a nonnegative [major, minor] minimum capability >= [8, 0]",
                         field="minimum_compute_capability")
    directory = profile["assets_directory"]
    if not isinstance(directory, str) or not directory:
        raise ModelError("INVALID_MODEL_CONFIG", "assets_directory must be a nonempty path")
    root = (project / directory).resolve()
    artifacts = profile["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelError("INVALID_MODEL_CONFIG", "model profile must pin artifact hashes")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) not in ({"path", "bytes", "sha256"}, {"path", "bytes", "git_blob_sha1"}):
            raise ModelError("INVALID_MODEL_CONFIG", "artifact lock must contain exact path/bytes/hash fields")
        name = artifact["path"]
        if not isinstance(name, str) or name in seen:
            raise ModelError("INVALID_MODEL_CONFIG", "artifact paths must be unique strings")
        if artifact != PINNED_REAL_ARTIFACTS.get(name):
            raise ModelError("INVALID_MODEL_CONFIG", "artifact identity is not from the exact pinned real-model revision", field=name)
        seen.add(name)
        asset = _asset_path(root, name)
        if type(artifact["bytes"]) is not int or asset.stat().st_size != artifact["bytes"]:
            raise ModelError("INVALID_WEIGHTS", "locked artifact byte size mismatch", field=name)
        if "sha256" in artifact:
            actual = _file_digest(asset)
            expected_hash = artifact["sha256"]
        else:
            raw = asset.read_bytes()
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw, usedforsecurity=False).hexdigest()
            expected_hash = artifact["git_blob_sha1"]
        if actual != expected_hash:
            raise ModelError("INVALID_WEIGHTS", "locked artifact checksum mismatch", field=name)
    if seen != {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}:
        raise ModelError("INVALID_MODEL_CONFIG", "real fixture artifact set differs from pinned profile")
    spec = SupportedModelSpec.from_hf_config(_json_file(root / "config.json"), model_revision=revision,
                                           tokenizer_revision=revision, runtime_dtype=profile["runtime_dtype"])
    context = profile["max_context_tokens"]
    if type(context) is not int or not 1 <= context <= spec.max_position_embeddings:
        raise ModelError("INVALID_MODEL_CONFIG", "max_context_tokens exceeds model position contract")
    return ModelSource(spec, root, (root / "model.safetensors",), "hf-tokenizers-json", None, context,
                       memory, (capability[0], capability[1]))


def inspect_model(model: str | Path = "fixture:tiny-decoder", *, project_root: Path | None = None) -> dict[str, Any]:
    source = resolve_model(model, project_root=project_root)
    weights = build_weight_manifest(source.weight_paths, source.spec)
    if source.fixture_id and weights != _json_file(source.assets_directory / "weight-load-manifest.json"):
        raise ModelError("INVALID_WEIGHTS", "computed fixture storage manifest differs from frozen artifact")
    return {"spec": {**source.spec.to_dict(), "spec_id": source.spec.spec_id}, "weight_manifest": weights,
            "parameter_count": source.spec.parameter_count, "canonical_tensor_shapes": {name: list(shape) for name, shape in source.spec.tensor_shapes().items()},
            "assets_directory": str(source.assets_directory), "fixture_id": source.fixture_id,
            "max_context_tokens": source.max_context_tokens,
            "hardware_requirements": {"minimum_gpu_memory_bytes": source.minimum_gpu_memory_bytes,
                                      "minimum_compute_capability": list(source.minimum_compute_capability),
                                      "policy": "checked_before_cuda_allocation; cpu_debug_does_not_qualify_gpu",
                                      "evidence_kind": "recorded"},
            "evidence_kind": "recorded"}


def _preflight_device(source: ModelSource, device: str) -> None:
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
        raise ModelError("INVALID_ARGUMENT", "device must be cpu, cuda or cuda:<index>")
    if device == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise ModelError("UNSUPPORTED_MODEL", "requested CUDA device is unavailable; no CPU fallback performed", field="device")
    index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ModelError("INVALID_ARGUMENT", "requested CUDA device index does not exist", field="device")
    actual_capability = torch.cuda.get_device_capability(index)
    if actual_capability < source.minimum_compute_capability:
        raise ModelError("UNSUPPORTED_MODEL", "CUDA compute capability is below model profile requirement", field="minimum_compute_capability")
    if source.spec.runtime_dtype == "bf16" and actual_capability < (8, 0):
        raise ModelError("UNSUPPORTED_MODEL", "BF16 requires native Ampere-or-newer CUDA capability", field="runtime_dtype")
    properties = torch.cuda.get_device_properties(index)
    if properties.total_memory < source.minimum_gpu_memory_bytes:
        raise ModelError("RESOURCE_EXHAUSTED", "CUDA total device memory is below model profile minimum; no allocation attempted",
                         field="minimum_gpu_memory_bytes")


def load_model(model: str | Path = "fixture:tiny-decoder", *, device: str = "cpu", project_root: Path | None = None) -> LoadedModel:
    source = resolve_model(model, project_root=project_root)
    _preflight_device(source, device)
    manifest = build_weight_manifest(source.weight_paths, source.spec)
    if source.fixture_id and manifest != _json_file(source.assets_directory / "weight-load-manifest.json"):
        raise ModelError("INVALID_WEIGHTS", "computed fixture storage manifest differs from frozen artifact")
    from illm.model.reference import load_reference
    from illm.tokenizer.reference import ByteTokenizer, LocalTokenizer

    reference = load_reference(source.spec, source.weight_paths, device=device)
    tokenizer: TokenizerAdapter
    if source.tokenizer_kind == "byte-v1":
        tokenizer = ByteTokenizer(**_json_file(source.assets_directory / "tokenizer.json"))
    else:
        tokenizer = LocalTokenizer(source.assets_directory / "tokenizer.json", revision=source.spec.tokenizer_revision,
                                   vocab_size=source.spec.vocab_size, bos_token_id=source.spec.bos_token_id)
    # Converted tensors need actual materialized checksums rather than source hashes.
    if not manifest["finalized"]:
        import torch

        for storage in manifest["storages"]:
            tensor = reference.canonical_weight(storage["canonical_name"])
            raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            storage["final_storage_checksum"] = hashlib.sha256(raw).hexdigest()
        manifest["finalized"] = True
        manifest.pop("manifest_digest")
        manifest["manifest_digest"] = digest(manifest)
    return LoadedModel(source.spec, manifest, reference, tokenizer, source.assets_directory,
                       source.fixture_id, source.max_context_tokens)

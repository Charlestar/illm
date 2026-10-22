"""Actual CPU producers for the V0.2 numerical-policy raw evidence.

The module is deliberately import-safe: importing it does not import Torch,
the model package, or the candidate decoder.  A native worker and a candidate
worker are intended to run in different fresh processes.  The output is raw,
DEVELOPMENT_ONLY evidence; completing a worker never grants qualification.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any, Iterator, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json


REPORT_KIND = "illm-policy-cpu-raw-worker-v1"
EVIDENCE_STAGE = "DEVELOPMENT_ONLY"
QUALIFICATION_STATUS = "unqualified"
_ROLES = frozenset({"native", "candidate", "candidate-reproduction"})
_REPARSE = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


class RuntimeWorkerError(RuntimeError):
    """The runtime worker cannot produce a complete truthful result."""


def _fail(message: str) -> NoReturn:
    raise RuntimeWorkerError(message)


def _is_linklike(path: Path) -> bool:
    try:
        information = path.lstat()
    except OSError as exc:
        raise RuntimeWorkerError(f"cannot inspect path: {path}") from exc
    return stat.S_ISLNK(information.st_mode) or bool(
        int(getattr(information, "st_file_attributes", 0)) & _REPARSE
    )


def _plain_existing_root(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail("project root cannot contain links or reparse points")
    if not lexical.is_dir() or not (lexical / "src" / "illm").is_dir():
        _fail("project root must be an existing iLLM source workspace")
    return lexical.resolve()


def _new_output_root(path: Path, project_root: Path, *, test_subset: bool) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    for candidate in (lexical, *lexical.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail("output path cannot contain links or reparse points")
    if os.path.lexists(lexical):
        _fail("output root must not already exist")
    if not lexical.parent.is_dir() or _is_linklike(lexical.parent):
        _fail("output parent must be an existing ordinary directory")
    if not test_subset:
        evidence_parent = project_root / "out"
        try:
            lexical.relative_to(evidence_parent)
        except ValueError:
            _fail("public worker output must be a new descendant of project_root/out")
    try:
        lexical.mkdir()
    except OSError as exc:
        raise RuntimeWorkerError("cannot create exclusive output root") from exc
    return lexical


def _exclusive_write(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise RuntimeWorkerError(f"cannot write {path.name!r} exclusively") from exc


@contextmanager
def _cpu_environment() -> Iterator[None]:
    names = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _measured_flags(torch: Any) -> dict[str, object]:
    return {
        "autocast_cpu": bool(torch.is_autocast_enabled("cpu")),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "cpu_threads": int(torch.get_num_threads()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "compile": False,
    }


def _cuda_initialized(torch: Any) -> bool:
    inspect_state = cast(Callable[[], bool], torch.cuda.is_initialized)
    return bool(inspect_state())


@contextmanager
def _native_cpu_flags(torch: Any) -> Iterator[dict[str, object]]:
    """Apply ADR-005 CPU flags without importing the candidate implementation."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    previous_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    previous_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        with torch.autocast(device_type="cpu", enabled=False):
            yield _measured_flags(torch)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_fp16
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_bf16
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
        torch.set_float32_matmul_precision(previous_precision)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)


def _tensor_identifier(role: str, case_id: str, step: int, op: str, position: int | None) -> str:
    from illm.policy_cpu_numerical_contract import tensor_id

    return tensor_id(role, case_id, step, op, position)


def _tensor_bytes(tensor: Any) -> tuple[str, list[int], bytes]:
    import torch

    if sys.byteorder != "little":
        _fail("raw tensor evidence requires a little-endian host")
    if getattr(getattr(tensor, "device", None), "type", None) != "cpu":
        _fail("CPU runtime worker cannot serialize a non-CPU tensor")
    dtype_name = str(getattr(tensor, "dtype", ""))
    dtype = {"torch.float32": "fp32", "torch.float64": "fp64"}.get(dtype_name)
    if dtype is None:
        _fail("raw tensor evidence only accepts native FP32/FP64 tensors")
    value = tensor.detach().contiguous()
    shape = [int(dimension) for dimension in value.shape]
    if not shape or any(dimension < 1 for dimension in shape):
        _fail("raw tensor evidence rejects empty/scalar tensors")
    raw = cast(bytes, value.view(torch.uint8).numpy().tobytes())
    # A dtype-specific itemsize check prevents a future tensor runtime from
    # silently converting rather than exposing the underlying bytes.
    expected = value.numel() * value.element_size()
    if len(raw) != expected:
        _fail("tensor byte view changed precision or length")
    return dtype, shape, raw


def _write_tensor(writer: Any, tensor_id: str, tensor: Any) -> str:
    dtype, shape, raw = _tensor_bytes(tensor)
    writer.write_tensor(tensor_id, dtype, shape, raw)
    return tensor_id


def _state_record(state: Any) -> dict[str, object]:
    return {"digest": state.digest, "value": state.to_dict()}


def _zero_resources() -> dict[str, object]:
    return {
        "closed": True,
        "active_leases": 0,
        "allocated_bytes": 0,
        "valid_tokens": 0,
        "staged_tokens": 0,
    }


def _store_resources(store: Any) -> dict[str, object]:
    return {
        "closed": True,
        "active_leases": int(store.active_leases),
        "allocated_bytes": int(store.allocated_bytes),
        "valid_tokens": int(store.valid_tokens),
        "staged_tokens": int(store.staged_tokens),
    }


def _case_projection(case: Mapping[str, object]) -> dict[str, object]:
    required = {
        "case_id",
        "input_token_ids",
        "max_new_tokens",
        "sampling_spec",
        "input_digest",
        "sampling_spec_digest",
        "suite_id",
        "expected_output_token_ids",
        "case_digest",
    }
    if set(case) != required:
        _fail("CPU case differs from the closed worker input contract")
    return {name: case[name] for name in sorted(required)}


def _validate_trace_names(trace: Mapping[str, object]) -> None:
    from illm.policy_cpu_numerical_contract import canonical_tiny_ops

    expected = canonical_tiny_ops()
    if len(expected) != 39 or set(trace) != set(expected):
        _fail("runtime trace differs from the canonical 39-op inventory")


def _native_trace(
    *,
    writer: Any,
    role: str,
    case_id: str,
    step: int,
    forward: Callable[[Callable[[Any, Any], None]], Any],
) -> tuple[Any, dict[str, str]]:
    identifiers: dict[str, str] = {}

    def hook(record: Any, tensor: Any) -> None:
        if record.name in identifiers:
            _fail("native forward emitted a duplicate op")
        identifier = _tensor_identifier(role, case_id, step, str(record.name), None)
        identifiers[str(record.name)] = _write_tensor(writer, identifier, tensor)

    logits = forward(hook)
    _validate_trace_names(identifiers)
    return logits, identifiers


def _candidate_trace(
    *,
    writer: Any,
    role: str,
    case_id: str,
    step: int,
    torch: Any,
    forward: Callable[[Callable[[Any, Any], None]], Any],
) -> tuple[Any, dict[str, dict[str, str]], dict[str, object]]:
    identifiers: dict[str, dict[str, str]] = {}
    observed_flags: dict[str, object] | None = None

    def hook(record: Any, tensor: Any) -> None:
        nonlocal observed_flags
        position = int(record.position)
        positions = identifiers.setdefault(str(record.name), {})
        key = str(position)
        if key in positions:
            _fail("candidate forward emitted a duplicate op position")
        identifier = _tensor_identifier(role, case_id, step, str(record.name), position)
        positions[key] = _write_tensor(writer, identifier, tensor)
        flags = _measured_flags(torch)
        if observed_flags is None:
            observed_flags = flags
        elif observed_flags != flags:
            _fail("candidate execution flags changed during a forward")

    logits = forward(hook)
    _validate_trace_names(identifiers)
    if observed_flags is None:
        _fail("candidate forward did not expose measured execution flags")
    return logits, identifiers, observed_flags


def _sample_logits(logits: Any) -> tuple[float, ...]:
    row = logits[0, -1]
    return tuple(float(value) for value in row.tolist())


def _run_native_path(
    *,
    path_id: str,
    case: Mapping[str, object],
    loaded: Any,
    oracle: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    from illm.model.validation import collect_transformers_ops, oracle_forward_logits
    from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample

    case_id = cast(str, case["case_id"])
    context = tuple(cast(list[int], case["input_token_ids"]))
    maximum = cast(int, case["max_new_tokens"])
    sampling = SamplingSpec.from_dict(cast(Mapping[str, object], case["sampling_spec"]))
    state = SamplingState.for_request(case_id, sampling)
    output: list[int] = []
    steps: list[dict[str, object]] = []
    flags_record: dict[str, object] | None = None
    for step in range(maximum):
        state_before = _state_record(state)
        inputs = torch.tensor([context], dtype=torch.int64)
        with _native_cpu_flags(torch) as flags:
            if flags_record is None:
                flags_record = flags
            elif flags_record != flags:
                _fail("native execution flags changed between steps")
            if path_id == "A":
                logits, op_ids = _native_trace(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    forward=lambda hook: loaded.reference.forward(inputs, op_hook=hook),
                )
                ordinary_logits_id: str | None = None
            elif path_id == "D-native":
                from illm.policy_cpu_numerical_contract import canonical_tiny_ops

                trace = collect_transformers_ops(oracle, inputs, loaded)

                def replay(hook: Callable[[Any, Any], None]) -> Any:
                    # Adapter tensors are already complete Transformers-owned
                    # facts; feed them through the same closed raw hook.
                    from illm.model.reference import OpRecord

                    for name in canonical_tiny_ops():
                        tensor = trace[name]
                        layer = int(name.split(".")[1]) if name.startswith("layer.") else None
                        hook(OpRecord(name, tuple(tensor.shape), str(tensor.dtype), layer, str(tensor.device)), tensor)
                    return trace["logits"]

                logits, op_ids = _native_trace(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    forward=replay,
                )
                ordinary = oracle_forward_logits(
                    oracle,
                    inputs,
                    vocab_size=loaded.spec.vocab_size,
                )
                if not torch.equal(logits, ordinary):
                    _fail("D-native adapter logits differ bitwise from ordinary Transformers forward")
                ordinary_identifier = _tensor_identifier(
                    path_id, case_id, step, "ordinary_logits", None
                )
                ordinary_logits_id = _write_tensor(writer, ordinary_identifier, ordinary)
            else:  # pragma: no cover - closed caller
                _fail("unknown native path")
        logits_id = op_ids["logits"]
        raw_logits = _sample_logits(logits)
        decision = prepare_sample(raw_logits, sampling, state)
        state = commit_sample(state, decision, spec=sampling, logits=raw_logits)
        token = int(decision.token_id)
        output.append(token)
        steps.append(
            {
                "step": step,
                "context": list(context),
                "output_token_id": token,
                "logits_id": logits_id,
                "state_before": state_before,
                "state_after": _state_record(state),
                "query_tokens": len(context),
                "kv_before": 0,
                "kv_after": 0,
                "op_ids": op_ids,
                "prefix_ids": {},
                "ordinary_logits_id": ordinary_logits_id,
                "fence_complete": None,
            }
        )
        context = (*context, token)
        if token == loaded.spec.eos_token_id:
            break
    finish = "EOS" if output and output[-1] == loaded.spec.eos_token_id else "MAX_NEW_TOKENS"
    if flags_record is None:
        _fail("native path produced no steps")
    return {
        "path_id": path_id,
        "execution_flags": flags_record,
        "steps": steps,
        "output_token_ids": output,
        "finish_reason": finish,
        "resource_status": _zero_resources(),
    }


def _prefix_ids(
    *,
    writer: Any,
    role: str,
    case_id: str,
    step: int,
    lease: Any,
    num_layers: int,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if lease.start == 0:
        return result
    for layer in range(num_layers):
        key, value = lease.prefix(layer)
        key_id = _tensor_identifier(role, case_id, step, f"prefix.layer.{layer}.key", None)
        value_id = _tensor_identifier(role, case_id, step, f"prefix.layer.{layer}.value", None)
        result[str(layer)] = {
            "key": _write_tensor(writer, key_id, key),
            "value": _write_tensor(writer, value_id, value),
        }
    return result


def _run_candidate_path(
    *,
    path_id: str,
    case: Mapping[str, object],
    decoder: Any,
    loaded: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    from illm.executor.fence import ImmediateFence
    from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
    from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample

    case_id = cast(str, case["case_id"])
    context = tuple(cast(list[int], case["input_token_ids"]))
    maximum = cast(int, case["max_new_tokens"])
    sampling = SamplingSpec.from_dict(cast(Mapping[str, object], case["sampling_spec"]))
    state = SamplingState.for_request(case_id, sampling)
    output: list[int] = []
    steps: list[dict[str, object]] = []
    flags_record: dict[str, object] | None = None
    store: Any | None = None
    if path_id == "C":
        store = ContiguousKVStore(
            KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256)
        )
    try:
        for step in range(maximum):
            state_before = _state_record(state)
            if path_id == "B":
                query = context
                logits, op_ids, flags = _candidate_trace(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    torch=torch,
                    forward=lambda hook: decoder.forward(
                        torch.tensor([context], dtype=torch.int64), op_hook=hook
                    ),
                )
                prefix: dict[str, dict[str, str]] = {}
                kv_before = kv_after = 0
                fence_complete: bool | None = None
            elif path_id == "C" and store is not None:
                query = context[store.valid_tokens :]
                kv_before = int(store.valid_tokens)
                lease = store.begin_append(query)
                prefix = _prefix_ids(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    lease=lease,
                    num_layers=loaded.spec.num_layers,
                )

                logits, op_ids, flags = _candidate_trace(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    torch=torch,
                    forward=lambda hook: decoder.forward_incremental(
                        torch.tensor([query], dtype=torch.int64), lease, op_hook=hook
                    ),
                )
                fence = ImmediateFence()
                kv_after = int(lease.commit(fence))
                fence_complete = fence.is_complete()
                if fence_complete is not True or fence.succeeded is not True:
                    _fail("candidate CPU append did not complete its immediate fence")
            else:  # pragma: no cover - closed caller
                _fail("unknown candidate path")
            if flags_record is None:
                flags_record = flags
            elif flags_record != flags:
                _fail("candidate flags changed between steps")
            full_logits_id = _tensor_identifier(path_id, case_id, step, "full_logits", None)
            logits_id = _write_tensor(writer, full_logits_id, logits)
            raw_logits = _sample_logits(logits)
            decision = prepare_sample(raw_logits, sampling, state)
            state = commit_sample(state, decision, spec=sampling, logits=raw_logits)
            token = int(decision.token_id)
            output.append(token)
            steps.append(
                {
                    "step": step,
                    "context": list(context),
                    "output_token_id": token,
                    "logits_id": logits_id,
                    "state_before": state_before,
                    "state_after": _state_record(state),
                    "query_tokens": len(query),
                    "kv_before": kv_before,
                    "kv_after": kv_after,
                    "op_ids": op_ids,
                    "prefix_ids": prefix,
                    "ordinary_logits_id": None,
                    "fence_complete": fence_complete,
                }
            )
            context = (*context, token)
            if token == loaded.spec.eos_token_id:
                break
    finally:
        if store is not None:
            store.close()
    if store is not None:
        resources = _store_resources(store)
        if resources != _zero_resources():
            _fail("candidate C resources did not return to zero")
    else:
        resources = _zero_resources()
    finish = "EOS" if output and output[-1] == loaded.spec.eos_token_id else "MAX_NEW_TOKENS"
    if flags_record is None:
        _fail("candidate path produced no steps")
    return {
        "path_id": path_id,
        "execution_flags": flags_record,
        "steps": steps,
        "output_token_ids": output,
        "finish_reason": finish,
        "resource_status": resources,
    }


def _build_independent_transformers_oracle(
    project_root: Path,
    torch: Any,
) -> tuple[Any, Any, dict[str, object]]:
    """Build D-native from frozen bytes without observing A/B/C parameters."""

    from safetensors.torch import load_file

    from illm.model.loader import build_weight_manifest, resolve_model
    from illm.policy_cpu_numerical_contract import read_json

    transformers = importlib.import_module("transformers")
    if getattr(transformers, "__version__", None) != "4.57.1":
        _fail("D-native contract pins Transformers 4.57.1")
    source = resolve_model("fixture:tiny-decoder", project_root=project_root)
    expected_root = project_root / "fixtures" / "tiny-decoder"
    if source.fixture_id != "tiny-decoder" or source.assets_directory != expected_root:
        _fail("D-native did not resolve the project-owned frozen tiny fixture")
    if source.spec.weight_dtype != "fp32" or source.spec.runtime_dtype != "fp32":
        _fail("D-native requires the frozen FP32 tiny model")
    if _is_linklike(source.assets_directory):
        _fail("D-native asset root cannot be a link or reparse point")

    asset_names = (
        "config.json",
        "model.safetensors",
        "supported-model-spec.json",
        "weight-load-manifest.json",
        "tokenizer.json",
        "fixture-manifest.json",
    )
    asset_files: dict[str, str] = {}
    for name in asset_names:
        path = source.assets_directory / name
        if _is_linklike(path) or not path.is_file():
            _fail("D-native asset must be one ordinary frozen file")
        asset_files[name] = sha256_file(path)

    fixture = read_json(source.assets_directory / "fixture-manifest.json")
    if not isinstance(fixture, dict):
        _fail("D-native fixture manifest must be an object")
    fixture_digest = fixture.get("manifest_digest")
    if type(fixture_digest) is not str or fixture_digest != sha256_json(
        {name: value for name, value in fixture.items() if name != "manifest_digest"}
    ):
        _fail("D-native fixture manifest self-identity differs")

    fresh_manifest = build_weight_manifest(source.weight_paths, source.spec)
    stored_manifest = read_json(source.assets_directory / "weight-load-manifest.json")
    if not isinstance(stored_manifest, dict) or canonical_json_bytes(fresh_manifest) != canonical_json_bytes(
        stored_manifest
    ):
        _fail("D-native fresh storage inspection differs from the frozen load manifest")

    config_data = read_json(source.assets_directory / "config.json")
    if not isinstance(config_data, dict):
        _fail("D-native config must be an object")
    config_factory = transformers.Qwen2Config
    model_factory = transformers.Qwen2ForCausalLM._from_config
    config = config_factory(**config_data)
    config._attn_implementation = "eager"
    config.use_cache = False
    with torch.random.fork_rng(devices=[]):
        oracle = model_factory(config, dtype=torch.float32, attn_implementation="eager")

    tensors: dict[str, Any] = {}
    source_shards = fresh_manifest.get("source_shards")
    if not isinstance(source_shards, list):
        _fail("D-native load manifest source shards are invalid")
    for shard in source_shards:
        if not isinstance(shard, dict) or set(shard) != {"shard", "sha256"}:
            _fail("D-native load manifest shard identity is invalid")
        name, expected_digest = shard["shard"], shard["sha256"]
        if type(name) is not str or Path(name).name != name or type(expected_digest) is not str:
            _fail("D-native load manifest shard identity is invalid")
        path = source.assets_directory / name
        if _is_linklike(path) or sha256_file(path) != expected_digest:
            _fail("D-native source shard differs from the fresh load identity")
        values = load_file(str(path), device="cpu")
        if set(tensors) & set(values):
            _fail("D-native source shards repeat tensor storage")
        tensors.update(values)
    if set(tensors) != set(source.spec.tensor_shapes()):
        _fail("D-native source tensor coverage differs from the frozen spec")
    if source.spec.tie_word_embeddings:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    oracle.load_state_dict(tensors, strict=True)
    oracle = oracle.to(device="cpu")
    parameter_dtypes = {str(parameter.dtype) for parameter in oracle.parameters()}
    parameter_devices = {str(parameter.device) for parameter in oracle.parameters()}
    if parameter_dtypes != {"torch.float32"} or parameter_devices != {"cpu"}:
        _fail("D-native parameters are not independently materialized as CPU FP32")
    rope_dtype = str(oracle.model.rotary_emb.inv_freq.dtype)
    if rope_dtype != "torch.float32":
        _fail("D-native RoPE inverse frequencies must remain FP32")
    oracle.eval()

    modeling_qwen2 = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    module_path_value = getattr(modeling_qwen2, "__file__", None)
    if type(module_path_value) is not str:
        _fail("D-native Transformers source module has no file identity")
    module_path = Path(module_path_value)
    if not module_path.is_file():
        _fail("D-native Transformers source module is not a file")
    identity: dict[str, object] = {
        "kind": "independent-pinned-transformers-oracle-v1",
        "transformers_version": "4.57.1",
        "model_class": f"{oracle.__class__.__module__}.{oracle.__class__.__qualname__}",
        "source": {
            "module": "transformers.models.qwen2.modeling_qwen2",
            "sha256": sha256_file(module_path),
        },
        "device": "cpu",
        "parameter_dtype": "torch.float32",
        "rope_inv_freq_dtype": rope_dtype,
        "attention_implementation": "eager",
        "use_cache": False,
        "asset_files": asset_files,
        "fixture_manifest_digest": fixture_digest,
        "spec_id": source.spec.spec_id,
        "weight_manifest_digest": fresh_manifest["manifest_digest"],
        "weights_digest": fresh_manifest["weights_digest"],
        "source_shards": source_shards,
    }
    return oracle, source, identity


def _run_native_cases(
    cases: Sequence[Mapping[str, object]], writer: Any
) -> tuple[list[dict[str, object]], Any, dict[str, object]]:
    if "illm.model.policy" in sys.modules:
        _fail("native worker refuses a process that imported the candidate decoder")
    import torch
    from illm.model import load_model

    loaded = load_model(project_root=_ACTIVE_PROJECT_ROOT)
    oracle, oracle_source, oracle_identity = _build_independent_transformers_oracle(
        _ACTIVE_PROJECT_ROOT, torch
    )
    records: list[dict[str, object]] = []
    for case in cases:
        record = _case_projection(case)
        record["paths"] = {
            "A": _run_native_path(
                path_id="A", case=case, loaded=loaded, oracle=oracle, writer=writer, torch=torch
            ),
            "D-native": _run_native_path(
                path_id="D-native",
                case=case,
                loaded=oracle_source,
                oracle=oracle,
                writer=writer,
                torch=torch,
            ),
        }
        records.append(record)
    if "illm.model.policy" in sys.modules:
        _fail("native worker imported the candidate decoder")
    return records, loaded, oracle_identity


def _run_candidate_cases(
    cases: Sequence[Mapping[str, object]], writer: Any
) -> tuple[list[dict[str, object]], Any, Any, Any]:
    import torch
    from illm.model import load_model
    from illm.model.policy import PolicyDecoder
    from illm.numerical_policy import candidate_policy

    loaded = load_model(project_root=_ACTIVE_PROJECT_ROOT)
    policy = candidate_policy("fp32")
    b = PolicyDecoder(loaded.reference, policy)
    c = PolicyDecoder(loaded.reference, policy)
    records: list[dict[str, object]] = []
    for case in cases:
        record = _case_projection(case)
        record["paths"] = {
            "B": _run_candidate_path(
                path_id="B", case=case, decoder=b, loaded=loaded, writer=writer, torch=torch
            ),
            "C": _run_candidate_path(
                path_id="C", case=case, decoder=c, loaded=loaded, writer=writer, torch=torch
            ),
        }
        records.append(record)
    return records, loaded, policy, (b, c)


def _high_full_path(
    *,
    path_id: str,
    high: Mapping[str, object],
    decoder: Any,
    loaded: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    from illm.executor.fence import ImmediateFence
    from illm.kv_cache import ContiguousKVStore, KVLayoutSpec

    case_id = cast(str, high["case_id"])
    tokens = tuple(cast(Sequence[int], high["tokens"]))
    steps: list[dict[str, object]] = []
    flags_record: dict[str, object] | None = None
    if path_id == "B-high":
        logits, op_ids, flags = _candidate_trace(
            writer=writer,
            role=path_id,
            case_id=case_id,
            step=0,
            torch=torch,
            forward=lambda hook: decoder.forward(
                torch.tensor([tokens], dtype=torch.int64), op_hook=hook
            ),
        )
        logits_id = _write_tensor(
            writer, _tensor_identifier(path_id, case_id, 0, "full_logits", None), logits
        )
        steps.append(
            {
                "step": 0,
                "context": list(tokens),
                "output_token_id": None,
                "logits_id": logits_id,
                "state_before": None,
                "state_after": None,
                "query_tokens": len(tokens),
                "kv_before": 0,
                "kv_after": 0,
                "op_ids": op_ids,
                "prefix_ids": {},
                "ordinary_logits_id": None,
                "fence_complete": None,
            }
        )
        flags_record = flags
        resources = _zero_resources()
        full_logits_id = logits_id
    elif path_id == "C-high":
        store = ContiguousKVStore(
            KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256)
        )
        rows: list[Any] = []
        try:
            for step, token in enumerate(tokens):
                lease = store.begin_append((int(token),))
                prefix = _prefix_ids(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    lease=lease,
                    num_layers=loaded.spec.num_layers,
                )

                def incremental(hook: Callable[[Any, Any], None]) -> Any:
                    return decoder.forward_incremental(
                        torch.tensor([[int(token)]], dtype=torch.int64), lease, op_hook=hook
                    )

                logits, op_ids, flags = _candidate_trace(
                    writer=writer,
                    role=path_id,
                    case_id=case_id,
                    step=step,
                    torch=torch,
                    forward=incremental,
                )
                if flags_record is None:
                    flags_record = flags
                elif flags_record != flags:
                    _fail("C-high flags changed between positions")
                row_id = _write_tensor(
                    writer,
                    _tensor_identifier(path_id, case_id, step, "full_logits", None),
                    logits,
                )
                fence = ImmediateFence()
                before = int(store.valid_tokens)
                after = int(lease.commit(fence))
                if fence.is_complete() is not True or fence.succeeded is not True:
                    _fail("C-high immediate fence did not complete")
                rows.append(logits)
                steps.append(
                    {
                        "step": step,
                        "context": list(tokens[: step + 1]),
                        "output_token_id": None,
                        "logits_id": row_id,
                        "state_before": None,
                        "state_after": None,
                        "query_tokens": 1,
                        "kv_before": before,
                        "kv_after": after,
                        "op_ids": op_ids,
                        "prefix_ids": prefix,
                        "ordinary_logits_id": None,
                        "fence_complete": True,
                    }
                )
        finally:
            store.close()
        resources = _store_resources(store)
        if resources != _zero_resources():
            _fail("C-high resources did not return to zero")
        combined = torch.cat(rows, dim=1).contiguous()
        full_logits_id = _write_tensor(
            writer,
            _tensor_identifier(path_id, case_id, 0, "concatenated_logits", None),
            combined,
        )
    else:  # pragma: no cover - closed caller
        _fail("unknown high path")
    if flags_record is None:
        _fail("high path did not record execution flags")
    return {
        "path_id": path_id,
        "execution_flags": flags_record,
        "steps": steps,
        "output_token_ids": [],
        "finish_reason": "NOT_APPLICABLE",
        "resource_status": resources,
        "full_logits_id": full_logits_id,
    }


def _high_rope_path(
    *,
    path_id: str,
    high: Mapping[str, object],
    decoder: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    case_id = cast(str, high["case_id"])
    positions = tuple(cast(Sequence[int], high["positions"]))
    if positions != (31, 32):
        _fail("rope-only high case must use positions 31 and 32")
    query = torch.empty((1, 4, 2, 8), dtype=torch.float32)
    key = torch.empty((1, 2, 2, 8), dtype=torch.float32)
    for head in range(4):
        for offset in range(2):
            for dimension in range(8):
                query[0, head, offset, dimension] = (2 * head + dimension - 9) / 16
    for head in range(2):
        for offset in range(2):
            for dimension in range(8):
                key[0, head, offset, dimension] = (3 * head - dimension + 5) / 32
    with torch.inference_mode(), _native_cpu_flags(torch) as flags:
        query_rows: list[Any] = []
        key_rows: list[Any] = []
        for offset, position in enumerate(positions):
            q_result, k_result = decoder._rope_position(
                query[:, :, offset, :].contiguous(),
                key[:, :, offset, :].contiguous(),
                int(position),
            )
            query_rows.append(q_result.unsqueeze(2))
            key_rows.append(k_result.unsqueeze(2))
    outputs = {
        "input_q": query,
        "input_k": key,
        "rope_q": torch.cat(query_rows, dim=2).contiguous(),
        "rope_k": torch.cat(key_rows, dim=2).contiguous(),
    }
    op_ids: dict[str, dict[str, str]] = {}
    for name, tensor in outputs.items():
        identifier = _tensor_identifier(path_id, case_id, 0, name, None)
        op_ids[name] = {"full": _write_tensor(writer, identifier, tensor)}
    step = {
        "step": 0,
        "context": [],
        "output_token_id": None,
        "logits_id": None,
        "state_before": None,
        "state_after": None,
        "query_tokens": 0,
        "kv_before": 0,
        "kv_after": 0,
        "op_ids": op_ids,
        "prefix_ids": {},
        "ordinary_logits_id": None,
        "fence_complete": None,
    }
    return {
        "path_id": path_id,
        "execution_flags": flags,
        "steps": [step],
        "output_token_ids": [],
        "finish_reason": "NOT_APPLICABLE",
        "resource_status": _zero_resources(),
        "full_logits_id": None,
    }


def _run_high_cases(
    high_cases: Sequence[Mapping[str, object]],
    *,
    b: Any,
    c: Any,
    loaded: Any,
    writer: Any,
) -> list[dict[str, object]]:
    import torch

    records: list[dict[str, object]] = []
    for high in high_cases:
        if set(high) != {"case_id", "mode", "tokens", "positions"}:
            _fail("D-high candidate input differs from the closed contract")
        mode = high["mode"]
        if mode == "forward":
            paths = {
                "B-high": _high_full_path(
                    path_id="B-high", high=high, decoder=b, loaded=loaded, writer=writer, torch=torch
                ),
                "C-high": _high_full_path(
                    path_id="C-high", high=high, decoder=c, loaded=loaded, writer=writer, torch=torch
                ),
            }
        elif mode == "rope_only":
            paths = {
                "B-high": _high_rope_path(
                    path_id="B-high", high=high, decoder=b, writer=writer, torch=torch
                ),
                "C-high": _high_rope_path(
                    path_id="C-high", high=high, decoder=c, writer=writer, torch=torch
                ),
            }
        else:
            _fail("unknown D-high candidate mode")
        records.append(
            {
                "case_id": high["case_id"],
                "mode": mode,
                "input_token_ids": list(cast(Sequence[int], high["tokens"])),
                "positions": list(cast(Sequence[int], high["positions"])),
                "paths": paths,
            }
        )
    return records


def _authority_forward_path(
    *,
    path_id: str,
    case_id: str,
    tokens: tuple[int, ...],
    decoder: Any,
    loaded: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    from illm.executor.fence import ImmediateFence
    from illm.kv_cache import ContiguousKVStore, KVLayoutSpec

    prefix: dict[str, dict[str, str]] = {}
    if path_id == "B-authority":
        logits, op_ids, flags = _candidate_trace(
            writer=writer,
            role=path_id,
            case_id=case_id,
            step=0,
            torch=torch,
            forward=lambda hook: decoder.forward(
                torch.tensor([tokens], dtype=torch.int64), op_hook=hook
            ),
        )
        kv_after = 0
        fence_complete: bool | None = None
        resources = _zero_resources()
    elif path_id == "C-authority":
        store = ContiguousKVStore(
            KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256)
        )
        try:
            lease = store.begin_append(tokens)
            logits, op_ids, flags = _candidate_trace(
                writer=writer,
                role=path_id,
                case_id=case_id,
                step=0,
                torch=torch,
                forward=lambda hook: decoder.forward_incremental(
                    torch.tensor([tokens], dtype=torch.int64), lease, op_hook=hook
                ),
            )
            fence = ImmediateFence()
            kv_after = int(lease.commit(fence))
            fence_complete = fence.is_complete()
            if fence_complete is not True or fence.succeeded is not True:
                _fail("authority prefill did not complete its immediate fence")
        finally:
            store.close()
        resources = _store_resources(store)
        if resources != _zero_resources():
            _fail("authority C resources did not return to zero")
    else:  # pragma: no cover - closed caller
        _fail("unknown authority path")
    full_logits_id = _write_tensor(
        writer,
        _tensor_identifier(path_id, case_id, 0, "full_logits", None),
        logits,
    )
    return {
        "path_id": path_id,
        "execution_flags": flags,
        "steps": [
            {
                "step": 0,
                "context": list(tokens),
                "output_token_id": None,
                "logits_id": full_logits_id,
                "state_before": None,
                "state_after": None,
                "query_tokens": len(tokens),
                "kv_before": 0,
                "kv_after": kv_after,
                "op_ids": op_ids,
                "prefix_ids": prefix,
                "ordinary_logits_id": None,
                "fence_complete": fence_complete,
            }
        ],
        "output_token_ids": [],
        "finish_reason": "NOT_APPLICABLE",
        "resource_status": resources,
    }


def _authority_rope_path(
    *,
    path_id: str,
    case_id: str,
    expected: Mapping[str, Any],
    decoder: Any,
    loaded: Any,
    writer: Any,
    torch: Any,
) -> dict[str, object]:
    op_ids: dict[str, dict[str, str]] = {}
    with torch.inference_mode(), _native_cpu_flags(torch) as flags:
        for layer in range(loaded.spec.num_layers):
            q_name = f"layer.{layer}.q_projection"
            k_name = f"layer.{layer}.k_projection"
            rope_q_name = f"layer.{layer}.rope_q"
            rope_k_name = f"layer.{layer}.rope_k"
            query = expected[q_name]
            key = expected[k_name]
            rope_q, rope_k = decoder._rope_position(query, key, 255)
            for name, tensor in (
                (q_name, query),
                (k_name, key),
                (rope_q_name, rope_q),
                (rope_k_name, rope_k),
            ):
                identifier = _tensor_identifier(path_id, case_id, 0, name, None)
                op_ids[name] = {"full": _write_tensor(writer, identifier, tensor)}
    return {
        "path_id": path_id,
        "execution_flags": flags,
        "steps": [
            {
                "step": 0,
                "context": [258],
                "output_token_id": None,
                "logits_id": None,
                "state_before": None,
                "state_after": None,
                "query_tokens": 0,
                "kv_before": 0,
                "kv_after": 0,
                "op_ids": op_ids,
                "prefix_ids": {},
                "ordinary_logits_id": None,
                "fence_complete": None,
            }
        ],
        "output_token_ids": [],
        "finish_reason": "NOT_APPLICABLE",
        "resource_status": _zero_resources(),
    }


def _run_op_authority_cases(
    *,
    project_root: Path,
    b: Any,
    c: Any,
    loaded: Any,
    writer: Any,
) -> list[dict[str, object]]:
    import torch
    from safetensors.torch import load_file

    from illm.policy_cpu_numerical_contract import (
        TINY_ORACLE_MANIFEST_DIGEST,
        canonical_tiny_ops,
        strict_json,
    )

    directory = project_root / "fixtures" / "tiny-decoder" / "oracle-v1"
    manifest_path = directory / "oracle-manifest.json"
    if _is_linklike(directory) or _is_linklike(manifest_path) or not manifest_path.is_file():
        _fail("frozen op authority path must be an ordinary local tree")
    manifest = strict_json(manifest_path.read_bytes(), label="frozen tiny oracle manifest")
    if not isinstance(manifest, dict):
        _fail("frozen tiny oracle manifest must be an object")
    claimed = manifest.get("manifest_digest")
    if (
        claimed != TINY_ORACLE_MANIFEST_DIGEST
        or claimed
        != sha256_json({key: value for key, value in manifest.items() if key != "manifest_digest"})
    ):
        _fail("frozen tiny oracle manifest identity differs")
    sources = manifest.get("op_cases")
    if not isinstance(sources, list) or len(sources) != 3:
        _fail("frozen op authority must contain exactly three cases")
    expected_ids = ("hello", "vocab-min-position-zero", "vocab-max-position-last")
    records: list[dict[str, object]] = []
    for index, source_value in enumerate(sources):
        if not isinstance(source_value, dict):
            _fail("frozen op authority case must be an object")
        source = cast(dict[str, object], source_value)
        case_id = source.get("fixture_id")
        file_name = source.get("file")
        digest = source.get("sha256")
        tokens = source.get("input_ids")
        position_ids = source.get("position_ids")
        if (
            case_id != expected_ids[index]
            or type(file_name) is not str
            or Path(file_name).name != file_name
            or type(digest) is not str
            or not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int for token in tokens)
            or source.get("op_order") != list(canonical_tiny_ops())
        ):
            _fail("frozen op authority case metadata differs")
        case_id_value = cast(str, case_id)
        source_path = directory / file_name
        if _is_linklike(source_path) or not source_path.is_file() or sha256_file(source_path) != digest:
            _fail("frozen op authority tensor file differs")
        token_tuple = tuple(cast(list[int], tokens))
        if position_ids == [255]:
            if case_id_value != "vocab-max-position-last" or token_tuple != (258,):
                _fail("frozen position-255 authority metadata differs")
            expected = load_file(str(source_path), device="cpu")
            paths = {
                "B-authority": _authority_rope_path(
                    path_id="B-authority",
                    case_id=case_id_value,
                    expected=expected,
                    decoder=b,
                    loaded=loaded,
                    writer=writer,
                    torch=torch,
                ),
                "C-authority": _authority_rope_path(
                    path_id="C-authority",
                    case_id=case_id_value,
                    expected=expected,
                    decoder=c,
                    loaded=loaded,
                    writer=writer,
                    torch=torch,
                ),
            }
            mode = "rope_only"
            positions = [255]
        else:
            if position_ids not in (None, list(range(len(token_tuple)))):
                _fail("frozen full-forward authority positions differ")
            paths = {
                "B-authority": _authority_forward_path(
                    path_id="B-authority",
                    case_id=case_id_value,
                    tokens=token_tuple,
                    decoder=b,
                    loaded=loaded,
                    writer=writer,
                    torch=torch,
                ),
                "C-authority": _authority_forward_path(
                    path_id="C-authority",
                    case_id=case_id_value,
                    tokens=token_tuple,
                    decoder=c,
                    loaded=loaded,
                    writer=writer,
                    torch=torch,
                ),
            }
            mode = "forward"
            positions = list(range(len(token_tuple)))
        records.append(
            {
                "case_id": case_id_value,
                "input_token_ids": list(token_tuple),
                "positions": positions,
                "mode": mode,
                "source_file": source_path.relative_to(project_root).as_posix(),
                "source_sha256": digest,
                "paths": paths,
            }
        )
    return records


def _model_identity(
    loaded: Any,
    *,
    native_oracle: Mapping[str, object] | None,
) -> dict[str, object]:
    assets: dict[str, str] = {}
    if _is_linklike(loaded.assets_directory):
        _fail("model asset root cannot be a link or reparse point")
    pending = [loaded.assets_directory]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if _is_linklike(path):
                _fail("model asset inventory contains a link or reparse point")
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                assets[path.relative_to(loaded.assets_directory).as_posix()] = sha256_file(path)
            else:
                _fail("model asset inventory contains a non-regular entry")
    tokenizer = loaded.tokenizer.to_dict()
    specification = {**loaded.spec.to_dict(), "spec_id": loaded.spec.spec_id}
    return {
        "model_spec": specification,
        "model_spec_id": loaded.spec.spec_id,
        "weight_manifest": loaded.weight_manifest,
        "weight_manifest_digest": loaded.weight_manifest["manifest_digest"],
        "weights_digest": loaded.weight_manifest["weights_digest"],
        "tokenizer": tokenizer,
        "tokenizer_digest": sha256_json(tokenizer),
        "asset_files": assets,
        "native_oracle": dict(native_oracle) if native_oracle is not None else None,
    }


def _coverage(
    role: str,
    cases: Sequence[Mapping[str, object]],
    high_cases: Sequence[Mapping[str, object]],
    *,
    test_subset: bool,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for case in cases:
        suite = cast(str, case["suite_id"])
        counts[suite] = counts.get(suite, 0) + 1
    expected = 20 if role == "candidate-reproduction" else 70
    expected_high = 8 if role == "candidate" else 0
    full = not test_subset and len(cases) == expected and len(high_cases) == expected_high
    if not test_subset and not full:
        _fail("public worker refuses incomplete CPU protocol coverage")
    return {
        "kind": "test-subset" if test_subset else (
            "reproduction20" if role == "candidate-reproduction" else "full70"
        ),
        "case_count": len(cases),
        "expected_case_count": expected,
        "suite_counts": dict(sorted(counts.items())),
        "high_case_count": len(high_cases),
        "expected_high_case_count": expected_high,
        "full_contract": full,
    }


_ACTIVE_PROJECT_ROOT = Path(".")


def _run_runtime_worker(
    project_root: Path,
    output: Path,
    *,
    role: str,
    cases: Sequence[Mapping[str, object]],
    high_case_records: Sequence[Mapping[str, object]],
    test_subset: bool,
) -> dict[str, object]:
    global _ACTIVE_PROJECT_ROOT
    if role not in _ROLES:
        _fail("role must be native, candidate, or candidate-reproduction")
    root = _plain_existing_root(project_root)
    _ACTIVE_PROJECT_ROOT = root
    coverage = _coverage(role, cases, high_case_records, test_subset=test_subset)
    destination = _new_output_root(output, root, test_subset=test_subset)
    _exclusive_write(destination / "INCOMPLETE", b"policy_cpu_runtime_worker_in_progress\n")

    from illm.policy_binding import build_execution_binding
    from illm.policy_tensor_evidence import TensorPackWriter

    binding = build_execution_binding(root)
    binding_value = binding.to_dict()
    toolchain = cast(Mapping[str, object], binding_value["toolchain"])
    if not test_subset and toolchain.get("binding") != "source-bound-manifest":
        _fail("public worker requires a finalized source-bound environment manifest")
    policy_loaded_before = "illm.model.policy" in sys.modules
    writer = TensorPackWriter(destination / "pack")
    with _cpu_environment():
        import torch

        cuda_before = _cuda_initialized(torch)
        if cuda_before:
            _fail("CPU worker refuses a process with initialized CUDA state")
        if role == "native":
            records, loaded, native_oracle = _run_native_cases(cases, writer)
            policy_value: dict[str, object] | None = None
            high_results: list[dict[str, object]] = []
            op_authority_results: list[dict[str, object]] = []
        else:
            records, loaded, policy, decoders = _run_candidate_cases(cases, writer)
            native_oracle = None
            b, c = decoders
            policy_value = {
                "descriptor": policy.to_dict(),
                "digest": policy.digest,
                "sidecar_bytes": {"B": int(b.sidecar_bytes), "C": int(c.sidecar_bytes)},
            }
            high_results = (
                _run_high_cases(high_case_records, b=b, c=c, loaded=loaded, writer=writer)
                if role == "candidate"
                else []
            )
            op_authority_results = (
                _run_op_authority_cases(
                    project_root=root,
                    b=b,
                    c=c,
                    loaded=loaded,
                    writer=writer,
                )
                if role == "candidate"
                else []
            )
        cuda_after = _cuda_initialized(torch)
        if cuda_after:
            _fail("CPU worker initialized CUDA")
    pack_manifest = writer.finalize()
    pack_digest = pack_manifest.get("manifest_digest")
    if type(pack_digest) is not str:
        _fail("tensor pack finalize did not return a manifest digest")
    process = {
        "pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "package_version": importlib.metadata.version("illm"),
        "worker_source_sha256": sha256_file(Path(__file__)),
        "implementation_source_digest": binding_value["implementation_source_digest"],
        "cuda_visible_devices": "",
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "native_policy_module_loaded_before": policy_loaded_before if role == "native" else None,
        "native_policy_module_loaded_after": (
            "illm.model.policy" in sys.modules if role == "native" else None
        ),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": REPORT_KIND,
        "status": "complete",
        "evidence_stage": EVIDENCE_STAGE,
        "qualification_status": QUALIFICATION_STATUS,
        "role": role,
        "coverage": coverage,
        "binding": binding_value,
        "binding_digest": binding.digest,
        "process": process,
        "model": _model_identity(loaded, native_oracle=native_oracle),
        "policy": policy_value,
        "pack_manifest_digest": pack_digest,
        "cases": records,
        "high_cases": high_results,
        "op_authority_cases": op_authority_results,
    }
    encoded = canonical_json_bytes(report)
    _exclusive_write(destination / "worker.json", encoded)
    worker_digest = sha256_bytes(encoded)
    _exclusive_write(destination / "COMPLETE", (worker_digest + "\n").encode("ascii"))
    try:
        (destination / "INCOMPLETE").unlink()
    except OSError as exc:
        try:
            (destination / "COMPLETE").unlink()
        except OSError:
            pass
        raise RuntimeWorkerError("cannot clear worker INCOMPLETE marker") from exc
    return report


def run_runtime_worker(project_root: Path, output: Path, *, role: str) -> dict[str, object]:
    """Run one fixed public worker role in a fresh source-bound CPU process."""

    from illm.policy_cpu_numerical_contract import cpu_cases, high_cases

    all_cases = cpu_cases(project_root)
    if role == "candidate-reproduction":
        selected = [case for case in all_cases if case["expected_output_token_ids"] is not None]
        highs: list[dict[str, object]] = []
    else:
        selected = all_cases
        highs = high_cases() if role == "candidate" else []
    return _run_runtime_worker(
        project_root,
        output,
        role=role,
        cases=selected,
        high_case_records=highs,
        test_subset=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=sorted(_ROLES), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_runtime_worker(args.project_root, args.output, role=args.role)
    except (RuntimeWorkerError, OSError, ValueError) as exc:
        print(
            canonical_json_bytes(
                {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 1
    print(
        canonical_json_bytes(
            {
                "status": report["status"],
                "role": report["role"],
                "coverage": report["coverage"],
                "pack_manifest_digest": report["pack_manifest_digest"],
                "evidence_stage": EVIDENCE_STAGE,
                "qualification_status": QUALIFICATION_STATUS,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RuntimeWorkerError", "main", "run_runtime_worker"]

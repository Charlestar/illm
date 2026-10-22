"""CPU-only executable mutation evidence for numerical policy section 4.4.

This worker deliberately runs counterexamples against the real candidate
implementation.  It produces raw tensor bytes for every numerical detector,
keeps policy-acceptance failures separate from execution failures, and never
uses the formula worker in the same process.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any, Callable, Mapping, NoReturn, Sequence, cast


REPORT_KIND = "illm-policy-cpu-raw-worker-v1"
_FORMULA_MODULE = "illm.policy_cpu_formula_worker"


class MutationWorkerError(RuntimeError):
    """The mutation worker could not produce truthful complete evidence."""


def _fail(message: str) -> NoReturn:
    raise MutationWorkerError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MutationWorkerError("mutation worker report is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MutationWorkerError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise MutationWorkerError(f"cannot inspect path {path}") from exc
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _plain_existing_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail(f"{label} cannot contain a link or reparse point")
    if not absolute.is_dir() or _is_linklike(absolute):
        _fail(f"{label} must be an existing ordinary directory")
    return absolute


def _begin_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in absolute.parents:
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail("mutation worker output ancestors cannot contain links or reparse points")
    if os.path.lexists(absolute):
        _fail("mutation worker output must not already exist")
    parent = absolute.parent
    if not parent.is_dir() or _is_linklike(parent):
        _fail("mutation worker output parent must be an existing ordinary directory")
    try:
        absolute.mkdir()
        with (absolute / "INCOMPLETE").open("xb") as stream:
            stream.write(b"mutation_worker_in_progress\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise MutationWorkerError("cannot initialize mutation worker output") from exc
    return absolute


def _exclusive_write(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise MutationWorkerError(f"cannot exclusively write {path.name}") from exc


def _finish_output(root: Path, report: Mapping[str, object]) -> None:
    encoded = _canonical_bytes(report) + b"\n"
    _exclusive_write(root / "worker.json", encoded)
    _exclusive_write(root / "COMPLETE", (_sha256_bytes(encoded) + "\n").encode("ascii"))
    try:
        (root / "INCOMPLETE").unlink()
    except OSError as exc:
        try:
            (root / "COMPLETE").unlink()
        except OSError:
            pass
        raise MutationWorkerError("cannot clear mutation worker incomplete marker") from exc


def _assert_process_isolation(stage: str) -> None:
    if _FORMULA_MODULE in sys.modules:
        _fail(f"formula and mutation evidence shared one process at {stage}")


def _exception_record(exc: BaseException) -> dict[str, object]:
    return {
        "type": type(exc).__name__,
        "code": getattr(exc, "code", None),
        "field": getattr(exc, "field", None),
        "message": str(exc),
    }


def _resource_not_applicable() -> dict[str, object]:
    return {
        "required": False,
        "fence_kind": None,
        "fence_complete": None,
        "fence_succeeded": None,
        "lease_state": None,
        "active_leases_after_cleanup": None,
        "store_closed": None,
    }


def _json_identity(value: object) -> dict[str, object]:
    encoded = _canonical_bytes(value)
    return {"descriptor": value, "sha256": _sha256_bytes(encoded)}


def run_mutation_worker(project_root: Path, output: Path) -> dict[str, object]:
    """Execute all nine frozen CPU policy counterexamples in one role process."""

    _assert_process_isolation("startup")
    root = _plain_existing_directory(Path(project_root), label="project root")
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "illm").is_dir():
        _fail("mutation worker project root is not an iLLM source workspace")
    destination = _begin_output(Path(output))

    import torch
    from torch import Tensor
    from torch.nn import functional as F

    from illm.executor import ImmediateFence
    from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
    from illm.model import LoadedModel, load_model
    from illm.model.policy import PolicyDecoder, PolicyOpRecord
    from illm.model.reference import ReferenceDecoder
    from illm.model.spec import ModelError
    from illm.model.validation import collect_transformers_ops, load_transformers_oracle
    from illm.numerical_policy import (
        NumericalExecutionPolicyV1,
        NumericalPolicyError,
        candidate_policy,
        parse_policy,
    )
    from illm.policy_binding import build_execution_binding
    from illm.policy_cpu_numerical_contract import FIXED_MUTATION_IDS, tensor_id
    from illm.policy_tensor_evidence import TensorPackWriter

    Trace = list[tuple[PolicyOpRecord, Tensor]]

    if sys.byteorder != "little":
        _fail("tensor evidence v1 requires a little-endian CPU process")
    cuda_initialized_start = bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
    if cuda_initialized_start:
        _fail("mutation worker inherited an initialized CUDA runtime")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        binding = build_execution_binding(root)
        loaded: LoadedModel = load_model(project_root=root)
        fp32_policy = candidate_policy("fp32")
        bf16_policy = candidate_policy("bf16")
        pack = TensorPackWriter(destination / "pack")
        mutations: list[dict[str, object]] = []

        def tensor_raw(tensor: Tensor, dtype: torch.dtype) -> bytes:
            value = tensor.detach().cpu().contiguous()
            if (
                value.dtype != dtype
                or value.layout != torch.strided
                or not bool(torch.isfinite(value).all().item())
            ):
                _fail("mutation tensor differs from its finite contiguous CPU dtype contract")
            return value.view(torch.uint8).numpy().tobytes(order="C")

        def tensor_identity(tensor: Tensor) -> dict[str, object]:
            value = tensor.detach().cpu().contiguous()
            raw = value.view(torch.uint8).numpy().tobytes(order="C")
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "raw_sha256": _sha256_bytes(raw),
            }

        def write_fp32(mutation_id: str, label: str, tensor: Tensor) -> str:
            value = tensor.detach().cpu().contiguous()
            semantic_id = tensor_id("D-mutation", mutation_id, 0, label, None)
            pack.write_tensor(semantic_id, "fp32", tuple(value.shape), tensor_raw(value, torch.float32))
            return semantic_id

        def capture(trace: Trace) -> Callable[[PolicyOpRecord, Tensor], None]:
            def hook(record: PolicyOpRecord, tensor: Tensor) -> None:
                trace.append((record, tensor.detach().cpu().clone().contiguous()))

            return hook

        def run_full(decoder: PolicyDecoder, tokens: tuple[int, ...]) -> Trace:
            trace: Trace = []
            decoder(torch.tensor([tokens], dtype=torch.int64), op_hook=capture(trace))
            return trace

        def first_divergence(left: Trace, right: Trace) -> tuple[PolicyOpRecord, Tensor, Tensor]:
            if len(left) != len(right):
                _fail("mutation traces have different operation counts")
            for (left_record, left_tensor), (right_record, right_tensor) in zip(
                left, right, strict=True
            ):
                if left_record != right_record:
                    _fail("mutation traces have different operation identities")
                if not torch.equal(left_tensor, right_tensor):
                    return left_record, left_tensor, right_tensor
            _fail("mutation did not create an observable tensor divergence")

        def trace_map(trace: Trace) -> dict[tuple[str, int], Tensor]:
            result: dict[tuple[str, int], Tensor] = {}
            for record, tensor in trace:
                key = (record.name, record.position)
                if key in result:
                    _fail("mutation trace repeated an operation identity")
                result[key] = tensor
            return result

        def reference_for_dtype(dtype: str) -> ReferenceDecoder:
            weights = {
                name: loaded.reference.canonical_weight(name).detach().cpu().float().clone().contiguous()
                for name in loaded.spec.tensor_shapes()
            }
            return ReferenceDecoder(replace(loaded.spec, runtime_dtype=dtype), weights)

        # 1. A square Q projection can be transposed without a shape error.
        mutation_id = "transpose-square-q-weight"
        tokens = (3, 4)
        baseline = run_full(PolicyDecoder(loaded.reference, fp32_policy), tokens)
        transposed_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        q_name = "model.layers.0.self_attn.q_proj.weight"
        q_weight = transposed_decoder.canonical_sidecar(q_name)
        original_q = q_weight.detach().clone().contiguous()
        modified_q = original_q.transpose(0, 1).contiguous()
        if original_q.shape[0] != original_q.shape[1] or torch.equal(original_q, modified_q):
            _fail("transpose mutation requires a non-symmetric square Q weight")
        with torch.no_grad():
            q_weight.copy_(modified_q)
        faulted = run_full(transposed_decoder, tokens)
        record, expected_tensor, actual_tensor = first_divergence(baseline, faulted)
        if (record.name, record.position) != ("layer.0.q_projection", 0):
            _fail("transpose mutation did not first diverge at layer0 Q projection")
        paths = {
            "original_q_weight": write_fp32(mutation_id, "original_q_weight", original_q),
            "modified_q_weight": write_fp32(mutation_id, "modified_q_weight", modified_q),
            "expected_first_tensor": write_fp32(mutation_id, "expected_first_tensor", expected_tensor),
            "actual_first_tensor": write_fp32(mutation_id, "actual_first_tensor", actual_tensor),
        }
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"input_token_ids": list(tokens), "weight": tensor_identity(original_q)}),
            "modified_input_identity": _json_identity({"mutation": "transpose(0,1)", "weight": tensor_identity(modified_q)}),
            "expected_detector": {"kind": "first_trace_divergence", "name": record.name, "position": 0},
            "observation": {
                "kind": "first_trace_divergence",
                "name": record.name,
                "position": record.position,
                "expected_tensor": tensor_identity(expected_tensor),
                "actual_tensor": tensor_identity(actual_tensor),
            },
            "tensor_paths": paths,
            "resource_fence": _resource_not_applicable(),
        })

        # 2. Decode position is shifted only after a correct committed prefill.
        mutation_id = "decode-position-plus-one"
        prompt = (3, 4)
        decode_token = 5
        control_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        shifted_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        control_store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=8))
        shifted_store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=8))
        control_decode = None
        shifted_decode = None
        fence = ImmediateFence()
        try:
            for decoder, store in ((control_decoder, control_store), (shifted_decoder, shifted_store)):
                prefill = store.begin_append(prompt)
                decoder.forward_incremental(torch.tensor([prompt], dtype=torch.int64), prefill)
                if prefill.commit(fence) != len(prompt):
                    _fail("decode-position mutation prefill did not commit")
            original_rope = shifted_decoder._rope_position

            def shifted_rope(query: Tensor, key: Tensor, position: int) -> tuple[Tensor, Tensor]:
                return original_rope(query, key, position + 1)

            shifted_decoder._rope_position = shifted_rope  # type: ignore[method-assign]
            control_decode = control_store.begin_append((decode_token,))
            shifted_decode = shifted_store.begin_append((decode_token,))
            control_trace: Trace = []
            shifted_trace: Trace = []
            control_decoder.forward_incremental(
                torch.tensor([[decode_token]], dtype=torch.int64), control_decode, op_hook=capture(control_trace)
            )
            shifted_decoder.forward_incremental(
                torch.tensor([[decode_token]], dtype=torch.int64), shifted_decode, op_hook=capture(shifted_trace)
            )
            record, expected_tensor, actual_tensor = first_divergence(control_trace, shifted_trace)
            if (record.name, record.position) != ("layer.0.rope_q", len(prompt)):
                _fail("decode position mutation did not first diverge at decode RoPE")
            paths = {
                "expected_first_tensor": write_fp32(mutation_id, "expected_first_tensor", expected_tensor),
                "actual_first_tensor": write_fp32(mutation_id, "actual_first_tensor", actual_tensor),
            }
            control_decode.abort(fence)
            shifted_decode.abort(fence)
            control_state = control_decode.state
            shifted_state = shifted_decode.state
            control_decode = None
            shifted_decode = None
        finally:
            if control_decode is not None and control_decode.state == "staged":
                control_decode.abort(fence)
            if shifted_decode is not None and shifted_decode.state == "staged":
                shifted_decode.abort(fence)
            control_active = control_store.active_leases
            shifted_active = shifted_store.active_leases
            control_store.close()
            shifted_store.close()
        if control_active != 0 or shifted_active != 0:
            _fail("decode position mutation leaked an active lease")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({
                "prefill_token_ids": list(prompt), "decode_token_id": decode_token,
                "absolute_decode_position": len(prompt), "prefill_committed": True,
            }),
            "modified_input_identity": _json_identity({
                "field": "decode_absolute_position", "original": len(prompt), "modified": len(prompt) + 1,
            }),
            "expected_detector": {"kind": "first_trace_divergence", "name": "layer.0.rope_q", "position": len(prompt)},
            "observation": {
                "kind": "first_trace_divergence", "name": record.name, "position": record.position,
                "expected_tensor": tensor_identity(expected_tensor), "actual_tensor": tensor_identity(actual_tensor),
            },
            "tensor_paths": paths,
            "resource_fence": {
                "required": True, "fence_kind": "ImmediateFence", "fence_complete": fence.is_complete(),
                "fence_succeeded": fence.succeeded, "lease_state": [control_state, shifted_state],
                "active_leases_after_cleanup": control_active + shifted_active, "store_closed": True,
            },
        })

        # 3. Flip the two KV heads while keeping legal shapes.
        mutation_id = "wrong-gqa-group"
        tokens = (3, 4)
        baseline = run_full(PolicyDecoder(loaded.reference, fp32_policy), tokens)
        gqa_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        original_attention = gqa_decoder._attention_position

        def swapped_attention(
            query: Tensor,
            key_prefix: Tensor,
            value_prefix: Tensor,
            *,
            position: int,
            layer: int,
            hook: Callable[[PolicyOpRecord, Tensor], None] | None,
        ) -> Tensor:
            return original_attention(
                query,
                key_prefix.flip(1).contiguous(),
                value_prefix.flip(1).contiguous(),
                position=position,
                layer=layer,
                hook=hook,
            )

        gqa_decoder._attention_position = swapped_attention  # type: ignore[method-assign]
        faulted = run_full(gqa_decoder, tokens)
        record, expected_tensor, actual_tensor = first_divergence(baseline, faulted)
        if (record.name, record.position) != ("layer.0.attention_scores", 0):
            _fail("GQA mutation did not first diverge at attention scores")
        base_map = trace_map(baseline)
        fault_map = trace_map(faulted)
        context_key = ("layer.0.attention_context", 0)
        if torch.equal(base_map[context_key], fault_map[context_key]):
            _fail("GQA mutation did not change attention context")
        paths = {
            "expected_scores": write_fp32(mutation_id, "expected_scores", expected_tensor),
            "actual_scores": write_fp32(mutation_id, "actual_scores", actual_tensor),
            "expected_context": write_fp32(mutation_id, "expected_context", base_map[context_key]),
            "actual_context": write_fp32(mutation_id, "actual_context", fault_map[context_key]),
        }
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"input_token_ids": list(tokens), "gqa_mapping": "floor(h/groups)"}),
            "modified_input_identity": _json_identity({"input_token_ids": list(tokens), "gqa_mapping": "flip-kv-head-axis"}),
            "expected_detector": {"kind": "scores_and_context", "first_name": "layer.0.attention_scores", "position": 0},
            "observation": {
                "kind": "scores_and_context", "first_name": record.name, "position": record.position,
                "scores_changed": True, "context_changed": True,
            },
            "tensor_paths": paths,
            "resource_fence": _resource_not_applicable(),
        })

        # 4. A future key/value row is rejected by the exact support shape.
        mutation_id = "future-key-visible"
        support_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        query = torch.arange(
            loaded.spec.num_attention_heads * loaded.spec.head_dim, dtype=torch.float32
        ).reshape(1, loaded.spec.num_attention_heads, loaded.spec.head_dim).div_(32.0)
        bad_key = torch.arange(
            loaded.spec.num_kv_heads * 2 * loaded.spec.head_dim, dtype=torch.float32
        ).reshape(1, loaded.spec.num_kv_heads, 2, loaded.spec.head_dim).div_(64.0)
        bad_value = bad_key.flip(-1).contiguous()
        try:
            support_decoder._attention_position(
                query, bad_key, bad_value, position=0, layer=0, hook=None
            )
        except ModelError as exc:
            support_error = exc
        else:
            _fail("future-visible support mutation was accepted")
        if support_error.code != "INVALID_ARGUMENT" or "prefix shape" not in str(support_error):
            _fail("future-visible mutation hit the wrong detector")
        paths = {
            "query": write_fp32(mutation_id, "query", query),
            "bad_key_prefix": write_fp32(mutation_id, "bad_key_prefix", bad_key),
            "bad_value_prefix": write_fp32(mutation_id, "bad_value_prefix", bad_value),
        }
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"position": 0, "legal_prefix_length": 1, "query": tensor_identity(query)}),
            "modified_input_identity": _json_identity({
                "position": 0, "provided_prefix_length": 2,
                "key": tensor_identity(bad_key), "value": tensor_identity(bad_value),
            }),
            "expected_detector": {"kind": "exception", "code": "INVALID_ARGUMENT", "phase": "support-shape"},
            "observation": {"kind": "exception", "phase": "support-shape", "error": _exception_record(support_error)},
            "tensor_paths": paths,
            "resource_fence": _resource_not_applicable(),
        })

        # 5. Corrupt one actually committed K element and compare C to B/native.
        mutation_id = "corrupt-committed-prefix-k"
        b_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        c_decoder = PolicyDecoder(loaded.reference, fp32_policy)
        native = load_transformers_oracle(loaded)
        prefix_store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=8))
        decode_lease = None
        prefix_fence = ImmediateFence()
        prompt = (3, 4)
        context = (*prompt, 5)
        try:
            prefill = prefix_store.begin_append(prompt)
            c_decoder.forward_incremental(torch.tensor([prompt], dtype=torch.int64), prefill)
            if prefill.commit(prefix_fence) != len(prompt):
                _fail("prefix mutation prefill did not commit")
            decode_lease = prefix_store.begin_append((5,))
            actual_key, _ = decode_lease.prefix(0)
            original_prefix = actual_key.detach().clone().contiguous()
            with torch.no_grad():
                actual_key[0, 0, 0, 0] += 1.0
            modified_prefix = actual_key.detach().clone().contiguous()
            if torch.equal(original_prefix, modified_prefix):
                _fail("committed prefix mutation did not alter actual KV storage")
            b_trace: Trace = []
            c_trace: Trace = []
            full_ids = torch.tensor([context], dtype=torch.int64)
            b_decoder(full_ids, op_hook=capture(b_trace))
            c_decoder.forward_incremental(
                torch.tensor([[5]], dtype=torch.int64), decode_lease, op_hook=capture(c_trace)
            )
            native_trace = collect_transformers_ops(native, full_ids, loaded)
            position = len(prompt)
            b_map = trace_map(b_trace)
            c_map = trace_map(c_trace)
            score_name = "layer.0.attention_scores"
            score_key = (score_name, position)
            expected_b = b_map[score_key]
            native_full = native_trace[score_name]
            expected_native = native_full[:, :, position : position + 1, : position + 1].contiguous()
            if not torch.allclose(expected_b, expected_native, atol=1e-4, rtol=1e-3):
                _fail("B and native score controls disagree before prefix detector")
            if torch.allclose(c_map[score_key], expected_native, atol=1e-4, rtol=1e-3):
                _fail("corrupted committed prefix escaped B/native comparison")
            first = next(
                (
                    (record, tensor, b_map[(record.name, record.position)])
                    for record, tensor in c_trace
                    if not torch.equal(tensor, b_map[(record.name, record.position)])
                ),
                None,
            )
            if first is None:
                _fail("corrupted committed prefix did not diverge from B")
            first_record, first_actual, first_expected = first
            if (first_record.name, first_record.position) != score_key:
                _fail("committed prefix mutation did not first diverge at decode scores")
            paths = {
                "original_committed_prefix_k": write_fp32(mutation_id, "original_committed_prefix_k", original_prefix),
                "modified_committed_prefix_k": write_fp32(mutation_id, "modified_committed_prefix_k", modified_prefix),
                "b_scores": write_fp32(mutation_id, "b_scores", expected_b),
                "native_scores": write_fp32(mutation_id, "native_scores", expected_native),
                "c_scores": write_fp32(mutation_id, "c_scores", c_map[score_key]),
            }
            decode_lease.abort(prefix_fence)
            lease_state = decode_lease.state
            decode_lease = None
        finally:
            if decode_lease is not None and decode_lease.state == "staged":
                decode_lease.abort(prefix_fence)
            active_after_cleanup = prefix_store.active_leases
            prefix_store.close()
        if active_after_cleanup != 0:
            _fail("prefix mutation leaked an active lease")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({
                "input_token_ids": list(context), "committed_prefix_k": tensor_identity(original_prefix),
            }),
            "modified_input_identity": _json_identity({
                "element": [0, 0, 0, 0], "delta": 1.0, "committed_prefix_k": tensor_identity(modified_prefix),
            }),
            "expected_detector": {"kind": "b_native_prefix_hard_gate", "name": score_name, "position": position},
            "observation": {
                "kind": "b_native_prefix_hard_gate", "name": first_record.name,
                "position": first_record.position, "b_native_within_tolerance": True,
                "candidate_native_within_tolerance": False,
            },
            "tensor_paths": paths,
            "resource_fence": {
                "required": True, "fence_kind": "ImmediateFence", "fence_complete": prefix_fence.is_complete(),
                "fence_succeeded": prefix_fence.succeeded, "lease_state": lease_state,
                "active_leases_after_cleanup": active_after_cleanup, "store_closed": True,
            },
        })

        # 6. An incorrect digest cannot create an accepted policy instance.
        mutation_id = "wrong-policy-digest"
        fp32_descriptor = fp32_policy.to_dict()
        canonical_policy = _canonical_bytes(fp32_descriptor)
        try:
            NumericalExecutionPolicyV1(
                _canonical_bytes=canonical_policy,
                family=fp32_policy.family,
                storage_dtype=fp32_policy.storage_dtype,
                digest="0" * 64,
                is_candidate=True,
                _legal_kv_modes=("none", "contiguous"),
            )
        except NumericalPolicyError as exc:
            digest_error = exc
        else:
            _fail("wrong policy digest was accepted")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"descriptor": fp32_descriptor, "digest": fp32_policy.digest}),
            "modified_input_identity": _json_identity({"descriptor": fp32_descriptor, "digest": "0" * 64}),
            "expected_detector": {"kind": "exception", "code": "INVALID_NUMERICAL_POLICY", "phase": "pre-acceptance"},
            "observation": {"kind": "exception", "phase": "pre-acceptance", "error": _exception_record(digest_error)},
            "tensor_paths": {},
            "resource_fence": _resource_not_applicable(),
        })

        # 7. Closed policy descriptors reject unknown fields before acceptance.
        mutation_id = "unknown-policy-field"
        unknown_descriptor = {**fp32_descriptor, "unknown_field": "must-reject"}
        try:
            parse_policy(unknown_descriptor)
        except NumericalPolicyError as exc:
            unknown_error = exc
        else:
            _fail("unknown policy field was accepted")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"descriptor": fp32_descriptor, "digest": fp32_policy.digest}),
            "modified_input_identity": _json_identity({"descriptor": unknown_descriptor}),
            "expected_detector": {"kind": "exception", "code": "UNSUPPORTED_NUMERICAL_POLICY", "phase": "pre-acceptance"},
            "observation": {"kind": "exception", "phase": "pre-acceptance", "error": _exception_record(unknown_error)},
            "tensor_paths": {},
            "resource_fence": _resource_not_applicable(),
        })

        def run_linear_guard(mutation_id: str, dtype: str) -> tuple[dict[str, object], dict[str, str]]:
            reference = reference_for_dtype(dtype)
            policy = fp32_policy if dtype == "fp32" else bf16_policy
            control = PolicyDecoder(reference, policy)
            mutated = PolicyDecoder(reference, policy)
            original_linear = F.linear
            attempts: list[dict[str, object]] = []

            def guarded_linear(input_tensor: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
                attempts.append({
                    "input": input_tensor.detach().cpu().clone().contiguous(),
                    "weight": weight.detach().cpu().clone().contiguous(),
                    "bias": None if bias is None else bias.detach().cpu().clone().contiguous(),
                })
                if bias is not None:
                    raise AssertionError("policy conformance rejected fused bias")
                if input_tensor.dtype != torch.float32 or weight.dtype != torch.float32:
                    raise AssertionError("policy conformance rejected non-FP32 contraction")
                return original_linear(input_tensor, weight, bias)

            F.linear = guarded_linear  # type: ignore[assignment]
            try:
                control(torch.tensor([[3, 4]], dtype=torch.int64))
                attempts.clear()
                before_cast: dict[str, Tensor] = {}
                storage_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16

                def wrong_linear_row(hidden: Tensor, name: str, *, bias: bool = False) -> Tensor:
                    row = hidden.contiguous()
                    weight = mutated.canonical_sidecar(name + ".weight")
                    if mutation_id == "fused-bias":
                        fused = mutated.canonical_sidecar(name + ".bias") if bias else None
                        return F.linear(row.float(), weight, fused).to(dtype=storage_dtype)
                    before_cast["input"] = row.float().detach().cpu().clone().contiguous()
                    before_cast["weight"] = weight.detach().cpu().clone().contiguous()
                    cast_input = row.to(dtype=storage_dtype)
                    cast_weight = weight.to(dtype=storage_dtype)
                    result = F.linear(cast_input, cast_weight, None)
                    if bias:
                        result = result + mutated.canonical_sidecar(name + ".bias").to(dtype=storage_dtype)
                    return result

                mutated._linear_row = wrong_linear_row  # type: ignore[method-assign]
                try:
                    mutated(torch.tensor([[3, 4]], dtype=torch.int64))
                except AssertionError as exc:
                    guard_error = exc
                else:
                    _fail(f"{mutation_id} escaped the real F.linear conformance guard")
            finally:
                F.linear = original_linear
            if not attempts:
                _fail(f"{mutation_id} did not invoke real F.linear")
            attempt = attempts[0]
            attempted_input = cast(Tensor, attempt["input"])
            attempted_weight = cast(Tensor, attempt["weight"])
            attempted_bias = cast(Tensor | None, attempt["bias"])
            paths: dict[str, str] = {}
            observation: dict[str, object] = {
                "kind": "linear_conformance_guard",
                "phase": "first-real-F.linear-call",
                "call_count_before_rejection": len(attempts),
                "attempted_input": tensor_identity(attempted_input),
                "attempted_weight": tensor_identity(attempted_weight),
                "attempted_bias": None if attempted_bias is None else tensor_identity(attempted_bias),
                "error": _exception_record(guard_error),
            }
            if mutation_id == "fused-bias":
                if attempted_bias is None or attempted_input.dtype != torch.float32 or attempted_weight.dtype != torch.float32:
                    _fail("fused-bias mutation did not present FP32 input/weight plus actual bias")
                paths["attempted_input"] = write_fp32(mutation_id, "attempted_input", attempted_input)
                paths["attempted_weight"] = write_fp32(mutation_id, "attempted_weight", attempted_weight)
                paths["attempted_bias"] = write_fp32(mutation_id, "attempted_bias", attempted_bias)
                observation["attempted_bf16_raw"] = None
            else:
                if attempted_bias is not None or attempted_input.dtype != torch.bfloat16 or attempted_weight.dtype != torch.bfloat16:
                    _fail("early-storage-cast mutation did not present actual BF16 contraction inputs")
                pre_input = before_cast.get("input")
                pre_weight = before_cast.get("weight")
                if pre_input is None or pre_weight is None:
                    _fail("early-storage-cast mutation did not retain pre-cast FP32 values")
                paths["pre_cast_input"] = write_fp32(mutation_id, "pre_cast_input", pre_input)
                paths["pre_cast_weight"] = write_fp32(mutation_id, "pre_cast_weight", pre_weight)

                def bf16_raw_record(value: Tensor) -> dict[str, object]:
                    contiguous = value.detach().cpu().contiguous()
                    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
                    encoded = raw.hex()
                    if len(encoded) != 2 * len(raw) or encoded != encoded.lower():
                        _fail("BF16 raw evidence is not complete lowercase hex")
                    return {
                        "dtype": "torch.bfloat16", "shape": list(contiguous.shape),
                        "nbytes": len(raw), "sha256": _sha256_bytes(raw), "raw_hex": encoded,
                    }

                observation["attempted_bf16_raw"] = {
                    "input": bf16_raw_record(attempted_input),
                    "weight": bf16_raw_record(attempted_weight),
                }
            return observation, paths

        # 8. Passing a real bias to F.linear violates the unfused policy.
        mutation_id = "fused-bias"
        fused_observation, fused_paths = run_linear_guard(mutation_id, "fp32")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"input_token_ids": [3, 4], "linear": "fp32-no-bias"}),
            "modified_input_identity": _json_identity({"input_token_ids": [3, 4], "linear": "fp32-fused-bias"}),
            "expected_detector": {"kind": "linear_conformance_guard", "reason": "fused_bias"},
            "observation": fused_observation,
            "tensor_paths": fused_paths,
            "resource_fence": _resource_not_applicable(),
        })

        # 9. Casting contraction inputs to BF16 early is negative evidence only.
        mutation_id = "early-storage-cast"
        cast_observation, cast_paths = run_linear_guard(mutation_id, "bf16")
        mutations.append({
            "mutation_id": mutation_id,
            "status": "detected",
            "input_identity": _json_identity({"input_token_ids": [3, 4], "storage_dtype": "bf16", "contraction_dtype": "fp32"}),
            "modified_input_identity": _json_identity({"input_token_ids": [3, 4], "storage_dtype": "bf16", "contraction_dtype": "bf16"}),
            "expected_detector": {"kind": "linear_conformance_guard", "reason": "non_fp32_contraction"},
            "observation": cast_observation,
            "tensor_paths": cast_paths,
            "resource_fence": _resource_not_applicable(),
        })

        expected_ids = list(FIXED_MUTATION_IDS)
        actual_ids = [cast(str, mutation["mutation_id"]) for mutation in mutations]
        if actual_ids != expected_ids or any(mutation.get("status") != "detected" for mutation in mutations):
            _fail("mutation coverage or ordering differs from the frozen protocol")
        for mutation in mutations:
            if set(mutation) != {
                "mutation_id", "status", "input_identity", "modified_input_identity",
                "expected_detector", "observation", "tensor_paths", "resource_fence",
            }:
                _fail("mutation record differs from the closed worker contract")

        _assert_process_isolation("completion")
        cuda_initialized_end = bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
        if cuda_initialized_end:
            _fail("mutation worker initialized CUDA")
        pack_manifest = pack.finalize()
        source_path = Path(__file__).resolve()
        process = {
            "pid": os.getpid(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byte_order": sys.byteorder,
            "cpu_threads": 1,
            "packages": {
                "torch": importlib.metadata.version("torch"),
                "safetensors": importlib.metadata.version("safetensors"),
                "transformers": importlib.metadata.version("transformers"),
            },
            "cuda_internal_initialized_start": cuda_initialized_start,
            "cuda_internal_initialized_end": cuda_initialized_end,
            "formula_worker_module_loaded_start": False,
            "formula_worker_module_loaded_end": _FORMULA_MODULE in sys.modules,
        }
        report: dict[str, object] = {
            "schema_version": 1,
            "report_kind": REPORT_KIND,
            "status": "pass",
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
            "role": "mutation",
            "coverage": {
                "protocol_section": "4.4",
                "mutation_count": len(mutations),
                "mutation_ids": actual_ids,
                "all_mutations_detected": True,
                "pack_path": "pack",
                "gpu_execution": "not_run_not_authorized",
                "claims": ["cpu_executable_counterexamples", "raw_tensor_bytes"],
                "limitations": [
                    "development mutation evidence is not numerical qualification",
                    "CPU BF16 early-cast rejection is not a BF16 profile pass",
                    "no GPU correctness or performance evidence was produced",
                ],
            },
            "binding": binding.to_dict(),
            "binding_digest": binding.digest,
            "process": process,
            "model": {
                "fixture_id": loaded.fixture_id,
                "spec_id": loaded.spec.spec_id,
                "runtime_dtype": loaded.spec.runtime_dtype,
                "weight_manifest_digest": loaded.weight_manifest["manifest_digest"],
                "source": {"path": source_path.relative_to(root).as_posix(), "sha256": _sha256_file(source_path)},
            },
            "policy": {
                "fp32": {"descriptor": fp32_policy.to_dict(), "digest": fp32_policy.digest},
                "bf16": {"descriptor": bf16_policy.to_dict(), "digest": bf16_policy.digest},
            },
            "pack_manifest_digest": pack_manifest["manifest_digest"],
            "mutations": mutations,
        }
        _finish_output(destination, report)
        return cast(dict[str, object], json.loads(_canonical_bytes(report)))
    finally:
        torch.set_num_threads(previous_threads)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce CPU numerical-policy mutation evidence")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_mutation_worker(arguments.project_root, arguments.output)
    print(_canonical_bytes({
        "role": report["role"],
        "status": report["status"],
        "pack_manifest_digest": report["pack_manifest_digest"],
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MutationWorkerError", "REPORT_KIND", "main", "run_mutation_worker"]

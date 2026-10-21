"""CPU-first execution of the V0.2 rowwise numerical policy.

This module is deliberately separate from the immutable V0.1 reference and
the diagnostic C0 incremental decoder.  It implements both the full-context
control (B) and contiguous-KV candidate (C) with one policy object and one set
of FP32 compute sidecars derived from the reference's verified storage
weights.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import os
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from illm.kv_cache import KVWriteLease
from illm.model.reference import ReferenceDecoder, TORCH_DTYPES
from illm.model.spec import ModelError, SupportedModelSpec
from illm.numerical_policy import NumericalExecutionPolicyV1, candidate_policy


@dataclass(frozen=True, slots=True)
class PolicyOpRecord:
    """One storage-dtype observation at one absolute token position.

    Attention score and probability tensors are ragged: their final dimension
    is exactly ``position + 1``.  They never contain future-padding sentinels.
    """

    name: str
    position: int
    shape: tuple[int, ...]
    dtype: str
    layer: int | None
    device: str


PolicyOpHook = Callable[[PolicyOpRecord, Tensor], None]


_EXECUTION_FLAGS_LOCK = RLock()


@contextmanager
def _cpu_execution_flags() -> Iterator[None]:
    """Apply candidate CPU flags for one call and restore all global state."""

    with _EXECUTION_FLAGS_LOCK:
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_precision = torch.get_float32_matmul_precision()
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        previous_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        previous_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_float32_matmul_precision("highest")
            # Host-side backend switches do not create a CUDA context. They
            # implement the complete descriptor but are not GPU evidence.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            with torch.autocast(device_type="cpu", enabled=False):
                yield
        finally:
            try:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous_fp16
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_bf16
                if previous_workspace is None:
                    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                else:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
            finally:
                try:
                    # allow_tf32's setter maps medium back to high. Restore
                    # the precise three-state setting last, not merely its bool.
                    torch.set_float32_matmul_precision(previous_precision)
                finally:
                    try:
                        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
                    finally:
                        torch.set_num_threads(previous_threads)


class PolicyDecoder(nn.Module):
    """Execute candidate B/C arithmetic without changing V0.1 or C0.

    The first implementation is intentionally CPU-only.  Constructing it from
    a non-CPU reference fails before a sidecar or any other tensor is created.
    """

    def __init__(self, reference: ReferenceDecoder, policy: NumericalExecutionPolicyV1) -> None:
        super().__init__()
        if not isinstance(reference, ReferenceDecoder):
            raise ModelError("INVALID_ARGUMENT", "policy decoder requires a ReferenceDecoder")
        if not isinstance(policy, NumericalExecutionPolicyV1):
            raise ModelError("INVALID_ARGUMENT", "policy decoder requires a numerical policy v1")
        if not policy.is_candidate or policy.family != "illm-rowwise-fp32-contraction":
            raise ModelError("UNSUPPORTED_MODEL", "policy decoder only accepts the candidate policy family")
        if policy.storage_dtype != reference.spec.runtime_dtype:
            raise ModelError(
                "INVALID_MODEL_CONFIG",
                "policy storage dtype differs from the model runtime dtype",
                field="storage_dtype",
            )
        registered = candidate_policy(policy.storage_dtype)
        if policy.digest != registered.digest:
            raise ModelError("UNSUPPORTED_MODEL", "candidate policy instance is not registered")
        # Validate both routes at construction so a malformed descriptor cannot
        # be accepted for one path and fail only after execution has started.
        policy.validate_kv_mode("none")
        policy.validate_kv_mode("contiguous")

        embedding = reference.canonical_weight("model.embed_tokens.weight")
        if embedding.device.type != "cpu":
            raise ModelError("UNSUPPORTED_MODEL", "policy decoder CPU implementation rejects non-CPU weights")

        self.reference = reference
        self.policy = policy
        self._storage_dtype = TORCH_DTYPES[policy.storage_dtype]

        sidecar_names: dict[str, str] = {}
        for index, name in enumerate(sorted(reference.spec.tensor_shapes())):
            storage = reference.canonical_weight(name)
            if (
                storage.device.type != "cpu"
                or storage.dtype != self._storage_dtype
                or storage.layout != torch.strided
                or not storage.is_contiguous()
            ):
                raise ModelError(
                    "INVALID_WEIGHTS",
                    "canonical storage is incompatible with the numerical policy",
                    field=name,
                )
            if not bool(torch.isfinite(storage).all().item()):
                raise ModelError("NUMERICAL_ERROR", "canonical storage contains non-finite values", field=name)
            buffer_name = f"policy_fp32_sidecar_{index:04d}"
            # The clone is derived from already-rounded storage exactly once.
            self.register_buffer(
                buffer_name,
                storage.detach().float().clone().contiguous(),
                persistent=True,
            )
            sidecar_names[name] = buffer_name
        self._sidecar_names: Mapping[str, str] = MappingProxyType(sidecar_names)
        self.register_buffer(
            "policy_rms_eps_fp32",
            torch.tensor(reference.spec.rms_norm_eps, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "policy_rope_theta_fp32",
            torch.tensor(reference.spec.rope_theta, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "policy_attention_scale_fp32",
            torch.tensor(reference.spec.head_dim ** -0.5, dtype=torch.float32),
            persistent=True,
        )
        self.eval()

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> PolicyDecoder:
        # ``nn.Module.to/cuda/half`` all route through _apply.  Moving or
        # recasting after the verified storage-derived sidecars exist would
        # violate both the CPU-only boundary and the policy identity.
        raise ModelError(
            "UNSUPPORTED_MODEL",
            "policy decoder placement and dtype are frozen to verified CPU storage",
        )

    @property
    def spec(self) -> SupportedModelSpec:
        return self.reference.spec

    def canonical_sidecar(self, name: str) -> Tensor:
        """Return the unique FP32 sidecar for one canonical storage."""

        if name == "lm_head.weight" and self.spec.tie_word_embeddings:
            name = "model.embed_tokens.weight"
        try:
            value = getattr(self, self._sidecar_names[name])
        except (KeyError, AttributeError) as exc:
            raise ModelError("INVALID_WEIGHTS", "unknown policy sidecar", field=name) from exc
        if not isinstance(value, Tensor):
            raise ModelError("INVALID_WEIGHTS", "policy sidecar is no longer a tensor", field=name)
        return value

    @property
    def sidecar_bytes(self) -> int:
        """Actual bytes held by unique FP32 compute mirrors."""

        return sum(
            self.canonical_sidecar(name).numel() * self.canonical_sidecar(name).element_size()
            for name in self._sidecar_names
        )

    def _storage_weight(self, name: str) -> Tensor:
        value = self.reference.canonical_weight(name)
        if value.device.type != "cpu" or value.dtype != self._storage_dtype:
            raise ModelError("INVALID_WEIGHTS", "canonical storage changed after policy construction", field=name)
        return value

    def _fp32_buffer(self, name: str) -> Tensor:
        value = self.get_buffer(name)
        if not isinstance(value, Tensor):  # pragma: no cover - guarded by registration
            raise ModelError("INTERNAL_ERROR", "policy scalar buffer is unavailable", field=name)
        return value

    def _record(
        self,
        name: str,
        tensor: Tensor,
        position: int,
        layer: int | None,
        hook: PolicyOpHook | None,
    ) -> Tensor:
        if tensor.device.type != "cpu" or tensor.dtype != self._storage_dtype:
            raise ModelError("NUMERICAL_ERROR", "policy op escaped storage dtype or CPU", field=name)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ModelError("NUMERICAL_ERROR", "policy op produced non-finite values", field=name)
        if hook is not None:
            hook(
                PolicyOpRecord(
                    name=name,
                    position=position,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    layer=layer,
                    device=str(tensor.device),
                ),
                tensor.detach(),
            )
        return tensor

    def _validate_input(self, input_ids: Tensor, *, message: str) -> int:
        embedding = self._storage_weight("model.embed_tokens.weight")
        if input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ModelError("INVALID_ARGUMENT", f"{message} must be single-request int64 [1,T]")
        length = input_ids.shape[1]
        if not 1 <= length <= self.spec.max_position_embeddings:
            raise ModelError("INVALID_ARGUMENT", "context length is outside supported bounds")
        if input_ids.device != embedding.device:
            raise ModelError("INVALID_ARGUMENT", "input IDs and policy weights must be on the same CPU device")
        if bool(((input_ids < 0) | (input_ids >= self.spec.vocab_size)).any().item()):
            raise ModelError("INVALID_ARGUMENT", "input token is outside vocabulary")
        return length

    @staticmethod
    def _sequence_from_rows(rows: list[Tensor]) -> Tensor:
        if not rows:
            raise ModelError("INVALID_ARGUMENT", "policy cannot assemble an empty token sequence")
        return torch.cat(rows, dim=0).unsqueeze(0).contiguous()

    @staticmethod
    def _kv_from_rows(rows: list[Tensor]) -> Tensor:
        if not rows:
            raise ModelError("INVALID_ARGUMENT", "policy cannot assemble empty K/V rows")
        return torch.stack(rows, dim=2).contiguous()

    def _embedding_rows(
        self,
        input_ids: Tensor,
        *,
        position_start: int,
        hook: PolicyOpHook | None,
    ) -> Tensor:
        storage = self._storage_weight("model.embed_tokens.weight")
        rows: list[Tensor] = []
        for offset in range(input_ids.shape[1]):
            row = F.embedding(input_ids[:, offset], storage).contiguous()
            rows.append(self._record("embedding", row, position_start + offset, None, hook))
        return self._sequence_from_rows(rows)

    def _norm_row(self, hidden: Tensor, name: str) -> Tensor:
        row = hidden.contiguous()
        promoted = row.float()
        variance = (promoted * promoted).mean(dim=-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self._fp32_buffer("policy_rms_eps_fp32"))
        storage_normalized = normalized.to(dtype=self._storage_dtype)
        result = storage_normalized * self._storage_weight(name)
        if result.dtype != self._storage_dtype:
            raise ModelError("NUMERICAL_ERROR", "RMSNorm multiplication escaped storage dtype", field=name)
        return result

    def _linear_row(self, hidden: Tensor, name: str, *, bias: bool = False) -> Tensor:
        row = hidden.contiguous()
        result = F.linear(row.float(), self.canonical_sidecar(name + ".weight"), None)
        if bias:
            # Bias is a distinct FP32 eager op, never fused into F.linear.
            result = result + self.canonical_sidecar(name + ".bias")
        return result.to(dtype=self._storage_dtype)

    def _rope_position(self, query: Tensor, key: Tensor, position: int) -> tuple[Tensor, Tensor]:
        indices = torch.arange(0, self.spec.head_dim, 2, device=query.device, dtype=torch.float32)
        inv_frequencies = 1.0 / torch.pow(
            self._fp32_buffer("policy_rope_theta_fp32"),
            indices / self.spec.head_dim,
        )
        frequency = inv_frequencies * torch.tensor(float(position), device=query.device, dtype=torch.float32)
        angles = torch.cat((frequency, frequency), dim=-1)
        cosine = torch.cos(angles).to(dtype=self._storage_dtype)
        sine = torch.sin(angles).to(dtype=self._storage_dtype)
        half = self.spec.head_dim // 2
        query_rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
        key_rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
        query_result = query * cosine + query_rotated * sine
        key_result = key * cosine + key_rotated * sine
        if query_result.dtype != self._storage_dtype or key_result.dtype != self._storage_dtype:
            raise ModelError("NUMERICAL_ERROR", "RoPE escaped storage dtype")
        return query_result, key_result

    def _attention_position(
        self,
        query: Tensor,
        key_prefix: Tensor,
        value_prefix: Tensor,
        *,
        position: int,
        layer: int,
        hook: PolicyOpHook | None,
    ) -> Tensor:
        expected_prefix = position + 1
        expected_shape = (1, self.spec.num_kv_heads, expected_prefix, self.spec.head_dim)
        if tuple(key_prefix.shape) != expected_shape or tuple(value_prefix.shape) != expected_shape:
            raise ModelError("INVALID_ARGUMENT", "K/V prefix shape differs from absolute position")
        if (
            key_prefix.dtype != self._storage_dtype
            or value_prefix.dtype != self._storage_dtype
            or key_prefix.device.type != "cpu"
            or value_prefix.device.type != "cpu"
        ):
            raise ModelError("INVALID_ARGUMENT", "K/V prefix dtype or device differs from policy")
        if not bool(torch.isfinite(key_prefix).all().item()) or not bool(torch.isfinite(value_prefix).all().item()):
            raise ModelError("NUMERICAL_ERROR", "K/V prefix contains non-finite values")

        groups = self.spec.num_attention_heads // self.spec.num_kv_heads
        score_rows: list[Tensor] = []
        value_rows: list[Tensor] = []
        for head in range(self.spec.num_attention_heads):
            kv_head = head // groups
            query_row = query[:, head, :].contiguous()
            key_rows = key_prefix[0, kv_head, :, :].contiguous()
            key_matrix = key_rows.float().transpose(0, 1).contiguous()
            score = (query_row.float() @ key_matrix) * self._fp32_buffer("policy_attention_scale_fp32")
            score_rows.append(score.to(dtype=self._storage_dtype))
            value_rows.append(value_prefix[0, kv_head, :, :].contiguous())
        scores = torch.stack(score_rows, dim=1).contiguous()
        scores = self._record(
            f"layer.{layer}.attention_scores",
            scores.unsqueeze(2),
            position,
            layer,
            hook,
        ).squeeze(2)

        probability_rows: list[Tensor] = []
        for head in range(self.spec.num_attention_heads):
            probability_rows.append(
                F.softmax(scores[:, head, :].float(), dim=-1).to(dtype=self._storage_dtype)
            )
        probabilities = torch.stack(probability_rows, dim=1).contiguous()
        probabilities = self._record(
            f"layer.{layer}.attention_probabilities",
            probabilities.unsqueeze(2),
            position,
            layer,
            hook,
        ).squeeze(2)

        context_rows: list[Tensor] = []
        for head in range(self.spec.num_attention_heads):
            context_rows.append(
                (probabilities[:, head, :].float() @ value_rows[head].float().contiguous()).to(
                    dtype=self._storage_dtype
                )
            )
        context = torch.stack(context_rows, dim=1).contiguous()
        context = self._record(
            f"layer.{layer}.attention_context",
            context.unsqueeze(2),
            position,
            layer,
            hook,
        ).squeeze(2)
        return context.reshape(1, self.spec.hidden_size).contiguous()

    def _layer(
        self,
        hidden: Tensor,
        *,
        layer: int,
        position_start: int,
        previous_key: Tensor | None,
        previous_value: Tensor | None,
        hook: PolicyOpHook | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (previous_key is None) != (previous_value is None):
            raise ModelError("INVALID_ARGUMENT", "K/V prefix pair is incomplete")
        if position_start == 0:
            if previous_key is not None or previous_value is not None:
                raise ModelError("INVALID_ARGUMENT", "position-zero policy call cannot have prior K/V")
        else:
            expected = (1, self.spec.num_kv_heads, position_start, self.spec.head_dim)
            if previous_key is None or previous_value is None:
                raise ModelError("INVALID_ARGUMENT", "decode policy call requires prior K/V")
            if tuple(previous_key.shape) != expected or tuple(previous_value.shape) != expected:
                raise ModelError("INVALID_ARGUMENT", "committed K/V prefix shape differs from lease start")
            if (
                previous_key.dtype != self._storage_dtype
                or previous_value.dtype != self._storage_dtype
                or previous_key.device.type != "cpu"
                or previous_value.device.type != "cpu"
            ):
                raise ModelError("INVALID_ARGUMENT", "committed K/V prefix differs from policy dtype or device")

        prefix = f"model.layers.{layer}"
        op = f"layer.{layer}"
        key_rows: list[Tensor] = []
        value_rows: list[Tensor] = []
        output_rows: list[Tensor] = []
        for offset in range(hidden.shape[1]):
            position = position_start + offset
            row = hidden[:, offset, :].contiguous()
            residual = row
            norm = self._record(
                f"{op}.input_norm",
                self._norm_row(row, prefix + ".input_layernorm.weight"),
                position,
                layer,
                hook,
            )
            query = self._linear_row(norm, prefix + ".self_attn.q_proj", bias=True).view(
                1, self.spec.num_attention_heads, self.spec.head_dim
            )
            key = self._linear_row(norm, prefix + ".self_attn.k_proj", bias=True).view(
                1, self.spec.num_kv_heads, self.spec.head_dim
            )
            value = self._linear_row(norm, prefix + ".self_attn.v_proj", bias=True).view(
                1, self.spec.num_kv_heads, self.spec.head_dim
            )
            self._record(f"{op}.q_projection", query.unsqueeze(2), position, layer, hook)
            self._record(f"{op}.k_projection", key.unsqueeze(2), position, layer, hook)
            self._record(f"{op}.v_projection", value.unsqueeze(2), position, layer, hook)
            query, key = self._rope_position(query, key, position)
            self._record(f"{op}.rope_q", query.unsqueeze(2), position, layer, hook)
            self._record(f"{op}.rope_k", key.unsqueeze(2), position, layer, hook)
            key_rows.append(key)
            value_rows.append(value)

            current_key = self._kv_from_rows(key_rows)
            current_value = self._kv_from_rows(value_rows)
            if position_start:
                assert previous_key is not None and previous_value is not None
                key_prefix = torch.cat((previous_key, current_key), dim=2).contiguous()
                value_prefix = torch.cat((previous_value, current_value), dim=2).contiguous()
            else:
                key_prefix = current_key
                value_prefix = current_value
            context = self._attention_position(
                query,
                key_prefix,
                value_prefix,
                position=position,
                layer=layer,
                hook=hook,
            )
            attention = self._record(
                f"{op}.attention_output",
                self._linear_row(context, prefix + ".self_attn.o_proj"),
                position,
                layer,
                hook,
            )
            row = self._record(f"{op}.attention_residual", residual + attention, position, layer, hook)
            residual = row
            norm = self._record(
                f"{op}.post_attention_norm",
                self._norm_row(row, prefix + ".post_attention_layernorm.weight"),
                position,
                layer,
                hook,
            )
            gate = self._record(
                f"{op}.mlp_gate",
                self._linear_row(norm, prefix + ".mlp.gate_proj"),
                position,
                layer,
                hook,
            )
            up = self._record(
                f"{op}.mlp_up",
                self._linear_row(norm, prefix + ".mlp.up_proj"),
                position,
                layer,
                hook,
            )
            activated = self._record(f"{op}.mlp_silu", F.silu(gate), position, layer, hook)
            product = self._record(f"{op}.mlp_product", activated * up, position, layer, hook)
            mlp = self._record(
                f"{op}.mlp_output",
                self._linear_row(product, prefix + ".mlp.down_proj"),
                position,
                layer,
                hook,
            )
            output_rows.append(
                self._record(f"{op}.mlp_residual", residual + mlp, position, layer, hook)
            )
        return (
            self._sequence_from_rows(output_rows),
            self._kv_from_rows(key_rows),
            self._kv_from_rows(value_rows),
        )

    def _final_rows(
        self,
        hidden: Tensor,
        *,
        position_start: int,
        hook: PolicyOpHook | None,
    ) -> Tensor:
        logits_rows: list[Tensor] = []
        for offset in range(hidden.shape[1]):
            position = position_start + offset
            norm = self._record(
                "final_norm",
                self._norm_row(hidden[:, offset, :].contiguous(), "model.norm.weight"),
                position,
                None,
                hook,
            )
            logits_rows.append(
                self._record(
                    "logits",
                    self._linear_row(norm, "lm_head"),
                    position,
                    None,
                    hook,
                )
            )
        return self._sequence_from_rows(logits_rows)

    @torch.inference_mode()
    def forward(self, input_ids: Tensor, *, op_hook: PolicyOpHook | None = None) -> Tensor:
        """Run stateless full-context control B and return every position."""

        self.policy.validate_kv_mode("none")
        self._validate_input(input_ids, message="policy input_ids")
        with _cpu_execution_flags():
            hidden = self._embedding_rows(input_ids, position_start=0, hook=op_hook)
            for layer in range(self.spec.num_layers):
                hidden, _, _ = self._layer(
                    hidden,
                    layer=layer,
                    position_start=0,
                    previous_key=None,
                    previous_value=None,
                    hook=op_hook,
                )
            return self._final_rows(hidden, position_start=0, hook=op_hook)

    @torch.inference_mode()
    def forward_incremental(
        self,
        input_ids: Tensor,
        lease: KVWriteLease,
        *,
        op_hook: PolicyOpHook | None = None,
    ) -> Tensor:
        """Run candidate C for a prefill or one-token decode append lease."""

        self.policy.validate_kv_mode("contiguous")
        query_tokens = self._validate_input(input_ids, message="incremental policy input_ids")
        if not isinstance(lease, KVWriteLease) or lease.state != "staged":
            raise ModelError("INVALID_ARGUMENT", "incremental policy requires an active KV append lease")
        if query_tokens != lease.count:
            raise ModelError("INVALID_ARGUMENT", "incremental query length differs from KV append lease")
        if lease.start > 0 and query_tokens != 1:
            raise ModelError("INVALID_ARGUMENT", "incremental decode after prefill must contain one query token")
        if lease.end > self.spec.max_position_embeddings:
            raise ModelError("RESOURCE_EXHAUSTED", "incremental positions exceed model context")
        if tuple(int(token) for token in input_ids[0].tolist()) != lease.token_ids:
            raise ModelError("INVALID_ARGUMENT", "incremental input IDs differ from the KV append lease")

        with _cpu_execution_flags():
            hidden = self._embedding_rows(input_ids, position_start=lease.start, hook=op_hook)
            for layer in range(self.spec.num_layers):
                previous_key, previous_value = lease.prefix(layer)
                hidden, key, value = self._layer(
                    hidden,
                    layer=layer,
                    position_start=lease.start,
                    previous_key=previous_key if lease.start else None,
                    previous_value=previous_value if lease.start else None,
                    hook=op_hook,
                )
                lease.stage_layer(layer, key, value)
            return self._final_rows(hidden, position_start=lease.start, hook=op_hook)


__all__ = [
    "NumericalExecutionPolicyV1",
    "PolicyDecoder",
    "PolicyOpHook",
    "PolicyOpRecord",
]

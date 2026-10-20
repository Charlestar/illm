"""Single-sequence contiguous K/V storage with staged append leases."""

from __future__ import annotations

from typing import Literal, Protocol

import torch
from torch import Tensor

from illm.kv_cache.layout import KVLayoutSpec
from illm.model.reference import TORCH_DTYPES
from illm.model.spec import ModelError


LeaseState = Literal["staged", "committed", "aborted"]


class CompletionFence(Protocol):
    def is_complete(self) -> bool | None: ...

    @property
    def succeeded(self) -> bool | None: ...


class ContiguousKVStore:
    """Preallocated K/V tensors owned by one sequence."""

    def __init__(self, layout: KVLayoutSpec) -> None:
        self.layout = layout
        dtype = TORCH_DTYPES[layout.dtype]
        self._keys: Tensor | None = torch.empty(layout.tensor_shape, dtype=dtype, device=layout.device)
        self._values: Tensor | None = torch.empty(layout.tensor_shape, dtype=dtype, device=layout.device)
        self._valid_tokens = 0
        self._committed_token_ids: tuple[int, ...] = ()
        self._active_lease: KVWriteLease | None = None
        self._closed = False

    def _require_open(self) -> tuple[Tensor, Tensor]:
        if self._closed or self._keys is None or self._values is None:
            raise ModelError("INVALID_ARGUMENT", "KV store is closed")
        return self._keys, self._values

    @property
    def valid_tokens(self) -> int:
        return self._valid_tokens

    @property
    def committed_token_ids(self) -> tuple[int, ...]:
        return self._committed_token_ids

    @property
    def staged_tokens(self) -> int:
        return self._active_lease.count if self._active_lease is not None else 0

    @property
    def active_leases(self) -> int:
        return int(self._active_lease is not None)

    @property
    def allocated_bytes(self) -> int:
        return 0 if self._closed else self.layout.reserved_bytes

    def begin_append(self, token_ids: tuple[int, ...]) -> KVWriteLease:
        self._require_open()
        if self._active_lease is not None:
            raise ModelError("INVALID_ARGUMENT", "KV store already has an active append lease")
        if not isinstance(token_ids, tuple) or not token_ids or any(type(value) is not int or value < 0 for value in token_ids):
            raise ModelError("INVALID_ARGUMENT", "KV append token IDs must be a nonempty integer tuple")
        start = self._valid_tokens
        end = start + len(token_ids)
        if end > self.layout.capacity_tokens:
            raise ModelError("RESOURCE_EXHAUSTED", "contiguous KV capacity exceeded", field="capacity")
        lease = KVWriteLease(self, start, end, token_ids)
        self._active_lease = lease
        return lease

    def _stage_layer(self, lease: KVWriteLease, layer: int, key: Tensor, value: Tensor) -> None:
        keys, values = self._require_open()
        if lease is not self._active_lease or lease.state != "staged":
            raise ModelError("INVALID_ARGUMENT", "KV append lease is not active")
        if type(layer) is not int or not 0 <= layer < self.layout.num_layers:
            raise ModelError("INVALID_ARGUMENT", "KV layer index is outside layout")
        if layer in lease._staged_layers:
            raise ModelError("INVALID_ARGUMENT", "KV layer was staged more than once")
        expected = (1, self.layout.num_kv_heads, lease.count, self.layout.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ModelError("INVALID_ARGUMENT", "staged K/V shape differs from append range")
        if key.dtype != keys.dtype or value.dtype != values.dtype or key.device != keys.device or value.device != values.device:
            raise ModelError("INVALID_ARGUMENT", "staged K/V dtype or device differs from layout")
        keys[layer, :, lease.start:lease.end, :].copy_(key[0])
        values[layer, :, lease.start:lease.end, :].copy_(value[0])
        lease._staged_layers.add(layer)

    def _prefix(self, lease: KVWriteLease, layer: int) -> tuple[Tensor, Tensor]:
        keys, values = self._require_open()
        if lease is not self._active_lease or lease.state != "staged":
            raise ModelError("INVALID_ARGUMENT", "KV append lease is not active")
        if type(layer) is not int or not 0 <= layer < self.layout.num_layers:
            raise ModelError("INVALID_ARGUMENT", "KV layer index is outside layout")
        return (
            keys[layer:layer + 1, :, :lease.start, :],
            values[layer:layer + 1, :, :lease.start, :],
        )

    def _commit(self, lease: KVWriteLease, fence: CompletionFence) -> int:
        self._require_open()
        if lease is not self._active_lease or lease.state != "staged":
            raise ModelError("INVALID_ARGUMENT", "KV append lease is not active")
        if fence.is_complete() is not True:
            raise ModelError("EXECUTION_STATE_UNKNOWN", "KV write fence is pending or unknown")
        if fence.succeeded is not True:
            raise ModelError("EXECUTION_FAILED", "failed KV write cannot be committed")
        if lease._staged_layers != set(range(self.layout.num_layers)):
            raise ModelError("INVALID_ARGUMENT", "KV append did not stage every layer")
        self._valid_tokens = lease.end
        self._committed_token_ids += lease.token_ids
        lease._state = "committed"
        self._active_lease = None
        return self._valid_tokens

    def _abort(self, lease: KVWriteLease, fence: CompletionFence) -> None:
        self._require_open()
        if lease.state == "aborted":
            return
        if lease is not self._active_lease or lease.state != "staged":
            raise ModelError("INVALID_ARGUMENT", "KV append lease is not active")
        if fence.is_complete() is not True:
            raise ModelError("EXECUTION_STATE_UNKNOWN", "KV write completion is not known; range cannot be reused")
        lease._state = "aborted"
        self._active_lease = None

    def close(self) -> None:
        if self._closed:
            return
        if self._active_lease is not None:
            raise ModelError("EXECUTION_STATE_UNKNOWN", "cannot close KV store with an active append lease")
        self._keys = None
        self._values = None
        self._valid_tokens = 0
        self._committed_token_ids = ()
        self._closed = True


class KVWriteLease:
    """A staged, contiguous append range whose bytes are not yet valid."""

    def __init__(self, store: ContiguousKVStore, start: int, end: int, token_ids: tuple[int, ...]) -> None:
        self._store = store
        self.start = start
        self.end = end
        self.token_ids = token_ids
        self._state: LeaseState = "staged"
        self._staged_layers: set[int] = set()

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def state(self) -> LeaseState:
        return self._state

    def stage_layer(self, layer: int, key: Tensor, value: Tensor) -> None:
        self._store._stage_layer(self, layer, key, value)

    def prefix(self, layer: int) -> tuple[Tensor, Tensor]:
        return self._store._prefix(self, layer)

    def commit(self, fence: CompletionFence) -> int:
        return self._store._commit(self, fence)

    def abort(self, fence: CompletionFence) -> None:
        self._store._abort(self, fence)

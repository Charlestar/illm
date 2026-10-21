"""Synchronous eager executor with explicit ticket/fence/lease ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from illm.domain.errors import EmergencyAbandon
from illm.executor.fence import CompletionFence, ImmediateFence, record_completion_fence
from illm.kv_cache import ContiguousKVStore, KVLayoutSpec, KVWriteLease
from illm.model.incremental import IncrementalDecoder
from illm.model.loader import LoadedModel
from illm.model.spec import ModelError
from illm.numerical_policy import NumericalExecutionPolicyV1
from illm.model.policy import PolicyDecoder


TicketState = Literal["launched", "resolved", "committed", "aborted"]


@dataclass(frozen=True, slots=True)
class ExecutorResourceStatus:
    active_tickets: int
    input_leases: int
    output_leases: int
    workspace_leases: int
    kv_write_leases: int
    kv_reserved_bytes: int
    kv_valid_tokens: int
    kv_staged_tokens: int


@dataclass(slots=True, eq=False)
class ExecutionTicket:
    ticket_id: int
    fence: CompletionFence
    query_tokens: int
    before_valid_tokens: int
    after_valid_tokens: int
    resource_refs: tuple[object, ...]
    kv_write_lease: KVWriteLease | None
    _owner: object
    _state: TicketState = "launched"
    _host_logits: tuple[float, ...] | None = None

    @property
    def state(self) -> TicketState:
        return self._state


@dataclass(slots=True)
class _QuarantinedLaunch:
    """References retained when a failed launch has no trustworthy fence."""

    resource_refs: tuple[object, ...]
    has_input: bool
    has_output: bool


class EagerExecutor:
    """One-request eager execution context.

    ``launch`` always accepts the full committed input sequence.  In
    contiguous mode it derives either the whole prompt (prefill) or the one
    uncached suffix token (decode). Without an explicit numerical policy,
    ``none`` deliberately calls the unchanged V0.1 full-context reference.
    An explicit candidate selects separate policy B/C arithmetic; it never
    replaces that historical reference or silently changes the default.
    """

    def __init__(self, loaded: LoadedModel, kv_mode: str, capacity: int, *,
                 numerical_policy: NumericalExecutionPolicyV1 | None = None) -> None:
        if kv_mode not in {"none", "contiguous"}:
            raise ModelError("INVALID_ARGUMENT", "kv_mode must be none or contiguous", field="kv_mode")
        if type(capacity) is not int or not 1 <= capacity <= min(
            loaded.max_context_tokens, loaded.spec.max_position_embeddings
        ):
            raise ModelError("INVALID_ARGUMENT", "executor capacity is outside model context contract", field="capacity")
        self.loaded = loaded
        self.kv_mode = kv_mode
        self.capacity = capacity
        self._device = loaded.reference.canonical_weight("model.embed_tokens.weight").device
        self._owner = object()
        self._next_ticket_id = 0
        self._active: dict[int, ExecutionTicket] = {}
        self._quarantined: list[_QuarantinedLaunch] = []
        self._closed = False
        self._unhealthy = False
        self._store: ContiguousKVStore | None = None
        self._incremental: IncrementalDecoder | None = None
        self._policy_decoder: PolicyDecoder | None = None
        self.numerical_policy = numerical_policy
        if numerical_policy is not None:
            numerical_policy.validate_kv_mode(kv_mode)
            if numerical_policy.storage_dtype != loaded.spec.runtime_dtype:
                raise ModelError("INVALID_ARGUMENT", "policy dtype differs from loaded model")
            if self._device.type != "cpu":
                raise ModelError("UNSUPPORTED_MODEL", "policy implementation is currently CPU-only; GPU is deferred")
            if numerical_policy.is_candidate:
                self._policy_decoder = PolicyDecoder(loaded.reference, numerical_policy)
        if kv_mode == "contiguous":
            layout = KVLayoutSpec.for_model(loaded.spec, device=str(self._device), capacity=capacity)
            self._store = ContiguousKVStore(layout)
            self._incremental = IncrementalDecoder(loaded.reference, layout)

    def _wrap_fence(self, fence: CompletionFence) -> CompletionFence:
        """Testing seam for a delayed fence wrapper; production is identity."""

        return fence

    def _require_available(self) -> None:
        if self._closed:
            raise ModelError("INVALID_ARGUMENT", "executor is closed")
        if self._unhealthy:
            raise ModelError("EXECUTION_STATE_UNKNOWN", "executor is unhealthy after unknown completion state")
        if self._active:
            raise ModelError("OVERLOADED", "executor already has an active ticket")

    def _validate_context(self, context_ids: tuple[int, ...]) -> None:
        if not isinstance(context_ids, tuple) or not context_ids:
            raise ModelError("INVALID_ARGUMENT", "context_ids must be a nonempty tuple")
        if len(context_ids) > self.capacity:
            raise ModelError("RESOURCE_EXHAUSTED", "context exceeds executor capacity", field="capacity")
        if any(type(token) is not int or not 0 <= token < self.loaded.spec.vocab_size for token in context_ids):
            raise ModelError("INVALID_ARGUMENT", "context contains a token outside vocabulary")

    def _quarantine_failed_launch(
        self,
        inputs: Tensor | None,
        logits: Tensor | None,
        lease: KVWriteLease | None,
        fence: CompletionFence | None,
        error: BaseException | None = None,
    ) -> None:
        # The traceback owns forward-local tensors that were not returned as
        # logits. Retain it too when completion is unknown, rather than relying
        # on unwinding/GC to release possibly in-flight intermediates.
        refs = tuple(value for value in (inputs, logits, lease, fence, error) if value is not None)
        self._quarantined.append(_QuarantinedLaunch(refs, inputs is not None, logits is not None))
        self._unhealthy = True

    def _prove_failed_launch_complete(
        self,
        inputs: Tensor | None,
        logits: Tensor | None,
        lease: KVWriteLease | None,
        original: BaseException,
    ) -> None:
        """Release a failed launch only after a new fence proves writers stopped.

        The original operation error is re-raised by ``launch`` when this proof
        succeeds. If recording or observing the cleanup fence fails, every
        reference owned here remains quarantined and the executor fail-stops.
        """

        cleanup_fence: CompletionFence | None = None
        try:
            cleanup_fence = record_completion_fence(self._device)
            try:
                cleanup_fence.wait()
            except EmergencyAbandon:
                raise
            except BaseException:
                # A failed operation can still be known complete. Only the
                # tri-state completion query decides whether reuse is safe.
                pass
            if cleanup_fence.is_complete() is not True:
                raise ModelError("EXECUTION_STATE_UNKNOWN", "failed launch cleanup fence is pending or unknown")
            if lease is not None:
                lease.abort(cleanup_fence)
        except EmergencyAbandon:
            self._quarantine_failed_launch(inputs, logits, lease, cleanup_fence, original)
            raise
        except BaseException:
            self._quarantine_failed_launch(inputs, logits, lease, cleanup_fence, original)
            raise ModelError(
                "EXECUTION_STATE_UNKNOWN",
                "failed launch completion could not be established; resources are quarantined",
            ) from original

    def launch(self, context_ids: tuple[int, ...]) -> ExecutionTicket:
        self._require_available()
        self._validate_context(context_ids)
        lease: KVWriteLease | None = None
        inputs: Tensor | None = None
        logits: Tensor | None = None
        fence: CompletionFence | None = None
        ticket: ExecutionTicket | None = None
        execution_started = False
        before = 0
        after = 0
        try:
            if self.kv_mode == "none":
                query_ids = context_ids
                execution_started = True
                inputs = torch.tensor([query_ids], dtype=torch.int64, device=self._device)
                logits = (self._policy_decoder.forward(inputs) if self._policy_decoder is not None
                          else self.loaded.reference.forward(inputs))
            else:
                store = self._store
                incremental = self._incremental
                if store is None or incremental is None:
                    raise ModelError("EXECUTION_STATE_UNKNOWN", "contiguous executor lost its KV resources")
                before = store.valid_tokens
                if before:
                    if len(context_ids) != before + 1 or context_ids[:before] != store.committed_token_ids:
                        raise ModelError(
                            "INVALID_ARGUMENT",
                            "decode context must equal committed KV prefix plus exactly one query token",
                        )
                    query_ids = context_ids[before:]
                else:
                    query_ids = context_ids
                lease = store.begin_append(query_ids)
                after = lease.end
                execution_started = True
                inputs = torch.tensor([query_ids], dtype=torch.int64, device=self._device)
                logits = (self._policy_decoder.forward_incremental(inputs, lease)
                          if self._policy_decoder is not None else incremental.forward(inputs, lease))
            base_fence = record_completion_fence(self._device)
            fence = self._wrap_fence(base_fence)
            assert inputs is not None and logits is not None
            ticket = ExecutionTicket(
                ticket_id=self._next_ticket_id,
                fence=fence,
                query_tokens=len(query_ids),
                before_valid_tokens=before,
                after_valid_tokens=after,
                resource_refs=(inputs, logits),
                kv_write_lease=lease,
                _owner=self._owner,
            )
            self._next_ticket_id += 1
            self._active[ticket.ticket_id] = ticket
            return ticket
        except BaseException as exc:
            if isinstance(exc, EmergencyAbandon):
                # A forced second interrupt is not a normal execution failure.
                # Retain all owned addresses without starting another blocking
                # synchronize while the caller is explicitly forcing unwind.
                # This also covers an interrupt inside begin_append before its
                # return is assigned, and inside ticket construction/registration.
                # The store and traceback retain any not-yet-returned lease.
                self._quarantine_failed_launch(inputs, logits, lease, fence, exc)
                if ticket is not None:
                    self._active.pop(ticket.ticket_id, None)  # ownership transferred, not released
                raise
            if execution_started:
                try:
                    self._prove_failed_launch_complete(inputs, logits, lease, exc)
                finally:
                    if ticket is not None:
                        self._active.pop(ticket.ticket_id, None)
            elif lease is not None:
                # No tensor creation or model work started, so this range has
                # no reader/writer and is synchronously safe to roll back.
                lease.abort(ImmediateFence())
            raise

    def _require_ticket(self, ticket: ExecutionTicket) -> ExecutionTicket:
        if not isinstance(ticket, ExecutionTicket) or ticket._owner is not self._owner:
            raise ModelError("INVALID_ARGUMENT", "execution ticket belongs to a different executor")
        active = self._active.get(ticket.ticket_id)
        if active is not ticket:
            raise ModelError("INVALID_ARGUMENT", "execution ticket is not active")
        return ticket

    def resolve(self, ticket: ExecutionTicket) -> tuple[float, ...]:
        ticket = self._require_ticket(ticket)
        if ticket._state == "resolved" and ticket._host_logits is not None:
            return ticket._host_logits
        if ticket._state != "launched":
            raise ModelError("INVALID_ARGUMENT", "execution ticket cannot be resolved in its current state")
        ticket.fence.wait()
        if ticket.fence.is_complete() is not True:
            self._unhealthy = True
            raise ModelError("EXECUTION_STATE_UNKNOWN", "execution fence completion remains unknown")
        if ticket.fence.succeeded is not True:
            raise ModelError("EXECUTION_FAILED", "execution fence completed with failure")
        output = ticket.resource_refs[1]
        if not isinstance(output, Tensor):
            self._unhealthy = True
            raise ModelError("EXECUTION_STATE_UNKNOWN", "ticket output resource is not a tensor")
        values = output[0, -1].detach().float().cpu().tolist()
        ticket._host_logits = tuple(float(value) for value in values)
        ticket._state = "resolved"
        return ticket._host_logits

    def commit(self, ticket: ExecutionTicket) -> int:
        ticket = self._require_ticket(ticket)
        if ticket._state != "resolved":
            raise ModelError("INVALID_ARGUMENT", "execution ticket must resolve before commit")
        valid_tokens = 0
        if ticket.kv_write_lease is not None:
            valid_tokens = ticket.kv_write_lease.commit(ticket.fence)
            if valid_tokens != ticket.after_valid_tokens:
                self._unhealthy = True
                raise ModelError("EXECUTION_STATE_UNKNOWN", "KV commit count differs from ticket contract")
        ticket.resource_refs = ()
        ticket._state = "committed"
        self._active.pop(ticket.ticket_id)
        return valid_tokens

    def abort(self, ticket: ExecutionTicket) -> None:
        ticket = self._require_ticket(ticket)
        try:
            ticket.fence.wait()
        except EmergencyAbandon:
            self._unhealthy = True
            raise
        except BaseException:
            # Cleanup is permitted only when the fence can still prove that
            # every writer has terminated, even if execution itself failed.
            pass
        if ticket.fence.is_complete() is not True:
            self._unhealthy = True
            raise ModelError("EXECUTION_STATE_UNKNOWN", "cannot abort while fence completion is pending or unknown")
        if ticket.kv_write_lease is not None:
            ticket.kv_write_lease.abort(ticket.fence)
        ticket.resource_refs = ()
        ticket._host_logits = None
        ticket._state = "aborted"
        self._active.pop(ticket.ticket_id)

    @property
    def resource_status(self) -> ExecutorResourceStatus:
        store = self._store
        active = len(self._active)
        quarantined = len(self._quarantined)
        return ExecutorResourceStatus(
            active_tickets=active + quarantined,
            input_leases=active + sum(item.has_input for item in self._quarantined),
            output_leases=active + sum(item.has_output for item in self._quarantined),
            workspace_leases=active + quarantined,
            kv_write_leases=store.active_leases if store is not None else 0,
            kv_reserved_bytes=store.allocated_bytes if store is not None else 0,
            kv_valid_tokens=store.valid_tokens if store is not None else 0,
            kv_staged_tokens=store.staged_tokens if store is not None else 0,
        )

    def close(self) -> None:
        if self._closed:
            return
        for ticket in list(self._active.values()):
            self.abort(ticket)
        if self._quarantined:
            self._unhealthy = True
            raise ModelError("EXECUTION_STATE_UNKNOWN", "cannot close executor with quarantined launch resources")
        if self._store is not None:
            self._store.close()
        self._incremental = None
        self._policy_decoder = None
        self._closed = True

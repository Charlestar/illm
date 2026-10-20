# ADR-006: V0.2 stateful execution, streaming and capture boundaries

- Status: accepted for V0.2, 2026-10-21; implementation and qualification pending.
- Scope: one accepted request, synchronous eager executor, contiguous/no KV,
  ordered streaming, cancellation, immutable MemoryPlan and metadata-only R0.

## Preserve the mathematical oracle

Keep the V0.1 `illm.engine.Engine` and DTOs, model spec IDs, weight/tokenizer
digests, canonical sampler and full-context forward unchanged. New entry points
are `illm.engine.streaming.Engine` and `streaming_contracts`. The CLI adds `run`,
`verify` and `inspect request`, without changing `generate` or `model` behavior.
`KVLayoutSpec` is separate: never replace the old model spec's frozen
`none-reference-full-context-v1` layout identity to describe a runtime cache.

The tiny model capacity remains256. New real-model V0.2 profiles use capacity512
for the prescribed256-prompt/64-output workload; old V0.1 profiles remain256.
The native Qwen2 model limit and pinned upstream revision do not change.
Requests require `prompt_tokens + max_new_tokens <= capacity` before acceptance.

## Commit and cancellation

Prefill commits prompt KV then generates output0. Each later decode processes
the last committed output token and only then commits another output. Report KV
coverage and output count separately; a final output need not be written to KV.
Prepare sampling is pure; commit of KV, sampling state and output publication is
serialized against cancel. Greedy consumes no RNG draw. The counter addresses
and seed semantics remain V0.1's canonical contract.

A lock arbitrates exactly one logical terminal winner. It is not permission to
reuse memory. ExecutionTicket holds leases until its fence is known complete;
cancel suppresses uncommitted output, waits, aborts staged KV, then reclaims.
Unknown fence status quarantines references and fail-stops the engine. A failed
release cannot be reported as a resource-baseline success.

`request.terminal_decided` records the unique logical winner independently of
terminal-ready. A permanently unknown completion, or a forced second interrupt,
closes the stream exceptionally with an incomplete capture and quarantined
resources. It cannot produce a normal terminal-ready chunk or a normal latency
sample, because safe completion has not been proved. This explicit failure path
must be tested without classifying the affected request as successfully cleaned
up. Known-complete success/failure/cancel paths still deliver exactly one ready
terminal chunk. The release fault test validates fail-stop, not model success.

Each stream contains one metadata chunk, committed token chunks and one
terminal chunk. Terminal includes `final_text_delta` to flush an incomplete
UTF-8 sequence on finish, cancellation or error. This does not create a token.
Concatenated deltas must equal canonical decode of the committed output prefix.
Special tokens are skipped consistently with the V0.1 tokenizer default.

## Capture, observability and compatibility

New captures use manifest schema2, event envelope v1 and independent catalog
`illm-stateful-v02-v1`, with capabilities exactly `[R0]`. Old schema0/1 readers
retain their original contracts; old V0.1 binaries reject schema2. No new R1,
numerical reexecution or what-if capability is implied by a timeline.
Default storage is uncompressed UTF-8 JSONL and metadata-only: no raw text,
tokens, logits or HMAC key. Local HMACs may correlate facts within one capture.

Required capture creation failure rejects before acceptance. A required write
failure after acceptance marks capture incomplete and is reported separately;
it does not rewrite an already committed inference outcome. Optional exporters
receive bounded diagnostic copies on existing background workers. A stalled
exporter can lose copies but cannot stall inference or invalidate an otherwise
complete independent capture. Tests distinguish the two failure modes.

## Predeclared measurement and verification

Use the five experimental metric IDs and exact censor/denominator rules in the
Version SPEC. Store integer monotonic timestamps; subtract only within their
own server or client domain. Missing client observation is missing, not zero.
Use nearest-rank p99 and preserve latency samples formed before a later failure.

CPU differential validation covers50 distinct cache-off/on cases and the20
existing V0.1 goldens. GPU smoke covers both BF16 and FP16 on the fixed model,
using ADR-005 tolerances without widening after a failure. Report query tokens,
raw timing and memory for the256/64 baseline and telemetry off/on; a speed win
is not a release condition. Release still needs CPU, real GPU, clean-room,
100-cycle/failure evidence, rollback, derived Request Story PNG and all docs.

On the first CLI interrupt request cancellation and wait for safe fence cleanup.
After30seconds warn rather than invent completed cleanup. A second interrupt
may force exit only with an incomplete marker; no success claim is allowed.
There is no deadline API or execution of the reserved DEADLINE_EXCEEDED outcome.

The exception is a client interruption **after proved server completion**:
server terminal bookkeeping/latency commits before capture finalization and
delivery. An already completed immutable server capture stays COMPLETE, because
lost client delivery does not invalidate recorded server facts. The CLI still
returns130. Do not fabricate quarantined GPU leases after they were proved
reclaimed. If reclaim succeeded but terminal recording did not, retain
INCOMPLETE/fail-stop and withhold a complete latency report.

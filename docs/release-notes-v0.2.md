# iLLM V0.2 development release notes

> **Not a release:** these notes describe the current `0.2.0.dev0` development
> snapshot. Qualification is `unqualified`; no `v0.2.0` tag or release bundle is
> implied. The notes become final only after the
> [V0.2 SPEC](specs/versions/v0.2-stateful-streaming.md) gates are sealed.

## What this version adds

- A separate stateful `illm.engine.streaming.Engine` facade with single-request
  `generate()` and ordered `stream()` entry points.
- Explicit metadata, token and terminal DTOs, including incremental text deltas,
  terminal `final_text_delta` and a complete schema2 result.
- Synchronous prefill/decode execution with a contiguous KV cache and a
  full-context `none` correctness oracle.
- Immutable single-request `MemoryPlan`, capacity checks, execution tickets,
  completion fences, resource leases and fail-stop quarantine.
- Idempotent cancellation, unique logical terminal arbitration and separation
  between terminal decision and terminal-ready cleanup.
- schema2 metadata-only capture, typed stateful events, R0 Request Story, derived
  Chrome trace, and fail-closed rejection of higher replay modes.
- Experimental `MetricRegistryV1` with five frozen, fully-qualified streaming
  metric definitions and deterministic report/conformance validation.
- Local CLI commands `run`, `verify --fixture tiny-decoder` and `inspect request`.

## Compatibility

V0.1 model math, strict loader, canonical `SamplingSpec`, model/tokenizer identity
and original facade/DTO imports remain unchanged. V0.2 lives at new import paths
and uses a new capture family rather than silently changing schema1 semantics.
Old readers may reject schema2; the V0.2 reader keeps schema0/schema1 behavior
and advertises only R0 for the new family.

The real-model V0.2 profiles increase the declared context capacity to 512 for
the predeclared 256-prompt/64-output workload. They do not mutate the old V0.1
capacity-256 profiles or the upstream model revision.

## User-visible behavior

- A normal stream emits one metadata chunk, committed token chunks with
  contiguous output indexes, and one terminal chunk after safe cleanup.
- `contiguous` mode performs one full prefill then one-token decode steps;
  `none` recomputes full context and must produce the same tokens.
- `FINISHED`, `CANCELLED` and `FAILED` are implemented accepted outcomes.
  `DEADLINE_EXCEEDED` is reserved and must remain absent in V0.2 reports.
- The first CLI interrupt requests cancellation after acceptance and waits. A
  second interrupt uses emergency abandon and exits 130 without claiming safe
  release.
- Captures default to metadata-only and R0. The live result may expose text and
  token IDs, but the capture does not retain them by implication.

## Evidence status

Targeted CPU/tiny development checks have exercised the core paths, including a
tiny contiguous run whose four query lengths were `[6, 1, 1, 1]`. These are
development observations only. An earlier source-frozen Linux clean-room gate
and an isolated V0.2-to-V0.1 rollback rehearsal passed at different source
snapshots; neither can be promoted into evidence for later source changes.
The final source still needs matching CPU, real GPU BF16/FP16 and clean-room
profiles, stress and benchmark evidence, rollback, derived images and a portable
release bundle. The real BF16/FP16 development smoke remains failed. No final
qualification, GPU-performance or production-speed claim is made.

See [implementation status](implementation/v0.2-status.md),
[known limitations](known-limitations-v0.2.md), and
[teach-back](teach-back-v0.2.md).

## Upgrade and rollback boundary

Create a fresh source-bound V0.2 environment; do not copy an environment or
manifest from another checkout. V0.1 and V0.2 capture/API families remain
separate, so callers should select explicit import paths and reject unknown
schema/capability combinations.

Until V0.2 is formally qualified, the release fallback remains the independently
verified V0.1 artifact/tag in an isolated checkout and its own locked environment.
Rollback must not use `git reset --hard` on the active development tree, reuse a
V0.2 environment for V0.1, or reinterpret schema2 evidence as V0.1 evidence.


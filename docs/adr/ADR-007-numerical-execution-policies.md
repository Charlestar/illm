# ADR-007: Versioned numerical execution policies for V0.2

- Status: accepted for specification and CPU implementation, 2026-10-21.
- Qualification: unqualified; GPU execution explicitly deferred.
- Authority: [human approval](../implementation/v0.2-policy-approval.md) of commit04144ab;
  [frozen protocol](../implementation/v0.2-numerical-policy-protocol.md).

## Context

Frozen V0.1 full-context A remains unchanged. Existing cached C0 failed the
predeclared real BF16/FP16 elementwise tolerances. Controlled diagnostics
support shape-dependent rounding, not a proof that no implementation can meet
the old contract. Old failed/INCOMPLETE evidence must remain byte-for-byte.

KV mode alone cannot identify numerical execution. Separate model, policy,
implementation and environment identities make arithmetic and compatibility
inspectable without silently replacing the old reference.

## Decision

Register immutable, closed-schema, canonical-digest numerical policies.
Evaluate the frozen rowwise FP32-contraction/cast-boundary candidate: storage
remains the profile dtype; explicit compute boundaries and per-row/head shapes
are versioned. Different storage dtypes have different instances; B/C of the
same profile share one instance. KV mode and environment are separate bindings.

B genuinely recomputes the whole prefix and every position's LM head. C uses
only new queries with committed post-RoPE K and projected V. Physical row/head
calls, FP32 sidecars, copies, full logits, memory and initialization are counted.
No candidate is enabled by default before qualification. Legacy+none selects
unchanged A; C0 remains historical/unrecorded diagnostics, not a new registered
legacy+contiguous release combination.

Direct B↔C, ordinary native-dtype/batched Transformers D-native, independent
tiny CPU FP64 mathematical D-high, frozen goldens, complete A/B/C fixed20 greedy
sequences, state/RNG/terminal/resources remain hard gates with original limits.
D-native is not rewritten to candidate scheduling or downgraded after failure.
Only new A↔B/C low-precision elementwise compatibility becomes report-only:
original thresholds still run, failed comparisons retain fail and raw first-op/
error counts, never a claim of numerical equivalence. C0 failures are not promoted.

Use explicit policy-aware versions: capture3/envelope1/new policy catalog/[R0],
MemoryPlan2, stream API revision2/GenerationResult JSON3, new numerical and
benchmark report kinds, qualification/release3, rollback-v2. Old schema0/1/2
readers retain their contracts; schema2 has unrecorded policy, not a retroactive ID.
Format numbers do not grant replay capabilities. Default capture stays metadata-only.

Keep the cumulative3600-second GPU-occupancy budget, full matrix and bounded
pilot/cleanup rules. Unknown balance forbids CUDA initialization. This approval
permits CPU work only; GPU remains deferred even if a balance is later inferred.
No automatic extra budget, dependency, custom kernel or expanded experiment scope.

## Supersedes and preserves

In new-policy V0.2 scope only, supersede ADR-006's implicit single numerical
oracle/fallback routing. Its request lifecycle, fences/leases, cancellation,
stream/UTF8, privacy, metrics and R0 rules remain. ADR-005 retains authority for
V0.1, original tolerances, sampler, model assets and independent oracle.
Do not alter historical tags, bundles, goldens or failed observations.

## Alternatives and rejected shortcuts

Keeping direction A remains valid but unresolved. Native-low-dtype rowwise is
a different policy, not an unapproved fallback on failure. CUDA/Rust/Triton or
new dependencies require separate review. Reject threshold widening, shared-
error self-certification, first-token-only checks, tiny/FP32 substitutes for GPU,
prefix recompute disguised as query1, hidden padding/cost, reduced matrices or
rewritten history.

## Consequences, no-go and rollback

Inspectability costs more identities, formats and validation. FP32 mirrors and
rowwise launches may OOM or exceed budget; D-native or full greedy may fail.
No speed, release or cross-environment bitwise guarantee is made. Any hard-gate
failure/incomplete, unknown budget or unsafe cleanup is no-go, not permission
to lower standards. Final source must rerun the complete qualification chain.

Rollback only for a new request after proved safe cleanup, explicitly A+none;
policy changes invalidate KV/MemoryPlan/derived caches. Unknown fences remain
quarantined/fail-stopped. Independently verify old V0.1 binary/tag/bundle and its
schema3 rejection; retain schema2 rejection and old schema1 replay checks.
New rollback evidence binds from-policy, schema3 capture, target/config/source
and11 real resource/fence facts. Old binary receives no invented policy ID.

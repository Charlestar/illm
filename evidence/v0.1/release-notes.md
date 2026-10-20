# V0.1 Reference Runner — release contract

Target tag: `v0.1.0`. This source document is not a publication announcement.
The verified release bundle and immutable Git tag determine actual release state.
It is frozen before final qualification; a failure must prevent publication,
not be hidden by the release-oriented metadata in this document or the SPEC.

## New learning checkpoint

V0.0 recorded simulated scheduling facts. V0.1 adds real Transformer computation:
strict Qwen2 config/storage contracts, deterministic tiny safetensors, pinned
real-model assets, a tokenizer adapter, eager GQA/RMSNorm/RoPE/gated-MLP decoder,
and synchronous full-context generation.

Canonical sampling freezes greedy tie behavior, FP32 stages, ordered FP64
reductions, top-k/top-p/CDF rules and request-bound Philox counters. Token commit
and RNG consumption are checked together; discarded proposals consume neither.

`model inspect`, `generate`, `model validate`, R0 and R1_REFERENCE expose model
structure, per-op comparisons and control state. Metadata-only capture remains
separate from the normal text/token result. Exporter faults cannot choose tokens.

## What the diagnostics teach

The independent oracle loads original verified safetensors rather than copying
the engine's buffers. Native dtype-aware construction preserves FP32 RoPE
frequencies with BF16/FP16 parameters. Masked attention may legitimately contain
negative infinity at strictly future positions; raw logits, probabilities,
unmasked scores, NaN and positive infinity are still rejected.

Pure wrong-Q-transpose and explicitly labelled compound faults demonstrate
why unchanged greedy output alone does not establish correctness, and how the
first failing op is traced back from an actual token mismatch. Predeclared
tolerances are not widened to accommodate observed failures.

The portable bundle derives two static model/difference SVG diagrams from the
qualified tiny model facts and reproduces their exact bytes when verifying.
They are learning diagrams, not screenshots of a Web application. The baseline
separately checks off/request/ops sampled-output equality on public fixtures,
then records independent forward, host-export, sampling and total timing spans.

## Qualification and compatibility

The completed release requires tiny CPU, real single-GPU and clean-room profiles,
their actual transcripts, fixed-context baseline, allocator-budget failure
capture, immutable-source matches, and a V0.0 rollback rehearsal. CPU or download
success cannot stand in for GPU qualification. The performance result is a
baseline only, with no speedup/adoption claim.

V0.0 readers/bundle evidence remain available and the old tag is not rewritten.
V0.1 formats are versioned and experimental. See `teach-back.md`, `spec.md`,
`known-limitations.md`, and `qualification/release-state.json` in the final bundle.

The official Windows qualification uses CPython3.12.14; GitHub Actions uses a
separate CPython3.12.10 CI-only profile with the same verified wheel bytes. It
cannot grant a functional release qualification. A narrow test adapter retains
each executed pytest subtest as its own JUnit case without weakening the strict
test-accounting reader or changing failures into passes.

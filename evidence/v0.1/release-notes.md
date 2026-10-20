# V0.1 Reference Runner — candidate notes

Target tag: `v0.1.0`. This source document is not a publication announcement.
The verified release bundle and immutable Git tag determine actual release state.

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

## Qualification and compatibility

The completed release requires tiny CPU, real single-GPU and clean-room profiles,
their actual transcripts, fixed-context baseline, allocator-budget failure
capture, immutable-source matches, and a V0.0 rollback rehearsal. CPU or download
success cannot stand in for GPU qualification. The performance result is a
baseline only, with no speedup/adoption claim.

V0.0 readers/bundle evidence remain available and the old tag is not rewritten.
V0.1 formats are versioned and experimental. See `teach-back.md`, `spec.md`,
`known-limitations.md`, and `qualification/release-state.json` in the final bundle.

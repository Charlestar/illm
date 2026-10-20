# ADR-005: V0.1 model, numerical oracle, and sampling contract

- Status: accepted for V0.1, 2026-10-21; numerical thresholds are fixed before qualification.
- Scope: synchronous full-context Qwen2 execution, one active request, no KV,
  streaming, cancellation API, batching, compile, or custom CUDA kernel.

## Frozen model and assets

Use `Qwen/Qwen2.5-0.5B` at model and tokenizer revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. Its official
[config](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json)
declares Qwen2, hidden896/intermediate4864, 24 layers, Q14/KV2/head64,
vocab151936, RMSNorm1e-6, RoPE theta1e6, and tied embeddings. Execute only
local, digest-verified safetensors and tokenizer assets. No floating revision,
remote code, pickle weights, or Transformers runtime model loader is allowed.

The repository tiny model uses the same semantics, 2 layers, hidden32,
intermediate64, Q4/KV2/head8 and vocab259. Its byte tokenizer and actual FP32
safetensors are committed with deterministic generation and asset manifests.
The tiny model teaches computation; its output is not trained language quality.
Every storage appears once in WeightLoadManifest, with tied LM-head alias
pointing to the embedding storage. Audit JSON duplicate keys, shape, dtype,
range coverage and hashes before tensor allocation.

## Dependency and oracle boundary

Pin Torch2.8.0 and Transformers4.57.1 with complete per-platform wheel hashes.
Torch implements our eager numerical primitives; Transformers is an independent
validation oracle. Tokenization uses a local tokenizer adapter. CPU Windows,
CPU Linux and CUDA12.6 Linux have separate Python3.12 artifact locks. An
explicit fetch step prepares a content-addressed cache; bootstrap consumes it
offline, then seals the exact installed package set. This extends ADR-004 for
large platform wheels while keeping the V0.0 tag and default minimal profile.

Include configs, fixtures and runtime locks in source identity. Large real
weights and runtime wheels stay in cache, with published URL, size and SHA256
in committed indexes. They are not ordinary Git blobs.

## Predeclared numerical validation

Disable TF32 and select eager attention in both paths; never use global model
initialization RNG as a fixture. Tiny FP32 per-op tolerance is atol1e-4,
rtol1e-3. For the real model, freeze BF16 atol0.0625/rtol0.02 and FP16
atol0.0078125/rtol0.01 for activations/logits; normalized attention probabilities
use BF16 atol0.005/rtol0.02 and FP16 atol0.001/rtol0.01. These are validation
limits, not observed results. A failure requires finding the first divergent
op, not relaxing limits after seeing output. At least20 fixed greedy inputs
must have exactly equal token IDs, regardless of floating-point tolerance.

Op reports bind spec, weights/load manifest, tokenizer, input, tolerance and
capture identities. Record shapes/dtypes and first mismatching index, maximum
absolute/relative error, and per-op links. A deliberately wrong transpose
fixture must fail at the affected projection, not only at final text.

Construct the testing oracle with the pinned Transformers native dtype-aware
factory, independently loading verified source safetensors, then transfer its
device without a blanket dtype conversion. Qwen2's RoPE `inv_freq` is explicitly
FP32 even when parameters are BF16/FP16. Converting every buffer to the parameter
dtype first loses frequency precision; converting that rounded buffer back to
FP32 does not restore it. Record the oracle factory, parameter dtype and original
FP32 frequency contract in the numerical report. The initial BF16 development
failure remains in `out/v0.1/real-bf16-numerical-development`; the repaired native
oracle rerun passes without changing the predeclared tolerances.

## Canonical sampling clarification

The canonical sampler stays on CPU. Its raw logits are RN-even binary32.
Temperature remains a binary64 scalar; each divided score rounds once to
binary32. `exp_fp32(x)` means RN32(math.exp(RN32(x))). Z/T/S reductions are
explicit binary64 left folds in ascending token ID. p is binary32; r/q and
CDF are binary64. This resolves the earlier shorthand 'retained FP32
probabilities': p is the frozen FP32 stage, without an extra rounding of q.

Top-k and top-p ties use ascending token ID. CDF prefixes use
`cumulative = min(1.0, cumulative + q)`; force the last positive prefix to1.
Saturation prevents a finite rounding overshoot before a tiny tail probability
from making the final CDF non-monotone. The four-logit `[0,1,-3,-45]` regression
freezes this edge case. Draw chooses the first token with `u < cumulative`.

Philox4x32-10 uses seed low/high32 as key, and little-endian counter words
(output-position low32, high32, draw-kind0, draw-index0). Use lane0/2^32.
Published [Random123 vectors](https://github.com/DEShawResearch/random123/blob/v1.14.0/tests/kat_vectors)
anchor the arithmetic. Keep output position separate from draw consumption.
Greedy consumes no draw. Prepare is pure; commit validates against trusted
request/spec/current logits and advances token and RNG state atomically.
Discarded proposals do not advance either. These tests cover proposal discard,
not a public cancellation API in V0.1.

## Events, privacy and replay

Keep V0.0 schema readers; add schema1 reference events and capture dispatch.
R1_REFERENCE reconstructs request/model/sampling control state and counters,
never reruns tensor operations. Default capture stores shapes, lengths and
capture-local digests, with no prompt, output IDs, logits or activation values.
Numerical oracle fixtures are explicit public test artifacts. Every accepted
request has exactly one FINISHED or FAILED outcome, including OOM.

Qualification remains unqualified until actual tiny/oracle/real-GPU/clean-room
gates run. Prepared weights, CPU inspection, or a successful download cannot
stand in for a numerical or hardware pass.

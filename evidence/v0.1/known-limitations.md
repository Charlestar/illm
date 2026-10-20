# V0.1 known limitations

This is a correctness and learning reference, not a production serving engine.
Its final qualification is determined by the verified release manifest and
embedded profiles, not by this document or by a development test snapshot.

- One synchronous active request, batch1; every output token recomputes the full
  context. No KV cache, streaming, cancellation, batching, HTTP or Web UI.
- One exact Qwen2.5-0.5B model/tokenizer revision. Raw-text tokenization only;
  no chat template, automatic downloads, remote code, pickle weights or plugins.
- Tiny weights are deterministic synthetic fixtures, not a trained language
  model. Repeated or nonsensical text is not an inference-quality benchmark.
- GPU validation is limited to the recorded RTX3070/SM8.6/CUDA12.6 profile.
  No universal GPU portability or distributed-runtime claim is made.
- The eager path performs finite checks and exposes op hooks. It deliberately
  favors diagnosis over speed and has not established superiority to vLLM,
  SGLang or any production engine.
- Forward wall time includes synchronous reference work and configured hooks;
  it is not a per-kernel GPU duration. CPU peak memory is not claimed measured.
  Separate logits host export and canonical sampling can be important costs.
- Canonical sampling is scalar CPU work. No vectorized/CUDA sampler, quantized
  kernel, CUDA Graph, speculative decoding, prefix cache or model compilation.
- R0 plays events. R1_REFERENCE reconstructs recorded controls/counters only;
  neither proves numerical reexecution. R2 is not implemented.
- Default capture omits prompts, output IDs, logits and activation values.
  Public oracle fixtures are separately identified test data. Normal generation
  results intentionally return text/token IDs to the caller.
- Optional Python exporter callbacks cannot be forcibly killed. Their queues
  are bounded and close can report timeout; required capture remains independent.
- OOM fails the accepted request. The budget probe changes only its own process
  allocator, restores it, and does not establish admission/recovery behavior.
- Rollback evidence is an isolated artifact rehearsal, not proof that an
  unrelated live service was stopped or a production deployment rolled back.

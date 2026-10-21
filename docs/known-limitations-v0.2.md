# V0.2 known limitations

> This list describes the current `0.2.0.dev0` implementation. V0.2 is
> **unqualified** and has not been released. See the
> [development status](implementation/v0.2-status.md) for the evidence boundary.

## Execution and scale

- The explicit numerical-policy CPU slice uses a separately versioned rowwise
  FP32-contraction candidate, not a replacement for the frozen V0.1 reference.
  It is opt-in; result/metadata schema3 and MemoryPlan2 do not imply a released
  format. Policy-aware capture3, full numerical raw/portable qualification and
  release/rollback migration are pending. Explicit-policy capture is rejected
  before acceptance, and explicit-policy CUDA is rejected before model loading.
  This development API requires the local source workspace to bind its current
  implementation digest; it is not yet a portable standalone wheel interface.
  The ordered CPU driver is a `DEVELOPMENT_ONLY` check, not the full retained
  raw-tensor/portable qualification report.

- The engine supports one accepted non-terminal request. A concurrent offer is
  rejected with `SINGLE_ACTIVE_LIMIT`; there is no queue, fairness policy,
  batching, admission scheduler or backpressure protocol.
- Execution is synchronous eager reference execution. There is no overlap,
  `torch.compile`, CUDA Graph, Triton or custom CUDA kernel.
- KV mode is either full-context `none` or single-sequence `contiguous`. There is
  no paged KV, block table, sharing, eviction, prefix cache, preemption or
  recompute policy.
- Tiny context capacity remains 256. The V0.2 real-model profiles declare 512,
  but real-GPU qualification is still pending.
- Real CUDA contiguous correctness is currently **unresolved**, not merely
  unmeasured: BF16 and FP16 development smoke both exceeded the frozen logits
  tolerance. Finite logits or matching a selected token do not establish
  equivalence. Use the published V0.1 checkpoint, or explicitly select
  `legacy+none=A` before a new diagnostic request. Candidate+none is B, not A;
  changing only KV mode does not identify the arithmetic. Never hot-switch an
  accepted request. See the
  [preserved numerical investigation](implementation/v0.2-numerical-investigation.md).
- The built-in tiny decoder is deterministic synthetic test data. It is not a
  chat model and its text is not an indication of language quality.

## Lifecycle and failure handling

- Accepted terminal outcomes reserve `DEADLINE_EXCEEDED`, but V0.2 exposes no
  deadline API and valid reports require its count to remain zero.
- Cancellation becomes logically final before it can become terminal-ready.
  Resource reuse still waits for a known-complete fence and successful cleanup.
- A permanently unknown fence or forced second CLI interrupt quarantines retained
  references and fail-stops the engine. The process may exit 130 with an
  incomplete capture; it cannot promise that resources were released.
- The CLI's first interrupt waits for cancellation cleanup. A warning after 30
  seconds is only a warning and never a release-success signal.
- Client interruption after proved server completion still exits130, but does
  not invalidate an already completed immutable server capture or fabricate
  quarantined GPU leases. Server completion and client delivery are distinct.

## Observability and replay

- schema2 capture on the retained unrecorded C0 path is metadata-only and has
  capability exactly `[R0]`. Explicit-policy capture is still rejected. It omits
  raw prompt/text, token IDs, logits and activation values. Live CLI results may
  contain token IDs and text, but those values are not silently copied into the
  capture.
- Request Story and Chrome trace are derived timelines. There is no V0.2 R1
  control-state reconstruction, R2 numerical re-execution or R3 what-if policy
  simulation. Unsupported replay modes fail closed.
- Timings are host monotonic wall-time boundaries. They are not per-kernel GPU
  timings and do not allocate all overhead to model execution.
- A bounded optional exporter may drop diagnostic copies and increment a counter.
  It is intentionally not a lossless required capture channel.

## Metrics

- `MetricRegistryV1` is experimental. Only its five fully-qualified streaming
  IDs are defined; changing a boundary, denominator, weighting, unit or censor
  rule requires a new metric ID.
- Engine observations currently contain server `perf_counter_ns` timestamps.
  CLI execution does not fabricate client E2E from server timestamps, so missing
  client pairs increment `client_e2e_missing`. The separate benchmark driver
  records an independently paired client clock domain; those observations are
  not invented by the engine or ordinary CLI.
- The registry uses nearest-rank p99. With small sample counts, p99 is often the
  maximum; it should not be interpreted as a production SLO distribution.
- V0.2 has no open-loop workload generator and no multi-request throughput or
  scheduling fairness metric.

## Interfaces and configuration

- `illm.engine.streaming` and `streaming_contracts` are V0.2 versioned surfaces;
  the V0.1 facade remains separate. The new APIs are not yet declared stable.
- `--model-config` accepts an exact model profile document. There is no general
  engine/session JSON configuration consumer, so this repository does not ship a
  misleading `configs/examples/v0_2_streaming.json`. Use the
  [Python API example](examples/v0_2_streaming.py) or explicit CLI flags.
- The CLI is local process I/O only. There is no HTTP/OpenAI-compatible endpoint,
  WebSocket, daemon, authentication, tenant isolation or interactive Web UI.
- Only the fixed model/spec/tokenizer and canonical sampling contracts are
  accepted. Unsupported model fields, dtype, processor or context values fail
  closed rather than being approximated.

## Qualification and portability

- The final-source V0.2 CPU/GPU/clean-room set is not complete. An earlier
  `44e7ddc` clean-room profile, `7d17bcd` isolated rollback rehearsal and
  `9ca0507` Windows CPU preflight passed, but their different source identities
  cannot be combined into a final release. The public-example changes are later
  source again. Development tests and tiny-model runs do not replace the GPU gate.
- Source-bound toolchain manifests cannot be moved between checkouts or retained
  after source changes. Re-bootstrap the current source; do not copy an old
  environment.
- Windows and Linux locked profiles use different Python patch releases
  (3.12.14 and 3.12.3 respectively). A pass in one environment does not imply a
  pass in the other.
- Promotion uses an exclusive lock outside the release bundle. A concurrent or
  crash-left lock blocks promotion; the tool does not reclaim it automatically.
  Inspect the interrupted transaction and verify its metadata before manually
  resolving a stale lock. Unconfirmed metadata restoration or lock ownership is
  a no-publish condition, not evidence of a completed release.


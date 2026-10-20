# Version learning index

Each completed version has an annotated immutable tag, a runnable demo, a
Version SPEC, a teach-back, and a verified evidence bundle. Work toward later
versions stays visible on `main`; a planned version has no release tag.

| Checkpoint | New concept | Tag / progress | Reading |
|---|---|---|---|
| V0.0 | Facts, deterministic scheduling, capture and replay | `v0.0.0` | [SPEC](../specs/versions/v0.0-bootstrap-lab.md), [teach-back](../teach-back-v0.0.md) |
| V0.1 | Transformer forward, weights and canonical sampling | planned; no tag | [SPEC](../specs/versions/v0.1-reference-runner.md) |
| V0.2–V0.8 | KV, streaming, batching, pages, preemption, chunks, prefix and visual replay | planned; no tags | [release train](../specs/README.md) |
| V1.0–V3.1 | Stable single GPU, performance experiments, distributed execution and operations | planned; no tags | [project plan](../../AI_INFERENCE_ENGINE_PROJECT_PLAN.md) |

## Study a checkpoint

Keep unfinished work on a branch before switching versions. Use a detached
checkout to read a tag, or create your own branch to experiment:

```text
git fetch origin --tags
git switch --detach v0.0.0
python scripts/bootstrap.py --profile study-v00 --locked --write-manifest out/study-v00/toolchain.json --write-manifest-digest out/study-v00/toolchain.sha256
python scripts/in_env.py --manifest out/study-v00/toolchain.json --expected-manifest-digest-file out/study-v00/toolchain.sha256 -- python -m illm demo toy-scheduler --seed 7 --steps 20 --capture out/study-v00/capture
python scripts/in_env.py --manifest out/study-v00/toolchain.json --expected-manifest-digest-file out/study-v00/toolchain.sha256 -- python -m illm replay out/study-v00/capture --mode trace --step 0
python scripts/in_env.py --manifest out/study-v00/toolchain.json --expected-manifest-digest-file out/study-v00/toolchain.sha256 -- python -m illm replay out/study-v00/capture --mode state --assert-state-digest
python scripts/in_env.py --manifest out/study-v00/toolchain.json --expected-manifest-digest-file out/study-v00/toolchain.sha256 -- python -m illm release verify release/v0.0.0/manifest.json
git switch main
```

An old toolchain manifest intentionally becomes invalid when source or lock
bytes change. Re-bootstrap after changing tags; never copy an environment from
another checkout. A historical bundle can be audited using its embedded files;
its absolute recorded environment paths identify the original run, not your
local execution environment.

For V0.0, first read the seven events in step 0. Match arrivals and selection to
the scheduler plan, then find the per-step state digest. Change or delete one
line in a **copy** of `events.jsonl` and replay it: rejection demonstrates why
critical facts and checksums are necessary. The 10,000-step blocking-exporter
test explains how telemetry can lose optional deliveries while capture keeps
every state-changing fact.

## Release rules

Run [the documented gates](../../evidence/v0.0/commands.txt), merge platform
evidence, verify the entire bundle, and promote before creating a tag. The tag
annotation records the manifest digest. Existing tags and release bundles are
never rewritten to describe a newer implementation. A correction after release
uses a new patch tag; a new concept follows the next Version SPEC.

Compare adjacent checkpoints with `git diff v0.0.0..v0.1.0` once both tags
exist. Read the successor's Scope, Architecture Delta, invariants and rollback
alongside that diff. GPU or distributed-resource gaps remain explicit in the
qualification profiles; a CPU learning artifact cannot claim a hardware pass.

# iLLM

iLLM is an inspectable inference-runtime learning project. **V0.1 Reference
Runner** (`0.1.0`) implements a tiny eager Transformer, strict local weight
loading, canonical sampling and metadata-only capture/control replay. Its real
model is pinned Qwen2.5-0.5B, with separate BF16/FP16 numerical profiles.
The tiny weights are synthetic correctness fixtures, not a trained chatbot.

A version is published only when its annotated remote tag and verified evidence
bundle both exist. Version strings and release documents are prepared before
the final gates; they alone are not publication proof. See the
[release evidence guide](docs/implementation/v0.1-status.md) and
[learning index](docs/releases/README.md).

The preserved `v0.0.0` checkpoint contains a CPU-only deterministic toy
scheduler, events, capture and replay, with no tensor/model execution.
V0.1 deliberately has no KV cache, streaming, batching, HTTP or Web UI.

## V0.0 quickstart

Run these commands in a separate checkout of tag `v0.0.0`, not the V0.1
working tree. See the [learning index](docs/releases/README.md).

Use any supported system Python 3.11+ only for the first command. All project
commands after bootstrap are resolved from the finalized toolchain manifest and
do not depend on shell activation or global `PATH` entries.

```text
python scripts/bootstrap.py --profile dev --locked --write-manifest out/v0.0/toolchain.json --write-manifest-digest out/v0.0/toolchain.sha256
python scripts/in_env.py --manifest out/v0.0/toolchain.json --expected-manifest-digest-file out/v0.0/toolchain.sha256 -- python scripts/check_quality.py --output out/v0.0/quality.json
python scripts/in_env.py --manifest out/v0.0/toolchain.json --expected-manifest-digest-file out/v0.0/toolchain.sha256 -- python -m pytest tests/cpu --junitxml out/v0.0/junit.xml
python scripts/in_env.py --manifest out/v0.0/toolchain.json --expected-manifest-digest-file out/v0.0/toolchain.sha256 -- python -m illm demo toy-scheduler --seed 7 --steps 20 --capture out/v0.0
python scripts/in_env.py --manifest out/v0.0/toolchain.json --expected-manifest-digest-file out/v0.0/toolchain.sha256 -- python -m illm replay out/v0.0 --mode trace
python scripts/in_env.py --manifest out/v0.0/toolchain.json --expected-manifest-digest-file out/v0.0/toolchain.sha256 -- python -m illm replay out/v0.0 --mode state --assert-state-digest
```

The normative requirements are in
[`docs/specs/versions/v0.0-bootstrap-lab.md`](docs/specs/versions/v0.0-bootstrap-lab.md).
Later version documents are plans, not implemented capabilities.

## V0.1 quickstart

Use CPython **3.12.14 on Windows x64**, or **3.12.3 on Linux x86_64**,
matching the recorded runtime profiles.
Downloading artifacts and installing them are deliberately separate operations;
the bootstrap resolves no dependencies from the network.

```text
python scripts/fetch_runtime.py --runtime-profile v01-cpu --artifact-cache .illm/cache/runtime
python scripts/bootstrap.py --profile study-v01 --runtime-profile v01-cpu --locked --offline --write-manifest out/study-v01/toolchain.json --write-manifest-digest out/study-v01/toolchain.sha256
python scripts/in_env.py --manifest out/study-v01/toolchain.json --expected-manifest-digest-file out/study-v01/toolchain.sha256 -- python -m illm model inspect --model fixture:tiny-decoder
python scripts/in_env.py --manifest out/study-v01/toolchain.json --expected-manifest-digest-file out/study-v01/toolchain.sha256 -- python -m illm generate --model fixture:tiny-decoder --prompt hello --max-new-tokens 8 --seed 7 --telemetry ops --capture out/study-v01/capture
python scripts/in_env.py --manifest out/study-v01/toolchain.json --expected-manifest-digest-file out/study-v01/toolchain.sha256 -- python -m illm replay out/study-v01/capture --mode state --assert-state-digest
python scripts/in_env.py --manifest out/study-v01/toolchain.json --expected-manifest-digest-file out/study-v01/toolchain.sha256 -- python -m illm model validate --model fixture:tiny-decoder --against transformers --output-directory out/study-v01/oracle
```

Use `python3` in WSL if that is its Python3.12 command. Re-bootstrap after source
changes; reusing an old source-bound manifest intentionally fails. A capture path
cannot be reused or overwritten. The [V0.1 teach-back](docs/teach-back-v0.1.md)
explains shapes, tied storage, sampling and first-op divergence. The complete
[qualification recipe](evidence/v0.1/commands.txt) runs actual phases and seals
their facts. Development runs are not substitutes for source-frozen profiles.

The completed release bundle includes static learning diagrams in
`release/v0.1.0/screenshots/model-and-diff/`: model structure and first-op
divergence, each bound to the recorded model and numerical evidence. They are
generated SVG diagrams, not screenshots of a nonexistent Web UI.

## Qualification boundaries

V0.1 requires all three source-matched profiles: tiny CPU, real single GPU, and
an independently installed Linux clean room. Its bundle includes numerical
comparisons, fixed-context baselines, off/request/ops output invariance, failure
capture, V0.0 rollback, static learning diagrams and their portable verifiers.
See `release/v0.1.0/manifest.json` in a completed checkout. The Windows Actions
CI profile uses a separately pinned Python 3.12.10 and cannot grant a release
qualification; the official Windows profile remains 3.12.14.

V0.0 is the first tagged learning release (`v0.0.0`), with
`functionally_qualified` evidence: CPU, Windows, and Linux profiles pass. The
Linux clean-room profile is produced in WSL Ubuntu and merged only after its
sealed project-root, bootstrap, lock, environment, and evidence digests match
the Windows-built release bundle. GitHub Actions independently repeats the
Windows/Linux matrix after each push. The quality gate includes offline
`mypy --strict`, and the runner rechecks the exact locked distribution set
before importing iLLM. See
[`docs/implementation/v0.0-status.md`](docs/implementation/v0.0-status.md) and
the generated [`release/v0.0.0/manifest.json`](release/v0.0.0/manifest.json).

Use the [version learning index](docs/releases/README.md) to check out a release,
run its demo, and study the change from its predecessor. Each completed version
receives its own annotated tag; later SPECs stay plans until their gates pass.

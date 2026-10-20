# iLLM

iLLM is an inspectable inference-runtime learning project. The current runtime
is **V0.0 Bootstrap Lab**: a CPU-only deterministic toy scheduler with typed
events, capture, R0 trace playback, and R1 control-state replay. It intentionally
contains no tokenizer, tensor, model, GPU, KV cache, or HTTP server yet.

## V0.0 quickstart

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

## Current qualification

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

# V0.0 clean-room quickstart

1. Run `scripts/bootstrap.py` from a supported system Python 3.11+ with both
   `--write-manifest` and `--write-manifest-digest`.
2. Inspect `out/v0.0/toolchain.json`; it must say `finalized: true` and list the
   exact locked distributions.
3. Pass `--expected-manifest-digest-file` to `scripts/in_env.py`, then run the
   quality gate, CPU tests, demo, and replay only through that runner.
4. Seal platform evidence with `scripts/qualify_v00.py`; it binds the toolchain,
   JUnit, `mypy --strict` quality report, and replayed capture.
5. Delete or reorder one event in a copied capture and verify replay exits
   non-zero with `CAPTURE_CORRUPT`.

No shell activation, global `uv`, global `pytest`, or global `illm` command is
part of qualification evidence.

Linux CI uploads a finalized qualification directory. Import it with
`scripts/merge_v00_qualification.py`; the merge fails unless its project,
bootstrap, and lock digests match the local release source.

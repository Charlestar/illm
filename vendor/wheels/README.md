# Offline wheelhouse

These wheels are the complete V0.0 build/test toolchain. `scripts/bootstrap.py`
installs them with `--no-index --require-hashes`; it never falls back to PyPI or
the global environment.

The authoritative package versions and SHA-256 values are in
`requirements-bootstrap.lock` and `requirements-dev.lock`. Refreshing this directory is a toolchain contract
change and requires updating ADR-004, the lock digest, and clean-room evidence.

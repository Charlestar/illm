# ADR-004: Lockfile plus vendored wheelhouse is the V0 environment authority

- Status: accepted for V0.0
- Decision: `requirements-dev.lock` and `vendor/wheels` are the primary build
  facts. Bootstrap uses standard-library `venv` and pip with `--no-index` and
  `--require-hashes`; a container is supplementary, not authoritative.
- Consequence: clean-room tests remain network-independent. Changing any wheel
  or hash changes `lock_digest` and invalidates existing toolchain manifests.
- Consequence: the V0 type gate pins the pure-Python `mypy` wheel and its
  pure-Python dependencies so the same offline wheelhouse works on the declared
  Windows and Linux CPU profiles.

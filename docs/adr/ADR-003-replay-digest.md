# ADR-003: Checkpoint plus event reducer replay

- Status: accepted for V0.0
- Decision: R1 toy replay reads an initial checkpoint and canonical events,
  applies a pure reducer, and compares a SHA-256 canonical state digest after
  every scheduler step.
- Consequence: replay never imports or invokes the online scheduler. Missing,
  reordered, corrupt, or unknown critical facts cannot produce a success claim.


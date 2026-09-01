# ADR-002: JSONL canonical DomainEvent v0

- Status: accepted for V0.0
- Decision: use UTF-8 canonical JSON with sorted keys, compact separators,
  stable numeric codes, producer-local sequence numbers, and monotonic time.
- Consequence: timestamps remain facts but never enter the semantic state
  digest. Unknown required codes fail closed. Protobuf remains a future option.


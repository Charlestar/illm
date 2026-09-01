# ADR-001: Keep the V0 domain independent

- Status: accepted for V0.0
- Decision: `domain` and `scheduler.reference` use only the Python standard
  library and cannot import CLI, capture, HTTP, tensor, or GPU modules.
- Consequence: the scheduler is a pure `ToyState → ToySchedulePlanV0`
  function; replay can validate events without re-running it.


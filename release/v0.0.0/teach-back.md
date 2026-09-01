# V0.0 teach-back: request → plan → event → replay

A request is durable domain state. The reference scheduler reads a snapshot and
returns an immutable plan; it does not mutate that state. The online loop turns
the plan into canonical events. Capture persists an initial checkpoint, those
ordered facts, per-step digests, and checksums. R0 displays the recorded events.
R1 applies them to a separate pure reducer and verifies the state digest after
every step. The reducer also treats a scheduler step as a transaction: selected,
deferred, completed, and the step summary must agree and cover every eligible
request.

Logs are human-oriented explanations. Metrics aggregate bounded numeric facts.
Traces correlate work across time. Domain events are the versioned facts from
which V0.0 can reconstruct state. If a replay-critical fact is missing, the
correct answer is “non-replayable,” never a guessed state.

Qualification is another fact chain. The locked environment identifies the
exact interpreter and distributions; JUnit and the strict type report describe
the checks; the capture proves online/replay agreement; a sealed platform
evidence digest binds those files before a release profile may say `pass`.

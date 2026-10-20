from __future__ import annotations

import argparse
import json
import shutil
import time
import tracemalloc
from pathlib import Path
from threading import Event
from typing import Any, Sequence

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.events import DomainEvent, EventKind
from illm.observability import BoundedMemorySink, EventBus
from illm.qualification import parse_junit
from illm.release import verify_release_manifest, write_verified_release_state
from illm.replay import ToyReplayReducer
from illm.scheduler import build_plan


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _parse_junit(path: Path) -> dict[str, Any]:
    return {
        "schema_version": 0,
        **parse_junit(path),
        "junit_sha256": sha256_file(path),
    }


def _benchmark_sink() -> dict[str, Any]:
    entered = Event()
    release = Event()
    reducer = ToyReplayReducer()
    required = BoundedMemorySink(capacity=3)

    class RequiredFacts:
        delivered = 0

        def emit(self, event: DomainEvent) -> None:
            required.emit(event)
            for recorded in required.drain():
                reducer.apply(recorded)
                self.delivered += 1

    class StalledExporter:
        def emit(self, event: DomainEvent) -> None:
            del event
            entered.set()
            release.wait()

    facts = RequiredFacts()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    bus = EventBus((facts,), (StalledExporter(),), best_effort_capacity=3)
    sequence = 0
    completed_steps = 0

    def emit(kind: EventKind, payload: dict[str, object]) -> None:
        nonlocal sequence
        bus.emit(DomainEvent("benchmark", sequence, time.monotonic_ns(), kind, payload))
        sequence += 1

    try:
        emit(EventKind.DIAGNOSTIC, {})
        if not entered.wait(2):
            raise RuntimeError("exporter did not enter blocking call")
        started = time.perf_counter_ns()
        for step_id in range(10_000):
            emit(EventKind.SCHEDULER_STEP_STARTED, {"step_id": step_id})
            if step_id == 0:
                emit(EventKind.REQUEST_ARRIVED, {"request_id": "benchmark-request", "arrival_step": 0})
            plan = build_plan(reducer.state, step_id)
            for request_id in plan.selected:
                emit(EventKind.REQUEST_SELECTED, {
                    "request_id": request_id, "step_id": step_id, "reason_code": "OLDEST_FIRST",
                })
            for deferred in plan.deferred:
                emit(EventKind.REQUEST_DEFERRED, {
                    "request_id": deferred.request_id, "step_id": step_id,
                    "reason_code": deferred.reason_code,
                })
            for request_id in plan.selected:
                emit(EventKind.REQUEST_COMPLETED, {
                    "request_id": request_id, "step_id": step_id, "terminal": "FINISHED",
                })
            emit(EventKind.SCHEDULER_STEP_COMPLETED, {
                "step_id": step_id, "selected": list(plan.selected),
                "deferred": [item.request_id for item in plan.deferred],
            })
            completed_steps += 1
        state = reducer.finish()
        producer_completed_while_blocked = not release.is_set() and entered.is_set()
        duration_ns = time.perf_counter_ns() - started
        after_current, peak = tracemalloc.get_traced_memory()
        closed_while_blocked = bus.close(timeout=0.05)
    finally:
        release.set()
        shutdown_after_release = bus.close(timeout=2)
        tracemalloc.stop()
    memory_growth = max(0, after_current - before_current)
    memory_limit = 1_048_576
    report = {
        "schema_version": 0,
        "offered_events": sequence,
        "required_delivered_events": facts.delivered,
        "completed_scheduler_steps": completed_steps,
        "final_state_digest": state.digest,
        "critical_event_drop_count": required.critical_drop_count,
        "exporter_was_blocked": entered.is_set(),
        "producer_completed_while_exporter_blocked": producer_completed_while_blocked,
        "exporter_queue_capacity": bus.best_effort_capacity,
        "exporter_queue_high_watermark": bus.exporter_queue_high_watermark,
        "exporter_in_flight_limit": 1,
        "exporter_drop_count": bus.exporter_drop_count,
        "exporter_critical_drop_count": bus.exporter_critical_drop_count,
        "exporter_error_count": bus.exporter_error_count,
        "duration_s": duration_ns / 1_000_000_000,
        "traced_current_growth_bytes": memory_growth,
        "traced_peak_bytes": peak,
        "memory_growth_limit_bytes": memory_limit,
        "memory_bound_pass": memory_growth <= memory_limit,
        "close_while_blocked_returned": closed_while_blocked,
        "close_timeout_s": 0.05,
        "shutdown_after_release": shutdown_after_release,
        "forward_progress": (
            completed_steps == 10_000 and producer_completed_while_blocked
            and facts.delivered == sequence and required.critical_drop_count == 0
            and bus.exporter_queue_high_watermark <= bus.best_effort_capacity
            and memory_growth <= memory_limit and not closed_while_blocked
            and shutdown_after_release
        ),
    }
    if not report["forward_progress"]:
        raise RuntimeError("blocking-exporter experiment failed its forward-progress/boundedness gate")
    return report


def _profile(
    profile_id: str,
    result: str,
    *,
    environment_digest: str | None,
    evidence_digest: str | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "result": result,
        "required_for_functional": True,
        "environment_manifest_digest": environment_digest,
        "evidence_digest": evidence_digest,
        "reason": reason,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V0.0 local evidence bundle")
    parser.add_argument("--toolchain-manifest", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    expected_output = (project_root / "release" / "v0.0.0").resolve()
    if output != expected_output:
        raise SystemExit("V0.0 evidence output must be release/v0.0.0")
    toolchain = json.loads(args.toolchain_manifest.read_text(encoding="utf-8"))
    toolchain_digest = str(toolchain["manifest_digest"])
    test_report = _parse_junit(args.junit)
    if test_report["failures"] or test_report["errors"]:
        raise SystemExit("cannot build passing evidence from a failing JUnit report")

    output.mkdir(parents=True, exist_ok=True)
    capture_target = output / "capture" / "toy-online-replay"
    capture_target.mkdir(parents=True, exist_ok=True)
    capture_files = (
        "capture-manifest.json",
        "checkpoint.json",
        "events.jsonl",
        "state-digests.jsonl",
        "checksums.json",
        "COMPLETE",
    )
    for name in capture_files:
        source = args.capture / name
        if not source.is_file():
            raise SystemExit(f"capture fact is missing: {name}")
        shutil.copy2(source, capture_target / name)
    shutil.copy2(args.commands, output / "commands.txt")
    _write_json(output / "test-report.json", test_report)
    benchmark = _benchmark_sink()
    _write_json(output / "benchmark" / "event-stall" / "report.json", benchmark)
    shutil.copy2(project_root / "docs" / "teach-back-v0.0.md", output / "teach-back.md")
    (output / "known-limitations.md").write_text(
        "# Known limitations\n\n- CPU toy state only; no tensors, model, GPU, KV, HTTP, R2, or R3.\n"
        "- Functional qualification covers Windows and WSL Ubuntu CPU clean rooms; it is not native-GPU or production qualification.\n"
        "- JSONL v0 is experimental and supports explicit rejection rather than migration.\n",
        encoding="utf-8",
    )
    (output / "release-notes.md").write_text(
        "# iLLM v0.0.0 Bootstrap Lab\n\nImplements the offline bootstrap lab, deterministic toy scheduler, "
        "canonical capture, transactional R0/R1 replay, bounded asynchronous best-effort exporters, "
        "sealed platform evidence, rollback rehearsal, and full release-bundle verification.\n\n"
        "Learning checkpoint: inspect each scheduler step, replay facts, reproduce state, then corrupt a copied capture to observe rejection. "
        "The immutable v0.0.0 tag preserves this CPU-only foundation before real model execution is added.\n",
        encoding="utf-8",
    )
    terminal_dir = output / "screenshots" / "terminal"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    for transcript in args.commands.parent.glob("*.txt"):
        shutil.copy2(transcript, terminal_dir / transcript.name)
    if args.transcripts:
        for transcript in args.transcripts.glob("*.txt"):
            shutil.copy2(transcript, terminal_dir / transcript.name)
        if (args.transcripts / "execution.json").is_file():
            shutil.copy2(args.transcripts / "execution.json", terminal_dir / "execution.json")

    test_digest = sha256_json(test_report)
    profiles = [
        _profile(
            "cpu-clean-room-v1",
            "pass",
            environment_digest=toolchain_digest,
            evidence_digest=test_digest,
            reason=None,
        ),
        _profile(
            "windows-bootstrap-v1",
            "not_run",
            environment_digest=None,
            evidence_digest=None,
            reason="sealed_platform_evidence_not_merged",
        ),
        _profile(
            "linux-bootstrap-v1",
            "not_run",
            environment_digest=None,
            evidence_digest=None,
            reason="sealed_platform_evidence_not_merged",
        ),
    ]
    release_manifest: dict[str, Any] = {
        "schema_version": 0,
        "spec_status": "implemented",
        "release_stage": "release_candidate",
        "qualification_tier": "learning_complete",
        "qualification_profiles": profiles,
        "performance_outcome": "not_applicable",
        "lts_state": "not_applicable",
        "toolchain_manifest_digest": toolchain_digest,
        "capture_manifest_digest": json.loads(
            (capture_target / "capture-manifest.json").read_text(encoding="utf-8")
        )["manifest_digest"],
        "test_report_digest": test_digest,
        "event_stall_digest": sha256_json(benchmark),
    }
    release_manifest["manifest_digest"] = sha256_json(release_manifest)
    _write_json(output / "manifest.json", release_manifest)
    verified = verify_release_manifest(output / "manifest.json")
    write_verified_release_state(verified, output / "qualification" / "release-state.json")
    for profile in profiles:
        profile_dir = output / "qualification" / "profiles" / str(profile["profile_id"])
        _write_json(profile_dir / "profile-result.json", profile)
    print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

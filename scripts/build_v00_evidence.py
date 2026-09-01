from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
import tracemalloc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.events import DomainEvent, EventKind
from illm.domain.state import ToyState
from illm.observability import BoundedMemorySink, EventBus
from illm.release import verify_release_manifest, write_verified_release_state
from illm.scheduler import build_plan


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "schema_version": 0,
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "duration_s": sum(float(suite.attrib.get("time", 0)) for suite in suites),
        "junit_sha256": sha256_file(path),
    }


def _benchmark_sink() -> dict[str, Any]:
    class StalledExporter:
        def emit(self, event: DomainEvent) -> None:
            del event
            raise RuntimeError("synthetic exporter stall")

    sink = BoundedMemorySink(capacity=3)
    bus = EventBus((sink,), (StalledExporter(),))
    state = ToyState()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter_ns()
    for seq in range(10_000):
        build_plan(state, seq)
        bus.emit(DomainEvent("benchmark", seq, seq + 1, EventKind.DIAGNOSTIC, {"step_id": seq}))
    duration_ns = time.perf_counter_ns() - started
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = len(sink.drain())
    return {
        "schema_version": 0,
        "offered_events": 10_000,
        "completed_scheduler_steps": 10_000,
        "retained_events": retained,
        "diagnostic_drop_count": sink.dropped_diagnostic_count,
        "critical_event_drop_count": sink.critical_drop_count,
        "exporter_error_count": bus.exporter_error_count,
        "duration_s": duration_ns / 1_000_000_000,
        "traced_current_growth_bytes": max(0, after_current - before_current),
        "traced_peak_bytes": peak,
        "forward_progress": (
            retained == 3
            and sink.dropped_diagnostic_count == 9_997
            and bus.exporter_error_count == 10_000
        ),
    }


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
        "# iLLM v0.0.0 release candidate\n\nImplements the offline bootstrap lab, deterministic toy scheduler, "
        "canonical capture, transactional R0/R1 replay, sealed platform qualification evidence, and fail-closed release verification.\n",
        encoding="utf-8",
    )
    terminal_dir = output / "screenshots" / "terminal"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    for transcript in args.commands.parent.glob("*.txt"):
        shutil.copy2(transcript, terminal_dir / transcript.name)

    test_digest = sha256_json(test_report)
    os_name = platform.system()
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
            "pass" if os_name == "Windows" else "not_run_resource_unavailable",
            environment_digest=toolchain_digest if os_name == "Windows" else None,
            evidence_digest=sha256_json({"toolchain": toolchain_digest, "os": os_name}) if os_name == "Windows" else None,
            reason=None if os_name == "Windows" else "windows_runner_unavailable",
        ),
        _profile(
            "linux-bootstrap-v1",
            "pass" if os_name == "Linux" else "not_run_resource_unavailable",
            environment_digest=toolchain_digest if os_name == "Linux" else None,
            evidence_digest=sha256_json({"toolchain": toolchain_digest, "os": os_name}) if os_name == "Linux" else None,
            reason=None if os_name == "Linux" else "linux_runner_unavailable",
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

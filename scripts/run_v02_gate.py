"""Run source-bound V0.2 profiles; never promote a release or create a tag.

Downloads are separate. Every post-bootstrap project command goes through the
finalized manifest. A new managed environment is required, so this orchestrator
can never replace a user's existing environment while claiming a clean run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from _toolchain_common import PROFILE_PATTERN, atomic_write_json, is_within, lock_digest, sha256_bytes, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> tuple[dict[str, str], dict[str, str], Callable[[str], dict[str, dict[str, Any]]]]:
    # This is a pure-stdlib control-plane module, like _toolchain_common. Loading
    # its source does not run the project/runtime before the manifest handshake.
    path = PROJECT_ROOT / "src/illm/stateful_gate_contract.py"
    spec = importlib.util.spec_from_file_location("illm_v02_gate_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError("V0.2 gate contract is absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (cast(dict[str, str], module.STATEFUL_PROFILE_RUNTIME_IDS), cast(dict[str, str], module.MODEL_PROFILES),
            cast(Callable[[str], dict[str, dict[str, Any]]], module.phase_parameters))


def phase_commands(profile_id: str, output: Path, *, device: str = "cuda:0") -> dict[str, list[str]]:
    """Build the exact project argv after bootstrap, without executing it."""
    runtimes, models, parameters = _contract()
    if profile_id not in runtimes:
        raise ValueError("unknown V0.2 profile")
    commands = {
        "quality": ["python", "scripts/check_quality.py", "--output", str(output / "quality.json")],
        "pytest": ["python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--basetemp", str(output / "pytest-temp"),
                   "--junitxml", str(output / "junit.xml"), "--tb=short"],
    }
    if profile_id == "stateful-streaming-single-gpu-v1":
        for dtype, model in models.items():
            commands[f"stateful-{dtype}"] = [
                "python", "scripts/validate_stateful.py", "--model", model, "--device", device,
                "--scope", "real-smoke", "--output", str(output / f"stateful-{dtype}"),
            ]
        for dtype, model in models.items():
            for suite in ("tuning", "heldout"):
                commands[f"benchmark-{dtype}-{suite}"] = [
                    "python", "scripts/benchmark_streaming.py", "--model", model, "--device", device,
                    "--target", "real", "--suite", suite, "--warmups", "3", "--repetitions", "7",
                    "--output", str(output / f"benchmark-{dtype}-{suite}"),
                ]
    else:
        capture = output / "demo-capture"
        learning_capture = output / "exercise/captures/cancel-after-launch"
        learning = ["python", "scripts/render_streaming_learning.py", str(learning_capture), "--output", str(output / "learning")]
        commands.update({
            "verify": ["python", "-m", "illm", "verify", "--fixture", "tiny-decoder"],
            "run": ["python", "-m", "illm", "run", "--model", "fixture:tiny-decoder", "--device", "cpu",
                    "--prompt", "hello", "--max-new-tokens", "8", "--stream", "--request-id", "req-0001",
                    "--kv-mode", "contiguous", "--telemetry", "request", "--capture", str(capture)],
            "replay-trace": ["python", "-m", "illm", "replay", str(capture), "--mode", "trace"],
            "inspect-request": ["python", "-m", "illm", "inspect", "request", str(capture), "--request", "req-0001"],
            "stateful-full": ["python", "scripts/validate_stateful.py", "--model", "fixture:tiny-decoder", "--device", "cpu",
                              "--scope", "full", "--output", str(output / "stateful-full")],
            "exercise": ["python", "scripts/exercise_streaming.py", "--output", str(output / "exercise")],
            "learning-build": learning,
            "learning-verify": [*learning, "--verify-existing"],
        })
    if list(commands) != list(parameters(profile_id))[1:]:
        raise ValueError("gate argv and shared phase contract differ")
    return commands


def run_gate(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    runtimes, _, parameters_for = _contract()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-id", choices=tuple(runtimes), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    project = PROJECT_ROOT
    runtime = runtimes[args.profile_id]
    if PROFILE_PATTERN.fullmatch(args.profile) is None:
        parser.error("--profile must be a safe managed environment name")
    if (runtime.startswith("py312-win") and platform.system() != "Windows") or (
        runtime.startswith("py312-linux") and platform.system() != "Linux"
    ):
        parser.error("qualification profile belongs to a different operating system")
    if args.profile_id == "stateful-streaming-single-gpu-v1" and args.device != "cuda:0":
        parser.error("the frozen GPU profile uses exactly --device cuda:0")
    environment_root = project / ".illm/envs" / args.profile
    if not environment_root.resolve().is_relative_to(project) or not (project / "out").resolve().is_relative_to(project):
        parser.error("managed environments and gate outputs cannot follow links outside the project")
    if environment_root.exists() or environment_root.is_symlink():
        parser.error("qualification requires a new managed environment; existing environments are never removed")
    output = args.output.resolve()
    if args.output.is_symlink() or not is_within(output, project / "out") or output == (project / "out").resolve() or output.exists():
        parser.error("--output must be a new directory strictly inside workspace out/")
    commands_by_phase = phase_commands(args.profile_id, output, device=args.device)
    parameters = parameters_for(args.profile_id)
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    incomplete = output / "INCOMPLETE"
    atomic_write_json(incomplete, {"reason": "gate_in_progress", "profile_id": args.profile_id})
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST"):
        environment.pop(name, None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    manifest = output / "toolchain.json"
    sidecar = output / "toolchain.sha256"
    prefix = [sys.executable, str(project / "scripts/in_env.py"), "--manifest", str(manifest),
              "--expected-manifest-digest-file", str(sidecar), "--"]
    records: list[dict[str, Any]] = []

    def phase(name: str, command: list[str], *, bootstrap: bool = False) -> None:
        actual = command if bootstrap else [*prefix, *command]
        process = subprocess.run(actual, cwd=project, env=environment, capture_output=True, check=False)
        stdout, stderr = logs / f"{name}.stdout.txt", logs / f"{name}.stderr.txt"
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        record = {
            "phase": name, "argv": actual, "parameters": parameters[name], "returncode": process.returncode,
            "stdout_file": stdout.relative_to(output).as_posix(), "stderr_file": stderr.relative_to(output).as_posix(),
            "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr),
        }
        records.append(record)
        atomic_write_json(output / "execution.json", {"schema_version": 2, "profile_id": args.profile_id, "commands": records})
        print(json.dumps({"phase": name, "returncode": process.returncode, "stdout": record["stdout_file"],
                          "stderr": record["stderr_file"]}, sort_keys=True), flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"phase {name} failed; preserved transcripts remain authoritative")

    try:
        bootstrap = [sys.executable, str(project / "scripts/bootstrap.py"), "--profile", args.profile,
                     "--runtime-profile", runtime, "--locked", "--offline", "--require-new-environment", "--write-manifest", str(manifest),
                     "--write-manifest-digest", str(sidecar)]
        if args.artifact_cache is not None:
            bootstrap.extend(["--artifact-cache", str(args.artifact_cache.resolve())])
        phase("bootstrap", bootstrap, bootstrap=True)
        for name, command in commands_by_phase.items():
            phase(name, command)
        toolchain = json.loads(manifest.read_text(encoding="utf-8"))
        report: dict[str, Any] = {
            "schema_version": 2, "report_kind": "illm-v02-profile-gate-v1", "runtime_profile_id": runtime,
            "profile_id": args.profile_id, "status": "pass", "fresh_environment": True,
            "offline_bootstrap": True, "network_used": "not_observed", "network_blocking": "not_enforced",
            "network_policy": "offline_resolution_and_local_model_only",
            "toolchain_manifest_digest": toolchain["manifest_digest"],
            "source_identity": {"project_root_digest": toolchain["project_root_digest"],
                                "bootstrap_digest": toolchain["bootstrap_digest"], "base_lock_digest": lock_digest(project)},
            "commands": records,
        }
        report["report_digest"] = sha256_bytes(json.dumps(report, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=False, allow_nan=False).encode("utf-8"))
        atomic_write_json(output / "gate-report.json", report)
        actual = [*prefix, "python", "scripts/qualify_v02.py", "--profile-id", args.profile_id,
                  "--gate-root", str(output), "--output", str(output / "qualification")]
        process = subprocess.run(actual, cwd=project, env=environment, capture_output=True, check=False)
        (output / "qualify.stdout.txt").write_bytes(process.stdout)
        (output / "qualify.stderr.txt").write_bytes(process.stderr)
        print(json.dumps({"phase": "qualify", "returncode": process.returncode}, sort_keys=True), flush=True)
        if process.returncode != 0:
            raise RuntimeError("qualification sealing failed; inspect qualify transcripts")
        if not (output / "qualification/COMPLETE").is_file():
            raise RuntimeError("qualification returned without its sealed completion artifact")
        (output / "COMPLETE").write_text("complete\n", encoding="ascii")
        incomplete.unlink()
        return 0
    except BaseException as exc:
        atomic_write_json(incomplete, {"reason": "gate_failed", "profile_id": args.profile_id,
                                       "exception_type": type(exc).__name__, "completed_phases": len(records)})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_gate(argv)
    except (RuntimeError, OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "exception_type": type(exc).__name__, "message": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

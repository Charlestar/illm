"""Execute real V0.1 qualification phases and preserve their actual transcripts.

Downloads are a separate explicit preparation step. This gate bootstraps from
the verified local wheel cache, then runs every project command through the
finalized manifest. It never promotes a release or creates a tag.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from _toolchain_common import atomic_write_json, is_within, lock_digest, sha256_bytes, sha256_file


def run_gate(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="managed environment name")
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--profile-id", choices=("tiny-reference-cpu-v1", "real-model-single-gpu-v1", "clean-room-release-v1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--fp16-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--artifact-cache", type=Path)
    args = parser.parse_args(argv)
    if args.profile_id == "real-model-single-gpu-v1":
        if not args.device.startswith("cuda"):
            parser.error("the real-model GPU profile requires --device cuda[:index]")
        if not args.fp16_model:
            parser.error("the real-model GPU profile requires --fp16-model")
    elif args.device != "cpu" or args.model != "fixture:tiny-decoder":
        parser.error("the tiny and clean-room CPU profiles require the tiny fixture on cpu")
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not is_within(output, project / "out") or output == (project / "out").resolve() or output.exists():
        parser.error("--output must be a new directory strictly below workspace out/")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    (output / "INCOMPLETE").write_text("gate_in_progress\n", encoding="ascii")
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    commands: list[dict[str, Any]] = []
    manifest = output / "toolchain.json"
    sidecar = output / "toolchain.sha256"
    prefix = [sys.executable, str(project / "scripts" / "in_env.py"), "--manifest", str(manifest),
              "--expected-manifest-digest-file", str(sidecar), "--"]

    def phase(name: str, command: list[str], *, project_command: bool = True) -> None:
        actual = [*prefix, *command] if project_command else command
        process = subprocess.run(actual, cwd=project, env=environment, capture_output=True, check=False)
        stdout = logs / f"{name}.stdout.txt"
        stderr = logs / f"{name}.stderr.txt"
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        record = {"phase": name, "argv": actual, "returncode": process.returncode,
                  "stdout_file": stdout.relative_to(output).as_posix(), "stderr_file": stderr.relative_to(output).as_posix(),
                  "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr)}
        commands.append(record)
        atomic_write_json(output / "execution.json", {"schema_version": 1, "commands": commands})
        print(json.dumps({"phase": name, "returncode": process.returncode, "stdout": record["stdout_file"],
                          "stderr": record["stderr_file"]}, sort_keys=True), flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"qualification phase {name} failed; inspect preserved transcripts")

    bootstrap = [sys.executable, str(project / "scripts" / "bootstrap.py"), "--profile", args.profile,
                 "--runtime-profile", args.runtime_profile, "--locked", "--offline", "--write-manifest", str(manifest),
                 "--write-manifest-digest", str(sidecar)]
    if args.artifact_cache:
        bootstrap.extend(["--artifact-cache", str(args.artifact_cache.resolve())])
    phase("bootstrap", bootstrap, project_command=False)
    phase("quality", ["python", "scripts/check_quality.py", "--output", str(output / "quality.json")])
    phase("pytest", ["python", "-m", "pytest", "tests", "-p", "no:cacheprovider", "--basetemp", str(output / "pytest-temp"),
                     "--junitxml", str(output / "junit.xml"), "--tb=short"])
    phase("inspect", ["python", "-m", "illm", "model", "inspect", "--model", args.model])
    capture = output / "capture"
    phase("generate", ["python", "-m", "illm", "generate", "--model", args.model, "--device", args.device,
                       "--prompt", "hello", "--max-new-tokens", "8", "--seed", "7", "--telemetry", "ops", "--capture", str(capture)])
    validation_directories = [output / "validation"]
    phase("validate", ["python", "-m", "illm", "model", "validate", "--model", args.model, "--device", args.device,
                       "--against", "transformers", "--output-directory", str(validation_directories[0])])
    if args.fp16_model:
        validation_directories.append(output / "validation-fp16")
        phase("validate-fp16", ["python", "-m", "illm", "model", "validate", "--model", args.fp16_model,
                                "--device", args.device, "--against", "transformers", "--output-directory", str(validation_directories[-1])])
    phase("replay-trace", ["python", "-m", "illm", "replay", str(capture), "--mode", "trace", "--step-id", "0"])
    phase("replay-steps", ["python", "-m", "illm", "replay", str(capture), "--mode", "state", "--assert-state-digest"])
    benchmark = output / "benchmark"
    phase("benchmark", ["python", "scripts/benchmark_reference.py", "--model", args.model, "--device", args.device,
                         "--output", str(benchmark), "--contexts", "8", "32", "128", "--warmups", "3", "--repetitions", "7"])
    gpu_failure_evidence = None
    if args.profile_id == "real-model-single-gpu-v1":
        gpu_failure_evidence = output / "gpu-oom"
        phase("gpu-oom", ["python", "scripts/probe_gpu_oom.py", "--model", args.model, "--device", args.device,
                          "--output", str(gpu_failure_evidence)])
    toolchain = json.loads(manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1, "status": "pass", "fresh_environment": True, "offline_bootstrap": True,
        "network_used": "not_observed", "network_blocking": "not_enforced",
        "network_policy": "offline_resolution_and_local_model_only",
        "toolchain_manifest_digest": toolchain["manifest_digest"],
        "source_identity": {"project_root_digest": toolchain["project_root_digest"], "bootstrap_digest": toolchain["bootstrap_digest"],
                            "base_lock_digest": lock_digest(project)},
        "commands": commands,
    }
    report["report_digest"] = sha256_bytes(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    atomic_write_json(output / "gate-report.json", report)
    qualify = ["python", "scripts/qualify_v01.py", "--profile-id", args.profile_id, "--toolchain-manifest", str(manifest),
               "--junit", str(output / "junit.xml"), "--quality", str(output / "quality.json"), "--capture", str(capture),
               "--benchmark", str(benchmark), "--sampler-fixture", "fixtures/sampling/sampler-known-answer-v1.json",
               "--output", str(output / "qualification")]
    for directory in validation_directories:
        qualify.extend(["--validation-directory", str(directory)])
    if args.profile_id == "clean-room-release-v1":
        qualify.extend(["--clean-room-report", str(output / "gate-report.json")])
    if gpu_failure_evidence is not None:
        qualify.extend(["--gpu-failure-evidence", str(gpu_failure_evidence)])
    phase("qualify", qualify)
    (output / "COMPLETE").write_text("complete\n", encoding="ascii")
    (output / "INCOMPLETE").unlink()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_gate(argv)
    except (RuntimeError, OSError, ValueError) as exc:
        print(json.dumps({"status": "fail", "error_type": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the approved CPU-only policy profile in a fresh locked environment.

No GPU command is defined, no dependency is downloaded, and no release/tag is
created. Existing environments, attempts and failed outputs are never replaced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from _toolchain_common import PROFILE_PATTERN, atomic_write_json, lock_digest, sha256_bytes, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> tuple[
    dict[str, str], Callable[[str], dict[str, dict[str, Any]]], Callable[[str, Path], dict[str, list[str]]],
]:
    path = PROJECT_ROOT / "src/illm/stateful_policy_gate_contract.py"
    spec = importlib.util.spec_from_file_location("illm_v02_policy_gate_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError("policy gate contract is absent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (cast(dict[str, str], module.POLICY_PROFILE_RUNTIME_IDS),
            cast(Callable[[str], dict[str, dict[str, Any]]], module.policy_phase_parameters),
            cast(Callable[[str, Path], dict[str, list[str]]], module.policy_phase_commands))


def _plain(path: Path) -> Path:
    requested = Path(os.path.abspath(path))
    for item in (requested, *requested.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("managed output/environment cannot follow a link or reparse point")
    return requested


def run_gate(argv: Sequence[str] | None = None) -> int:
    runtimes, parameters_for, commands_for = _contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-id", required=True, choices=tuple(runtimes))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-cache", type=Path)
    args = parser.parse_args(argv)
    # Fail before subprocess/environment/output creation for the GPU profile.
    parameters = parameters_for(args.profile_id)
    runtime = runtimes[args.profile_id]
    if PROFILE_PATTERN.fullmatch(args.profile) is None:
        parser.error("--profile must be a safe managed environment name")
    if (runtime.startswith("py312-win") and platform.system() != "Windows") or (
        runtime.startswith("py312-linux") and platform.system() != "Linux"
    ):
        parser.error("CPU profile belongs to another operating system")
    project = _plain(PROJECT_ROOT)
    environment_root = _plain(project / ".illm/envs" / args.profile)
    output = _plain(args.output)
    if environment_root.exists() or not environment_root.is_relative_to(project / ".illm/envs"):
        parser.error("profile requires a new managed environment; existing environments are preserved")
    if not output.is_relative_to(project / "out") or output == project / "out" or output.exists():
        parser.error("--output must be a new directory strictly inside project out/")
    commands = commands_for(args.profile_id, output)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    incomplete = output / "INCOMPLETE"
    atomic_write_json(incomplete, {"reason": "gate_in_progress", "profile_id": args.profile_id})
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST"):
        environment.pop(name, None)
    environment.update(
        PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="",
    )
    manifest, sidecar = output / "toolchain.json", output / "toolchain.sha256"
    prefix = [sys.executable, str(project / "scripts/in_env.py"), "--manifest", str(manifest),
              "--expected-manifest-digest-file", str(sidecar), "--"]
    records: list[dict[str, Any]] = []

    def phase(name: str, command: list[str], *, bootstrap: bool = False) -> None:
        actual = command if bootstrap else [*prefix, *command]
        process = subprocess.run(actual, cwd=project, env=environment, capture_output=True, check=False)
        stdout, stderr = logs / f"{name}.stdout.txt", logs / f"{name}.stderr.txt"
        with stdout.open("xb") as stream:
            stream.write(process.stdout)
        with stderr.open("xb") as stream:
            stream.write(process.stderr)
        record = {"phase": name, "argv": actual, "parameters": parameters[name], "returncode": process.returncode,
                  "stdout_file": stdout.relative_to(output).as_posix(), "stderr_file": stderr.relative_to(output).as_posix(),
                  "stdout_sha256": sha256_file(stdout), "stderr_sha256": sha256_file(stderr)}
        records.append(record)
        atomic_write_json(output / "execution.json", {
            "schema_version": 3, "contract_revision": 2, "profile_id": args.profile_id, "commands": records,
        })
        print(json.dumps({"phase": name, "returncode": process.returncode,
                          "stdout": record["stdout_file"], "stderr": record["stderr_file"]}, sort_keys=True), flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"policy CPU phase {name} failed; original partial evidence is retained")

    try:
        bootstrap = [sys.executable, str(project / "scripts/bootstrap.py"), "--profile", args.profile,
                     "--runtime-profile", runtime, "--locked", "--offline", "--require-new-environment",
                     "--write-manifest", str(manifest), "--write-manifest-digest", str(sidecar)]
        if args.artifact_cache is not None:
            bootstrap.extend(["--artifact-cache", str(_plain(args.artifact_cache))])
        phase("bootstrap", bootstrap, bootstrap=True)
        for name, command in commands.items():
            phase(name, command)
        toolchain = json.loads(manifest.read_text(encoding="utf-8"))
        contract = parameters["bootstrap"]["contract"]
        report: dict[str, Any] = {
            "schema_version": 3, "report_kind": "illm-v02-policy-profile-gate-v1", "contract_revision": 2,
            "runtime_profile_id": runtime, "profile_id": args.profile_id, "status": "pass", "fresh_environment": True,
            "offline_bootstrap": True, "network_used": "not_observed", "network_blocking": "not_enforced",
            "network_policy": "offline_resolution_and_local_model_only", "toolchain_manifest_digest": toolchain["manifest_digest"],
            "source_identity": {"project_root_digest": toolchain["project_root_digest"],
                                "bootstrap_digest": toolchain["bootstrap_digest"], "base_lock_digest": lock_digest(project)},
            "protocol_sha256": contract["protocol"]["sha256"], "approval_sha256": contract["approval"]["sha256"],
            "commands": records, "qualification_status": "unqualified", "global_qualification_granted": False,
            "gpu_qualification_granted": False, **{axis: "incomplete" for axis in contract["result_axes"]},
        }
        report["report_digest"] = sha256_bytes(json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"))
        atomic_write_json(output / "gate-report.json", report)
        process = subprocess.run([
            *prefix, "python", "scripts/qualify_v02_policy.py", "--profile-id", args.profile_id,
            "--gate-root", str(output), "--output", str(output / "qualification"),
        ], cwd=project, env=environment, capture_output=True, check=False)
        for stream_name, payload in (("stdout", process.stdout), ("stderr", process.stderr)):
            with (output / f"qualify.{stream_name}.txt").open("xb") as stream:
                stream.write(payload)
        print(json.dumps({"phase": "seal-cpu-profile", "returncode": process.returncode}, sort_keys=True), flush=True)
        if process.returncode != 0 or not (output / "qualification/COMPLETE").is_file():
            raise RuntimeError("CPU profile sealing failed; inspect retained qualification output/logs")
        with (output / "COMPLETE").open("xb") as marker:
            marker.write(b"cpu-profile-complete;global-unqualified;gpu-not-run\n")
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
        print(json.dumps({"status": "fail", "exception_type": type(exc).__name__, "message": str(exc)},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

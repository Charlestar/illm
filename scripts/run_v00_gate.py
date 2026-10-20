from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run an isolated V0.0 platform release gate")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / "out") or output == root / "out" or output.exists():
        parser.error("--output must be a new directory below this checkout's out/")
    output.mkdir(parents=True)
    manifest = output / "toolchain.json"
    sidecar = output / "toolchain.sha256"
    transcript: list[dict[str, object]] = []
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"

    def execute(command: list[str], label: str, timeout: int = 300) -> None:
        print(f"[{label}]", flush=True)
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                env=child_environment)
        (output / f"{label}.stdout.txt").write_bytes(result.stdout.encode("utf-8"))
        (output / f"{label}.stderr.txt").write_bytes(result.stderr.encode("utf-8"))
        transcript.append({"phase": label, "argv": command, "returncode": result.returncode})
        (output / "execution.json").write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

    def run(child: list[str], label: str) -> None:
        execute([sys.executable, "scripts/in_env.py", "--manifest", str(manifest),
                 "--expected-manifest-digest-file", str(sidecar), "--", "python", *child], label)

    execute([sys.executable, "scripts/bootstrap.py", "--profile", args.profile, "--locked",
             "--write-manifest", str(manifest), "--write-manifest-digest", str(sidecar)], "bootstrap")
    run(["scripts/check_quality.py", "--output", str(output / "quality.json")], "quality")
    run(["-m", "pytest", "-p", "no:cacheprovider", "tests/cpu", "--tb=short",
         "--basetemp", str(output / "pytest-temp"), "--junitxml", str(output / "junit.xml")], "tests")
    run(["-m", "illm", "demo", "toy-scheduler", "--seed", "7", "--steps", "20",
         "--capture", str(output)], "demo")
    run(["-m", "illm", "replay", str(output), "--mode", "trace"], "r0")
    run(["-m", "illm", "replay", str(output), "--mode", "state", "--assert-state-digest"], "r1")
    run(["scripts/qualify_v00.py", "--toolchain-manifest", str(manifest),
         "--junit", str(output / "junit.xml"), "--quality", str(output / "quality.json"),
         "--capture", str(output), "--output", str(output / "qualification")], "qualify")
    if args.rollback:
        rollback_manifest = output / "rollback-toolchain.json"
        rollback_sidecar = output / "rollback-toolchain.sha256"
        execute([sys.executable, "scripts/bootstrap.py", "--profile", args.profile + "-rollback",
                 "--locked", "--write-manifest", str(rollback_manifest),
                 "--write-manifest-digest", str(rollback_sidecar)], "rollback-bootstrap")
        execute([sys.executable, "scripts/in_env.py", "--manifest", str(rollback_manifest),
                 "--expected-manifest-digest-file", str(rollback_sidecar), "--", "python",
                 "scripts/rollback_v00.py", "--toolchain-manifest", str(rollback_manifest),
                 "--evidence-dir", "release/v0.0.0/rollback"], "rollback")
    print(json.dumps({"platform": platform.system(), "output": str(output), "status": "pass"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

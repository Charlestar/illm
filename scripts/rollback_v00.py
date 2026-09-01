from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rehearse rollback to the pre-V0.0 empty runtime")
    parser.add_argument("--toolchain-manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    evidence_dir = args.evidence_dir.resolve()
    release_root = (project_root / "release" / "v0.0.0").resolve()
    if evidence_dir != release_root / "rollback":
        raise SystemExit("rollback evidence must target release/v0.0.0/rollback")
    manifest_path = args.toolchain_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise SystemExit("rollback must run with the manifest interpreter")
    sentinel_path = Path(manifest["environment_root"]).resolve() / ".illm-toolchain.json"
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if sentinel.get("state") != "ready":
        raise SystemExit("rollback environment is not ready")
    installed_before = importlib.util.find_spec("illm") is not None
    process = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "illm"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    importlib.invalidate_caches()
    installed_after = importlib.util.find_spec("illm") is not None
    success = installed_before and process.returncode == 0 and not installed_after
    report = {
        "schema_version": 0,
        "source_release": "v0.0.0",
        "target_state": "pre-v0.0-empty-runtime",
        "toolchain_manifest": str(manifest_path),
        "toolchain_manifest_digest": manifest.get("manifest_digest"),
        "environment_id": manifest.get("environment_id"),
        "python_executable": sys.executable,
        "installed_before": installed_before,
        "pip_returncode": process.returncode,
        "installed_after": installed_after,
        "background_processes_started": 0,
        "ports_opened": 0,
        "success": success,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(evidence_dir / "manifest.json", report)
    (evidence_dir / "commands.txt").write_text(
        "python scripts/in_env.py --manifest out/v0.0/rollback-toolchain.json -- "
        "python scripts/rollback_v00.py --toolchain-manifest out/v0.0/rollback-toolchain.json "
        "--evidence-dir release/v0.0.0/rollback\n",
        encoding="utf-8",
    )
    (evidence_dir / "uninstall.stdout.txt").write_text(process.stdout, encoding="utf-8")
    (evidence_dir / "uninstall.stderr.txt").write_text(process.stderr, encoding="utf-8")
    (evidence_dir / "rollback-report.md").write_text(
        "# V0.0 rollback rehearsal\n\n"
        f"- Installed before rollback: `{installed_before}`\n"
        f"- pip uninstall exit code: `{process.returncode}`\n"
        f"- Importable after rollback: `{installed_after}`\n"
        f"- Background processes/ports created: `0/0`\n"
        f"- Result: `{'PASS' if success else 'FAIL'}`\n",
        encoding="utf-8",
    )
    sentinel["state"] = "rolled_back"
    sentinel["rollback_evidence"] = str(evidence_dir / "manifest.json")
    _atomic_json(sentinel_path, sentinel)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Collect approved tiny CPU raw evidence; never authorize a GPU or release.

Run collection with the finalized source-bound CPU interpreter through in_env.
Verification of an existing bundle uses only stdlib and its copied reader.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


_READER_PROGRAM = r'''
import builtins, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
attempted, network_events = [], []
forbidden = ("torch", "transformers", "tokenizers", "safetensors")
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".", 1)[0] in forbidden:
        attempted.append(name)
        raise RuntimeError("tensor runtime import is forbidden in raw reader")
    return original_import(name, globals, locals, fromlist, level)
def audit(event, args):
    if event.startswith("socket."):
        network_events.append(event)
        raise RuntimeError("network is forbidden in raw reader")
builtins.__import__ = guarded_import
sys.addaudithook(audit)
reader_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(reader_root))
from illm.canonical import canonical_json_bytes, sha256_json
from illm.policy_cpu_raw_bundle import verify_payload, verify_bundle
root = Path(sys.argv[2])
report = verify_bundle(root) if sys.argv[3] == "sealed" else verify_payload(root)
modules = {}
for name, module in sorted(sys.modules.items()):
    if name != "illm" and not name.startswith("illm."):
        continue
    path = Path(module.__file__).resolve()
    if path.suffix != ".py":
        raise RuntimeError("reader module is not Python source")
    modules[name] = path.relative_to(reader_root).as_posix()
loaded = sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
if attempted or loaded or network_events:
    raise RuntimeError("reader isolation violation")
result = {"schema_version": 1, "report_digest": sha256_json(report),
          "comparison_records": report["comparison_records"], "module_files": modules,
          "forbidden_import_attempts": attempted, "forbidden_loaded_modules": loaded,
          "network_events": network_events}
print(canonical_json_bytes(result).decode("utf-8"))
'''


def _imports() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))


def _write(path: Path, raw: bytes) -> None:
    from illm.policy_cpu_numerical_contract import plain_path

    with plain_path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _environment() -> dict[str, str]:
    result = dict(os.environ)
    result.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return result


def _run(root: Path, label: str, argv: list[str], cwd: Path) -> dict[str, Any]:
    from illm.policy_cpu_raw_bundle import file_identity
    from illm.policy_cpu_numerical_contract import NumericalEvidenceError

    with (root / "logs" / f"{label}.stdout.log").open("xb") as stdout, (
        root / "logs" / f"{label}.stderr.log"
    ).open("xb") as stderr:
        result = subprocess.run(argv, cwd=cwd, env=_environment(), stdout=stdout, stderr=stderr, check=False)
    command = {"label": label, "argv": argv, "cwd": str(cwd), "returncode": result.returncode,
               **{stream: file_identity(root / "logs" / f"{label}.{stream}.log", f"logs/{label}.{stream}.log")
                  for stream in ("stdout", "stderr")}}
    # Keep each command even if a later worker fails. No failed attempt is sealed.
    from illm.canonical import canonical_json_bytes
    _write(root / "logs" / f"{label}.command.json", canonical_json_bytes(command) + b"\n")
    if result.returncode != 0:
        raise NumericalEvidenceError(f"CPU subprocess {label} failed; INCOMPLETE and logs retained")
    return command


def _copy_authorities(project: Path, root: Path) -> None:
    from illm.policy_cpu_numerical_contract import (
        PROTOCOL_SHA256, NumericalEvidenceError, cpu_cases_from_snapshots, plain_path,
    )
    from illm.policy_cpu_raw_bundle import APPROVAL_SHA256, AUTHORITY_NAMES, file_identity

    fixture = project / "fixtures/tiny-decoder"
    sources = {
        "inputs.json": project / "docs/implementation/v0.2-policy-inputs-draft.json",
        "oracle-manifest.json": fixture / "oracle-v1/oracle-manifest.json",
        **{name: fixture / "oracle-v1" / name for name in AUTHORITY_NAMES if name.endswith(".safetensors")
           and name != "model.safetensors"},
        **{name: fixture / name for name in AUTHORITY_NAMES if name in {
            "config.json", "model.safetensors", "supported-model-spec.json", "weight-load-manifest.json",
            "tokenizer.json", "fixture-manifest.json"}},
        "protocol.md": project / "docs/implementation/v0.2-numerical-policy-protocol.md",
        "approval.json": project / "docs/implementation/v0.2-policy-approval.json",
    }
    for name in AUTHORITY_NAMES:
        source = plain_path(sources[name])
        if not source.is_file():
            raise NumericalEvidenceError("CPU authority is not a plain file")
        _write(root / "authorities" / name, source.read_bytes())
    for name, digest in (("protocol.md", PROTOCOL_SHA256), ("approval.json", APPROVAL_SHA256)):
        if file_identity(root / "authorities" / name, name)["sha256"] != digest:
            raise NumericalEvidenceError("CPU human approval or protocol differs")
    cpu_cases_from_snapshots((root / "authorities/inputs.json").read_bytes(),
                             (root / "authorities/oracle-manifest.json").read_bytes())


def collect(project: Path, output: Path) -> dict[str, Any]:
    from illm.canonical import canonical_json_bytes, sha256_json
    from illm.policy_binding import build_execution_binding
    from illm.policy_cpu_numerical_contract import NumericalEvidenceError, plain_path, read_json, write_json_exclusive
    from illm.policy_cpu_raw_bundle import BUNDLE_KIND, inventory, verify_seal, worker_program
    from illm.policy_cpu_raw_validation import derive_cpu_numerical

    project = plain_path(project)
    binding = build_execution_binding(project)
    if binding.to_dict()["toolchain"]["binding"] != "source-bound-manifest":
        raise NumericalEvidenceError("raw CPU collection requires a finalized source-bound manifest via in_env")
    destination = plain_path(output)
    if os.path.lexists(destination) or not destination.parent.is_dir():
        raise NumericalEvidenceError("CPU output must be new and have an existing parent")
    if destination == project / "out" or not destination.is_relative_to(project / "out"):
        raise NumericalEvidenceError("CPU output must be a new descendant of project_root/out")
    destination.mkdir()
    _write(destination / "INCOMPLETE", b"cpu_raw_collection_in_progress\n")
    for directory in ("authorities", "workers", "logs", "reader"):
        (destination / directory).mkdir()
    _copy_authorities(project, destination)
    commands = []
    roles = (("native", "runtime", "native"), ("candidate", "runtime", "candidate"),
             ("formula", "formula", None), ("mutations", "mutation", None),
             ("reproduction", "runtime", "candidate-reproduction"))
    for label, module, role in roles:
        program = worker_program(module)
        argv = [sys.executable, "-I", "-X", "utf8", "-c", program, str(project / "src"),
                "--project-root", str(project), "--output", str(destination / "workers" / label)]
        if role is not None:
            argv.extend(("--role", role))
        commands.append(_run(destination, label, argv, project))
    # All numerical conclusions are calculated from raw bytes, never producer booleans.
    report = derive_cpu_numerical(destination, records_path=destination / "comparisons.jsonl")
    write_json_exclusive(destination / "numerical-report.json", report)
    source_argv = [sys.executable, "-I", "-S", "-X", "utf8", "-c", _READER_PROGRAM,
                   str(project / "src"), str(destination), "payload"]
    commands.append(_run(destination, "reader-source", source_argv, destination / "authorities"))
    source_result = read_json(destination / "logs/reader-source.stdout.log")
    from illm.policy_cpu_raw_bundle import safe_relative
    for relative in source_result["module_files"].values():
        safe_relative(relative)
        source = plain_path(project / "src" / relative)
        target = plain_path(destination / "reader" / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write(target, source.read_bytes())
    copy_argv = [sys.executable, "-I", "-S", "-X", "utf8", "-c", _READER_PROGRAM,
                 str(destination / "reader"), str(destination), "payload"]
    commands.append(_run(destination, "reader-copy", copy_argv, destination / "reader"))
    if source_result != read_json(destination / "logs/reader-copy.stdout.log"):
        raise NumericalEvidenceError("fresh source/copy reader results differ")
    write_json_exclusive(destination / "commands.json", commands)
    # Individual transcripts remain members so an interrupted run is inspectable.
    manifest = {"schema_version": 1, "bundle_kind": BUNDLE_KIND, "evidence_stage": "DEVELOPMENT_ONLY",
                "qualification_status": "unqualified", "binding": binding.to_dict(), "binding_digest": binding.digest,
                "files": inventory(destination, exclude=frozenset({"INCOMPLETE"}))}
    manifest["manifest_digest"] = sha256_json(manifest)
    write_json_exclusive(destination / "bundle-manifest.json", manifest)
    _write(destination / "COMPLETE", (str(manifest["manifest_digest"]) + "\n").encode("ascii"))
    (destination / "INCOMPLETE").unlink()
    try:
        verify_seal(destination)
    except BaseException:
        # Never leave a supposedly complete container after a failed sealer.
        (destination / "COMPLETE").unlink(missing_ok=True)
        _write(destination / "INCOMPLETE", b"cpu_raw_seal_verification_failed\n")
        raise
    # A complete evidence container is not a numerically qualified release.
    return {"status": report["status"], "cpu_results": report["cpu_results"],
            "manifest_digest": manifest["manifest_digest"], "report_kind": report["report_kind"],
            "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify-existing", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    _imports()
    from illm.canonical import canonical_json_bytes
    from illm.policy_cpu_numerical_contract import NumericalEvidenceError, plain_path
    try:
        if args.verify_existing is not None:
            root = plain_path(args.verify_existing)
            from illm.policy_cpu_raw_bundle import verify_seal
            verify_seal(root)
            process = subprocess.run(
                [sys.executable, "-I", "-S", "-X", "utf8", "-c", _READER_PROGRAM,
                 str(root / "reader"), str(root), "sealed"], cwd=root,
                env=_environment(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            sys.stdout.buffer.write(process.stdout)
            sys.stderr.buffer.write(process.stderr)
            return process.returncode
        result = collect(args.project_root, args.output)
    except (NumericalEvidenceError, OSError, ValueError) as exc:
        print(canonical_json_bytes({"status": "error", "message": str(exc), "error_type": type(exc).__name__,
                                    "evidence_stage": "DEVELOPMENT_ONLY", "qualification_status": "unqualified"}).decode(),
              file=sys.stderr)
        return 1
    print(canonical_json_bytes(result).decode("utf-8"))
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

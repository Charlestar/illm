"""Actual tiny CPU CLI checks; unbound test runs are not profile evidence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes
from illm.policy_cpu_numerical_contract import NumericalEvidenceError
from illm.stateful_policy_gate_contract import policy_phase_commands
from illm.stateful_policy_qualification import _demo


ROOT = Path(__file__).resolve().parents[2]


def _run_demos(output: Path) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("actual tiny policy CLI needs the optional locked CPU runtime")
    logs = output / "gate/logs"
    logs.mkdir(parents=True)
    environment = os.environ.copy()
    for name in ("ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST", "PYTHONHOME"):
        environment.pop(name, None)
    environment.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    commands = policy_phase_commands("state-machine-cpu-v1", output)
    for phase in ("demo-b", "demo-c", "replay-c", "inspect-c"):
        process = subprocess.run([sys.executable, *commands[phase][1:]], cwd=ROOT, env=environment,
                                 capture_output=True, check=False)
        assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
        (logs / f"{phase}.stdout.txt").write_bytes(process.stdout)
        (logs / f"{phase}.stderr.txt").write_bytes(process.stderr)
    manifest = json.loads((output / "demo-c/capture-manifest.json").read_bytes())
    binding: dict[str, Any] = manifest["execution_binding"]
    assert binding["toolchain"]["binding"] == "unbound-development"
    return binding


def test_actual_candidate_b_c_cli_match_fixed_public_sequences_and_capture_story(tmp_path: Path) -> None:
    binding = _run_demos(tmp_path)
    b, c = _demo(tmp_path, "demo-b", binding), _demo(tmp_path, "demo-c", binding)
    assert b["manifest_digest"] != c["manifest_digest"]
    assert b["memory_plan_digest"] != c["memory_plan_digest"]
    assert b["event_count"] > 0 and c["event_count"] > 0


def test_actual_demo_uses_literal_finish_reason_not_a_normalized_alias(tmp_path: Path) -> None:
    binding = _run_demos(tmp_path)
    path = tmp_path / "gate/logs/demo-c.stdout.txt"
    chunks = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert chunks[-1]["finish_reason"] == "MAX_NEW_TOKENS"
    assert chunks[-1]["result"]["finish_reason"] == "MAX_NEW_TOKENS"
    original = canonical_json_bytes(chunks)
    chunks[-1]["finish_reason"] = "length"
    chunks[-1]["result"]["finish_reason"] = "length"
    path.write_bytes(b"\n".join(canonical_json_bytes(chunk) for chunk in chunks) + b"\n")
    with pytest.raises(NumericalEvidenceError, match="metadata/terminal/result envelope"):
        _demo(tmp_path, "demo-c", binding)
    path.write_bytes(b"\n".join(canonical_json_bytes(chunk) for chunk in json.loads(original)) + b"\n")
    for phase in ("replay-c", "inspect-c"):
        story_path = tmp_path / "gate/logs" / (phase + ".stdout.txt")
        original_story = story_path.read_bytes()
        story_path.write_bytes(canonical_json_bytes({"schema_version": 2, "capability": "R0", "invented": True}) + b"\n")
        with pytest.raises(NumericalEvidenceError, match="capture-derived story"):
            _demo(tmp_path, "demo-c", binding)
        story_path.write_bytes(original_story)
    restored = json.loads(original)
    restored[0]["unapproved_payload"] = "not part of the closed public fixture"
    path.write_bytes(b"\n".join(canonical_json_bytes(chunk) for chunk in restored) + b"\n")
    with pytest.raises(NumericalEvidenceError, match="metadata/terminal/result envelope"):
        _demo(tmp_path, "demo-c", binding)

"""CPU-only tests for the independent D-high raw evidence worker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from illm.policy_cpu_numerical_contract import canonical_tiny_ops, high_cases, tensor_id
from illm.policy_tensor_evidence import TensorPack


ROOT = Path(__file__).resolve().parents[2]


def _environment() -> dict[str, str]:
    result = dict(os.environ)
    result["PYTHONPATH"] = str(ROOT / "src")
    result["CUDA_VISIBLE_DEVICES"] = ""
    return result


def _run(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "illm.policy_cpu_formula_worker",
            "--project-root",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )


def _report(output: Path) -> dict[str, Any]:
    value = json.loads((output / "worker.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_formula_worker_writes_complete_raw_d_high_bundle(tmp_path: Path) -> None:
    output = tmp_path / "formula"
    completed = _run(output)
    assert completed.returncode == 0, completed.stderr
    assert set(path.name for path in output.iterdir()) == {"pack", "worker.json", "COMPLETE"}

    report_raw = (output / "worker.json").read_bytes()
    assert (output / "COMPLETE").read_bytes() == (
        hashlib.sha256(report_raw).hexdigest() + "\n"
    ).encode("ascii")
    report = _report(output)
    assert set(report) == {
        "schema_version", "report_kind", "status", "evidence_stage",
        "qualification_status", "role", "coverage", "binding", "binding_digest",
        "process", "model", "policy", "pack_manifest_digest", "cases", "high_cases",
    }
    assert report["schema_version"] == 1
    assert report["report_kind"] == "illm-policy-cpu-raw-worker-v1"
    assert report["status"] == "pass"
    assert report["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert report["qualification_status"] == "unqualified"
    assert report["role"] == "formula"
    assert report["cases"] == []
    assert report["policy"] is None
    assert report["model"]["candidate_loaded"] is False
    assert report["process"]["cuda_internal_initialized_start"] is False
    assert report["process"]["cuda_internal_initialized_end"] is False
    assert report["process"]["forbidden_modules_loaded_start"] == []
    assert report["process"]["forbidden_modules_loaded_end"] == []

    pack = TensorPack.open(output / "pack")
    expected_count = 39 * sum(
        len(case["positions"]) for case in high_cases() if case["mode"] == "forward"
    ) + 4
    assert expected_count == 11860
    assert len(pack.records) == expected_count
    assert report["coverage"]["tensor_count"] == expected_count

    cases = report["high_cases"]
    assert [case["case_id"] for case in cases] == [f"high-{index:02d}" for index in range(8)]
    for case, contract in zip(cases, high_cases(), strict=True):
        assert set(case) == {"case_id", "mode", "input_token_ids", "positions", "paths"}
        assert case["case_id"] == contract["case_id"]
        assert case["mode"] == contract["mode"]
        assert case["input_token_ids"] == contract["tokens"]
        assert case["positions"] == contract["positions"]
        if case["mode"] == "forward":
            assert set(case["paths"]) == set(canonical_tiny_ops())
            for op in canonical_tiny_ops():
                assert set(case["paths"][op]) == {f"{position:03d}" for position in contract["positions"]}
                for position in contract["positions"]:
                    identity = tensor_id("D-high", case["case_id"], 0, op, position)
                    assert case["paths"][op][f"{position:03d}"] == identity
                    assert identity in pack.records
        else:
            assert set(case["paths"]) == {"input_q", "input_k", "rope_q", "rope_k"}
            for label, identity in case["paths"].items():
                assert identity == tensor_id("D-high", "high-05", 0, label, None)
                assert identity in pack.records
            assert pack.records[case["paths"]["input_q"]]["dtype"] == "fp32"
            assert pack.records[case["paths"]["input_k"]]["dtype"] == "fp32"
            assert pack.records[case["paths"]["rope_q"]]["dtype"] == "fp64"
            assert pack.records[case["paths"]["rope_k"]]["dtype"] == "fp64"


def test_formula_worker_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    output = tmp_path / "foreign"
    output.mkdir()
    foreign = output / "owner.txt"
    foreign.write_bytes(b"foreign-owner\n")
    completed = _run(output)
    assert completed.returncode != 0
    assert foreign.read_bytes() == b"foreign-owner\n"
    assert set(output.iterdir()) == {foreign}


def test_formula_worker_failure_retains_incomplete_bundle(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    script = "\n".join(
        (
            "from pathlib import Path",
            "from illm.model.high_precision import HighPrecisionOracle",
            "from illm.policy_cpu_formula_worker import run_formula_worker",
            "def fail(*args, **kwargs): raise RuntimeError('injected formula failure')",
            "HighPrecisionOracle.forward = fail",
            f"run_formula_worker(Path({str(ROOT)!r}), Path({str(output)!r}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert (output / "INCOMPLETE").read_bytes() == b"formula_worker_in_progress\n"
    assert not (output / "COMPLETE").exists()


def test_formula_worker_refuses_a_shared_mutation_process(tmp_path: Path) -> None:
    output = tmp_path / "shared-process"
    script = "\n".join(
        (
            "from pathlib import Path",
            "import illm.policy_cpu_mutation_worker",
            "from illm.policy_cpu_formula_worker import run_formula_worker",
            f"run_formula_worker(Path({str(ROOT)!r}), Path({str(output)!r}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert not output.exists()


def test_formula_worker_rejects_link_ancestor_when_supported(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(ordinary, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable on this host")
    completed = _run(linked / "formula")
    assert completed.returncode != 0
    assert list(ordinary.iterdir()) == []

"""CPU-only tests for executable numerical-policy mutation evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import pytest

from illm.policy_cpu_numerical_contract import FIXED_MUTATION_IDS, tensor_id
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
            "illm.policy_cpu_mutation_worker",
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


def test_mutation_worker_executes_all_frozen_counterexamples(tmp_path: Path) -> None:
    output = tmp_path / "mutations"
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
        "process", "model", "policy", "pack_manifest_digest", "mutations",
    }
    assert report["schema_version"] == 1
    assert report["report_kind"] == "illm-policy-cpu-raw-worker-v1"
    assert report["status"] == "pass"
    assert report["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert report["qualification_status"] == "unqualified"
    assert report["role"] == "mutation"
    assert report["process"]["cuda_internal_initialized_start"] is False
    assert report["process"]["cuda_internal_initialized_end"] is False
    assert report["process"]["formula_worker_module_loaded_start"] is False
    assert report["process"]["formula_worker_module_loaded_end"] is False

    mutations = report["mutations"]
    assert [item["mutation_id"] for item in mutations] == list(FIXED_MUTATION_IDS)
    assert all(item["status"] == "detected" for item in mutations)
    assert report["coverage"]["all_mutations_detected"] is True
    pack = TensorPack.open(output / "pack")
    assert report["coverage"]["mutation_count"] == 9

    for item in mutations:
        assert set(item) == {
            "mutation_id", "status", "input_identity", "modified_input_identity",
            "expected_detector", "observation", "tensor_paths", "resource_fence",
        }
        assert item["input_identity"]["sha256"] != item["modified_input_identity"]["sha256"]
        for label, identity in item["tensor_paths"].items():
            assert identity == tensor_id("D-mutation", item["mutation_id"], 0, label, None)
            assert identity in pack.records

    by_id = {item["mutation_id"]: item for item in mutations}
    assert by_id["transpose-square-q-weight"]["observation"]["name"] == "layer.0.q_projection"
    assert by_id["decode-position-plus-one"]["observation"]["name"] == "layer.0.rope_q"
    assert by_id["decode-position-plus-one"]["observation"]["position"] == 2
    assert by_id["wrong-gqa-group"]["observation"]["scores_changed"] is True
    assert by_id["wrong-gqa-group"]["observation"]["context_changed"] is True
    assert by_id["future-key-visible"]["observation"]["error"]["code"] == "INVALID_ARGUMENT"
    assert by_id["corrupt-committed-prefix-k"]["observation"]["b_native_within_tolerance"] is True
    assert by_id["corrupt-committed-prefix-k"]["observation"]["candidate_native_within_tolerance"] is False
    for mutation_id in ("decode-position-plus-one", "corrupt-committed-prefix-k"):
        resource = by_id[mutation_id]["resource_fence"]
        assert resource["required"] is True
        assert resource["fence_complete"] is True
        assert resource["fence_succeeded"] is True
        assert resource["active_leases_after_cleanup"] == 0
        assert resource["store_closed"] is True
    for mutation_id in ("wrong-policy-digest", "unknown-policy-field"):
        assert by_id[mutation_id]["observation"]["phase"] == "pre-acceptance"
        assert by_id[mutation_id]["tensor_paths"] == {}
    assert by_id["fused-bias"]["observation"]["attempted_bias"] is not None

    early = by_id["early-storage-cast"]["observation"]["attempted_bf16_raw"]
    for value in (early["input"], early["weight"]):
        raw = bytes.fromhex(value["raw_hex"])
        assert value["dtype"] == "torch.bfloat16"
        assert value["raw_hex"] == value["raw_hex"].lower()
        assert len(raw) == value["nbytes"]
        assert len(value["raw_hex"]) == 2 * value["nbytes"]
        assert hashlib.sha256(raw).hexdigest() == value["sha256"]


def test_mutation_worker_refuses_existing_output_without_touching_it(tmp_path: Path) -> None:
    output = tmp_path / "foreign"
    output.mkdir()
    foreign = output / "owner.txt"
    foreign.write_bytes(b"foreign-owner\n")
    completed = _run(output)
    assert completed.returncode != 0
    assert foreign.read_bytes() == b"foreign-owner\n"
    assert set(output.iterdir()) == {foreign}


def test_mutation_worker_failure_retains_incomplete_bundle(tmp_path: Path) -> None:
    output = tmp_path / "failed"
    script = "\n".join(
        (
            "from pathlib import Path",
            "import illm.model",
            "from illm.policy_cpu_mutation_worker import run_mutation_worker",
            "def fail(*args, **kwargs): raise RuntimeError('injected mutation failure')",
            "illm.model.load_model = fail",
            f"run_mutation_worker(Path({str(ROOT)!r}), Path({str(output)!r}))",
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
    assert (output / "INCOMPLETE").read_bytes() == b"mutation_worker_in_progress\n"
    assert not (output / "COMPLETE").exists()


def test_mutation_worker_refuses_a_shared_formula_process(tmp_path: Path) -> None:
    output = tmp_path / "shared-process"
    script = "\n".join(
        (
            "from pathlib import Path",
            "import illm.policy_cpu_formula_worker",
            "from illm.policy_cpu_mutation_worker import run_mutation_worker",
            f"run_mutation_worker(Path({str(ROOT)!r}), Path({str(output)!r}))",
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


def test_mutation_worker_rejects_link_ancestor_when_supported(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(ordinary, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable on this host")
    completed = _run(linked / "mutations")
    assert completed.returncode != 0
    assert list(ordinary.iterdir()) == []

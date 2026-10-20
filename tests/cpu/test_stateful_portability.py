"""Portable readers cannot import tensor runtimes or equate booleans to counts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import IllmError
from illm.stateful_validation import verify_stateful_output


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def stateful_evidence(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("portable-stateful") / "evidence"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run([
        sys.executable, "scripts/validate_stateful.py", "--development-no-manifest", "--output", str(output),
    ], cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    return output


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _resign(root: Path) -> None:
    report_path = root / "report/stateful-validation-report.json"
    report = _read(report_path)
    raw_digests: dict[str, str] = {}
    for name, key in (("differential-cases", "differential"), ("fixed-prompts", "fixed_greedy")):
        path = root / f"raw/{name}.json"
        raw = _read(path)
        raw.pop("suite_digest", None)
        raw["suite_digest"] = sha256_json(raw)
        _write(path, raw)
        raw_digests[key] = raw["suite_digest"]
    report["raw_digests"] = raw_digests
    report.pop("report_digest", None)
    report["report_digest"] = sha256_json(report)
    _write(report_path, report)
    _write(root / "DEVELOPMENT_ONLY", {
        "qualification_tier": "unqualified", "formal_evidence_eligible": False, "report_digest": report["report_digest"],
    })
    ignored = {"checksums.json", "COMPLETE", "INCOMPLETE", "DEVELOPMENT_ONLY"}
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*"))
             if path.is_file() and path.name not in ignored}
    _write(root / "checksums.json", {"schema_version": 1, "files": files, "inventory_digest": sha256_json(files)})


def test_reader_verifies_actual_evidence_without_any_tensor_runtime(stateful_evidence: Path) -> None:
    # -S also removes installed site packages; this works with the stdlib only.
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from pathlib import Path; from illm.stateful_validation import verify_stateful_output; "
        "r=verify_stateful_output(Path(sys.argv[2]), require_formal=False); "
        "assert r['actual_counts']=={'differential':50,'fixed_greedy':20}; "
        "assert not {'torch','transformers','safetensors','tokenizers'} & set(sys.modules)"
    )
    process = subprocess.run([sys.executable, "-I", "-S", "-c", program, str(ROOT / "src"), str(stateful_evidence)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("mutation", [
    "report-schema-bool", "raw-schema-bool", "query-bool", "position-bool", "extra-file", "golden-token", "golden-authority",
])
def test_resigned_type_and_inventory_mutations_fail_closed(stateful_evidence: Path, tmp_path: Path, mutation: str) -> None:
    root = tmp_path / "evidence"
    shutil.copytree(stateful_evidence, root)
    report_path = root / "report/stateful-validation-report.json"
    raw_path = root / ("raw/fixed-prompts.json" if mutation.startswith("golden-") else "raw/differential-cases.json")
    report, raw = _read(report_path), _read(raw_path)
    if mutation == "report-schema-bool":
        report["schema_version"] = True
    elif mutation == "raw-schema-bool":
        raw["schema_version"] = True
    elif mutation == "query-bool":
        raw["cases"][0]["steps"][0]["query_tokens"]["contiguous"] = True
    elif mutation == "position-bool":
        raw["cases"][0]["steps"][0]["position"] = False
    elif mutation == "golden-token":
        changed = raw["cases"][0]["generated_tokens"][0] + 1
        raw["cases"][0]["generated_tokens"][0] = changed
        raw["cases"][0]["steps"][0]["chosen_token"] = {"none": changed, "contiguous": changed}
    elif mutation == "golden-authority":
        oracle_path = root / "authority/tiny-oracle.json"
        oracle = _read(oracle_path)
        oracle["greedy_cases"][0]["expected_tokens"][0] += 1
        oracle.pop("manifest_digest")
        oracle["manifest_digest"] = sha256_json(oracle)
        raw["tiny_oracle_manifest_digest"] = oracle["manifest_digest"]
        _write(oracle_path, oracle)
    else:
        _write(root / "unexpected.json", {"not_evidence": True})
    _write(report_path, report)
    _write(raw_path, raw)
    _resign(root)
    with pytest.raises(IllmError):
        verify_stateful_output(root, require_formal=False)


def test_duplicate_json_fields_fail_even_when_bytes_are_checksummed(stateful_evidence: Path, tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    shutil.copytree(stateful_evidence, root)
    report = root / "report/stateful-validation-report.json"
    report.write_bytes(b'{"schema_version":1,' + report.read_bytes().lstrip()[1:])
    checksums = _read(root / "checksums.json")
    checksums["files"]["report/stateful-validation-report.json"] = sha256_file(report)
    checksums["inventory_digest"] = sha256_json(checksums["files"])
    _write(root / "checksums.json", checksums)
    with pytest.raises(IllmError, match="unreadable"):
        verify_stateful_output(root, require_formal=False)


def test_shared_numerical_contract_has_not_changed() -> None:
    from illm.model.validation import FIXED_PROMPTS, TensorTolerance, _tolerances
    from illm.numerical_contracts import FIXED_PROMPTS as portable_prompts
    from illm.numerical_contracts import TensorTolerance as portable_tolerance
    from illm.numerical_contracts import tensor_tolerances

    assert FIXED_PROMPTS is portable_prompts
    assert TensorTolerance is portable_tolerance
    for dtype in ("fp32", "bf16", "fp16"):
        assert _tolerances(dtype) == tensor_tolerances(dtype)

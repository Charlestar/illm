from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import canonical_json_bytes, sha256_bytes
from illm.policy_cpu_numerical_contract import (
    NumericalEvidenceError,
    canonical_tiny_ops,
    cpu_cases,
    cpu_cases_from_snapshots,
    high_cases,
    read_authority_tensors,
    strict_json,
    tensor_id,
)


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "docs/implementation/v0.2-policy-inputs-draft.json"
ORACLE = ROOT / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json"


def test_cases_are_frozen_input_only_and_do_not_regenerate_goldens() -> None:
    cases = cpu_cases(ROOT)
    assert len(cases) == 70
    assert [case["case_id"] for case in cases[:50]] == [f"differential-{index:02d}" for index in range(50)]
    assert [case["case_id"] for case in cases[50:]] == [f"fixed-greedy-{index:02d}" for index in range(20)]
    assert cases[2]["max_new_tokens"] == 1 and len(cases[2]["input_token_ids"]) == 255
    assert all(case["expected_output_token_ids"] is None for case in cases[:50])
    assert all(len(case["expected_output_token_ids"]) == 8 for case in cases[50:])
    detached = cpu_cases(ROOT)
    detached[0]["input_token_ids"][0] = 99
    assert cpu_cases(ROOT)[0]["input_token_ids"] == [3]


def test_resigned_authority_and_different_input_bytes_cannot_become_new_goldens() -> None:
    authority = json.loads(ORACLE.read_bytes())
    authority["greedy_cases"][0]["expected_tokens"][0] = 99
    from illm.canonical import sha256_json
    authority["manifest_digest"] = sha256_json({key: value for key, value in authority.items() if key != "manifest_digest"})
    with pytest.raises(NumericalEvidenceError):
        cpu_cases_from_snapshots(INPUTS.read_bytes(), canonical_json_bytes(authority))
    with pytest.raises(NumericalEvidenceError):
        cpu_cases_from_snapshots(INPUTS.read_bytes() + b"\n", ORACLE.read_bytes())
    equivalent = json.dumps(json.loads(ORACLE.read_bytes()), indent=7).encode("utf-8")
    with pytest.raises(NumericalEvidenceError, match="frozen original bytes"):
        cpu_cases_from_snapshots(INPUTS.read_bytes(), equivalent)


def test_high05_is_only_rope_and_cases_keep_full_capacity_boundary() -> None:
    cases = high_cases()
    assert [case["case_id"] for case in cases] == [f"high-{index:02d}" for index in range(8)]
    assert cases[5] == {"case_id": "high-05", "mode": "rope_only", "tokens": [], "positions": [31, 32]}
    assert len(cases[7]["tokens"]) == 255
    assert cases[7]["positions"] == list(range(255))
    assert len(canonical_tiny_ops()) == 39


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'{"x":Infinity}'])
def test_strict_json_rejects_duplicates_and_all_nonfinite_spellings(raw: bytes) -> None:
    with pytest.raises(NumericalEvidenceError):
        strict_json(raw, label="negative")


@pytest.mark.parametrize("position", [True, -1, "0"])
def test_tensor_identity_does_not_guess_positions(position: Any) -> None:
    with pytest.raises(NumericalEvidenceError):
        tensor_id("B", "case", 0, "logits", position)


def test_semantic_ids_are_not_paths_or_invented_nonzero_origins() -> None:
    assert tensor_id("D-native", "fixed-greedy-00", 3, "logits", None) == "D-native/fixed-greedy-00/003/logits/full"
    assert tensor_id("B", "case", 0, "layer.0.rope_k", 31).endswith("/031")
    with pytest.raises(NumericalEvidenceError):
        tensor_id("../B", "case", 0, "logits", 0)


def test_original_op_authorities_are_read_from_exact_bytes_without_torch() -> None:
    authority = json.loads(ORACLE.read_bytes())
    for case in authority["op_cases"]:
        tensors = read_authority_tensors(ORACLE.parent / case["file"], case["sha256"])
        assert set(tensors) == set(canonical_tiny_ops())
        assert tensors["logits"].dtype == "fp32"
        expected = tensors["logits"]
        assert expected.raw == struct.pack("<" + "f" * len(expected.values), *expected.values)


def test_authority_reader_rejects_checksum_or_malformed_storage(tmp_path: Path) -> None:
    original = json.loads(ORACLE.read_bytes())["op_cases"][0]
    path = tmp_path / "bad.safetensors"
    raw = (ORACLE.parent / original["file"]).read_bytes()
    path.write_bytes(raw)
    with pytest.raises(NumericalEvidenceError):
        read_authority_tensors(path, "0" * 64)
    broken = struct.pack("<Q", len(raw) + 1) + raw[8:]
    path.write_bytes(broken)
    with pytest.raises(NumericalEvidenceError):
        read_authority_tensors(path, sha256_bytes(broken))


def test_contract_and_original_f32_authority_reader_are_isolated_stdlib_only() -> None:
    script = (
        "import sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(ROOT / 'src')!r});"
        "from illm.policy_cpu_numerical_contract import cpu_cases,read_authority_tensors;"
        f"assert len(cpu_cases(Path({str(ROOT)!r})))==70;"
        "assert not any(n.split('.')[0] in {'torch','transformers','tokenizers','safetensors'} for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

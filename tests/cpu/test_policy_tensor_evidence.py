from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any

import pytest

import illm.policy_tensor_evidence as tensor_evidence
from illm.policy_tensor_evidence import (
    EvidenceError,
    EvidenceTensor,
    TensorPack,
    TensorPackWriter,
    compare_evidence,
    slice_position,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw(dtype: str, values: list[float]) -> bytes:
    code = {"fp32": "f", "fp64": "d"}[dtype]
    return b"".join(struct.pack("<" + code, value) for value in values)


def _tensor(dtype: str, shape: tuple[int, ...], values: list[float]) -> EvidenceTensor:
    raw = _raw(dtype, values)
    code = {"fp32": "f", "fp64": "d"}[dtype]
    decoded = tuple(item[0] for item in struct.iter_unpack("<" + code, raw))
    return EvidenceTensor(dtype, shape, decoded, _sha256(raw), raw)


def _pack(root: Path) -> tuple[dict[str, object], TensorPack]:
    writer = TensorPackWriter(root)
    first = writer.write_tensor("B/tiny-00/layer.0/q_projection/full", "fp32", (2, 2), _raw("fp32", [1, 2, 3, 4]))
    second = writer.write_tensor("D-high/high-00/logits/000", "fp64", (2,), _raw("fp64", [5, 6]))
    assert first["offset"] == 0
    assert second["offset"] == 16
    manifest = writer.finalize()
    return manifest, TensorPack.open(root)


def _read_index(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()]


def _resign(root: Path, *, records: list[dict[str, Any]] | None = None) -> str:
    if records is not None:
        (root / "index.jsonl").write_bytes(b"".join(_canonical(record) + b"\n" for record in records))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["record_count"] = len(_read_index(root))
    manifest["data_bytes"] = (root / "data.bin").stat().st_size
    manifest["files"] = [
        {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": _sha256((root / name).read_bytes()),
        }
        for name in ("data.bin", "index.jsonl")
    ]
    projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = _sha256(_canonical(projection))
    (root / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    (root / "COMPLETE").write_bytes((str(manifest["manifest_digest"]) + "\n").encode("ascii"))
    return str(manifest["manifest_digest"])


def test_pack_round_trip_is_exact_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    manifest, pack = _pack(root)
    assert set(path.name for path in root.iterdir()) == {"data.bin", "index.jsonl", "manifest.json", "COMPLETE"}
    assert manifest["kind"] == "illm-policy-tensor-pack-v1"
    assert manifest["schema_version"] == 1
    assert manifest["byte_order"] == "little"
    assert manifest["record_count"] == 2
    assert tuple(pack.records) == (
        "B/tiny-00/layer.0/q_projection/full",
        "D-high/high-00/logits/000",
    )
    assert pack.records["B/tiny-00/layer.0/q_projection/full"]["shape"] == (2, 2)
    with pytest.raises(TypeError):
        pack.records["new"] = {}  # type: ignore[index]
    tensor = pack.tensor("B/tiny-00/layer.0/q_projection/full")
    assert tensor.dtype == "fp32"
    assert tensor.shape == (2, 2)
    assert tensor.values == (1.0, 2.0, 3.0, 4.0)
    assert tensor.raw == _raw("fp32", [1, 2, 3, 4])


def test_writer_refuses_existing_root_and_failure_stays_incomplete(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(EvidenceError, match="must not already exist"):
        TensorPackWriter(existing)

    root = tmp_path / "failed"
    writer = TensorPackWriter(root)
    writer.write_tensor("one", "fp32", (1,), _raw("fp32", [1]))
    with pytest.raises(EvidenceError, match="unique"):
        writer.write_tensor("one", "fp32", (1,), _raw("fp32", [2]))
    assert (root / "INCOMPLETE").is_file()
    assert not (root / "COMPLETE").exists()
    with pytest.raises(EvidenceError, match="not active"):
        writer.finalize()


@pytest.mark.parametrize(
    ("dtype", "shape", "raw", "message"),
    [
        ("bf16", (1,), b"\x00\x00", "dtype"),
        ("fp32", (), b"", "shape"),
        ("fp32", (True,), b"\x00" * 4, "shape"),
        ("fp32", (2,), b"\x00" * 4, "length"),
    ],
)
def test_writer_rejects_invalid_tensor_contract(
    tmp_path: Path, dtype: str, shape: tuple[int | bool, ...], raw: bytes, message: str
) -> None:
    writer = TensorPackWriter(tmp_path / f"bad-{message}")
    with pytest.raises(EvidenceError, match=message):
        writer.write_tensor("tensor", dtype, shape, raw)


def test_reader_rejects_resigned_offset_shape_and_unknown_record_fields(tmp_path: Path) -> None:
    for name, index, field, changed in (
        ("offset", 1, "offset", 17),
        ("shape", 0, "shape", [5]),
        ("unknown", 0, "prompt", "secret"),
    ):
        root = tmp_path / name
        _pack(root)
        records = _read_index(root)
        records[index][field] = changed
        _resign(root, records=records)
        with pytest.raises(EvidenceError):
            TensorPack.open(root)


def test_reader_rejects_resigned_manifest_unknown_field_and_unused_tail(tmp_path: Path) -> None:
    unknown = tmp_path / "manifest-unknown"
    _pack(unknown)
    manifest = json.loads((unknown / "manifest.json").read_text(encoding="utf-8"))
    manifest["qualification_tier"] = "qualified"
    projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = _sha256(_canonical(projection))
    (unknown / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
    (unknown / "COMPLETE").write_bytes((str(manifest["manifest_digest"]) + "\n").encode("ascii"))
    with pytest.raises(EvidenceError, match="closed schema"):
        TensorPack.open(unknown)

    tail = tmp_path / "tail"
    _pack(tail)
    with (tail / "data.bin").open("ab") as stream:
        stream.write(b"tail")
    _resign(tail)
    with pytest.raises(EvidenceError, match="trailing bytes"):
        TensorPack.open(tail)


def test_reader_rejects_duplicate_and_noncanonical_index_even_when_outer_resigned(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    _pack(duplicate)
    line = (duplicate / "index.jsonl").read_text(encoding="utf-8").splitlines()[0]
    duplicate_line = line[:-1] + ',"id":"duplicate"}'
    second = (duplicate / "index.jsonl").read_text(encoding="utf-8").splitlines()[1]
    (duplicate / "index.jsonl").write_bytes((duplicate_line + "\n" + second + "\n").encode("utf-8"))
    _resign(duplicate)
    with pytest.raises(EvidenceError, match="duplicate JSON key"):
        TensorPack.open(duplicate)

    noncanonical = tmp_path / "noncanonical"
    _pack(noncanonical)
    records = _read_index(noncanonical)
    (noncanonical / "index.jsonl").write_bytes(
        ("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n").encode("utf-8")
    )
    _resign(noncanonical)
    with pytest.raises(EvidenceError, match="not canonical"):
        TensorPack.open(noncanonical)


def test_byte_change_requires_record_identity_and_full_resign_is_a_new_pack(tmp_path: Path) -> None:
    outer_only = tmp_path / "outer-only"
    old_manifest, _ = _pack(outer_only)
    data = bytearray((outer_only / "data.bin").read_bytes())
    data[0] ^= 1
    (outer_only / "data.bin").write_bytes(data)
    changed_outer_digest = _resign(outer_only)
    assert changed_outer_digest != old_manifest["manifest_digest"]
    with pytest.raises(EvidenceError, match="payload hash"):
        TensorPack.open(outer_only)

    fully_resigned = tmp_path / "fully-resigned"
    original, _ = _pack(fully_resigned)
    changed = bytearray((fully_resigned / "data.bin").read_bytes())
    changed[0] ^= 1
    (fully_resigned / "data.bin").write_bytes(changed)
    records = _read_index(fully_resigned)
    records[0]["sha256"] = _sha256(bytes(changed[: int(records[0]["length"])]))
    new_digest = _resign(fully_resigned, records=records)
    assert new_digest != original["manifest_digest"]
    assert TensorPack.open(fully_resigned).tensor(str(records[0]["id"])).raw == bytes(changed[:16])


def test_nonfinite_bytes_are_retained_but_comparison_fails_canonically(tmp_path: Path) -> None:
    root = tmp_path / "nonfinite"
    writer = TensorPackWriter(root)
    writer.write_tensor("nonfinite", "fp64", (3,), _raw("fp64", [1.0, math.nan, math.inf]))
    writer.finalize()
    value = TensorPack.open(root).tensor("nonfinite")
    report = compare_evidence(_tensor("fp64", (3,), [1.0, 2.0, 3.0]), value, atol=0.0, rtol=0.0)
    assert report["status"] == "fail"
    assert report["reason"] == "non_finite"
    assert report["nonfinite_count"] == 2
    assert report["mismatch_count"] == 2
    assert report["first_flat_index"] == 1
    assert report["expected_value"] == 2.0
    assert report["actual_value"] is None
    json.dumps(report, allow_nan=False)


def test_compare_evidence_exact_directional_metrics_and_fixed_keys() -> None:
    expected = _tensor("fp64", (4,), [1.0, 2.0, 0.0, 4.0])
    actual = _tensor("fp64", (4,), [1.0, 2.3, 1e-4, 3.0])
    report = compare_evidence(expected, actual, atol=0.1, rtol=0.05)
    assert set(report) == {
        "status", "reason", "expected_shape", "actual_shape", "expected_dtype", "actual_dtype",
        "expected_sha256", "actual_sha256", "cross_precision", "atol", "rtol", "element_count",
        "first_flat_index", "expected_value", "actual_value", "max_absolute_error",
        "max_relative_error", "mismatch_count", "nonfinite_count", "bitwise_equal",
    }
    assert report["status"] == "fail"
    assert report["reason"] == "outside_tolerance"
    assert report["first_flat_index"] == 1
    assert report["expected_value"] == 2.0
    assert report["actual_value"] == pytest.approx(2.3)
    assert report["max_absolute_error"] == 1.0
    assert report["max_relative_error"] == pytest.approx(1e26)
    assert report["mismatch_count"] == 2
    assert report["nonfinite_count"] == 0
    assert report["bitwise_equal"] is False


def test_compare_is_expected_directional_and_cross_precision_does_not_precast() -> None:
    expected = _tensor("fp64", (1,), [100.0])
    actual = _tensor("fp64", (1,), [111.0])
    assert compare_evidence(expected, actual, atol=0, rtol=0.1)["status"] == "fail"
    assert compare_evidence(actual, expected, atol=0, rtol=0.1)["status"] == "pass"

    fp64 = _tensor("fp64", (1,), [1.00000006])
    fp32 = _tensor("fp32", (1,), [1.00000006])
    native = compare_evidence(fp64, fp32, atol=0, rtol=0)
    assert native["reason"] == "dtype_mismatch"
    assert native["mismatch_count"] is None
    cross = compare_evidence(fp64, fp32, atol=0, rtol=0, cross_precision=True)
    assert cross["reason"] == "outside_tolerance"
    assert cross["expected_value"] == 1.00000006
    assert cross["actual_value"] != cross["expected_value"]


def test_compare_shape_mismatch_and_statistics_overflow_do_not_emit_infinity() -> None:
    shape = compare_evidence(
        _tensor("fp64", (1,), [1.0]),
        _tensor("fp64", (1, 1), [1.0]),
        atol=0,
        rtol=0,
    )
    assert shape["reason"] == "shape_mismatch"
    assert shape["element_count"] is None
    assert shape["mismatch_count"] is None

    overflow = compare_evidence(
        _tensor("fp64", (1,), [1e308]),
        _tensor("fp64", (1,), [-1e308]),
        atol=0,
        rtol=0,
    )
    assert overflow["reason"] == "statistics_overflow"
    assert overflow["first_flat_index"] == 0
    assert overflow["max_absolute_error"] is None
    assert overflow["max_relative_error"] is None
    assert overflow["mismatch_count"] is None
    json.dumps(overflow, allow_nan=False)


@pytest.mark.parametrize(
    ("expected", "actual", "reason", "first", "left", "right"),
    [
        ([2.0, math.nan], [3.0, 4.0], "outside_tolerance", 0, 2.0, 3.0),
        ([2.0, 3.0], [2.0, math.nan], "non_finite", 1, 3.0, None),
        ([1.0, 1e308], [2.0, -1e308], "outside_tolerance", 0, 1.0, 2.0),
        ([1.0, 1e308], [1.0, -1e308], "statistics_overflow", 1, 1e308, -1e308),
    ],
)
def test_compare_mixed_failures_report_the_earliest_actual_failure(
    expected: list[float],
    actual: list[float],
    reason: str,
    first: int,
    left: float,
    right: float | None,
) -> None:
    report = compare_evidence(
        _tensor("fp64", (2,), expected),
        _tensor("fp64", (2,), actual),
        atol=0.0,
        rtol=0.0,
    )
    assert report["status"] == "fail"
    assert report["reason"] == reason
    assert report["first_flat_index"] == first
    assert report["expected_value"] == left
    assert report["actual_value"] == right
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(("atol", "rtol"), [(math.inf, 0.0), (math.nan, 0.0), (-1.0, 0.0), (0.0, True)])
def test_compare_rejects_invalid_tolerance(atol: float, rtol: float) -> None:
    value = _tensor("fp32", (1,), [1.0])
    with pytest.raises(EvidenceError, match="finite non-negative"):
        compare_evidence(value, value, atol=atol, rtol=rtol)


def test_slice_position_preserves_exact_selected_bytes() -> None:
    scores = _tensor("fp32", (1, 2, 3, 3), [float(value) for value in range(18)])
    score_row = slice_position(scores, "layer.0.attention_scores", 1)
    assert score_row.shape == (1, 2, 1, 2)
    assert score_row.values == (3.0, 4.0, 12.0, 13.0)
    assert score_row.raw == _raw("fp32", [3, 4, 12, 13])
    probability_row = slice_position(scores, "attention_probabilities", 2)
    assert probability_row.shape == (1, 2, 1, 3)
    assert probability_row.values == (6.0, 7.0, 8.0, 15.0, 16.0, 17.0)

    heads = _tensor("fp32", (1, 2, 3, 2), [float(value) for value in range(12)])
    head_row = slice_position(heads, "layer.0.rope_q", 2)
    assert head_row.shape == (1, 2, 1, 2)
    assert head_row.values == (4.0, 5.0, 10.0, 11.0)

    hidden = _tensor("fp64", (1, 3, 2), [float(value) for value in range(6)])
    hidden_row = slice_position(hidden, "logits", 1)
    assert hidden_row.shape == (1, 2)
    assert hidden_row.values == (2.0, 3.0)


def test_simulated_ancestor_and_entry_links_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = tensor_evidence._is_linklike
    monkeypatch.setattr(tensor_evidence, "_is_linklike", lambda path: path == tmp_path or original(path))
    with pytest.raises(EvidenceError, match="links or reparse"):
        TensorPackWriter(tmp_path / "blocked")
    monkeypatch.setattr(tensor_evidence, "_is_linklike", original)

    root = tmp_path / "pack"
    _pack(root)
    monkeypatch.setattr(tensor_evidence, "_is_linklike", lambda path: path.name == "data.bin" or original(path))
    with pytest.raises(EvidenceError, match="ordinary files"):
        TensorPack.open(root)


def test_writer_and_reader_reject_external_hardlinks_when_supported(tmp_path: Path) -> None:
    writer_root = tmp_path / "writer-pack"
    writer = TensorPackWriter(writer_root)
    writer.write_tensor("one", "fp32", (1,), _raw("fp32", [1.0]))
    try:
        (tmp_path / "writer-data-alias.bin").hardlink_to(writer_root / "data.bin")
    except (OSError, NotImplementedError):
        pytest.skip("hardlink creation is unavailable on this host")
    with pytest.raises(EvidenceError, match="single-link ordinary files"):
        writer.finalize()
    assert (writer_root / "INCOMPLETE").is_file()
    assert not (writer_root / "COMPLETE").exists()

    reader_root = tmp_path / "reader-pack"
    _pack(reader_root)
    (tmp_path / "reader-data-alias.bin").hardlink_to(reader_root / "data.bin")
    with pytest.raises(EvidenceError, match="single-link ordinary files"):
        TensorPack.open(reader_root)


def test_copied_reader_works_under_isolated_stdlib_without_tensor_packages(tmp_path: Path) -> None:
    source_root = Path(tensor_evidence.__file__).resolve().parents[1]
    pack_root = tmp_path / "source-pack"
    _pack(pack_root)
    isolated = tmp_path / "isolated"
    package = isolated / "reader" / "illm"
    package.mkdir(parents=True)
    shutil.copy2(source_root / "illm" / "policy_tensor_evidence.py", package / "policy_tensor_evidence.py")
    (package / "__init__.py").write_text("", encoding="utf-8")
    copied_pack = isolated / "pack"
    shutil.copytree(pack_root, copied_pack)
    cwd = isolated / "cwd"
    cwd.mkdir()
    code = (
        "import importlib.util,json,pathlib,sys;"
        f"sys.path.insert(0,{str(isolated / 'reader')!r});"
        "from illm.policy_tensor_evidence import TensorPack;"
        f"p=TensorPack.open(pathlib.Path({str(copied_pack)!r}));"
        "t=p.tensor('B/tiny-00/layer.0/q_projection/full');"
        "names=('torch','transformers','tokenizers','safetensors');"
        "assert all(importlib.util.find_spec(n) is None for n in names);"
        "assert all(n not in sys.modules for n in names);"
        "print(json.dumps({'dtype':t.dtype,'shape':list(t.shape),'records':len(p.records)},sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"dtype": "fp32", "shape": [2, 2], "records": 2}

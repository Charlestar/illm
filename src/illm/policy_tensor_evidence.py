"""Portable, stdlib-only tensor evidence containers for V0.2 development.

The format proves the internal consistency of one tensor pack.  It deliberately
does not authenticate who produced the bytes and does not grant qualification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Any, BinaryIO, Final


_KIND: Final = "illm-policy-tensor-pack-v1"
_SCHEMA_VERSION: Final = 1
_BYTE_ORDER: Final = "little"
_FINAL_FILES: Final = frozenset({"data.bin", "index.jsonl", "manifest.json", "COMPLETE"})
_INDEX_FIELDS: Final = frozenset({"id", "dtype", "shape", "offset", "length", "sha256"})
_MANIFEST_FIELDS: Final = frozenset(
    {"schema_version", "kind", "byte_order", "record_count", "data_bytes", "files", "manifest_digest"}
)
_FILE_FIELDS: Final = frozenset({"path", "size_bytes", "sha256"})
_DTYPE_FORMAT: Final = {"fp32": "f", "fp64": "d"}
_DTYPE_BYTES: Final = {"fp32": 4, "fp64": 8}
_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_MAX_BYTE_OFFSET: Final = (1 << 63) - 1


class EvidenceError(ValueError):
    """A closed-format or evidence-consistency violation."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"cannot hash evidence file {path.name!r}") from exc
    return digest.hexdigest()


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite_json(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number {value!r}")


def _decode_json_object(raw: bytes, *, label: str, newline: bool) -> dict[str, Any]:
    payload = raw[:-1] if newline and raw.endswith(b"\n") else raw
    if newline and (not raw.endswith(b"\n") or payload.endswith(b"\r")):
        raise EvidenceError(f"{label} is not newline-terminated canonical JSON")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise EvidenceError(f"{label} must be a JSON object")
    if raw != _canonical_bytes(value) + (b"\n" if newline else b""):
        raise EvidenceError(f"{label} is not canonical JSON")
    return value


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot inspect path {path}") from exc
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _require_single_link_regular_file(path: Path) -> None:
    if _is_linklike(path):
        raise EvidenceError("tensor pack entries must be single-link ordinary files")
    try:
        info = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot inspect evidence file {path.name!r}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or int(info.st_nlink) != 1
    ):
        raise EvidenceError("tensor pack entries must be single-link ordinary files")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _check_existing_chain(path: Path, *, include_self: bool) -> None:
    candidates = (path, *path.parents) if include_self else tuple(path.parents)
    for candidate in candidates:
        if os.path.lexists(candidate) and _is_linklike(candidate):
            raise EvidenceError("tensor pack path cannot contain links or reparse points")


def _new_pack_root(path: Path) -> Path:
    root = _lexical_absolute(path)
    _check_existing_chain(root, include_self=True)
    if os.path.lexists(root):
        raise EvidenceError("tensor pack root must not already exist")
    parent = root.parent
    if not parent.is_dir() or _is_linklike(parent):
        raise EvidenceError("tensor pack parent must be an existing ordinary directory")
    return root


def _existing_pack_root(path: Path) -> Path:
    root = _lexical_absolute(path)
    _check_existing_chain(root, include_self=True)
    if not root.is_dir() or _is_linklike(root):
        raise EvidenceError("tensor pack root must be an ordinary directory")
    return root


def _exclusive_write(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise EvidenceError(f"cannot exclusively write {path.name!r}") from exc


def _validate_id(value: object) -> str:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise EvidenceError("tensor id must be a safe ASCII semantic key")
    return value


def _validate_dtype(value: object) -> str:
    if type(value) is not str or value not in _DTYPE_BYTES:
        raise EvidenceError("tensor dtype must be fp32 or fp64")
    return value


def _validate_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EvidenceError("tensor shape must be a sequence")
    shape = tuple(value)
    if not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise EvidenceError("tensor shape dimensions must be positive integers")
    count = 1
    for dimension in shape:
        if count > _MAX_BYTE_OFFSET // dimension:
            raise EvidenceError("tensor shape exceeds the supported byte range")
        count *= dimension
    return shape


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _expected_length(dtype: str, shape: tuple[int, ...]) -> int:
    count = _element_count(shape)
    item_size = _DTYPE_BYTES[dtype]
    if count > _MAX_BYTE_OFFSET // item_size:
        raise EvidenceError("tensor byte length exceeds the supported range")
    return count * item_size


def _strict_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > _MAX_BYTE_OFFSET:
        raise EvidenceError(f"{label} must be a bounded integer")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _decode_values(dtype: str, raw: bytes) -> tuple[float, ...]:
    item_size = _DTYPE_BYTES[dtype]
    if len(raw) % item_size:
        raise EvidenceError("tensor raw bytes are not aligned to dtype")
    return tuple(item[0] for item in struct.iter_unpack("<" + _DTYPE_FORMAT[dtype], raw))


def _same_float(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if left == 0.0 and right == 0.0:
        return math.copysign(1.0, left) == math.copysign(1.0, right)
    return left == right


@dataclass(frozen=True, slots=True)
class EvidenceTensor:
    """A tensor's exact bytes plus independently decoded Python values."""

    dtype: str
    shape: tuple[int, ...]
    values: tuple[float, ...]
    raw_sha256: str
    raw: bytes

    def __post_init__(self) -> None:
        dtype = _validate_dtype(self.dtype)
        shape = _validate_shape(self.shape)
        if type(self.shape) is not tuple or shape != self.shape:
            raise EvidenceError("EvidenceTensor shape must be a tuple of positive integers")
        if type(self.raw) is not bytes or len(self.raw) != _expected_length(dtype, shape):
            raise EvidenceError("EvidenceTensor raw bytes disagree with shape and dtype")
        if _strict_sha256(self.raw_sha256, label="raw_sha256") != _sha256_bytes(self.raw):
            raise EvidenceError("EvidenceTensor raw SHA-256 mismatch")
        if type(self.values) is not tuple or any(type(value) is not float for value in self.values):
            raise EvidenceError("EvidenceTensor values must be a tuple of floats")
        decoded = _decode_values(dtype, self.raw)
        if len(decoded) != len(self.values) or any(
            not _same_float(expected, actual) for expected, actual in zip(decoded, self.values, strict=True)
        ):
            raise EvidenceError("EvidenceTensor values do not decode from raw bytes")


def _tensor_from_raw(dtype: str, shape: tuple[int, ...], raw: bytes) -> EvidenceTensor:
    return EvidenceTensor(
        dtype=dtype,
        shape=shape,
        values=_decode_values(dtype, raw),
        raw_sha256=_sha256_bytes(raw),
        raw=raw,
    )


@dataclass(frozen=True, slots=True)
class _Record:
    tensor_id: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int
    sha256: str

    def public(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "id": self.tensor_id,
                "dtype": self.dtype,
                "shape": self.shape,
                "offset": self.offset,
                "length": self.length,
                "sha256": self.sha256,
            }
        )


class TensorPackWriter:
    """Exclusive streaming writer for a development tensor evidence pack."""

    def __init__(self, root: Path) -> None:
        self._root = _new_pack_root(root)
        self._data: BinaryIO | None = None
        self._index: BinaryIO | None = None
        self._records: list[_Record] = []
        self._ids: set[str] = set()
        self._offset = 0
        self._state = "initializing"
        try:
            self._root.mkdir()
            _exclusive_write(self._root / "INCOMPLETE", b"tensor_pack_in_progress\n")
            self._data = (self._root / "data.bin").open("xb")
            self._index = (self._root / "index.jsonl").open("xb")
            self._state = "active"
        except (OSError, EvidenceError) as exc:
            self._state = "failed"
            self._close_streams(sync=False)
            self._retain_incomplete()
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError("cannot initialize tensor pack") from exc

    def _retain_incomplete(self) -> None:
        if not self._root.is_dir() or _is_linklike(self._root):
            return
        marker = self._root / "INCOMPLETE"
        if not os.path.lexists(marker):
            try:
                _exclusive_write(marker, b"tensor_pack_failed\n")
            except EvidenceError:
                pass

    def _close_streams(self, *, sync: bool) -> None:
        for stream in (self._data, self._index):
            if stream is None or stream.closed:
                continue
            try:
                if sync:
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                self._state = "failed"
            finally:
                try:
                    stream.close()
                except OSError:
                    self._state = "failed"
        self._data = None
        self._index = None

    def _fail(self) -> None:
        self._state = "failed"
        self._close_streams(sync=False)
        self._retain_incomplete()

    def _require_active(self) -> tuple[BinaryIO, BinaryIO]:
        if self._state != "active" or self._data is None or self._index is None:
            raise EvidenceError("tensor pack writer is not active")
        return self._data, self._index

    def write_tensor(
        self,
        tensor_id: str,
        dtype: str,
        shape: Sequence[int],
        raw: bytes,
    ) -> dict[str, object]:
        data_stream, index_stream = self._require_active()
        try:
            checked_id = _validate_id(tensor_id)
            checked_dtype = _validate_dtype(dtype)
            checked_shape = _validate_shape(shape)
            if checked_id in self._ids:
                raise EvidenceError("tensor ids must be unique")
            if type(raw) is not bytes:
                raise EvidenceError("tensor raw payload must be bytes")
            expected_length = _expected_length(checked_dtype, checked_shape)
            if len(raw) != expected_length:
                raise EvidenceError("tensor raw length disagrees with shape and dtype")
            if self._offset > _MAX_BYTE_OFFSET - len(raw):
                raise EvidenceError("tensor pack byte offset exceeds the supported range")
            record = _Record(
                tensor_id=checked_id,
                dtype=checked_dtype,
                shape=checked_shape,
                offset=self._offset,
                length=len(raw),
                sha256=_sha256_bytes(raw),
            )
            encoded = _canonical_bytes(
                {
                    "id": record.tensor_id,
                    "dtype": record.dtype,
                    "shape": list(record.shape),
                    "offset": record.offset,
                    "length": record.length,
                    "sha256": record.sha256,
                }
            ) + b"\n"
            data_stream.write(raw)
            index_stream.write(encoded)
            self._records.append(record)
            self._ids.add(checked_id)
            self._offset += len(raw)
            return {
                "id": record.tensor_id,
                "dtype": record.dtype,
                "shape": list(record.shape),
                "offset": record.offset,
                "length": record.length,
                "sha256": record.sha256,
            }
        except (OSError, EvidenceError) as exc:
            self._fail()
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError("cannot append tensor evidence") from exc

    def finalize(self) -> dict[str, object]:
        self._require_active()
        if not self._records:
            self._fail()
            raise EvidenceError("tensor pack must contain at least one tensor")
        self._close_streams(sync=True)
        if self._state == "failed":
            self._retain_incomplete()
            raise EvidenceError("cannot synchronize tensor pack")
        try:
            for name in ("data.bin", "index.jsonl"):
                _require_single_link_regular_file(self._root / name)
            files = [
                {
                    "path": name,
                    "size_bytes": (self._root / name).stat().st_size,
                    "sha256": _sha256_file(self._root / name),
                }
                for name in ("data.bin", "index.jsonl")
            ]
            projection: dict[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "kind": _KIND,
                "byte_order": _BYTE_ORDER,
                "record_count": len(self._records),
                "data_bytes": self._offset,
                "files": files,
            }
            manifest = {**projection, "manifest_digest": _sha256_bytes(_canonical_bytes(projection))}
            _exclusive_write(self._root / "manifest.json", _canonical_bytes(manifest) + b"\n")
            _exclusive_write(
                self._root / "COMPLETE",
                (str(manifest["manifest_digest"]) + "\n").encode("ascii"),
            )
            try:
                (self._root / "INCOMPLETE").unlink()
            except OSError as exc:
                try:
                    (self._root / "COMPLETE").unlink()
                except OSError:
                    pass
                raise EvidenceError("cannot clear tensor pack incomplete marker") from exc
            self._state = "finalized"
            return copy.deepcopy(manifest)
        except (OSError, EvidenceError) as exc:
            self._state = "failed"
            self._retain_incomplete()
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError("cannot finalize tensor pack") from exc


class TensorPack:
    """A fully validated, read-only view of a tensor evidence pack."""

    def __init__(self, root: Path, manifest: Mapping[str, object], records: tuple[_Record, ...]) -> None:
        self._root = root
        self._manifest = MappingProxyType(copy.deepcopy(dict(manifest)))
        self._record_data = records
        self._by_id = {record.tensor_id: record for record in records}
        self._records: Mapping[str, Mapping[str, object]] = MappingProxyType(
            {record.tensor_id: record.public() for record in records}
        )

    @property
    def records(self) -> Mapping[str, Mapping[str, object]]:
        return self._records

    @classmethod
    def open(cls, root: Path) -> TensorPack:
        checked = _existing_pack_root(root)
        try:
            entries = list(checked.iterdir())
        except OSError as exc:
            raise EvidenceError("cannot enumerate tensor pack") from exc
        names = {entry.name for entry in entries}
        if names != _FINAL_FILES or len(entries) != len(_FINAL_FILES):
            raise EvidenceError("finalized tensor pack has an unexpected inventory")
        for entry in entries:
            _require_single_link_regular_file(entry)

        try:
            manifest_raw = (checked / "manifest.json").read_bytes()
        except OSError as exc:
            raise EvidenceError("cannot read tensor pack manifest") from exc
        manifest = _decode_json_object(manifest_raw, label="manifest", newline=True)
        if set(manifest) != _MANIFEST_FIELDS:
            raise EvidenceError("manifest fields differ from the closed schema")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != _SCHEMA_VERSION
            or manifest.get("kind") != _KIND
            or manifest.get("byte_order") != _BYTE_ORDER
        ):
            raise EvidenceError("manifest identity is unsupported")
        record_count = _strict_int(manifest.get("record_count"), label="record_count", minimum=1)
        data_bytes = _strict_int(manifest.get("data_bytes"), label="data_bytes", minimum=1)
        digest = _strict_sha256(manifest.get("manifest_digest"), label="manifest_digest")
        projection = {key: value for key, value in manifest.items() if key != "manifest_digest"}
        if digest != _sha256_bytes(_canonical_bytes(projection)):
            raise EvidenceError("manifest digest mismatch")
        try:
            complete = (checked / "COMPLETE").read_bytes()
        except OSError as exc:
            raise EvidenceError("cannot read tensor pack completion marker") from exc
        if complete != (digest + "\n").encode("ascii"):
            raise EvidenceError("completion marker does not bind the manifest")

        files_value = manifest.get("files")
        if type(files_value) is not list or len(files_value) != 2:
            raise EvidenceError("manifest files must be the exact data/index inventory")
        for expected_name, value in zip(("data.bin", "index.jsonl"), files_value, strict=True):
            if type(value) is not dict or set(value) != _FILE_FIELDS or value.get("path") != expected_name:
                raise EvidenceError("manifest file record differs from the closed schema")
            size = _strict_int(value.get("size_bytes"), label="file size")
            sha256 = _strict_sha256(value.get("sha256"), label="file sha256")
            path = checked / expected_name
            try:
                actual_size = path.stat().st_size
            except OSError as exc:
                raise EvidenceError("cannot inspect tensor pack file") from exc
            if actual_size != size or _sha256_file(path) != sha256:
                raise EvidenceError("manifest file hash or size mismatch")
        if files_value[0]["size_bytes"] != data_bytes:
            raise EvidenceError("manifest data byte count disagrees with data.bin")

        try:
            index_raw = (checked / "index.jsonl").read_bytes()
        except OSError as exc:
            raise EvidenceError("cannot read tensor pack index") from exc
        if not index_raw or not index_raw.endswith(b"\n"):
            raise EvidenceError("tensor pack index must be non-empty canonical JSONL")
        lines = index_raw.splitlines(keepends=True)
        if len(lines) != record_count:
            raise EvidenceError("index record count disagrees with manifest")
        records: list[_Record] = []
        ids: set[str] = set()
        next_offset = 0
        try:
            data_stream = (checked / "data.bin").open("rb")
        except OSError as exc:
            raise EvidenceError("cannot read tensor pack data") from exc
        try:
            for line_number, line in enumerate(lines, start=1):
                value = _decode_json_object(line, label=f"index line {line_number}", newline=True)
                if set(value) != _INDEX_FIELDS:
                    raise EvidenceError("index record fields differ from the closed schema")
                tensor_id = _validate_id(value.get("id"))
                dtype = _validate_dtype(value.get("dtype"))
                shape = _validate_shape(value.get("shape"))
                offset = _strict_int(value.get("offset"), label="record offset")
                length = _strict_int(value.get("length"), label="record length", minimum=1)
                sha256 = _strict_sha256(value.get("sha256"), label="record sha256")
                if tensor_id in ids:
                    raise EvidenceError("tensor ids must be unique")
                if offset != next_offset:
                    raise EvidenceError("tensor offsets must be contiguous without gaps or overlap")
                if length != _expected_length(dtype, shape):
                    raise EvidenceError("record length disagrees with shape and dtype")
                if offset > data_bytes - length:
                    raise EvidenceError("tensor record exceeds data.bin")
                data_stream.seek(offset)
                raw = data_stream.read(length)
                if len(raw) != length or _sha256_bytes(raw) != sha256:
                    raise EvidenceError("tensor payload hash mismatch")
                records.append(_Record(tensor_id, dtype, shape, offset, length, sha256))
                ids.add(tensor_id)
                next_offset += length
        finally:
            data_stream.close()
        if next_offset != data_bytes:
            raise EvidenceError("data.bin contains unused trailing bytes")
        return cls(checked, manifest, tuple(records))

    def tensor(self, tensor_id: str) -> EvidenceTensor:
        checked_id = _validate_id(tensor_id)
        try:
            record = self._by_id[checked_id]
        except KeyError as exc:
            raise EvidenceError("tensor id is absent from pack") from exc
        try:
            with (self._root / "data.bin").open("rb") as stream:
                stream.seek(record.offset)
                raw = stream.read(record.length)
        except OSError as exc:
            raise EvidenceError("cannot read tensor payload") from exc
        if len(raw) != record.length or _sha256_bytes(raw) != record.sha256:
            raise EvidenceError("tensor payload changed after pack validation")
        return _tensor_from_raw(record.dtype, record.shape, raw)


def slice_position(tensor: EvidenceTensor, op: str, position: int) -> EvidenceTensor:
    """Extract one query position while preserving the selected source bytes."""

    if type(op) is not str or not op or any(ord(character) < 33 or ord(character) > 126 for character in op):
        raise EvidenceError("op must be a non-empty printable ASCII identifier")
    if type(position) is not int or position < 0:
        raise EvidenceError("position must be a non-negative integer")
    shape = tensor.shape
    item_size = _DTYPE_BYTES[tensor.dtype]
    chunks: list[bytes] = []
    result_shape: tuple[int, ...]
    if len(shape) == 4:
        batch, heads, queries, width = shape
        if batch != 1 or position >= queries:
            raise EvidenceError("position is outside the rank-4 tensor query axis")
        attention = op in {"attention_scores", "attention_probabilities"} or op.endswith(
            (".attention_scores", ".attention_probabilities")
        )
        selected_width = position + 1 if attention else width
        if selected_width > width:
            raise EvidenceError("attention tensor does not contain the legal prefix")
        for head in range(heads):
            start_element = (head * queries + position) * width
            start = start_element * item_size
            chunks.append(tensor.raw[start : start + selected_width * item_size])
        result_shape = (1, heads, 1, selected_width)
    elif len(shape) == 3:
        batch, queries, width = shape
        if batch != 1 or position >= queries:
            raise EvidenceError("position is outside the rank-3 tensor query axis")
        start = position * width * item_size
        chunks.append(tensor.raw[start : start + width * item_size])
        result_shape = (1, width)
    else:
        raise EvidenceError("position slicing supports only rank-3 and rank-4 tensors")
    raw = b"".join(chunks)
    return _tensor_from_raw(tensor.dtype, result_shape, raw)


def _tolerance(value: object, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise EvidenceError(f"{label} must be a finite non-negative number")
    if not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EvidenceError(f"{label} must be a finite non-negative number")
    return result


def compare_evidence(
    expected: EvidenceTensor,
    actual: EvidenceTensor,
    *,
    atol: float,
    rtol: float,
    cross_precision: bool = False,
) -> dict[str, object]:
    """Recompute directional tensor metrics without importing a tensor runtime."""

    checked_atol = _tolerance(atol, label="atol")
    checked_rtol = _tolerance(rtol, label="rtol")
    if type(cross_precision) is not bool:
        raise EvidenceError("cross_precision must be a bool")
    bitwise_equal = (
        expected.dtype == actual.dtype
        and expected.shape == actual.shape
        and expected.raw == actual.raw
    )
    report: dict[str, object] = {
        "status": "fail",
        "reason": None,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": expected.dtype,
        "actual_dtype": actual.dtype,
        "expected_sha256": expected.raw_sha256,
        "actual_sha256": actual.raw_sha256,
        "cross_precision": cross_precision,
        "atol": checked_atol,
        "rtol": checked_rtol,
        "element_count": None,
        "first_flat_index": None,
        "expected_value": None,
        "actual_value": None,
        "max_absolute_error": None,
        "max_relative_error": None,
        "mismatch_count": None,
        "nonfinite_count": None,
        "bitwise_equal": bitwise_equal,
    }
    if expected.shape != actual.shape:
        report["reason"] = "shape_mismatch"
        return report
    count = _element_count(expected.shape)
    report["element_count"] = count
    dtype_ok = expected.dtype == actual.dtype if not cross_precision else (
        expected.dtype == "fp64" and actual.dtype == "fp32"
    )
    if not dtype_ok:
        report["reason"] = "dtype_mismatch"
        return report

    nonfinite_indexes: list[int] = []
    mismatch_indexes: list[int] = []
    max_absolute = 0.0
    max_relative = 0.0
    overflow_index: int | None = None
    for index, (expected_value, actual_value) in enumerate(
        zip(expected.values, actual.values, strict=True)
    ):
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            nonfinite_indexes.append(index)
            continue
        absolute = abs(actual_value - expected_value)
        denominator = max(abs(expected_value), 1e-30)
        relative = absolute / denominator
        threshold = checked_atol + checked_rtol * abs(expected_value)
        if not math.isfinite(absolute) or not math.isfinite(relative) or not math.isfinite(threshold):
            if overflow_index is None:
                overflow_index = index
            continue
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
        if absolute > threshold:
            mismatch_indexes.append(index)

    report["nonfinite_count"] = len(nonfinite_indexes)
    if overflow_index is None:
        report["max_absolute_error"] = max_absolute
        report["max_relative_error"] = max_relative
        report["mismatch_count"] = len(nonfinite_indexes) + len(mismatch_indexes)
    failures = sorted(
        (*nonfinite_indexes, *mismatch_indexes, *((overflow_index,) if overflow_index is not None else ()))
    )
    if failures:
        first = failures[0]
    else:
        report["status"] = "pass"
        report["reason"] = "within_tolerance"
        return report
    if first in nonfinite_indexes:
        report["reason"] = "non_finite"
        report["first_flat_index"] = first
        left = expected.values[first]
        right = actual.values[first]
        report["expected_value"] = left if math.isfinite(left) else None
        report["actual_value"] = right if math.isfinite(right) else None
        return report
    if overflow_index is not None and first == overflow_index:
        report["reason"] = "statistics_overflow"
        report["first_flat_index"] = first
        report["expected_value"] = expected.values[first]
        report["actual_value"] = actual.values[first]
        return report
    report["reason"] = "outside_tolerance"
    report["first_flat_index"] = first
    report["expected_value"] = expected.values[first]
    report["actual_value"] = actual.values[first]
    return report


__all__ = [
    "EvidenceError",
    "EvidenceTensor",
    "TensorPack",
    "TensorPackWriter",
    "compare_evidence",
    "slice_position",
]

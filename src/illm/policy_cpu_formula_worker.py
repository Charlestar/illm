"""Independent CPU binary64 formula evidence worker for protocol section 4.3.

The module stays stdlib-only until :func:`run_formula_worker` has established a
new, fail-closed output root.  The worker never imports or executes A/B/C model
implementations, runtime KV/executor code, or Transformers helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any, Mapping, NoReturn, Sequence, cast


REPORT_KIND = "illm-policy-cpu-raw-worker-v1"
_FORBIDDEN_MODULE_PREFIXES = (
    "illm.policy_cpu_mutation_worker",
    "illm.model.reference",
    "illm.model.policy",
    "illm.model.incremental",
    "illm.model.validation",
    "illm.kv_cache",
    "illm.executor",
    "illm.engine",
    "transformers",
)


class FormulaWorkerError(RuntimeError):
    """The formula worker could not produce truthful complete evidence."""


def _fail(message: str) -> NoReturn:
    raise FormulaWorkerError(message)


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
        raise FormulaWorkerError("formula worker report is not canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FormulaWorkerError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise FormulaWorkerError(f"cannot inspect path {path}") from exc
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _plain_existing_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail(f"{label} cannot contain a link or reparse point")
    if not absolute.is_dir() or _is_linklike(absolute):
        _fail(f"{label} must be an existing ordinary directory")
    return absolute


def _begin_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in absolute.parents:
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail("formula worker output ancestors cannot contain links or reparse points")
    if os.path.lexists(absolute):
        _fail("formula worker output must not already exist")
    parent = absolute.parent
    if not parent.is_dir() or _is_linklike(parent):
        _fail("formula worker output parent must be an existing ordinary directory")
    try:
        absolute.mkdir()
        with (absolute / "INCOMPLETE").open("xb") as stream:
            stream.write(b"formula_worker_in_progress\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise FormulaWorkerError("cannot initialize formula worker output") from exc
    return absolute


def _exclusive_write(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise FormulaWorkerError(f"cannot exclusively write {path.name}") from exc


def _finish_output(root: Path, report: Mapping[str, object]) -> None:
    encoded = _canonical_bytes(report) + b"\n"
    _exclusive_write(root / "worker.json", encoded)
    _exclusive_write(root / "COMPLETE", (hashlib.sha256(encoded).hexdigest() + "\n").encode("ascii"))
    try:
        (root / "INCOMPLETE").unlink()
    except OSError as exc:
        try:
            (root / "COMPLETE").unlink()
        except OSError:
            pass
        raise FormulaWorkerError("cannot clear formula worker incomplete marker") from exc


def _forbidden_loaded() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_MODULE_PREFIXES)
    )


def _assert_formula_isolation(stage: str) -> None:
    loaded = _forbidden_loaded()
    if loaded:
        _fail(f"formula worker imported forbidden modules at {stage}: {loaded}")


def _tensor_raw(tensor: Any, *, expected_dtype: Any) -> bytes:
    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != expected_dtype
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all().item())
    ):
        _fail("formula tensor differs from its finite contiguous CPU dtype contract")
    return tensor.view(torch.uint8).numpy().tobytes(order="C")


def _validate_high_cases(cases: Sequence[Mapping[str, object]]) -> None:
    expected_ids = [f"high-{index:02d}" for index in range(8)]
    if [case.get("case_id") for case in cases] != expected_ids:
        _fail("high-case order or identity differs from the frozen protocol")
    for index, case in enumerate(cases):
        if set(case) != {"case_id", "mode", "tokens", "positions"}:
            _fail("high case fields differ from the closed worker contract")
        mode = case["mode"]
        if mode != ("rope_only" if index == 5 else "forward"):
            _fail("high case mode differs from the frozen protocol")
        tokens = case["tokens"]
        positions = case["positions"]
        if type(tokens) is not list or type(positions) is not list:
            _fail("high case tokens and positions must be lists")
        if index == 5:
            if tokens != [] or positions != [31, 32]:
                _fail("high-05 must remain the RoPE-only absolute-position case")
        elif not tokens or positions != list(range(len(tokens))):
            _fail("full high case must start at canonical position zero")


def run_formula_worker(project_root: Path, output: Path) -> dict[str, object]:
    """Write one exclusive, CPU-only D-high raw evidence bundle."""

    _assert_formula_isolation("startup")
    root = _plain_existing_directory(Path(project_root), label="project root")
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "illm").is_dir():
        _fail("formula worker project root is not an iLLM source workspace")
    destination = _begin_output(Path(output))

    # Imports below this boundary are intentional evidence dependencies.  The
    # isolation assertion above proves no forbidden implementation was inherited.
    import torch

    from illm.model.high_precision import HighPrecisionOracle, OpRecord
    from illm.policy_binding import build_execution_binding
    from illm.policy_cpu_numerical_contract import canonical_tiny_ops, high_cases, tensor_id
    from illm.policy_tensor_evidence import TensorPackWriter

    if sys.byteorder != "little":
        _fail("tensor evidence v1 requires a little-endian CPU process")
    cuda_initialized_start = bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
    if cuda_initialized_start:
        _fail("formula worker inherited an initialized CUDA runtime")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        binding = build_execution_binding(root)
        oracle = HighPrecisionOracle.from_fixture(root)
        contract_cases = cast(list[Mapping[str, object]], high_cases())
        _validate_high_cases(contract_cases)
        canonical_ops = tuple(canonical_tiny_ops())
        if len(canonical_ops) != 39 or len(set(canonical_ops)) != 39:
            _fail("canonical tiny operation registry must contain 39 unique operations")

        pack = TensorPackWriter(destination / "pack")
        case_reports: list[dict[str, object]] = []
        for case in contract_cases:
            case_id = cast(str, case["case_id"])
            mode = cast(str, case["mode"])
            tokens = cast(list[int], case["tokens"])
            positions = cast(list[int], case["positions"])
            if mode == "forward":
                paths: dict[str, dict[str, str]] = {name: {} for name in canonical_ops}
                seen: set[tuple[str, int]] = set()

                def hook(record: OpRecord, tensor: torch.Tensor) -> None:
                    key = (record.name, record.position)
                    if (
                        key in seen
                        or record.name not in paths
                        or record.position not in positions
                        or record.dtype != "torch.float64"
                        or record.device != "cpu"
                        or tuple(tensor.shape) != record.shape
                    ):
                        _fail("D-high hook emitted an unexpected operation record")
                    raw = _tensor_raw(tensor.detach().contiguous(), expected_dtype=torch.float64)
                    semantic_id = tensor_id("D-high", case_id, 0, record.name, record.position)
                    pack.write_tensor(semantic_id, "fp64", record.shape, raw)
                    paths[record.name][f"{record.position:03d}"] = semantic_id
                    seen.add(key)

                input_ids = torch.tensor([tokens], dtype=torch.int64, device="cpu")
                position_ids = torch.tensor([positions], dtype=torch.int64, device="cpu")
                oracle.forward(input_ids, position_ids=position_ids, op_hook=hook)
                expected_keys = {(name, position) for name in canonical_ops for position in positions}
                if seen != expected_keys:
                    _fail(f"{case_id} D-high operation coverage is incomplete")
                path_report: object = paths
            else:
                query = torch.tensor(
                    [[[[((2 * head + dimension - 9) / 16) for dimension in range(8)] for _ in range(2)]
                      for head in range(4)]],
                    dtype=torch.float32,
                    device="cpu",
                )
                key = torch.tensor(
                    [[[[((3 * head - dimension + 5) / 32) for dimension in range(8)] for _ in range(2)]
                      for head in range(2)]],
                    dtype=torch.float32,
                    device="cpu",
                )
                position_ids = torch.tensor([positions], dtype=torch.int64, device="cpu")
                expected_query, expected_key = oracle.rope(query, key, position_ids)
                rope_paths: dict[str, str] = {}
                for label, tensor, dtype_name, torch_dtype in (
                    ("input_q", query, "fp32", torch.float32),
                    ("input_k", key, "fp32", torch.float32),
                    ("rope_q", expected_query.contiguous(), "fp64", torch.float64),
                    ("rope_k", expected_key.contiguous(), "fp64", torch.float64),
                ):
                    semantic_id = tensor_id("D-high", case_id, 0, label, None)
                    pack.write_tensor(
                        semantic_id,
                        dtype_name,
                        tuple(tensor.shape),
                        _tensor_raw(tensor, expected_dtype=torch_dtype),
                    )
                    rope_paths[label] = semantic_id
                path_report = rope_paths
            case_reports.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "input_token_ids": list(tokens),
                    "positions": list(positions),
                    "paths": path_report,
                }
            )

        _assert_formula_isolation("completion")
        cuda_initialized_end = bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]
        if cuda_initialized_end:
            _fail("formula worker initialized CUDA")
        pack_manifest = pack.finalize()
        source_path = Path(cast(str, sys.modules[HighPrecisionOracle.__module__].__file__)).resolve()
        source_relative = source_path.relative_to(root).as_posix()
        process = {
            "pid": os.getpid(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "byte_order": sys.byteorder,
            "cpu_threads": 1,
            "packages": {
                "torch": importlib.metadata.version("torch"),
                "safetensors": importlib.metadata.version("safetensors"),
            },
            "cuda_internal_initialized_start": cuda_initialized_start,
            "cuda_internal_initialized_end": cuda_initialized_end,
            "forbidden_modules_loaded_start": [],
            "forbidden_modules_loaded_end": _forbidden_loaded(),
        }
        report: dict[str, object] = {
            "schema_version": 1,
            "report_kind": REPORT_KIND,
            "status": "pass",
            "evidence_stage": "DEVELOPMENT_ONLY",
            "qualification_status": "unqualified",
            "role": "formula",
            "coverage": {
                "protocol_section": "4.3",
                "full_forward_cases": 7,
                "rope_only_cases": 1,
                "canonical_op_count": len(canonical_ops),
                "tensor_count": pack_manifest["record_count"],
                "pack_path": "pack",
                "gpu_execution": "not_run_not_authorized",
                "claims": [
                    "independent_cpu_binary64_tiny_formula",
                    "raw_tensor_bytes",
                ],
                "limitations": [
                    "not a real-FP64 full-model qualification",
                    "not native-dtype equality evidence",
                    "not GPU correctness or performance evidence",
                    "high-05 is RoPE-only and does not establish a nonzero public request origin",
                ],
            },
            "binding": binding.to_dict(),
            "binding_digest": binding.digest,
            "process": process,
            "model": {
                "kind": "independent-binary64-tiny-formula",
                "spec_id": oracle.spec_id,
                "oracle_identity": dict(oracle.identity),
                "source": {"path": source_relative, "sha256": _sha256_file(source_path)},
                "candidate_loaded": False,
            },
            "policy": None,
            "pack_manifest_digest": pack_manifest["manifest_digest"],
            "cases": [],
            "high_cases": case_reports,
        }
        _finish_output(destination, report)
        return cast(dict[str, object], json.loads(_canonical_bytes(report)))
    finally:
        torch.set_num_threads(previous_threads)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce independent CPU D-high raw evidence")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_formula_worker(arguments.project_root, arguments.output)
    print(_canonical_bytes({
        "role": report["role"],
        "status": report["status"],
        "pack_manifest_digest": report["pack_manifest_digest"],
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FormulaWorkerError", "REPORT_KIND", "main", "run_formula_worker"]

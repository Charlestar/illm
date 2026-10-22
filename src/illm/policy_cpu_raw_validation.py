"""Portable derivation of the V0.2 CPU raw-tensor numerical report.

This module is a reader, not an execution oracle.  It imports no tensor
runtime.  Worker claims are accepted only after their closed metadata, tensor
pack inventory, sampling transitions, comparison records, and cross-worker
bindings have been independently recomputed from the saved bytes.

The resulting report is deliberately ``DEVELOPMENT_ONLY`` and
``unqualified``.  A successful CPU derivation does not complete the GPU,
native-dtype, release, or LTS gates described by the protocol.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, BinaryIO, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.numerical_policy import parse_policy
from illm.policy_cpu_numerical_contract import (
    FIXED_MUTATION_IDS,
    NUMERICAL_REPORT_KIND,
    PROTOCOL_SHA256,
    PUBLIC_SCOPE,
    TINY_LOAD_MANIFEST_DIGEST,
    TINY_ORACLE_MANIFEST_DIGEST,
    TINY_SPEC_ID,
    TINY_TOKENIZER_DIGEST,
    TINY_WEIGHTS_DIGEST,
    WORKER_REPORT_KIND,
    NumericalEvidenceError,
    canonical_tiny_ops,
    cpu_cases_from_snapshots,
    high_cases,
    read_authority_tensors,
    strict_json,
    tensor_id,
)
from illm.policy_tensor_evidence import (
    EvidenceTensor,
    TensorPack,
    compare_evidence,
    slice_position,
)
from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample


_SHA256 = re.compile(r"[0-9a-f]{64}")
_WORKER_DIRECTORIES = {
    "native": "native",
    "candidate": "candidate",
    "formula": "formula",
    "mutations": "mutation",
    "reproduction": "candidate-reproduction",
}
_RUNTIME_TOP = {
    "schema_version", "report_kind", "status", "evidence_stage",
    "qualification_status", "role", "coverage", "binding", "binding_digest",
    "process", "model", "policy", "pack_manifest_digest", "cases", "high_cases",
    "op_authority_cases",
}
_FORMULA_TOP = {
    "schema_version", "report_kind", "status", "evidence_stage",
    "qualification_status", "role", "coverage", "binding", "binding_digest",
    "process", "model", "policy", "pack_manifest_digest", "cases", "high_cases",
}
_MUTATION_TOP = {
    "schema_version", "report_kind", "status", "evidence_stage",
    "qualification_status", "role", "coverage", "binding", "binding_digest",
    "process", "model", "policy", "pack_manifest_digest", "mutations",
}
_CASE_FIELDS = {
    "case_id", "input_token_ids", "max_new_tokens", "sampling_spec", "input_digest",
    "sampling_spec_digest", "suite_id", "expected_output_token_ids", "case_digest", "paths",
}
_PATH_FIELDS = {
    "path_id", "execution_flags", "steps", "output_token_ids", "finish_reason",
    "resource_status",
}
_HIGH_PATH_FIELDS = _PATH_FIELDS | {"full_logits_id"}
_STEP_FIELDS = {
    "step", "context", "output_token_id", "logits_id", "state_before", "state_after",
    "query_tokens", "kv_before", "kv_after", "op_ids", "prefix_ids",
    "ordinary_logits_id", "fence_complete",
}
_FLAGS_FIELDS = {
    "autocast_cpu", "deterministic_algorithms", "deterministic_warn_only",
    "float32_matmul_precision", "tf32_matmul", "tf32_cudnn",
    "fp16_reduced_precision_reduction", "bf16_reduced_precision_reduction",
    "cpu_threads", "cublas_workspace_config", "compile",
}
_RESOURCE_FIELDS = {
    "closed", "active_leases", "allocated_bytes", "valid_tokens", "staged_tokens",
}
_BASE_MODEL_FIELDS = {
    "model_spec", "model_spec_id", "weight_manifest", "weight_manifest_digest",
    "weights_digest", "tokenizer", "tokenizer_digest", "asset_files",
}
_RUNTIME_MODEL_FIELDS = _BASE_MODEL_FIELDS | {"native_oracle"}
_MODEL_AUTHORITY_FILES = {
    "config.json", "model.safetensors", "supported-model-spec.json",
    "weight-load-manifest.json", "tokenizer.json", "fixture-manifest.json",
}
_NATIVE_ORACLE_FIELDS = {
    "kind", "transformers_version", "model_class", "source", "device",
    "parameter_dtype", "rope_inv_freq_dtype", "attention_implementation",
    "use_cache", "asset_files", "fixture_manifest_digest", "spec_id",
    "weight_manifest_digest", "weights_digest", "source_shards",
}
_MUTATION_FIELDS = {
    "mutation_id", "status", "input_identity", "modified_input_identity",
    "expected_detector", "observation", "tensor_paths", "resource_fence",
}
_FENCE_FIELDS = {
    "required", "fence_kind", "fence_complete", "fence_succeeded", "lease_state",
    "active_leases_after_cleanup", "store_closed",
}
_COMPARISON_KEYS = {
    "status", "reason", "expected_shape", "actual_shape", "expected_dtype",
    "actual_dtype", "expected_sha256", "actual_sha256", "cross_precision", "atol",
    "rtol", "element_count", "first_flat_index", "expected_value", "actual_value",
    "max_absolute_error", "max_relative_error", "mismatch_count", "nonfinite_count",
    "bitwise_equal",
}
_ATOL = 1.0e-4
_RTOL = 1.0e-3
_REPARSE = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _fail(message: str) -> NoReturn:
    raise NumericalEvidenceError(message)


def _mapping(value: object, *, fields: set[str] | None, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{label} must be an object with string keys")
    result = dict(cast(Mapping[str, Any], value))
    if fields is not None and set(result) != fields:
        _fail(f"{label} fields differ from the closed contract")
    return result


def _list(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be a list")
    return value


def _string(value: object, *, label: str, choices: set[str] | None = None) -> str:
    if type(value) is not str or not value or (choices is not None and value not in choices):
        _fail(f"{label} is invalid")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be bool")
    return value


def _sha(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _json_equal(left: object, right: object) -> bool:
    try:
        return canonical_json_bytes(left) == canonical_json_bytes(right)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise NumericalEvidenceError(f"cannot inspect evidence path {path}") from exc
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _ordinary_directory(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail(f"{label} cannot contain links or reparse points")
    if not absolute.is_dir() or _is_linklike(absolute):
        _fail(f"{label} must be an existing ordinary directory")
    return absolute


def _ordinary_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail(f"{label} cannot contain links or reparse points")
    if not absolute.is_file() or _is_linklike(absolute):
        _fail(f"{label} must be an existing ordinary file")
    return absolute


def _exclusive_output(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in absolute.parents:
        if os.path.lexists(candidate) and _is_linklike(candidate):
            _fail(f"{label} ancestors cannot contain links or reparse points")
    if os.path.lexists(absolute):
        _fail(f"{label} must not already exist")
    if not absolute.parent.is_dir() or _is_linklike(absolute.parent):
        _fail(f"{label} parent must be an ordinary directory")
    return absolute


@dataclass(slots=True)
class _Axis:
    comparisons: int = 0
    failures: int = 0
    incomplete_reasons: list[str] = field(default_factory=list)

    def observe(self, result: Mapping[str, object], *, require_bitwise: bool = False) -> None:
        self.comparisons += 1
        if result.get("status") != "pass" or (require_bitwise and result.get("bitwise_equal") is not True):
            self.failures += 1

    def incomplete(self, reason: str) -> None:
        if reason not in self.incomplete_reasons:
            self.incomplete_reasons.append(reason)

    def report(self, *, allow_empty: bool = False) -> dict[str, object]:
        if self.failures:
            status = "fail"
        elif self.incomplete_reasons or (self.comparisons == 0 and not allow_empty):
            status = "incomplete"
        else:
            status = "pass"
        return {
            "status": status,
            "comparison_count": self.comparisons,
            "failure_count": self.failures,
            "incomplete_reasons": list(self.incomplete_reasons),
        }


class _RecordSink:
    def __init__(self, *, output: Path | None, verify: Path | None) -> None:
        if output is not None and verify is not None:
            _fail("records_path and verify_records_path are mutually exclusive")
        self._output_path = _exclusive_output(output, label="comparison records") if output else None
        self._verify_path = _ordinary_file(verify, label="comparison records") if verify else None
        self._output: BinaryIO | None = None
        self._verify: BinaryIO | None = None
        if self._output_path is not None:
            self._output = self._output_path.open("xb")
        if self._verify_path is not None:
            self._verify = self._verify_path.open("rb")
        self._digest = hashlib.sha256()
        self.count = 0

    def add(self, record: Mapping[str, object]) -> None:
        encoded = canonical_json_bytes(dict(record)) + b"\n"
        self._digest.update(encoded)
        self.count += 1
        if self._output is not None:
            self._output.write(encoded)
        if self._verify is not None:
            existing = self._verify.readline()
            if existing != encoded:
                _fail(f"comparison record {self.count - 1} differs from independent recomputation")

    def finish(self) -> dict[str, object]:
        try:
            if self._verify is not None and self._verify.read(1) != b"":
                _fail("comparison records contain an unverified trailing record or byte")
            if self._output is not None:
                self._output.flush()
                os.fsync(self._output.fileno())
        finally:
            if self._output is not None:
                self._output.close()
                self._output = None
            if self._verify is not None:
                self._verify.close()
                self._verify = None
        return {
            "path": "comparisons.jsonl",
            "record_count": self.count,
            "sha256": self._digest.hexdigest(),
        }

    def abort(self) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None
        if self._verify is not None:
            self._verify.close()
            self._verify = None


@dataclass(slots=True)
class _Worker:
    directory_name: str
    role: str
    root: Path
    report: dict[str, Any]
    report_sha256: str
    pack_manifest_digest: str
    pack: TensorPack
    consumed: set[str] = field(default_factory=set)
    tensor_cache: OrderedDict[str, EvidenceTensor] = field(default_factory=OrderedDict)

    def tensor(self, semantic_id: object, *, expected_id: str | None = None) -> EvidenceTensor:
        identifier = _string(semantic_id, label=f"{self.role} tensor id")
        if expected_id is not None and identifier != expected_id:
            _fail(f"{self.role} tensor id differs from its semantic identity")
        if identifier in self.tensor_cache:
            tensor = self.tensor_cache.pop(identifier)
            self.tensor_cache[identifier] = tensor
            return tensor
        if identifier not in self.pack.records:
            _fail(f"{self.role} references an absent tensor {identifier!r}")
        self.consumed.add(identifier)
        tensor = self.pack.tensor(identifier)
        self.tensor_cache[identifier] = tensor
        if len(self.tensor_cache) > 32:
            self.tensor_cache.popitem(last=False)
        return tensor

    def require_complete_inventory(self) -> None:
        actual = set(self.pack.records)
        if self.consumed != actual:
            missing = sorted(actual - self.consumed)
            absent = sorted(self.consumed - actual)
            _fail(f"{self.role} tensor inventory is not exact; unreferenced={missing[:3]}, absent={absent[:3]}")


def _read_worker(root: Path, directory_name: str, expected_role: str) -> _Worker:
    worker_root = _ordinary_directory(root / "workers" / directory_name, label=f"{directory_name} worker")
    names = {entry.name for entry in worker_root.iterdir()}
    if names != {"pack", "worker.json", "COMPLETE"}:
        _fail(f"{directory_name} worker inventory must be exactly pack/worker.json/COMPLETE")
    report_path = _ordinary_file(worker_root / "worker.json", label=f"{directory_name} worker report")
    complete_path = _ordinary_file(worker_root / "COMPLETE", label=f"{directory_name} COMPLETE")
    raw = report_path.read_bytes()
    value = strict_json(raw, label=f"{directory_name} worker report")
    report = _mapping(value, fields=None, label=f"{directory_name} worker report")
    canonical = canonical_json_bytes(report)
    if raw not in (canonical, canonical + b"\n"):
        _fail(f"{directory_name} worker report is not canonical JSON")
    digest = sha256_bytes(raw)
    if complete_path.read_bytes() != (digest + "\n").encode("ascii"):
        _fail(f"{directory_name} COMPLETE does not bind exact worker.json bytes")
    fields = _MUTATION_TOP if expected_role == "mutation" else (
        _FORMULA_TOP if expected_role == "formula" else _RUNTIME_TOP
    )
    if set(report) != fields:
        _fail(f"{directory_name} worker top-level fields differ")
    if (
        report.get("schema_version") != 1
        or report.get("report_kind") != WORKER_REPORT_KIND
        or report.get("role") != expected_role
        or report.get("evidence_stage") != "DEVELOPMENT_ONLY"
        or report.get("qualification_status") != "unqualified"
        or report.get("status") != ("complete" if expected_role in {"native", "candidate", "candidate-reproduction"} else "pass")
    ):
        _fail(f"{directory_name} worker lifecycle or role fields differ")
    pack = TensorPack.open(worker_root / "pack")
    pack_manifest = _mapping(
        strict_json((worker_root / "pack" / "manifest.json").read_bytes(), label="tensor pack manifest"),
        fields=None,
        label="tensor pack manifest",
    )
    pack_digest = _sha(pack_manifest.get("manifest_digest"), label="tensor pack manifest digest")
    if report.get("pack_manifest_digest") != pack_digest:
        _fail(f"{directory_name} report does not bind its tensor pack")
    return _Worker(directory_name, expected_role, worker_root, report, digest, pack_digest, pack)


def _validate_binding(worker: _Worker) -> dict[str, Any]:
    binding = _mapping(
        worker.report["binding"],
        fields={
            "schema_version", "implementation_source_scope", "implementation_source_digest",
            "environment", "environment_digest", "toolchain", "protocol_sha256",
        },
        label=f"{worker.role} binding",
    )
    if (
        binding["schema_version"] != 1
        or binding["implementation_source_scope"] != "project-root-v1"
        or binding["protocol_sha256"] != PROTOCOL_SHA256
        or worker.report["binding_digest"] != sha256_json(binding)
    ):
        _fail(f"{worker.role} execution binding identity differs")
    _sha(binding["implementation_source_digest"], label="implementation source digest")
    environment = _mapping(
        binding["environment"],
        fields={
            "backend", "python_version", "platform", "machine", "torch_version",
            "cuda_arithmetic", "cuda_execution_authorized",
        },
        label=f"{worker.role} environment",
    )
    if (
        environment["backend"] != "cpu"
        or environment["cuda_arithmetic"] != "not_applicable"
        or environment["cuda_execution_authorized"] is not False
        or any(type(environment[name]) is not str or not environment[name]
               for name in ("python_version", "platform", "machine", "torch_version"))
        or binding["environment_digest"] != sha256_json(environment)
    ):
        _fail(f"{worker.role} CPU environment binding differs")
    toolchain = _mapping(binding["toolchain"], fields=None, label=f"{worker.role} toolchain")
    if set(toolchain) != {"binding", "manifest_digest", "environment_id", "lock_digest"}:
        _fail(f"{worker.role} requires a source-bound finalized toolchain")
    if toolchain.get("binding") != "source-bound-manifest":
        _fail(f"{worker.role} toolchain is not source-bound")
    for name in ("manifest_digest", "environment_id", "lock_digest"):
        _sha(toolchain[name], label=f"toolchain {name}")
    return binding


def _validate_process(worker: _Worker) -> int:
    process = _mapping(worker.report["process"], fields=None, label=f"{worker.role} process")
    if worker.role in {"native", "candidate", "candidate-reproduction"}:
        fields = {
            "pid", "python_executable", "python_version", "package_version",
            "worker_source_sha256", "implementation_source_digest", "cuda_visible_devices",
            "cuda_initialized_before", "cuda_initialized_after",
            "native_policy_module_loaded_before", "native_policy_module_loaded_after",
        }
        if set(process) != fields:
            _fail(f"{worker.role} process fields differ")
        _sha(process["worker_source_sha256"], label="runtime worker source")
        _sha(process["implementation_source_digest"], label="runtime implementation source")
        if process["implementation_source_digest"] != worker.report["binding"]["implementation_source_digest"]:
            _fail(f"{worker.role} process/source binding differs")
        if (
            process["cuda_visible_devices"] != ""
            or process["cuda_initialized_before"] is not False
            or process["cuda_initialized_after"] is not False
        ):
            _fail(f"{worker.role} CPU process used or exposed CUDA")
        if worker.role == "native" and (
            process["native_policy_module_loaded_before"] is not False
            or process["native_policy_module_loaded_after"] is not False
        ):
            _fail("native worker imported the candidate policy module")
        if worker.role != "native" and (
            process["native_policy_module_loaded_before"] is not None
            or process["native_policy_module_loaded_after"] is not None
        ):
            _fail(f"{worker.role} native-isolation fields must be null")
    elif worker.role == "formula":
        fields = {
            "pid", "python_executable", "python_version", "platform", "machine", "byte_order",
            "cpu_threads", "packages", "cuda_internal_initialized_start",
            "cuda_internal_initialized_end", "forbidden_modules_loaded_start",
            "forbidden_modules_loaded_end",
        }
        if set(process) != fields:
            _fail("formula process fields differ")
        if (
            process["byte_order"] != "little" or process["cpu_threads"] != 1
            or process["cuda_internal_initialized_start"] is not False
            or process["cuda_internal_initialized_end"] is not False
            or process["forbidden_modules_loaded_start"] != []
            or process["forbidden_modules_loaded_end"] != []
        ):
            _fail("formula process isolation differs")
        packages = _mapping(process["packages"], fields={"torch", "safetensors"}, label="formula packages")
        if any(type(value) is not str or not value for value in packages.values()):
            _fail("formula package versions are invalid")
    else:
        fields = {
            "pid", "python_executable", "python_version", "platform", "machine", "byte_order",
            "cpu_threads", "packages", "cuda_internal_initialized_start",
            "cuda_internal_initialized_end", "formula_worker_module_loaded_start",
            "formula_worker_module_loaded_end",
        }
        if set(process) != fields:
            _fail("mutation process fields differ")
        if (
            process["byte_order"] != "little" or process["cpu_threads"] != 1
            or process["cuda_internal_initialized_start"] is not False
            or process["cuda_internal_initialized_end"] is not False
            or process["formula_worker_module_loaded_start"] is not False
            or process["formula_worker_module_loaded_end"] is not False
        ):
            _fail("mutation process isolation differs")
        packages = _mapping(
            process["packages"], fields={"torch", "safetensors", "transformers"},
            label="mutation packages",
        )
        if any(type(value) is not str or not value for value in packages.values()):
            _fail("mutation package versions are invalid")
    for name in ("python_executable", "python_version"):
        if type(process[name]) is not str or not process[name]:
            _fail(f"{worker.role} process {name} is invalid")
    return _integer(process["pid"], label=f"{worker.role} pid", minimum=1)


def _validate_flags(value: object, *, label: str) -> None:
    flags = _mapping(value, fields=_FLAGS_FIELDS, label=label)
    expected: dict[str, object] = {
        "autocast_cpu": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "float32_matmul_precision": "highest",
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
        "cpu_threads": 1,
        "cublas_workspace_config": ":16:8",
        "compile": False,
    }
    if flags != expected:
        _fail(f"{label} differs from the canonical CPU policy flags")


def _validate_zero_resources(value: object, *, label: str) -> None:
    resources = _mapping(value, fields=_RESOURCE_FIELDS, label=label)
    if resources != {
        "closed": True,
        "active_leases": 0,
        "allocated_bytes": 0,
        "valid_tokens": 0,
        "staged_tokens": 0,
    }:
        _fail(f"{label} does not prove safe resource reclamation")


def _validate_models(workers: Mapping[str, _Worker]) -> None:
    native = _mapping(
        workers["native"].report["model"], fields=_RUNTIME_MODEL_FIELDS, label="native model"
    )
    if (
        native["model_spec_id"] != TINY_SPEC_ID
        or native["weights_digest"] != TINY_WEIGHTS_DIGEST
        or native["weight_manifest_digest"] != TINY_LOAD_MANIFEST_DIGEST
        or native["tokenizer_digest"] != sha256_json(native["tokenizer"])
        or native["tokenizer_digest"] != TINY_TOKENIZER_DIGEST
    ):
        _fail("runtime model identity differs from the approved tiny model")
    spec = _mapping(native["model_spec"], fields=None, label="runtime model spec")
    if spec.get("spec_id") != TINY_SPEC_ID or spec.get("runtime_dtype") != "fp32":
        _fail("runtime model spec is not the approved FP32 tiny spec")
    if native["weight_manifest_digest"] != sha256_json(
        {key: value for key, value in _mapping(native["weight_manifest"], fields=None, label="weight manifest").items()
         if key != "manifest_digest"}
    ):
        _fail("runtime weight manifest digest is not canonical")
    base_native = {key: native[key] for key in _BASE_MODEL_FIELDS}
    for name in ("candidate", "reproduction"):
        model = _mapping(
            workers[name].report["model"], fields=_RUNTIME_MODEL_FIELDS,
            label=f"{name} model",
        )
        if model["native_oracle"] is not None or not _json_equal(
            {key: model[key] for key in _BASE_MODEL_FIELDS}, base_native
        ):
            _fail(f"{name} model identity differs from native")

    oracle = _mapping(
        native["native_oracle"], fields=_NATIVE_ORACLE_FIELDS,
        label="D-native oracle identity",
    )
    if (
        oracle["kind"] != "independent-pinned-transformers-oracle-v1"
        or oracle["transformers_version"] != "4.57.1"
        or oracle["model_class"]
        != "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM"
        or oracle["device"] != "cpu"
        or oracle["parameter_dtype"] != "torch.float32"
        or oracle["rope_inv_freq_dtype"] != "torch.float32"
        or oracle["attention_implementation"] != "eager"
        or oracle["use_cache"] is not False
        or oracle["spec_id"] != native["model_spec_id"]
        or oracle["weight_manifest_digest"] != native["weight_manifest_digest"]
        or oracle["weights_digest"] != native["weights_digest"]
    ):
        _fail("D-native oracle execution or model identity differs")
    oracle_source = _mapping(
        oracle["source"], fields={"module", "sha256"}, label="D-native oracle source"
    )
    if oracle_source["module"] != "transformers.models.qwen2.modeling_qwen2":
        _fail("D-native oracle source module differs")
    _sha(oracle_source["sha256"], label="D-native oracle source digest")
    _sha(oracle["fixture_manifest_digest"], label="D-native fixture manifest digest")

    base_assets = _mapping(native["asset_files"], fields=None, label="runtime model asset inventory")
    oracle_assets = _mapping(
        oracle["asset_files"], fields=_MODEL_AUTHORITY_FILES,
        label="D-native oracle asset inventory",
    )
    if not _json_equal(
        oracle_assets, {name: base_assets.get(name) for name in _MODEL_AUTHORITY_FILES}
    ):
        _fail("D-native oracle asset bytes differ from the runtime model")
    for name, digest in oracle_assets.items():
        _sha(digest, label=f"D-native oracle asset {name}")

    manifest = _mapping(native["weight_manifest"], fields=None, label="weight manifest")
    source_shards = _list(oracle["source_shards"], label="D-native source shards")
    if not source_shards or not _json_equal(source_shards, manifest.get("source_shards")):
        _fail("D-native source shard identity differs from the fresh weight manifest")
    for index, shard_value in enumerate(source_shards):
        shard = _mapping(
            shard_value, fields={"shard", "sha256"}, label=f"D-native source shard {index}"
        )
        shard_name = _string(shard["shard"], label="D-native source shard name")
        if Path(shard_name).name != shard_name:
            _fail("D-native source shard name is not a basename")
        _sha(shard["sha256"], label="D-native source shard digest")

    formula = _mapping(
        workers["formula"].report["model"],
        fields={"kind", "spec_id", "oracle_identity", "source", "candidate_loaded"},
        label="formula model",
    )
    oracle_identity = _mapping(
        formula["oracle_identity"],
        fields={"fixture_manifest_digest", "model_sha256", "spec_id", "weight_manifest_digest", "weights_digest"},
        label="formula oracle identity",
    )
    if (
        formula["kind"] != "independent-binary64-tiny-formula"
        or formula["spec_id"] != TINY_SPEC_ID
        or formula["candidate_loaded"] is not False
        or oracle_identity["spec_id"] != TINY_SPEC_ID
        or oracle_identity["weight_manifest_digest"] != TINY_LOAD_MANIFEST_DIGEST
        or oracle_identity["weights_digest"] != TINY_WEIGHTS_DIGEST
    ):
        _fail("formula model identity differs from the approved tiny model")
    for name in ("fixture_manifest_digest", "model_sha256"):
        _sha(oracle_identity[name], label=f"formula {name}")
    source = _mapping(formula["source"], fields={"path", "sha256"}, label="formula source")
    _string(source["path"], label="formula source path")
    _sha(source["sha256"], label="formula source digest")

    mutation = _mapping(
        workers["mutations"].report["model"],
        fields={"fixture_id", "spec_id", "runtime_dtype", "weight_manifest_digest", "source"},
        label="mutation model",
    )
    if (
        mutation["spec_id"] != TINY_SPEC_ID
        or mutation["runtime_dtype"] != "fp32"
        or mutation["weight_manifest_digest"] != TINY_LOAD_MANIFEST_DIGEST
    ):
        _fail("mutation model identity differs")
    mutation_source = _mapping(mutation["source"], fields={"path", "sha256"}, label="mutation source")
    _string(mutation_source["path"], label="mutation source path")
    _sha(mutation_source["sha256"], label="mutation source digest")


def _validate_authority_assets(root: Path, workers: Mapping[str, _Worker], oracle_raw: bytes) -> dict[str, str]:
    model = cast(Mapping[str, Any], workers["native"].report["model"])
    assets = _mapping(model["asset_files"], fields=None, label="runtime model asset inventory")
    required = tuple(sorted(_MODEL_AUTHORITY_FILES))
    result: dict[str, str] = {}
    for name in required:
        raw = _ordinary_file(root / "authorities" / name, label=f"authority {name}").read_bytes()
        digest = sha256_bytes(raw)
        if assets.get(name) != digest:
            _fail(f"authority {name} bytes differ from the runtime worker asset identity")
        result[name] = digest
    if assets.get("oracle-v1/oracle-manifest.json") != sha256_bytes(oracle_raw):
        _fail("copied oracle manifest differs from the runtime worker asset identity")
    weight_manifest = _mapping(
        strict_json((root / "authorities" / "weight-load-manifest.json").read_bytes(),
                    label="copied weight manifest"),
        fields=None,
        label="copied weight manifest",
    )
    tokenizer = _mapping(
        strict_json((root / "authorities" / "tokenizer.json").read_bytes(), label="copied tokenizer"),
        fields=None,
        label="copied tokenizer",
    )
    spec = _mapping(
        strict_json((root / "authorities" / "supported-model-spec.json").read_bytes(),
                    label="copied model spec"),
        fields=None,
        label="copied model spec",
    )
    if (
        not _json_equal(weight_manifest, model["weight_manifest"])
        or not _json_equal(tokenizer, model["tokenizer"])
        or not _json_equal(spec, model["model_spec"])
    ):
        _fail("copied spec/weight/tokenizer authorities differ from runtime parsed identities")
    fixture = _mapping(
        strict_json((root / "authorities" / "fixture-manifest.json").read_bytes(),
                    label="copied fixture manifest"),
        fields=None,
        label="copied fixture manifest",
    )
    fixture_digest = fixture.get("manifest_digest")
    if fixture_digest != sha256_json({key: value for key, value in fixture.items() if key != "manifest_digest"}):
        _fail("copied fixture manifest self-identity differs")
    formula_identity = cast(Mapping[str, Any], cast(Mapping[str, Any], workers["formula"].report["model"])["oracle_identity"])
    if formula_identity.get("fixture_manifest_digest") != fixture_digest:
        _fail("formula worker does not bind the copied fixture manifest")
    native_oracle = cast(
        Mapping[str, Any], cast(Mapping[str, Any], workers["native"].report["model"])["native_oracle"]
    )
    if native_oracle.get("fixture_manifest_digest") != fixture_digest:
        _fail("D-native oracle does not bind the copied fixture manifest")
    result["oracle-manifest.json"] = sha256_bytes(oracle_raw)
    return result


def _validate_policies(workers: Mapping[str, _Worker]) -> None:
    if workers["native"].report["policy"] is not None or workers["formula"].report["policy"] is not None:
        _fail("native and formula workers cannot claim a candidate policy")
    candidate = _mapping(
        workers["candidate"].report["policy"],
        fields={"descriptor", "digest", "sidecar_bytes"},
        label="candidate policy",
    )
    policy = parse_policy(_mapping(candidate["descriptor"], fields=None, label="candidate descriptor"))
    if (
        policy.digest != candidate["digest"]
        or not policy.is_candidate
        or policy.family != "illm-rowwise-fp32-contraction"
        or policy.storage_dtype != "fp32"
    ):
        _fail("candidate policy identity differs from the approved FP32 policy")
    sidecars = _mapping(candidate["sidecar_bytes"], fields={"B", "C"}, label="candidate sidecars")
    if (
        _integer(sidecars["B"], label="B sidecar bytes", minimum=1)
        != _integer(sidecars["C"], label="C sidecar bytes", minimum=1)
    ):
        _fail("B and C sidecar byte identities differ")
    if not _json_equal(workers["reproduction"].report["policy"], candidate):
        _fail("reproduction policy identity differs from the main candidate")
    mutation = _mapping(
        workers["mutations"].report["policy"], fields={"fp32", "bf16"}, label="mutation policies"
    )
    for dtype in ("fp32", "bf16"):
        item = _mapping(mutation[dtype], fields={"descriptor", "digest"}, label=f"mutation {dtype} policy")
        parsed = parse_policy(_mapping(item["descriptor"], fields=None, label=f"mutation {dtype} descriptor"))
        if parsed.digest != item["digest"] or parsed.storage_dtype != dtype or not parsed.is_candidate:
            _fail(f"mutation {dtype} policy identity differs")


def _candidate_shape(op: str, position: int) -> tuple[int, ...]:
    suffix = op.split(".")[-1]
    if op in {"embedding", "final_norm"} or suffix in {
        "input_norm", "attention_output", "attention_residual", "post_attention_norm",
        "mlp_output", "mlp_residual",
    }:
        return (1, 32)
    if suffix in {"q_projection", "rope_q"}:
        return (1, 4, 1, 8)
    if suffix in {"k_projection", "v_projection", "rope_k"}:
        return (1, 2, 1, 8)
    if suffix in {"attention_scores", "attention_probabilities"}:
        return (1, 4, 1, position + 1)
    if suffix == "attention_context":
        return (1, 4, 1, 8)
    if suffix in {"mlp_gate", "mlp_up", "mlp_silu", "mlp_product"}:
        return (1, 64)
    if op == "logits":
        return (1, 259)
    _fail(f"unknown canonical op {op!r}")


def _native_shape(op: str, length: int) -> tuple[int, ...]:
    row = _candidate_shape(op, max(length - 1, 0))
    if len(row) == 2:
        return (1, length, row[-1])
    if len(row) == 3:
        return (1, row[1], length, row[2])
    if op.endswith("attention_scores") or op.endswith("attention_probabilities"):
        return (1, 4, length, length)
    return (1, row[1], length, row[-1])


def _finite(tensor: EvidenceTensor, *, label: str) -> None:
    if any(not math.isfinite(value) for value in tensor.values):
        _fail(f"{label} contains a non-finite value")


def _row_from_full_logits(tensor: EvidenceTensor) -> tuple[float, ...]:
    if tensor.dtype != "fp32" or len(tensor.shape) != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 259:
        _fail("sampling logits must be FP32 [1,Q,259]")
    _finite(tensor, label="sampling logits")
    return tensor.values[-259:]


def _state_record(state: SamplingState) -> dict[str, object]:
    return {"digest": state.digest, "value": state.to_dict()}


def _base_case(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value if key != "paths"}


def _validate_case_projection(value: object, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    case = _mapping(value, fields=_CASE_FIELDS, label=label)
    if not _json_equal(_base_case(case), expected):
        _fail(f"{label} differs from its approved case bytes")
    return case


def _validate_path_base(value: object, path_id: str, *, high: bool = False) -> dict[str, Any]:
    path = _mapping(value, fields=_HIGH_PATH_FIELDS if high else _PATH_FIELDS, label=f"{path_id} path")
    if path["path_id"] != path_id:
        _fail(f"{path_id} path identity differs")
    _validate_flags(path["execution_flags"], label=f"{path_id} flags")
    _validate_zero_resources(path["resource_status"], label=f"{path_id} resources")
    return path


def _validate_stateful_path(
    worker: _Worker,
    value: object,
    *,
    path_id: str,
    case: Mapping[str, Any],
    candidate: bool,
) -> dict[str, Any]:
    path = _validate_path_base(value, path_id)
    steps = _list(path["steps"], label=f"{path_id} steps")
    maximum = _integer(case["max_new_tokens"], label="max_new_tokens", minimum=1)
    if not 1 <= len(steps) <= maximum:
        _fail(f"{path_id} step count is outside the approved request bound")
    context = tuple(_integer(token, label="input token") for token in _list(case["input_token_ids"], label="input ids"))
    sampling = SamplingSpec.from_dict(_mapping(case["sampling_spec"], fields=None, label="sampling spec"))
    state = SamplingState.for_request(cast(str, case["case_id"]), sampling)
    output: list[int] = []
    for index, step_value in enumerate(steps):
        step = _mapping(step_value, fields=_STEP_FIELDS, label=f"{path_id} step")
        if step["step"] != index or step["context"] != list(context):
            _fail(f"{path_id} step context or index differs from its own committed history")
        if step["state_before"] != _state_record(state):
            _fail(f"{path_id} state_before was not independently derived")
        query = len(context) if not candidate or path_id == "B" or index == 0 else 1
        if step["query_tokens"] != query:
            _fail(f"{path_id} query_tokens differs from the execution contract")
        if path_id == "C":
            before = 0 if index == 0 else len(context) - 1
            after = len(context)
            if step["kv_before"] != before or step["kv_after"] != after or step["fence_complete"] is not True:
                _fail("C KV commit/fence accounting differs")
        elif step["kv_before"] != 0 or step["kv_after"] != 0 or step["fence_complete"] is not None:
            _fail(f"{path_id} cannot claim KV or a completion fence")
        logits_id = _string(step["logits_id"], label=f"{path_id} logits id")
        logits = worker.tensor(logits_id, expected_id=tensor_id(path_id, cast(str, case["case_id"]), index,
                                                                 "full_logits" if candidate else "logits", None))
        raw_logits = _row_from_full_logits(logits)
        decision = prepare_sample(raw_logits, sampling, state)
        state = commit_sample(state, decision, spec=sampling, logits=raw_logits)
        if step["output_token_id"] != decision.token_id or step["state_after"] != _state_record(state):
            _fail(f"{path_id} token or committed sampling state differs from raw logits")
        output.append(decision.token_id)
        context = (*context, decision.token_id)
        if decision.token_id == 2:
            if index != len(steps) - 1:
                _fail(f"{path_id} continued after EOS")
            break
    finish = "EOS" if output[-1] == 2 else "MAX_NEW_TOKENS"
    if finish == "MAX_NEW_TOKENS" and len(steps) != maximum:
        _fail(f"{path_id} stopped without EOS before max_new_tokens")
    if path["output_token_ids"] != output or path["finish_reason"] != finish:
        _fail(f"{path_id} output or finish reason differs from recomputation")
    expected_outputs = case["expected_output_token_ids"]
    if expected_outputs is not None and output != expected_outputs:
        _fail(f"{path_id} differs from the frozen tiny20 golden output")
    return path


def _candidate_positions(step: Mapping[str, Any], *, path_id: str) -> list[int]:
    context = cast(list[int], step["context"])
    if path_id == "B" or (path_id == "C" and step["step"] == 0):
        return list(range(len(context)))
    if path_id == "C":
        return [len(context) - 1]
    _fail("candidate position request used an unsupported path")


def _validate_candidate_ops(
    worker: _Worker,
    path: Mapping[str, Any],
    *,
    path_id: str,
    case_id: str,
) -> None:
    ops = canonical_tiny_ops()
    for step_value in cast(list[Any], path["steps"]):
        step = cast(dict[str, Any], step_value)
        positions = _candidate_positions(step, path_id=path_id)
        op_ids = _mapping(step["op_ids"], fields=set(ops), label=f"{path_id} op ids")
        for op in ops:
            per_position = _mapping(op_ids[op], fields={str(pos) for pos in positions}, label=f"{path_id} {op} ids")
            for position in positions:
                identifier = tensor_id(path_id, case_id, cast(int, step["step"]), op, position)
                tensor = worker.tensor(per_position[str(position)], expected_id=identifier)
                if tensor.dtype != "fp32" or tensor.shape != _candidate_shape(op, position):
                    _fail(f"{path_id} {op} tensor shape/dtype differs at position {position}")
                _finite(tensor, label=f"{path_id} {op}")
        full = worker.tensor(step["logits_id"])
        expected_values: list[float] = []
        expected_raw = bytearray()
        for position in positions:
            row = worker.tensor(cast(Mapping[str, Any], op_ids["logits"])[str(position)])
            expected_values.extend(row.values)
            expected_raw.extend(row.raw)
        assembled = EvidenceTensor("fp32", (1, len(positions), 259), tuple(expected_values),
                                   sha256_bytes(bytes(expected_raw)), bytes(expected_raw))
        result = compare_evidence(assembled, full, atol=0.0, rtol=0.0)
        if result["bitwise_equal"] is not True:
            _fail(f"{path_id} full logits are not the exact emitted logits rows")
        prefix_ids = _mapping(step["prefix_ids"], fields=None, label=f"{path_id} prefix ids")
        if path_id == "C" and cast(int, step["step"]) > 0:
            if set(prefix_ids) != {"0", "1"}:
                _fail("C decode must expose both committed K/V layer prefixes")
            prefix_length = len(cast(list[int], step["context"])) - 1
            for layer in range(2):
                item = _mapping(prefix_ids[str(layer)], fields={"key", "value"}, label="C prefix layer")
                for kind in ("key", "value"):
                    identifier = tensor_id(path_id, case_id, cast(int, step["step"]),
                                           f"prefix.layer.{layer}.{kind}", None)
                    prefix = worker.tensor(item[kind], expected_id=identifier)
                    if prefix.dtype != "fp32" or prefix.shape != (1, 2, prefix_length, 8):
                        _fail("C committed prefix tensor shape/dtype differs")
                    _finite(prefix, label="C committed prefix")
        elif prefix_ids:
            _fail(f"{path_id} cannot expose committed prefixes at this step")


def _validate_native_ops(
    worker: _Worker,
    path: Mapping[str, Any],
    *,
    path_id: str,
    case_id: str,
) -> None:
    ops = canonical_tiny_ops()
    for step_value in cast(list[Any], path["steps"]):
        step = cast(dict[str, Any], step_value)
        length = len(cast(list[int], step["context"]))
        op_ids = _mapping(step["op_ids"], fields=set(ops), label=f"{path_id} op ids")
        for op in ops:
            identifier = tensor_id(path_id, case_id, cast(int, step["step"]), op, None)
            tensor = worker.tensor(op_ids[op], expected_id=identifier)
            if tensor.dtype != "fp32" or tensor.shape != _native_shape(op, length):
                _fail(f"{path_id} {op} full tensor shape/dtype differs")
            if op.endswith("attention_scores"):
                for position in range(length):
                    _finite(slice_position(tensor, op, position), label=f"{path_id} legal score prefix")
            else:
                _finite(tensor, label=f"{path_id} {op}")
            if op.endswith("attention_probabilities"):
                _, heads, queries, keys = tensor.shape
                for head in range(heads):
                    for query in range(queries):
                        base = ((head * queries) + query) * keys
                        if any(tensor.values[base + key] != 0.0 for key in range(query + 1, keys)):
                            _fail(f"{path_id} future attention probability is nonzero")
        if step["logits_id"] != op_ids["logits"]:
            _fail(f"{path_id} sampling logits do not reference the canonical logits op")
        ordinary = step["ordinary_logits_id"]
        if path_id == "D-native":
            expected = tensor_id(path_id, case_id, cast(int, step["step"]), "ordinary_logits", None)
            other = worker.tensor(ordinary, expected_id=expected)
            logits = worker.tensor(step["logits_id"])
            if compare_evidence(logits, other, atol=0.0, rtol=0.0)["bitwise_equal"] is not True:
                _fail("D-native trace adapter logits differ from ordinary Transformers logits")
        elif ordinary is not None:
            _fail(f"{path_id} cannot claim ordinary Transformers logits")


def _tensor_descriptor(worker: _Worker, tensor_id_value: str, source: EvidenceTensor,
                       compared: EvidenceTensor) -> dict[str, object]:
    return {
        "worker": worker.directory_name,
        "path": worker.role,
        "pack_manifest_digest": worker.pack_manifest_digest,
        "tensor_id": tensor_id_value,
        "source_raw_sha256": source.raw_sha256,
        "source_shape": list(source.shape),
        "source_dtype": source.dtype,
        "compared_raw_sha256": compared.raw_sha256,
        "compared_shape": list(compared.shape),
        "compared_dtype": compared.dtype,
    }


def _compare(
    sink: _RecordSink,
    axis: _Axis,
    *,
    axis_name: str,
    case_id: str,
    step: int,
    op: str,
    position: int | None,
    expected_worker: _Worker,
    expected_id: str,
    expected_source: EvidenceTensor,
    expected: EvidenceTensor,
    actual_worker: _Worker,
    actual_id: str,
    actual_source: EvidenceTensor,
    actual: EvidenceTensor,
    cross_precision: bool = False,
    require_bitwise: bool = False,
    expected_direction: str | None = None,
) -> dict[str, object]:
    result = compare_evidence(
        expected, actual, atol=_ATOL if not require_bitwise else 0.0,
        rtol=_RTOL if not require_bitwise else 0.0, cross_precision=cross_precision,
    )
    if set(result) != _COMPARISON_KEYS:
        _fail("tensor comparison returned an unexpected schema")
    axis.observe(result, require_bitwise=require_bitwise)
    record = {
        "schema_version": 1,
        "axis": axis_name,
        "case_id": case_id,
        "step": step,
        "op": op,
        "position": position,
        "expected_direction": expected_direction or f"{expected_worker.role}-to-{actual_worker.role}",
        "expected": _tensor_descriptor(expected_worker, expected_id, expected_source, expected),
        "actual": _tensor_descriptor(actual_worker, actual_id, actual_source, actual),
        "result": result,
    }
    record["comparison_id"] = sha256_json(record)
    sink.add(record)
    return result


def _op_row(worker: _Worker, path: Mapping[str, Any], path_id: str, step: int,
            op: str, position: int, *, native: bool) -> tuple[str, EvidenceTensor, EvidenceTensor]:
    step_value = cast(list[dict[str, Any]], path["steps"])[step]
    op_ids = cast(Mapping[str, Any], step_value["op_ids"])
    if native:
        identifier = cast(str, op_ids[op])
        source = worker.tensor(identifier)
        return identifier, source, slice_position(source, op, position)
    identifier = cast(Mapping[str, str], op_ids[op])[str(position)]
    source = worker.tensor(identifier)
    return identifier, source, source


def _compare_main_case(
    sink: _RecordSink,
    axes: Mapping[str, _Axis],
    *,
    case_id: str,
    native_worker: _Worker,
    candidate_worker: _Worker,
    a: Mapping[str, Any],
    d_native: Mapping[str, Any],
    b: Mapping[str, Any],
    c: Mapping[str, Any],
) -> None:
    outputs = [cast(list[int], path["output_token_ids"]) for path in (a, d_native, b, c)]
    finishes = [path["finish_reason"] for path in (a, d_native, b, c)]
    if not all(output == outputs[0] for output in outputs[1:]) or not all(item == finishes[0] for item in finishes[1:]):
        axes["sequence"].failures += 1
        axes["sequence"].incomplete("free-running histories diverged; cross-history element comparison forbidden")
        axes["legacy"].incomplete("A and candidate inputs diverged")
        axes["candidate"].incomplete("B and C inputs diverged")
        axes["native"].incomplete("D-native and candidate inputs diverged")
        return
    axes["sequence"].comparisons += 1
    steps = min(len(cast(list[Any], b["steps"])), len(cast(list[Any], c["steps"])))
    ops = canonical_tiny_ops()
    for step in range(steps):
        b_step = cast(dict[str, Any], cast(list[Any], b["steps"])[step])
        c_step = cast(dict[str, Any], cast(list[Any], c["steps"])[step])
        if b_step["context"] != c_step["context"]:
            _fail("B/C step contexts differ after equal output histories")
        positions = _candidate_positions(c_step, path_id="C")
        for op in ops:
            for position in positions:
                b_id, b_source, b_row = _op_row(candidate_worker, b, "B", step, op, position, native=False)
                c_id, c_source, c_row = _op_row(candidate_worker, c, "C", step, op, position, native=False)
                d_id, d_source, d_row = _op_row(native_worker, d_native, "D-native", step, op, position, native=True)
                a_id, a_source, a_row = _op_row(native_worker, a, "A", step, op, position, native=True)
                _compare(sink, axes["candidate"], axis_name="B-to-C", case_id=case_id, step=step,
                         op=op, position=position, expected_worker=candidate_worker, expected_id=b_id,
                         expected_source=b_source, expected=b_row, actual_worker=candidate_worker,
                         actual_id=c_id, actual_source=c_source, actual=c_row, expected_direction="B-expected")
                for actual_id, actual_source, actual_row, label in (
                    (b_id, b_source, b_row, "D-native-to-B"),
                    (c_id, c_source, c_row, "D-native-to-C"),
                ):
                    _compare(sink, axes["native"], axis_name=label, case_id=case_id, step=step,
                             op=op, position=position, expected_worker=native_worker, expected_id=d_id,
                             expected_source=d_source, expected=d_row, actual_worker=candidate_worker,
                             actual_id=actual_id, actual_source=actual_source, actual=actual_row,
                             expected_direction="D-native-expected")
                for actual_id, actual_source, actual_row, label in (
                    (b_id, b_source, b_row, "A-to-B"),
                    (c_id, c_source, c_row, "A-to-C"),
                ):
                    _compare(sink, axes["legacy"], axis_name=label, case_id=case_id, step=step,
                             op=op, position=position, expected_worker=native_worker, expected_id=a_id,
                             expected_source=a_source, expected=a_row, actual_worker=candidate_worker,
                             actual_id=actual_id, actual_source=actual_source, actual=actual_row,
                             expected_direction="A-expected-report-only")
        if step > 0:
            prefix_ids = cast(Mapping[str, Any], c_step["prefix_ids"])
            prefix_len = len(cast(list[int], c_step["context"])) - 1
            for layer in range(2):
                for kind, op_suffix in (("key", "rope_k"), ("value", "v_projection")):
                    prefix_id = cast(Mapping[str, str], prefix_ids[str(layer)])[kind]
                    prefix = candidate_worker.tensor(prefix_id)
                    for position in range(prefix_len):
                        b_id, b_source, b_row = _op_row(
                            candidate_worker, b, "B", step, f"layer.{layer}.{op_suffix}", position,
                            native=False,
                        )
                        values = b_row.values
                        raw = b_row.raw
                        expected = EvidenceTensor(
                            "fp32", (1, 2, 1, 8), values, sha256_bytes(raw), raw
                        )
                        actual = slice_position(prefix, f"prefix_{kind}", position)
                        _compare(
                            sink, axes["candidate"], axis_name="B-to-C-prefix", case_id=case_id,
                            step=step, op=f"layer.{layer}.prefix_{kind}", position=position,
                            expected_worker=candidate_worker, expected_id=b_id, expected_source=b_source,
                            expected=expected, actual_worker=candidate_worker, actual_id=prefix_id,
                            actual_source=prefix, actual=actual, require_bitwise=True,
                            expected_direction="B-current-full-context-row-expected",
                        )


def _runtime_cases(
    workers: Mapping[str, _Worker], expected_cases: Sequence[Mapping[str, Any]], sink: _RecordSink,
    axes: Mapping[str, _Axis],
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    expected_ids = [cast(str, case["case_id"]) for case in expected_cases]
    native_cases = _list(workers["native"].report["cases"], label="native cases")
    candidate_cases = _list(workers["candidate"].report["cases"], label="candidate cases")
    if len(native_cases) != len(expected_cases) or len(candidate_cases) != len(expected_cases):
        _fail("native/candidate case coverage differs from approved 50+20 order")
    for index, expected in enumerate(expected_cases):
        case_id = expected_ids[index]
        native_case = _validate_case_projection(native_cases[index], expected, label="native case")
        candidate_case = _validate_case_projection(candidate_cases[index], expected, label="candidate case")
        native_paths = _mapping(native_case["paths"], fields={"A", "D-native"}, label="native paths")
        candidate_paths = _mapping(candidate_case["paths"], fields={"B", "C"}, label="candidate paths")
        a = _validate_stateful_path(workers["native"], native_paths["A"], path_id="A", case=expected, candidate=False)
        d_native = _validate_stateful_path(
            workers["native"], native_paths["D-native"], path_id="D-native", case=expected, candidate=False
        )
        b = _validate_stateful_path(workers["candidate"], candidate_paths["B"], path_id="B",
                                    case=expected, candidate=True)
        c = _validate_stateful_path(workers["candidate"], candidate_paths["C"], path_id="C",
                                    case=expected, candidate=True)
        _validate_native_ops(workers["native"], a, path_id="A", case_id=case_id)
        _validate_native_ops(workers["native"], d_native, path_id="D-native", case_id=case_id)
        _validate_candidate_ops(workers["candidate"], b, path_id="B", case_id=case_id)
        _validate_candidate_ops(workers["candidate"], c, path_id="C", case_id=case_id)
        _compare_main_case(
            sink, axes, case_id=case_id, native_worker=workers["native"],
            candidate_worker=workers["candidate"], a=a, d_native=d_native, b=b, c=c,
        )
        by_id[case_id] = {"A": a, "D-native": d_native, "B": b, "C": c}
    return by_id


def _validate_reproduction(
    workers: Mapping[str, _Worker], expected_cases: Sequence[Mapping[str, Any]],
    main_cases: Mapping[str, Mapping[str, Any]], axis: _Axis,
) -> None:
    golden = [case for case in expected_cases if case["expected_output_token_ids"] is not None]
    reported = _list(workers["reproduction"].report["cases"], label="reproduction cases")
    if len(reported) != len(golden):
        _fail("reproduction must cover the exact tiny20 order")
    for index, expected in enumerate(golden):
        case = _validate_case_projection(reported[index], expected, label="reproduction case")
        case_id = cast(str, expected["case_id"])
        paths = _mapping(case["paths"], fields={"B", "C"}, label="reproduction paths")
        for path_id in ("B", "C"):
            path = _validate_stateful_path(workers["reproduction"], paths[path_id], path_id=path_id,
                                           case=expected, candidate=True)
            _validate_candidate_ops(workers["reproduction"], path, path_id=path_id, case_id=case_id)
            if not _json_equal(path, main_cases[case_id][path_id]):
                axis.failures += 1
            main_ids = _path_tensor_ids(main_cases[case_id][path_id])
            reproduction_ids = _path_tensor_ids(path)
            if main_ids != reproduction_ids:
                _fail("reproduction semantic tensor inventory differs from the main run")
            for identifier in sorted(main_ids):
                left = workers["candidate"].tensor(identifier)
                right = workers["reproduction"].tensor(identifier)
                axis.comparisons += 1
                if left.dtype != right.dtype or left.shape != right.shape or left.raw != right.raw:
                    axis.failures += 1


def _path_tensor_ids(path: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for step in cast(list[Mapping[str, Any]], path["steps"]):
        for name in ("logits_id", "ordinary_logits_id"):
            if step[name] is not None:
                result.add(cast(str, step[name]))
        for positions in cast(Mapping[str, Mapping[str, str]], step["op_ids"]).values():
            if isinstance(positions, Mapping):
                result.update(positions.values())
            else:
                result.add(cast(str, positions))
        for layer in cast(Mapping[str, Mapping[str, str]], step["prefix_ids"]).values():
            result.update(layer.values())
    full = path.get("full_logits_id")
    if full is not None:
        result.add(cast(str, full))
    return result


def _high_and_formula(
    workers: Mapping[str, _Worker], sink: _RecordSink, axis: _Axis,
) -> None:
    expected_cases = high_cases()
    candidate_values = _list(workers["candidate"].report["high_cases"], label="candidate high cases")
    formula_values = _list(workers["formula"].report["high_cases"], label="formula high cases")
    if len(candidate_values) != len(expected_cases) or len(formula_values) != len(expected_cases):
        _fail("D-high coverage differs from the frozen eight cases")
    if workers["formula"].report["cases"] != []:
        _fail("formula worker cannot claim public sampling cases")
    for index, expected in enumerate(expected_cases):
        case_id = cast(str, expected["case_id"])
        candidate = _mapping(
            candidate_values[index], fields={"case_id", "mode", "input_token_ids", "positions", "paths"},
            label="candidate high case",
        )
        formula = _mapping(
            formula_values[index], fields={"case_id", "mode", "input_token_ids", "positions", "paths"},
            label="formula high case",
        )
        base = {
            "case_id": case_id,
            "mode": expected["mode"],
            "input_token_ids": list(cast(list[int], expected["tokens"])),
            "positions": list(cast(list[int], expected["positions"])),
        }
        if not _json_equal({key: candidate[key] for key in base}, base) or not _json_equal(
            {key: formula[key] for key in base}, base
        ):
            _fail("D-high case inputs differ from the frozen protocol")
        candidate_paths = _mapping(candidate["paths"], fields={"B-high", "C-high"}, label="candidate high paths")
        if expected["mode"] == "rope_only":
            _validate_high_rope(workers, sink, axis, case_id=case_id,
                                candidate_paths=candidate_paths, formula_paths=formula["paths"])
        else:
            _validate_high_forward(
                workers, sink, axis, case_id=case_id, positions=cast(list[int], expected["positions"]),
                tokens=cast(list[int], expected["tokens"]),
                candidate_paths=candidate_paths, formula_paths=formula["paths"],
            )


def _validate_high_forward(
    workers: Mapping[str, _Worker], sink: _RecordSink, axis: _Axis, *, case_id: str,
    positions: list[int], tokens: list[int], candidate_paths: Mapping[str, Any],
    formula_paths: object,
) -> None:
    b = _validate_path_base(candidate_paths["B-high"], "B-high", high=True)
    c = _validate_path_base(candidate_paths["C-high"], "C-high", high=True)
    b_steps = _list(b["steps"], label="B-high steps")
    c_steps = _list(c["steps"], label="C-high steps")
    if len(b_steps) != 1 or len(c_steps) != len(positions):
        _fail("B-high/C-high step coverage differs")
    formula = _mapping(formula_paths, fields=set(canonical_tiny_ops()), label="formula forward paths")
    for position in positions:
        for op in canonical_tiny_ops():
            formula_positions = _mapping(
                formula[op], fields={f"{item:03d}" for item in positions}, label="formula op positions"
            )
            formula_id = tensor_id("D-high", case_id, 0, op, position)
            expected_source = workers["formula"].tensor(formula_positions[f"{position:03d}"], expected_id=formula_id)
            if expected_source.dtype != "fp64" or expected_source.shape != _candidate_shape(op, position):
                _fail("D-high formula tensor shape/dtype differs")
            _finite(expected_source, label="D-high formula")
            b_id, b_source, b_row = _high_candidate_row(workers["candidate"], b, "B-high", case_id, op, position)
            c_id, c_source, c_row = _high_candidate_row(workers["candidate"], c, "C-high", case_id, op, position)
            for actual_id, actual_source, actual, label in (
                (b_id, b_source, b_row, "D-high-to-B-high"),
                (c_id, c_source, c_row, "D-high-to-C-high"),
            ):
                _compare(
                    sink, axis, axis_name=label, case_id=case_id, step=position, op=op,
                    position=position, expected_worker=workers["formula"], expected_id=formula_id,
                    expected_source=expected_source, expected=expected_source,
                    actual_worker=workers["candidate"], actual_id=actual_id,
                    actual_source=actual_source, actual=actual, cross_precision=True,
                    expected_direction="D-high-FP64-expected",
                )
            _compare(
                sink, axis, axis_name="B-high-to-C-high", case_id=case_id, step=position,
                op=op, position=position, expected_worker=workers["candidate"], expected_id=b_id,
                expected_source=b_source, expected=b_row, actual_worker=workers["candidate"],
                actual_id=c_id, actual_source=c_source, actual=c_row,
                expected_direction="B-high-expected",
            )
    _validate_high_path_semantics(workers["candidate"], b, "B-high", case_id, positions, tokens)
    _validate_high_path_semantics(workers["candidate"], c, "C-high", case_id, positions, tokens)
    c_steps = cast(list[Mapping[str, Any]], c["steps"])
    for step in range(1, len(positions)):
        prefix_ids = cast(Mapping[str, Mapping[str, str]], c_steps[step]["prefix_ids"])
        for layer in range(2):
            for kind, suffix in (("key", "rope_k"), ("value", "v_projection")):
                prefix_id = prefix_ids[str(layer)][kind]
                prefix = workers["candidate"].tensor(prefix_id)
                for position in range(step):
                    b_id, b_source, b_row = _high_candidate_row(
                        workers["candidate"], b, "B-high", case_id,
                        f"layer.{layer}.{suffix}", position,
                    )
                    expected = EvidenceTensor(
                        "fp32", (1, 2, 1, 8), b_row.values, sha256_bytes(b_row.raw), b_row.raw
                    )
                    actual = slice_position(prefix, f"prefix_{kind}", position)
                    _compare(
                        sink, axis, axis_name="B-high-to-C-high-prefix", case_id=case_id,
                        step=step, op=f"layer.{layer}.prefix_{kind}", position=position,
                        expected_worker=workers["candidate"], expected_id=b_id,
                        expected_source=b_source, expected=expected,
                        actual_worker=workers["candidate"], actual_id=prefix_id,
                        actual_source=prefix, actual=actual, require_bitwise=True,
                        expected_direction="B-high-row-expected",
                    )


def _high_candidate_row(worker: _Worker, path: Mapping[str, Any], path_id: str, case_id: str,
                        op: str, position: int) -> tuple[str, EvidenceTensor, EvidenceTensor]:
    steps = cast(list[Mapping[str, Any]], path["steps"])
    step = steps[0] if path_id == "B-high" else steps[position]
    positions = cast(Mapping[str, str], cast(Mapping[str, Any], step["op_ids"])[op])
    identifier = positions[str(position)]
    expected_id = tensor_id(path_id, case_id, 0 if path_id == "B-high" else position, op, position)
    tensor = worker.tensor(identifier, expected_id=expected_id)
    if tensor.dtype != "fp32" or tensor.shape != _candidate_shape(op, position):
        _fail(f"{path_id} high tensor shape/dtype differs")
    _finite(tensor, label=f"{path_id} tensor")
    return identifier, tensor, tensor


def _validate_high_path_semantics(worker: _Worker, path: Mapping[str, Any], path_id: str,
                                  case_id: str, positions: list[int], tokens: list[int]) -> None:
    steps = cast(list[Mapping[str, Any]], path["steps"])
    _validate_flags(path["execution_flags"], label=f"{path_id} flags")
    _validate_zero_resources(path["resource_status"], label=f"{path_id} resources")
    if path["output_token_ids"] != [] or path["finish_reason"] != "NOT_APPLICABLE":
        _fail(f"{path_id} high forward cannot claim sampled output")
    for index, step in enumerate(steps):
        _mapping(step, fields=_STEP_FIELDS, label=f"{path_id} high step")
        expected_context = tokens if path_id == "B-high" else tokens[:index + 1]
        if step["step"] != index or not _json_equal(step["context"], expected_context):
            _fail(f"{path_id} high step coverage differs")
        if any(step[name] is not None for name in ("output_token_id", "state_before", "state_after", "ordinary_logits_id")):
            _fail(f"{path_id} high forward cannot claim sampling or an ordinary adapter")
        query = len(tokens) if path_id == "B-high" else 1
        if step["query_tokens"] != query:
            _fail(f"{path_id} high query length differs")
        if path_id == "C-high":
            if step["kv_before"] != index or step["kv_after"] != index + 1 or step["fence_complete"] is not True:
                _fail("C-high KV/fence accounting differs")
        elif step["kv_before"] != 0 or step["kv_after"] != 0 or step["fence_complete"] is not None:
            _fail("B-high cannot claim KV/fence state")
        expected_positions = positions if path_id == "B-high" else [positions[index]]
        op_ids = _mapping(step["op_ids"], fields=set(canonical_tiny_ops()), label=f"{path_id} high ops")
        for op in canonical_tiny_ops():
            position_ids = _mapping(
                op_ids[op], fields={str(position) for position in expected_positions},
                label=f"{path_id} high {op}",
            )
            for position in expected_positions:
                identifier = tensor_id(path_id, case_id, index if path_id == "C-high" else 0, op, position)
                tensor = worker.tensor(position_ids[str(position)], expected_id=identifier)
                if tensor.dtype != "fp32" or tensor.shape != _candidate_shape(op, position):
                    _fail(f"{path_id} high op shape/dtype differs")
                _finite(tensor, label=f"{path_id} high op")
        logits_identifier = tensor_id(path_id, case_id, index if path_id == "C-high" else 0,
                                      "full_logits", None)
        logits = worker.tensor(step["logits_id"], expected_id=logits_identifier)
        if logits.dtype != "fp32" or logits.shape != (1, len(expected_positions), 259):
            _fail(f"{path_id} per-step logits shape/dtype differs")
        prefix_ids = _mapping(step["prefix_ids"], fields=None, label=f"{path_id} high prefixes")
        if path_id == "C-high" and index > 0:
            if set(prefix_ids) != {"0", "1"}:
                _fail("C-high decode must expose both layer prefixes")
            for layer in range(2):
                item = _mapping(prefix_ids[str(layer)], fields={"key", "value"}, label="C-high prefix layer")
                for kind in ("key", "value"):
                    identifier = tensor_id(path_id, case_id, index, f"prefix.layer.{layer}.{kind}", None)
                    prefix = worker.tensor(item[kind], expected_id=identifier)
                    if prefix.dtype != "fp32" or prefix.shape != (1, 2, index, 8):
                        _fail("C-high prefix shape/dtype differs")
                    _finite(prefix, label="C-high prefix")
        elif prefix_ids:
            _fail(f"{path_id} cannot expose a prefix at this high step")
    full_id = _string(path["full_logits_id"], label=f"{path_id} full logits")
    expected_name = "full_logits" if path_id == "B-high" else "concatenated_logits"
    full = worker.tensor(full_id, expected_id=tensor_id(path_id, case_id, 0, expected_name, None))
    if full.dtype != "fp32" or full.shape != (1, len(positions), 259):
        _fail(f"{path_id} full logits shape differs")


def _validate_high_rope(
    workers: Mapping[str, _Worker], sink: _RecordSink, axis: _Axis, *, case_id: str,
    candidate_paths: Mapping[str, Any], formula_paths: object,
) -> None:
    formula = _mapping(
        formula_paths, fields={"input_q", "input_k", "rope_q", "rope_k"}, label="high-05 formula paths"
    )
    formula_tensors: dict[str, tuple[str, EvidenceTensor]] = {}
    for name, shape, dtype in (
        ("input_q", (1, 4, 2, 8), "fp32"), ("input_k", (1, 2, 2, 8), "fp32"),
        ("rope_q", (1, 4, 2, 8), "fp64"), ("rope_k", (1, 2, 2, 8), "fp64"),
    ):
        identifier = tensor_id("D-high", case_id, 0, name, None)
        tensor = workers["formula"].tensor(formula[name], expected_id=identifier)
        if tensor.shape != shape or tensor.dtype != dtype:
            _fail("high-05 formula tensor shape/dtype differs")
        _finite(tensor, label="high-05 formula tensor")
        formula_tensors[name] = (identifier, tensor)
    candidate_tensors: dict[str, dict[str, tuple[str, EvidenceTensor]]] = {}
    for path_id in ("B-high", "C-high"):
        path = _validate_path_base(candidate_paths[path_id], path_id, high=True)
        steps = _list(path["steps"], label=f"{path_id} rope steps")
        if len(steps) != 1:
            _fail("high-05 candidate path must contain one primitive step")
        step = _mapping(steps[0], fields=_STEP_FIELDS, label=f"{path_id} rope step")
        if (
            step["step"] != 0 or step["context"] != [] or step["query_tokens"] != 0
            or step["logits_id"] is not None or step["output_token_id"] is not None
            or step["state_before"] is not None or step["state_after"] is not None
            or step["prefix_ids"] != {} or step["ordinary_logits_id"] is not None
            or step["fence_complete"] is not None or path["full_logits_id"] is not None
            or path["output_token_ids"] != [] or path["finish_reason"] != "NOT_APPLICABLE"
        ):
            _fail("high-05 is a RoPE-only primitive, not a public request")
        op_ids = _mapping(step["op_ids"], fields={"input_q", "input_k", "rope_q", "rope_k"},
                          label=f"{path_id} rope ids")
        candidate_tensors[path_id] = {}
        for name, shape in (
            ("input_q", (1, 4, 2, 8)), ("input_k", (1, 2, 2, 8)),
            ("rope_q", (1, 4, 2, 8)), ("rope_k", (1, 2, 2, 8)),
        ):
            item = _mapping(op_ids[name], fields={"full"}, label=f"{path_id} {name}")
            identifier = tensor_id(path_id, case_id, 0, name, None)
            tensor = workers["candidate"].tensor(item["full"], expected_id=identifier)
            if tensor.dtype != "fp32" or tensor.shape != shape:
                _fail("high-05 candidate tensor shape/dtype differs")
            _finite(tensor, label="high-05 candidate tensor")
            candidate_tensors[path_id][name] = (identifier, tensor)
    for name in ("input_q", "input_k", "rope_q", "rope_k"):
        formula_id, expected = formula_tensors[name]
        for path_id in ("B-high", "C-high"):
            actual_id, actual = candidate_tensors[path_id][name]
            _compare(
                sink, axis, axis_name=f"D-high-to-{path_id}", case_id=case_id, step=0,
                op=name, position=None, expected_worker=workers["formula"], expected_id=formula_id,
                expected_source=expected, expected=expected, actual_worker=workers["candidate"],
                actual_id=actual_id, actual_source=actual, actual=actual,
                cross_precision=name.startswith("rope_"), require_bitwise=name.startswith("input_"),
                expected_direction="D-high-expected",
            )


def _authority_cases(
    root: Path, workers: Mapping[str, _Worker], sink: _RecordSink, axis: _Axis,
) -> dict[str, object]:
    oracle_path = _ordinary_file(root / "authorities" / "oracle-manifest.json", label="oracle authority")
    oracle = _mapping(strict_json(oracle_path.read_bytes(), label="oracle authority"), fields=None,
                      label="oracle authority")
    sources = _list(oracle.get("op_cases"), label="oracle op cases")
    reported = _list(workers["candidate"].report["op_authority_cases"], label="reported op authority cases")
    if len(sources) != 3 or len(reported) != 3:
        _fail("op authority must cover exactly the three frozen files")
    for worker_name in ("native", "reproduction"):
        if workers[worker_name].report["op_authority_cases"] != []:
            _fail(f"{worker_name} cannot claim candidate op-authority execution")
    source_summaries: list[dict[str, object]] = []
    runtime_assets = cast(
        Mapping[str, Any], cast(Mapping[str, Any], workers["native"].report["model"])["asset_files"]
    )
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, fields=None, label="op authority source")
        case_id = _string(source.get("fixture_id"), label="op authority fixture id")
        file_name = _string(source.get("file"), label="op authority filename")
        digest = _sha(source.get("sha256"), label="op authority digest")
        if Path(file_name).name != file_name:
            _fail("op authority filename is not a basename")
        if runtime_assets.get(f"oracle-v1/{file_name}") != digest:
            _fail("op authority bytes differ from the runtime worker asset identity")
        tensors = read_authority_tensors(root / "authorities" / file_name, digest)
        report = _mapping(
            reported[index],
            fields={"case_id", "input_token_ids", "positions", "mode", "source_file", "source_sha256", "paths"},
            label="reported op authority case",
        )
        tokens = _list(source.get("input_ids"), label="op authority input ids")
        position_ids = source.get("position_ids")
        rope_only = position_ids == [255]
        expected_positions = [255] if rope_only else list(range(len(tokens)))
        if (
            report["case_id"] != case_id or report["input_token_ids"] != tokens
            or report["positions"] != expected_positions
            or report["mode"] != ("rope_only" if rope_only else "forward")
            or report["source_file"] != f"fixtures/tiny-decoder/oracle-v1/{file_name}"
            or report["source_sha256"] != digest
        ):
            _fail("reported op authority metadata differs from frozen bytes")
        paths = _mapping(report["paths"], fields={"B-authority", "C-authority"}, label="authority paths")
        if rope_only:
            _authority_rope(workers["candidate"], sink, axis, case_id, tensors, paths)
        else:
            _authority_forward(workers["candidate"], sink, axis, case_id, tokens, tensors, paths)
        source_summaries.append({"case_id": case_id, "file": file_name, "sha256": digest})
    return {"status": axis.report()["status"], "op_cases": source_summaries}


def _authority_forward(worker: _Worker, sink: _RecordSink, axis: _Axis, case_id: str,
                       tokens: list[Any], tensors: Mapping[str, EvidenceTensor],
                       paths: Mapping[str, Any]) -> None:
    validated: dict[str, Mapping[str, Any]] = {}
    for path_id in ("B-authority", "C-authority"):
        path = _validate_path_base(paths[path_id], path_id)
        steps = _list(path["steps"], label=f"{path_id} steps")
        if len(steps) != 1:
            _fail("authority forward must contain one prefill step")
        step = _mapping(steps[0], fields=_STEP_FIELDS, label=f"{path_id} step")
        if (
            step["step"] != 0 or step["context"] != tokens or step["output_token_id"] is not None
            or step["state_before"] is not None or step["state_after"] is not None
            or step["query_tokens"] != len(tokens) or path["output_token_ids"] != []
            or path["finish_reason"] != "NOT_APPLICABLE"
        ):
            _fail("authority path is not a no-sampling prefill")
        if path_id == "C-authority":
            if step["kv_before"] != 0 or step["kv_after"] != len(tokens) or step["fence_complete"] is not True:
                _fail("C-authority KV/fence accounting differs")
        elif step["kv_before"] != 0 or step["kv_after"] != 0 or step["fence_complete"] is not None:
            _fail("B-authority cannot claim KV/fence state")
        op_ids = _mapping(step["op_ids"], fields=set(canonical_tiny_ops()), label=f"{path_id} ops")
        for op in canonical_tiny_ops():
            ids = _mapping(op_ids[op], fields={str(pos) for pos in range(len(tokens))}, label=f"{path_id} {op}")
            source = tensors[op]
            for position in range(len(tokens)):
                identifier = tensor_id(path_id, case_id, 0, op, position)
                actual = worker.tensor(ids[str(position)], expected_id=identifier)
                expected = slice_position(source, op, position)
                _compare(
                    sink, axis, axis_name=f"authority-to-{path_id}", case_id=case_id, step=0,
                    op=op, position=position, expected_worker=worker,
                    expected_id=f"authority:{case_id}:{op}", expected_source=source, expected=expected,
                    actual_worker=worker, actual_id=identifier, actual_source=actual, actual=actual,
                    expected_direction="frozen-original-op-authority-expected",
                )
        if step["prefix_ids"] != {} or step["ordinary_logits_id"] is not None:
            _fail("authority prefill cannot claim a prior prefix or ordinary adapter")
        logits_identifier = tensor_id(path_id, case_id, 0, "full_logits", None)
        full_logits = worker.tensor(step["logits_id"], expected_id=logits_identifier)
        rows = [worker.tensor(cast(Mapping[str, str], op_ids["logits"])[str(position)])
                for position in range(len(tokens))]
        raw = b"".join(row.raw for row in rows)
        assembled = EvidenceTensor(
            "fp32", (1, len(tokens), 259), tuple(value for row in rows for value in row.values),
            sha256_bytes(raw), raw,
        )
        if compare_evidence(assembled, full_logits, atol=0.0, rtol=0.0)["bitwise_equal"] is not True:
            _fail("authority full logits differ from emitted logits rows")
        validated[path_id] = path
    # B/C equality is already implied by both comparing to authority, but this
    # explicit record prevents a future report from relabeling either path.
    if set(validated) != {"B-authority", "C-authority"}:
        _fail("authority path validation did not cover both candidate modes")


def _authority_rope(worker: _Worker, sink: _RecordSink, axis: _Axis, case_id: str,
                    tensors: Mapping[str, EvidenceTensor], paths: Mapping[str, Any]) -> None:
    names = {f"layer.{layer}.{suffix}" for layer in range(2)
             for suffix in ("q_projection", "k_projection", "rope_q", "rope_k")}
    for path_id in ("B-authority", "C-authority"):
        path = _validate_path_base(paths[path_id], path_id)
        steps = _list(path["steps"], label=f"{path_id} rope steps")
        if len(steps) != 1:
            _fail("position-255 authority must contain one primitive step")
        step = _mapping(steps[0], fields=_STEP_FIELDS, label=f"{path_id} rope step")
        if (
            step["context"] != [258] or step["query_tokens"] != 0 or step["logits_id"] is not None
            or step["output_token_id"] is not None or step["state_before"] is not None
            or step["state_after"] is not None or step["prefix_ids"] != {}
            or path["output_token_ids"] != [] or path["finish_reason"] != "NOT_APPLICABLE"
        ):
            _fail("position-255 authority was misrepresented as a public request")
        ids = _mapping(step["op_ids"], fields=names, label=f"{path_id} position-255 ops")
        for op in sorted(names):
            item = _mapping(ids[op], fields={"full"}, label=f"{path_id} {op}")
            identifier = tensor_id(path_id, case_id, 0, op, None)
            actual = worker.tensor(item["full"], expected_id=identifier)
            source = tensors[op]
            _compare(
                sink, axis, axis_name=f"authority-to-{path_id}", case_id=case_id, step=0,
                op=op, position=255, expected_worker=worker, expected_id=f"authority:{case_id}:{op}",
                expected_source=source, expected=source, actual_worker=worker, actual_id=identifier,
                actual_source=actual, actual=actual,
                expected_direction="frozen-position-255-op-authority-expected",
            )


def _validate_identity(value: object, *, label: str) -> dict[str, Any]:
    identity = _mapping(value, fields={"descriptor", "sha256"}, label=label)
    if identity["sha256"] != sha256_bytes(canonical_json_bytes(identity["descriptor"])):
        _fail(f"{label} digest differs from its descriptor")
    return identity


def _validate_mutations(worker: _Worker, axis: _Axis) -> None:
    records = _list(worker.report["mutations"], label="mutation records")
    if len(records) != len(FIXED_MUTATION_IDS):
        _fail("mutation worker does not cover all nine fixed mutations")
    expected_detectors: dict[str, tuple[str, str | None]] = {
        "transpose-square-q-weight": ("first_trace_divergence", "layer.0.q_projection"),
        "decode-position-plus-one": ("first_trace_divergence", "layer.0.rope_q"),
        "wrong-gqa-group": ("scores_and_context", "layer.0.attention_scores"),
        "future-key-visible": ("exception", None),
        "corrupt-committed-prefix-k": ("b_native_prefix_hard_gate", "layer.0.attention_scores"),
        "wrong-policy-digest": ("exception", None),
        "unknown-policy-field": ("exception", None),
        "fused-bias": ("linear_conformance_guard", None),
        "early-storage-cast": ("linear_conformance_guard", None),
    }
    expected_pairs: dict[str, tuple[tuple[str, str], ...]] = {
        "transpose-square-q-weight": (("expected_first_tensor", "actual_first_tensor"),),
        "decode-position-plus-one": (("expected_first_tensor", "actual_first_tensor"),),
        "wrong-gqa-group": (("expected_scores", "actual_scores"), ("expected_context", "actual_context")),
        "corrupt-committed-prefix-k": (("b_scores", "c_scores"), ("native_scores", "c_scores")),
    }
    expected_path_labels: dict[str, set[str]] = {
        "transpose-square-q-weight": {
            "original_q_weight", "modified_q_weight", "expected_first_tensor", "actual_first_tensor",
        },
        "decode-position-plus-one": {"expected_first_tensor", "actual_first_tensor"},
        "wrong-gqa-group": {"expected_scores", "actual_scores", "expected_context", "actual_context"},
        "future-key-visible": {"query", "bad_key_prefix", "bad_value_prefix"},
        "corrupt-committed-prefix-k": {
            "original_committed_prefix_k", "modified_committed_prefix_k", "b_scores",
            "native_scores", "c_scores",
        },
        "wrong-policy-digest": set(),
        "unknown-policy-field": set(),
        "fused-bias": {"attempted_input", "attempted_weight", "attempted_bias"},
        "early-storage-cast": {"pre_cast_input", "pre_cast_weight"},
    }
    for index, mutation_id in enumerate(FIXED_MUTATION_IDS):
        record = _mapping(records[index], fields=_MUTATION_FIELDS, label="mutation record")
        if record["mutation_id"] != mutation_id or record["status"] != "detected":
            _fail("mutation order/status differs from the frozen protocol")
        _validate_identity(record["input_identity"], label=f"{mutation_id} input identity")
        _validate_identity(record["modified_input_identity"], label=f"{mutation_id} modified identity")
        detector = _mapping(record["expected_detector"], fields=None, label=f"{mutation_id} detector")
        observation = _mapping(record["observation"], fields=None, label=f"{mutation_id} observation")
        kind, name = expected_detectors[mutation_id]
        if detector.get("kind") != kind or observation.get("kind") != kind:
            _fail(f"{mutation_id} hit an unexpected detector kind")
        observed_name = observation.get("name", observation.get("first_name"))
        expected_name = detector.get("name", detector.get("first_name"))
        if name is not None and (observed_name != name or expected_name != name):
            _fail(f"{mutation_id} first divergent op differs")
        paths = _mapping(record["tensor_paths"], fields=None, label=f"{mutation_id} tensor paths")
        if set(paths) != expected_path_labels[mutation_id]:
            _fail(f"{mutation_id} raw tensor path inventory differs")
        for label, identifier_value in paths.items():
            identifier = tensor_id("D-mutation", mutation_id, 0, label, None)
            tensor = worker.tensor(identifier_value, expected_id=identifier)
            if tensor.dtype != "fp32":
                _fail(f"{mutation_id} pack contains a non-FP32 tensor")
            _finite(tensor, label=f"{mutation_id} tensor")
        detected = True
        for expected_label, actual_label in expected_pairs.get(mutation_id, ()):
            if expected_label not in paths or actual_label not in paths:
                _fail(f"{mutation_id} omits tensors needed to recompute its detector")
            expected = worker.tensor(paths[expected_label])
            actual = worker.tensor(paths[actual_label])
            result = compare_evidence(expected, actual, atol=_ATOL, rtol=_RTOL)
            if result["status"] != "fail":
                detected = False
        if mutation_id in {"transpose-square-q-weight", "decode-position-plus-one"}:
            if set(observation) != {"kind", "name", "position", "expected_tensor", "actual_tensor"}:
                _fail(f"{mutation_id} trace observation fields differ")
            _require_tensor_identity(
                observation["expected_tensor"], worker.tensor(paths["expected_first_tensor"]),
                label=f"{mutation_id} expected observation",
            )
            _require_tensor_identity(
                observation["actual_tensor"], worker.tensor(paths["actual_first_tensor"]),
                label=f"{mutation_id} actual observation",
            )
            if observation["position"] != detector.get("position"):
                _fail(f"{mutation_id} detector position differs from observation")
        elif mutation_id == "wrong-gqa-group":
            if set(observation) != {"kind", "first_name", "position", "scores_changed", "context_changed"}:
                _fail("wrong-gqa-group observation fields differ")
            detected = detected and observation["scores_changed"] is True and observation["context_changed"] is True
        elif mutation_id == "corrupt-committed-prefix-k":
            if set(observation) != {
                "kind", "name", "position", "b_native_within_tolerance",
                "candidate_native_within_tolerance",
            }:
                _fail("corrupt-prefix observation fields differ")
            control = compare_evidence(
                worker.tensor(paths["native_scores"]), worker.tensor(paths["b_scores"]),
                atol=_ATOL, rtol=_RTOL,
            )
            detected = (
                detected and control["status"] == "pass"
                and observation["b_native_within_tolerance"] is True
                and observation["candidate_native_within_tolerance"] is False
            )
        if mutation_id == "future-key-visible":
            if set(observation) != {"kind", "phase", "error"}:
                _fail("future-key-visible exception observation fields differ")
            key = worker.tensor(paths.get("bad_key_prefix"))
            value = worker.tensor(paths.get("bad_value_prefix"))
            query = worker.tensor(paths.get("query"))
            error = _mapping(
                observation.get("error"), fields={"type", "code", "field", "message"},
                label="future support error",
            )
            detected = (
                key.shape == (1, 2, 2, 8) and value.shape == (1, 2, 2, 8)
                and query.shape == (1, 4, 8) and detector.get("code") == "INVALID_ARGUMENT"
                and detector.get("phase") == "support-shape"
                and error.get("code") == "INVALID_ARGUMENT"
            )
        if mutation_id in {"wrong-policy-digest", "unknown-policy-field"}:
            if set(observation) != {"kind", "phase", "error"}:
                _fail(f"{mutation_id} exception observation fields differ")
            expected_code = "INVALID_NUMERICAL_POLICY" if mutation_id == "wrong-policy-digest" else "UNSUPPORTED_NUMERICAL_POLICY"
            error = _mapping(
                observation.get("error"), fields={"type", "code", "field", "message"},
                label=f"{mutation_id} error",
            )
            detected = (
                detector.get("code") == expected_code and detector.get("phase") == "pre-acceptance"
                and observation.get("phase") == "pre-acceptance" and error.get("code") == expected_code
                and paths == {}
            )
        if mutation_id in {"fused-bias", "early-storage-cast"}:
            if set(observation) != {
                "kind", "phase", "call_count_before_rejection", "attempted_input",
                "attempted_weight", "attempted_bias", "error", "attempted_bf16_raw",
            }:
                _fail(f"{mutation_id} linear guard observation fields differ")
            error = _mapping(
                observation.get("error"), fields={"type", "code", "field", "message"},
                label=f"{mutation_id} error",
            )
            detected = (
                observation.get("phase") == "first-real-F.linear-call"
                and error.get("type") == "AssertionError"
                and _integer(observation.get("call_count_before_rejection"), label="linear guard count", minimum=1) >= 1
            )
            if mutation_id == "fused-bias":
                if observation["attempted_bf16_raw"] is not None:
                    _fail("fused-bias cannot claim BF16 raw inputs")
                for label in ("attempted_input", "attempted_weight", "attempted_bias"):
                    _require_tensor_identity(
                        observation[label], worker.tensor(paths[label]), label=f"fused-bias {label}"
                    )
            else:
                if observation["attempted_bias"] is not None:
                    _fail("early-storage-cast cannot claim a fused bias")
                bf16 = _mapping(
                    observation["attempted_bf16_raw"], fields={"input", "weight"},
                    label="early-storage-cast BF16 raw",
                )
                for label in ("input", "weight"):
                    _validate_inline_bf16(bf16[label], label=f"early-storage-cast {label}")
                _require_inline_identity(
                    observation["attempted_input"], bf16["input"],
                    label="early-storage-cast attempted input",
                )
                _require_inline_identity(
                    observation["attempted_weight"], bf16["weight"],
                    label="early-storage-cast attempted weight",
                )
        _validate_mutation_fence(record["resource_fence"], mutation_id=mutation_id)
        axis.comparisons += 1
        if not detected:
            axis.failures += 1


def _require_tensor_identity(value: object, tensor: EvidenceTensor, *, label: str) -> None:
    identity = _mapping(value, fields={"dtype", "shape", "raw_sha256"}, label=label)
    expected_dtype = "torch.float32" if tensor.dtype == "fp32" else "torch.float64"
    if (
        identity["dtype"] != expected_dtype
        or identity["shape"] != list(tensor.shape)
        or identity["raw_sha256"] != tensor.raw_sha256
    ):
        _fail(f"{label} does not bind its raw tensor")


def _validate_inline_bf16(value: object, *, label: str) -> None:
    item = _mapping(
        value, fields={"dtype", "shape", "nbytes", "sha256", "raw_hex"}, label=label
    )
    shape = _list(item["shape"], label=f"{label} shape")
    count = 1
    for dimension in shape:
        count *= _integer(dimension, label=f"{label} dimension", minimum=1)
    raw_hex = _string(item["raw_hex"], label=f"{label} raw hex")
    if re.fullmatch(r"[0-9a-f]*", raw_hex) is None or len(raw_hex) % 2:
        _fail(f"{label} raw hex is not complete lowercase hexadecimal")
    raw = bytes.fromhex(raw_hex)
    if (
        item["dtype"] != "torch.bfloat16"
        or item["nbytes"] != count * 2
        or len(raw) != count * 2
        or item["sha256"] != sha256_bytes(raw)
    ):
        _fail(f"{label} BF16 raw identity differs")


def _require_inline_identity(value: object, inline: object, *, label: str) -> None:
    identity = _mapping(value, fields={"dtype", "shape", "raw_sha256"}, label=label)
    item = cast(Mapping[str, Any], inline)
    if (
        identity["dtype"] != item["dtype"]
        or identity["shape"] != item["shape"]
        or identity["raw_sha256"] != item["sha256"]
    ):
        _fail(f"{label} does not bind the inline BF16 raw bytes")


def _validate_mutation_fence(value: object, *, mutation_id: str) -> None:
    fence = _mapping(value, fields=_FENCE_FIELDS, label=f"{mutation_id} resource fence")
    required = mutation_id in {"decode-position-plus-one", "corrupt-committed-prefix-k"}
    if fence["required"] is not required:
        _fail(f"{mutation_id} fence-required flag differs")
    if required:
        if (
            fence["fence_kind"] != "ImmediateFence" or fence["fence_complete"] is not True
            or fence["fence_succeeded"] is not True or fence["active_leases_after_cleanup"] != 0
            or fence["store_closed"] is not True
        ):
            _fail(f"{mutation_id} did not prove fence completion and cleanup")
    elif any(fence[name] is not None for name in _FENCE_FIELDS - {"required"}):
        _fail(f"{mutation_id} claims a resource fence that was not required")


def _validate_coverage(worker: _Worker, expected_cases: Sequence[Mapping[str, Any]]) -> None:
    coverage = _mapping(worker.report["coverage"], fields=None, label=f"{worker.role} coverage")
    if worker.role in {"native", "candidate", "candidate-reproduction"}:
        required = {
            "kind", "case_count", "expected_case_count", "suite_counts", "high_case_count",
            "expected_high_case_count", "full_contract",
        }
        if set(coverage) != required:
            _fail(f"{worker.role} coverage fields differ")
        expected = 20 if worker.role == "candidate-reproduction" else len(expected_cases)
        selected = (
            [case for case in expected_cases if case["expected_output_token_ids"] is not None]
            if worker.role == "candidate-reproduction" else list(expected_cases)
        )
        suite_counts: dict[str, int] = {}
        for case in selected:
            suite = cast(str, case["suite_id"])
            suite_counts[suite] = suite_counts.get(suite, 0) + 1
        if (
            coverage.get("kind") != ("reproduction20" if worker.role == "candidate-reproduction" else "full70")
            or coverage.get("case_count") != expected
            or coverage.get("expected_case_count") != expected
            or coverage.get("full_contract") is not True
            or coverage.get("high_case_count") != (8 if worker.role == "candidate" else 0)
            or coverage.get("expected_high_case_count") != (8 if worker.role == "candidate" else 0)
            or coverage.get("suite_counts") != dict(sorted(suite_counts.items()))
        ):
            _fail(f"{worker.role} coverage claim differs from the required matrix")
    elif worker.role == "formula":
        required = {
            "protocol_section", "full_forward_cases", "rope_only_cases", "canonical_op_count",
            "tensor_count", "pack_path", "gpu_execution", "claims", "limitations",
        }
        if (
            set(coverage) != required or coverage.get("protocol_section") != "4.3"
            or coverage.get("pack_path") != "pack" or coverage.get("full_forward_cases") != 7
            or coverage.get("rope_only_cases") != 1 or coverage.get("canonical_op_count") != 39
            or coverage.get("tensor_count") != len(worker.pack.records)
            or coverage.get("gpu_execution") != "not_run_not_authorized"
        ):
            _fail("formula coverage fields differ")
    else:
        required = {
            "protocol_section", "mutation_count", "mutation_ids", "all_mutations_detected",
            "pack_path", "gpu_execution", "claims", "limitations",
        }
        if (
            set(coverage) != required or coverage.get("protocol_section") != "4.4"
            or coverage.get("mutation_count") != 9 or coverage.get("mutation_ids") != list(FIXED_MUTATION_IDS)
            or coverage.get("all_mutations_detected") is not True or coverage.get("pack_path") != "pack"
            or coverage.get("gpu_execution") != "not_run_not_authorized"
        ):
            _fail("mutation coverage fields differ")


def derive_cpu_numerical(
    root: Path,
    *,
    records_path: Path | None = None,
    verify_records_path: Path | None = None,
) -> dict[str, object]:
    """Derive a closed CPU numerical report from five raw worker bundles.

    ``records_path`` exclusively writes the canonical comparison stream.
    ``verify_records_path`` instead requires byte-for-byte equality with an
    existing stream.  Omitting both still computes the same digest and count.
    Structural corruption raises :class:`NumericalEvidenceError`; genuine
    numerical mismatches are represented as failing report axes.
    """

    evidence_root = _ordinary_directory(Path(root), label="CPU numerical evidence root")
    authorities = _ordinary_directory(evidence_root / "authorities", label="CPU authority directory")
    inputs_raw = _ordinary_file(authorities / "inputs.json", label="approved CPU inputs").read_bytes()
    oracle_raw = _ordinary_file(authorities / "oracle-manifest.json", label="tiny oracle authority").read_bytes()
    expected_cases = cpu_cases_from_snapshots(inputs_raw, oracle_raw)
    if len(expected_cases) != 70:
        _fail("approved CPU case authority must contain exactly 50+20 cases")

    workers = {
        directory: _read_worker(evidence_root, directory, role)
        for directory, role in _WORKER_DIRECTORIES.items()
    }
    bindings = {name: _validate_binding(worker) for name, worker in workers.items()}
    reference_binding = bindings["native"]
    if any(not _json_equal(binding, reference_binding) for binding in bindings.values()):
        _fail("workers do not share one source/environment/toolchain/protocol binding")
    pids = [_validate_process(worker) for worker in workers.values()]
    if len(set(pids)) != len(pids):
        _fail("worker roles did not execute in distinct process identities")
    _validate_models(workers)
    authority_asset_hashes = _validate_authority_assets(evidence_root, workers, oracle_raw)
    _validate_policies(workers)
    for worker in workers.values():
        _validate_coverage(worker, expected_cases)

    axes = {
        "candidate": _Axis(),
        "native": _Axis(),
        "legacy": _Axis(),
        "sequence": _Axis(),
        "high": _Axis(),
        "mutation": _Axis(),
        "reproduction": _Axis(),
        "authority": _Axis(),
    }
    sink = _RecordSink(output=records_path, verify=verify_records_path)
    try:
        main_cases = _runtime_cases(workers, expected_cases, sink, axes)
        _validate_reproduction(workers, expected_cases, main_cases, axes["reproduction"])
        _high_and_formula(workers, sink, axes["high"])
        authority = _authority_cases(evidence_root, workers, sink, axes["authority"])
        _validate_mutations(workers["mutations"], axes["mutation"])
        for worker in workers.values():
            worker.require_complete_inventory()
        records = sink.finish()
    except BaseException:
        sink.abort()
        raise

    axis_reports = {name: axis.report() for name, axis in axes.items()}
    hard_names = ("candidate", "native", "sequence", "high", "mutation", "reproduction", "authority")
    cpu_hard = "pass" if all(axis_reports[name]["status"] == "pass" for name in hard_names) else "fail"
    cpu_results: dict[str, object] = {
        "hard_gate_result": cpu_hard,
        "legacy_elementwise_compatibility": axis_reports["legacy"]["status"],
        "sequence_compatibility": axis_reports["sequence"]["status"],
        "candidate": axis_reports["candidate"],
        "native": axis_reports["native"],
        "high": axis_reports["high"],
        "mutation": axis_reports["mutation"],
        "reproduction": axis_reports["reproduction"],
        "authority": axis_reports["authority"],
    }
    worker_summary = {
        name: {
            "role": worker.role,
            "worker_json_sha256": worker.report_sha256,
            "pack_manifest_digest": worker.pack_manifest_digest,
            "pid": worker.report["process"]["pid"],
        }
        for name, worker in workers.items()
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": NUMERICAL_REPORT_KIND,
        "scope": PUBLIC_SCOPE,
        "evidence_stage": "DEVELOPMENT_ONLY",
        "qualification_status": "unqualified",
        "status": "incomplete" if cpu_hard == "pass" else "fail",
        "hard_gate_result": "incomplete",
        "legacy_elementwise_compatibility": "incomplete",
        "sequence_compatibility": "incomplete",
        "cpu_results": cpu_results,
        "authority": {
            "inputs_sha256": sha256_bytes(inputs_raw),
            "oracle_manifest_digest": TINY_ORACLE_MANIFEST_DIGEST,
            "case_count": len(expected_cases),
            "protocol_sha256": PROTOCOL_SHA256,
            "asset_sha256": authority_asset_hashes,
            "op_authority": authority,
        },
        "workers": worker_summary,
        "comparison_records": records,
    }
    # This is an internal consistency report.  The source-bound outer bundle,
    # not this checksum, establishes which frozen source produced it.
    return cast(dict[str, object], strict_json(canonical_json_bytes(report), label="CPU numerical report"))


__all__ = ["NumericalEvidenceError", "derive_cpu_numerical"]

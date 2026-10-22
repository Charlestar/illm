"""Input-only CPU numerical evidence contracts; no tensor runtime imports.

These helpers name the already approved public cases. They neither generate
goldens nor grant numerical, GPU, or release qualification.
"""

from __future__ import annotations

import json
import os
import re
import stat
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_json
from illm.numerical_contracts import FIXED_PROMPTS
from illm.policy_inputs import (
    APPROVED_INPUT_BYTES_SHA256,
    INPUT_MANIFEST_DIGEST,
    TINY_SUITE_ID,
    parse_policy_inputs_json,
)
from illm.sampling import SamplingSpec

if TYPE_CHECKING:
    from illm.policy_tensor_evidence import EvidenceTensor


PROTOCOL_SHA256 = "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"
TINY_ORACLE_MANIFEST_DIGEST = "d424fe676820b748bf51023ca15150211d3414371314d63ce3fa647c4264fadc"
TINY_ORACLE_BYTES_SHA256 = "186931d05be0bc0cd0c667240c103f017503962b86a6221d78fe3b9aae9a2298"
TINY_SPEC_ID = "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873"
TINY_WEIGHTS_DIGEST = "3868a9840d0b7e418e26c6a31d5022fd829e68090fe75d617964eee7ea09ac92"
TINY_LOAD_MANIFEST_DIGEST = "dd0827b9ed7e4b1e2808e30552602a38cdda4666bf8b1417482b69ce1e5840bd"
TINY_TOKENIZER_DIGEST = "f8972f8627f9f6af04431b14a6cedc02cba870e9494d76f72d3df82ccb161616"
V01_REFERENCE_SHA256 = "19fb396156e5739a3ddd6783a2fbbc610528fff8ed099abe6e0a17a5c63b44bf"
WORKER_REPORT_KIND = "illm-policy-cpu-raw-worker-v1"
NUMERICAL_REPORT_KIND = "illm-stateful-numerical-policy-validation-v1"
PUBLIC_SCOPE = "approved-tiny-fp32-cpu-only"
FIXED_MUTATION_IDS = (
    "transpose-square-q-weight", "decode-position-plus-one", "wrong-gqa-group",
    "future-key-visible", "corrupt-committed-prefix-k", "wrong-policy-digest",
    "unknown-policy-field", "fused-bias", "early-storage-cast",
)
_SAFE_ID_PART = re.compile(r"[A-Za-z0-9_.-]+")


class NumericalEvidenceError(ValueError):
    """Malformed, inconsistent, or incomplete CPU numerical evidence."""


def strict_json(raw: str | bytes, *, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise NumericalEvidenceError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value: str) -> NoReturn:
        raise NumericalEvidenceError(f"{label} contains non-finite JSON: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        # Also rejects an exponent such as 1e999, which json.loads turns into Inf.
        canonical_json_bytes(result)
    except NumericalEvidenceError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        raise NumericalEvidenceError(f"{label} is not finite JSON") from exc
    return result


def plain_path(path: Path) -> Path:
    """Reject existing links before normalizing a caller-visible output path."""
    requested = Path(os.path.abspath(os.fspath(path)))
    for item in (requested, *requested.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NumericalEvidenceError("evidence path is unreadable") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise NumericalEvidenceError("evidence path cannot contain a link or reparse point")
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise NumericalEvidenceError("evidence file cannot have a hard-link alias")
    return requested


def read_json(path: Path) -> Any:
    requested = plain_path(path)
    if not requested.is_file():
        raise NumericalEvidenceError("evidence JSON must be a plain file")
    return strict_json(requested.read_bytes(), label=requested.name)


def write_json_exclusive(path: Path, value: Any) -> None:
    requested = plain_path(path)
    with requested.open("xb") as output:
        output.write(canonical_json_bytes(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())


def cpu_cases_from_snapshots(inputs_raw: bytes, oracle_raw: bytes) -> list[dict[str, Any]]:
    if sha256_bytes(inputs_raw) != APPROVED_INPUT_BYTES_SHA256:
        raise NumericalEvidenceError("CPU input snapshot differs from the human-approved bytes")
    if sha256_bytes(oracle_raw) != TINY_ORACLE_BYTES_SHA256:
        raise NumericalEvidenceError("CPU tiny oracle differs from the frozen original bytes")
    proposal = parse_policy_inputs_json(inputs_raw)
    if proposal.manifest_digest != INPUT_MANIFEST_DIGEST:
        raise NumericalEvidenceError("CPU input proposal identity differs")
    proposal_value = proposal.to_dict()
    suites = cast(list[dict[str, Any]], proposal_value["suites"])
    tiny = next(suite for suite in suites if suite["suite_id"] == TINY_SUITE_ID)

    oracle = strict_json(oracle_raw, label="frozen tiny oracle manifest")
    if not isinstance(oracle, dict):
        raise NumericalEvidenceError("tiny oracle manifest must be an object")
    claimed = oracle.get("manifest_digest")
    if (claimed != TINY_ORACLE_MANIFEST_DIGEST
            or claimed != sha256_json({key: value for key, value in oracle.items() if key != "manifest_digest"})):
        raise NumericalEvidenceError("tiny oracle authority cannot be regenerated or re-signed")
    for field, expected in (
        ("spec_id", TINY_SPEC_ID), ("weights_digest", TINY_WEIGHTS_DIGEST),
        ("load_manifest_digest", TINY_LOAD_MANIFEST_DIGEST),
        ("tokenizer_revision", f"sha256:{TINY_TOKENIZER_DIGEST}"),
    ):
        if oracle.get(field) != expected:
            raise NumericalEvidenceError(f"tiny oracle {field} differs")

    result: list[dict[str, Any]] = []
    for source in cast(list[dict[str, Any]], tiny["cases"]):
        case = {**source, "suite_id": TINY_SUITE_ID, "expected_output_token_ids": None}
        case["case_digest"] = sha256_json(case)
        result.append(case)
    greedy = oracle.get("greedy_cases")
    if not isinstance(greedy, list) or len(greedy) != 20:
        raise NumericalEvidenceError("tiny20 authority must contain exactly twenty ordered cases")
    sampling = SamplingSpec(mode="greedy", actual_seed=7)
    for index, source in enumerate(greedy):
        if (not isinstance(source, dict) or source.get("fixture_id") != f"fixed-greedy-{index:02d}"
                or source.get("public_prompt") != FIXED_PROMPTS[index]):
            raise NumericalEvidenceError("tiny20 authority case order differs")
        tokens, outputs = source["input_ids"], source["expected_tokens"]
        if (not isinstance(tokens, list) or not tokens or not isinstance(outputs, list) or len(outputs) != 8
                or any(type(token) is not int or not 0 <= token < 259 for token in (*tokens, *outputs))):
            raise NumericalEvidenceError("tiny20 authority token vectors are malformed")
        case = {
            "case_id": source["fixture_id"], "input_token_ids": list(tokens), "max_new_tokens": 8,
            "sampling_spec": sampling.to_dict(),
            "input_digest": sha256_json({"schema_version": 1, "input_token_ids": tokens}),
            "sampling_spec_digest": sampling.digest, "suite_id": "v01-tiny-greedy20",
            "expected_output_token_ids": list(outputs),
        }
        case["case_digest"] = sha256_json(case)
        result.append(case)
    return result


def cpu_cases(project_root: Path) -> list[dict[str, Any]]:
    return cpu_cases_from_snapshots(
        plain_path(project_root / "docs/implementation/v0.2-policy-inputs-draft.json").read_bytes(),
        plain_path(project_root / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_bytes(),
    )


def high_cases() -> list[dict[str, Any]]:
    inputs: list[list[int]] = [
        [3], [4, 5], [2], [0] * 7, [1, 2, 258, 3, 257, 4, 256], [],
        [3 + ((43 * 6 + 17 * p) % 256) for p in range(31)],
        [3 + ((43 * 7 + 17 * p) % 256) for p in range(255)],
    ]
    return [{"case_id": f"high-{index:02d}", "mode": "rope_only" if index == 5 else "forward",
             "tokens": tokens, "positions": [31, 32] if index == 5 else list(range(len(tokens)))}
            for index, tokens in enumerate(inputs)]


def canonical_tiny_ops() -> tuple[str, ...]:
    suffixes = (
        "input_norm", "q_projection", "k_projection", "v_projection", "rope_q", "rope_k",
        "attention_scores", "attention_probabilities", "attention_context", "attention_output",
        "attention_residual", "post_attention_norm", "mlp_gate", "mlp_up", "mlp_silu",
        "mlp_product", "mlp_output", "mlp_residual",
    )
    return ("embedding", *(f"layer.{layer}.{name}" for layer in range(2) for name in suffixes),
            "final_norm", "logits")


def tensor_id(role: str, case_id: str, step: int, op: str, position: int | None) -> str:
    if any(type(value) is not str or _SAFE_ID_PART.fullmatch(value) is None for value in (role, case_id, op)):
        raise NumericalEvidenceError("tensor semantic identity contains an invalid component")
    if type(step) is not int or step < 0 or (position is not None and (type(position) is not int or position < 0)):
        raise NumericalEvidenceError("tensor semantic position must be a nonnegative integer")
    row = "full" if position is None else f"{position:03d}"
    return f"{role}/{case_id}/{step:03d}/{op}/{row}"


def read_authority_tensors(path: Path, expected_sha256: str) -> dict[str, EvidenceTensor]:
    """Read the immutable public F32 op goldens without importing safetensors.

    This is not a general safetensors loader. It accepts only the existing
    checksummed tiny-op authority, preserving its original bytes and -Inf future
    score sentinels. Numerical comparisons must still inspect only legal prefix.
    """
    from illm.policy_tensor_evidence import EvidenceTensor

    raw = plain_path(path).read_bytes()
    if sha256_bytes(raw) != expected_sha256 or len(raw) < 10:
        raise NumericalEvidenceError("frozen op authority bytes differ")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length < 2 or header_length > len(raw) - 8:
        raise NumericalEvidenceError("frozen op authority header is truncated")
    header = strict_json(raw[8:8 + header_length], label="frozen F32 tensor header")
    if not isinstance(header, dict):
        raise NumericalEvidenceError("frozen tensor header must be an object")
    data = raw[8 + header_length:]
    ranges: list[tuple[int, int]] = []
    result: dict[str, EvidenceTensor] = {}
    for name, description in header.items():
        if name == "__metadata__":
            if (not isinstance(description, dict)
                    or any(type(key) is not str or type(value) is not str for key, value in description.items())):
                raise NumericalEvidenceError("frozen tensor metadata must contain strings")
            continue
        if (type(name) is not str or not name or not isinstance(description, dict)
                or set(description) != {"dtype", "shape", "data_offsets"} or description["dtype"] != "F32"):
            raise NumericalEvidenceError("frozen op authority contains an unsupported storage")
        shape, offsets = description["shape"], description["data_offsets"]
        if (type(shape) is not list or not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(offsets) is not list or len(offsets) != 2
                or any(type(value) is not int or value < 0 for value in offsets)):
            raise NumericalEvidenceError("frozen op authority shape or offsets are malformed")
        start, end = offsets
        count = 1
        for dim in shape:
            count *= dim
        if start > end or end > len(data) or end - start != count * 4:
            raise NumericalEvidenceError("frozen op authority storage bounds disagree with shape")
        storage = data[start:end]
        values = tuple(value[0] for value in struct.iter_unpack("<f", storage))
        result[name] = EvidenceTensor("fp32", tuple(shape), values, sha256_bytes(storage), storage)
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise NumericalEvidenceError("frozen op authority has overlapping or unused storage")
        cursor = end
    if cursor != len(data) or set(result) != set(canonical_tiny_ops()):
        raise NumericalEvidenceError("frozen op authority does not cover the exact canonical ops")
    return result

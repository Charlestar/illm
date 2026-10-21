"""Portable validation for the approved V0.2 input-only proposal.

This module has no tensor, tokenizer, model-loader, or network dependency.  It
validates the frozen source identities, exact public inputs, deterministic tiny
formula, sampling contracts, and every nested canonical digest.  No generated
tokens or candidate outputs are accepted in this input-only schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, cast

from illm.canonical import canonical_json_bytes, sha256_bytes
from illm.numerical_contracts import FIXED_PROMPTS


PROTOCOL_ID = "v02-numerical-policy-protocol-draft-1"
DOCUMENT_KIND = "illm-policy-input-proposal-v1"
DOCUMENT_STATUS = "proposed-not-runtime-authority"
APPROVED_INPUT_BYTES_SHA256 = "93d0de1c5c606a58830eeef2d6b95f4a2b389b4defd17735507d2e95b9adb754"
INPUT_MANIFEST_DIGEST = "a5362433ff4f870cd4c0f1df1b5330ee42ace6be9e934ec159749e642b5bb36e"
REAL_SUITE_ID = "v02-policy-real-fixed20-inputs-v1"
REAL_SUITE_DIGEST = "57163407317f5e9f9f0cc26931c364718b299a79e3ec77cb1c7da4bac52a6f7a"
TINY_SUITE_ID = "v02-policy-tiny50-inputs-v1"
TINY_SUITE_DIGEST = "659337351c7fd93ce2ca101a76908bf537db586ded85e89ac41e45072e97f9cb"
GREEDY_SAMPLING_DIGEST = "d39d74929204c7df8a62c21a4520da2bc30cd10e26cf1f0e56f526d61d3cfc06"
SAMPLE_SAMPLING_DIGEST = "3b5576ca2a41c973073d99b4733355c6eb1f1ff4cdc5c9bc047fdc8456e67d40"

_TOP_FIELDS = frozenset(
    {"schema_version", "document_kind", "status", "protocol_id", "suites", "manifest_digest"}
)
_REAL_SUITE_FIELDS = frozenset(
    {
        "suite_id",
        "model_revision",
        "tokenizer_revision",
        "tokenizer_digest",
        "weights_digest",
        "chat_template",
        "add_bos",
        "cases",
        "suite_digest",
    }
)
_TINY_SUITE_FIELDS = frozenset(
    {
        "suite_id",
        "model_spec_id",
        "tokenizer_digest",
        "weights_digest",
        "capacity_tokens",
        "cases",
        "suite_digest",
    }
)
_REAL_CASE_FIELDS = frozenset(
    {
        "case_id",
        "public_prompt",
        "input_token_ids",
        "max_new_tokens",
        "sampling_spec",
        "input_digest",
        "sampling_spec_digest",
    }
)
_TINY_CASE_FIELDS = frozenset(
    {
        "case_id",
        "input_token_ids",
        "max_new_tokens",
        "sampling_spec",
        "input_digest",
        "sampling_spec_digest",
    }
)
_SAMPLING_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "actual_seed",
        "compute_dtype",
        "processor_order_version",
        "rng_algorithm",
        "rng_contract_version",
        "temperature",
        "top_k",
        "top_p",
    }
)

_REAL_SOURCE_AUTHORITY: Mapping[str, object] = {
    "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "tokenizer_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "tokenizer_digest": "e14f0b0445b5fa49758860c20a7ad12200f65a1f23d144b6ac0afbab16fc765b",
    "weights_digest": "477045d85bedba0a9bb31d42995212339f566729636c2881021de52bda84a08a",
    "chat_template": None,
    "add_bos": False,
}
_TINY_SOURCE_AUTHORITY: Mapping[str, object] = {
    "model_spec_id": "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873",
    "tokenizer_digest": "f8972f8627f9f6af04431b14a6cedc02cba870e9494d76f72d3df82ccb161616",
    "weights_digest": "3868a9840d0b7e418e26c6a31d5022fd829e68090fe75d617964eee7ea09ac92",
    "capacity_tokens": 256,
}

_REAL_INPUT_IDS: tuple[tuple[int, ...], ...] = (
    (14990,),
    (64,),
    (785, 6722, 315, 9625, 374),
    (16, 488, 220, 16, 284),
    (840, 20772, 264, 3950),
    (641, 2202),
    (82707, 6500),
    (48, 16948),
    (108386,),
    (104949,),
    (32, 425, 356),
    (15,),
    (16, 17, 18, 19, 20),
    (198,),
    (220,),
    (12669,),
    (750, 282, 2075, 1648),
    (12522, 5193, 264, 882),
    (31615, 323, 22670),
    (145691,),
)


class PolicyInputError(ValueError):
    """The input-only proposal is malformed or differs from its authority."""

    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field


def _fail(code: str, message: str, *, field: str | None = None) -> NoReturn:
    raise PolicyInputError(code, message, field=field)


def _sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _field_names(value: Mapping[object, object]) -> list[str]:
    keys = list(value.keys())
    if any(type(key) is not str for key in keys):
        _fail("INVALID_POLICY_INPUTS", "JSON field names must be strings")
    names = cast(list[str], keys)
    if len(names) != len(set(names)):
        _fail("INVALID_POLICY_INPUTS", "duplicate JSON field")
    return names


def _closed_fields(value: Mapping[object, object], expected: frozenset[str], what: str) -> None:
    names = set(_field_names(value))
    missing = sorted(expected - names)
    unknown = sorted(names - expected)
    if missing or unknown:
        _fail(
            "INVALID_POLICY_INPUTS",
            f"{what} has missing or unknown fields: missing={missing}, unknown={unknown}",
        )


def _same(actual: object, expected: object) -> bool:
    """Compare JSON values without allowing bool to masquerade as integer."""

    try:
        return canonical_json_bytes(actual) == canonical_json_bytes(expected)
    except (TypeError, ValueError):
        return False


def _greedy_sampling() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "greedy",
        "actual_seed": 7,
        "compute_dtype": "fp32",
        "processor_order_version": 1,
        "rng_algorithm": "philox4x32-10",
        "rng_contract_version": 1,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
    }


def _sample_sampling() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "sample",
        "actual_seed": 7,
        "compute_dtype": "fp32",
        "processor_order_version": 1,
        "rng_algorithm": "philox4x32-10",
        "rng_contract_version": 1,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    }


def _input_digest(tokens: list[int]) -> str:
    return _sha256_json({"schema_version": 1, "input_token_ids": tokens})


def _tiny_tokens(index: int, length: int) -> list[int]:
    return [3 + ((43 * index + 17 * position) % 256) for position in range(length)]


def _expected_tiny_case(index: int) -> tuple[list[int], int, dict[str, object]]:
    if index == 0:
        tokens, max_new_tokens = [3], 1
    elif index == 1:
        tokens, max_new_tokens = [4, 5], 2
    elif index == 2:
        tokens, max_new_tokens = _tiny_tokens(2, 255), 1
    elif index == 3:
        tokens, max_new_tokens = [2], 3
    else:
        max_new_tokens = 1 + index % 3
        maximum = min(48, 256 - max_new_tokens)
        length = 3 + ((11 * index) % (maximum - 2))
        tokens = _tiny_tokens(index, length)
    sampling = _greedy_sampling() if index < 25 else _sample_sampling()
    return tokens, max_new_tokens, sampling


def _validate_sampling(value: object, expected: dict[str, object], expected_digest: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail("INVALID_POLICY_INPUTS", "sampling_spec must be an object")
    _closed_fields(cast(Mapping[object, object], value), _SAMPLING_FIELDS, "sampling_spec")
    result = dict(cast(Mapping[str, object], value))
    if not _same(result, expected) or _sha256_json(result) != expected_digest:
        _fail("INVALID_POLICY_INPUTS", "sampling_spec differs from its registered contract")
    return result


def _validate_case(
    value: object,
    *,
    index: int,
    real: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail("INVALID_POLICY_INPUTS", "input case must be an object")
    fields = _REAL_CASE_FIELDS if real else _TINY_CASE_FIELDS
    _closed_fields(cast(Mapping[object, object], value), fields, "input case")
    case = dict(cast(Mapping[str, object], value))
    expected_id = f"fixed-greedy-{index:02d}" if real else f"differential-{index:02d}"
    if type(case.get("case_id")) is not str or case.get("case_id") != expected_id:
        _fail("INVALID_POLICY_INPUTS", f"input case order or ID differs at index {index}")

    tokens_value = case.get("input_token_ids")
    if (
        type(tokens_value) is not list
        or not tokens_value
        or any(type(token) is not int or token < 0 for token in tokens_value)
    ):
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} input_token_ids are invalid")
    tokens = cast(list[int], tokens_value)

    if real:
        expected_tokens = list(_REAL_INPUT_IDS[index])
        expected_max_new = 8
        expected_sampling = _greedy_sampling()
        expected_sampling_digest = GREEDY_SAMPLING_DIGEST
        if type(case.get("public_prompt")) is not str or case.get("public_prompt") != FIXED_PROMPTS[index]:
            _fail("INVALID_POLICY_INPUTS", f"{expected_id} public prompt differs from authority")
    else:
        expected_tokens, expected_max_new, expected_sampling = _expected_tiny_case(index)
        expected_sampling_digest = (
            GREEDY_SAMPLING_DIGEST if index < 25 else SAMPLE_SAMPLING_DIGEST
        )

    if tokens != expected_tokens:
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} token IDs differ from authority")
    if type(case.get("max_new_tokens")) is not int or case.get("max_new_tokens") != expected_max_new:
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} max_new_tokens differs from authority")
    if not real and len(tokens) + expected_max_new > 256:
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} exceeds tiny capacity")

    sampling = _validate_sampling(case.get("sampling_spec"), expected_sampling, expected_sampling_digest)
    if case.get("sampling_spec_digest") != expected_sampling_digest:
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} sampling digest differs from authority")
    computed_input_digest = _input_digest(tokens)
    if type(case.get("input_digest")) is not str or case.get("input_digest") != computed_input_digest:
        _fail("INVALID_POLICY_INPUTS", f"{expected_id} input digest differs from its token IDs")
    case["sampling_spec"] = sampling
    case["input_token_ids"] = list(tokens)
    return case


def _validate_suite(value: object, *, real: bool) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail("INVALID_POLICY_INPUTS", "suite must be an object")
    expected_fields = _REAL_SUITE_FIELDS if real else _TINY_SUITE_FIELDS
    _closed_fields(cast(Mapping[object, object], value), expected_fields, "input suite")
    suite = dict(cast(Mapping[str, object], value))
    suite_id = REAL_SUITE_ID if real else TINY_SUITE_ID
    expected_digest = REAL_SUITE_DIGEST if real else TINY_SUITE_DIGEST
    expected_count = 20 if real else 50
    if suite.get("suite_id") != suite_id:
        _fail("INVALID_POLICY_INPUTS", f"expected suite {suite_id}")
    authority = _REAL_SOURCE_AUTHORITY if real else _TINY_SOURCE_AUTHORITY
    for field, expected in authority.items():
        if not _same(suite.get(field), expected):
            _fail("INVALID_POLICY_INPUTS", f"{suite_id} source authority differs: {field}")

    cases_value = suite.get("cases")
    if type(cases_value) is not list or len(cases_value) != expected_count:
        _fail("INVALID_POLICY_INPUTS", f"{suite_id} must contain exactly {expected_count} cases")
    cases = [_validate_case(item, index=index, real=real) for index, item in enumerate(cases_value)]
    identifiers = [cast(str, item["case_id"]) for item in cases]
    if len(identifiers) != len(set(identifiers)):
        _fail("INVALID_POLICY_INPUTS", f"{suite_id} contains duplicate case IDs")
    suite["cases"] = cases
    reported_digest = suite.get("suite_digest")
    unsigned = dict(suite)
    unsigned.pop("suite_digest")
    computed_digest = _sha256_json(unsigned)
    if (
        type(reported_digest) is not str
        or reported_digest != expected_digest
        or computed_digest != expected_digest
    ):
        _fail("INVALID_POLICY_INPUTS", f"{suite_id} canonical digest differs from authority")
    return suite


@dataclass(frozen=True, slots=True)
class PolicyInputManifestV1:
    """Immutable, validated input-only manifest."""

    _canonical_bytes: bytes
    manifest_digest: str
    suite_digests: tuple[str, str]
    case_counts: tuple[int, int]

    def __post_init__(self) -> None:
        value = json.loads(self._canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict) or value.get("manifest_digest") != self.manifest_digest:
            _fail("INVALID_POLICY_INPUTS", "manifest metadata disagrees with canonical bytes")

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self._canonical_bytes.decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - guarded in __post_init__
            _fail("INVALID_POLICY_INPUTS", "canonical input manifest is not an object")
        return cast(dict[str, object], value)


def validate_policy_inputs(mapping: Mapping[str, object]) -> PolicyInputManifestV1:
    """Validate the complete input-only manifest against registered authorities."""

    if not isinstance(mapping, Mapping):
        _fail("INVALID_POLICY_INPUTS", "policy inputs must be a mapping")
    _closed_fields(cast(Mapping[object, object], mapping), _TOP_FIELDS, "input manifest")
    if (
        type(mapping.get("schema_version")) is not int
        or mapping.get("schema_version") != 1
        or mapping.get("document_kind") != DOCUMENT_KIND
        or mapping.get("status") != DOCUMENT_STATUS
        or mapping.get("protocol_id") != PROTOCOL_ID
    ):
        _fail("INVALID_POLICY_INPUTS", "input manifest identity differs from the approved proposal")
    suites_value = mapping.get("suites")
    if type(suites_value) is not list or len(suites_value) != 2:
        _fail("INVALID_POLICY_INPUTS", "input manifest must contain the real and tiny suites")
    real_suite = _validate_suite(suites_value[0], real=True)
    tiny_suite = _validate_suite(suites_value[1], real=False)

    normalized = dict(mapping)
    normalized["suites"] = [real_suite, tiny_suite]
    reported_digest = normalized.get("manifest_digest")
    unsigned = dict(normalized)
    unsigned.pop("manifest_digest")
    computed_digest = _sha256_json(unsigned)
    if (
        type(reported_digest) is not str
        or reported_digest != INPUT_MANIFEST_DIGEST
        or computed_digest != INPUT_MANIFEST_DIGEST
    ):
        _fail("INVALID_POLICY_INPUTS", "input proposal manifest digest differs from authority")
    encoded = canonical_json_bytes(normalized)
    return PolicyInputManifestV1(
        _canonical_bytes=encoded,
        manifest_digest=INPUT_MANIFEST_DIGEST,
        suite_digests=(REAL_SUITE_DIGEST, TINY_SUITE_DIGEST),
        case_counts=(20, 50),
    )


def parse_policy_inputs_json(value: str | bytes) -> PolicyInputManifestV1:
    """Parse policy-input JSON with recursive duplicate/non-finite rejection."""

    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeError as exc:
            raise PolicyInputError("INVALID_POLICY_INPUTS", "policy inputs are not UTF-8") from exc
    elif isinstance(value, str):
        text = value
    else:
        _fail("INVALID_POLICY_INPUTS", "policy input JSON must be str or bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail("INVALID_POLICY_INPUTS", f"duplicate JSON field: {key}", field=key)
            result[key] = item
        return result

    def constant(name: str) -> NoReturn:
        _fail("INVALID_POLICY_INPUTS", f"non-finite JSON number: {name}")

    try:
        parsed = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    except PolicyInputError:
        raise
    except (TypeError, ValueError) as exc:
        raise PolicyInputError("INVALID_POLICY_INPUTS", "policy input JSON is invalid") from exc
    if not isinstance(parsed, dict):
        _fail("INVALID_POLICY_INPUTS", "policy input JSON must contain an object")
    return validate_policy_inputs(cast(dict[str, object], parsed))


def load_policy_inputs(path: str | Path) -> PolicyInputManifestV1:
    """Load the byte-frozen approved draft and validate all embedded facts."""

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            _fail("INVALID_POLICY_INPUTS", "policy input source must be a regular file")
        raw = source.read_bytes()
    except PolicyInputError:
        raise
    except OSError as exc:
        raise PolicyInputError("INVALID_POLICY_INPUTS", "policy input source is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != APPROVED_INPUT_BYTES_SHA256:
        _fail("INVALID_POLICY_INPUTS", "policy input source bytes differ from the approved draft")
    return parse_policy_inputs_json(raw)


def copy_policy_inputs_snapshot(source: str | Path, destination: str | Path) -> PolicyInputManifestV1:
    """Validate then copy the approved input snapshot byte-for-byte without overwrite."""

    source_path = Path(source)
    manifest = load_policy_inputs(source_path)
    target = Path(destination)
    created = False
    try:
        raw = source_path.read_bytes()
        if target.is_symlink():
            _fail("INVALID_POLICY_INPUTS", "policy input snapshot target cannot be a symlink")
        with target.open("xb") as handle:
            created = True
            handle.write(raw)
        if target.read_bytes() != raw:
            _fail("INVALID_POLICY_INPUTS", "policy input snapshot copy is not byte-identical")
    except PolicyInputError:
        if created:
            target.unlink(missing_ok=True)
        raise
    except FileExistsError as exc:
        raise PolicyInputError("INVALID_POLICY_INPUTS", "policy input snapshot target already exists") from exc
    except OSError as exc:
        if created:
            target.unlink(missing_ok=True)
        raise PolicyInputError("INVALID_POLICY_INPUTS", "policy input snapshot copy failed") from exc
    return manifest


# Explicit aliases use manifest-oriented names for callers that prefer them.
validate_policy_input_manifest = validate_policy_inputs
load_policy_input_manifest = load_policy_inputs
copy_validated_policy_inputs = copy_policy_inputs_snapshot


__all__ = [
    "APPROVED_INPUT_BYTES_SHA256",
    "DOCUMENT_KIND",
    "DOCUMENT_STATUS",
    "GREEDY_SAMPLING_DIGEST",
    "INPUT_MANIFEST_DIGEST",
    "PROTOCOL_ID",
    "PolicyInputError",
    "PolicyInputManifestV1",
    "REAL_SUITE_DIGEST",
    "REAL_SUITE_ID",
    "SAMPLE_SAMPLING_DIGEST",
    "TINY_SUITE_DIGEST",
    "TINY_SUITE_ID",
    "copy_policy_inputs_snapshot",
    "copy_validated_policy_inputs",
    "load_policy_input_manifest",
    "load_policy_inputs",
    "parse_policy_inputs_json",
    "validate_policy_input_manifest",
    "validate_policy_inputs",
]

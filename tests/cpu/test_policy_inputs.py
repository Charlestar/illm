from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from illm.canonical import sha256_json
from illm.policy_inputs import (
    APPROVED_INPUT_BYTES_SHA256,
    GREEDY_SAMPLING_DIGEST,
    INPUT_MANIFEST_DIGEST,
    REAL_SUITE_DIGEST,
    SAMPLE_SAMPLING_DIGEST,
    TINY_SUITE_DIGEST,
    PolicyInputError,
    copy_policy_inputs_snapshot,
    load_policy_inputs,
    parse_policy_inputs_json,
    validate_policy_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "docs/implementation/v0.2-policy-inputs-draft.json"


def _draft() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(DRAFT.read_text(encoding="utf-8")))


def _resign(value: dict[str, Any]) -> None:
    for suite in value["suites"]:
        unsigned_suite = dict(suite)
        unsigned_suite.pop("suite_digest", None)
        suite["suite_digest"] = sha256_json(unsigned_suite)
    unsigned_manifest = dict(value)
    unsigned_manifest.pop("manifest_digest", None)
    value["manifest_digest"] = sha256_json(unsigned_manifest)


def test_approved_input_draft_validates_all_registered_identities() -> None:
    manifest = load_policy_inputs(DRAFT)
    value = manifest.to_dict()

    assert manifest.manifest_digest == INPUT_MANIFEST_DIGEST
    assert manifest.suite_digests == (REAL_SUITE_DIGEST, TINY_SUITE_DIGEST)
    assert manifest.case_counts == (20, 50)
    assert value["status"] == "proposed-not-runtime-authority"
    suites = cast(list[dict[str, Any]], value["suites"])
    assert len(suites[0]["cases"]) == 20
    assert len(suites[1]["cases"]) == 50
    assert suites[0]["model_revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert suites[1]["model_spec_id"] == "2951c0cfd9c3525db490dd2fd97a768e726e79ba366c8b24e820779a307ee873"
    assert APPROVED_INPUT_BYTES_SHA256 == "93d0de1c5c606a58830eeef2d6b95f4a2b389b4defd17735507d2e95b9adb754"


def test_manifest_to_dict_is_detached() -> None:
    manifest = load_policy_inputs(DRAFT)
    first = manifest.to_dict()
    cast(list[dict[str, Any]], first["suites"])[0]["cases"] = []

    second = manifest.to_dict()
    assert len(cast(list[dict[str, Any]], second["suites"])[0]["cases"]) == 20


def test_snapshot_copy_is_byte_identical_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "policy-inputs.json"
    manifest = copy_policy_inputs_snapshot(DRAFT, target)

    assert target.read_bytes() == DRAFT.read_bytes()
    assert manifest.manifest_digest == INPUT_MANIFEST_DIGEST
    with pytest.raises(PolicyInputError, match="already exists"):
        copy_policy_inputs_snapshot(DRAFT, target)


def test_loader_requires_the_approved_bytes_even_when_semantics_match(tmp_path: Path) -> None:
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(
        json.dumps(_draft(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(PolicyInputError, match="source bytes differ"):
        load_policy_inputs(reformatted)
    # Portable semantic validation remains available for an embedded snapshot.
    assert parse_policy_inputs_json(reformatted.read_bytes()).manifest_digest == INPUT_MANIFEST_DIGEST


@pytest.mark.parametrize(
    "mutation",
    [
        "real-token",
        "tiny-token",
        "tiny-length",
        "case-order",
        "sampling",
        "bool-counter",
        "real-authority",
        "tiny-authority",
        "suite-order",
        "result-field",
    ],
)
def test_re_signed_semantic_mutations_still_fail_source_authority(mutation: str) -> None:
    value = _draft()
    real = value["suites"][0]
    tiny = value["suites"][1]
    if mutation == "real-token":
        real["cases"][0]["input_token_ids"][0] += 1
        real["cases"][0]["input_digest"] = sha256_json(
            {"schema_version": 1, "input_token_ids": real["cases"][0]["input_token_ids"]}
        )
    elif mutation == "tiny-token":
        tiny["cases"][4]["input_token_ids"][0] += 1
        tiny["cases"][4]["input_digest"] = sha256_json(
            {"schema_version": 1, "input_token_ids": tiny["cases"][4]["input_token_ids"]}
        )
    elif mutation == "tiny-length":
        tiny["cases"][2]["input_token_ids"].pop()
        tiny["cases"][2]["input_digest"] = sha256_json(
            {"schema_version": 1, "input_token_ids": tiny["cases"][2]["input_token_ids"]}
        )
    elif mutation == "case-order":
        tiny["cases"][4], tiny["cases"][5] = tiny["cases"][5], tiny["cases"][4]
    elif mutation == "sampling":
        tiny["cases"][25]["sampling_spec"]["top_p"] = 0.8
        tiny["cases"][25]["sampling_spec_digest"] = sha256_json(tiny["cases"][25]["sampling_spec"])
    elif mutation == "bool-counter":
        tiny["cases"][0]["max_new_tokens"] = True
    elif mutation == "real-authority":
        real["tokenizer_digest"] = "0" * 64
    elif mutation == "tiny-authority":
        tiny["capacity_tokens"] = 512
    elif mutation == "suite-order":
        value["suites"] = [tiny, real]
    else:
        real["cases"][0]["generated_tokens"] = [1]
    _resign(value)

    with pytest.raises(PolicyInputError):
        validate_policy_inputs(value)


def test_sampling_and_nested_digests_match_approved_contract() -> None:
    value = load_policy_inputs(DRAFT).to_dict()
    real, tiny = cast(list[dict[str, Any]], value["suites"])
    real_cases = cast(list[dict[str, Any]], real["cases"])
    tiny_cases = cast(list[dict[str, Any]], tiny["cases"])

    assert {case["sampling_spec_digest"] for case in real_cases} == {GREEDY_SAMPLING_DIGEST}
    assert {case["sampling_spec_digest"] for case in tiny_cases[:25]} == {GREEDY_SAMPLING_DIGEST}
    assert {case["sampling_spec_digest"] for case in tiny_cases[25:]} == {SAMPLE_SAMPLING_DIGEST}
    assert tiny_cases[2]["max_new_tokens"] == 1
    assert len(tiny_cases[2]["input_token_ids"]) == 255


def test_raw_parser_rejects_duplicate_fields_nonfinite_and_nonobject() -> None:
    raw = DRAFT.read_text(encoding="utf-8")
    duplicate = raw.replace('"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1)
    nonfinite = raw.replace('"temperature": 0.0', '"temperature": NaN', 1)

    with pytest.raises(PolicyInputError, match="duplicate"):
        parse_policy_inputs_json(duplicate)
    with pytest.raises(PolicyInputError, match="non-finite"):
        parse_policy_inputs_json(nonfinite)
    with pytest.raises(PolicyInputError, match="object"):
        parse_policy_inputs_json("[]")


def test_manifest_and_suite_digest_tampering_is_rejected() -> None:
    manifest = _draft()
    manifest["manifest_digest"] = "0" * 64
    with pytest.raises(PolicyInputError, match="manifest digest"):
        validate_policy_inputs(manifest)

    suite = _draft()
    suite["suites"][0]["suite_digest"] = "0" * 64
    unsigned = dict(suite)
    unsigned.pop("manifest_digest")
    suite["manifest_digest"] = sha256_json(unsigned)
    with pytest.raises(PolicyInputError, match="digest differs"):
        validate_policy_inputs(suite)


def test_portable_reader_imports_without_tensor_or_tokenizer_packages() -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from illm.numerical_policy import candidate_policy; "
        "from illm.policy_inputs import load_policy_inputs; "
        f"m=load_policy_inputs(Path({str(DRAFT)!r})); "
        "assert m.case_counts == (20, 50); "
        "assert candidate_policy('bf16').is_candidate; "
        "assert not ({'torch','transformers','tokenizers','safetensors'} & set(sys.modules))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

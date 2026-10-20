"""Portable, tensor-free verification of V0.2 differential evidence.

This reader inspects embedded facts only, never the author environment paths.
Checksums prove internal consistency, not authenticity against fabricated inputs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.numerical_contracts import FIXED_PROMPTS, tensor_tolerances as _tolerances
from illm.sampling import SamplingSpec


REPORT_KIND = "illm-stateful-differential-validation-v1"
SCOPES = ("full", "real-smoke")
TINY_ORACLE_MANIFEST_DIGEST = "d424fe676820b748bf51023ca15150211d3414371314d63ce3fa647c4264fadc"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,96}")
_ZERO_RESOURCE_FIELDS = {
    "active_tickets",
    "input_leases",
    "output_leases",
    "workspace_leases",
    "kv_write_leases",
    "kv_reserved_bytes",
    "kv_valid_tokens",
    "kv_staged_tokens",
}


def _fail(code: ErrorCode, message: str, details: Mapping[str, Any] | None = None) -> NoReturn:
    raise IllmError(code, message, details)


def _object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

    def constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence must be an object: {path.name}")
    return cast(dict[str, Any], value)


def _sampling_specs() -> tuple[SamplingSpec, SamplingSpec]:
    return (
        SamplingSpec(mode="greedy", actual_seed=7),
        SamplingSpec(mode="sample", temperature=0.8, top_k=50, top_p=0.9, actual_seed=7),
    )


def _resources_zero(value: Mapping[str, object]) -> bool:
    return set(value) == _ZERO_RESOURCE_FIELDS and all(type(item) is int and item == 0 for item in value.values())


def _finite_nonnegative(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    number = cast(int | float, value)
    return math.isfinite(float(number)) and number >= 0


def _qualification_role(target: str, scope: str, actual_device: str) -> str:
    if target == "tiny" and scope == "full":
        return "tiny-full-authority"
    if target == "real" and scope == "real-smoke" and actual_device.startswith("cuda"):
        return "real-model-gpu-smoke"
    if target == "real" and scope == "real-smoke":
        return "real-model-cpu-smoke-diagnostic-not-gpu"
    if target == "tiny":
        return "tiny-smoke-diagnostic-not-full"
    return "real-model-full-diagnostic-not-tiny-authority"


def _inventory(output: Path) -> dict[str, str]:
    ignored = {"INCOMPLETE", "COMPLETE", "DEVELOPMENT_ONLY", "checksums.json"}
    expected = {"report/stateful-validation-report.json", "raw/differential-cases.json", "raw/fixed-prompts.json"}
    optional = {"authority/tiny-oracle.json"}
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            _fail(ErrorCode.CAPTURE_CORRUPT, "stateful evidence cannot contain links or escaped paths")
        if path.is_file():
            name = path.relative_to(output).as_posix()
            if name in ignored:
                continue
            if name not in expected | optional:
                _fail(ErrorCode.CAPTURE_CORRUPT, "stateful evidence includes an unexpected file")
            files[name] = sha256_file(path)
    if not expected <= set(files) <= expected | optional:
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful evidence is missing required raw/report files")
    return files


def _same(actual: object, expected: object) -> bool:
    # Python container equality equates True with 1 and False with 0. Evidence
    # counters and positions must retain their actual JSON types instead.
    return canonical_json_bytes(actual) == canonical_json_bytes(expected)


def _verified_case_records(
    raw: Mapping[str, Any], *, suite_id: str, scope: str, expected_count: int, fixed: bool,
) -> list[dict[str, Any]]:
    cases_value = raw.get("cases")
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("suite_id") != suite_id
        or raw.get("scope") != scope
        or raw.get("case_count") != expected_count
        or not isinstance(cases_value, list)
        or len(cases_value) != expected_count
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} scope/count identity is invalid")
    cases = cast(list[dict[str, Any]], cases_value)
    identifiers: set[str] = set()
    inputs: set[tuple[int, ...]] = set()
    sampled = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} case is not an object")
        case_id = case.get("case_id")
        input_ids = case.get("input_token_ids")
        sampling = case.get("sampling_spec")
        generated = case.get("generated_tokens")
        steps = case.get("steps")
        resources = case.get("resource_status_after_close")
        max_new_tokens = case.get("max_new_tokens")
        if (
            not isinstance(case_id, str)
            or _SAFE_ID.fullmatch(case_id) is None
            or case_id in identifiers
            or not isinstance(input_ids, list)
            or not input_ids
            or any(type(token) is not int or token < 0 for token in input_ids)
            or case.get("input_digest") != sha256_json({"schema_version": 1, "input_token_ids": input_ids})
            or case.get("source") != "public-synthetic-token-ids-v1"
            or type(max_new_tokens) is not int
            or max_new_tokens < 1
            or not isinstance(sampling, dict)
            or case.get("sampling_spec_digest") != sha256_json(sampling)
            or not isinstance(generated, list)
            or not 1 <= len(generated) <= max_new_tokens
            or any(type(token) is not int or token < 0 for token in generated)
            or not isinstance(steps, list)
            or len(steps) != len(generated)
            or case.get("status") != "pass"
            or case.get("first_divergence") is not None
            or case.get("resource_baseline") is not True
            or case.get("close_errors") != []
            or not isinstance(resources, dict)
            or set(resources) != {"none", "contiguous"}
            or any(not isinstance(value, dict) or not _resources_zero(value) for value in resources.values())
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} case status/identity/resource facts are invalid")
        mode = sampling.get("mode")
        if mode not in {"greedy", "sample"}:
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} sampling mode is invalid")
        if mode == "sample":
            sampled += 1
        if fixed and (mode != "greedy" or case_id != f"fixed-greedy-{index:02d}"):
            _fail(ErrorCode.CAPTURE_CORRUPT, "fixed prompt suite is not frozen greedy order")
        if fixed and (
            case.get("public_prompt") != FIXED_PROMPTS[index]
            or max_new_tokens != 8
            or len(generated) != 8
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "fixed prompt content/output count is invalid")
        if not fixed and case_id != f"differential-{index:02d}":
            _fail(ErrorCode.CAPTURE_CORRUPT, "differential case order/identity is invalid")
        identifiers.add(case_id)
        input_key = tuple(cast(list[int], input_ids))
        if not fixed and input_key in inputs:
            _fail(ErrorCode.CAPTURE_CORRUPT, "differential input cases are not distinct")
        inputs.add(input_key)
        expected_draws = 0 if mode == "greedy" else len(generated)
        expected_addresses = [] if mode == "greedy" else [[position, 0, 0] for position in range(len(generated))]
        if (
            not _same(case.get("next_output_position"), len(generated))
            or not _same(case.get("rng_draws_consumed"), expected_draws)
            or not _same(case.get("committed_draw_addresses"), expected_addresses)
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} final sampling state is invalid")
        initial = len(input_ids)
        for position, step in enumerate(steps):
            if not isinstance(step, dict):
                _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} step is not an object")
            expected_query = {"none": initial + position, "contiguous": initial if position == 0 else 1}
            expected_before = {"none": 0, "contiguous": 0 if position == 0 else initial + position - 1}
            expected_after = {"none": 0, "contiguous": initial + position}
            expected_draw = None if mode == "greedy" else [position, 0, 0]
            draws_before = 0 if mode == "greedy" else position
            draws_after = 0 if mode == "greedy" else position + 1
            logits = step.get("logits")
            chosen = step.get("chosen_token")
            if (
                step.get("status") != "pass"
                or not _same(step.get("position"), position)
                or not _same(step.get("context_tokens"), initial + position)
                or not _same(step.get("query_tokens"), expected_query)
                or not _same(step.get("kv_valid_before"), expected_before)
                or not _same(step.get("kv_valid_after"), expected_after)
                or not _same(step.get("fence_complete"), {"none": True, "contiguous": True})
                or not isinstance(logits, dict)
                or logits.get("status") != "pass"
                or logits.get("reason") is not None
                or type(logits.get("expected_values")) is not int
                or logits.get("expected_values", 0) <= 0
                or not _same(logits.get("actual_values"), logits.get("expected_values"))
                or not _finite_nonnegative(logits.get("max_absolute_error"))
                or not _finite_nonnegative(logits.get("max_relative_error"))
                or not _same(logits.get("mismatch_count"), 0)
                or logits.get("first_mismatch_index") is not None
                or not isinstance(chosen, dict)
                or not _same(chosen.get("none"), generated[position])
                or not _same(chosen.get("contiguous"), generated[position])
                or not _same(step.get("draw_address"), {"none": expected_draw, "contiguous": expected_draw})
                or not _same(step.get("next_output_position_before"), {"none": position, "contiguous": position})
                or not _same(step.get("next_output_position_after"), {"none": position + 1, "contiguous": position + 1})
                or not _same(step.get("rng_draws_before"), {"none": draws_before, "contiguous": draws_before})
                or not _same(step.get("rng_draws_after"), {"none": draws_after, "contiguous": draws_after})
                or not _same(step.get("committed_draw_addresses"),
                             {"none": expected_addresses[: position + 1], "contiguous": expected_addresses[: position + 1]})
            ):
                _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} step contract is invalid")
            finish = step.get("finish")
            if not isinstance(finish, dict) or finish.get("none") != finish.get("contiguous"):
                _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} finish facts differ")
            if position + 1 < len(generated) and finish.get("none") is not None:
                _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} has a premature terminal step")
        final_finish = None if not steps else cast(dict[str, Any], steps[-1])["finish"]["none"]
        if (
            case.get("finish_reason") != final_finish
            or final_finish not in {"EOS", "MAX_NEW_TOKENS"}
            or (final_finish == "MAX_NEW_TOKENS" and len(generated) != max_new_tokens)
            or (fixed and final_finish != "MAX_NEW_TOKENS")
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} final finish reason is invalid")
    expected_sampled = 0 if fixed or expected_count == 0 else 25
    if sampled != expected_sampled:
        _fail(ErrorCode.CAPTURE_CORRUPT, f"{suite_id} sampling case count is invalid")
    return cases


def _verify_report_semantics(
    report: Mapping[str, Any],
    differential_cases: Sequence[Mapping[str, Any]],
    fixed_cases: Sequence[Mapping[str, Any]],
) -> None:
    identity = report.get("identity")
    hardware = report.get("hardware")
    contract = report.get("contract")
    coverage = report.get("coverage")
    if not all(isinstance(value, dict) for value in (identity, hardware, contract, coverage)):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report contract sections are invalid")
    identity = cast(dict[str, Any], identity)
    hardware = cast(dict[str, Any], hardware)
    contract = cast(dict[str, Any], contract)
    coverage = cast(dict[str, Any], coverage)
    target = report.get("target")
    scope = report.get("scope")
    actual_device = hardware.get("actual_device")
    expected_sampled = 25 if scope == "full" else 0
    if (
        target not in {"tiny", "real"}
        or not isinstance(actual_device, str)
        or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", actual_device) is None
        or report.get("qualification_role") != _qualification_role(cast(str, target), cast(str, scope), actual_device)
        or report.get("qualification_tier") != "unqualified"
        or (target == "tiny") != (identity.get("fixture_id") == "tiny-decoder")
        or hardware.get("cpu_threads") != 1
        or hardware.get("tf32_matmul_enabled") is not False
        or hardware.get("tf32_cudnn_enabled") is not False
        or hardware.get("hardware_digest")
        != sha256_json({key: value for key, value in hardware.items() if key != "hardware_digest"})
        or identity.get("hardware_digest") != hardware.get("hardware_digest")
        or coverage.get("sampled_case_count") != expected_sampled
        or coverage.get("tiny_natural_eos_first") is not (scope == "full" and target == "tiny")
        or contract.get("transformers_model_loaded") is not False
        or contract.get("last_row_all_vocab_logits_compared") is not True
        or contract.get("frozen_tolerance_no_posthoc_change") is not True
        or contract.get("cpu_threads") != 1
        or contract.get("tf32_enabled") is not False
        or contract.get("full_scope_sample_case_minimum") != 25
        or contract.get("fixed_prompt_count") != len(FIXED_PROMPTS)
        or contract.get("fixed_outputs_per_prompt") != 8
        or contract.get("scope_selected_before_execution") is not True
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report scope/hardware/contract claims are invalid")
    digest_fields = (
        "model_spec_id", "normalized_spec_digest", "weights_digest", "load_manifest_digest", "tokenizer_digest",
    )
    if any(not isinstance(identity.get(name), str) or _DIGEST.fullmatch(identity[name]) is None for name in digest_fields):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report model identity digests are invalid")
    runtime_dtype = identity.get("runtime_dtype")
    if runtime_dtype not in {"fp32", "bf16", "fp16"}:
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report runtime dtype is invalid")
    tolerance, probability_tolerance = _tolerances(cast(str, runtime_dtype))
    if (
        identity.get("tolerance") != asdict(tolerance)
        or identity.get("attention_probability_tolerance_recorded_not_used_for_last_row")
        != asdict(probability_tolerance)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report tolerance differs from the frozen profile")
    greedy, sampled = _sampling_specs()
    expected_specs = {
        "greedy": greedy.to_dict(),
        "greedy_digest": greedy.digest,
        "sample": sampled.to_dict(),
        "sample_digest": sampled.digest,
    }
    if identity.get("sampling_specs") != expected_specs:
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report sampling specs differ from the frozen profiles")
    for case in [*differential_cases, *fixed_cases]:
        sampling = cast(dict[str, Any], case["sampling_spec"])
        if sampling != expected_specs[sampling["mode"]]:
            _fail(ErrorCode.CAPTURE_CORRUPT, "stateful raw case sampling spec is not frozen")
    source = identity.get("source")
    toolchain = identity.get("toolchain")
    if (
        not isinstance(source, dict)
        or any(not isinstance(source.get(name), str) or _DIGEST.fullmatch(source[name]) is None
               for name in ("project_root_digest", "bootstrap_digest"))
        or type(source.get("git_dirty")) is not bool
        or not isinstance(toolchain, dict)
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful report source/toolchain identity is invalid")


def _verify_frozen_tiny_golden(root: Path, report: Mapping[str, Any], fixed: Mapping[str, Any]) -> None:
    path = root / "authority/tiny-oracle.json"
    if report.get("target") != "tiny":
        if path.exists() or fixed.get("tiny_oracle_manifest_digest") is not None:
            _fail(ErrorCode.CAPTURE_CORRUPT, "real evidence cannot masquerade as the tiny golden authority")
        return
    oracle = _object(path)
    identity = report["identity"]
    if (
        not isinstance(identity, dict)
        or oracle.get("manifest_digest") != TINY_ORACLE_MANIFEST_DIGEST
        or sha256_json({key: value for key, value in oracle.items() if key != "manifest_digest"}) != TINY_ORACLE_MANIFEST_DIGEST
        or fixed.get("tiny_oracle_manifest_digest") != TINY_ORACLE_MANIFEST_DIGEST
        or oracle.get("spec_id") != identity.get("model_spec_id")
        or oracle.get("weights_digest") != identity.get("weights_digest")
        or oracle.get("load_manifest_digest") != identity.get("load_manifest_digest")
        or oracle.get("tokenizer_revision") != identity.get("tokenizer_revision")
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "tiny golden authority is not the frozen V0.1 oracle")
    golden = oracle.get("greedy_cases")
    cases = fixed.get("cases")
    if not isinstance(golden, list) or not isinstance(cases, list) or len(golden) != 20 or len(cases) != 20:
        _fail(ErrorCode.CAPTURE_CORRUPT, "tiny golden case set is incomplete")
    for expected, actual in zip(golden, cases, strict=True):
        if (
            not isinstance(expected, dict) or not isinstance(actual, dict)
            or actual.get("case_id") != expected.get("fixture_id")
            or actual.get("public_prompt") != expected.get("public_prompt")
            or not _same(actual.get("input_token_ids"), expected.get("input_ids"))
            or not _same(actual.get("generated_tokens"), expected.get("expected_tokens"))
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "tiny output differs from the frozen V0.1 golden tokens")


def verify_stateful_output(output: Path, *, require_formal: bool | None = None) -> dict[str, Any]:
    """Verify a completed formal or explicitly unqualified development output."""

    if output.is_symlink():
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful evidence root cannot be a symbolic link")
    root = output.resolve()
    if not root.is_dir() or (root / "INCOMPLETE").exists():
        _fail(ErrorCode.CAPTURE_INCOMPLETE, "stateful validation output is incomplete")
    formal = (root / "COMPLETE").is_file()
    development = (root / "DEVELOPMENT_ONLY").is_file()
    if require_formal is not None and type(require_formal) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "require_formal must be bool or None")
    if formal == development or (require_formal is True and not formal) or (require_formal is False and formal):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful validation completion markers are inconsistent")
    report_path = root / "report" / "stateful-validation-report.json"
    differential_path = root / "raw" / "differential-cases.json"
    fixed_path = root / "raw" / "fixed-prompts.json"
    report = _object(report_path)
    differential = _object(differential_path)
    fixed = _object(fixed_path)
    checksums = _object(root / "checksums.json")
    files = checksums.get("files")
    if (
        checksums.get("schema_version") != 1
        or not isinstance(files, dict)
        or checksums.get("inventory_digest") != sha256_json(files)
        or files != _inventory(root)
        or any(not isinstance(name, str) or not isinstance(digest, str) or sha256_file(root / name) != digest
               for name, digest in files.items())
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful validation checksum inventory is invalid")
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("report_kind") != REPORT_KIND
        or report.get("status") != "pass"
        or report.get("report_digest") != sha256_json({key: value for key, value in report.items() if key != "report_digest"})
        or report.get("first_divergence") is not None
        or report.get("all_fenced_closed_resources_zero") is not True
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful validation report identity/status is invalid")
    scope = report.get("scope")
    expected = {"differential": 50 if scope == "full" else 0, "fixed_greedy": 20}
    if scope not in SCOPES or report.get("expected_counts") != expected or report.get("actual_counts") != expected:
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful validation scope/counts are invalid")
    differential_cases = _verified_case_records(
        differential,
        suite_id="illm.v02.stateful.full-differential.v1",
        scope=cast(str, scope),
        expected_count=expected["differential"],
        fixed=False,
    )
    fixed_cases = _verified_case_records(
        fixed,
        suite_id="illm.v02.stateful.fixed-prompts-greedy.v1",
        scope=cast(str, scope),
        expected_count=expected["fixed_greedy"],
        fixed=True,
    )
    if (
        differential.get("suite_digest") != sha256_json({key: value for key, value in differential.items() if key != "suite_digest"})
        or fixed.get("suite_digest") != sha256_json({key: value for key, value in fixed.items() if key != "suite_digest"})
        or report.get("raw_digests") != {
            "differential": differential["suite_digest"],
            "fixed_greedy": fixed["suite_digest"],
        }
        or len(differential_cases) != expected["differential"]
        or len(fixed_cases) != expected["fixed_greedy"]
    ):
        _fail(ErrorCode.CAPTURE_CORRUPT, "stateful validation raw suite linkage is invalid")
    _verify_report_semantics(report, differential_cases, fixed_cases)
    _verify_frozen_tiny_golden(root, report, fixed)
    if formal:
        toolchain = cast(dict[str, Any], cast(dict[str, Any], report["identity"])["toolchain"])
        if (
            report.get("development_no_manifest") is not False
            or report.get("formal_evidence_eligible") is not True
            or toolchain.get("binding") != "finalized-source-bound-manifest"
            or toolchain.get("formal_evidence_eligible") is not True
            or not isinstance(toolchain.get("manifest_digest"), str)
            or _DIGEST.fullmatch(toolchain["manifest_digest"]) is None
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "formal stateful report lost its manifest binding")
        if (root / "COMPLETE").read_text(encoding="ascii").strip() != report["report_digest"]:
            _fail(ErrorCode.CAPTURE_CORRUPT, "COMPLETE does not bind the stateful report")
    else:
        marker = _object(root / "DEVELOPMENT_ONLY")
        toolchain = cast(dict[str, Any], cast(dict[str, Any], report["identity"])["toolchain"])
        if (
            report.get("development_no_manifest") is not True
            or report.get("formal_evidence_eligible") is not False
            or toolchain.get("binding") != "development-unqualified"
            or toolchain.get("formal_evidence_eligible") is not False
            or toolchain.get("manifest_digest") is not None
            or marker != {
                "qualification_tier": "unqualified",
                "formal_evidence_eligible": False,
                "report_digest": report["report_digest"],
            }
        ):
            _fail(ErrorCode.CAPTURE_CORRUPT, "development marker/report boundary is invalid")
    return report

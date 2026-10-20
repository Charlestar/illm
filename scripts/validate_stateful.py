"""Produce source-bound V0.2 cache-off/cache-on differential evidence.

``full`` is the tiny-model authority: 50 new public differential cases plus
the 20 frozen V0.1 public prompts. ``real-smoke`` is the predeclared real-model
GPU scope: only the 20 frozen prompts. A scope is never selected from results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence, cast

from illm import __version__
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.domain.errors import EXIT_CODE, ErrorCode, IllmError
from illm.engine.facade import as_illm_error
from illm.executor import EagerExecutor, ExecutionTicket
from illm.model import LoadedModel, load_model
from illm.model.validation import FIXED_PROMPTS, TensorTolerance, _tolerances
from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample
from illm.stateful_validation import verify_stateful_output as verify_stateful_output

from _toolchain_common import bootstrap_digest, project_digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_KIND = "illm-stateful-differential-validation-v1"
SCOPES = ("full", "real-smoke")
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


def _signed(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        _fail(ErrorCode.INTERNAL, f"document already contains {field}")
    value[field] = sha256_json(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical_json_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        _fail(ErrorCode.CAPTURE_CORRUPT, f"JSON evidence must be an object: {path.name}")
    return cast(dict[str, Any], value)


def _source_identity() -> dict[str, Any]:
    commit = "unversioned"
    dirty = True
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )
        if revision.returncode == status.returncode == 0 and revision.stdout.strip():
            commit = revision.stdout.strip()
            dirty = bool(status.stdout.strip())
    except OSError:
        pass
    return {
        "project_root_digest": project_digest(PROJECT_ROOT),
        "bootstrap_digest": bootstrap_digest(PROJECT_ROOT),
        "git_commit": commit,
        "git_dirty": dirty,
    }


def _toolchain_binding(development_no_manifest: bool, source: Mapping[str, Any]) -> dict[str, Any]:
    if type(development_no_manifest) is not bool:
        _fail(ErrorCode.INVALID_ARGUMENT, "development_no_manifest must be bool")
    if development_no_manifest:
        return {
            "binding": "development-unqualified",
            "formal_evidence_eligible": False,
            "manifest_digest": None,
            "manifest_environment_ignored": bool(os.environ.get("ILLM_TOOLCHAIN_MANIFEST")),
            "python_executable": str(Path(sys.executable).resolve()),
        }

    manifest_name = os.environ.get("ILLM_TOOLCHAIN_MANIFEST")
    if not manifest_name:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "run stateful validation through a finalized manifest runner")
    manifest_path = Path(manifest_name).resolve()
    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest is unreadable") from exc
    if not isinstance(manifest_value, dict):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest must be an object")
    manifest = cast(dict[str, Any], manifest_value)
    digest = manifest.get("manifest_digest")
    required = {
        "schema_version",
        "finalized",
        "environment_id",
        "python_executable",
        "python_version",
        "project_root",
        "project_root_digest",
        "bootstrap_digest",
        "lock_digest",
        "runtime_profile_id",
        "runtime_lock_digest",
        "manifest_digest",
    }
    if not required <= set(manifest) or manifest.get("schema_version") != 1 or manifest.get("finalized") is not True:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "stateful validation requires a finalized selected runtime profile")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None or digest != sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest checksum mismatch")
    environment_id = manifest.get("environment_id")
    python_executable = manifest.get("python_executable")
    lock_identity = manifest.get("lock_digest")
    runtime_lock = manifest.get("runtime_lock_digest")
    if (
        not isinstance(environment_id, str)
        or os.environ.get("ILLM_ENVIRONMENT_ID") != environment_id
        or not isinstance(python_executable, str)
        or Path(python_executable).resolve() != Path(sys.executable).resolve()
        or not isinstance(lock_identity, str)
        or _DIGEST.fullmatch(lock_identity) is None
        or os.environ.get("ILLM_LOCK_DIGEST") != lock_identity
        or not isinstance(runtime_lock, str)
        or _DIGEST.fullmatch(runtime_lock) is None
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "current interpreter/environment differs from the finalized manifest")
    if (
        Path(str(manifest.get("project_root"))).resolve() != PROJECT_ROOT
        or manifest.get("project_root_digest") != source["project_root_digest"]
        or manifest.get("bootstrap_digest") != source["bootstrap_digest"]
        or manifest.get("python_version") != platform.python_version()
    ):
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "toolchain manifest is not bound to the current source/runtime")
    runtime_profile_id = manifest.get("runtime_profile_id")
    if not isinstance(runtime_profile_id, str) or not runtime_profile_id:
        _fail(ErrorCode.TOOLCHAIN_MISMATCH, "runtime profile identity is missing")
    return {
        "binding": "finalized-source-bound-manifest",
        "formal_evidence_eligible": True,
        "manifest_path": str(manifest_path),
        "manifest_digest": digest,
        "environment_id": environment_id,
        "python_executable": str(Path(python_executable).resolve()),
        "lock_digest": lock_identity,
        "runtime_profile_id": runtime_profile_id,
        "runtime_lock_digest": runtime_lock,
    }


@dataclass(frozen=True, slots=True)
class PublicCase:
    case_id: str
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingSpec
    coverage: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": "public-synthetic-token-ids-v1",
            "input_token_ids": list(self.input_token_ids),
            "input_digest": sha256_json({"schema_version": 1, "input_token_ids": list(self.input_token_ids)}),
            "max_new_tokens": self.max_new_tokens,
            "sampling_spec": self.sampling.to_dict(),
            "sampling_spec_digest": self.sampling.digest,
            "coverage": list(self.coverage),
        }


def _sampling_specs() -> tuple[SamplingSpec, SamplingSpec]:
    return (
        SamplingSpec(mode="greedy", actual_seed=7),
        SamplingSpec(mode="sample", temperature=0.8, top_k=50, top_p=0.9, actual_seed=7),
    )


def build_differential_cases(loaded: LoadedModel) -> tuple[PublicCase, ...]:
    """Build the fixed 50-case public suite without using validation results."""

    capacity = loaded.max_context_tokens
    vocabulary = loaded.spec.vocab_size
    if capacity < 8 or vocabulary < 8:
        _fail(ErrorCode.INVALID_MODEL_CONFIG, "the frozen differential suite requires capacity/vocabulary >= 8")
    greedy, sampled = _sampling_specs()

    def tokens(case_index: int, length: int) -> tuple[int, ...]:
        usable = vocabulary - 3
        return tuple(3 + ((case_index * 43 + position * 17) % usable) for position in range(length))

    cases: list[PublicCase] = []
    for index in range(50):
        sampling = greedy if index < 25 else sampled
        coverage: list[str] = ["gqa", "rope"]
        token_ids: tuple[int, ...]
        outputs: int
        if index == 0:
            token_ids, outputs = (3,), 1
            coverage.extend(("length-1", "max-new-1"))
        elif index == 1:
            token_ids, outputs = (4, 5), 2
            coverage.append("length-2")
        elif index == 2:
            token_ids, outputs = tokens(index, capacity - 1), 1
            coverage.extend(("capacity-minus-1", "max-new-1"))
        elif index == 3:
            token_ids, outputs = (loaded.spec.eos_token_id,), 3
            coverage.extend(("eos-input", "tiny-natural-eos-first"))
        else:
            outputs = 1 + index % 3
            maximum_length = min(48, capacity - outputs)
            length = 3 + ((index * 11) % (maximum_length - 2))
            token_ids = tokens(index, length)
            if outputs == 1:
                coverage.append("max-new-1")
        cases.append(PublicCase(f"differential-{index:02d}", token_ids, outputs, sampling, tuple(coverage)))
    _validate_differential_cases(cases, loaded)
    return tuple(cases)


def _validate_differential_cases(cases: Sequence[PublicCase], loaded: LoadedModel) -> None:
    if len(cases) != 50 or any(type(case) is not PublicCase for case in cases):
        _fail(ErrorCode.INVALID_ARGUMENT, "full scope requires exactly 50 PublicCase values")
    identifiers = [case.case_id for case in cases]
    inputs = [case.input_token_ids for case in cases]
    fixed_inputs = {loaded.tokenizer.encode(prompt) for prompt in FIXED_PROMPTS}
    if (
        len(set(identifiers)) != 50
        or len(set(inputs)) != 50
        or any(_SAFE_ID.fullmatch(case_id) is None for case_id in identifiers)
        or any(input_ids in fixed_inputs for input_ids in inputs)
    ):
        _fail(ErrorCode.INVALID_ARGUMENT, "differential cases must be distinct and separate from FIXED_PROMPTS")
    sampled = 0
    coverage: set[str] = set()
    _, frozen_sampled = _sampling_specs()
    for case in cases:
        if not case.input_token_ids or any(
            type(token) is not int or not 0 <= token < loaded.spec.vocab_size for token in case.input_token_ids
        ):
            _fail(ErrorCode.INVALID_ARGUMENT, "differential input token IDs are invalid")
        if type(case.max_new_tokens) is not int or case.max_new_tokens < 1:
            _fail(ErrorCode.INVALID_ARGUMENT, "differential max_new_tokens must be positive")
        if len(case.input_token_ids) + case.max_new_tokens > loaded.max_context_tokens:
            _fail(ErrorCode.CONTEXT_LIMIT_EXCEEDED, "differential case exceeds the selected model profile")
        if case.sampling.mode == "sample":
            sampled += 1
            if case.sampling != frozen_sampled:
                _fail(ErrorCode.INVALID_ARGUMENT, "sample cases must use the frozen seed/top-k/top-p spec")
        coverage.update(case.coverage)
    if sampled < 25:
        _fail(ErrorCode.INVALID_ARGUMENT, "at least half of full-scope differential cases must sample")
    required = {"length-1", "length-2", "capacity-minus-1", "eos-input", "max-new-1", "gqa", "rope"}
    if not required <= coverage:
        _fail(ErrorCode.INVALID_ARGUMENT, "differential suite coverage is incomplete")


def _compare_logits(
    expected: tuple[float, ...], actual: tuple[float, ...], tolerance: TensorTolerance,
) -> dict[str, Any]:
    if not expected or len(expected) != len(actual):
        return {
            "status": "fail",
            "reason": "shape_mismatch",
            "expected_values": len(expected),
            "actual_values": len(actual),
            "max_absolute_error": None,
            "max_relative_error": None,
            "mismatch_count": max(len(expected), len(actual)),
        }
    if not all(math.isfinite(value) for value in (*expected, *actual)):
        return {
            "status": "fail",
            "reason": "non_finite",
            "expected_values": len(expected),
            "actual_values": len(actual),
            "max_absolute_error": None,
            "max_relative_error": None,
            "mismatch_count": len(expected),
        }
    absolute = [abs(left - right) for left, right in zip(expected, actual)]
    relative = [difference / max(abs(left), 1e-30) for difference, left in zip(absolute, expected)]
    mismatches = [
        index
        for index, (difference, left) in enumerate(zip(absolute, expected))
        if difference > tolerance.atol + tolerance.rtol * abs(left)
    ]
    return {
        "status": "pass" if not mismatches else "fail",
        "reason": None if not mismatches else "outside_frozen_tolerance",
        "expected_values": len(expected),
        "actual_values": len(actual),
        "max_absolute_error": max(absolute),
        "max_relative_error": max(relative),
        "mismatch_count": len(mismatches),
        "first_mismatch_index": mismatches[0] if mismatches else None,
    }


def _status(executor: EagerExecutor) -> dict[str, int]:
    return asdict(executor.resource_status)


def _resources_zero(value: Mapping[str, object]) -> bool:
    return set(value) == _ZERO_RESOURCE_FIELDS and all(type(item) is int and item == 0 for item in value.values())


def _finite_nonnegative(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    number = cast(int | float, value)
    return math.isfinite(float(number)) and number >= 0


def _abort_active(executor: EagerExecutor, ticket: ExecutionTicket | None) -> dict[str, str] | None:
    if ticket is None or ticket.state not in {"launched", "resolved"}:
        return None
    try:
        executor.abort(ticket)
        return None
    except Exception as exc:
        return {"exception_type": type(exc).__name__, "message": str(exc)}


def _draw_address(decision: Any) -> list[int] | None:
    if decision.draw is None:
        return None
    address = decision.draw.address
    return [address.absolute_output_pos, address.draw_kind, address.draw_index]


def _execute_case(
    loaded: LoadedModel,
    case: PublicCase,
    tolerance: TensorTolerance,
    *,
    expected_tokens: tuple[int, ...] | None = None,
    require_exact_steps: bool = False,
    require_tiny_eos_first: bool = False,
) -> dict[str, Any]:
    none = EagerExecutor(loaded, "none", loaded.max_context_tokens)
    contiguous = EagerExecutor(loaded, "contiguous", loaded.max_context_tokens)
    request_id = f"stateful-validation-{case.case_id}"
    none_state = SamplingState.for_request(request_id, case.sampling)
    cached_state = SamplingState.for_request(request_id, case.sampling)
    none_output: list[int] = []
    cached_output: list[int] = []
    steps: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    finish_reason: str | None = None
    close_errors: list[dict[str, str]] = []

    for position in range(case.max_new_tokens):
        none_ticket: ExecutionTicket | None = None
        cached_ticket: ExecutionTicket | None = None
        step: dict[str, Any] = {
            "position": position,
            "sampling_mode": case.sampling.mode,
            "context_tokens": len(case.input_token_ids) + position,
            "status": "pass",
        }
        try:
            none_context = (*case.input_token_ids, *none_output)
            cached_context = (*case.input_token_ids, *cached_output)
            if none_context != cached_context:
                _fail(ErrorCode.ORACLE_DIVERGED, "cache-off/cache-on committed contexts differ")
            none_ticket = none.launch(none_context)
            cached_ticket = contiguous.launch(cached_context)
            none_logits = none.resolve(none_ticket)
            cached_logits = contiguous.resolve(cached_ticket)
            comparison = _compare_logits(none_logits, cached_logits, tolerance)
            none_decision = prepare_sample(none_logits, case.sampling, none_state)
            cached_decision = prepare_sample(cached_logits, case.sampling, cached_state)
            reasons: list[str] = []
            expected_query = {"none": len(none_context), "contiguous": len(case.input_token_ids) if position == 0 else 1}
            expected_valid_before = {
                "none": 0,
                "contiguous": 0 if position == 0 else len(case.input_token_ids) + position - 1,
            }
            fence_complete = {
                "none": none_ticket.fence.is_complete(),
                "contiguous": cached_ticket.fence.is_complete(),
            }
            expected_draw_address = None if case.sampling.mode == "greedy" else [position, 0, 0]
            expected_draws_before = 0 if case.sampling.mode == "greedy" else position
            if comparison["status"] != "pass":
                reasons.append("last_row_logits")
            if {
                "none": none_ticket.query_tokens,
                "contiguous": cached_ticket.query_tokens,
            } != expected_query:
                reasons.append("query_tokens")
            if {
                "none": none_ticket.before_valid_tokens,
                "contiguous": cached_ticket.before_valid_tokens,
            } != expected_valid_before:
                reasons.append("kv_valid_before")
            if not all(fence_complete.values()):
                reasons.append("fence_incomplete_at_resolve")
            if none_state.next_output_position != position or cached_state.next_output_position != position:
                reasons.append("next_output_position_before")
            if (
                none_state.rng_draws_consumed != expected_draws_before
                or cached_state.rng_draws_consumed != expected_draws_before
            ):
                reasons.append("rng_draws_before")
            if none_decision.token_id != cached_decision.token_id:
                reasons.append("chosen_token")
            if (
                none_decision.output_position != position
                or cached_decision.output_position != position
            ):
                reasons.append("output_position")
            if (
                _draw_address(none_decision) != expected_draw_address
                or _draw_address(cached_decision) != expected_draw_address
            ):
                reasons.append("rng_draw_address")
            step.update({
                "query_tokens": {"none": none_ticket.query_tokens, "contiguous": cached_ticket.query_tokens},
                "kv_valid_before": {
                    "none": none_ticket.before_valid_tokens,
                    "contiguous": cached_ticket.before_valid_tokens,
                },
                "fence_complete": fence_complete,
                "logits": comparison,
                "chosen_token": {"none": none_decision.token_id, "contiguous": cached_decision.token_id},
                "draw_address": {
                    "none": _draw_address(none_decision),
                    "contiguous": _draw_address(cached_decision),
                },
                "next_output_position_before": {
                    "none": none_state.next_output_position,
                    "contiguous": cached_state.next_output_position,
                },
                "rng_draws_before": {
                    "none": none_state.rng_draws_consumed,
                    "contiguous": cached_state.rng_draws_consumed,
                },
            })
            if reasons:
                step.update(status="fail", reasons=reasons, finish={"none": None, "contiguous": None})
                _abort_active(none, none_ticket)
                _abort_active(contiguous, cached_ticket)
            else:
                none_valid = none.commit(none_ticket)
                cached_valid = contiguous.commit(cached_ticket)
                none_state = commit_sample(none_state, none_decision, spec=case.sampling, logits=none_logits)
                cached_state = commit_sample(cached_state, cached_decision, spec=case.sampling, logits=cached_logits)
                none_output.append(none_decision.token_id)
                cached_output.append(cached_decision.token_id)
                expected_draws_after = 0 if case.sampling.mode == "greedy" else position + 1
                expected_addresses = (
                    []
                    if case.sampling.mode == "greedy"
                    else [[item, 0, 0] for item in range(position + 1)]
                )
                if none_state.to_dict() != cached_state.to_dict():
                    reasons.append("sampling_state")
                if none_output != cached_output:
                    reasons.append("committed_output")
                if none_valid != 0 or cached_valid != len(cached_context):
                    reasons.append("kv_valid_after")
                if (
                    none_state.next_output_position != position + 1
                    or cached_state.next_output_position != position + 1
                ):
                    reasons.append("next_output_position_after")
                if (
                    none_state.rng_draws_consumed != expected_draws_after
                    or cached_state.rng_draws_consumed != expected_draws_after
                ):
                    reasons.append("rng_draws_after")
                none_addresses = [
                    [item.absolute_output_pos, item.draw_kind, item.draw_index]
                    for item in none_state.committed_draw_addresses
                ]
                cached_addresses = [
                    [item.absolute_output_pos, item.draw_kind, item.draw_index]
                    for item in cached_state.committed_draw_addresses
                ]
                if none_addresses != expected_addresses or cached_addresses != expected_addresses:
                    reasons.append("committed_draw_addresses")
                if expected_tokens is not None and (
                    position >= len(expected_tokens) or none_decision.token_id != expected_tokens[position]
                ):
                    reasons.append("committed_v01_oracle_token")
                none_finish = (
                    "EOS"
                    if none_decision.token_id == loaded.spec.eos_token_id
                    else "MAX_NEW_TOKENS" if position + 1 == case.max_new_tokens else None
                )
                cached_finish = (
                    "EOS"
                    if cached_decision.token_id == loaded.spec.eos_token_id
                    else "MAX_NEW_TOKENS" if position + 1 == case.max_new_tokens else None
                )
                if none_finish != cached_finish:
                    reasons.append("finish_reason")
                if require_tiny_eos_first and position == 0 and none_finish != "EOS":
                    reasons.append("tiny_natural_eos_first")
                if require_exact_steps and none_finish == "EOS" and position + 1 < case.max_new_tokens:
                    reasons.append("unexpected_early_eos")
                finish_reason = none_finish
                step.update({
                    "kv_valid_after": {"none": none_valid, "contiguous": cached_valid},
                    "next_output_position_after": {
                        "none": none_state.next_output_position,
                        "contiguous": cached_state.next_output_position,
                    },
                    "rng_draws_after": {
                        "none": none_state.rng_draws_consumed,
                        "contiguous": cached_state.rng_draws_consumed,
                    },
                    "committed_draw_addresses": {
                        "none": none_addresses,
                        "contiguous": cached_addresses,
                    },
                    "finish": {"none": none_finish, "contiguous": cached_finish},
                })
                if reasons:
                    step.update(status="fail", reasons=reasons)
            steps.append(step)
            if step["status"] != "pass":
                first_divergence = {
                    "case_id": case.case_id,
                    "position": position,
                    "reasons": step.get("reasons", []),
                    "step": step,
                }
                break
            if finish_reason == "EOS":
                break
        except Exception as exc:
            cleanup = [
                item
                for item in (_abort_active(none, none_ticket), _abort_active(contiguous, cached_ticket))
                if item is not None
            ]
            step.update(
                status="fail",
                reasons=["execution_exception"],
                exception={"exception_type": type(exc).__name__, "message": str(exc)},
                cleanup_errors=cleanup,
            )
            steps.append(step)
            first_divergence = {
                "case_id": case.case_id,
                "position": position,
                "reasons": ["execution_exception"],
                "step": step,
            }
            break

    for mode, executor in (("none", none), ("contiguous", contiguous)):
        try:
            executor.close()
        except Exception as exc:
            close_errors.append({"mode": mode, "exception_type": type(exc).__name__, "message": str(exc)})
    resources = {"none": _status(none), "contiguous": _status(contiguous)}
    resource_baseline = not close_errors and all(_resources_zero(item) for item in resources.values())
    if require_exact_steps and len(none_output) != case.max_new_tokens and first_divergence is None:
        first_divergence = {
            "case_id": case.case_id,
            "position": len(none_output),
            "reasons": ["generated_count"],
            "expected": case.max_new_tokens,
            "actual": len(none_output),
        }
    if not resource_baseline and first_divergence is None:
        first_divergence = {
            "case_id": case.case_id,
            "position": len(steps),
            "reasons": ["resource_baseline"],
            "resources": resources,
            "close_errors": close_errors,
        }
    return {
        **case.to_dict(),
        "tokenizer_input_identity": loaded.tokenizer.input_identity(case.input_token_ids, add_bos=False),
        "status": "pass" if first_divergence is None else "fail",
        "generated_tokens": none_output,
        "finish_reason": finish_reason,
        "next_output_position": none_state.next_output_position,
        "rng_draws_consumed": none_state.rng_draws_consumed,
        "committed_draw_addresses": [
            [item.absolute_output_pos, item.draw_kind, item.draw_index]
            for item in none_state.committed_draw_addresses
        ],
        "steps": steps,
        "first_divergence": first_divergence,
        "resource_status_after_close": resources,
        "resource_baseline": resource_baseline,
        "close_errors": close_errors,
    }


def _tiny_oracle_manifest(loaded: LoadedModel) -> dict[str, Any] | None:
    if loaded.fixture_id != "tiny-decoder":
        return None
    path = loaded.assets_directory / "oracle-v1" / "oracle-manifest.json"
    value = _object(path)
    claimed = value.get("manifest_digest")
    if not isinstance(claimed, str) or claimed != sha256_json(
        {key: item for key, item in value.items() if key != "manifest_digest"}
    ):
        _fail(ErrorCode.INVALID_WEIGHTS, "tiny oracle manifest checksum mismatch")
    prompts = value.get("greedy_cases")
    if (
        value.get("spec_id") != loaded.spec.spec_id
        or value.get("weights_digest") != loaded.weight_manifest.get("weights_digest")
        or value.get("load_manifest_digest") != loaded.weight_manifest.get("manifest_digest")
        or value.get("tokenizer_revision") != loaded.tokenizer.revision
        or not isinstance(prompts, list)
        or len(prompts) != len(FIXED_PROMPTS)
        or [item.get("public_prompt") for item in prompts if isinstance(item, dict)] != list(FIXED_PROMPTS)
    ):
        _fail(ErrorCode.INVALID_WEIGHTS, "tiny oracle manifest identity/input set mismatch")
    return value


def _fixed_cases(loaded: LoadedModel, oracle: Mapping[str, Any] | None) -> tuple[tuple[PublicCase, tuple[int, ...] | None], ...]:
    greedy, _ = _sampling_specs()
    expected_by_id: dict[str, tuple[int, ...]] = {}
    if oracle is not None:
        for raw in cast(list[dict[str, Any]], oracle["greedy_cases"]):
            expected_by_id[str(raw["fixture_id"])] = tuple(int(token) for token in raw["expected_tokens"])
    cases: list[tuple[PublicCase, tuple[int, ...] | None]] = []
    for index, prompt in enumerate(FIXED_PROMPTS):
        case_id = f"fixed-greedy-{index:02d}"
        token_ids = loaded.tokenizer.encode(prompt)
        if len(token_ids) + 8 > loaded.max_context_tokens:
            _fail(ErrorCode.CONTEXT_LIMIT_EXCEEDED, "FIXED_PROMPTS case exceeds selected model capacity")
        expected = expected_by_id.get(case_id)
        if oracle is not None and (expected is None or len(expected) != 8):
            _fail(ErrorCode.INVALID_WEIGHTS, "tiny oracle must contain eight tokens for every FIXED_PROMPTS case")
        cases.append((PublicCase(case_id, token_ids, 8, greedy, ("v01-fixed-public-prompt",)), expected))
    return tuple(cases)


def _hardware(torch: Any, actual_device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "actual_device": actual_device,
        "cpu_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_matmul_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_enabled": bool(torch.backends.cudnn.allow_tf32),
    }
    if actual_device.startswith("cuda"):
        index = torch.device(actual_device).index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "gpu_index": index,
            "gpu_name": properties.name,
            "total_device_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
        })
        try:
            process = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={index}",
                    "--query-gpu=index,uuid,name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "NVIDIA driver identity cannot be recorded") from exc
        lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if process.returncode != 0 or len(lines) != 1:
            _fail(ErrorCode.TOOLCHAIN_MISMATCH, "NVIDIA driver identity cannot be recorded")
        result["nvidia_smi_identity"] = lines[0]
    else:
        result["nvidia_smi_identity"] = None
    result["hardware_digest"] = sha256_json(result)
    return result


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
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def _write_checksums(output: Path) -> None:
    files = _inventory(output)
    _atomic_json(output / "checksums.json", {
        "schema_version": 1,
        "files": files,
        "inventory_digest": sha256_json(files),
    })











def run_validation(
    model: str | Path,
    *,
    device: str,
    output: Path,
    scope: str = "full",
    development_no_manifest: bool = False,
) -> dict[str, Any]:
    """Run the predeclared differential scope and write immutable raw evidence."""

    if scope not in SCOPES:
        _fail(ErrorCode.INVALID_ARGUMENT, "scope must be full or real-smoke")
    if not isinstance(model, (str, Path)) or not str(model):
        _fail(ErrorCode.INVALID_ARGUMENT, "model must be a fixture identifier or local profile")
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
        _fail(ErrorCode.INVALID_ARGUMENT, "device must be cpu, cuda or cuda:<index>")
    source = _source_identity()
    toolchain = _toolchain_binding(development_no_manifest, source)
    root = output.resolve()
    if root.exists() and (root.is_symlink() or not root.is_dir() or any(root.iterdir())):
        _fail(ErrorCode.INVALID_ARGUMENT, "stateful validation output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "INCOMPLETE", {"reason": "stateful_validation_in_progress", "scope": scope})

    import torch

    original_threads = torch.get_num_threads()
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        loaded = load_model(model, device=device, project_root=PROJECT_ROOT)
        actual_device = str(loaded.reference.canonical_weight("model.embed_tokens.weight").device)
        hardware = _hardware(torch, actual_device)
        tolerance, probability_tolerance = _tolerances(loaded.spec.runtime_dtype)
        oracle = _tiny_oracle_manifest(loaded)
        if oracle is not None:
            _atomic_json(root / "authority" / "tiny-oracle.json", oracle)
        fixed_cases = _fixed_cases(loaded, oracle)
        differential_cases = build_differential_cases(loaded) if scope == "full" else ()

        differential_results: list[dict[str, Any]] = []
        fixed_results: list[dict[str, Any]] = []
        first_divergence: dict[str, Any] | None = None
        for case in differential_cases:
            result = _execute_case(
                loaded,
                case,
                tolerance,
                require_tiny_eos_first=loaded.fixture_id == "tiny-decoder" and "tiny-natural-eos-first" in case.coverage,
            )
            differential_results.append(result)
            if result["status"] != "pass" and first_divergence is None:
                first_divergence = {"suite": "differential", **cast(dict[str, Any], result["first_divergence"])}
            if result["resource_baseline"] is not True:
                break
        if len(differential_results) == len(differential_cases) and all(
            result["resource_baseline"] is True for result in differential_results
        ):
            for case, expected_tokens in fixed_cases:
                result = _execute_case(
                    loaded,
                    case,
                    tolerance,
                    expected_tokens=expected_tokens,
                    require_exact_steps=True,
                )
                result["public_prompt"] = FIXED_PROMPTS[int(case.case_id.rsplit("-", 1)[1])]
                result["tokenizer_input_identity"] = loaded.tokenizer.input_identity(
                    case.input_token_ids, add_bos=loaded.fixture_id is not None,
                )
                fixed_results.append(result)
                if result["status"] != "pass" and first_divergence is None:
                    first_divergence = {"suite": "fixed_greedy", **cast(dict[str, Any], result["first_divergence"])}
                if result["resource_baseline"] is not True:
                    break

        differential_raw = _signed({
            "schema_version": 1,
            "suite_id": "illm.v02.stateful.full-differential.v1",
            "scope": scope,
            "case_count": len(differential_results),
            "cases": differential_results,
        }, "suite_digest")
        fixed_raw = _signed({
            "schema_version": 1,
            "suite_id": "illm.v02.stateful.fixed-prompts-greedy.v1",
            "scope": scope,
            "case_count": len(fixed_results),
            "cases": fixed_results,
            "tiny_oracle_manifest_digest": oracle.get("manifest_digest") if oracle is not None else None,
        }, "suite_digest")
        _atomic_json(root / "raw" / "differential-cases.json", differential_raw)
        _atomic_json(root / "raw" / "fixed-prompts.json", fixed_raw)

        expected_counts = {"differential": 50 if scope == "full" else 0, "fixed_greedy": 20}
        actual_counts = {"differential": len(differential_results), "fixed_greedy": len(fixed_results)}
        resources_zero = all(
            result.get("resource_baseline") is True for result in [*differential_results, *fixed_results]
        )
        passed = (
            actual_counts == expected_counts
            and first_divergence is None
            and resources_zero
            and all(result.get("status") == "pass" for result in [*differential_results, *fixed_results])
        )
        greedy, sampled = _sampling_specs()
        target = "tiny" if loaded.fixture_id == "tiny-decoder" else "real"
        qualification_role = _qualification_role(target, scope, actual_device)
        report = _signed({
            "schema_version": 1,
            "report_kind": REPORT_KIND,
            "status": "pass" if passed else "fail",
            "scope": scope,
            "target": target,
            "qualification_role": qualification_role,
            "qualification_tier": "unqualified",
            "formal_evidence_eligible": bool(passed and not development_no_manifest),
            "development_no_manifest": development_no_manifest,
            "expected_counts": expected_counts,
            "actual_counts": actual_counts,
            "first_divergence": first_divergence,
            "all_fenced_closed_resources_zero": resources_zero,
            "identity": {
                "engine_version": __version__,
                "model_argument": str(model),
                "fixture_id": loaded.fixture_id,
                "model_spec_id": loaded.spec.spec_id,
                "normalized_spec_digest": sha256_json(loaded.spec.to_dict()),
                "weights_digest": loaded.weight_manifest["weights_digest"],
                "load_manifest_digest": loaded.weight_manifest["manifest_digest"],
                "tokenizer_revision": loaded.tokenizer.revision,
                "tokenizer_digest": sha256_json(loaded.tokenizer.to_dict()),
                "runtime_dtype": loaded.spec.runtime_dtype,
                "model_profile_capacity_tokens": loaded.max_context_tokens,
                "sampling_specs": {
                    "greedy": greedy.to_dict(),
                    "greedy_digest": greedy.digest,
                    "sample": sampled.to_dict(),
                    "sample_digest": sampled.digest,
                },
                "tolerance": asdict(tolerance),
                "attention_probability_tolerance_recorded_not_used_for_last_row": asdict(probability_tolerance),
                "source": source,
                "toolchain": toolchain,
                "hardware_digest": hardware["hardware_digest"],
            },
            "hardware": hardware,
            "contract": {
                "oracle": "same LoadedModel immutable V0.1 reference weights; none full-context vs contiguous incremental",
                "transformers_model_loaded": False,
                "last_row_all_vocab_logits_compared": True,
                "frozen_tolerance_no_posthoc_change": True,
                "cpu_threads": 1,
                "tf32_enabled": False,
                "full_scope_sample_case_minimum": 25,
                "fixed_prompt_count": len(FIXED_PROMPTS),
                "fixed_outputs_per_prompt": 8,
                "scope_selected_before_execution": True,
            },
            "coverage": {
                "length_1": scope == "full",
                "length_2": scope == "full",
                "capacity_minus_1_max_new_1": scope == "full",
                "eos_input": scope == "full",
                "tiny_natural_eos_first": scope == "full" and target == "tiny",
                "max_new_1": scope == "full",
                "gqa": loaded.spec.num_attention_heads > loaded.spec.num_kv_heads,
                "rope_nonzero_positions": True,
                "sampled_case_count": sum(case.sampling.mode == "sample" for case in differential_cases),
            },
            "raw_digests": {
                "differential": differential_raw["suite_digest"],
                "fixed_greedy": fixed_raw["suite_digest"],
            },
        }, "report_digest")
        _atomic_json(root / "report" / "stateful-validation-report.json", report)
        _write_checksums(root)
        if passed and development_no_manifest:
            _atomic_json(root / "DEVELOPMENT_ONLY", {
                "qualification_tier": "unqualified",
                "formal_evidence_eligible": False,
                "report_digest": report["report_digest"],
            })
            (root / "INCOMPLETE").unlink()
            verify_stateful_output(root, require_formal=False)
        elif passed:
            _atomic_text(root / "COMPLETE", str(report["report_digest"]) + "\n")
            (root / "INCOMPLETE").unlink()
            try:
                verify_stateful_output(root, require_formal=True)
            except Exception:
                (root / "COMPLETE").unlink(missing_ok=True)
                _atomic_json(root / "INCOMPLETE", {"reason": "portable_self_verification_failed", "scope": scope})
                raise
        else:
            _atomic_json(root / "INCOMPLETE", {
                "reason": "stateful_validation_failed",
                "scope": scope,
                "first_divergence": first_divergence,
                "report_digest": report["report_digest"],
            })
        return report
    except BaseException as exc:
        (root / "COMPLETE").unlink(missing_ok=True)
        (root / "DEVELOPMENT_ONLY").unlink(missing_ok=True)
        existing_reason = None
        if (root / "INCOMPLETE").is_file():
            try:
                existing_reason = _object(root / "INCOMPLETE").get("reason")
            except IllmError:
                pass
        if existing_reason not in {"stateful_validation_failed", "portable_self_verification_failed"}:
            try:
                _atomic_json(root / "INCOMPLETE", {
                    "reason": "stateful_validation_exception",
                    "scope": scope,
                    "exception_type": type(exc).__name__,
                })
            except OSError:
                pass
        raise
    finally:
        torch.set_num_threads(original_threads)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="fixture:tiny-decoder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, default="full")
    parser.add_argument("--development-no-manifest", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_validation(
            args.model,
            device=args.device,
            output=args.output,
            scope=args.scope,
            development_no_manifest=args.development_no_manifest,
        )
        print(json.dumps({
            "status": report["status"],
            "scope": report["scope"],
            "report_digest": report["report_digest"],
            "formal_evidence_eligible": report["formal_evidence_eligible"],
            "output": str(args.output),
        }, sort_keys=True))
        return 0 if report["status"] == "pass" else EXIT_CODE[ErrorCode.ORACLE_DIVERGED]
    except Exception as exc:
        error = as_illm_error(exc)
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


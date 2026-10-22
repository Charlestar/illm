"""Actual CPU-only C -> explicit A transition tests; no old bootstrap/GPU."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import pytest

from illm.capture.policy_streaming import PolicyStreamingCaptureBundle
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.stateful_policy_rollback import (
    CANDIDATE_REJECTED_REQUEST,
    EXPECTED_INPUT_TOKEN_IDS,
    EXPECTED_OUTPUT_TOKEN_IDS,
    LEGACY_REJECTED_REQUEST,
    RESOURCE_FIELDS,
    _fence_facts,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
producer = importlib.import_module("rehearse_policy_rollback")


@pytest.fixture(scope="module")
def actual_transition(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path_factory.mktemp("actual-policy-rollback-transition")
    candidate = producer._run_one_policy(output=root / "candidate-c", candidate=True)
    legacy = producer._run_one_policy(output=root / "legacy-a", candidate=False)
    return root, candidate, legacy


def test_actual_candidate_c_reaches_fenced_zero_resource_close(actual_transition: tuple[Path, dict[str, Any], dict[str, Any]]) -> None:
    root, segment, _ = actual_transition
    result = segment["result"]
    bundle = PolicyStreamingCaptureBundle.open(root / "candidate-c")
    assert result["schema_version"] == 3
    assert result["numerical_policy_digest"] == candidate_policy("fp32").digest
    assert result["input_token_ids"] == EXPECTED_INPUT_TOKEN_IDS
    assert result["output_token_ids"] == EXPECTED_OUTPUT_TOKEN_IDS
    assert result["query_tokens_per_step"] == [6, 1, 1, 1, 1, 1, 1, 1]
    assert result["kv_valid_tokens_per_step"] == list(range(6, 14))
    for name in ("resource_status",):
        assert set(result[name]) == RESOURCE_FIELDS and not any(result[name].values())
    assert not any(segment["engine_status_before_close"].values())
    assert segment["engine_close_returned"] is True
    assert not any(segment["engine_status_after_close"].values())
    assert segment["post_close_admission"] == {
        "request_id": CANDIDATE_REJECTED_REQUEST,
        "accepted": False,
        "error_code": "INVALID_ARGUMENT",
        "reason": "engine is closed",
    }
    assert _fence_facts(bundle, candidate=True) == {
        "writes_started": 8,
        "writes_committed": 8,
        "writes_aborted": 0,
        "releases": 1,
        "all_fences_completed": True,
        "terminal_physical_cleanup_complete": True,
        "terminal_active_leases": 0,
        "terminal_pending_writes": 0,
    }
    assert not (root / "candidate-c-must-not-exist").exists()


def test_actual_explicit_legacy_none_is_a_new_full_context_request(actual_transition: tuple[Path, dict[str, Any], dict[str, Any]]) -> None:
    root, candidate, segment = actual_transition
    result = segment["result"]
    bundle = PolicyStreamingCaptureBundle.open(root / "legacy-a")
    assert result["schema_version"] == 3
    assert result["numerical_policy_digest"] == legacy_policy("fp32").digest
    assert result["kv_mode"] == "none"
    assert result["input_token_ids"] == EXPECTED_INPUT_TOKEN_IDS
    assert result["output_token_ids"] == EXPECTED_OUTPUT_TOKEN_IDS
    assert result["query_tokens_per_step"] == list(range(6, 14))
    assert result["kv_valid_tokens_per_step"] == [0] * 8
    assert bundle.manifest["memory_plan"]["sidecar_bytes"] == 0
    assert bundle.manifest["memory_plan"]["kv_bytes"] == 0
    assert not any(result["resource_status"].values())
    assert not any(segment["engine_status_before_close"].values())
    assert segment["engine_close_returned"] is True
    assert not any(segment["engine_status_after_close"].values())
    assert segment["post_close_admission"] == {
        "request_id": LEGACY_REJECTED_REQUEST,
        "accepted": False,
        "error_code": "INVALID_ARGUMENT",
        "reason": "engine is closed",
    }
    assert _fence_facts(bundle, candidate=False)["writes_started"] == 0
    assert candidate["result"]["input_token_ids"] == result["input_token_ids"]
    assert candidate["result"]["output_token_ids"] == result["output_token_ids"]
    assert candidate["result"]["text"] == result["text"]
    assert not (root / "legacy-a-must-not-exist").exists()

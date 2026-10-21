"""Actual CPU B/C/sampler/resource tests, never a GPU qualification surrogate."""

from __future__ import annotations

import json
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from illm.executor import EagerExecutor, ImmediateFence
from illm.memory.policy import build_policy_memory_plan
from illm.model import LoadedModel, load_model
from illm.model.policy import PolicyDecoder
from illm.model.spec import ModelError
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample


ROOT = Path(__file__).resolve().parents[2]
INPUTS = json.loads((ROOT / "docs/implementation/v0.2-policy-inputs-draft.json").read_text(encoding="utf-8"))
CASES = INPUTS["suites"][1]["cases"]
GOLDENS = json.loads((ROOT / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_text(encoding="utf-8"))["greedy_cases"]


@pytest.fixture(scope="module")
def loaded() -> Generator[LoadedModel, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield load_model(project_root=ROOT)
    torch.set_num_threads(previous)


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_frozen_fifty_b_c_logits_sampling_state_and_resources(loaded: LoadedModel, case: dict[str, Any]) -> None:
    policy = candidate_policy("fp32")
    b = EagerExecutor(loaded, "none", 256, numerical_policy=policy)
    c = EagerExecutor(loaded, "contiguous", 256, numerical_policy=policy)
    context = tuple(case["input_token_ids"])
    sampling = SamplingSpec(**case["sampling_spec"])
    b_state = SamplingState(case["case_id"], sampling.digest)
    c_state = SamplingState(case["case_id"], sampling.digest)
    try:
        for index in range(case["max_new_tokens"]):
            b_ticket, c_ticket = b.launch(context), c.launch(context)
            b_logits, c_logits = b.resolve(b_ticket), c.resolve(c_ticket)
            torch.testing.assert_close(torch.tensor(c_logits), torch.tensor(b_logits), atol=1e-4, rtol=1e-3)
            b_decision = prepare_sample(b_logits, sampling, b_state)
            c_decision = prepare_sample(c_logits, sampling, c_state)
            assert b_decision.token_id == c_decision.token_id
            assert b_ticket.query_tokens == len(context)
            assert c_ticket.query_tokens == (len(context) if index == 0 else 1)
            assert c_ticket.before_valid_tokens == (0 if index == 0 else len(context) - 1)
            assert b.commit(b_ticket) == 0
            assert c.commit(c_ticket) == len(context)
            b_state = commit_sample(b_state, b_decision, spec=sampling, logits=b_logits)
            c_state = commit_sample(c_state, c_decision, spec=sampling, logits=c_logits)
            assert b_state == c_state
            assert b_state.next_output_position == index + 1
            assert b_state.rng_draws_consumed == (0 if sampling.mode == "greedy" else index + 1)
            context += (b_decision.token_id,)
            if b_decision.token_id == loaded.spec.eos_token_id:
                break
    finally:
        b.close()
        c.close()
    assert set(asdict(b.resource_status).values()) == {0}
    assert set(asdict(c.resource_status).values()) == {0}


@pytest.mark.parametrize("case", GOLDENS, ids=[case["fixture_id"] for case in GOLDENS])
def test_complete_frozen_v01_greedy_sequence_still_matches_b_and_c(loaded: LoadedModel, case: dict[str, Any]) -> None:
    executors = [EagerExecutor(loaded, mode, 256, numerical_policy=candidate_policy("fp32"))
                 for mode in ("none", "contiguous")]
    contexts = [tuple(case["input_ids"]), tuple(case["input_ids"])]
    sampling = SamplingSpec(mode="greedy", actual_seed=7)
    states = [SamplingState(case["fixture_id"], sampling.digest) for _ in executors]
    observed: list[list[int]] = [[], []]
    try:
        for index, executor in enumerate(executors):
            for _ in range(8):
                ticket = executor.launch(contexts[index])
                logits = executor.resolve(ticket)
                decision = prepare_sample(logits, sampling, states[index])
                executor.commit(ticket)
                states[index] = commit_sample(states[index], decision, spec=sampling, logits=logits)
                observed[index].append(decision.token_id)
                contexts[index] += (decision.token_id,)
                if decision.token_id == loaded.spec.eos_token_id:
                    break
            assert observed[index] == case["expected_tokens"]
            assert states[index].rng_draws_consumed == 0
    finally:
        for executor in executors:
            executor.close()


def test_policy_memory_plan_accounts_for_actual_unique_mirrors(loaded: LoadedModel) -> None:
    policy = candidate_policy("fp32")
    decoder = PolicyDecoder(loaded.reference, policy)
    plan = build_policy_memory_plan(loaded, policy, "contiguous", 256, 1 << 40)
    assert plan.sidecar_bytes == decoder.sidecar_bytes == plan.base.weights_bytes
    assert decoder.canonical_sidecar("lm_head.weight") is decoder.canonical_sidecar("model.embed_tokens.weight")
    assert plan.required_bytes > plan.base.required_bytes
    assert plan.prefix_scratch_bytes > 0
    assert plan.to_dict()["schema_version"] == 2
    assert plan.to_dict()["numerical_policy_digest"] == policy.digest
    assert plan.base.to_dict()["schema_version"] == 1
    insufficient = build_policy_memory_plan(loaded, policy, "none", 256, plan.base.required_bytes)
    with pytest.raises(ModelError, match="policy MemoryPlan"):
        insufficient.require_admission(1)


def test_explicit_legacy_none_preserves_old_executor_result(loaded: LoadedModel) -> None:
    old = EagerExecutor(loaded, "none", 256)
    explicit = EagerExecutor(loaded, "none", 256, numerical_policy=legacy_policy("fp32"))
    try:
        a, b = old.launch((1, 107)), explicit.launch((1, 107))
        assert old.resolve(a) == explicit.resolve(b)
        old.commit(a)
        explicit.commit(b)
    finally:
        old.close()
        explicit.close()


def test_policy_aborted_write_does_not_become_committed_cache(loaded: LoadedModel) -> None:
    executor = EagerExecutor(loaded, "contiguous", 256, numerical_policy=candidate_policy("fp32"))
    try:
        ticket = executor.launch((3, 4))
        assert executor.resource_status.kv_staged_tokens == 2
        executor.abort(ticket)
        assert executor.resource_status.kv_valid_tokens == 0
        assert executor.resource_status.kv_staged_tokens == 0
        retry = executor.launch((5,))
        assert retry.before_valid_tokens == 0
        executor.resolve(retry)
        executor.commit(retry)
    finally:
        executor.close()
    assert executor.resource_status.active_tickets == 0
    assert ImmediateFence().is_complete() is True

"""Canonical sampler known answers, numeric boundaries and commit ownership."""

from __future__ import annotations

import math
import json
import random
import struct
import unittest
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Sequence

from illm.sampling import (
    CounterAddress, SamplingDecision, SamplingError, SamplingSpec, SamplingState,
    commit_sample, fp32, philox4x32_10, philox_draw, prepare_sample, select_cdf,
)


def initial(spec: SamplingSpec, request_id: str = "test-request") -> SamplingState:
    return SamplingState.for_request(request_id, spec)


def propose(logits: Sequence[float], spec: SamplingSpec) -> SamplingDecision:
    return prepare_sample(logits, spec, initial(spec))


class PhiloxKnownAnswerTests(unittest.TestCase):
    # Independent published source (Random123, tests/kat_vectors):
    # https://github.com/DEShawResearch/random123/blob/v1.14.0/tests/kat_vectors
    def test_random123_all_zero(self) -> None:
        self.assertEqual(philox4x32_10([0] * 4, [0] * 2),
                         (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8))

    def test_random123_all_ones(self) -> None:
        self.assertEqual(philox4x32_10([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2),
                         (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD))

    def test_random123_digits_of_pi(self) -> None:
        self.assertEqual(
            philox4x32_10((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
                         (0xA4093822, 0x299F31D0)),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        )

    def test_little_endian_identity_and_lane_zero(self) -> None:
        address = CounterAddress(0x1122334455667788, 0x99AABBCC, 0xDDEEFF00)
        self.assertEqual(address.to_bytes().hex(), "8877665544332211ccbbaa9900ffeedd")
        self.assertEqual(address.to_words(), (0x55667788, 0x11223344, 0x99AABBCC, 0xDDEEFF00))
        draw = philox_draw(0x0123456789ABCDEF, address)
        self.assertEqual(draw.to_dict()["key_bytes_le"], "efcdab8967452301")
        self.assertEqual(draw.uniform, draw.lanes[0] / 4294967296)
        self.assertGreaterEqual(draw.uniform, 0)
        self.assertLess(draw.uniform, 1)

    def test_counter_domain_limits(self) -> None:
        for args in [(-1,), (1 << 64,), (0, 1 << 32), (0, 0, -1), (True,)]:
            with self.subTest(args=args), self.assertRaises(SamplingError):
                CounterAddress(*args)
        with self.assertRaises(SamplingError):
            philox_draw(1 << 64, CounterAddress(0))


class SamplingContractTests(unittest.TestCase):
    def test_spec_is_frozen_and_canonical(self) -> None:
        first = SamplingSpec()
        second = SamplingSpec(temperature=0, top_p=1)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(SamplingSpec.from_dict(first.to_dict()), first)
        self.assertEqual(SamplingSpec(temperature=-0.0).digest, first.digest)
        with self.assertRaises(FrozenInstanceError):
            first.temperature = 1  # type: ignore[misc]

    def test_invalid_spec_fails_closed(self) -> None:
        cases: list[dict[str, Any]] = [
            {"mode": "other"}, {"mode": []}, {"temperature": 1}, {"top_k": 1}, {"top_p": 0.9},
            {"mode": "sample", "temperature": 0}, {"mode": "sample", "temperature": -1},
            {"mode": "sample", "temperature": math.inf}, {"top_p": 0}, {"top_p": 1.1},
            {"top_p": math.nan}, {"top_k": -1}, {"top_k": True}, {"temperature": True},
            {"actual_seed": -1}, {"actual_seed": 1 << 64}, {"actual_seed": True},
            {"schema_version": True}, {"processor_order_version": 2}, {"rng_contract_version": 2},
            {"compute_dtype": "bf16"}, {"rng_algorithm": "global-rng"},
            {"temperature": 1 << 10000},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(SamplingError):
                SamplingSpec(**kwargs)
        with self.assertRaises(SamplingError) as raised:
            SamplingSpec.from_dict({"repetition_penalty": 1.1})
        self.assertEqual(raised.exception.code, "UNSUPPORTED_SAMPLING_PARAMETER")

    def test_raw_nonfinite_is_rejected_before_greedy_or_masks(self) -> None:
        for value in [math.nan, math.inf, -math.inf, 1e100]:
            for spec in [SamplingSpec(), SamplingSpec(mode="sample", temperature=1, top_k=1)]:
                with self.subTest(value=value, spec=spec), self.assertRaises(SamplingError) as raised:
                    propose([0, value], spec)
                self.assertEqual(raised.exception.code, "NUMERICAL_ERROR")
        with self.assertRaises(SamplingError):
            propose([], SamplingSpec())

    def test_fp32_rounding_and_greedy_tie(self) -> None:
        # Values distinct in binary64 tie after the required binary32 conversion.
        self.assertEqual(fp32(1.0 + 2.0 ** -25), 1.0)
        logits, spec = [1.0, 1.0 + 2.0 ** -25, -2.0], SamplingSpec()
        result = propose(logits, spec)
        self.assertEqual(result.token_id, 0)
        self.assertIsNone(result.draw)
        state = commit_sample(initial(spec), result, spec=spec, logits=logits)
        self.assertEqual(state, SamplingState("test-request", spec.digest, next_output_position=1))

    def test_top_k_score_ties_and_large_k(self) -> None:
        spec = SamplingSpec(mode="sample", temperature=1, top_k=2)
        decision = propose([2, 2, 2, -1], spec)
        self.assertEqual(decision.top_k_ids_by_score, (0, 1))
        self.assertEqual(decision.retained_probabilities_by_token, ((0, 0.5), (1, 0.5)))
        large = propose([2, 1], SamplingSpec(mode="sample", temperature=1, top_k=200))
        self.assertEqual(large.top_k_ids_by_score, (0, 1))

    def test_top_p_exact_boundary_uses_minimal_nonempty_prefix(self) -> None:
        expected = [(0.25, (0,)), (0.5, (0, 1)), (0.75, (0, 1, 2)), (1.0, (0, 1, 2, 3))]
        for top_p, retained in expected:
            with self.subTest(top_p=top_p):
                decision = propose([0] * 4, SamplingSpec(mode="sample", temperature=1, top_p=top_p))
                self.assertEqual(decision.top_p_ids_by_score, retained)
                self.assertEqual(decision.cdf_by_token[-1][1], 1.0)
        decision = propose([0] * 4, SamplingSpec(mode="sample", temperature=1, top_p=math.nextafter(0.5, 1)))
        self.assertEqual(decision.top_p_ids_by_score, (0, 1, 2))

    def test_cdf_is_token_order_not_probability_order(self) -> None:
        decision = propose([0, 3, 2, 1], SamplingSpec(mode="sample", temperature=1, top_p=0.9))
        self.assertEqual(decision.top_p_ids_by_score, (1, 2, 3))
        self.assertEqual(tuple(index for index, _ in decision.cdf_by_token), (1, 2, 3))
        decision = propose([2, 3, 1, 0], SamplingSpec(mode="sample", temperature=1, top_p=0.9))
        self.assertEqual(decision.top_p_ids_by_score, (1, 0, 2))
        self.assertEqual(tuple(index for index, _ in decision.cdf_by_token), (0, 1, 2))

    def test_half_open_cdf_and_zero_probability(self) -> None:
        cdf = [(0, 0), (1, 0.25), (2, 0.25), (3, 1.0), (4, 1.0)]
        self.assertEqual(select_cdf(cdf, 0), 1)
        self.assertEqual(select_cdf(cdf, math.nextafter(0.25, 0)), 1)
        self.assertEqual(select_cdf(cdf, 0.25), 3)
        self.assertEqual(select_cdf(cdf, math.nextafter(1.0, 0)), 3)
        for invalid in [math.nan, math.inf, -1e-10, 1, True]:
            with self.subTest(invalid=invalid), self.assertRaises(SamplingError):
                select_cdf(cdf, invalid)

    def test_underflow_does_not_remove_top_p_one_candidates(self) -> None:
        decision = propose([0, -200, -300], SamplingSpec(mode="sample", temperature=1))
        self.assertEqual(decision.top_p_ids_by_score, (0, 1, 2))
        self.assertEqual(decision.retained_probabilities_by_token, ((0, 1), (1, 0), (2, 0)))
        self.assertEqual(decision.token_id, 0)
        self.assertEqual(decision.cdf_by_token, ((0, 1), (1, 1), (2, 1)))

    def test_full_eight_token_fixture_stage_bits(self) -> None:
        decision = propose([2, -1, 2, 0, 0.5, -3, 1, 2], SamplingSpec(mode="sample", temperature=1, top_k=5, top_p=0.8, actual_seed=0))
        self.assertEqual(decision.top_k_ids_by_score, (0, 2, 7, 6, 4))
        self.assertEqual(decision.top_p_ids_by_score, (0, 2, 7))
        self.assertEqual(decision.token_id, 2)  # Random123 lane0 / 2**32 is ~0.399.
        self.assertEqual(decision.draw.lanes if decision.draw else None,
                         (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8))
        self.assertEqual([struct.pack("<f", value).hex() for _, value in decision.weights_by_token],
                         ["0000803f", "0000803f", "3c7c643e", "b25abc3e", "0000803f"])
        self.assertEqual(decision.cdf_by_token, ((0, 1 / 3), (2, 2 / 3), (7, 1)))

    def test_frozen_fixture_includes_every_stage_and_commit_abort(self) -> None:
        fixture = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "sampling" / "sampler-known-answer-v1.json").read_text())
        spec = SamplingSpec.from_dict(fixture["sampling_spec"])
        state = initial(spec, "fixture-eight")
        decision = prepare_sample(fixture["logits"], spec, state)
        self.assertEqual(json.loads(json.dumps(decision.to_dict())), fixture["decision"])
        self.assertEqual(commit_sample(state, decision, spec=spec, logits=fixture["logits"]).to_dict(), fixture["committed_state"])
        self.assertEqual(state.to_dict(), fixture["abort_state"])

    def test_exp_primitive_against_independent_high_precision_decimal(self) -> None:
        # Decimal.exp is correctly rounded in a separate arithmetic library.
        # The frozen inputs cover exact weights, ordinary weights and underflow.
        for value in [0, -0.5, -1, -1.5, -10, -80, -100, -200]:
            with self.subTest(value=value), localcontext() as context:
                context.prec = 100
                expected = fp32(float(Decimal(value).exp()))
                decision = propose([0, value], SamplingSpec(mode="sample", temperature=1))
                self.assertEqual(decision.weights_by_token[1][1], expected)

    def test_temperature_numerical_overflow_is_explicit(self) -> None:
        with self.assertRaises(SamplingError) as raised:
            propose([1, 2], SamplingSpec(mode="sample", temperature=1e-300))
        self.assertEqual(raised.exception.code, "NUMERICAL_ERROR")

    def test_temperature_is_binary64_scalar_before_binary32_score_round(self) -> None:
        decision = propose([0, 1], SamplingSpec(mode="sample", temperature=1.00000003))
        self.assertEqual(decision.temperature_scores_by_token[1][1], 0.9999999403953552)
        self.assertNotEqual(decision.temperature_scores_by_token[1][1], fp32(1 / fp32(1.00000003)))

    def test_valid_cdf_rounding_overshoot_saturates_prefix_not_validation(self) -> None:
        decision = propose([0, 1, -3, -45], SamplingSpec(mode="sample", temperature=1))
        self.assertEqual(decision.cdf_by_token[-2:], ((2, 1.0), (3, 1.0)))
        self.assertTrue(all(0 <= value <= 1 for _, value in decision.cdf_by_token))
        with self.assertRaises(SamplingError):
            select_cdf([(0, 1.0000000000000002), (1, 1.0)], 0.2)

    def test_malformed_cdf_yields_structured_error(self) -> None:
        malformed: list[Any] = [[(0, True)], [(0, "bad"), (1, 1.0)], [(0,)],
                                [(0, 0.5, 1), (1, 1.0)], [42], None,
                                [(0, 1 << 10000), (1, 1.0)], [(True, 1.0)]]
        for cdf in malformed:
            with self.subTest(cdf=cdf), self.assertRaises(SamplingError):
                select_cdf(cdf, 0.2)


class SamplingTransactionTests(unittest.TestCase):
    def test_discarded_proposal_consumes_no_draw_and_retry_is_identical(self) -> None:
        spec = SamplingSpec(mode="sample", temperature=1, actual_seed=7)
        state = initial(spec)
        first = prepare_sample([1, 2, 3], spec, state)
        retry = prepare_sample([1, 2, 3], spec, state)
        self.assertEqual(first, retry)
        self.assertEqual(state, initial(spec))
        state = commit_sample(state, retry, spec=spec, logits=[1, 2, 3])
        self.assertEqual(state.rng_draws_consumed, 1)
        self.assertEqual(state.committed_draw_addresses, (CounterAddress(0),))
        with self.assertRaises(SamplingError):
            commit_sample(state, retry, spec=spec, logits=[1, 2, 3])

    def test_greedy_and_sample_positions_have_separate_draw_count(self) -> None:
        greedy = SamplingSpec()
        sample = SamplingSpec(mode="sample", temperature=1, actual_seed=7)
        for spec in [greedy, sample]:
            state = initial(spec)
            for _ in range(5):
                state = commit_sample(state, prepare_sample([0, 1], spec, state), spec=spec, logits=[0, 1])
            self.assertEqual(state.next_output_position, 5)
            self.assertEqual(state.rng_draws_consumed, 0 if spec.mode == "greedy" else 5)
            self.assertEqual(state.committed_draw_addresses, () if spec.mode == "greedy" else tuple(CounterAddress(i) for i in range(5)))

    def test_invalid_state_rejects_duplicate_or_future_address(self) -> None:
        for args in [(1, 1, ()), (1, 1, (CounterAddress(1),)), (2, 2, (CounterAddress(0), CounterAddress(0))),
                     (3, 2, (CounterAddress(2), CounterAddress(0)))]:
            with self.subTest(args=args), self.assertRaises(SamplingError):
                SamplingState("test-request", SamplingSpec().digest, *args)

    def test_commit_rejects_replaced_token_spec_seed_lanes_and_probabilities(self) -> None:
        spec = SamplingSpec(mode="sample", temperature=1, actual_seed=7)
        state = initial(spec)
        logits = [0, 2, 1]
        decision = prepare_sample(logits, spec, state)
        assert decision.draw is not None
        for invalid in [replace(decision, token_id=-999), replace(decision, token_id=True),
                        replace(decision, spec_digest="forged"),
                        replace(decision, draw=replace(decision.draw, seed=8)),
                        replace(decision, draw=replace(decision.draw, lanes=(1, 2, 3, 4))),
                        replace(decision, draw=replace(decision.draw, uniform=0.25)),
                        replace(decision, cdf_by_token=((0, 1.0),))]:
            with self.subTest(decision=invalid), self.assertRaises(SamplingError):
                commit_sample(state, invalid, spec=spec, logits=logits)

    def test_request_and_spec_binding_and_trusted_logits(self) -> None:
        spec = SamplingSpec(mode="sample", temperature=1, actual_seed=7)
        state = initial(spec, "request-a")
        logits = [0, 2, 1]
        decision = prepare_sample(logits, spec, state)
        with self.assertRaises(SamplingError):
            commit_sample(initial(spec, "request-b"), decision, spec=spec, logits=logits)
        with self.assertRaises(SamplingError):
            commit_sample(state, decision, spec=replace(spec, actual_seed=8), logits=logits)
        with self.assertRaises(SamplingError):
            commit_sample(state, decision, spec=spec, logits=[0, 2, 1.1])
        with self.assertRaises(SamplingError):
            prepare_sample(logits, replace(spec, temperature=2), state)

    def test_exhausted_output_counter_rejected_before_proposal(self) -> None:
        spec = SamplingSpec()
        state = SamplingState("exhausted", spec.digest, next_output_position=(1 << 64) - 1)
        with self.assertRaises(SamplingError):
            prepare_sample([0, 1], spec, state)

    def test_interleaving_other_requests_and_global_rng_changes_nothing(self) -> None:
        spec = SamplingSpec(mode="sample", temperature=0.7, top_k=5, top_p=0.85, actual_seed=1234)
        def run(interleave: bool) -> tuple[list[int], SamplingState]:
            state = initial(spec)
            output: list[int] = []
            for position in range(30):
                decision = prepare_sample([position % 3, 1, -1, 2, 0, 3], spec, state)
                if interleave:
                    random.seed(position)
                    random.random()
                    other = SamplingSpec(mode="sample", temperature=1, actual_seed=position)
                    other_state = initial(other, "other-request")
                    for _ in range(5):
                        other_decision = prepare_sample([1, 3, 2], other, other_state)
                        other_state = commit_sample(other_state, other_decision, spec=other, logits=[1, 3, 2])
                state = commit_sample(state, decision, spec=spec, logits=[position % 3, 1, -1, 2, 0, 3])
                output.append(decision.token_id)
            return output, state
        self.assertEqual(run(False), run(True))

    def test_uniform_distribution_regression_is_broad_not_a_proof(self) -> None:
        # Bounds are frozen before running and deliberately broad (about 7 sigma).
        # The implementation has no vectorized path requiring a differential suite.
        counts = [0] * 4
        for position in range(4096):
            draw = philox_draw(11, CounterAddress(position))
            counts[select_cdf([(0, 0.25), (1, 0.5), (2, 0.75), (3, 1.0)], draw.uniform)] += 1
        self.assertTrue(all(800 <= count <= 1250 for count in counts), counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)

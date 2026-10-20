from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence, cast

from illm.metrics.registry import METRIC_IDS, metric_registry, registry_digest


TERMINAL_OUTCOMES = ("FINISHED", "CANCELLED", "FAILED", "DEADLINE_EXCEEDED")
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class RequestObservation:
    """One offered request's immutable server/client timing facts.

    Server timestamps all belong to ``server_clock_domain``. Client timestamps
    all belong to ``client_clock_domain``. The builder never subtracts a
    server timestamp from a client timestamp.
    """

    request_id: str
    accepted_ns: int | None
    token_ready_ns: tuple[int, ...]
    terminal_ready_ns: int | None
    outcome: str | None
    client_send_ns: int | None
    client_terminal_ns: int | None
    preaccept_reason: str | None
    server_clock_domain: str | None = None
    client_clock_domain: str | None = None


def _timestamp(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer nanosecond timestamp")
    return value


def _clock_domain(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_observation(observation: RequestObservation) -> bool:
    if type(observation.request_id) is not str or not observation.request_id:
        raise ValueError("request_id must be a non-empty string")
    if type(observation.token_ready_ns) is not tuple:
        raise ValueError("token_ready_ns must be a tuple")
    if observation.accepted_ns is None:
        if type(observation.preaccept_reason) is not str or not observation.preaccept_reason:
            raise ValueError("pre-accept observations require a non-empty preaccept_reason")
        if (
            observation.token_ready_ns
            or observation.terminal_ready_ns is not None
            or observation.outcome is not None
            or observation.client_send_ns is not None
            or observation.client_terminal_ns is not None
            or observation.server_clock_domain is not None
            or observation.client_clock_domain is not None
        ):
            raise ValueError("pre-accept observations cannot contain accepted or latency facts")
        return False

    if observation.preaccept_reason is not None:
        raise ValueError("accepted observations cannot contain a preaccept_reason")
    accepted_ns = _timestamp("accepted_ns", observation.accepted_ns)
    _clock_domain("server_clock_domain", observation.server_clock_domain)
    if observation.terminal_ready_ns is None:
        raise ValueError("accepted observations require terminal_ready_ns")
    if type(observation.outcome) is not str or observation.outcome not in TERMINAL_OUTCOMES:
        raise ValueError("accepted observations require one recognized terminal outcome")
    if observation.outcome == "DEADLINE_EXCEEDED":
        raise ValueError("V0.2 reserves DEADLINE_EXCEEDED and requires its count to remain zero")

    previous = accepted_ns
    for index, raw_timestamp in enumerate(observation.token_ready_ns):
        timestamp = _timestamp(f"token_ready_ns[{index}]", raw_timestamp)
        if timestamp <= previous:
            raise ValueError("server timestamps must be strictly increasing within one clock domain")
        previous = timestamp
    terminal_ns = _timestamp("terminal_ready_ns", observation.terminal_ready_ns)
    if terminal_ns <= previous:
        raise ValueError("terminal_ready_ns must follow accepted and token timestamps")

    client_values = (observation.client_send_ns, observation.client_terminal_ns)
    if client_values == (None, None):
        if observation.client_clock_domain is not None:
            raise ValueError("client_clock_domain requires a complete client timestamp pair")
    elif None in client_values:
        raise ValueError("client timestamps must be both present or both absent")
    else:
        _clock_domain("client_clock_domain", observation.client_clock_domain)
        client_send_ns = _timestamp("client_send_ns", observation.client_send_ns)
        client_terminal_ns = _timestamp("client_terminal_ns", observation.client_terminal_ns)
        if client_terminal_ns <= client_send_ns:
            raise ValueError("client timestamps must be strictly increasing within one clock domain")
    return True


def _p99(values: Sequence[Fraction]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (99 * len(ordered) + 99) // 100
    return float(ordered[rank - 1] / _NS_PER_SECOND)


def _metric_definitions() -> dict[str, dict[str, object]]:
    metrics = cast(list[dict[str, object]], metric_registry()["metrics"])
    return {cast(str, item["id"]): item for item in metrics}


def _metric_result(metric_id: str, samples: Sequence[Fraction]) -> dict[str, object]:
    definitions = _metric_definitions()
    definition = definitions[metric_id]
    return {
        "unit": definition["unit"],
        "sample_count": len(samples),
        "p99": _p99(samples),
    }


def _outcome_metric_result(
    metric_id: str,
    samples: Sequence[Fraction],
    by_outcome: Mapping[str, Sequence[Fraction]],
) -> dict[str, object]:
    result = _metric_result(metric_id, samples)
    result["by_outcome"] = {
        outcome: {"sample_count": len(by_outcome[outcome]), "p99": _p99(by_outcome[outcome])}
        for outcome in TERMINAL_OUTCOMES
    }
    return result


def build_streaming_metrics(observations: Iterable[RequestObservation]) -> dict[str, object]:
    """Build the V0.2 five-metric streaming report from immutable facts."""

    accepted = 0
    preaccept = 0
    request_ids: set[str] = set()
    terminal_outcomes = {outcome: 0 for outcome in TERMINAL_OUTCOMES}
    preaccept_reasons: dict[str, int] = {}
    counters = {
        "ttft_censored_before_first_token": 0,
        "itl_censored_insufficient_output": 0,
        "tpot_censored_insufficient_output": 0,
        "client_e2e_missing": 0,
    }
    ttft: list[Fraction] = []
    itl: list[Fraction] = []
    tpot: list[Fraction] = []
    server_e2e: list[Fraction] = []
    client_e2e: list[Fraction] = []
    server_by_outcome: dict[str, list[Fraction]] = {outcome: [] for outcome in TERMINAL_OUTCOMES}
    client_by_outcome: dict[str, list[Fraction]] = {outcome: [] for outcome in TERMINAL_OUTCOMES}

    for observation in observations:
        if type(observation) is not RequestObservation:
            raise ValueError("observations must contain RequestObservation values")
        if observation.request_id in request_ids:
            raise ValueError(f"duplicate request_id: {observation.request_id}")
        request_ids.add(observation.request_id)
        is_accepted = _validate_observation(observation)
        if not is_accepted:
            preaccept += 1
            assert observation.preaccept_reason is not None
            preaccept_reasons[observation.preaccept_reason] = preaccept_reasons.get(observation.preaccept_reason, 0) + 1
            continue

        accepted += 1
        assert observation.accepted_ns is not None
        assert observation.terminal_ready_ns is not None
        assert observation.outcome is not None
        outcome = observation.outcome
        terminal_outcomes[outcome] += 1
        tokens = observation.token_ready_ns
        if not tokens:
            counters["ttft_censored_before_first_token"] += 1
        else:
            ttft.append(Fraction(tokens[0] - observation.accepted_ns))
        if len(tokens) < 2:
            counters["itl_censored_insufficient_output"] += 1
            counters["tpot_censored_insufficient_output"] += 1
        else:
            itl.extend(Fraction(current - previous) for previous, current in zip(tokens, tokens[1:]))
            tpot.append(Fraction(tokens[-1] - tokens[0], len(tokens) - 1))

        server_sample = Fraction(observation.terminal_ready_ns - observation.accepted_ns)
        server_e2e.append(server_sample)
        server_by_outcome[outcome].append(server_sample)
        if observation.client_send_ns is None:
            counters["client_e2e_missing"] += 1
        else:
            assert observation.client_terminal_ns is not None
            client_sample = Fraction(observation.client_terminal_ns - observation.client_send_ns)
            client_e2e.append(client_sample)
            client_by_outcome[outcome].append(client_sample)

    metrics = {
        METRIC_IDS[0]: _metric_result(METRIC_IDS[0], ttft),
        METRIC_IDS[1]: _metric_result(METRIC_IDS[1], itl),
        METRIC_IDS[2]: _metric_result(METRIC_IDS[2], tpot),
        METRIC_IDS[3]: _outcome_metric_result(METRIC_IDS[3], server_e2e, server_by_outcome),
        METRIC_IDS[4]: _outcome_metric_result(METRIC_IDS[4], client_e2e, client_by_outcome),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "registry_digest": registry_digest(),
        "request_totals": {
            "offered": accepted + preaccept,
            "accepted": accepted,
            "preaccept": preaccept,
        },
        "terminal_outcomes": terminal_outcomes,
        "preaccept_reasons": dict(sorted(preaccept_reasons.items())),
        "counters": counters,
        "metrics": metrics,
    }
    validate_streaming_metrics(result)
    return result


def _count(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_metric_value(name: str, value: object) -> None:
    if value is None:
        return
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive SI-second float or null")


def validate_streaming_metrics(value: Mapping[str, object]) -> None:
    """Validate report accounting without accepting short IDs or implicit zeroes."""

    expected_fields = {
        "schema_version",
        "registry_digest",
        "request_totals",
        "terminal_outcomes",
        "preaccept_reasons",
        "counters",
        "metrics",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("streaming metric report fields do not match schema v1")
    if value["schema_version"] != 1 or value["registry_digest"] != registry_digest():
        raise ValueError("streaming metric report registry identity is invalid")

    totals = value["request_totals"]
    outcomes = value["terminal_outcomes"]
    reasons = value["preaccept_reasons"]
    counters = value["counters"]
    metrics = value["metrics"]
    if not isinstance(totals, Mapping) or set(totals) != {"offered", "accepted", "preaccept"}:
        raise ValueError("request totals are invalid")
    offered = _count("request_totals.offered", totals["offered"])
    accepted = _count("request_totals.accepted", totals["accepted"])
    preaccept = _count("request_totals.preaccept", totals["preaccept"])
    if offered != accepted + preaccept:
        raise ValueError("offered requests must equal accepted plus pre-accept requests")

    if not isinstance(outcomes, Mapping) or set(outcomes) != set(TERMINAL_OUTCOMES):
        raise ValueError("terminal outcome strata are invalid")
    outcome_counts = {outcome: _count(f"terminal_outcomes.{outcome}", outcomes[outcome]) for outcome in TERMINAL_OUTCOMES}
    if sum(outcome_counts.values()) != accepted or outcome_counts["DEADLINE_EXCEEDED"] != 0:
        raise ValueError("terminal outcome accounting is incomplete or enables V0.2 deadlines")

    if not isinstance(reasons, Mapping):
        raise ValueError("preaccept_reasons must be an object")
    reason_total = 0
    for reason, count in reasons.items():
        if type(reason) is not str or not reason:
            raise ValueError("pre-accept reasons must be non-empty strings")
        parsed_count = _count(f"preaccept_reasons.{reason}", count)
        if parsed_count == 0:
            raise ValueError("pre-accept reason entries must have positive counts")
        reason_total += parsed_count
    if reason_total != preaccept:
        raise ValueError("pre-accept reason accounting is incomplete")

    counter_names = {
        "ttft_censored_before_first_token",
        "itl_censored_insufficient_output",
        "tpot_censored_insufficient_output",
        "client_e2e_missing",
    }
    if not isinstance(counters, Mapping) or set(counters) != counter_names:
        raise ValueError("streaming metric censor counters are invalid")
    parsed_counters = {name: _count(f"counters.{name}", counters[name]) for name in counter_names}
    if any(count > accepted for count in parsed_counters.values()):
        raise ValueError("streaming metric censor counters exceed accepted requests")
    if parsed_counters["itl_censored_insufficient_output"] != parsed_counters["tpot_censored_insufficient_output"]:
        raise ValueError("ITL and TPOT insufficient-output censor counts must agree")
    if parsed_counters["ttft_censored_before_first_token"] > parsed_counters["tpot_censored_insufficient_output"]:
        raise ValueError("TTFT censors must be a subset of insufficient-output censors")

    definitions = _metric_definitions()
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_IDS):
        raise ValueError("streaming metrics must contain the five fully-qualified IDs")
    sample_counts: dict[str, int] = {}
    for metric_id in METRIC_IDS:
        metric = metrics[metric_id]
        expected_metric_fields = {"unit", "sample_count", "p99"}
        if definitions[metric_id]["outcome_stratified"]:
            expected_metric_fields.add("by_outcome")
        if not isinstance(metric, Mapping) or set(metric) != expected_metric_fields:
            raise ValueError(f"{metric_id} result fields are invalid")
        if metric["unit"] != definitions[metric_id]["unit"]:
            raise ValueError(f"{metric_id} unit differs from MetricRegistryV1")
        sample_count = _count(f"{metric_id}.sample_count", metric["sample_count"])
        sample_counts[metric_id] = sample_count
        _validate_metric_value(f"{metric_id}.p99", metric["p99"])
        if (sample_count == 0) != (metric["p99"] is None):
            raise ValueError(f"{metric_id} p99 nullability does not match its sample count")
        if "by_outcome" not in metric:
            continue
        strata = metric["by_outcome"]
        if not isinstance(strata, Mapping) or set(strata) != set(TERMINAL_OUTCOMES):
            raise ValueError(f"{metric_id} outcome strata are invalid")
        stratum_total = 0
        for outcome in TERMINAL_OUTCOMES:
            stratum = strata[outcome]
            if not isinstance(stratum, Mapping) or set(stratum) != {"sample_count", "p99"}:
                raise ValueError(f"{metric_id}.{outcome} result fields are invalid")
            count = _count(f"{metric_id}.{outcome}.sample_count", stratum["sample_count"])
            _validate_metric_value(f"{metric_id}.{outcome}.p99", stratum["p99"])
            if (count == 0) != (stratum["p99"] is None):
                raise ValueError(f"{metric_id}.{outcome} p99 nullability is invalid")
            if metric_id == METRIC_IDS[3] and count != outcome_counts[outcome]:
                raise ValueError("server E2E outcome strata must cover every terminal request")
            if metric_id == METRIC_IDS[4] and count > outcome_counts[outcome]:
                raise ValueError("client E2E outcome strata cannot exceed terminal outcome counts")
            stratum_total += count
        if stratum_total != sample_count:
            raise ValueError(f"{metric_id} outcome strata do not sum to its sample count")

    if sample_counts[METRIC_IDS[0]] != accepted - parsed_counters["ttft_censored_before_first_token"]:
        raise ValueError("TTFT sample/censor accounting is incomplete")
    if sample_counts[METRIC_IDS[2]] != accepted - parsed_counters["tpot_censored_insufficient_output"]:
        raise ValueError("TPOT sample/censor accounting is incomplete")
    if sample_counts[METRIC_IDS[3]] != accepted:
        raise ValueError("server E2E must cover all accepted terminal requests")
    if sample_counts[METRIC_IDS[4]] != accepted - parsed_counters["client_e2e_missing"]:
        raise ValueError("client E2E sample/missing accounting is incomplete")

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest

from illm.canonical import sha256_json
from illm.metrics import (
    METRIC_IDS,
    RequestObservation,
    build_streaming_metrics,
    metric_registry,
    registry_conformance,
    registry_digest,
    validate_metric_registry,
    validate_streaming_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
METRIC_FIXTURES = ROOT / "fixtures" / "metrics"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _accepted(request_id: str = "accepted") -> RequestObservation:
    return RequestObservation(
        request_id=request_id,
        accepted_ns=1_000_000_000,
        token_ready_ns=(1_100_000_000, 1_300_000_000),
        terminal_ready_ns=1_400_000_000,
        outcome="FINISHED",
        client_send_ns=900_000_000,
        client_terminal_ns=1_500_000_000,
        preaccept_reason=None,
        server_clock_domain="perf_counter_ns:server",
        client_clock_domain="perf_counter_ns:client",
    )


def _from_fixture(value: dict[str, Any]) -> RequestObservation:
    record = dict(value)
    record["token_ready_ns"] = tuple(record["token_ready_ns"])
    return RequestObservation(**record)


def test_registry_fixture_digest_schema_and_conformance_are_exact() -> None:
    fixture = _read_object(METRIC_FIXTURES / "metric-registry-v1.json")
    schema = _read_object(ROOT / "schemas" / "metric-registry-v1.schema.json")
    assert metric_registry() == fixture
    assert registry_digest() == sha256_json(fixture)
    assert registry_conformance(fixture) == {
        "schema_version": 1,
        "status": "pass",
        "registry_id": "illm.metric-registry.v1",
        "registry_digest": registry_digest(),
        "metric_count": 5,
        "metric_ids": list(METRIC_IDS),
    }
    prefix_items = schema["properties"]["metrics"]["prefixItems"]
    schema_ids = tuple(item["allOf"][1]["properties"]["id"]["const"] for item in prefix_items)
    assert schema_ids == METRIC_IDS

    tampered = deepcopy(fixture)
    tampered["metrics"][0]["unit"] = "ms"
    with pytest.raises(ValueError, match="frozen V0.2 contract"):
        validate_metric_registry(tampered)


def test_mixed_streaming_fixture_preserves_samples_censors_and_outcomes() -> None:
    fixture = _read_object(METRIC_FIXTURES / "streaming-mixed-v1.json")
    observations = [_from_fixture(item) for item in fixture["observations"]]
    actual = build_streaming_metrics(observations)
    assert actual["schema_version"] == 1
    assert actual["registry_digest"] == registry_digest()
    for field, expected in fixture["expected"].items():
        assert actual[field] == expected
    validate_streaming_metrics(actual)


def test_nearest_rank_p99_uses_ceil_without_interpolation() -> None:
    observations = []
    for index, duration_ns in enumerate((1, 2, 3, 4)):
        accepted_ns = index * 100
        observations.append(
            RequestObservation(
                request_id=f"p99-{index}",
                accepted_ns=accepted_ns,
                token_ready_ns=(accepted_ns + duration_ns,),
                terminal_ready_ns=accepted_ns + duration_ns + 1,
                outcome="FINISHED",
                client_send_ns=None,
                client_terminal_ns=None,
                preaccept_reason=None,
                server_clock_domain="test-server-clock",
            )
        )
    metrics = build_streaming_metrics(observations)["metrics"]
    assert isinstance(metrics, dict)
    assert metrics[METRIC_IDS[0]] == {"unit": "s", "sample_count": 4, "p99": 4e-9}


def test_server_and_client_clock_domains_are_never_cross_subtracted() -> None:
    observation = RequestObservation(
        request_id="separate-domains",
        accepted_ns=10_000_000_000,
        token_ready_ns=(11_000_000_000,),
        terminal_ready_ns=12_000_000_000,
        outcome="FINISHED",
        client_send_ns=100,
        client_terminal_ns=200,
        preaccept_reason=None,
        server_clock_domain="server-monotonic",
        client_clock_domain="client-monotonic",
    )
    metrics = build_streaming_metrics([observation])["metrics"]
    assert isinstance(metrics, dict)
    assert metrics[METRIC_IDS[3]]["p99"] == 2.0
    assert metrics[METRIC_IDS[4]]["p99"] == 1e-7


def test_missing_client_pair_is_counted_and_never_substituted_with_zero() -> None:
    observation = replace(
        _accepted(),
        client_send_ns=None,
        client_terminal_ns=None,
        client_clock_domain=None,
    )
    report = build_streaming_metrics([observation])
    counters = report["counters"]
    metrics = report["metrics"]
    assert isinstance(counters, dict)
    assert isinstance(metrics, dict)
    assert counters["client_e2e_missing"] == 1
    assert metrics[METRIC_IDS[4]]["sample_count"] == 0
    assert metrics[METRIC_IDS[4]]["p99"] is None
    assert metrics[METRIC_IDS[3]]["p99"] == 0.4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"accepted_ns": True}, "accepted_ns"),
        ({"accepted_ns": -1}, "accepted_ns"),
        ({"token_ready_ns": cast(Any, [1_100_000_000])}, "tuple"),
        ({"token_ready_ns": (cast(Any, True),)}, "token_ready_ns"),
        ({"token_ready_ns": (-1,)}, "token_ready_ns"),
        ({"token_ready_ns": (1_100_000_000, 1_100_000_000)}, "strictly increasing"),
        ({"terminal_ready_ns": None}, "terminal_ready_ns"),
        ({"terminal_ready_ns": 1_300_000_000}, "terminal_ready_ns"),
        ({"outcome": None}, "terminal outcome"),
        ({"outcome": "REJECTED"}, "terminal outcome"),
        ({"outcome": "DEADLINE_EXCEEDED"}, "requires its count to remain zero"),
        ({"server_clock_domain": None}, "server_clock_domain"),
        ({"client_terminal_ns": None}, "both present or both absent"),
        ({"client_terminal_ns": 900_000_000}, "strictly increasing"),
        ({"client_clock_domain": None}, "client_clock_domain"),
        (
            {"client_send_ns": None, "client_terminal_ns": None, "client_clock_domain": "dangling"},
            "complete client timestamp pair",
        ),
        ({"preaccept_reason": "OVERLOADED"}, "cannot contain a preaccept_reason"),
    ],
)
def test_invalid_accepted_observations_fail_closed(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_streaming_metrics([replace(_accepted(), **changes)])


@pytest.mark.parametrize(
    "changes",
    [
        {"preaccept_reason": None},
        {"token_ready_ns": (1,)},
        {"terminal_ready_ns": 1},
        {"outcome": "FAILED"},
        {"client_send_ns": 1},
        {"server_clock_domain": "server"},
    ],
)
def test_invalid_preaccept_observations_fail_closed(changes: dict[str, Any]) -> None:
    observation = RequestObservation(
        request_id="rejected",
        accepted_ns=None,
        token_ready_ns=(),
        terminal_ready_ns=None,
        outcome=None,
        client_send_ns=None,
        client_terminal_ns=None,
        preaccept_reason="SINGLE_ACTIVE_LIMIT",
    )
    with pytest.raises(ValueError, match="pre-accept"):
        build_streaming_metrics([replace(observation, **changes)])


def test_duplicate_request_ids_and_non_observation_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate request_id"):
        build_streaming_metrics([_accepted(), _accepted()])
    with pytest.raises(ValueError, match="RequestObservation"):
        build_streaming_metrics(cast(Any, [{"request_id": "not-a-dto"}]))


def test_request_observation_is_immutable() -> None:
    observation = _accepted()
    with pytest.raises(FrozenInstanceError):
        observation.request_id = "changed"  # type: ignore[misc]


def test_streaming_report_conformance_rejects_identity_and_accounting_tamper() -> None:
    report = build_streaming_metrics([_accepted()])

    bad_digest = deepcopy(report)
    bad_digest["registry_digest"] = "0" * 64
    with pytest.raises(ValueError, match="registry identity"):
        validate_streaming_metrics(bad_digest)

    bad_count = deepcopy(report)
    counters = cast(dict[str, object], bad_count["counters"])
    counters["client_e2e_missing"] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_streaming_metrics(bad_count)

    short_id = deepcopy(report)
    metrics = cast(dict[str, object], short_id["metrics"])
    metrics["p99_ttft"] = metrics.pop(METRIC_IDS[0])
    with pytest.raises(ValueError, match="fully-qualified IDs"):
        validate_streaming_metrics(short_id)

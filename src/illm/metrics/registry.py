from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from illm.canonical import canonical_json_bytes, sha256_json


METRIC_IDS = (
    "illm.server.ttft.request_weighted.p99.v1",
    "illm.server.itl.token_weighted.p99.v1",
    "illm.server.tpot.request_weighted.p99.v1",
    "illm.server.e2e.request_weighted.p99.v1",
    "illm.client.e2e.request_weighted.p99.v1",
)

_NEAREST_RANK_P99 = {
    "kind": "nearest_rank",
    "quantile_numerator": 99,
    "quantile_denominator": 100,
    "rank": "ceil(0.99*N)",
    "interpolation": "none",
}

_REGISTRY: dict[str, object] = {
    "schema_version": 1,
    "registry_id": "illm.metric-registry.v1",
    "stability": "experimental",
    "metrics": [
        {
            "id": METRIC_IDS[0],
            "stability": "experimental",
            "kind": "derived_distribution",
            "unit": "s",
            "measurement_boundary": "request.accepted to first committed token server_ready_ns",
            "aggregation": dict(_NEAREST_RANK_P99),
            "weighting": "request",
            "denominator": "accepted requests with at least one committed output token",
            "censor_rule": "accepted requests with zero output increment ttft_censored_before_first_token",
            "outcome_stratified": False,
        },
        {
            "id": METRIC_IDS[1],
            "stability": "experimental",
            "kind": "derived_distribution",
            "unit": "s",
            "measurement_boundary": "adjacent committed token server_ready_ns timestamps",
            "aggregation": dict(_NEAREST_RANK_P99),
            "weighting": "token",
            "denominator": "all formed adjacent output-token intervals of accepted requests",
            "censor_rule": "accepted requests with fewer than two outputs increment itl_censored_insufficient_output",
            "outcome_stratified": False,
        },
        {
            "id": METRIC_IDS[2],
            "stability": "experimental",
            "kind": "derived_distribution",
            "unit": "s/token",
            "measurement_boundary": "(last_server_ready_ns-first_server_ready_ns)/(committed_output_count-1)",
            "aggregation": dict(_NEAREST_RANK_P99),
            "weighting": "request",
            "denominator": "accepted requests with at least two committed output tokens",
            "censor_rule": "accepted requests with fewer than two outputs increment tpot_censored_insufficient_output",
            "outcome_stratified": False,
        },
        {
            "id": METRIC_IDS[3],
            "stability": "experimental",
            "kind": "derived_distribution",
            "unit": "s",
            "measurement_boundary": "request.accepted to terminal server commit/ready",
            "aggregation": dict(_NEAREST_RANK_P99),
            "weighting": "request",
            "denominator": "all accepted requests with exactly one terminal outcome",
            "censor_rule": "missing terminal is invalid; results are stratified by terminal outcome",
            "outcome_stratified": True,
        },
        {
            "id": METRIC_IDS[4],
            "stability": "experimental",
            "kind": "derived_distribution",
            "unit": "s",
            "measurement_boundary": "client send start to terminal chunk or structured close observed by client",
            "aggregation": dict(_NEAREST_RANK_P99),
            "weighting": "request",
            "denominator": "accepted terminal requests with a complete client timestamp pair",
            "censor_rule": "missing client timestamp pairs increment client_e2e_missing; results are outcome stratified",
            "outcome_stratified": True,
        },
    ],
}


def metric_registry() -> dict[str, object]:
    """Return an independent copy of the frozen MetricRegistryV1 document."""

    return deepcopy(_REGISTRY)


def registry_digest() -> str:
    """Return the canonical SHA-256 identity of MetricRegistryV1."""

    return sha256_json(_REGISTRY)


def validate_metric_registry(value: Mapping[str, object]) -> None:
    """Fail closed unless *value* is exactly the V1 five-metric registry."""

    if not isinstance(value, Mapping):
        raise ValueError("MetricRegistryV1 must be an object")
    try:
        encoded = canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("MetricRegistryV1 must be canonical-JSON encodable") from exc
    if encoded != canonical_json_bytes(_REGISTRY):
        raise ValueError("MetricRegistryV1 does not match the frozen V0.2 contract")


def registry_conformance(value: Mapping[str, object] | None = None) -> dict[str, object]:
    """Validate a registry and return a deterministic conformance record."""

    candidate = metric_registry() if value is None else value
    validate_metric_registry(candidate)
    return {
        "schema_version": 1,
        "status": "pass",
        "registry_id": "illm.metric-registry.v1",
        "registry_digest": registry_digest(),
        "metric_count": len(METRIC_IDS),
        "metric_ids": list(METRIC_IDS),
    }

"""Explicit CPU facade and version dispatch without changing legacy schema2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from illm.domain.errors import IllmError
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest, MetadataChunk
from illm.numerical_policy import candidate_policy, legacy_policy
from illm.sampling import SamplingSpec


def _request(engine: Engine, *, capture: Path | None = None) -> GenerationRequest:
    return GenerationRequest(model_spec_id=engine.model_spec_id, prompt="hello", max_new_tokens=4,
                             sampling=SamplingSpec(mode="greedy", actual_seed=7), capture=capture)


def test_explicit_cpu_policy_generates_versioned_result_and_metadata() -> None:
    policy = candidate_policy("fp32")
    engine = Engine(numerical_policy=policy)
    try:
        stream = engine.stream(_request(engine))
        try:
            chunks = list(stream)
        finally:
            stream.close()
        metadata = chunks[0]
        assert isinstance(metadata, MetadataChunk)
        assert metadata.to_dict()["schema_version"] == 3
        assert metadata.to_dict()["numerical_policy_digest"] == policy.digest
        result = engine.last_result
        assert result is not None
        assert result.output_token_ids == (114, 114, 114, 114)
        assert result.query_tokens_per_step == (6, 1, 1, 1)
        assert result.kv_valid_tokens_per_step == (6, 7, 8, 9)
        assert result.to_dict()["schema_version"] == 3
        assert result.to_dict()["numerical_policy"] == policy.to_dict()
        assert result.to_dict()["numerical_policy_digest"] == policy.digest
        assert not any(result.resource_status.values())
    finally:
        assert engine.close(timeout=1.0)


def test_old_development_surface_still_serializes_schema2_without_policy() -> None:
    engine = Engine()
    try:
        result = engine.generate(_request(engine))
        assert result.to_dict()["schema_version"] == 2
        assert "numerical_policy" not in result.to_dict()
    finally:
        assert engine.close(timeout=1.0)


def test_gpu_policy_rejected_before_model_loader_or_cuda_initialization() -> None:
    with patch("illm.engine.streaming.load_model", side_effect=AssertionError("must not load CUDA")) as loader:
        with pytest.raises(IllmError, match="rejects CUDA before model loading"):
            Engine(device="cuda:0", numerical_policy=candidate_policy("bf16"))
        loader.assert_not_called()


def test_new_policy_writes_a_separate_schema3_capture(tmp_path: Path) -> None:
    capture = tmp_path / "policy-capture"
    engine = Engine(numerical_policy=candidate_policy("fp32"))
    try:
        result = engine.generate(_request(engine, capture=capture))
        import json
        manifest = json.loads((capture / "capture-manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 3
        assert result.capture_manifest_digest == manifest["manifest_digest"]
        assert all(event.event_catalog == "illm-stateful-v02-policy-v1" for event in engine.last_events)
    finally:
        assert engine.close(timeout=1.0)


def test_legacy_cannot_select_contiguous_under_new_explicit_identity() -> None:
    with pytest.raises(IllmError):
        Engine(kv_mode="contiguous", numerical_policy=legacy_policy("fp32"))

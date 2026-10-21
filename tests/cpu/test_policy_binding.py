"""Policy source/environment bindings without rewriting historical wire formats."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import sha256_json
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import (
    GenerationRequest,
    GenerationResult,
    MetadataChunk,
    TerminalChunk,
)
from illm.memory.policy import PolicyMemoryPlan, build_policy_memory_plan
from illm.model import LoadedModel, load_model
from illm.model.spec import ModelError
from illm.numerical_policy import candidate_policy
from illm.policy_binding import build_execution_binding
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _toolchain_common as toolchain_common


@pytest.fixture(scope="module")
def loaded() -> LoadedModel:
    return load_model(project_root=ROOT)


def _clear_toolchain_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST"):
        monkeypatch.delenv(name, raising=False)


def _minimal_project(root: Path) -> Path:
    (root / "src" / "illm").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname='binding-fixture'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("source-v1\n", encoding="utf-8")
    return root


def _manifest(root: Path, *, environment_id: str, lock_digest: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "finalized": True,
        "project_root": str(root.resolve()),
        "project_root_digest": toolchain_common.project_digest(root),
        "environment_id": environment_id,
        "lock_digest": lock_digest,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
    }
    value["manifest_digest"] = sha256_json(value)
    return value


def test_binding_source_digest_matches_authoritative_toolchain_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_toolchain_binding(monkeypatch)
    binding = build_execution_binding(ROOT).to_dict()

    assert binding["implementation_source_scope"] == "project-root-v1"
    assert binding["implementation_source_digest"] == toolchain_common.project_digest(ROOT)
    assert binding["protocol_sha256"] == "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"
    assert binding["toolchain"] == {"binding": "unbound-development", "manifest_digest": None}
    assert binding["environment"]["backend"] == "cpu"
    assert binding["environment"]["cuda_arithmetic"] == "not_applicable"
    assert binding["environment"]["cuda_execution_authorized"] is False
    assert binding["environment_digest"] == sha256_json(binding["environment"])


def test_source_and_environment_changes_invalidate_binding_and_memory_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedModel,
) -> None:
    _clear_toolchain_binding(monkeypatch)
    project = _minimal_project(tmp_path / "project")
    policy = candidate_policy("fp32")

    initial = build_policy_memory_plan(
        loaded,
        policy,
        "contiguous",
        16,
        1 << 40,
        project_root=project,
    )
    (project / "README.md").write_text("source-v2\n", encoding="utf-8")
    changed_source = build_policy_memory_plan(
        loaded,
        policy,
        "contiguous",
        16,
        1 << 40,
        project_root=project,
    )

    assert initial.numerical_policy.digest == changed_source.numerical_policy.digest
    assert initial.execution_binding.digest != changed_source.execution_binding.digest
    assert initial.digest != changed_source.digest
    assert (
        initial.execution_binding.to_dict()["implementation_source_digest"]
        != changed_source.execution_binding.to_dict()["implementation_source_digest"]
    )

    monkeypatch.setattr("illm.policy_binding.platform.platform", lambda: "synthetic-platform-v2")
    changed_environment = build_policy_memory_plan(
        loaded,
        policy,
        "contiguous",
        16,
        1 << 40,
        project_root=project,
    )
    source_binding = changed_source.execution_binding.to_dict()
    environment_binding = changed_environment.execution_binding.to_dict()
    assert source_binding["implementation_source_digest"] == environment_binding["implementation_source_digest"]
    assert source_binding["environment_digest"] != environment_binding["environment_digest"]
    assert changed_source.execution_binding.digest != changed_environment.execution_binding.digest
    assert changed_source.digest != changed_environment.digest


def test_policy_plan_metadata_and_result_share_one_binding_and_plan_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_toolchain_binding(monkeypatch)
    policy = candidate_policy("fp32")
    engine = Engine(project_root=ROOT, numerical_policy=policy)
    try:
        request = GenerationRequest(
            model_spec_id=engine.model_spec_id,
            input_token_ids=(1, 107),
            max_new_tokens=1,
            sampling=SamplingSpec(mode="greedy", actual_seed=7),
            request_id="policy-binding-request",
            telemetry="off",
        )
        chunks = list(engine.stream(request))
        metadata = chunks[0]
        terminal = chunks[-1]
        assert isinstance(metadata, MetadataChunk)
        assert isinstance(terminal, TerminalChunk)
        assert isinstance(engine.memory_plan, PolicyMemoryPlan)

        plan = engine.memory_plan
        plan_wire = plan.to_dict()
        metadata_wire = metadata.to_dict()
        result_wire = terminal.result.to_dict()
        assert metadata_wire["schema_version"] == result_wire["schema_version"] == 3
        assert metadata_wire["execution_binding"] == result_wire["execution_binding"] == plan_wire["execution_binding"]
        assert (
            metadata_wire["execution_binding_digest"]
            == result_wire["execution_binding_digest"]
            == plan_wire["execution_binding_digest"]
            == plan.execution_binding.digest
        )
        assert metadata_wire["memory_plan_digest"] == result_wire["memory_plan_digest"] == plan.digest
        assert metadata_wire["numerical_policy_digest"] == result_wire["numerical_policy_digest"] == policy.digest
    finally:
        assert engine.close(timeout=5.0)


def test_historical_schema2_and_metadata_wire_do_not_gain_null_policy_fields() -> None:
    result = GenerationResult(
        request_id="old-result",
        input_token_ids=(1,),
        output_token_ids=(2,),
        text="",
        outcome="FINISHED",
        finish_reason="MAX_NEW_TOKENS",
        error_code=None,
        actual_seed=7,
        model_spec_id="0" * 64,
        sampling_spec_digest="1" * 64,
        kv_mode="contiguous",
        next_output_position=1,
        rng_draws_consumed=0,
        committed_draw_addresses=(),
        query_tokens_per_step=(1,),
        kv_valid_tokens_per_step=(1,),
        resource_status={},
        timings={},
        capture_manifest_digest=None,
        capture_error=None,
        event_count=0,
    ).to_dict()
    assert set(result) == {
        "schema_version",
        "request_id",
        "input_token_ids",
        "output_token_ids",
        "text",
        "outcome",
        "finish_reason",
        "error_code",
        "actual_seed",
        "model_spec_id",
        "sampling_spec_digest",
        "kv_mode",
        "next_output_position",
        "rng_draws_consumed",
        "committed_draw_addresses",
        "query_tokens_per_step",
        "kv_valid_tokens_per_step",
        "resource_status",
        "timings",
        "capture_manifest_digest",
        "capture_error",
        "event_count",
    }
    assert result["schema_version"] == 2

    metadata = MetadataChunk("old", "0" * 64, "1" * 64, "2" * 64, 1, "contiguous", 3).to_dict()
    assert set(metadata) == {
        "request_id",
        "model_spec_id",
        "sampling_spec_digest",
        "memory_plan_digest",
        "input_tokens",
        "kv_mode",
        "accepted_ns",
        "kind",
    }


def test_stale_or_environment_mismatched_supplied_manifest_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _minimal_project(tmp_path / "project")
    manifest_path = tmp_path / "toolchain.json"
    environment_id = "synthetic-environment-v1"
    lock_digest = "f" * 64
    manifest_path.write_text(
        json.dumps(_manifest(project, environment_id=environment_id, lock_digest=lock_digest)),
        encoding="utf-8",
    )
    monkeypatch.setenv("ILLM_TOOLCHAIN_MANIFEST", str(manifest_path))
    monkeypatch.setenv("ILLM_ENVIRONMENT_ID", environment_id)
    monkeypatch.setenv("ILLM_LOCK_DIGEST", lock_digest)

    bound = build_execution_binding(project).to_dict()
    assert bound["toolchain"]["binding"] == "source-bound-manifest"

    (project / "README.md").write_text("stale-source\n", encoding="utf-8")
    with pytest.raises(ModelError, match="stale or mismatched environment manifest"):
        build_execution_binding(project)

    refreshed = _manifest(project, environment_id=environment_id, lock_digest=lock_digest)
    manifest_path.write_text(json.dumps(refreshed), encoding="utf-8")
    monkeypatch.setenv("ILLM_ENVIRONMENT_ID", "wrong-environment")
    with pytest.raises(ModelError, match="stale or mismatched environment manifest"):
        build_execution_binding(project)

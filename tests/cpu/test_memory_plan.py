"""Pure planning tests: no model execution and no live-memory queries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from illm.kv_cache import CONTIGUOUS_LAYOUT_ID, KVLayoutSpec
from illm.memory import MemoryPlan, build_memory_plan
from illm.model.loader import build_weight_manifest, resolve_model
from illm.model.spec import ModelError


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model_contract() -> Any:
    source = resolve_model(project_root=ROOT)
    manifest = build_weight_manifest(source.weight_paths, source.spec)
    return SimpleNamespace(
        spec=source.spec,
        weight_manifest=manifest,
        max_context_tokens=source.max_context_tokens,
        reference=SimpleNamespace(
            canonical_weight=lambda name: SimpleNamespace(device="cpu")
        ),
    )


def test_memory_plan_is_immutable_exactly_summed_and_bound(model_contract: Any) -> None:
    plan = build_memory_plan(model_contract, "contiguous", 64, 1 << 40, device="cpu")
    assert isinstance(plan, MemoryPlan)
    assert plan.model_spec_id == model_contract.spec.spec_id
    assert plan.weights_digest == model_contract.weight_manifest["weights_digest"]
    assert plan.weight_manifest_digest == model_contract.weight_manifest["manifest_digest"]
    assert plan.runtime_dtype == model_contract.spec.runtime_dtype
    assert plan.device == "cpu"
    assert plan.capacity_tokens == 64
    assert plan.kv_layout_id == CONTIGUOUS_LAYOUT_ID
    assert plan.kv_bytes == KVLayoutSpec.for_model(model_contract.spec, device="cpu", capacity=64).reserved_bytes
    assert plan.required_bytes == sum(
        (
            plan.weights_bytes,
            plan.kv_bytes,
            plan.activation_bytes,
            plan.workspace_bytes,
            plan.io_bytes,
            plan.telemetry_bytes,
            plan.headroom_bytes,
        )
    )
    assert plan.fits_budget
    assert plan.digest == build_memory_plan(model_contract, "contiguous", 64, 1 << 40, device="cpu").digest
    with pytest.raises(FrozenInstanceError):
        plan.capacity_tokens = 32  # type: ignore[misc]


def test_none_plan_keeps_v01_layout_and_has_no_kv_allocation(model_contract: Any) -> None:
    old_spec_id = model_contract.spec.spec_id
    plan = build_memory_plan(model_contract, "none", 32, 1 << 40, device="cpu", telemetry_bytes=0)
    assert plan.kv_bytes == 0
    assert plan.kv_layout_id == "none-reference-full-context-v1"
    assert model_contract.spec.kv_layout_id == "none-reference-full-context-v1"
    assert model_contract.spec.spec_id == old_spec_id


def test_plan_accounts_for_full_context_logits_and_retained_eager_temporaries(model_contract: Any) -> None:
    plan = build_memory_plan(model_contract, "none", 128, 1 << 40, device="cpu")
    assert plan.io_bytes >= 2 * 128 * model_contract.spec.vocab_size * 4
    assert plan.activation_bytes >= 8 * 128 * model_contract.spec.intermediate_size * 4
    assert plan.telemetry_bytes >= 128 * 64 * 1024


def test_preaccept_capacity_and_budget_rejection_are_deterministic(model_contract: Any) -> None:
    roomy = build_memory_plan(model_contract, "contiguous", 16, 1 << 40, device="cpu")
    roomy.require_admission(16)
    with pytest.raises(ModelError) as capacity:
        roomy.require_admission(17)
    assert capacity.value.code == "RESOURCE_EXHAUSTED"
    short = build_memory_plan(model_contract, "contiguous", 16, roomy.required_bytes - 1, device="cpu")
    assert short.required_bytes == roomy.required_bytes
    assert not short.fits_budget
    with pytest.raises(ModelError) as budget:
        short.require_admission(1)
    assert budget.value.code == "RESOURCE_EXHAUSTED"
    assert budget.value.field == "memory_budget_bytes"


def test_plan_changes_when_frozen_capacity_device_or_kv_contract_changes(model_contract: Any) -> None:
    base = build_memory_plan(model_contract, "contiguous", 16, 1 << 40, device="cpu")
    capacity = build_memory_plan(model_contract, "contiguous", 17, 1 << 40, device="cpu")
    none = build_memory_plan(model_contract, "none", 16, 1 << 40, device="cpu")
    cuda_contract: Any = SimpleNamespace(
        spec=model_contract.spec,
        weight_manifest=model_contract.weight_manifest,
        max_context_tokens=model_contract.max_context_tokens,
        reference=SimpleNamespace(
            canonical_weight=lambda name: SimpleNamespace(device="cuda:0")
        ),
    )
    cuda_binding = build_memory_plan(cuda_contract, "contiguous", 16, 1 << 40, device="cuda:0")
    assert len({base.digest, capacity.digest, none.digest, cuda_binding.digest}) == 4
    assert base.weights_bytes == sum(item["final_storage_bytes"] for item in model_contract.weight_manifest["storages"])
    with pytest.raises(ModelError, match="loaded model device"):
        build_memory_plan(model_contract, "contiguous", 16, 1 << 40, device="cuda:0")


@pytest.mark.parametrize("field,value", [("capacity", 0), ("capacity", 257), ("memory", 0), ("telemetry", -1)])
def test_invalid_plan_inputs_fail_before_admission(model_contract: Any, field: str, value: int) -> None:
    arguments = {"capacity": 16, "memory_budget_bytes": 1 << 40, "telemetry_bytes": 0}
    if field == "memory":
        arguments["memory_budget_bytes"] = value
    elif field == "telemetry":
        arguments["telemetry_bytes"] = value
    else:
        arguments[field] = value
    with pytest.raises(ModelError):
        build_memory_plan(model_contract, "contiguous", arguments["capacity"], arguments["memory_budget_bytes"],
                          device="cpu", telemetry_bytes=arguments["telemetry_bytes"])

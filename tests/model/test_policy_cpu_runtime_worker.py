"""Actual small-scope CPU runs for the raw numerical evidence worker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_file
from illm.policy_cpu_numerical_contract import canonical_tiny_ops, cpu_cases, high_cases
from illm.policy_cpu_runtime_worker import (
    EVIDENCE_STAGE,
    QUALIFICATION_STATUS,
    REPORT_KIND,
    RuntimeWorkerError,
    _build_independent_transformers_oracle,
    _coverage,
    _new_output_root,
    _run_runtime_worker,
)
from illm.policy_tensor_evidence import EvidenceTensor, TensorPack


ROOT = Path(__file__).resolve().parents[2]
ZERO_RESOURCES = {
    "closed": True,
    "active_leases": 0,
    "allocated_bytes": 0,
    "valid_tokens": 0,
    "staged_tokens": 0,
}
TOP_LEVEL = {
    "schema_version",
    "report_kind",
    "status",
    "evidence_stage",
    "qualification_status",
    "role",
    "coverage",
    "binding",
    "binding_digest",
    "process",
    "model",
    "policy",
    "pack_manifest_digest",
    "cases",
    "high_cases",
    "op_authority_cases",
}


def _case(case_id: str) -> dict[str, Any]:
    return next(case for case in cpu_cases(ROOT) if case["case_id"] == case_id)


def _high(case_id: str) -> dict[str, Any]:
    return next(case for case in high_cases() if case["case_id"] == case_id)


def _case_report(report: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    cases = cast(list[dict[str, Any]], report["cases"])
    return next(case for case in cases if case["case_id"] == case_id)


def _path(case: Mapping[str, Any], path_id: str) -> dict[str, Any]:
    return cast(dict[str, Any], cast(Mapping[str, Any], case["paths"])[path_id])


def _assert_complete(output: Path, report: Mapping[str, Any]) -> TensorPack:
    assert set(report) == TOP_LEVEL
    assert report["schema_version"] == 1
    assert report["report_kind"] == REPORT_KIND
    assert report["status"] == "complete"
    assert report["evidence_stage"] == EVIDENCE_STAGE == "DEVELOPMENT_ONLY"
    assert report["qualification_status"] == QUALIFICATION_STATUS == "unqualified"
    encoded = canonical_json_bytes(report)
    assert (output / "worker.json").read_bytes() == encoded
    assert (output / "COMPLETE").read_bytes() == (sha256_bytes(encoded) + "\n").encode("ascii")
    assert not (output / "INCOMPLETE").exists()
    pack = TensorPack.open(output / "pack")
    pack_manifest = json.loads((output / "pack" / "manifest.json").read_bytes())
    assert pack_manifest["manifest_digest"] == report["pack_manifest_digest"]
    return pack


@pytest.fixture(scope="module")
def candidate_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any], TensorPack]:
    output = tmp_path_factory.mktemp("policy-worker-candidate") / "evidence"
    report = _run_runtime_worker(
        ROOT,
        output,
        role="candidate",
        cases=[_case("differential-01"), _case("differential-03")],
        high_case_records=[_high("high-00"), _high("high-05")],
        test_subset=True,
    )
    return output, report, _assert_complete(output, report)


def test_import_is_tensor_runtime_lazy() -> None:
    source = textwrap.dedent(
        """
        import pathlib
        import sys
        root = pathlib.Path(sys.argv[1])
        sys.path.insert(0, str(root / "src"))
        import illm.policy_cpu_runtime_worker
        assert "torch" not in sys.modules
        assert "illm.model.policy" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", source, str(ROOT)],
        cwd=ROOT,
        env=_child_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_candidate_free_runs_state_queries_prefix_and_natural_eos(
    candidate_run: tuple[Path, dict[str, Any], TensorPack],
) -> None:
    _, report, pack = candidate_run
    assert report["role"] == "candidate"
    coverage = cast(dict[str, Any], report["coverage"])
    assert coverage == {
        "kind": "test-subset",
        "case_count": 2,
        "expected_case_count": 70,
        "suite_counts": {"v02-policy-tiny50-inputs-v1": 2},
        "high_case_count": 2,
        "expected_high_case_count": 8,
        "full_contract": False,
    }
    process = cast(dict[str, Any], report["process"])
    assert process["cuda_visible_devices"] == ""
    assert process["cuda_initialized_before"] is False
    assert process["cuda_initialized_after"] is False
    assert cast(dict[str, Any], report["model"])["native_oracle"] is None

    regular = _case_report(report, "differential-01")
    b, c = _path(regular, "B"), _path(regular, "C")
    assert b["output_token_ids"] == c["output_token_ids"] == [5, 5]
    assert b["finish_reason"] == c["finish_reason"] == "MAX_NEW_TOKENS"
    b_steps = cast(list[dict[str, Any]], b["steps"])
    c_steps = cast(list[dict[str, Any]], c["steps"])
    assert [step["query_tokens"] for step in b_steps] == [2, 3]
    assert [step["query_tokens"] for step in c_steps] == [2, 1]
    assert [step["kv_before"] for step in c_steps] == [0, 2]
    assert [step["kv_after"] for step in c_steps] == [2, 3]
    assert [step["fence_complete"] for step in c_steps] == [True, True]
    assert [step["state_before"] for step in b_steps] == [step["state_before"] for step in c_steps]
    assert [step["state_after"] for step in b_steps] == [step["state_after"] for step in c_steps]
    assert b["resource_status"] == c["resource_status"] == ZERO_RESOURCES
    assert set(c_steps[1]["prefix_ids"]) == {"0", "1"}
    for layer in ("0", "1"):
        prefix = cast(dict[str, str], c_steps[1]["prefix_ids"][layer])
        for identifier in prefix.values():
            tensor = pack.tensor(identifier)
            assert tensor.dtype == "fp32"
            assert tensor.shape == (1, 2, 2, 8)

    expected_names = set(canonical_tiny_ops())
    for step, expected_positions in zip(b_steps, ({"0", "1"}, {"0", "1", "2"}), strict=True):
        op_ids = cast(dict[str, dict[str, str]], step["op_ids"])
        assert set(op_ids) == expected_names
        assert all(set(positions) == expected_positions for positions in op_ids.values())
    for step, expected_positions in zip(c_steps, ({"0", "1"}, {"2"}), strict=True):
        op_ids = cast(dict[str, dict[str, str]], step["op_ids"])
        assert set(op_ids) == expected_names
        assert all(set(positions) == expected_positions for positions in op_ids.values())
    q_id = cast(dict[str, str], b_steps[0]["op_ids"]["layer.0.q_projection"])["0"]
    score_id = cast(dict[str, str], b_steps[0]["op_ids"]["layer.0.attention_scores"])["1"]
    assert pack.tensor(q_id).shape == (1, 4, 1, 8)
    assert pack.tensor(score_id).shape == (1, 4, 1, 2)
    for b_step, c_step in zip(b_steps, c_steps, strict=True):
        b_logits = pack.tensor(cast(str, b_step["logits_id"]))
        c_logits = pack.tensor(cast(str, c_step["logits_id"]))
        assert b_logits.raw[-(259 * 4) :] == c_logits.raw[-(259 * 4) :]

    eos = _case_report(report, "differential-03")
    for path_id in ("B", "C"):
        path = _path(eos, path_id)
        assert path["output_token_ids"] == [2]
        assert path["finish_reason"] == "EOS"
        state = cast(dict[str, Any], cast(list[dict[str, Any]], path["steps"])[0]["state_after"])
        assert cast(dict[str, Any], state["value"])["request_id"] == "differential-03"


def test_candidate_high_and_frozen_op_authority_are_raw_and_distinct(
    candidate_run: tuple[Path, dict[str, Any], TensorPack],
) -> None:
    _, report, pack = candidate_run
    highs = {case["case_id"]: case for case in cast(list[dict[str, Any]], report["high_cases"])}
    assert set(highs) == {"high-00", "high-05"}
    forward = highs["high-00"]
    b_high, c_high = _path(forward, "B-high"), _path(forward, "C-high")
    assert pack.tensor(cast(str, b_high["full_logits_id"])).raw == pack.tensor(
        cast(str, c_high["full_logits_id"])
    ).raw
    rope = highs["high-05"]
    for name in ("input_q", "input_k", "rope_q", "rope_k"):
        b_id = cast(dict[str, str], _path(rope, "B-high")["steps"][0]["op_ids"][name])["full"]
        c_id = cast(dict[str, str], _path(rope, "C-high")["steps"][0]["op_ids"][name])["full"]
        assert pack.tensor(b_id).raw == pack.tensor(c_id).raw

    authorities = cast(list[dict[str, Any]], report["op_authority_cases"])
    assert [case["case_id"] for case in authorities] == [
        "hello",
        "vocab-min-position-zero",
        "vocab-max-position-last",
    ]
    for authority in authorities:
        source = ROOT / cast(str, authority["source_file"])
        assert sha256_file(source) == authority["source_sha256"]
        b_path, c_path = _path(authority, "B-authority"), _path(authority, "C-authority")
        b_ops = cast(dict[str, dict[str, str]], b_path["steps"][0]["op_ids"])
        c_ops = cast(dict[str, dict[str, str]], c_path["steps"][0]["op_ids"])
        assert set(b_ops) == set(c_ops)
        for name in b_ops:
            assert set(b_ops[name]) == set(c_ops[name])
            for position in b_ops[name]:
                assert pack.tensor(b_ops[name][position]).raw == pack.tensor(
                    c_ops[name][position]
                ).raw
        if authority["mode"] == "forward":
            assert set(b_ops) == set(canonical_tiny_ops())
            assert b_path["steps"][0]["logits_id"] is not None
            assert c_path["steps"][0]["logits_id"] is not None
        else:
            assert authority["positions"] == [255]
            assert len(b_ops) == 8
            assert b_path["steps"][0]["logits_id"] is None
            assert c_path["steps"][0]["logits_id"] is None


CHILD = textwrap.dedent(
    """
    from __future__ import annotations
    from pathlib import Path
    import sys
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    role = sys.argv[3]
    case_ids = set(sys.argv[4].split(","))
    sys.path.insert(0, str(root / "src"))
    from illm.policy_cpu_numerical_contract import cpu_cases
    from illm.policy_cpu_runtime_worker import _run_runtime_worker
    selected = [case for case in cpu_cases(root) if case["case_id"] in case_ids]
    _run_runtime_worker(
        root,
        output,
        role=role,
        cases=selected,
        high_case_records=[],
        test_subset=True,
    )
    """
)


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("ILLM_TOOLCHAIN_MANIFEST", "ILLM_ENVIRONMENT_ID", "ILLM_LOCK_DIGEST"):
        environment.pop(name, None)
    return environment


def _run_child(output: Path, *, role: str, case_ids: Sequence[str]) -> tuple[dict[str, Any], TensorPack]:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-c",
            CHILD,
            str(ROOT),
            str(output),
            role,
            ",".join(case_ids),
        ],
        cwd=ROOT,
        env=_child_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = cast(dict[str, Any], json.loads((output / "worker.json").read_bytes()))
    return report, _assert_complete(output, report)


def _future_probabilities(tensor: EvidenceTensor) -> list[float]:
    batch, heads, rows, columns = tensor.shape
    assert batch == 1 and rows == columns
    result: list[float] = []
    for head in range(heads):
        base = head * rows * columns
        for row in range(rows):
            result.extend(tensor.values[base + row * columns + row + 1 : base + (row + 1) * columns])
    return result


def test_native_fresh_process_uses_independent_oracle_and_keeps_future_masked(
    tmp_path: Path,
) -> None:
    report, pack = _run_child(
        tmp_path / "native",
        role="native",
        case_ids=("differential-01", "differential-03"),
    )
    assert report["role"] == "native"
    assert report["policy"] is None
    assert report["high_cases"] == report["op_authority_cases"] == []
    process = cast(dict[str, Any], report["process"])
    assert process["native_policy_module_loaded_before"] is False
    assert process["native_policy_module_loaded_after"] is False
    assert process["cuda_initialized_before"] is process["cuda_initialized_after"] is False
    model = cast(dict[str, Any], report["model"])
    oracle_identity = cast(dict[str, Any], model["native_oracle"])
    assert set(oracle_identity) == {
        "kind",
        "transformers_version",
        "model_class",
        "source",
        "device",
        "parameter_dtype",
        "rope_inv_freq_dtype",
        "attention_implementation",
        "use_cache",
        "asset_files",
        "fixture_manifest_digest",
        "spec_id",
        "weight_manifest_digest",
        "weights_digest",
        "source_shards",
    }
    assert oracle_identity["kind"] == "independent-pinned-transformers-oracle-v1"
    assert oracle_identity["transformers_version"] == "4.57.1"
    assert oracle_identity["model_class"] == (
        "transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM"
    )
    assert oracle_identity["device"] == "cpu"
    assert oracle_identity["parameter_dtype"] == "torch.float32"
    assert oracle_identity["rope_inv_freq_dtype"] == "torch.float32"
    assert oracle_identity["attention_implementation"] == "eager"
    assert oracle_identity["use_cache"] is False
    source_identity = cast(dict[str, Any], oracle_identity["source"])
    assert source_identity["module"] == "transformers.models.qwen2.modeling_qwen2"
    assert len(cast(str, source_identity["sha256"])) == 64
    oracle_assets = cast(dict[str, str], oracle_identity["asset_files"])
    all_assets = cast(dict[str, str], model["asset_files"])
    assert oracle_assets == {name: all_assets[name] for name in oracle_assets}
    assert set(oracle_assets) == {
        "config.json",
        "model.safetensors",
        "supported-model-spec.json",
        "weight-load-manifest.json",
        "tokenizer.json",
        "fixture-manifest.json",
    }
    assert oracle_identity["spec_id"] == model["model_spec_id"]
    assert oracle_identity["weight_manifest_digest"] == model["weight_manifest_digest"]
    assert oracle_identity["weights_digest"] == model["weights_digest"]
    assert oracle_identity["source_shards"] == cast(dict[str, Any], model["weight_manifest"])[
        "source_shards"
    ]
    regular = _case_report(report, "differential-01")
    a, d = _path(regular, "A"), _path(regular, "D-native")
    assert a["output_token_ids"] == d["output_token_ids"]
    assert a["resource_status"] == d["resource_status"] == ZERO_RESOURCES
    assert [step["query_tokens"] for step in a["steps"]] == [2, 3]
    assert [step["query_tokens"] for step in d["steps"]] == [2, 3]
    for step in cast(list[dict[str, Any]], d["steps"]):
        assert set(cast(dict[str, str], step["op_ids"])) == set(canonical_tiny_ops())
        assert pack.tensor(cast(str, step["logits_id"])).raw == pack.tensor(
            cast(str, step["ordinary_logits_id"])
        ).raw
        probabilities = pack.tensor(
            cast(dict[str, str], step["op_ids"])["layer.0.attention_probabilities"]
        )
        context_length = len(cast(list[int], step["context"]))
        assert probabilities.shape == (1, 4, context_length, context_length)
        assert _future_probabilities(probabilities)
        assert set(_future_probabilities(probabilities)) == {0.0}
    eos = _case_report(report, "differential-03")
    assert _path(eos, "A")["output_token_ids"] == [2]
    assert _path(eos, "D-native")["output_token_ids"] == [2]


def test_independent_oracle_construction_cannot_read_a_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from illm.model import load_model
    from illm.model.reference import ReferenceDecoder

    # Materialize A first, then poison its only canonical parameter accessor.
    # D construction receives only the project root and must still succeed from
    # the independently verified frozen assets/spec.
    load_model(project_root=ROOT)

    def reject_candidate_parameter_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("D-native observed A/B/C canonical parameters")

    monkeypatch.setattr(ReferenceDecoder, "canonical_weight", reject_candidate_parameter_read)
    oracle, source, identity = _build_independent_transformers_oracle(ROOT, torch)
    assert source.spec.spec_id == identity["spec_id"]
    assert {str(parameter.dtype) for parameter in oracle.parameters()} == {"torch.float32"}
    assert {str(parameter.device) for parameter in oracle.parameters()} == {"cpu"}


def test_reproduction_processes_match_every_raw_tensor_token_and_state(tmp_path: Path) -> None:
    first, first_pack = _run_child(
        tmp_path / "reproduction-1",
        role="candidate-reproduction",
        case_ids=("fixed-greedy-00",),
    )
    second, second_pack = _run_child(
        tmp_path / "reproduction-2",
        role="candidate-reproduction",
        case_ids=("fixed-greedy-00",),
    )
    assert first["process"]["pid"] != second["process"]["pid"]
    assert first["high_cases"] == first["op_authority_cases"] == []
    assert second["high_cases"] == second["op_authority_cases"] == []
    first_case = _case_report(first, "fixed-greedy-00")
    second_case = _case_report(second, "fixed-greedy-00")
    expected = _case("fixed-greedy-00")["expected_output_token_ids"]
    for path_id in ("B", "C"):
        first_path, second_path = _path(first_case, path_id), _path(second_case, path_id)
        assert first_path["output_token_ids"] == second_path["output_token_ids"] == expected
        assert first_path["steps"] == second_path["steps"]
    assert set(first_pack.records) == set(second_pack.records)
    for identifier in first_pack.records:
        assert first_pack.tensor(identifier).raw == second_pack.tensor(identifier).raw


def test_public_coverage_cannot_be_inferred_from_a_small_subset() -> None:
    with pytest.raises(RuntimeWorkerError, match="incomplete CPU protocol coverage"):
        _coverage("candidate", [_case("differential-00")], [], test_subset=False)


def test_output_refuses_existing_foreign_and_linked_roots(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeWorkerError, match="must not already exist"):
        _new_output_root(existing, ROOT, test_subset=True)

    foreign = tmp_path / "foreign"
    with pytest.raises(RuntimeWorkerError, match="project_root/out"):
        _new_output_root(foreign, tmp_path / "different-project", test_subset=False)
    assert not foreign.exists()

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(RuntimeWorkerError, match="links or reparse points"):
        _new_output_root(linked / "evidence", ROOT, test_subset=True)


def test_runtime_failure_retains_incomplete_markers(tmp_path: Path) -> None:
    output = tmp_path / "failed-worker"
    malformed = _case("differential-00")
    del malformed["input_digest"]
    with pytest.raises(RuntimeWorkerError, match="closed worker input contract"):
        _run_runtime_worker(
            ROOT,
            output,
            role="candidate",
            cases=[malformed],
            high_case_records=[],
            test_subset=True,
        )
    assert (output / "INCOMPLETE").read_bytes() == b"policy_cpu_runtime_worker_in_progress\n"
    assert (output / "pack" / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()
    assert not (output / "worker.json").exists()

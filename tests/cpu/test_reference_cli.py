from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import sha256_json
from illm.capture.reference import ReferenceCaptureSession
from illm.cli import main
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEventKind as K
from illm.engine.contracts import GenerationRequest
from illm.sampling import SamplingSpec


def test_model_inspect_reports_frozen_fixture_and_weight_coverage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model", "inspect", "--model", "fixture:tiny-decoder"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["spec"]["architecture"] == "Qwen2ForCausalLM"
    assert result["parameter_count"] == 27008
    assert len(result["canonical_tensor_shapes"]) == 26
    assert result["weight_manifest"]["finalized"] is True


@pytest.mark.parametrize("arguments", [
    ["generate", "--prompt", "hello", "--max-new-tokens", "0"],
    ["generate", "--prompt", "hello", "--temperature", "1"],
    ["generate", "--prompt", "hello", "--seed", "-1"],
    ["generate", "--input-token-ids", "[true]"],
    ["generate", "--input-token-ids", "[1.5]"],
    ["generate", "--input-token-ids", "{}"],
    ["generate", "--input-token-ids", "not-json"],
])
def test_invalid_generation_parameters_fail_before_loading_tensor_runtime(arguments: list[str]) -> None:
    assert main(arguments) == 64


def test_unsupported_model_is_structured_without_importing_tensor_runtime(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["model", "inspect", "--model", "fixture:unknown"]) == 80
    assert json.loads(capsys.readouterr().err)["code"] == "UNSUPPORTED_MODEL"


@pytest.mark.parametrize("patch", [
    {"telemetry": []}, {"telemetry": {}}, {"input_token_ids": [1], "prompt": None},
    {"request_id": "private prompt with spaces"}, {"max_new_tokens": True},
    {"add_bos": 1}, {"prompt": None}, {"input_token_ids": (1,)},
])
def test_request_dto_fails_closed_with_structured_errors(patch: dict[str, Any]) -> None:
    options: dict[str, Any] = {"prompt": "hello", **patch}
    with pytest.raises(IllmError) as failure:
        GenerationRequest(**options)
    assert failure.value.code is ErrorCode.INVALID_ARGUMENT


def test_schema_one_cli_replays_controls_not_model_numbers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = SamplingSpec(actual_seed=7)
    identity = {key: sha256_json(key) for key in (
        "model_spec_id", "weights_digest", "load_manifest_digest", "tokenizer_digest", "input_spec_digest",
    )}
    capture = ReferenceCaptureSession(tmp_path / "reference", identity, "public-request", spec.to_dict())
    capture.emit(K.REQUEST_ACCEPTED, {"max_new_tokens": 1})
    capture.emit(K.REQUEST_TOKENIZED, {"input_tokens": 2, "input_digest": sha256_json("input")})
    capture.emit(K.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": 2})
    capture.emit(K.MODEL_OP, {"output_position": 0, "op_name": "embedding", "shape": [1, 2, 32], "dtype": "torch.float32"})
    capture.emit(K.MODEL_FORWARD_COMPLETED, {"output_position": 0, "logits_digest": sha256_json("logits")})
    capture.emit(K.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": spec.digest})
    capture.emit(K.SAMPLING_SELECTED, {"output_position": 0, "sampling_spec_digest": spec.digest,
                                     "logits_digest": sha256_json("logits"), "draw_address": None,
                                     "selected_digest": sha256_json("selected")})
    capture.emit(K.REQUEST_FINISHED, {"output_tokens": 1, "output_digest": sha256_json("output"), "finish_reason": "MAX_NEW_TOKENS"})
    capture.finalize()
    assert main(["replay", str(capture.root), "--mode", "trace", "--step-id", "0"]) == 0
    trace = capsys.readouterr().out
    assert "embedding" in trace and '"mode":"R0"' in trace
    assert main(["replay", str(capture.root), "--mode", "state", "--assert-state-digest"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "R1_REFERENCE"
    assert result["numerical_reexecution"] is False
    assert result["rng_draws_consumed"] == 0 and result["output_tokens"] == 1
    assert main(["replay", str(capture.root), "--mode", "state", "--step-id", "0"]) == 64


@pytest.mark.parametrize("schema", [True, "1", 2])
def test_cli_never_guesses_unknown_capture_schema(tmp_path: Path, schema: object) -> None:
    (tmp_path / "capture-manifest.json").write_text(json.dumps({"schema_version": schema}), encoding="utf-8")
    assert main(["replay", str(tmp_path), "--mode", "trace"]) == 78


def test_cli_distinguishes_corrupt_and_missing_capture_manifests(tmp_path: Path) -> None:
    assert main(["replay", str(tmp_path), "--mode", "trace"]) == 74
    (tmp_path / "capture-manifest.json").write_text("not-json", encoding="utf-8")
    assert main(["replay", str(tmp_path), "--mode", "trace"]) == 75


def test_release_cli_dispatches_v1_without_guessing_v0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from illm import reference_release

    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":1}', encoding="utf-8")

    def unit_dispatch_only(actual: Path, *, expected_source_identity: object = None) -> dict[str, Any]:
        assert actual == path and expected_source_identity is None
        return {"schema_version": 1, "unit_dispatch_only": True}

    monkeypatch.setattr(reference_release, "verify_reference_release_bundle", unit_dispatch_only)
    assert main(["release", "verify", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["unit_dispatch_only"] is True


def test_release_dispatch_rejects_duplicate_version_discriminator(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":1,"schema_version":0}', encoding="utf-8")
    assert main(["release", "verify", str(path)]) == 64

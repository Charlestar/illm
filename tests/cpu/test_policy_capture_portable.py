from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureSession
from illm.domain.streaming_events import StreamingEventKind as K
from illm.numerical_policy import candidate_policy
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_policy_capture_portable.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata() -> dict[str, Any]:
    policy = candidate_policy("fp32").to_dict()
    environment = {
        "backend": "cpu",
        "python_version": "3.12.0",
        "platform": "portable-fixture-platform",
        "machine": "portable-fixture-machine",
        "torch_version": "portable-fixture-torch",
        "cuda_arithmetic": "not_applicable",
        "cuda_execution_authorized": False,
    }
    binding = {
        "schema_version": 1,
        "implementation_source_scope": "project-root-v1",
        "implementation_source_digest": "a" * 64,
        "environment": environment,
        "environment_digest": sha256_json(environment),
        "toolchain": {"binding": "unbound-development", "manifest_digest": None},
        "protocol_sha256": PROTOCOL_SHA256,
    }
    weights_bytes = 1024
    sidecar_bytes = 1024
    prefix_scratch_bytes = 256
    subtotal = weights_bytes + 256 + 128 + 64 + 32 + sidecar_bytes + prefix_scratch_bytes
    headroom_bytes = (((subtotal + 9) // 10 + 255) // 256) * 256
    plan = {
        "schema_version": 2,
        "model_spec_id": "2" * 64,
        "weights_digest": "3" * 64,
        "weight_manifest_digest": "4" * 64,
        "runtime_dtype": "fp32",
        "device": "cpu",
        "kv_mode": "none",
        "kv_layout_id": "none-reference-full-context-v1",
        "capacity_tokens": 4,
        "memory_budget_bytes": 4096,
        "weights_bytes": weights_bytes,
        "kv_bytes": 0,
        "activation_bytes": 256,
        "workspace_bytes": 128,
        "io_bytes": 64,
        "telemetry_bytes": 32,
        "headroom_bytes": headroom_bytes,
        "required_bytes": subtotal + headroom_bytes,
        "numerical_policy": policy,
        "numerical_policy_digest": sha256_json(policy),
        "compute_dtype": "fp32",
        "sidecar_bytes": sidecar_bytes,
        "prefix_scratch_bytes": prefix_scratch_bytes,
        "execution_binding": binding,
        "execution_binding_digest": sha256_json(binding),
        "evidence_kind": "static_conservative_estimate",
    }
    identity = {
        "engine_version": "0.2.0.dev0",
        "git_commit": "b" * 40,
        "git_dirty": False,
        "toolchain_digest": sha256_json({"binding": "unbound-development"}),
        "toolchain_bound": False,
        "device": "cpu",
        "dtype": "fp32",
        "model_spec_id": plan["model_spec_id"],
        "weights_digest": plan["weights_digest"],
        "load_manifest_digest": plan["weight_manifest_digest"],
        "tokenizer_digest": "5" * 64,
        "kv_mode": "none",
        "memory_plan_evidence_kind": "static_conservative_estimate",
    }
    return {
        "identity": identity,
        "sampling_spec": SamplingSpec(actual_seed=7).to_dict(),
        "memory_plan": plan,
        "numerical_policy": policy,
        "execution_binding": binding,
        "observation_profile": "request",
    }


def _complete_capture(path: Path) -> dict[str, Any]:
    metadata = _metadata()
    session = PolicyStreamingCaptureSession(
        path,
        metadata["identity"],
        "portable-request",
        metadata["sampling_spec"],
        metadata["memory_plan"],
        numerical_policy=metadata["numerical_policy"],
        execution_binding=metadata["execution_binding"],
        observation_profile=metadata["observation_profile"],
    )
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "TOKENIZED", "to_phase": "PREFILLING"})
    session.emit(
        K.MODEL_FORWARD_STARTED,
        {"phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0},
    )
    session.emit(
        K.MODEL_FORWARD_COMPLETED,
        {"phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0},
    )
    token = {
        "output_index": 0,
        "committed_output_count": 1,
        "kv_valid_token_count": 0,
        "server_ready_ns": 30,
        "token_hmac": "6" * 64,
    }
    session.emit(K.REQUEST_FIRST_TOKEN, token)
    session.emit(K.REQUEST_OUTPUT_CHUNK, token)
    session.emit(
        K.REQUEST_TERMINAL_DECIDED,
        {
            "outcome": "FINISHED",
            "reason": "MAX_NEW_TOKENS",
            "committed_output_count": 1,
            "kv_valid_token_count": 0,
        },
    )
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "PREFILLING", "to_phase": "FINISHED"})
    session.emit(
        K.REQUEST_COMPLETED,
        {
            "finish_reason": "MAX_NEW_TOKENS",
            "committed_output_count": 1,
            "kv_valid_token_count": 0,
            "server_terminal_ready_ns": 40,
        },
    )
    return session.finalize()


def _run(capture: Path, output: Path, *extra: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(capture), "--output", str(output), *extra],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
        check=False,
    )


def _read_report(output: Path) -> dict[str, Any]:
    value = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_build_and_verify_copied_minimal_reader_from_different_cwd(tmp_path: Path) -> None:
    capture = tmp_path / "source-capture"
    manifest = _complete_capture(capture)
    output = tmp_path / "portable-proof"

    created = _run(capture, output, cwd=tmp_path)
    assert created.returncode == 0, created.stderr
    summary = json.loads(created.stdout)
    assert summary["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert summary["qualification_status"] == "unqualified"
    assert summary["manifest_digest"] == manifest["manifest_digest"]
    assert set(item.name for item in output.iterdir()) == {"reader", "capture", "report.json", "COMPLETE"}
    assert set(item.name for item in (output / "capture").iterdir()) == {
        "events.jsonl",
        "checksums.json",
        "capture-manifest.json",
        "COMPLETE",
    }

    report = _read_report(output)
    assert report["report_kind"] == "illm-policy-capture-portable-development-v1"
    assert report["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert report["qualification_status"] == "unqualified"
    assert "not independent source truth" in report["limitations"][1]
    assert report["commands"]["source_validation"]["returncode"] == 0
    assert report["commands"]["copied_validation"]["returncode"] == 0
    assert report["commands"]["source_validation"]["result"] == report["commands"]["copied_validation"]["result"]
    assert report["commands"]["source_validation"]["result"]["forbidden_loaded_modules"] == []
    assert report["commands"]["source_validation"]["result"]["network_events"] == []
    assert report["reader_modules"]
    assert set(report["reader_modules"].values()) == set(report["reader_files"])
    assert all(name.startswith("illm/") and name.endswith(".py") for name in report["reader_files"])
    assert all(_sha256(output / "reader" / name) == digest for name, digest in report["reader_files"].items())

    verified = _run(tmp_path / "source-no-longer-required", output, "--verify-existing", cwd=tmp_path)
    assert verified.returncode == 0, verified.stderr
    verification = json.loads(verified.stdout)
    assert verification["source_capture_argument_used"] is False
    assert verification["manifest_digest"] == manifest["manifest_digest"]
    assert verification["report_sha256"] == (output / "COMPLETE").read_text(encoding="ascii").strip()


@pytest.mark.parametrize("target", ["reader", "capture"])
def test_verify_existing_rejects_copied_artifact_tampering(tmp_path: Path, target: str) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    output = tmp_path / "portable-proof"
    assert _run(capture, output).returncode == 0
    report = _read_report(output)
    if target == "reader":
        relative = next(iter(report["reader_files"]))
        path = output / "reader" / relative
    else:
        path = output / "capture" / "events.jsonl"
    path.write_bytes(path.read_bytes() + b"tamper\n")

    verified = _run(tmp_path / "ignored", output, "--verify-existing")
    assert verified.returncode == 1
    assert "digest mismatch" in verified.stderr


def _rewrite_report_and_completion(output: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    report = _read_report(output)
    mutate(report)
    encoded = canonical_json_bytes(report) + b"\n"
    (output / "report.json").write_bytes(encoded)
    (output / "COMPLETE").write_text(hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report.update(qualification_status="qualified"),
        lambda report: report["limitations"].clear(),
        lambda report: report["commands"]["source_validation"].update(stdout_sha256="0" * 64),
        lambda report: report["input_capture_manifest"].update(capabilities=["R1"]),
    ],
)
def test_resigned_report_cannot_promote_or_rewrite_embedded_facts(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    output = tmp_path / "portable-proof"
    assert _run(capture, output).returncode == 0
    _rewrite_report_and_completion(output, mutate)

    verified = _run(tmp_path / "ignored", output, "--verify-existing")
    assert verified.returncode == 1


@pytest.mark.parametrize("extra", ["top-file", "reader-empty-directory", "capture-extra-file"])
def test_verify_existing_rejects_nonexact_inventory(tmp_path: Path, extra: str) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    output = tmp_path / "portable-proof"
    assert _run(capture, output).returncode == 0
    if extra == "top-file":
        (output / "foreign.txt").write_text("foreign", encoding="utf-8")
    elif extra == "reader-empty-directory":
        (output / "reader" / "foreign").mkdir()
    else:
        (output / "capture" / "foreign.txt").write_text("foreign", encoding="utf-8")

    verified = _run(tmp_path / "ignored", output, "--verify-existing")
    assert verified.returncode == 1
    assert "inventory" in verified.stderr


def test_creation_refuses_existing_output_without_modifying_capture(tmp_path: Path) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    original = {item.name: _sha256(item) for item in capture.iterdir()}
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "owner.txt").write_text("keep", encoding="utf-8")

    completed = _run(capture, output)

    assert completed.returncode == 1
    assert (output / "owner.txt").read_text(encoding="utf-8") == "keep"
    assert {item.name: _sha256(item) for item in capture.iterdir()} == original


def _load_script_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location("policy_capture_portable_test_module", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_failure_after_output_creation_retains_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    output = tmp_path / "portable-proof"
    module = _load_script_module()
    original = cast(Callable[..., dict[str, Any]], getattr(module, "_run_child"))
    calls = 0

    def fail_second(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise cast(type[Exception], getattr(module, "PortableError"))("synthetic copied-reader failure")
        return original(**kwargs)

    monkeypatch.setattr(module, "_run_child", fail_second)
    build = cast(Callable[..., dict[str, Any]], getattr(module, "build_portable_proof"))

    with pytest.raises(Exception, match="synthetic copied-reader failure"):
        build(capture=capture, output=output, python=Path(sys.executable))

    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()
    assert not (output / "report.json").exists()


def test_invalid_or_incomplete_source_capture_is_not_packaged(tmp_path: Path) -> None:
    capture = tmp_path / "source-capture"
    _complete_capture(capture)
    (capture / "COMPLETE").unlink()
    (capture / "INCOMPLETE").write_text("not complete\n", encoding="utf-8")
    output = tmp_path / "portable-proof"

    completed = _run(capture, output)

    assert completed.returncode == 1
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()

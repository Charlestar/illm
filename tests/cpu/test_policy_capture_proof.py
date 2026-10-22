from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.policy_streaming import PolicyStreamingCaptureSession
from illm.domain.streaming_events import StreamingEventKind as K
from illm.numerical_policy import candidate_policy
from illm.policy_capture_contract import PROTOCOL_SHA256
from illm.policy_capture_proof import verify_policy_capture_proof
from illm.policy_cpu_numerical_contract import NumericalEvidenceError
from illm.sampling import SamplingSpec


ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "scripts/check_policy_capture_portable.py"
FORBIDDEN = {"torch", "transformers", "tokenizers", "safetensors"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata() -> dict[str, Any]:
    policy = candidate_policy("fp32").to_dict()
    environment = {
        "backend": "cpu",
        "python_version": "3.12.0",
        "platform": "proof-fixture-platform",
        "machine": "proof-fixture-machine",
        "torch_version": "proof-fixture-torch",
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


def _capture(path: Path, request_id: str) -> None:
    metadata = _metadata()
    session = PolicyStreamingCaptureSession(
        path,
        metadata["identity"],
        request_id,
        metadata["sampling_spec"],
        metadata["memory_plan"],
        numerical_policy=metadata["numerical_policy"],
        execution_binding=metadata["execution_binding"],
        observation_profile=metadata["observation_profile"],
    )
    session.emit(K.REQUEST_TOKENIZED, {"started_ns": 10, "ended_ns": 20, "input_tokens": 2})
    session.emit(K.REQUEST_ACCEPTED, {"input_tokens": 2, "max_new_tokens": 1})
    session.emit(K.REQUEST_PHASE_CHANGED, {"from_phase": "TOKENIZED", "to_phase": "PREFILLING"})
    forward = {"phase": "prefill", "query_tokens": 2, "context_tokens": 2, "output_position": 0}
    session.emit(K.MODEL_FORWARD_STARTED, forward)
    session.emit(K.MODEL_FORWARD_COMPLETED, forward)
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
    session.finalize()


def _build(capture: Path, output: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(PRODUCER), str(capture), "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _report(path: Path) -> dict[str, Any]:
    value = json.loads((path / "report.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _source_files(proof: Path) -> dict[str, str]:
    report = _report(proof)
    result = {"scripts/check_policy_capture_portable.py": _sha(PRODUCER)}
    for name in cast(dict[str, str], report["reader_files"]):
        result["src/" + name] = _sha(ROOT / "src" / Path(name))
    return result


def _resign_report(proof: Path, report: dict[str, Any]) -> None:
    raw = canonical_json_bytes(report) + b"\n"
    (proof / "report.json").write_bytes(raw)
    (proof / "COMPLETE").write_bytes((hashlib.sha256(raw).hexdigest() + "\n").encode("ascii"))


@pytest.fixture(scope="module")
def archived_proof(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path_factory.mktemp("policy-capture-proof")
    capture = root / "capture-source"
    proof = root / "proof"
    _capture(capture, "proof-request")
    _build(capture, proof)
    return proof, capture, _source_files(proof)


def test_archived_proof_is_recomputed_without_upgrading_qualification(
    archived_proof: tuple[Path, Path, dict[str, str]],
) -> None:
    proof, capture, source_files = archived_proof
    summary = verify_policy_capture_proof(
        proof,
        expected_capture=capture,
        producer_source=PRODUCER,
        source_files=source_files,
    )

    assert summary["evidence_stage"] == "DEVELOPMENT_ONLY"
    assert summary["qualification_status"] == "unqualified"
    assert summary["archived_transcripts_only"] is True
    assert summary["reader_executed_by_verifier"] is False
    assert summary["reader_module_count"] == len(_report(proof)["reader_modules"])


@pytest.mark.parametrize("mutation", ["qualification", "network", "bool-returncode"])
def test_fully_resigned_report_cannot_promote_or_hide_child_failure(
    archived_proof: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
    mutation: str,
) -> None:
    base, capture, source_files = archived_proof
    proof = tmp_path / "proof"
    shutil.copytree(base, proof)
    report = _report(proof)
    if mutation == "qualification":
        report["qualification_status"] = "qualified"
    elif mutation == "network":
        for name in ("source_validation", "copied_validation"):
            command = report["commands"][name]
            result = command["result"]
            result["network_events"] = ["socket.connect"]
            command["stdout"] = canonical_json_bytes(result).decode("utf-8") + "\n"
            command["stdout_sha256"] = hashlib.sha256(command["stdout"].encode("utf-8")).hexdigest()
    else:
        report["commands"]["copied_validation"]["returncode"] = True
    _resign_report(proof, report)

    with pytest.raises((NumericalEvidenceError, ValueError), match="portable|development-only|command"):
        verify_policy_capture_proof(
            proof,
            expected_capture=capture,
            producer_source=PRODUCER,
            source_files=source_files,
        )


def test_expected_profile_capture_and_profile_source_are_authorities(
    archived_proof: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    proof, _, source_files = archived_proof
    other_capture = tmp_path / "other-capture"
    _capture(other_capture, "other-request")
    with pytest.raises(NumericalEvidenceError, match="profile capture"):
        verify_policy_capture_proof(
            proof,
            expected_capture=other_capture,
            producer_source=PRODUCER,
            source_files=source_files,
        )

    changed_source = dict(source_files)
    one_reader = next(name for name in changed_source if name.startswith("src/"))
    changed_source[one_reader] = "0" * 64
    with pytest.raises(NumericalEvidenceError, match="profile source"):
        verify_policy_capture_proof(
            proof,
            expected_capture=archived_proof[1],
            producer_source=PRODUCER,
            source_files=changed_source,
        )


def test_resigned_reader_file_change_still_fails_profile_source_binding(
    archived_proof: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    base, capture, source_files = archived_proof
    proof = tmp_path / "proof"
    shutil.copytree(base, proof)
    report = _report(proof)
    relative = next(iter(cast(dict[str, str], report["reader_files"])))
    member = proof / "reader" / Path(relative)
    member.write_bytes(member.read_bytes() + b"\n# resigned mutation\n")
    digest = _sha(member)
    report["reader_files"][relative] = digest
    report["source_dependency_snapshot"]["files"][relative] = digest
    _resign_report(proof, report)

    with pytest.raises(NumericalEvidenceError, match="profile source"):
        verify_policy_capture_proof(
            proof,
            expected_capture=capture,
            producer_source=PRODUCER,
            source_files=source_files,
        )


def test_proof_reader_import_is_stdlib_only_in_isolated_python() -> None:
    source_root = ROOT / "src"
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import illm.policy_capture_proof;"
        f"blocked={FORBIDDEN!r};"
        "loaded=blocked.intersection({name.split('.')[0] for name in sys.modules});"
        "assert not loaded, sorted(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    assert result.returncode == 0, result.stderr

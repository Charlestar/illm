"""Control-plane unit checks, not execution evidence or release qualification."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from illm.canonical import sha256_json
from illm.stateful_gate_contract import STATEFUL_PROFILE_RUNTIME_IDS, phase_parameters


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bootstrap
import run_v02_gate as gate
import diagnose_stateful
from _toolchain_common import ToolchainFailure


@pytest.mark.parametrize("profile", tuple(STATEFUL_PROFILE_RUNTIME_IDS))
def test_shared_contract_and_argv_select_the_complete_profile(profile: str, tmp_path: Path) -> None:
    commands = gate.phase_commands(profile, tmp_path)
    parameters = phase_parameters(profile)
    assert list(commands) == list(parameters)[1:]
    assert len(parameters) == (9 if "single-gpu" in profile else 11)
    assert all("--development-no-manifest" not in command for command in commands.values())
    if "single-gpu" in profile:
        for dtype in ("bf16", "fp16"):
            assert commands[f"stateful-{dtype}"][-4:-2] == ["--scope", "real-smoke"]
            for suite in ("tuning", "heldout"):
                command = commands[f"benchmark-{dtype}-{suite}"]
                assert command[command.index("--target") + 1] == "real"
                assert command[command.index("--suite") + 1] == suite
    else:
        assert "--verify-existing" in commands["learning-verify"]
        assert "verify" in commands["verify"]
        assert "inspect" in commands["inspect-request"]


def test_contract_returns_independent_values() -> None:
    first = phase_parameters("stateful-streaming-single-gpu-v1")
    first["benchmark-bf16-heldout"]["groups"].clear()
    assert len(phase_parameters("stateful-streaming-single-gpu-v1")["benchmark-bf16-heldout"]["groups"]) == 4


def test_gate_preserves_existing_environment_before_spawning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = gate._contract()
    project = tmp_path / "project"
    environment = project / ".illm/envs/existing"
    environment.mkdir(parents=True)
    sentinel = environment / "notes.txt"
    sentinel.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(gate, "PROJECT_ROOT", project)
    monkeypatch.setattr(gate, "_contract", lambda: contract)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(SystemExit):
        gate.run_gate(["--profile", "existing", "--profile-id", "state-machine-cpu-v1", "--output", str(project / "out/new")])
    assert sentinel.read_text(encoding="ascii") == "preserve"
    assert not (project / "out").exists()


def test_bootstrap_fresh_flag_never_calls_replacement_or_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    environment = project / ".illm/envs/existing"
    environment.mkdir(parents=True)
    sentinel = environment / "notes.txt"
    sentinel.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(bootstrap, "__file__", str(project / "scripts/bootstrap.py"))
    monkeypatch.setattr(bootstrap, "_safe_remove_environment", lambda *a: pytest.fail("must not remove"))
    monkeypatch.setattr(bootstrap, "_install", lambda *a: pytest.fail("must not install"))
    with pytest.raises(ToolchainFailure, match="already exists"):
        bootstrap.bootstrap(["--profile", "existing", "--locked", "--require-new-environment",
                             "--write-manifest", str(project / "out/toolchain.json")])
    assert sentinel.read_text(encoding="ascii") == "preserve"


def test_gate_retains_failed_phase_and_cannot_qualify_from_exit_zero_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = gate._contract()
    project, output = tmp_path / "project", tmp_path / "project/out/gate"
    monkeypatch.setattr(gate, "PROJECT_ROOT", project)
    monkeypatch.setattr(gate, "_contract", lambda: contract)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(gate, "lock_digest", lambda project: "b" * 64)
    invoked: list[list[str]] = []

    def fake_process(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        invoked.append(command)
        if "--write-manifest" in command:
            manifest = {"manifest_digest": "a" * 64, "project_root_digest": "c" * 64, "bootstrap_digest": "d" * 64}
            Path(command[command.index("--write-manifest") + 1]).write_text(json.dumps(manifest), encoding="utf-8")
            Path(command[command.index("--write-manifest-digest") + 1]).write_text("a" * 64, encoding="ascii")
        return subprocess.CompletedProcess(command, 0, b'{"synthetic_unit_only":true}\n', b"")

    monkeypatch.setattr(subprocess, "run", fake_process)
    arguments = ["--profile", "new", "--profile-id", "state-machine-cpu-v1", "--output", str(output)]
    with pytest.raises(RuntimeError, match="completion artifact"):
        gate.run_gate(arguments)
    assert "--require-new-environment" in invoked[0]
    for command in invoked[1:]:
        assert Path(command[1]).name == "in_env.py"
        assert "--expected-manifest-digest-file" in command
    report = json.loads((output / "gate-report.json").read_text(encoding="utf-8"))
    assert [record["phase"] for record in report["commands"]] == list(phase_parameters("state-machine-cpu-v1"))
    assert report["report_digest"] == sha256_json({key: value for key, value in report.items() if key != "report_digest"})
    assert (output / "INCOMPLETE").is_file()
    assert not (output / "COMPLETE").exists()


def test_actual_tiny_first_op_diagnosis_restores_policy_and_never_overwrites(tmp_path: Path) -> None:
    import torch

    previous = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    output = tmp_path / "diagnosis.json"
    report = diagnose_stateful.diagnose("fixture:tiny-decoder", "cpu", "hello", output, full_precision_reduction=True)
    assert report["first_last_query_divergence"] is None
    assert report["first_cached_prefix_divergence"] is None
    assert report["formal_evidence_eligible"] is False
    assert previous == (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="new path"):
        diagnose_stateful.diagnose("fixture:tiny-decoder", "cpu", "hello", output)
    assert output.read_bytes() == before


def test_cached_precision_probe_never_changes_the_reference_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from torch import Tensor
    from torch.nn import functional as F
    from illm.model.reference import ReferenceDecoder

    native_forward, native_linear, native_matmul = ReferenceDecoder.forward, F.linear, Tensor.__matmul__
    calls = 0

    def original_oracle(self: ReferenceDecoder, *args: Any, **kwargs: Any) -> Tensor:
        nonlocal calls
        calls += 1
        assert F.linear is native_linear and Tensor.__matmul__ is native_matmul
        return native_forward(self, *args, **kwargs)

    monkeypatch.setattr(ReferenceDecoder, "forward", original_oracle)
    report = diagnose_stateful.diagnose("fixture:tiny-decoder", "cpu", "hello", tmp_path / "candidate.json",
                                       cached_policy="fp32-contractions")
    assert calls == 1
    assert report["oracle_policy"] == "v01-full-context-unchanged-v1"
    assert report["cached_contraction_policy"] == "fp32-contractions"
    assert report["formal_evidence_eligible"] is False
    assert F.linear is native_linear and Tensor.__matmul__ is native_matmul


def test_contraction_probe_restores_methods_on_exception() -> None:
    from torch import Tensor
    from torch.nn import functional as F

    original = F.linear, Tensor.__matmul__
    with pytest.raises(RuntimeError, match="probe failed"):
        with diagnose_stateful._cached_contractions("fp32-contractions"):
            assert F.linear is not original[0]
            raise RuntimeError("probe failed")
    assert (F.linear, Tensor.__matmul__) == original

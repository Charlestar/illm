from __future__ import annotations

from pathlib import Path

import pytest

from illm.cli import main


def test_demo_r0_r1_and_stable_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = tmp_path / "capture"
    assert main(
        [
            "demo",
            "toy-scheduler",
            "--seed",
            "7",
            "--steps",
            "20",
            "--capture",
            str(capture),
        ]
    ) == 0
    assert main(["replay", str(capture), "--mode", "trace", "--step-id", "0"]) == 0
    assert main(["replay", str(capture), "--mode", "state", "--assert-state-digest"]) == 0
    assert main(["replay"]) == 64
    assert "R1_TOY" in capsys.readouterr().out


def test_trace_filter_is_rejected_for_state_mode(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    assert main(
        [
            "demo",
            "toy-scheduler",
            "--seed",
            "1",
            "--steps",
            "5",
            "--capture",
            str(capture),
        ]
    ) == 0
    assert main(["replay", str(capture), "--mode", "state", "--request-id", "req-0000"]) == 64

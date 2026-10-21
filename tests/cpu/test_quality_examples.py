"""The existing strict type gate includes the public V0.2 API example."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_quality


def test_v02_public_example_is_an_explicit_mypy_target() -> None:
    relative = "docs/examples/v0_2_streaming.py"
    assert relative in check_quality.MYPY_TARGETS
    assert (ROOT / relative).is_file()


def test_quality_mypy_command_includes_v02_public_example() -> None:
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Success: no issues found\n",
        stderr="",
    )
    with (
        patch("subprocess.run", return_value=completed) as run,
        patch("importlib.metadata.version", return_value="test"),
    ):
        result, issues = check_quality._run_mypy(ROOT)

    command = run.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "mypy"]
    assert command[3:] == list(check_quality.MYPY_TARGETS)
    assert "docs/examples/v0_2_streaming.py" in command
    assert result["status"] == "pass"
    assert not issues

"""Portable boundaries for the policy-aware lifecycle sibling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from illm.domain.errors import ErrorCode, IllmError
from illm.stateful_policy_exercise import (
    COMPLETE_CAPTURE_IDS,
    CYCLE_COUNT,
    INCOMPLETE_CAPTURE_IDS,
    RACE_COUNT,
    REPORT_KIND,
    REQUIRED_FAULT_IDS,
    REQUIRED_POLICY_CAPTURE_IDS,
    SCOPE,
    _decode_json,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import exercise_policy_streaming as exercise


def test_policy_lifecycle_contract_has_independent_fixed_identity() -> None:
    assert REPORT_KIND == "illm-v02-policy-streaming-lifecycle-exercise-v1"
    assert SCOPE == "tiny-policy-state-machine-cpu-v2"
    assert CYCLE_COUNT == 100 and RACE_COUNT == 8
    assert len(REQUIRED_FAULT_IDS) == 12
    assert REQUIRED_POLICY_CAPTURE_IDS == (
        "success",
        "length",
        "eos",
        "cancel-after-launch",
        "no-kv",
    )
    assert COMPLETE_CAPTURE_IDS[-1] == "exporter-stall"
    assert INCOMPLETE_CAPTURE_IDS == (
        "required-write-failure",
        "unknown-fence",
        "forced-exit",
    )


def test_formal_policy_lifecycle_requires_bound_manifest_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    monkeypatch.delenv("ILLM_LOCK_DIGEST", raising=False)
    output = tmp_path / "must-not-exist"
    with pytest.raises(IllmError) as raised:
        exercise.run_policy_exercise(output=output)
    assert raised.value.code is ErrorCode.TOOLCHAIN_MISMATCH
    assert not output.exists()


def test_development_writer_refuses_foreign_output_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ILLM_TOOLCHAIN_MANIFEST", raising=False)
    monkeypatch.delenv("ILLM_ENVIRONMENT_ID", raising=False)
    monkeypatch.delenv("ILLM_LOCK_DIGEST", raising=False)
    output = tmp_path / "foreign"
    output.mkdir()
    sentinel = output / "sentinel.bin"
    sentinel.write_bytes(b"foreign-must-remain")
    with pytest.raises(IllmError) as raised:
        exercise.run_policy_exercise(output=output, development_no_manifest=True)
    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert sentinel.read_bytes() == b"foreign-must-remain"
    assert list(output.iterdir()) == [sentinel]


def test_portable_reader_imports_without_tensor_runtime() -> None:
    code = (
        f"import sys;sys.path.insert(0,{str(ROOT / 'src')!r});"
        "import illm.stateful_policy_exercise as m;"
        "assert m.REPORT_KIND.endswith('exercise-v1');"
        "assert 'torch' not in sys.modules;print('policy-reader-portable')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "policy-reader-portable"


@pytest.mark.parametrize("payload", ["{\"x\":NaN}", "{\"x\":Infinity}", "{\"x\":1e999}"])
def test_portable_json_decoder_rejects_every_nonfinite_spelling(payload: str) -> None:
    with pytest.raises(IllmError) as raised:
        _decode_json(payload, artifact="forged.json")
    assert raised.value.code is ErrorCode.CAPTURE_CORRUPT


def test_policy_producer_source_does_not_import_old_producer_or_test_framework() -> None:
    source = (ROOT / "scripts" / "exercise_policy_streaming.py").read_text(encoding="utf-8")
    assert "import exercise_streaming" not in source
    assert "import pytest" not in source
    assert "from tests" not in source
    assert "numerical_policy=POLICY" in source


def test_subprocess_environment_is_not_modified_by_import() -> None:
    before = dict(os.environ)
    __import__("illm.stateful_policy_exercise")
    assert dict(os.environ) == before

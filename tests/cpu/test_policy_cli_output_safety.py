from __future__ import annotations

import argparse
import importlib.util
import os
import secrets
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from illm.api import streaming_cli
from illm.canonical import canonical_json_bytes
from illm.domain.errors import ErrorCode, IllmError


ROOT = Path(__file__).resolve().parents[2]


def _trace() -> dict[str, Any]:
    return {"traceEvents": [], "metadata": {"capability": "R0"}}


def test_trace_publish_preserves_predictable_foreign_temp_and_leaves_no_owned_temp(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "trace.json"
    predictable = tmp_path / f".trace.json.tmp-{os.getpid()}"
    predictable.write_bytes(b"FOREIGN")

    streaming_cli._write_new_json(destination, _trace())

    assert destination.read_bytes() == canonical_json_bytes(_trace()) + b"\n"
    assert predictable.read_bytes() == b"FOREIGN"
    assert set(tmp_path.iterdir()) == {destination, predictable}


def test_random_temp_collision_is_not_deleted_or_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "trace.json"
    monkeypatch.setattr(secrets, "token_hex", lambda _size: "fixed-token")
    collision = tmp_path / f".trace.json.tmp-{os.getpid()}-fixed-token"
    collision.write_bytes(b"FOREIGN")

    with pytest.raises(IllmError, match="temporary output already exists") as raised:
        streaming_cli._write_new_json(destination, _trace())

    assert raised.value.code is ErrorCode.INVALID_ARGUMENT
    assert collision.read_bytes() == b"FOREIGN"
    assert not destination.exists()


def test_destination_created_during_publish_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "trace.json"
    real_link = os.link

    def racing_link(source: Path, target: Path) -> None:
        Path(target).write_bytes(b"FOREIGN_DESTINATION")
        real_link(source, target)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(IllmError, match="output already exists"):
        streaming_cli._write_new_json(destination, _trace())

    assert destination.read_bytes() == b"FOREIGN_DESTINATION"
    assert not list(tmp_path.glob(".trace.json.tmp-*"))


def test_unsupported_hardlink_fails_closed_without_fallback_or_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "trace.json"

    def unsupported(_source: object, _target: object) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(os, "link", unsupported)
    with pytest.raises(IllmError, match="without overwrite"):
        streaming_cli._write_new_json(destination, _trace())

    assert not destination.exists()
    assert not list(tmp_path.glob(".trace.json.tmp-*"))


def test_linklike_ancestor_is_rejected_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ancestor = (tmp_path / "linklike").absolute()
    ancestor.mkdir()
    destination = ancestor / "trace.json"
    real_lstat = Path.lstat

    def simulated_lstat(path: Path) -> os.stat_result | SimpleNamespace:
        if path == ancestor:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_lstat)
    with pytest.raises(IllmError, match="link or reparse"):
        streaming_cli._write_new_json(destination, _trace())

    assert not destination.exists()


def test_inspect_passes_original_output_path_to_safe_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()

    class Bundle:
        root = capture

        def request_story(self, request_id: str) -> dict[str, Any]:
            return {"request_id": request_id}

        def chrome_trace(self) -> dict[str, Any]:
            return _trace()

    supplied = Path("derived") / "trace.json"
    observed: list[Path] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(streaming_cli, "_streaming_bundle", lambda _path: Bundle())
    monkeypatch.setattr(
        streaming_cli,
        "_write_new_json",
        lambda path, _value: observed.append(path),
    )
    args = argparse.Namespace(capture=capture, request="req", chrome_trace=supplied)

    assert streaming_cli._inspect_request(args) == 0
    assert observed == [supplied]
    assert '"request_id":"req"' in capsys.readouterr().out


def test_policy_example_report_collision_preserves_foreign_file_and_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "illm_policy_capture_example_safety",
        ROOT / "docs" / "examples" / "v0_2_policy_capture.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    example: Any = module
    calls: list[str] = []

    def fake_case(root: Path, name: str, mode: str, *, cancel: bool = False) -> dict[str, Any]:
        if not calls:
            (root / "demo-report.json").write_bytes(b"FOREIGN_REPORT")
        calls.append(name)
        return {
            "case": name,
            "outcome": "CANCELLED" if cancel else "FINISHED",
            "output_token_ids": [1],
            "query_tokens_per_step": [6, 7, 8, 9] if mode == "none" else [6, 1, 1, 1],
        }

    monkeypatch.setattr(module, "run_case", fake_case)
    output = tmp_path / "example"
    with pytest.raises(RuntimeError, match="refuses to overwrite"):
        example.main(["--output", str(output)])

    assert (output / "demo-report.json").read_bytes() == b"FOREIGN_REPORT"
    assert (output / "INCOMPLETE").is_file()

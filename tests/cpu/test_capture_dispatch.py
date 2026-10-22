"""Version dispatch is strict and never substitutes for a family's verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from illm.capture.dispatch import capture_schema, open_capture, open_streaming_capture
from illm.domain.errors import ErrorCode, IllmError
from illm.toy import run_toy_scheduler


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_known_discriminator_does_not_claim_a_valid_bundle(tmp_path: Path, version: int) -> None:
    (tmp_path / "capture-manifest.json").write_text('{"schema_version":' + str(version) + '}', encoding="utf-8")
    assert capture_schema(tmp_path) == version
    with pytest.raises(IllmError):
        open_capture(tmp_path)


@pytest.mark.parametrize("raw", [
    '{"schema_version":true}', '{"schema_version":3.0}', '{"schema_version":99}',
    '{"schema_version":3,"schema_version":2}',
    '{"schema_version":2,"nested":{"x":1,"x":2}}',
    '{"schema_version":2,"x":NaN}', '[]', '{invalid',
])
def test_ambiguous_or_unknown_header_is_rejected(tmp_path: Path, raw: str) -> None:
    (tmp_path / "capture-manifest.json").write_text(raw, encoding="utf-8")
    with pytest.raises(IllmError):
        capture_schema(tmp_path)


def test_retained_toy_reader_and_capabilities_are_not_widened(tmp_path: Path) -> None:
    root = tmp_path / "toy"
    run_toy_scheduler(seed=7, steps=4, capture_root=root, project_root=Path.cwd())
    assert capture_schema(root) == 0
    assert open_capture(root).manifest["schema_version"] == 0
    with pytest.raises(IllmError) as caught:
        open_streaming_capture(root)
    assert caught.value.code is ErrorCode.UNSUPPORTED_SCHEMA


def test_linked_discriminator_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"schema_version":2}', encoding="utf-8")
    root = tmp_path / "capture"
    root.mkdir()
    try:
        (root / "capture-manifest.json").symlink_to(source)
    except OSError:
        pytest.skip("host does not permit symlinks")
    with pytest.raises(IllmError):
        capture_schema(root)

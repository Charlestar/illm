from __future__ import annotations

import binascii
import importlib.util
import json
import struct
import zlib
from pathlib import Path
from typing import Callable, Sequence, cast

import pytest

pytest.importorskip("torch")

from illm.canonical import canonical_json_bytes, sha256_json
from illm.domain.errors import IllmError
from illm.engine.streaming import Engine
from illm.engine.streaming_contracts import GenerationRequest
from illm.sampling import SamplingSpec
from illm.streaming_learning import (
    LEARNING_FILES, WIDTH, derive_streaming_learning_assets, verify_streaming_learning_assets,
)

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "illm_render_learning_test", Path(__file__).resolve().parents[2] / "scripts" / "render_streaming_learning.py",
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
render_main = cast(Callable[[Sequence[str]], int], _SCRIPT.main)


@pytest.fixture()
def captured(tmp_path: Path) -> Path:
    engine = Engine()
    capture = tmp_path / "capture"
    try:
        result = engine.generate(GenerationRequest(engine.model_spec_id, 4, SamplingSpec(), prompt="hello",
                                                  request_id="learning-fixed", capture=capture))
        assert result.capture_error is None
    finally:
        engine.close()
    return capture


def test_png_svg_trace_are_exactly_derived_and_portable(captured: Path, tmp_path: Path) -> None:
    assets = derive_streaming_learning_assets(captured)
    assert assets == derive_streaming_learning_assets(captured)
    assert set(assets) == LEARNING_FILES
    directory = tmp_path / "views"
    directory.mkdir()
    for name, data in assets.items():
        (directory / name).write_bytes(data)
    assert verify_streaming_learning_assets(captured, directory)["status"] == "pass"
    story = json.loads(assets["request-story.json"])
    assert story["terminal_outcome"] == story["logical_terminal_outcome"] == "FINISHED"
    assert story["kv"]["final_active_allocation_bytes"] == 0
    svg = assets["request-story.svg"].decode()
    assert "DERIVED FROM RECORDED R0 FACTS" in svg
    assert "NOT A LIVE UI" in svg
    assert "SHOWN 4 / 4" in svg
    png = assets["request-story.png"]
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert width == WIDTH and height >= 400
    offset = 8
    image_data = bytearray()
    while offset < len(png):
        length = struct.unpack(">I", png[offset:offset + 4])[0]
        kind = png[offset + 4:offset + 8]
        data = png[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", png[offset + 8 + length:offset + 12 + length])[0]
        assert expected_crc == binascii.crc32(kind + data) & 0xFFFFFFFF
        if kind == b"IDAT":
            image_data.extend(data)
        offset += length + 12
    pixels = zlib.decompress(bytes(image_data))
    assert len(pixels) == height * (1 + width * 3)
    assert len(set(pixels)) > 10


@pytest.mark.parametrize("mutation", ["png", "svg", "provenance", "extra"])
def test_resigning_outer_metadata_cannot_hide_changed_learning_facts(captured: Path, tmp_path: Path, mutation: str) -> None:
    assets = derive_streaming_learning_assets(captured)
    directory = tmp_path / "views"
    directory.mkdir()
    for name, data in assets.items():
        (directory / name).write_bytes(data)
    if mutation == "png":
        (directory / "request-story.png").write_bytes(assets["request-story.png"] + b"modified")
    elif mutation == "svg":
        (directory / "request-story.svg").write_bytes(assets["request-story.svg"].replace(b"SHOWN 4 / 4", b"SHOWN 9 / 9"))
    elif mutation == "extra":
        (directory / "extra.txt").write_text("not a declared learning asset", encoding="utf-8")
    else:
        provenance = json.loads(assets["provenance.json"])
        provenance["evidence_kind"] = "measured"
        provenance["provenance_digest"] = sha256_json({key: value for key, value in provenance.items() if key != "provenance_digest"})
        (directory / "provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")
    with pytest.raises(IllmError):
        verify_streaming_learning_assets(captured, directory)


def test_render_cli_does_not_overwrite_user_files_or_mutate_capture(captured: Path, tmp_path: Path) -> None:
    directory = tmp_path / "existing"
    directory.mkdir()
    sentinel = directory / "notes.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    before_capture = {path.name: path.read_bytes() for path in captured.iterdir()}
    assert render_main([str(captured), "--output", str(directory)]) == 64
    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert render_main([str(captured), "--output", str(captured / "views")]) == 64
    assert {path.name: path.read_bytes() for path in captured.iterdir()} == before_capture


def test_render_cli_verify_existing_is_read_only(captured: Path, tmp_path: Path) -> None:
    directory = tmp_path / "views"
    assert render_main([str(captured), "--output", str(directory)]) == 0
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert render_main([str(captured), "--output", str(directory), "--verify-existing"]) == 0
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    (directory / "request-story.svg").write_bytes(b"not the recorded request")
    modified = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert render_main([str(captured), "--output", str(directory), "--verify-existing"]) != 0
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == modified

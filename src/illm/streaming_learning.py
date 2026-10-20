"""Deterministic Request Story assets derived from validated R0 facts.

PNG is rendered with a tiny ASCII bitmap font and stored-DEFLATE, using only the
stdlib. It is a derived learning diagram, never a screenshot of a live Web UI
or an independent GPU profiler measurement. Rebuilding binds every visible row.
"""

from __future__ import annotations

import binascii
import html
import struct
import zlib
from pathlib import Path
from typing import Any

from illm.canonical import canonical_json_bytes, sha256_json
from illm.capture.streaming import StreamingCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.streaming_events import StreamingEventKind as K


LEARNING_FILES = frozenset({"request-story.json", "chrome-trace.json", "request-story.svg", "request-story.png", "provenance.json"})
WIDTH = 1280
MAX_VISIBLE_STEPS = 12
_FONT_ROWS = {
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01111",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "11111/00100/00100/00100/00100/00100/11111",
    "J": "00111/00010/00010/00010/10010/10010/01100",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    ".": "00000/00000/00000/00000/00000/00110/00110",
    ":": "00000/00110/00110/00000/00110/00110/00000",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    "_": "00000/00000/00000/00000/00000/00000/11111",
    "/": "00001/00010/00010/00100/01000/01000/10000",
    "=": "00000/11111/00000/11111/00000/00000/00000",
    "+": "00000/00100/00100/11111/00100/00100/00000",
    "?": "01110/10001/00001/00010/00100/00000/00100",
    " ": "00000/00000/00000/00000/00000/00000/00000",
}


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def _encode_png(width: int, height: int, pixels: bytearray) -> bytes:
    scanlines = b"".join(b"\0" + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))
    blocks = bytearray(b"\x78\x01")
    for start in range(0, len(scanlines), 65535):
        block = scanlines[start:start + 65535]
        blocks.extend(bytes([int(start + len(block) == len(scanlines))]))
        blocks.extend(struct.pack("<HH", len(block), len(block) ^ 0xFFFF))
        blocks.extend(block)
    blocks.extend(struct.pack(">I", zlib.adler32(scanlines) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", bytes(blocks)) + _png_chunk(b"IEND", b""))


class _Canvas:
    def __init__(self, height: int) -> None:
        self.height = height
        self.pixels = bytearray(bytes((14, 23, 40)) * WIDTH * height)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
                    f'<rect width="{WIDTH}" height="{height}" fill="#0e1728"/>']

    def rect(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x), min(WIDTH, x + width)
        y0, y1 = max(0, y), min(self.height, y + height)
        for row in range(y0, y1):
            self.pixels[(row * WIDTH + x0) * 3:(row * WIDTH + x1) * 3] = bytes(color) * (x1 - x0)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="rgb{color}"/>')

    def text(self, x: int, y: int, text: str, *, size: int = 2,
             color: tuple[int, int, int] = (222, 232, 245)) -> None:
        text = text.upper()
        # Both renderers use the same explicitly defined glyphs, not a system
        # font. This makes SVG/PNG geometry identical on all qualified hosts.
        for offset, character in enumerate(text):
            rows = _FONT_ROWS.get(character, _FONT_ROWS["?"]).split("/")
            for gy, bits in enumerate(rows):
                for gx, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(x + offset * 6 * size + gx * size, y + gy * size, size, size, color)
        self.svg.append(f'<desc>{html.escape(text)}</desc>')

    def assets(self) -> tuple[bytes, bytes]:
        return _encode_png(WIDTH, self.height, self.pixels), ("\n".join((*self.svg, "</svg>")) + "\n").encode("utf-8")


def derive_streaming_learning_assets(capture: Path) -> dict[str, bytes]:
    """Rebuild immutable views from a complete, checksum-validated R0 capture."""
    bundle = StreamingCaptureBundle.open(capture)
    manifest = bundle.manifest
    request_id = str(manifest["request_id"])
    story = bundle.request_story(request_id)
    accepted = next(event.monotonic_ns for event in bundle.events if event.kind is K.REQUEST_ACCEPTED)
    decided = next(event.monotonic_ns for event in bundle.events if event.kind is K.REQUEST_TERMINAL_DECIDED)
    ready_event = next(event for event in bundle.events if event.kind in {K.REQUEST_COMPLETED, K.REQUEST_CANCELLED, K.REQUEST_FAILED})
    end = ready_event.monotonic_ns
    span = max(1, end - accepted)
    started: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    committed_kv = {int(event.payload["output_index"]): int(event.payload["kv_valid_token_count"])
                    for event in bundle.events if event.kind is K.REQUEST_OUTPUT_CHUNK}
    for event in bundle.events:
        if event.kind is K.MODEL_FORWARD_STARTED:
            started[int(event.payload["output_position"])] = event
        elif event.kind is K.MODEL_FORWARD_COMPLETED:
            position = int(event.payload["output_position"])
            begin = started.pop(position, None)
            if begin is None:
                raise IllmError(ErrorCode.CAPTURE_CORRUPT, "learning view forward has no matching start")
            rows.append({"output_position": position, "phase": str(event.payload["phase"]),
                         "query_tokens": int(event.payload["query_tokens"]),
                         "context_tokens": int(event.payload["context_tokens"]),
                         "committed_kv_tokens": committed_kv.get(position),
                         "start_relative_ns": begin.monotonic_ns - accepted,
                         "end_relative_ns": event.monotonic_ns - accepted})
    visible = rows[:MAX_VISIBLE_STEPS]
    height = 380 + max(1, len(visible)) * 32
    canvas = _Canvas(height)
    canvas.text(28, 25, "ILLM V0.2 REQUEST STORY", size=3, color=(94, 224, 199))
    canvas.text(28, 65, "DERIVED FROM RECORDED R0 FACTS - NOT A LIVE UI OR GPU KERNEL TRACE")
    display_id = request_id if len(request_id) <= 70 else request_id[:67] + "..."
    canvas.text(28, 94, f"REQUEST: {display_id}")
    canvas.text(28, 121, f"OUTCOME: {manifest['terminal_outcome']}   PRIVACY: METADATA_ONLY   COMPLETE: YES")
    canvas.text(28, 148, f"CAPTURE SHA256: {manifest['manifest_digest'][:64]}", size=2)
    canvas.text(28, 177, "PREFILL PRODUCES OUTPUT 0. NEXT DECODE CONSUMES THE PREVIOUS OUTPUT.")
    canvas.rect(24, 210, 1232, 30, (31, 49, 72))
    canvas.text(32, 218, "STEP   PHASE          QUERY   CONTEXT   KV VALID   HOST TIMELINE")
    for index, row in enumerate(visible):
        y = 251 + index * 32
        if index % 2 == 0:
            canvas.rect(24, y - 5, 1232, 29, (19, 32, 51))
        valid = "-" if row["committed_kv_tokens"] is None else str(row["committed_kv_tokens"])
        canvas.text(32, y, f"{row['output_position']:>3}    {row['phase']:<13} {row['query_tokens']:>5}   {row['context_tokens']:>7}   {valid:>8}")
        x0 = 760 + row["start_relative_ns"] * 470 // span
        x1 = 760 + row["end_relative_ns"] * 470 // span
        color = (94, 224, 199) if row["phase"] == "PREFILLING" else (114, 160, 255)
        canvas.rect(x0, y - 1, max(2, x1 - x0), 17, color)
    footer = 258 + max(1, len(visible)) * 32
    canvas.text(28, footer, f"SHOWN {len(visible)} / {len(rows)} FORWARDS. FULL EVENTS: REQUEST-STORY.JSON / CHROME-TRACE.JSON")
    canvas.text(28, footer + 28, f"DECISION EVENT: +{decided - accepted} NS. READY EVENT: +{end - accepted} NS.")
    canvas.text(28, footer + 56, "ADDRESSES REUSABLE ONLY AFTER FENCE. FINAL ACTIVE KV BYTES: 0.", color=(94, 224, 199))
    canvas.text(28, footer + 84, "BAR POSITIONS ARE DERIVED HOST TIME. NO RAW PROMPT, TEXT OR TOKEN IDS ARE SHOWN.")
    png, svg = canvas.assets()
    assets = {"request-story.json": canonical_json_bytes(story) + b"\n",
              "chrome-trace.json": canonical_json_bytes(bundle.chrome_trace()) + b"\n",
              "request-story.png": png, "request-story.svg": svg}
    import hashlib

    provenance = {"schema_version": 1, "generator": "illm-stateful-request-story-v1",
                  "evidence_kind": "derived", "not_live_ui_screenshot": True,
                  "source_capture_manifest_digest": manifest["manifest_digest"],
                  "source_model_spec_id": manifest["identity"]["model_spec_id"],
                  "source_memory_plan_digest": manifest["memory_plan_digest"],
                  "source_sampling_spec_digest": manifest["sampling_spec_digest"],
                  "visible_forward_count": len(visible), "total_forward_count": len(rows),
                  "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(assets.items())}}
    provenance["provenance_digest"] = sha256_json(provenance)
    assets["provenance.json"] = canonical_json_bytes(provenance) + b"\n"
    return assets


def verify_streaming_learning_assets(capture: Path, directory: Path) -> dict[str, Any]:
    """Compare exact derived bytes, not only an easily re-signed outer checksum."""
    if directory.is_symlink() or not directory.is_dir():
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "learning asset directory must be an ordinary directory")
    entries = list(directory.iterdir())
    if {path.name for path in entries} != LEARNING_FILES or any(path.is_symlink() or not path.is_file() for path in entries):
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "learning asset inventory is not exact")
    expected = derive_streaming_learning_assets(capture)
    if any((directory / name).read_bytes() != data for name, data in expected.items()):
        raise IllmError(ErrorCode.CAPTURE_CORRUPT, "learning assets do not rebuild from recorded R0 facts")
    return {"status": "pass", "evidence_kind": "derived", "file_count": len(expected),
            "source_capture_manifest_digest": StreamingCaptureBundle.open(capture).manifest["manifest_digest"]}

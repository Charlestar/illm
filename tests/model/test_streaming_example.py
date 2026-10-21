"""The public V0.2 tiny example is executable development guidance, not qualification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs" / "examples" / "v0_2_streaming.py"


def test_public_example_executes_generate_and_stream_with_both_kv_paths() -> None:
    process = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert not process.stderr
    records = cast(list[dict[str, Any]], [json.loads(line) for line in process.stdout.splitlines()])
    assert [record["example"] for record in records] == [
        "generate-none",
        "stream-contiguous",
        "stream-contiguous",
        "stream-contiguous",
        "stream-contiguous",
        "stream-contiguous",
        "stream-contiguous",
        "comparison",
    ]

    generated = records[0]["payload"]
    assert generated["output_token_ids"] == [114, 114, 114, 114]
    assert generated["text"] == "oooo"
    assert generated["kv_mode"] == "none"
    assert generated["query_tokens_per_step"] == [6, 7, 8, 9]
    assert generated["kv_valid_tokens_per_step"] == [0, 0, 0, 0]

    stream_payloads = [record["payload"] for record in records[1:-1]]
    assert [payload["kind"] for payload in stream_payloads] == [
        "metadata", "token", "token", "token", "token", "terminal",
    ]
    assert [payload["output_index"] for payload in stream_payloads if payload["kind"] == "token"] == [0, 1, 2, 3]
    terminal = stream_payloads[-1]["result"]
    assert terminal["output_token_ids"] == generated["output_token_ids"]
    assert terminal["text"] == generated["text"]
    assert terminal["kv_mode"] == "contiguous"
    assert terminal["query_tokens_per_step"] == [6, 1, 1, 1]
    assert terminal["kv_valid_tokens_per_step"] == [6, 7, 8, 9]
    assert all(value == 0 for value in terminal["resource_status"].values())

    comparison = records[-1]["payload"]
    assert comparison == {
        "output_token_ids": [114, 114, 114, 114],
        "text": "oooo",
        "none_query_tokens_per_step": [6, 7, 8, 9],
        "contiguous_query_tokens_per_step": [6, 1, 1, 1],
        "same_committed_output": True,
    }

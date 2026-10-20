from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_generated_unicode_is_utf8_even_when_inherited_stdio_is_ascii() -> None:
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="ascii", PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-m", "illm", "generate", "--prompt", "hello", "--max-new-tokens", "8",
         "--sampling-mode", "sample", "--temperature", "0.8", "--top-k", "20", "--top-p", "0.95", "--seed", "7"],
        cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["rng_draws_consumed"] == 8
    assert len(value["output_token_ids"]) == 8
    assert any(ord(character) > 127 for character in value["text"])
    assert value["committed_draw_addresses"] == [[position, 0, 0] for position in range(8)]

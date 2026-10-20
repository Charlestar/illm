"""CPU-safe probe guards only; these tests are not real CUDA qualification."""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import patch

from illm.domain.errors import ErrorCode, IllmError


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "probe_gpu_oom.py"))
validate_arguments = cast(Callable[[Path, str, float], None], MODULE["validate_arguments"])
probe_gpu_oom = cast(Callable[..., dict[str, Any]], MODULE["probe_gpu_oom"])


def temporary_directory() -> tempfile.TemporaryDirectory[str]:
    parent = ROOT / "out" / "model-test-temp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class GpuOomProbeContractTests(unittest.TestCase):
    def test_invalid_device_and_budget_rejected_without_creating_artifacts(self) -> None:
        with temporary_directory() as temporary:
            output = Path(temporary) / "new-output"
            for device in ("cpu", "cuda", "cuda:-1", "cuda:one", "mps"):
                with self.subTest(device=device), self.assertRaises(IllmError) as raised:
                    validate_arguments(output, device, 1e-6)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
            fractions: tuple[Any, ...] = (0, -1, float("nan"), float("inf"), 0.1, True)
            for fraction in fractions:
                with self.subTest(fraction=fraction), self.assertRaises(IllmError) as raised:
                    validate_arguments(output, "cuda:0", fraction)
                self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
            self.assertFalse(output.exists())

    def test_existing_evidence_and_file_targets_are_never_overwritten(self) -> None:
        with temporary_directory() as temporary:
            directory = Path(temporary)
            marker = directory / "oom-report.json"
            marker.write_bytes(b"existing evidence")
            for target in (directory, marker):
                with self.subTest(target=target), self.assertRaises(IllmError):
                    validate_arguments(target, "cuda:0", 1e-6)
            self.assertEqual(marker.read_bytes(), b"existing evidence")

    def test_unavailable_backend_is_not_reported_as_an_oom_pass(self) -> None:
        with temporary_directory() as temporary:
            output = Path(temporary) / "unavailable"
            with patch("torch.cuda.is_available", return_value=False), self.assertRaises(IllmError) as raised:
                probe_gpu_oom("fixture:tiny-decoder", output)
            self.assertEqual(raised.exception.code, ErrorCode.UNSUPPORTED_MODEL)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

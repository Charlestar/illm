"""Real tiny eager engine integration; injected faults follow real model execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any, ClassVar
from unittest.mock import patch

import torch
from torch import Tensor

from illm.capture.reference import ReferenceCaptureBundle
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.events import DomainEvent
from illm.domain.reference_events import ReferenceEventKind as K
from illm.engine import Engine, GenerationRequest, GenerationResult
from illm.sampling import SamplingSpec
from illm.tokenizer.reference import ByteTokenizer


ROOT = Path(__file__).resolve().parents[2]


class RaisingExporter:
    def emit(self, event: DomainEvent) -> None:
        del event
        raise RuntimeError("optional exporter failure")


class MutatingExporter:
    def __init__(self) -> None:
        self.count = 0

    def emit(self, event: DomainEvent) -> None:
        copied: Any = event.payload["reference_event"]
        copied["payload"].clear()
        copied["payload"]["prompt"] = "exporter injected text"
        self.count += 1


class BlockingExporter:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()

    def emit(self, event: DomainEvent) -> None:
        del event
        self.entered.set()
        if not self.release.wait(10):
            raise RuntimeError("test did not release blocked exporter")


class EngineIntegrationTests(unittest.TestCase):
    original_threads: ClassVar[int]
    @classmethod
    def setUpClass(cls) -> None:
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        (ROOT / "out").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "out")
        self.directory = Path(self.temporary.name)
        self.engine = Engine(project_root=ROOT)

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary.cleanup()

    @staticmethod
    def _request(**kwargs: Any) -> GenerationRequest:
        return GenerationRequest(prompt="hello private engine fixture", max_new_tokens=4,
                                 request_id="engine-integration", **kwargs)

    def _assert_outcome(self, engine: Engine, capture: Path, status: str,
                        output: GenerationResult | None = None) -> ReferenceCaptureBundle:
        bundle = ReferenceCaptureBundle.open(capture)
        kinds = [event.kind for event in bundle.events]
        self.assertEqual(kinds.count(K.REQUEST_ACCEPTED), 1)
        self.assertEqual(sum(kind in {K.REQUEST_FINISHED, K.REQUEST_FAILED} for kind in kinds), 1)
        self.assertEqual(bundle.final_state["status"], status)
        self.assertEqual([event.to_dict() for event in engine.last_events], [event.to_dict() for event in bundle.events])
        self.assertTrue((capture / "COMPLETE").is_file())
        self.assertFalse((capture / "INCOMPLETE").exists())
        if output is not None:
            self.assertEqual(bundle.manifest["manifest_digest"], output.capture_manifest_digest)
            self.assertEqual(bundle.final_state["next_output_position"], output.next_output_position)
            self.assertEqual(bundle.final_state["rng_draws_consumed"], output.rng_draws_consumed)
            self.assertEqual(bundle.final_state["committed_draw_addresses"], [list(item) for item in output.committed_draw_addresses])
            self.assertEqual(bundle.final_state["output_tokens"], len(output.output_token_ids))
        return bundle

    def test_real_greedy_recomputes_complete_context_and_replays(self) -> None:
        observed: list[tuple[int, ...]] = []
        original = self.engine._loaded.reference.forward

        def forward(input_ids: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            observed.append(tuple(input_ids[0].tolist()))
            return original(input_ids, *args, **kwargs)

        capture = self.directory / "greedy"
        with patch.object(self.engine._loaded.reference, "forward", side_effect=forward):
            result = self.engine.generate(self._request(capture=capture, telemetry="ops"))
        self.assertEqual(len(result.output_token_ids), 4)
        self.assertEqual(result.finish_reason, "MAX_NEW_TOKENS")
        self.assertEqual(result.rng_draws_consumed, 0)
        self.assertEqual(result.committed_draw_addresses, ())
        self.assertEqual(observed, [(*result.input_token_ids, *result.output_token_ids[:position]) for position in range(4)])
        self.assertEqual(len(result.timings["per_forward_wall_time_ns"]), 4)
        self.assertTrue(all(duration > 0 for duration in result.timings["per_forward_wall_time_ns"]))
        self.assertEqual(result.timings["synchronization"], "synchronous_cpu")
        bundle = self._assert_outcome(self.engine, capture, "FINISHED", result)
        ops = [event for event in bundle.events if event.kind is K.MODEL_OP]
        self.assertEqual(len(ops), 39 * 4)
        self.assertTrue(all("duration_ns" not in event.payload for event in ops))

    def test_seeded_sampling_is_independent_of_telemetry_and_reuse(self) -> None:
        sampling = SamplingSpec(mode="sample", temperature=0.7, top_k=19, top_p=0.83, actual_seed=123456789)
        outputs: list[GenerationResult] = []
        for telemetry in ("off", "request", "ops", "off"):
            capture = self.directory / f"sample-{len(outputs)}"
            result = self.engine.generate(self._request(sampling=sampling, capture=capture, telemetry=telemetry))
            self._assert_outcome(self.engine, capture, "FINISHED", result)
            outputs.append(result)
        first = outputs[0]
        self.assertGreater(len(first.output_token_ids), 0)
        self.assertEqual(first.actual_seed, 123456789)
        for result in outputs:
            self.assertEqual(result.output_token_ids, first.output_token_ids)
            self.assertEqual(result.text, first.text)
            self.assertEqual(result.rng_draws_consumed, len(result.output_token_ids))
            self.assertEqual(result.committed_draw_addresses, tuple((position, 0, 0) for position in range(len(result.output_token_ids))))

    def test_real_input_ids_and_explicit_bos_contract(self) -> None:
        with_bos = self.engine.generate(GenerationRequest(input_token_ids=(), add_bos=True, max_new_tokens=2))
        self.assertEqual(with_bos.input_token_ids, (1,))
        raw = self.engine.generate(GenerationRequest(input_token_ids=(75, 104), add_bos=False, max_new_tokens=2))
        self.assertEqual(raw.input_token_ids, (75, 104))
        for input_ids in ((), (259,)):
            capture = self.directory / ("empty" if not input_ids else "out-of-vocabulary")
            with self.assertRaises(IllmError) as raised:
                self.engine.generate(GenerationRequest(input_token_ids=input_ids, capture=capture))
            self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
            self._assert_outcome(self.engine, capture, "FAILED")

    def test_context_limit_is_exact_for_full_recompute(self) -> None:
        result = self.engine.generate(GenerationRequest(input_token_ids=(75,) * 255, max_new_tokens=2))
        self.assertEqual(len(result.output_token_ids), 2)
        capture = self.directory / "too-long"
        with self.assertRaises(IllmError) as raised:
            self.engine.generate(GenerationRequest(input_token_ids=(75,) * 256, max_new_tokens=2, capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
        bundle = self._assert_outcome(self.engine, capture, "FAILED")
        self.assertNotIn(K.MODEL_FORWARD_STARTED, [event.kind for event in bundle.events])

    def test_capture_does_not_store_text_tokens_or_logits(self) -> None:
        capture = self.directory / "privacy"
        prompt = "UNIQUE-PRIVATE-PROMPT-for-engine-integration"
        result = self.engine.generate(GenerationRequest(prompt=prompt, capture=capture, telemetry="ops", max_new_tokens=3))
        self._assert_outcome(self.engine, capture, "FINISHED", result)
        forbidden_fields = {"prompt", "text", "input_token_ids", "output_token_ids", "token_id", "logits", "tensor", "values"}

        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                self.assertFalse(forbidden_fields.intersection(value))
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        for path in capture.iterdir():
            encoded = path.read_text(encoding="utf-8")
            self.assertNotIn(prompt, encoded)
            if path.suffix == ".json":
                inspect(json.loads(encoded))
            elif path.suffix == ".jsonl":
                for line in encoded.splitlines():
                    inspect(json.loads(line))
        second = self.directory / "same-private-input"
        self.engine.generate(GenerationRequest(prompt=prompt, capture=second, max_new_tokens=3))
        first_state = ReferenceCaptureBundle.open(capture).final_state
        second_state = ReferenceCaptureBundle.open(second).final_state
        self.assertNotEqual(first_state["input_digest"], second_state["input_digest"])
        self.assertNotEqual(first_state["last_logits_digest"], second_state["last_logits_digest"])

    def test_raising_and_mutating_exporters_cannot_change_real_generation(self) -> None:
        mutating = MutatingExporter()
        engine = Engine(project_root=ROOT, exporters=(RaisingExporter(), mutating), exporter_capacity=1024)
        try:
            total_events = 0
            result: GenerationResult | None = None
            capture = self.directory
            for sampling in (SamplingSpec(), SamplingSpec(mode="sample", temperature=0.7, top_k=19, top_p=0.83, actual_seed=123456789)):
                baseline = self.engine.generate(self._request(telemetry="off", sampling=sampling))
                capture = self.directory / f"bad-exporters-{sampling.mode}"
                result = engine.generate(self._request(capture=capture, telemetry="ops", sampling=sampling))
                self.assertEqual(result.output_token_ids, baseline.output_token_ids)
                self.assertEqual(result.committed_draw_addresses, baseline.committed_draw_addresses)
                self._assert_outcome(engine, capture, "FINISHED", result)
                total_events += result.event_count
            self.assertTrue(engine.close(timeout=5))
            self.assertEqual(mutating.count, total_events)
            self.assertEqual(engine.exporter_status["error_count"], total_events)
            self.assertEqual(engine.exporter_status["copy_drop_count"], 0)
            self.assertIsNotNone(result)
            self._assert_outcome(engine, capture, "FINISHED", result)
        finally:
            engine.close()

    def test_truly_blocked_exporter_is_bounded_and_does_not_block_outcome(self) -> None:
        exporter = BlockingExporter()
        engine = Engine(project_root=ROOT, exporters=(exporter,), exporter_capacity=2)
        try:
            sampling = SamplingSpec(mode="sample", temperature=0.7, top_k=19, top_p=0.83, actual_seed=123456789)
            baseline = self.engine.generate(self._request(telemetry="off", sampling=sampling))
            capture = self.directory / "blocked-exporter"
            result = engine.generate(self._request(capture=capture, telemetry="ops", sampling=sampling))
            self.assertEqual(result.output_token_ids, baseline.output_token_ids)
            self.assertEqual(result.committed_draw_addresses, baseline.committed_draw_addresses)
            self.assertTrue(exporter.entered.wait(2))
            self.assertFalse(exporter.release.is_set())
            self.assertFalse(engine.close(timeout=0))
            self.assertEqual(engine.exporter_status["queue_high_watermark"], 2)
            self.assertGreater(engine.exporter_status["copy_drop_count"], 0)
            self._assert_outcome(engine, capture, "FINISHED", result)
        finally:
            exporter.release.set()
            self.assertTrue(engine.close(timeout=5))

    def test_tokenization_failure_has_one_terminal_and_engine_remains_usable(self) -> None:
        capture = self.directory / "tokenizer-failed"
        with self.assertRaises(IllmError) as raised:
            self.engine.generate(GenerationRequest(prompt="\ud800", capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
        self._assert_outcome(self.engine, capture, "FAILED")
        result = self.engine.generate(self._request())
        self.assertGreater(len(result.output_token_ids), 0)

    def test_nonfinite_actual_forward_fails_before_rng_commit(self) -> None:
        original = self.engine._loaded.reference.forward
        executed = False

        def corrupt(input_ids: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            nonlocal executed
            logits = original(input_ids, *args, **kwargs).clone()
            executed = True
            logits[0, -1, 0] = float("nan")
            return logits

        capture = self.directory / "numerical-failed"
        with patch.object(self.engine._loaded.reference, "forward", side_effect=corrupt), self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=capture, sampling=SamplingSpec(mode="sample", temperature=1, actual_seed=5)))
        self.assertTrue(executed)
        self.assertEqual(raised.exception.code, ErrorCode.NUMERICAL_ERROR)
        bundle = self._assert_outcome(self.engine, capture, "FAILED")
        self.assertEqual(bundle.final_state["rng_draws_consumed"], 0)
        self.assertEqual(bundle.final_state["output_tokens"], 0)
        self.assertEqual(sum(event.kind is K.SAMPLING_FAILED for event in bundle.events), 1)
        self.assertNotIn(K.SAMPLING_SELECTED, [event.kind for event in bundle.events])

    def test_eos_stops_after_one_committed_real_forward(self) -> None:
        original = self.engine._loaded.reference.forward
        forwards = 0

        def force_eos(input_ids: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            nonlocal forwards
            logits = original(input_ids, *args, **kwargs).clone()
            forwards += 1
            logits[0, -1].fill_(-1)
            logits[0, -1, 2] = 1
            return logits

        capture = self.directory / "eos"
        with patch.object(self.engine._loaded.reference, "forward", side_effect=force_eos):
            result = self.engine.generate(self._request(capture=capture))
        self.assertEqual(forwards, 1)
        self.assertEqual(result.output_token_ids, (2,))
        self.assertEqual(result.text, "")
        self.assertEqual(result.finish_reason, "EOS")
        self.assertEqual(result.next_output_position, 1)
        self.assertEqual(result.rng_draws_consumed, 0)
        self._assert_outcome(self.engine, capture, "FINISHED", result)

    def test_real_torch_oom_type_maps_resource_exhausted_and_terminal_once(self) -> None:
        # Real successful eager call establishes the adapter before fault injection.
        self.assertGreater(len(self.engine.generate(self._request()).output_token_ids), 0)
        capture = self.directory / "oom-failed"
        with patch.object(self.engine._loaded.reference, "forward", side_effect=torch.OutOfMemoryError("allocation refused")), self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.RESOURCE_EXHAUSTED)
        bundle = self._assert_outcome(self.engine, capture, "FAILED")
        self.assertEqual(bundle.final_state["error_code"], "RESOURCE_EXHAUSTED")
        self.assertEqual(bundle.final_state["next_output_position"], 0)
        self.assertEqual(len(self.engine.generate(self._request()).output_token_ids), 4)

    def test_decode_failure_keeps_committed_controls_but_never_finishes(self) -> None:
        capture = self.directory / "decode-failed"
        with patch.object(ByteTokenizer, "decode", side_effect=RuntimeError("decoder failed")), self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.INTERNAL)
        bundle = self._assert_outcome(self.engine, capture, "FAILED")
        self.assertEqual(bundle.final_state["next_output_position"], 4)
        self.assertEqual(bundle.final_state["output_tokens"], 4)

    def test_one_active_request_rejects_concurrent_generate_and_close(self) -> None:
        entered, release = Event(), Event()
        original = self.engine._loaded.reference.forward

        def slow_forward(input_ids: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release active forward")
            return original(input_ids, *args, **kwargs)

        capture = self.directory / "active-request"
        rejected_capture = self.directory / "rejected-request"
        with ThreadPoolExecutor(max_workers=1) as executor, patch.object(self.engine._loaded.reference, "forward", side_effect=slow_forward):
            future = executor.submit(self.engine.generate, self._request(capture=capture))
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaises(IllmError) as raised:
                    self.engine.generate(self._request(capture=rejected_capture))
                self.assertEqual(raised.exception.code, ErrorCode.ENGINE_BUSY)
                with self.assertRaises(IllmError) as closing:
                    self.engine.close()
                self.assertEqual(closing.exception.code, ErrorCode.ENGINE_BUSY)
                self.assertFalse(rejected_capture.exists())
            finally:
                release.set()
            result = future.result(timeout=5)
        self._assert_outcome(self.engine, capture, "FINISHED", result)
        self.assertEqual(len(self.engine.generate(self._request()).output_token_ids), 4)

    def test_close_is_idempotent_and_rejects_future_generation(self) -> None:
        self.assertTrue(self.engine.close())
        self.assertTrue(self.engine.close())
        capture = self.directory / "closed"
        with self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
        self.assertFalse(capture.exists())

    def test_invalid_close_timeout_does_not_mutate_usable_engine(self) -> None:
        for timeout in (-1.0, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(IllmError) as raised:
                self.engine.close(timeout=timeout)
            self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
            self.assertEqual(len(self.engine.generate(self._request()).output_token_ids), 4)

    def test_invalid_capture_target_is_structured_before_request_acceptance(self) -> None:
        self.engine.generate(self._request())
        self.assertGreater(len(self.engine.last_events), 0)
        capture = self.directory / "existing-file"
        capture.touch()
        with self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=capture))
        self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
        self.assertEqual(self.engine.last_events, ())
        self.assertEqual(len(self.engine.generate(self._request()).output_token_ids), 4)

    def test_capture_initialization_io_failure_maps_capture_incomplete(self) -> None:
        with patch("illm.engine.facade.ReferenceCaptureSession", side_effect=PermissionError("private path denied")), self.assertRaises(IllmError) as raised:
            self.engine.generate(self._request(capture=self.directory / "denied"))
        self.assertEqual(raised.exception.code, ErrorCode.CAPTURE_INCOMPLETE)
        self.assertEqual(self.engine.last_events, ())
        self.assertNotIn("private path", raised.exception.message)
        self.assertEqual(len(self.engine.generate(self._request()).output_token_ids), 4)


class RequestContractTests(unittest.TestCase):
    def test_invalid_exporter_capacity_fails_before_loading_model(self) -> None:
        capacities: tuple[Any, ...] = (0, -1, False, True, 2.5, float("nan"), "2", None)
        for capacity in capacities:
            with self.subTest(capacity=capacity), patch("illm.engine.facade.load_model") as load, self.assertRaises(IllmError) as raised:
                Engine(exporter_capacity=capacity)
            self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)
            load.assert_not_called()

    def test_explicit_dto_rejects_mutable_ids_ambiguous_input_and_bad_fields(self) -> None:
        invalid: list[dict[str, Any]] = [
            {}, {"prompt": "a", "input_token_ids": (1,)}, {"input_token_ids": [1]},
            {"input_token_ids": (True,)}, {"input_token_ids": (-1,)}, {"prompt": "a", "max_new_tokens": False},
            {"prompt": "a", "max_new_tokens": 0}, {"prompt": "a", "request_id": "unsafe/id"},
            {"prompt": "a", "telemetry": "unknown"}, {"prompt": "a", "capture": "output"},
            {"prompt": "a", "add_bos": 1}, {"prompt": "a", "sampling": {}},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(IllmError) as raised:
                GenerationRequest(**kwargs)
            self.assertEqual(raised.exception.code, ErrorCode.INVALID_ARGUMENT)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Independent-process reproduction of the frozen tiny20 candidate run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ORACLE_MANIFEST_DIGEST = "d424fe676820b748bf51023ca15150211d3414371314d63ce3fa647c4264fadc"


CHILD = textwrap.dedent(
    r"""
    from __future__ import annotations

    import hashlib
    import json
    import os
    import sys
    from pathlib import Path

    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root / "src"))

    import torch

    from illm.executor import ImmediateFence
    from illm.kv_cache import ContiguousKVStore, KVLayoutSpec
    from illm.model import load_model
    from illm.model.policy import PolicyDecoder
    from illm.numerical_policy import candidate_policy
    from illm.sampling import SamplingSpec, SamplingState, commit_sample, prepare_sample


    def tensor_digest(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    manifest = json.loads(
        (root / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_text(encoding="utf-8")
    )
    cases = manifest["greedy_cases"]
    assert len(cases) == 20
    assert all(len(case["expected_tokens"]) == 8 for case in cases)

    loaded = load_model(project_root=root)
    policy = candidate_policy("fp32")
    b = PolicyDecoder(loaded.reference, policy)
    c = PolicyDecoder(loaded.reference, policy)
    sampling = SamplingSpec(mode="greedy", actual_seed=7)
    records = []
    observed_by_case = {}
    for case in cases:
        case_id = str(case["fixture_id"])
        context = tuple(int(token) for token in case["input_ids"])
        expected_tokens = tuple(int(token) for token in case["expected_tokens"])
        b_state = SamplingState(case_id, sampling.digest)
        c_state = SamplingState(case_id, sampling.digest)
        observed = []
        store = ContiguousKVStore(KVLayoutSpec.for_model(loaded.spec, device="cpu", capacity=256))
        try:
            for step, expected_token in enumerate(expected_tokens):
                query = context[store.valid_tokens :]
                lease = store.begin_append(query)
                b_tensor = b.forward(torch.tensor([context], dtype=torch.int64))[0, -1].contiguous()
                c_tensor = c.forward_incremental(
                    torch.tensor([query], dtype=torch.int64), lease
                )[0, -1].contiguous()
                assert b_tensor.dtype == c_tensor.dtype == torch.float32
                assert b_tensor.shape == c_tensor.shape == (loaded.spec.vocab_size,)
                assert torch.equal(b_tensor, c_tensor)
                b_logits = tuple(float(value) for value in b_tensor.tolist())
                c_logits = tuple(float(value) for value in c_tensor.tolist())
                b_decision = prepare_sample(b_logits, sampling, b_state)
                c_decision = prepare_sample(c_logits, sampling, c_state)
                assert b_decision.token_id == c_decision.token_id == expected_token
                assert lease.commit(ImmediateFence()) == len(context)
                b_state = commit_sample(b_state, b_decision, spec=sampling, logits=b_logits)
                c_state = commit_sample(c_state, c_decision, spec=sampling, logits=c_logits)
                assert b_state == c_state
                observed.append(expected_token)
                records.append(
                    {
                        "case_id": case_id,
                        "step": step,
                        "query_tokens": len(query),
                        "native_dtype": str(b_tensor.dtype),
                        "native_shape": list(b_tensor.shape),
                        "b_native_tensor_sha256": tensor_digest(b_tensor),
                        "c_native_tensor_sha256": tensor_digest(c_tensor),
                        "token_id": expected_token,
                        "state_digest": b_state.digest,
                        "state": b_state.to_dict(),
                        "kv_valid_tokens": store.valid_tokens,
                    }
                )
                context = (*context, expected_token)
        finally:
            store.close()
        assert observed == list(expected_tokens)
        assert store.valid_tokens == store.staged_tokens == store.allocated_bytes == 0
        observed_by_case[case_id] = observed

    assert len(records) == 20 * 8
    assert not torch.cuda.is_initialized()
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "cuda_initialized": torch.cuda.is_initialized(),
                "oracle_manifest_digest": manifest["manifest_digest"],
                "policy_digest": policy.digest,
                "case_count": len(cases),
                "step_count": len(records),
                "observed_by_case": observed_by_case,
                "records": records,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    """
)


def _run_child() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", CHILD, str(ROOT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, completed.stdout
    result = json.loads(lines[0])
    assert isinstance(result, dict)
    return result


def test_tiny20_reproduces_exactly_in_two_fresh_cpu_processes() -> None:
    first = _run_child()
    second = _run_child()

    assert first["pid"] != os.getpid()
    assert second["pid"] != os.getpid()
    assert first["pid"] != second["pid"]
    assert first["cuda_initialized"] is second["cuda_initialized"] is False
    assert first["oracle_manifest_digest"] == second["oracle_manifest_digest"]
    assert first["oracle_manifest_digest"] == EXPECTED_ORACLE_MANIFEST_DIGEST
    assert first["policy_digest"] == second["policy_digest"]
    assert first["case_count"] == second["case_count"] == 20
    assert first["step_count"] == second["step_count"] == 160
    assert first["records"] == second["records"]
    assert first["observed_by_case"] == second["observed_by_case"]

    manifest = json.loads(
        (ROOT / "fixtures/tiny-decoder/oracle-v1/oracle-manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        str(case["fixture_id"]): [int(token) for token in case["expected_tokens"]]
        for case in manifest["greedy_cases"]
    }
    assert first["observed_by_case"] == expected
    assert all(
        record["b_native_tensor_sha256"] == record["c_native_tensor_sha256"]
        and record["native_dtype"] == "torch.float32"
        and record["native_shape"] == [259]
        and record["state"]["next_output_position"] == record["step"] + 1
        for record in first["records"]
    )

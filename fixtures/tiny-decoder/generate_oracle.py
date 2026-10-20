"""Generate public tiny oracle goldens with pinned, local Transformers only.

This is an explicit maintainer operation, never part of fixture loading or a
test run. Existing outputs are immutable: regenerate to a new empty directory
and review the numerical diff before replacing a versioned oracle fixture.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from illm.model import load_model
from illm.model.spec import ModelError, canonical_bytes, digest
from illm.model.validation import FIXED_PROMPTS, collect_transformers_ops, load_transformers_oracle, oracle_forward_logits


@torch.inference_mode()
def generate_oracle(output: Path, *, project_root: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ModelError("INVALID_ARGUMENT", "oracle output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loaded = load_model("fixture:tiny-decoder", device="cpu", project_root=project_root)
    oracle = load_transformers_oracle(loaded)
    cases: list[dict[str, Any]] = []
    for case_id, ids, positions in (
        ("hello", loaded.tokenizer.encode("hello"), None),
        ("vocab-min-position-zero", (0,), (0,)),
        ("vocab-max-position-last", (258,), (loaded.spec.max_position_embeddings - 1,)),
    ):
        inputs = torch.tensor([ids], dtype=torch.int64)
        position_ids = torch.tensor([positions], dtype=torch.int64) if positions else None
        trace = collect_transformers_ops(oracle, inputs, loaded, position_ids=position_ids)
        filename = f"{case_id}.safetensors"
        save_file({name: value.cpu().contiguous() for name, value in trace.items()}, str(output / filename))
        cases.append({"fixture_id": case_id, "input_ids": list(ids),
                      "position_ids": list(positions) if positions else None,
                      "op_order": list(trace), "file": filename,
                      "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
    greedy: list[dict[str, Any]] = []
    for index, prompt in enumerate(FIXED_PROMPTS):
        initial_ids = loaded.tokenizer.encode(prompt)
        full = list(initial_ids)
        selected: list[int] = []
        for _ in range(8):
            logits = oracle_forward_logits(oracle, torch.tensor([full], dtype=torch.int64), vocab_size=loaded.spec.vocab_size)
            token = int(logits[0, -1].float().argmax().item())
            selected.append(token)
            full.append(token)
            if token == loaded.spec.eos_token_id:
                break
        greedy.append({"fixture_id": f"fixed-greedy-{index:02d}", "public_prompt": prompt,
                       "input_ids": list(initial_ids), "expected_tokens": selected})
    result: dict[str, Any] = {
        "schema_version": 1, "generator": "illm-pinned-transformers-oracle-v1",
        "torch_version": torch.__version__, "transformers_version": "4.57.1",
        "device": "cpu", "runtime_dtype": "fp32", "attention_implementation": "eager",
        "tf32_enabled": False, "spec_id": loaded.spec.spec_id,
        "weights_digest": loaded.weight_manifest["weights_digest"],
        "load_manifest_digest": loaded.weight_manifest["manifest_digest"],
        "tokenizer_revision": loaded.tokenizer.revision,
        "trace_adapter_verified_against_ordinary_forward": True,
        "tolerance": {"atol": 1e-4, "rtol": 1e-3, "profile_id": "tiny-fp32-op-v1"},
        "op_cases": cases, "greedy_cases": greedy,
    }
    result["manifest_digest"] = digest(result)
    (output / "oracle-manifest.json").write_bytes(canonical_bytes(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = generate_oracle(args.output, project_root=args.project_root)
    print(result["manifest_digest"])


if __name__ == "__main__":
    main()

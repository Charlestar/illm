# Tiny independent oracle fixtures

These are public numerical test assets, not captured user prompts. They were
generated on CPU with Torch 2.8.0+cpu, Transformers 4.57.1, eager attention,
TF32 disabled, and the original digest-verified safetensors loaded independently
of iLLM's reference buffers. The trace adapter checks its complete logits against
ordinary Transformers forward before recording a fixture.

`oracle-manifest.json` binds the model spec, original weights, load manifest,
tokenizer, three full 39-op tensor traces, and 20 fixed inputs with eight greedy
tokens each. The traces cover normal text plus the vocabulary/absolute-position
boundaries. Tensor comparisons use the predeclared FP32 `atol=1e-4, rtol=1e-3`;
greedy IDs must match exactly. The tiny model is deterministic and untrained:
repeating the last byte token is expected, not evidence of language quality.

Normal tests read these files; they never regenerate or overwrite them. To
review a candidate regeneration, use the selected locked CPU runtime and run
`fixtures/tiny-decoder/generate_oracle.py --output out/oracle-candidate` from the
repository root. The output must be a new empty directory. Compare all hashes
and numerical changes explicitly before introducing a new fixture version.

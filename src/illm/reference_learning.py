"""Deterministic, offline SVG learning diagrams derived from verified facts.

These are generated SVG diagrams, not UI screenshots or new numerical evidence.
Only stdlib code runs here: no model execution, JavaScript or external resources.
The release verifier independently verifies the profile before rebuilding them.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from illm.canonical import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from illm.domain.errors import ErrorCode, IllmError
from illm.model.spec import SupportedModelSpec


LEARNING_DIRECTORY = "screenshots/model-and-diff"
LEARNING_FILES = {"model-structure.svg", "first-divergence.svg", "provenance.json"}
GENERATOR_ID = "reference-learning-svg-v1"


def _fail(message: str) -> NoReturn:
    raise IllmError(ErrorCode.INVALID_ARGUMENT, message)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            _fail("learning input contains duplicate JSON keys")
        result[name] = value
    return result


def _object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        _fail("learning input must be a bounded ordinary JSON artifact")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                       parse_constant=lambda _: _fail("learning input contains nonfinite JSON"))
    if not isinstance(value, dict):
        _fail("learning input must be a JSON object")
    return dict(value)


def _signed(value: Mapping[str, Any], field: str) -> None:
    if value.get(field) != sha256_json({key: item for key, item in value.items() if key != field}):
        _fail("learning input has a stale semantic digest")


def _safe(value: object) -> str:
    text = str(value)
    if any(not (character in "\t\n\r" or 0x20 <= ord(character) <= 0xD7FF
                or 0xE000 <= ord(character) <= 0xFFFD or 0x10000 <= ord(character) <= 0x10FFFF)
           for character in text):
        _fail("learning label contains an invalid XML character")
    return html.escape(text, quote=True)


def _text(x: int, y: int, value: object, *, size: int = 17, color: str = "#1d2d44") -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{_safe(value)}</text>'


def _lines(x: int, y: int, values: Sequence[object], *, size: int = 17, gap: int = 26) -> str:
    return "\n".join(_text(x, y + index * gap, value, size=size) for index, value in enumerate(values))


def _box(x: int, y: int, width: int, height: int, title: str, lines: Sequence[object], *, fault: bool = False) -> str:
    fill, stroke = ("#fff1f0", "#bb3434") if fault else ("#f0f7ff", "#4a709a")
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="{stroke}"/>'
            + _text(x + 18, y + 30, title, size=20)
            + _lines(x + 18, y + 58, lines, size=16, gap=24))


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return (f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="#52677d" stroke-width="2" marker-end="url(#arrow)"/>')


def _svg(title: str, description: str, height: int, parts: Sequence[str]) -> bytes:
    header = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title description">\n'
              f'<title id="title">{_safe(title)}</title><desc id="description">{_safe(description)}</desc>\n'
              '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52677d"/></marker></defs>\n'
              f'<rect width="1440" height="{height}" fill="#ffffff"/>\n'
              '<g font-family="DejaVu Sans,Arial,sans-serif">\n')
    return (header + "\n".join(parts) + "\n</g></svg>\n").encode("utf-8")


def _projection(spec: SupportedModelSpec, weights: Mapping[str, Any]) -> dict[str, Any]:
    shapes = {name: list(shape) for name, shape in spec.tensor_shapes().items()}
    storages = weights.get("storages")
    if (not isinstance(storages, list) or len(storages) != len(shapes)
            or weights.get("spec_id") != spec.spec_id or weights.get("parameter_count") != spec.parameter_count
            or weights.get("canonical_storage_count") != len(shapes)):
        _fail("learning model storage/parameter facts disagree with inspect structure")
    actual = {item["canonical_name"]: item["shape"] for item in storages}
    if actual != shapes:
        _fail("learning weight shapes differ from inspect structure")
    tied = [item for item in storages if item.get("tied_weight_group") is not None]
    if (not spec.tie_word_embeddings or len(tied) != 1
            or tied[0]["canonical_name"] != "model.embed_tokens.weight"
            or tied[0]["parameter_aliases"] != ["model.embed_tokens.weight", "lm_head.weight"]):
        _fail("tiny learning diagram requires the measured embedding/lm-head storage alias")
    return {"spec": {**spec.to_dict(), "spec_id": spec.spec_id}, "parameter_count": spec.parameter_count,
            "canonical_tensor_shapes": shapes, "canonical_storage_count": len(storages),
            "tied_storage": tied[0], "weight_manifest_digest": weights["manifest_digest"]}


def _structure(projection: Mapping[str, Any], report_digest: str) -> bytes:
    spec = projection["spec"]
    h, i, q, kv, d, v = (spec[key] for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_kv_heads", "head_dim", "vocab_size"))
    parts = [_text(38, 46, "Tiny decoder: model structure", size=30),
             _text(38, 78, "Generated SVG diagram, not a UI screenshot. Shapes are derived from the verified inspect-compatible structure."),
             _text(38, 109, f"{spec['architecture']} | layers={spec['num_layers']} | {projection['parameter_count']} unique parameters | {projection['canonical_storage_count']} canonical storages | {spec['runtime_dtype']}"),
             _text(38, 139, "B = batch; T = current full context. No KV cache: each generated token recomputes the full context."),
             _box(275, 170, 890, 94, "Token IDs -> embedding", [f"IDs [B,T] -> hidden [B,T,{h}] | shared embedding weight [{v},{h}]"]),
             _arrow(420, 265, 360, 316)]
    for layer, x in ((0, 38), (1, 746)):
        parts.extend([_text(x + 15, 300, f"Decoder layer {layer}", size=24),
                      _box(x, 320, 655, 90, "Pre-attention RMSNorm", [f"[B,T,{h}] | epsilon={spec['rms_norm_eps']}"]),
                      _box(x, 440, 655, 118, "Q/K/V projections + RoPE", [f"Q [B,{q},T,{d}] | K,V [B,{kv},T,{d}]", f"Q/K rotary positions | theta={spec['rope_theta']} | Q/K/V bias"]),
                      _box(x, 588, 655, 118, "GQA + causal attention + residual", [f"{q} Q heads share {kv} KV heads ({q // kv} Q per KV)", f"scores/probabilities [B,{q},T,T] -> output [B,T,{h}]"]),
                      _box(x, 736, 655, 90, "Post-attention RMSNorm", [f"[B,T,{h}] | second pre-norm before the MLP"]),
                      _box(x, 856, 655, 118, "Gated SiLU MLP + residual", [f"gate/up [{i},{h}] -> SiLU(gate) * up [B,T,{i}]", f"down [{h},{i}] -> [B,T,{h}] + residual"])])
        for start, end in ((410, 440), (558, 588), (706, 736), (826, 856)):
            parts.append(_arrow(x + 327, start, x + 327, end))
    parts.extend([f'<path d="M 365 975 L 365 1000 L 720 1000 L 720 302 L 1073 302 L 1073 318" fill="none" stroke="#52677d" stroke-width="2" marker-end="url(#arrow)"/>',
                  _arrow(1073, 975, 1073, 1042),
                  _box(275, 1044, 890, 118, "Final RMSNorm -> tied LM head -> logits", [f"[B,T,{h}] -> [B,T,{v}] | LM head reuses embedding storage, no duplicate parameters", "Only the last-position logits feed the separately specified sampler."]),
                  _text(38, 1205, f"spec_id: {spec['spec_id']}", size=14),
                  _text(38, 1232, f"weight manifest digest: {projection['weight_manifest_digest']}", size=14),
                  _text(38, 1259, f"validation report digest: {report_digest}", size=14),
                  _text(38, 1290, "Provenance: recorded spec/storage metadata; arrows and symbolic tensor shapes are derived, not measured timings.", size=15)])
    return _svg("Tiny decoder model structure", "Two decoder layers with RMSNorm, RoPE, grouped-query causal attention, gated MLP and tied embedding/LM head.", 1320, parts)


def _diagnostic(directory: Path, validation: Mapping[str, Any], field: str, *, compound: bool) -> tuple[str, dict[str, Any]]:
    diagnostic = validation.get(field)
    if not isinstance(diagnostic, dict) or diagnostic.get("status") != "pass" or not isinstance(diagnostic.get("report"), dict):
        _fail("learning diagram requires an executed passing fault diagnostic")
    report = diagnostic["report"]
    name = report.get("report_link")
    expected = "fault-q-transpose-and-zero-final-norm.json" if compound else "fault-transposed-q.json"
    if name != expected or _object(directory / expected) != report:
        _fail("learning fault input differs from its exact referenced report")
    first, ops = report.get("first_divergence", {}), report.get("ops", [])
    identity = report.get("identity", {})
    comparison = report.get("output_comparison", {})
    transforms = identity.get("fault_transforms") if compound else [identity.get("fault_transform")]
    if (report.get("status") != "fail" or not isinstance(first, dict) or len(ops) != 39
            or first != ops[2] or first.get("op") != "layer.0.q_projection" or first.get("status") != "fail"
            or first.get("evidence_kind") != "measured" or first.get("reason") != "outside_tolerance"
            or any(item.get("status") != "pass" for item in ops[:2])
            or not isinstance(transforms, list) or len(transforms) != (2 if compound else 1)
            or transforms[0].get("transform_id") != "transpose-axes-0-1-contiguous-v1"
            or (compound and transforms[1].get("transform_id") != "zero-tensor-v1")
            or any(identity.get(key) != validation["identity"][key] for key in ("spec_id", "weights_digest", "load_manifest_digest"))
            or comparison.get("evidence_kind") != "measured" or comparison.get("token_mismatch") is not compound
            or (comparison.get("expected_next_token") != comparison.get("actual_next_token")) is not compound):
        _fail("learning diagram fault order, identity or measured token effect is inconsistent")
    for key in ("max_absolute_error", "max_relative_error", "expected_value", "actual_value"):
        if isinstance(first.get(key), bool) or not isinstance(first.get(key), (int, float)) or not math.isfinite(first[key]):
            _fail("learning fault metrics must be finite measured numbers")
    if first["max_absolute_error"] <= 0 or type(first.get("mismatch_count")) is not int or first["mismatch_count"] <= 0:
        _fail("learning fault must demonstrate actual divergence")
    return expected, dict(report)


def _divergence(simple: Mapping[str, Any], compound: Mapping[str, Any], validation_digest: str, hashes: Mapping[str, str]) -> bytes:
    parts = [_text(38, 46, "First divergence: find the broken operator", size=30),
             _text(38, 80, "Generated SVG diagram from actual fault reports, not a UI screenshot or another numerical run."),
             _text(38, 112, "Two distinct experiments: a numerical mismatch does not necessarily change greedy argmax."),
             _text(38, 146, f"Validation report digest: {validation_digest}", size=14)]
    for x, title, report in ((38, "A. Q weight transpose only", simple), (746, "B. Q transpose + zero final norm", compound)):
        first, output = report["first_divergence"], report["output_comparison"]
        tolerance = first["tolerance"]
        parts.extend([_text(x + 8, 205, title, size=23),
                      _box(x, 228, 655, 88, "embedding -> input RMSNorm: PASS", ["Both operators before Q projection match within tolerance."]),
                      _arrow(x + 327, 317, x + 327, 345),
                      _box(x, 348, 655, 118, "FIRST FAIL: " + first["op"], [f"expected {first['expected_shape']} / actual {first['actual_shape']}", f"{first['mismatch_count']} mismatches | first flat index {first['first_flat_index']}"], fault=True),
                      _lines(x + 16, 505, [f"max absolute error: {first['max_absolute_error']}",
                                           f"max relative error: {first['max_relative_error']}",
                                           f"first expected value: {first['expected_value']}",
                                           f"first actual value: {first['actual_value']}",
                                           f"tolerance: atol={tolerance['atol']}, rtol={tolerance['rtol']}",
                                           f"profile: {tolerance['profile_id']}"], size=16, gap=29),
                      _box(x, 702, 655, 118, "Measured final output (expected -> actual)", [f"greedy token: {output['expected_next_token']} -> {output['actual_next_token']} | token mismatch: {str(output['token_mismatch']).lower()}",
                                                                                             f"final-logit max absolute error: {output['final_logits_max_absolute_error']}"], fault=bool(output["token_mismatch"])),
                      _lines(x + 16, 859, ["Injected mutations: " + ("2 (Q transpose; final norm zero)" if x == 746 else "1 (Q transpose)"),
                                           "A changed argmax is not attributed to transpose alone." if x == 746 else "Same selected token does not imply correct intermediate tensors."], size=16),
                      _text(x + 16, 946, "Report: " + report["report_link"], size=13),
                      _text(x + 16, 975, "SHA256: " + hashes[report["report_link"]], size=12)])
    parts.extend([_text(38, 1040, "Fault experiments intentionally FAIL numerical comparison; their diagnostic checks PASS because the first wrong operator is found."),
                  _text(38, 1074, "The portable profile verifier checks fault transformations, per-op order, frozen tolerances and measured output effects before drawing.", size=16),
                  _text(38, 1106, "No performance, trained-model quality, UI rendering or universal hardware support claim is made by these derived diagrams.", size=16)])
    return _svg("First divergent Q projection and separate token effects", "Pure Q transpose preserves the measured greedy token; a separate compound fault changes it. Both first diverge at layer zero Q projection.", 1140, parts)


def derive_reference_learning_assets(validation_directory: Path, *, inspect_structure: Mapping[str, Any] | None = None) -> dict[str, bytes]:
    """Rebuild exactly three portable files from a previously verified tiny profile.

    The optional inspect result is checked, never trusted over the embedded spec
    and load manifest. Without an inspect transcript, the same structural fields
    are derived explicitly and labelled as an inspect-compatible projection.
    """
    if not validation_directory.is_dir() or validation_directory.is_symlink():
        _fail("learning source must be an ordinary validation directory")
    report = _object(validation_directory / "validation-report.json")
    spec_record = _object(validation_directory / "model-spec.json")
    weights = _object(validation_directory / "weight-load-manifest.json")
    _signed(report, "report_digest")
    _signed(weights, "manifest_digest")
    spec = SupportedModelSpec(**{key: value for key, value in spec_record.items() if key != "spec_id"})
    if (spec_record.get("spec_id") != spec.spec_id or spec.num_layers != 2 or spec.runtime_dtype != "fp32"
            or report.get("status") != "pass" or report.get("identity", {}).get("fixture_id") != "tiny-decoder"
            or report["identity"].get("spec_id") != spec.spec_id
            or report["identity"].get("load_manifest_digest") != weights["manifest_digest"]):
        _fail("learning diagrams require the validated two-layer tiny FP32 model")
    projection = _projection(spec, weights)
    if inspect_structure is not None:
        expected = {key: projection[key] for key in ("spec", "parameter_count", "canonical_tensor_shapes")}
        if any(inspect_structure.get(key) != value for key, value in expected.items()) or inspect_structure.get("weight_manifest") != weights:
            _fail("actual inspect output differs from validation-bound model structure")
    simple_name, simple = _diagnostic(validation_directory, report, "first_divergence_diagnostic", compound=False)
    compound_name, compound = _diagnostic(validation_directory, report, "token_mismatch_diagnostic", compound=True)
    names = ("validation-report.json", "model-spec.json", "weight-load-manifest.json", simple_name, compound_name)
    hashes = {name: sha256_file(validation_directory / name) for name in names}
    assets = {"model-structure.svg": _structure(projection, report["report_digest"]),
              "first-divergence.svg": _divergence(simple, compound, report["report_digest"], hashes)}
    provenance: dict[str, Any] = {"schema_version": 1, "generator_id": GENERATOR_ID,
                                 "artifact_kind": "generated_svg_diagram_not_screenshot", "new_model_execution": False,
                                 "structural_evidence": "derived_inspect_compatible_projection_of_recorded_spec_and_storage_metadata",
                                 "validation_report_digest": report["report_digest"], "inputs": hashes,
                                 "inspect_projection_digest": sha256_json(projection),
                                 "outputs": {name: sha256_bytes(data) for name, data in assets.items()}}
    provenance["provenance_digest"] = sha256_json(provenance)
    assets["provenance.json"] = canonical_json_bytes(provenance) + b"\n"
    return assets

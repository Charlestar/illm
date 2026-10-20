"""Publisher unit tests use explicit mocked profile/rollback boundaries.

These small records are not numerical or hardware evidence. The real profile
verifier rejects them, and its own adversarial suite tests semantic evidence.
"""

from __future__ import annotations

import json
import copy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

import pytest

from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.capture.reference import ReferenceCaptureSession
from illm.domain.errors import ErrorCode, IllmError
from illm.domain.reference_events import ReferenceEventKind as K
from illm.reference_qualification import PROFILES, _fs, _inventory, verify_reference_profile_evidence
from illm.reference_release import (
    DOCUMENTS, DOCUMENT_SOURCES, FROZEN_SPEC_AXES, FROZEN_SPEC_PROFILES, PRIMARY, _state,
    build_reference_release_bundle, verify_frozen_v01_spec, verify_reference_release_bundle,
)
from illm.sampling import SamplingSpec
from illm.reference_learning import LEARNING_DIRECTORY, LEARNING_FILES, derive_reference_learning_assets


def _read(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(_fs(path).read_text(encoding="utf-8"))
    return result


def _write(path: Path, value: dict[str, Any]) -> None:
    path = _fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _sign(path: Path, value: dict[str, Any], field: str) -> None:
    value[field] = sha256_json({key: item for key, item in value.items() if key != field})
    _write(path, value)


def _frozen_spec_text() -> str:
    lines = [
        f"status: {FROZEN_SPEC_AXES['status']}",
        f"spec_status: {FROZEN_SPEC_AXES['spec_status']}",
        f"release_stage: {FROZEN_SPEC_AXES['release_stage']}",
        f"qualification_tier: {FROZEN_SPEC_AXES['qualification_tier']}",
        "qualification_profiles:",
    ]
    for profile_id, facts in FROZEN_SPEC_PROFILES.items():
        lines.extend((f"  - profile_id: {profile_id}", f"    result: {facts['result']}", f"    reason: {facts['reason']}"))
    lines.extend((
        f"performance_outcome: {FROZEN_SPEC_AXES['performance_outcome']}",
        f"lts_state: {FROZEN_SPEC_AXES['lts_state']}",
        f"spec_version: {FROZEN_SPEC_AXES['spec_version']}",
        f"target_release: {FROZEN_SPEC_AXES['target_release']}",
    ))
    return ("# Publisher unit fixture spec.md\nThis is a frozen source snapshot, not release evidence.\n\n```yaml\n"
            + "\n".join(lines)
            + "\n```\n\n## 12. Definition of Done\n\n### 12.1 Must\n\n- [ ] source-bound gates\n\n"
              "### 12.2 Conditional\n\n- [ ] conditional source-bound gate\n")


def _learning_unit_validation(directory: Path) -> None:
    """Actual tiny storage metadata plus synthetic diagnostics, NOT model proof."""
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "tiny-decoder"
    spec = _read(fixture / "supported-model-spec.json")
    weights = _read(fixture / "weight-load-manifest.json")
    _write(directory / "model-spec.json", spec)
    _write(directory / "weight-load-manifest.json", weights)
    identity = {"fixture_id": "tiny-decoder", "spec_id": spec["spec_id"], "load_manifest_digest": weights["manifest_digest"],
                "weights_digest": weights["weights_digest"]}
    validation: dict[str, Any] = {"schema_version": 1, "status": "pass", "identity": identity, "unit_only": True}
    for compound, field, name in ((False, "first_divergence_diagnostic", "fault-transposed-q.json"),
                                   (True, "token_mismatch_diagnostic", "fault-q-transpose-and-zero-final-norm.json")):
        first = {"op": "layer.0.q_projection", "status": "fail", "reason": "outside_tolerance", "evidence_kind": "measured",
                 "expected_shape": [1, 4, 6, 8], "actual_shape": [1, 4, 6, 8], "mismatch_count": 3,
                 "first_flat_index": 1, "max_absolute_error": 0.25, "max_relative_error": 0.5,
                 "expected_value": 0.5, "actual_value": 0.25,
                 "tolerance": {"atol": 0.0001, "rtol": 0.001, "profile_id": "synthetic-unit-only"}}
        fault_identity: dict[str, Any] = dict(identity)
        transforms = [{"transform_id": "transpose-axes-0-1-contiguous-v1"}]
        if compound:
            transforms.append({"transform_id": "zero-tensor-v1"})
            fault_identity["fault_transforms"] = transforms
        else:
            fault_identity["fault_transform"] = transforms[0]
        fault: dict[str, Any] = {"status": "fail", "report_link": name, "identity": fault_identity,
                                 "first_divergence": first,
                                 "ops": [{"op": "embedding", "status": "pass"}, {"op": "layer.0.input_norm", "status": "pass"}, first,
                                         *[{"op": "synthetic-unmeasured-unit-placeholder", "status": "pass"} for _ in range(36)]],
                                 "output_comparison": {"evidence_kind": "measured", "token_mismatch": compound,
                                                       "expected_next_token": 114, "actual_next_token": 0 if compound else 114,
                                                       "final_logits_max_absolute_error": 0.75}}
        _write(directory / name, fault)
        validation[field] = {"status": "pass", "report": fault}
    _sign(directory / "validation-report.json", validation, "report_digest")


def _unit_capture(path: Path) -> dict[str, Any]:
    sampling = SamplingSpec(actual_seed=7)
    identity = {"model_spec_id": "4" * 64, "weights_digest": "5" * 64, "load_manifest_digest": "6" * 64,
                "tokenizer_digest": "7" * 64, "input_spec_digest": "8" * 64, "runtime_dtype": "fp32", "device": "cpu", "backend": "illm_eager"}
    capture = ReferenceCaptureSession(path, identity, "publisher-unit-not-model-proof", sampling.to_dict())
    for kind, payload in (
        (K.REQUEST_ACCEPTED, {"max_new_tokens": 1}),
        (K.REQUEST_TOKENIZED, {"input_tokens": 8, "input_digest": "9" * 64}),
        (K.MODEL_FORWARD_STARTED, {"output_position": 0, "context_tokens": 8}),
        (K.MODEL_FORWARD_COMPLETED, {"output_position": 0, "logits_digest": "a" * 64}),
        (K.SAMPLING_STARTED, {"output_position": 0, "sampling_spec_digest": sampling.digest}),
        (K.SAMPLING_SELECTED, {"output_position": 0, "sampling_spec_digest": sampling.digest, "logits_digest": "a" * 64,
                               "draw_address": None, "selected_digest": "b" * 64}),
        (K.REQUEST_FINISHED, {"output_tokens": 1, "finish_reason": "MAX_NEW_TOKENS", "output_digest": "c" * 64}),
    ):
        capture.emit(kind, payload)
    return capture.finalize()


def _profile_boundary(path: Path, *, expected_source_identity: Mapping[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read(path / "unit-boundary-record.json")
    if expected_source_identity is not None and value["source_identity"] != dict(expected_source_identity):
        raise IllmError(ErrorCode.TOOLCHAIN_MISMATCH, "unit profile source mismatch")
    return value["profile"], {"source_identity": value["source_identity"]}


def _rollback_boundary(path: Path) -> dict[str, Any]:
    return _read(path / "rollback-report.json")


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "unit-inputs"
    root.mkdir()
    source = {"project_root_digest": "1" * 64, "bootstrap_digest": "2" * 64, "base_lock_digest": "3" * 64}
    docs = root / "documents"
    docs.mkdir()
    for name in DOCUMENTS:
        text = (f"# Publisher unit fixture {name}\nThis is deliberately synthetic text for portable publisher boundary tests, not a release claim.\n")
        if name == "spec.md":
            text = _frozen_spec_text()
        (docs / name).write_text(text, encoding="utf-8")
    doc_hashes = {DOCUMENT_SOURCES[name]: sha256_file(docs / name) for name in DOCUMENTS}
    evidence: list[Path] = []
    for profile_id in sorted(PROFILES):
        directory = root / profile_id
        directory.mkdir()
        toolchain = {"schema_version": 1, "finalized": True, "profile": "synthetic-publisher-unit-boundary",
                     "installed_packages": {"illm": "0.1.0"}, "runtime_profile_id": profile_id,
                     "project_root_digest": source["project_root_digest"], "bootstrap_digest": source["bootstrap_digest"]}
        _sign(directory / "toolchain.json", toolchain, "manifest_digest")
        (directory / "junit.xml").write_text('<testsuite tests="1" failures="0" errors="0" skipped="0" time="0.1"><testcase name="publisher_unit_only"/></testsuite>', encoding="utf-8")
        _write(directory / "quality.json", {"unit_only": True})
        (directory / "locks").mkdir()
        (directory / "locks" / "pyproject.toml").write_text('[project]\nname = "illm"\nversion = "0.1.0"\n', encoding="utf-8")
        _write(directory / "source-provenance.json", {"project_files": doc_hashes})
        _unit_capture(directory / "capture")
        _sign(directory / "benchmark" / "benchmark-report.json", {"schema_version": 1, "unit_only": True}, "report_digest")
        profile = {"profile_id": profile_id, "result": "pass", "required_for_functional": True,
                   "environment_manifest_digest": toolchain["manifest_digest"], "evidence_digest": sha256_json(profile_id), "reason": None}
        _write(directory / "unit-boundary-record.json", {"profile": profile, "source_identity": source})
        evidence.append(directory)
    primary = root / PRIMARY
    _learning_unit_validation(primary / "validation" / "model-0")
    rollback = root / "rollback"
    rollback.mkdir()
    (rollback / "commands.txt").write_text("Unit boundary only; no historical process was run here.\n", encoding="utf-8")
    report = {"schema_version": 1, "status": "pass", "evidence_mode": "release", "source_identity": source,
              "live_v01_source_verified": True, "primary_v01_toolchain_manifest_digest": _read(primary / "toolchain.json")["manifest_digest"],
              "capture_manifest_digest": _read(primary / "capture" / "capture-manifest.json")["manifest_digest"],
              "files": {"commands.txt": sha256_file(rollback / "commands.txt")}}
    _sign(rollback / "rollback-report.json", report, "report_digest")
    _write(rollback / "checksums.json", _inventory(rollback))
    (rollback / "COMPLETE").write_text(report["report_digest"] + "\n", encoding="ascii")
    local = rollback / "local-only"
    local.mkdir()
    (local / "do-not-publish.txt").write_text("Regenerable local work is not release evidence.\n", encoding="utf-8")
    return {"evidence": evidence, "rollback": rollback, "docs": docs, "source": source}


@pytest.fixture
def assembled(inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("illm.reference_release.verify_reference_profile_evidence", _profile_boundary)
    monkeypatch.setattr("illm.reference_release.verify_reference_rollback", _rollback_boundary)
    output = tmp_path / "v0.1.0"
    result = build_reference_release_bundle(evidence_roots=inputs["evidence"], rollback=inputs["rollback"],
                                            documentation_root=inputs["docs"], output=output, expected_source_identity=inputs["source"])
    assert result["release_stage"] == "release_candidate"
    return output


def _resign(root: Path, *, update_state: bool = False) -> None:
    manifest = _read(root / "manifest.json")
    _sign(root / "manifest.json", manifest, "manifest_digest")
    if update_state:
        _write(root / "qualification" / "release-state.json", _state(root / "manifest.json"))
    (root / "COMPLETE").write_text(manifest["manifest_digest"] + "\n", encoding="ascii")
    _write(root / "SHA256SUMS.json", {name: digest for name, digest in _inventory(root).items() if name != "SHA256SUMS.json"})


def test_complete_candidate_derives_test_counts_and_omits_local_only_work(assembled: Path, inputs: dict[str, Any]) -> None:
    state = verify_reference_release_bundle(assembled / "manifest.json", expected_source_identity=inputs["source"])
    assert state["qualification_tier"] == "functionally_qualified"
    assert state["performance_outcome"] == "not_applicable"
    assert not any(path.name == "local-only" for path in assembled.rglob("*"))
    reports = _read(assembled / "test-report.json")["profiles"]
    assert set(reports) == PROFILES
    assert all(report["tests"] == 1 and report["failures"] == 0 for report in reports.values())
    assert {path.name for path in (assembled / LEARNING_DIRECTORY).iterdir()} == LEARNING_FILES
    frozen_spec = (assembled / "spec.md").read_bytes()
    manifest = _read(assembled / "manifest.json")
    manifest.update(spec_status="released", release_stage="released")
    _write(assembled / "manifest.json", manifest)
    _resign(assembled, update_state=True)
    assert verify_reference_release_bundle(assembled / "manifest.json")["release_stage"] == "released"
    assert (assembled / "spec.md").read_bytes() == frozen_spec


def test_frozen_spec_reader_rejects_target_state_and_checked_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "spec.md"
    source = _frozen_spec_text()
    path.write_text(source, encoding="utf-8")
    verify_frozen_v01_spec(path)
    for mutated in (
        source.replace("release_stage: dev_snapshot", "release_stage: released"),
        source.replace("    result: not_run", "    result: pass", 1),
        source.replace("- [ ] source-bound gates", "- [x] source-bound gates"),
    ):
        path.write_text(mutated, encoding="utf-8")
        with pytest.raises(IllmError):
            verify_frozen_v01_spec(path)


def test_repository_v01_spec_is_the_frozen_pre_gate_snapshot() -> None:
    path = Path(__file__).resolve().parents[2] / "docs" / "specs" / "versions" / "v0.1-reference-runner.md"
    result = verify_frozen_v01_spec(path)
    assert result["axes"]["release_stage"] == "dev_snapshot"
    assert all(profile["result"] == "not_run" for profile in result["qualification_profiles"].values())


@pytest.mark.parametrize("mutation", ["missing-profile", "status-only", "mixed-source", "checksum", "rollback-development", "fake-version", "fake-test-count", "missing-document", "path-traversal", "local-only", "stale-state", "speedup-claim"])
def test_candidate_rejects_resigned_incomplete_or_overclaimed_facts(assembled: Path, mutation: str) -> None:
    manifest = _read(assembled / "manifest.json")
    if mutation == "missing-profile":
        manifest["qualification_profiles"].pop()
        _write(assembled / "manifest.json", manifest)
    elif mutation == "status-only":
        manifest["qualification_profiles"][0] = {"profile_id": "clean-room-release-v1", "result": "pass"}
        _write(assembled / "manifest.json", manifest)
    elif mutation == "mixed-source":
        record_path = next((assembled / "qualification").rglob("unit-boundary-record.json"))
        record = _read(record_path)
        record["source_identity"]["project_root_digest"] = "f" * 64
        _write(record_path, record)
    elif mutation == "checksum":
        (assembled / "toolchain.json").write_text("{}", encoding="utf-8")
        with pytest.raises(IllmError):
            verify_reference_release_bundle(assembled / "manifest.json")
        return
    elif mutation == "rollback-development":
        path = assembled / "rollback" / "rollback-report.json"
        record = _read(path)
        record["evidence_mode"] = "development"
        _sign(path, record, "report_digest")
    elif mutation == "fake-version":
        manifest["release_version"] = "0.1.0.dev0"
        _write(assembled / "manifest.json", manifest)
    elif mutation == "fake-test-count":
        path = assembled / "test-report.json"
        report = _read(path)
        report["profiles"][PRIMARY]["tests"] += 100
        _sign(path, report, "report_digest")
        manifest["test_report_digest"] = report["report_digest"]
        _write(assembled / "manifest.json", manifest)
    elif mutation == "missing-document":
        (assembled / "teach-back.md").unlink()
    elif mutation == "path-traversal":
        sums = _read(assembled / "SHA256SUMS.json")
        sums["../outside"] = "0" * 64
        _write(assembled / "SHA256SUMS.json", sums)
        with pytest.raises(IllmError):
            verify_reference_release_bundle(assembled / "manifest.json")
        return
    elif mutation == "local-only":
        local = assembled / "rollback" / "local-only"
        local.mkdir()
        (local / "forbidden.txt").touch()
    elif mutation == "stale-state":
        _write(assembled / "qualification" / "release-state.json", {"qualification_tier": "stable_released"})
    else:
        manifest["performance_outcome"] = "Adopt"
        _write(assembled / "manifest.json", manifest)
    _resign(assembled)
    with pytest.raises(IllmError):
        verify_reference_release_bundle(assembled / "manifest.json")


def test_real_profile_boundary_rejects_publisher_unit_stubs(inputs: dict[str, Any]) -> None:
    with pytest.raises(IllmError):
        verify_reference_profile_evidence(inputs["evidence"][0])


def test_builder_refuses_development_package_even_with_passing_boundary_labels(inputs: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("illm.reference_release.verify_reference_profile_evidence", _profile_boundary)
    primary = next(path for path in inputs["evidence"] if path.name == PRIMARY)
    toolchain = _read(primary / "toolchain.json")
    toolchain["installed_packages"]["illm"] = "0.1.0.dev0"
    _sign(primary / "toolchain.json", toolchain, "manifest_digest")
    with pytest.raises(IllmError, match="final package version"):
        build_reference_release_bundle(evidence_roots=inputs["evidence"], rollback=inputs["rollback"],
                                        documentation_root=inputs["docs"], output=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_release_rejects_symbolic_link_root(assembled: Path, tmp_path: Path) -> None:
    link = tmp_path / "linked-release"
    try:
        link.symlink_to(assembled, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating symbolic links")
    with pytest.raises(IllmError, match="ordinary manifest"):
        verify_reference_release_bundle(link / "manifest.json")


@pytest.mark.parametrize("mutation", ["svg", "provenance", "extra", "extra-directory", "missing"])
def test_release_rebuilds_learning_diagrams_even_after_outer_checksums_are_resigned(assembled: Path, mutation: str) -> None:
    directory = assembled / LEARNING_DIRECTORY
    if mutation == "svg":
        path = directory / "first-divergence.svg"
        path.write_bytes(path.read_bytes().replace(b"114 -&gt; 114", b"114 -&gt; 0"))
    elif mutation == "provenance":
        record = _read(directory / "provenance.json")
        record["new_model_execution"] = True
        _sign(directory / "provenance.json", record, "provenance_digest")
    elif mutation == "extra":
        (directory / "extra.js").write_text("console.log('not allowed')", encoding="utf-8")
    elif mutation == "extra-directory":
        (directory / "extra-empty-directory").mkdir()
    else:
        (directory / "model-structure.svg").unlink()
    _resign(assembled)
    with pytest.raises(IllmError, match="learning diagram"):
        verify_reference_release_bundle(assembled / "manifest.json")


def test_learning_svg_is_deterministic_escaped_and_bound_to_actual_structure(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    _learning_unit_validation(validation)
    report = _read(validation / "validation-report.json")
    for key in ("first_divergence_diagnostic", "token_mismatch_diagnostic"):
        fault = report[key]["report"]
        fault["first_divergence"]["tolerance"]["profile_id"] = '<script>alert("x")</script>&test'
        fault["ops"][2] = copy.deepcopy(fault["first_divergence"])
        _write(validation / fault["report_link"], fault)
    _sign(validation / "validation-report.json", report, "report_digest")
    assets = derive_reference_learning_assets(validation)
    assert assets == derive_reference_learning_assets(validation)
    assert set(assets) == LEARNING_FILES
    for name in ("model-structure.svg", "first-divergence.svg"):
        xml = ET.fromstring(assets[name])
        assert xml.tag == "{http://www.w3.org/2000/svg}svg"
        assert all(element.tag.split("}")[-1] not in {"script", "image", "foreignObject", "a"} for element in xml.iter())
        assert all(not any("href" in key or key.startswith("on") for key in element.attrib) for element in xml.iter())
    text = assets["model-structure.svg"].decode("utf-8")
    assert "27008 unique parameters" in text and "26 canonical storages" in text
    assert "layers=2" in text and "LM head reuses embedding storage" in text
    diff = assets["first-divergence.svg"].decode("utf-8")
    assert "114 -&gt; 114" in diff and "114 -&gt; 0" in diff
    assert "&lt;script&gt;" in diff and "<script>" not in diff
    provenance = json.loads(assets["provenance.json"])
    assert provenance["new_model_execution"] is False
    assert provenance["inputs"]["validation-report.json"] == sha256_file(validation / "validation-report.json")
    assert provenance["outputs"]["first-divergence.svg"] == hashlib.sha256(assets["first-divergence.svg"]).hexdigest()
    spec, weights = _read(validation / "model-spec.json"), _read(validation / "weight-load-manifest.json")
    from illm.model.spec import SupportedModelSpec
    model = SupportedModelSpec(**{key: value for key, value in spec.items() if key != "spec_id"})
    inspect = {"spec": spec, "weight_manifest": weights, "parameter_count": model.parameter_count,
               "canonical_tensor_shapes": {name: list(shape) for name, shape in model.tensor_shapes().items()}}
    assert derive_reference_learning_assets(validation, inspect_structure=inspect) == assets
    inspect["parameter_count"] = 1
    with pytest.raises(IllmError, match="inspect output"):
        derive_reference_learning_assets(validation, inspect_structure=inspect)


@pytest.mark.parametrize("mutation", ["digest", "wrong-first-op", "false-token-claim", "report-link", "nonfinite", "wrong-shape", "duplicate-json"])
def test_learning_svg_rejects_inconsistent_bound_facts(tmp_path: Path, mutation: str) -> None:
    validation = tmp_path / "validation"
    _learning_unit_validation(validation)
    path = validation / "validation-report.json"
    report = _read(path)
    simple = report["first_divergence_diagnostic"]["report"]
    if mutation == "digest":
        report["report_digest"] = "0" * 64
        _write(path, report)
    elif mutation == "duplicate-json":
        path.write_text('{"status":"pass","status":"pass"}', encoding="utf-8")
    elif mutation == "wrong-shape":
        weights = _read(validation / "weight-load-manifest.json")
        weights["storages"][0]["shape"] = [1, 1]
        _sign(validation / "weight-load-manifest.json", weights, "manifest_digest")
        report["identity"]["load_manifest_digest"] = weights["manifest_digest"]
        _sign(path, report, "report_digest")
    else:
        if mutation == "wrong-first-op":
            simple["first_divergence"]["op"] = "logits"
            simple["ops"][2] = simple["first_divergence"]
        elif mutation == "false-token-claim":
            simple["output_comparison"]["token_mismatch"] = True
        elif mutation == "report-link":
            simple["report_link"] = "../outside.json"
        else:
            simple["first_divergence"]["actual_value"] = float("inf")
            # JSON nonfinite constants must be rejected before drawing.
            path.write_text(json.dumps(report), encoding="utf-8")
            with pytest.raises(IllmError):
                derive_reference_learning_assets(validation)
            return
        _write(validation / "fault-transposed-q.json", simple)
        _sign(path, report, "report_digest")
    with pytest.raises(IllmError):
        derive_reference_learning_assets(validation)

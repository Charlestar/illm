"""Release guardrails. Header fixtures are not execution or release evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, cast

import pytest

from illm.canonical import canonical_json_bytes, sha256_json
from illm.domain.errors import IllmError
import illm.stateful_release as release
from illm.stateful_qualification import FORMAL_PROFILES, SOURCE_FIELDS


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _header() -> dict[str, Any]:
    # Unit-only shape fixture; no actual artifacts or passing qualification.
    value: dict[str, Any] = {
        "schema_version": 2, "report_kind": release.REPORT_KIND,
        "release_version": release.VERSION, "release_tag": release.TAG,
        "spec_status": "implemented", "release_stage": "release_candidate",
        "qualification_tier": "functionally_qualified", "performance_outcome": "not_applicable",
        "lts_state": "not_applicable", "source_identity": {key: "a" * 64 for key in SOURCE_FIELDS},
        "primary_profile_id": release.PRIMARY,
        "qualification_profiles": [
            {"profile_id": name, "result": "pass", "evidence_mode": "formal", "simulated_profile_id": None,
             "required_for_functional": True, "reason": None, "evidence_digest": "b" * 64,
             "environment_manifest_digest": "c" * 64}
            for name in sorted(FORMAL_PROFILES)
        ],
        "capture_manifests": {}, "execution_identities": {},
        "benchmark_report_digests": {name: "d" * 64 for name in release.BENCHMARKS},
    }
    for name in ("toolchain_manifest_digest", "capture_manifest_digest", "metric_registry_digest",
                 "test_report_digest", "rollback_report_digest", "documentation_digest"):
        value[name] = "e" * 64
    value["manifest_digest"] = sha256_json(value)
    return value


def _resign(value: dict[str, Any]) -> None:
    value.pop("manifest_digest", None)
    value["manifest_digest"] = sha256_json(value)


@pytest.mark.parametrize("mutation", ["missing_gpu", "synthetic", "failed", "not_required", "development",
                                           "wrong_schema", "missing_suite", "performance_claim", "duplicate_profile"])
def test_header_never_grants_partial_development_or_synthetic_release(tmp_path: Path, mutation: str) -> None:
    value = _header()
    if mutation == "missing_gpu":
        value["qualification_profiles"] = [item for item in value["qualification_profiles"] if item["profile_id"] != release.GPU]
    elif mutation == "synthetic":
        value["qualification_profiles"][0]["evidence_mode"] = "synthetic_test_fixture"
    elif mutation == "failed":
        value["qualification_profiles"][0]["result"] = "fail"
    elif mutation == "not_required":
        value["qualification_profiles"][0]["required_for_functional"] = 1
    elif mutation == "development":
        value["release_version"] = "0.2.0.dev0"
    elif mutation == "wrong_schema":
        value["schema_version"] = True
    elif mutation == "missing_suite":
        value["benchmark_report_digests"].pop("fp16-heldout")
    elif mutation == "performance_claim":
        value["performance_outcome"] = "improved"
    else:
        value["qualification_profiles"][0] = value["qualification_profiles"][1]
    _resign(value)
    manifest = tmp_path / "manifest.json"
    _write(manifest, value)
    with pytest.raises(IllmError):
        release._state(manifest)


def test_header_shape_is_not_a_complete_bundle(tmp_path: Path) -> None:
    value = _header()
    manifest = tmp_path / "manifest.json"
    _write(manifest, value)
    with pytest.raises(IllmError, match="incomplete"):
        release.verify_stateful_release_bundle(manifest)


@pytest.mark.parametrize("source_version,installed_version", [("0.2.0.dev0", "0.2.0.dev0"),
                                                             ("0.2.0", "0.1.0"), ("0.1.0", "0.2.0")])
def test_final_package_must_be_final_in_both_metadata_and_environment(
    tmp_path: Path, source_version: str, installed_version: str,
) -> None:
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks/pyproject.toml").write_text(f'[project]\nname="illm"\nversion="{source_version}"\n', encoding="utf-8")
    _write(tmp_path / "toolchain.json", {"installed_packages": {"illm": installed_version}})
    with pytest.raises(IllmError, match="development package"):
        release._final_package(tmp_path)


def test_builder_rejects_missing_profiles_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(IllmError, match="three sealed profiles"):
        release.build_stateful_release_bundle(evidence_roots=[], rollback=tmp_path / "missing-rollback",
                                              documentation_root=tmp_path / "missing-docs", output=output)
    assert not output.exists()


def test_frozen_spec_cannot_claim_accepted_as_implemented_or_precheck_done(tmp_path: Path) -> None:
    metadata = {"status": "implemented", "spec_status": "implemented", "release_stage": "dev_snapshot",
                "qualification_tier": "unqualified", "target_release": "v0.2.0",
                "performance_outcome": "not_applicable", "lts_state": "not_applicable", "spec_version": "2"}
    profiles = "qualification_profiles:\n" + "".join(
        f"  - profile_id: {name}\n    result: not_run\n    reason: final_source_bound_gate_pending\n"
        for name in sorted(FORMAL_PROFILES)
    )
    header = "```yaml\n" + "\n".join(f"{key}: {value}" for key, value in metadata.items()) + "\n" + profiles + "```\n"
    headings = "## 12. Definition of Done\n### 12.1 Must\n### 12.2 Conditional\n"
    path = tmp_path / "spec.md"
    path.write_text(header + headings + "- [ ] Real gates pending\n", encoding="utf-8")
    release.verify_frozen_v02_spec(path)
    path.write_text(header.replace("status: implemented", "status: accepted") + headings, encoding="utf-8")
    with pytest.raises(IllmError, match="unqualified source-frozen"):
        release.verify_frozen_v02_spec(path)
    path.write_text(header + headings + "- [x] Falsely prechecked\n", encoding="utf-8")
    with pytest.raises(IllmError, match="pre-claim"):
        release.verify_frozen_v02_spec(path)
    path.write_text(header.replace("result: not_run", "result: pass") + headings, encoding="utf-8")
    with pytest.raises(IllmError, match="pre-gate facts"):
        release.verify_frozen_v02_spec(path)


def test_release_dispatch_is_schema_specific_and_portable_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from illm.release import verify_release_bundle

    manifest = tmp_path / "manifest.json"
    _write(manifest, {"schema_version": 2})
    calls: list[tuple[Path, object]] = []

    def reader(path: Path, *, expected_source_identity: object = None) -> dict[str, Any]:
        calls.append((path, expected_source_identity))
        return {"unit_dispatch_only": True}

    monkeypatch.setattr(release, "verify_stateful_release_bundle", reader)
    assert verify_release_bundle(manifest) == {"unit_dispatch_only": True}
    assert calls == [(manifest, None)]
    identity = {key: "a" * 64 for key in SOURCE_FIELDS}
    verify_release_bundle(manifest, expected_source_identity=identity)
    assert calls[-1] == (manifest, identity)


def test_current_source_cli_uses_common_dispatch_not_old_v01_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    import illm.cli as cli

    seen: dict[str, Any] = {}
    identity = {key: "a" * 64 for key in SOURCE_FIELDS}

    def reader(path: Path, **kwargs: Any) -> dict[str, Any]:
        seen.update(path=path, **kwargs)
        return {"unit_dispatch_only": True}

    monkeypatch.setattr(cli, "verify_release_bundle", reader)
    monkeypatch.setattr(cli, "reference_source_identity", lambda root: identity)
    args = argparse.Namespace(require_current_source=True, manifest=tmp_path / "manifest.json", output=None)
    _write(args.manifest, {"schema_version": 2})
    assert cli._release_verify(args) == 0
    assert seen == {"path": args.manifest, "expected_source_identity": identity}


def test_duplicate_json_header_is_not_resignable_evidence(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
    with pytest.raises(IllmError):
        release._state(tmp_path / "manifest.json")


def test_no_tensor_dependency_for_release_reader() -> None:
    import subprocess
    import sys

    program = (
        "import sys; from illm.stateful_release import verify_stateful_release_bundle; "
        "assert not any(name.split('.')[0] in {'torch','transformers','safetensors','tokenizers'} for name in sys.modules); "
        "print('portable-import-pass')"
    )
    process = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "portable-import-pass"


def test_release_bounds_reject_nonregular_entries_even_when_not_in_file_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden = tmp_path / "nonregular-entry"
    hidden.write_bytes(b"unit stand-in for a nonregular entry\n")
    original_is_file = Path.is_file

    def stand_in(path: Path) -> bool:
        return False if path == hidden else original_is_file(path)

    monkeypatch.setattr(Path, "is_file", stand_in)
    with pytest.raises(IllmError, match="nonregular"):
        release._bounded(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="actual FIFO safety is a POSIX profile check")
def test_public_reader_rejects_actual_fifo_before_hashing_or_opening_it(tmp_path: Path) -> None:
    make_fifo = cast(Callable[[Path], None], getattr(os, "mkfifo"))
    make_fifo(tmp_path / "unhashed-fifo")
    (tmp_path / "COMPLETE").write_text("not-a-real-bundle\n", encoding="ascii")
    with pytest.raises(IllmError, match="nonregular"):
        release.verify_stateful_release_bundle(tmp_path / "manifest.json")

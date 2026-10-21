"""CI routing/metadata checks; mocked reports are unit-only, never evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from illm.domain.errors import ErrorCode, IllmError


ROOT = Path(__file__).resolve().parents[2]


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v02_release_snapshot_unit", ROOT / "scripts/check_v02_release_snapshot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project(tmp_path: Path, version: str) -> tuple[Path, Path]:
    project = tmp_path / "unit-project-not-published"
    project.mkdir()
    (project / "pyproject.toml").write_text(f'[project]\nname="illm"\nversion="{version}"\n', encoding="utf-8")
    manifest = project / "release/v0.2.0/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"unit_only_not_evidence":true}', encoding="utf-8")
    return project, manifest


def _mock_report() -> dict[str, Any]:
    return {"release_tag": "v0.2.0", "release_stage": "released", "spec_status": "released",
            "manifest_digest": "a" * 64, "verification_digest": "b" * 64, "unit_only_not_evidence": True}


@pytest.mark.parametrize("version,ci,event_ref,binding", [
    ("0.2.0", False, "", "current"),
    ("0.2.0", True, "refs/heads/main", "current"),
    ("0.2.0", True, "refs/tags/v0.2.0", "current"),
    ("0.3.0.dev0", True, "refs/heads/main", "historical"),
    ("0.2.0.dev0", True, "refs/pull/1/merge", "historical"),
])
def test_ci_does_not_break_historical_versions_or_skip_exact_release_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, ci: bool, event_ref: str, binding: str,
) -> None:
    module = _module()
    project, manifest = _project(tmp_path, version)
    identity = {name: "c" * 64 for name in ("project_root_digest", "bootstrap_digest", "base_lock_digest")}
    calls: list[object] = []

    def source(root: Path) -> dict[str, str]:
        assert root == project
        assert binding == "current", "historical verification must not compute a current source identity"
        return identity

    def verify(path: Path, *, expected_source_identity: object = None) -> dict[str, Any]:
        assert path == manifest
        calls.append(expected_source_identity)
        return _mock_report()

    monkeypatch.setattr(module, "stateful_source_identity", source)
    monkeypatch.setattr(module, "verify_stateful_release_bundle", verify)
    result = module.check_snapshot(manifest, project, ci=ci, event_ref=event_ref)
    assert result["source_binding"] == binding and result["git_tag_verified"] is False
    assert calls == [identity if binding == "current" else None]


@pytest.mark.parametrize("version,ci,event_ref", [
    ("0.2.0.dev0", False, ""), ("0.3.0.dev0", False, ""),
    ("0.3.0.dev0", True, "refs/tags/v0.2.0"), ("0.2.0.dev0", True, "refs/tags/v0.2.0"),
])
def test_publication_and_tag_event_never_fall_back_to_historical(
    tmp_path: Path, version: str, ci: bool, event_ref: str,
) -> None:
    module = _module()
    project, manifest = _project(tmp_path, version)
    with pytest.raises(IllmError) as error:
        module.check_snapshot(manifest, project, ci=ci, event_ref=event_ref)
    assert error.value.code is ErrorCode.TOOLCHAIN_MISMATCH


def test_rc_is_rejected_even_if_the_bundle_reader_accepts_valid_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    project, manifest = _project(tmp_path, "0.3.0.dev0")
    candidate = {**_mock_report(), "release_stage": "release_candidate", "spec_status": "implemented"}
    monkeypatch.setattr(module, "verify_stateful_release_bundle", lambda *args, **kwargs: candidate)
    with pytest.raises(IllmError, match="candidate"):
        module.check_snapshot(manifest, project, ci=True)


def test_public_reader_still_rejects_unit_only_artifacts(tmp_path: Path) -> None:
    module = _module()
    project, manifest = _project(tmp_path, "0.3.0.dev0")
    with pytest.raises(IllmError):
        module.check_snapshot(manifest, project, ci=True)


def test_tag_event_with_missing_bundle_fails_real_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    project = tmp_path / "unit-project-without-release"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="illm"\nversion="0.2.0"\n', encoding="utf-8")
    identity = {name: "c" * 64 for name in ("project_root_digest", "bootstrap_digest", "base_lock_digest")}
    monkeypatch.setattr(module, "stateful_source_identity", lambda path: identity)
    with pytest.raises(IllmError):
        module.check_snapshot(project / "release/v0.2.0/manifest.json", project, ci=True, event_ref="refs/tags/v0.2.0")


def test_noncanonical_manifest_is_rejected(tmp_path: Path) -> None:
    module = _module()
    project, _ = _project(tmp_path, "0.3.0.dev0")
    with pytest.raises(IllmError, match="ordinary"):
        module.check_snapshot(project / "out/other-manifest.json", project, ci=True)


def test_ci_keeps_old_bundles_historical_and_checks_v02_released_source_mode() -> None:
    workflow = (ROOT / ".github/workflows/cpu.yml").read_text(encoding="utf-8")
    guards = [line.strip() for line in workflow.splitlines() if "hashFiles('release/v0.2.0/manifest.json')" in line]
    assert guards == ["if: ${{ github.ref == 'refs/tags/v0.2.0' || hashFiles('release/v0.2.0/manifest.json') != '' }}"]
    v02 = [line for line in workflow.splitlines() if "check_v02_release_snapshot.py" in line]
    assert len(v02) == 1 and "--ci" in v02[0] and "release/v0.2.0/manifest.json" in v02[0]
    old = [line for line in workflow.splitlines() if "illm release verify release/v0." in line]
    assert old and all("--require-current-source" not in line for line in old)

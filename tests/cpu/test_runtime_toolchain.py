from __future__ import annotations

import hashlib
import io
import json
import email
from pathlib import Path
import platform
import shutil
import sys
from typing import Any
import zipfile

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _runtime_artifacts as runtime
import _toolchain_common as common
import bootstrap
import fetch_model
import fetch_runtime
import in_env


def _resign(value: dict[str, Any]) -> None:
    value["lock_digest"] = common.sha256_bytes(common.canonical_bytes({key: item for key, item in value.items() if key != "lock_digest"}))


@pytest.mark.parametrize("profile_id,count", [
    ("py312-win-amd64-cpu", 25),
    ("py312-win-amd64-cpu-ci", 25),
    ("py312-linux-x86_64-cpu", 25),
    ("py312-linux-x86_64-cu126", 40),
])
def test_runtime_profiles_pin_each_wheel_and_match_shared_authority(profile_id: str, count: int) -> None:
    lock = runtime.load_runtime_lock(ROOT, profile_id)
    assert len(lock["artifacts"]) == count
    assert all(item["sha256"] in item["cache_key"] for item in lock["artifacts"])
    expected = common.expected_package_versions(ROOT, lock)
    assert expected["transformers"] == "4.57.1"
    assert expected["setuptools"] == "83.0.0"
    assert expected["packaging"] == "26.3"
    assert expected["torch"] == "2.8.0+" + lock["target"]["torch_channel"]
    assert common.lock_digest(ROOT, lock) != common.lock_digest(ROOT)


def test_ci_only_windows_profile_reuses_identical_wheels_with_unchanged_dependency_closure() -> None:
    formal = runtime.load_runtime_lock(ROOT, "py312-win-amd64-cpu")
    ci = runtime.load_runtime_lock(ROOT, "py312-win-amd64-cpu-ci")
    assert ci["target"]["python_validation_patch"] == "3.12.10"
    assert formal["target"]["python_validation_patch"] == "3.12.14"
    assert ci["marker_environment"]["implementation_version"] == "3.12.10"
    assert ci["marker_environment"]["python_full_version"] == "3.12.10"
    assert ci["artifacts"] == formal["artifacts"]
    assert ci["roots"] == formal["roots"]
    assert ci["dependency_edges"] == formal["dependency_edges"]
    assert ci["lock_digest"] != formal["lock_digest"]
    assert formal["lock_digest"] == "5149ad41ff2e0f261fbec8fe11dcefe2bed90ab8431b1c5339b8e08b59320a34"
    version = Version("3.12.10")
    supported = set(cpython_tags((3, 12), abis=["cp312"], platforms=["win_amd64"]))
    supported.update(compatible_tags((3, 12), "cp312", ["win_amd64"]))
    selected = runtime.runtime_packages(ci)
    edges: list[dict[str, str]] = []
    for artifact in ci["artifacts"]:
        assert not artifact["requires_python"] or version in SpecifierSet(artifact["requires_python"])
        assert parse_wheel_filename(artifact["filename"])[3] & supported
        for raw in artifact["requires_dist"]:
            requirement = Requirement(raw)
            original_active = requirement.marker is None or requirement.marker.evaluate(formal["marker_environment"])
            active = requirement.marker is None or requirement.marker.evaluate(ci["marker_environment"])
            assert active == original_active
            if active:
                name = common.normalize_distribution_name(requirement.name)
                assert name in selected
                assert Version(selected[name]) in requirement.specifier
                edges.append({"from": artifact["name"], "requirement": raw, "to": name})
    for raw in ci["roots"]:
        requirement = Requirement(raw)
        name = common.normalize_distribution_name(requirement.name)
        assert Version(selected[name]) in requirement.specifier
        edges.append({"from": "root", "requirement": raw, "to": name})
    assert len(edges) == 36
    assert sorted((edge["from"], edge["requirement"], edge["to"]) for edge in edges) == sorted(
        (edge["from"], edge["requirement"], edge["to"]) for edge in ci["dependency_edges"])
    vendor = sorted((ROOT / "vendor" / "wheels").glob("*.whl"))
    assert len(vendor) == 13
    for path in vendor:
        assert parse_wheel_filename(path.name)[3] & supported
        with zipfile.ZipFile(path) as wheel:
            metadata_names = [name for name in wheel.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            assert len(metadata_names) == 1
            metadata = email.message_from_bytes(wheel.read(metadata_names[0]))
        requires_python = metadata.get("Requires-Python")
        assert requires_python is None or version in SpecifierSet(requires_python)


def test_ci_only_runtime_does_not_replace_formal_alias_or_exact_python_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    formal = runtime.load_runtime_lock(ROOT, "v01-cpu")
    ci = runtime.load_runtime_lock(ROOT, "py312-win-amd64-cpu-ci")
    assert formal["profile_id"] == "py312-win-amd64-cpu"
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.10")
    runtime.verify_runtime_host(ci)
    with pytest.raises(common.ToolchainFailure, match="host identity mismatch"):
        runtime.verify_runtime_host(formal)
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.14")
    runtime.verify_runtime_host(formal)
    with pytest.raises(common.ToolchainFailure, match="host identity mismatch"):
        runtime.verify_runtime_host(ci)


def test_resealed_runtime_lock_cannot_change_registry_authority(tmp_path: Path) -> None:
    shutil.copytree(ROOT / "runtime", tmp_path / "runtime")
    path = tmp_path / "runtime" / "artifacts" / "py312-win-amd64-cpu.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    lock["artifacts"][0]["sha256"] = "0" * 64
    _resign(lock)
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(common.ToolchainFailure, match="lock identity mismatch"):
        runtime.load_runtime_lock(tmp_path, "py312-win-amd64-cpu")


def test_future_root_locks_do_not_contaminate_v00_package_authority(tmp_path: Path) -> None:
    (tmp_path / "vendor" / "wheels").mkdir(parents=True)
    (tmp_path / "vendor" / "wheels" / "fixture.whl").write_bytes(b"fixture")
    (tmp_path / "requirements-bootstrap.lock").write_text("base==1\n", encoding="utf-8")
    (tmp_path / "requirements-dev.lock").write_text("dev==2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="illm"\nversion="0.0.0"\n', encoding="utf-8")
    before = common.lock_digest(tmp_path)
    (tmp_path / "requirements-accidental.lock").write_text("base==99\nunrelated==1\n", encoding="utf-8")
    assert common.lock_digest(tmp_path) == before
    assert common.expected_package_versions(tmp_path) == {"base": "1", "dev": "2", "illm": "0.0.0"}
    with pytest.raises(common.ToolchainFailure, match="dependency pins conflict"):
        common.expected_package_versions(tmp_path, {"artifacts": [{"name": "base", "version": "99"}]})


def test_runtime_host_mismatch_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = runtime.load_runtime_lock(ROOT, "py312-win-amd64-cpu")
    monkeypatch.setattr(platform, "python_version", lambda: "3.13.0")
    with pytest.raises(common.ToolchainFailure, match="host identity mismatch"):
        runtime.verify_runtime_host(lock)


def test_missing_cache_never_removes_existing_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mutation: list[str] = []
    monkeypatch.setattr(bootstrap, "verify_runtime_host", lambda lock: None)
    monkeypatch.setattr(bootstrap, "_safe_remove_environment", lambda *args: mutation.append("removed"))
    with pytest.raises(common.ToolchainFailure, match="cache miss"):
        bootstrap.bootstrap([
            "--profile", "runtime-missing-cache-test", "--runtime-profile", "py312-win-amd64-cpu",
            "--locked", "--offline", "--artifact-cache", str(tmp_path / "empty-cache"),
            "--write-manifest", str(ROOT / "out" / "runtime-missing-cache-test.json"),
        ])
    assert mutation == []


def test_unknown_runtime_profile_fails_before_environment_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    mutation: list[str] = []
    monkeypatch.setattr(bootstrap, "_safe_remove_environment", lambda *args: mutation.append("removed"))
    with pytest.raises(common.ToolchainFailure, match="unknown runtime profile"):
        bootstrap.bootstrap([
            "--profile", "runtime-unknown-test", "--runtime-profile", "../../escape",
            "--locked", "--write-manifest", str(ROOT / "out" / "unknown-runtime.json"),
        ])
    assert mutation == []


def test_runtime_artifact_checks_bytes_and_only_root_metadata(tmp_path: Path) -> None:
    path = tmp_path / "fixture.whl"
    metadata = b"Name: fixture\nVersion: 1\n"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("fixture-1.dist-info/METADATA", metadata)
        wheel.writestr("fixture/_vendor/nested-2.dist-info/METADATA", b"vendored metadata")
    artifact = {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": common.sha256_file(path), "metadata_sha256": hashlib.sha256(metadata).hexdigest()}
    runtime.verify_artifact(path, artifact)
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(common.ToolchainFailure, match="hash/size mismatch"):
        runtime.verify_artifact(path, artifact)


def test_runtime_artifact_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("fixture-1.dist-info/METADATA", b"wrong metadata")
    artifact = {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": common.sha256_file(path), "metadata_sha256": "0" * 64}
    with pytest.raises(common.ToolchainFailure, match="metadata mismatch"):
        runtime.verify_artifact(path, artifact)


def test_v1_manifest_requires_runtime_fields(tmp_path: Path) -> None:
    value: dict[str, Any] = {field: None for field in common.TOOLCHAIN_FIELDS}
    value.update({"schema_version": 1, "finalized": True})
    value["manifest_digest"] = common.manifest_digest(value)
    path = tmp_path / "toolchain.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(common.ToolchainFailure, match="fields do not match schema"):
        in_env._load_manifest(path)


def _model_profile() -> tuple[dict[str, Any], dict[str, bytes]]:
    data = {name: f"fixture {name}".encode("ascii") for name in sorted(fetch_model.FILES)}
    artifacts: list[dict[str, Any]] = []
    for name, raw in data.items():
        artifact: dict[str, Any] = {"path": name, "bytes": len(raw)}
        if name == "model.safetensors":
            artifact["sha256"] = hashlib.sha256(raw).hexdigest()
        else:
            artifact["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw, usedforsecurity=False).hexdigest()
        artifacts.append(artifact)
    return {"schema_version": 1, "model_id": fetch_model.MODEL_ID,
            "model_revision": fetch_model.REVISION, "tokenizer_revision": fetch_model.REVISION,
            "assets_directory": ".illm/models/test-pinned", "runtime_dtype": "bf16", "max_context_tokens": 256,
            "minimum_gpu_memory_bytes": 6_442_450_944, "minimum_compute_capability": [8, 0],
            "tokenizer_kind": "hf-tokenizers-json", "artifacts": artifacts}, data


class _ModelResponse(io.BytesIO):
    def __init__(self, raw: bytes, filename: str) -> None:
        super().__init__(raw)
        self.url = f"https://huggingface.co/fixture/{filename}"
        self.headers = {"Content-Length": str(len(raw))}


class _ModelOpener:
    def __init__(self, data: dict[str, bytes]) -> None:
        self.data = data
        self.requests: list[str] = []

    def open(self, request: Any, *, timeout: int) -> _ModelResponse:
        assert timeout == 120
        self.requests.append(request.full_url)
        return _ModelResponse(self.data[request.full_url.rsplit("/", 1)[1]], request.full_url.rsplit("/", 1)[1])


def test_model_fetch_verifies_sha256_and_git_blob_identity_and_reuses_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, data = _model_profile()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    loaded = fetch_model.load_profile(tmp_path, path)
    opener = _ModelOpener(data)
    monkeypatch.setattr(fetch_model, "build_opener", lambda *args: opener)
    first = fetch_model.fetch_model(tmp_path, loaded, 1_000)
    assert first["downloaded_bytes"] == sum(map(len, data.values()))
    assert len(opener.requests) == 4
    receipt_path = tmp_path / loaded["assets_directory"] / "FETCH-COMPLETE.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = receipt.pop("receipt_digest")
    assert digest == common.sha256_bytes(common.canonical_bytes(receipt))
    second = fetch_model.fetch_model(tmp_path, loaded, 0)
    assert second["reused"] == 4
    assert second["downloaded_bytes"] == 0
    assert second["receipt_digest"] == first["receipt_digest"]
    assert len(opener.requests) == 4


def test_model_fetch_corrupt_cache_fails_without_network_or_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, data = _model_profile()
    destination = tmp_path / profile["assets_directory"]
    destination.mkdir(parents=True)
    artifact = profile["artifacts"][0]
    corrupt = b"x" * artifact["bytes"]
    (destination / artifact["path"]).write_bytes(corrupt)
    opener = _ModelOpener(data)
    monkeypatch.setattr(fetch_model, "build_opener", lambda *args: opener)
    with pytest.raises(common.ToolchainFailure, match="checksum differs"):
        fetch_model.fetch_model(tmp_path, profile, 1_000)
    assert opener.requests == []
    assert (destination / artifact["path"]).read_bytes() == corrupt
    assert not (destination / "FETCH-COMPLETE.json").exists()


def test_model_fetch_budget_fails_before_creating_cache(tmp_path: Path) -> None:
    profile, _ = _model_profile()
    with pytest.raises(common.ToolchainFailure, match="byte budget exceeded"):
        fetch_model.fetch_model(tmp_path, profile, 0)
    assert not (tmp_path / ".illm").exists()


def test_model_fetch_hash_failure_removes_partial_and_never_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile, data = _model_profile()
    first = profile["artifacts"][0]
    data[first["path"]] = b"x" * first["bytes"]
    monkeypatch.setattr(fetch_model, "build_opener", lambda *args: _ModelOpener(data))
    with pytest.raises(common.ToolchainFailure, match="checksum differs"):
        fetch_model.fetch_model(tmp_path, profile, 1_000)
    destination = tmp_path / profile["assets_directory"]
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("url", [
    "http://huggingface.co/model", "https://huggingface.co.evil.example/model",
    "https://user:password@huggingface.co/model", "https://localhost/model",
    "https://huggingface.co:444/model", "https://huggingface.co/model#fragment",
])
def test_model_fetch_rejects_unapproved_redirect_before_contact(url: str) -> None:
    with pytest.raises(common.ToolchainFailure, match="approved HTTPS artifact hosts"):
        fetch_model.ModelRedirectHandler().redirect_request(None, None, 302, "redirect", {}, url)


@pytest.mark.parametrize("change", ["revision", "path", "duplicate", "directory", "hash", "size"])
def test_model_profile_rejects_mutable_or_escaping_authority(tmp_path: Path, change: str) -> None:
    profile, _ = _model_profile()
    if change == "revision":
        profile["model_revision"] = "main"
    elif change == "path":
        profile["artifacts"][0]["path"] = "../config.json"
    elif change == "duplicate":
        profile["artifacts"][1] = profile["artifacts"][0]
    elif change == "directory":
        profile["assets_directory"] = "../escape"
    elif change == "hash":
        profile["artifacts"][0]["git_blob_sha1"] = "0" * 39
    else:
        profile["artifacts"][0]["bytes"] = True
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(common.ToolchainFailure):
        fetch_model.load_profile(tmp_path, path)


def test_model_profile_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(common.ToolchainFailure, match="duplicate JSON field"):
        fetch_model.load_profile(tmp_path, path)


def test_runtime_fetch_rejects_untrusted_redirect_before_contact() -> None:
    with pytest.raises(common.ToolchainFailure, match="untrusted host"):
        fetch_runtime.RuntimeRedirectHandler().redirect_request(None, None, 302, "redirect", {}, "https://untrusted.example/wheel.whl")


def test_runtime_alias_selects_os_without_silently_changing_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert runtime.load_runtime_lock(ROOT, "v01-cpu")["profile_id"] == "py312-linux-x86_64-cpu"
    assert runtime.load_runtime_lock(ROOT, "v01-cuda")["profile_id"] == "py312-linux-x86_64-cu126"
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert runtime.load_runtime_lock(ROOT, "v01-cpu")["profile_id"] == "py312-win-amd64-cpu"
    with pytest.raises(common.ToolchainFailure, match="unknown runtime profile"):
        runtime.load_runtime_lock(ROOT, "v01-cuda")


def test_model_fixture_config_and_runtime_authority_are_source_bound_but_outputs_are_not(tmp_path: Path) -> None:
    before = common.project_digest(tmp_path)
    for directory in ("fixtures", "configs", "runtime"):
        target = tmp_path / directory / "identity.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"version": 1}', encoding="utf-8")
        after = common.project_digest(tmp_path)
        assert after != before
        before = after
    output = tmp_path / "out" / "runtime.json"
    output.parent.mkdir()
    output.write_text('{"result": "changing"}', encoding="utf-8")
    assert common.project_digest(tmp_path) == before


def test_offline_child_environment_ignores_user_package_and_hub_network_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://untrusted.example/simple")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("PYTHONUSERBASE", "/untrusted/packages")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    environment = runtime.isolated_environment()
    assert "PIP_EXTRA_INDEX_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTHONUSERBASE" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"

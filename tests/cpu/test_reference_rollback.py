from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
from typing import Any
import zipfile

import pytest

from illm.canonical import sha256_json
from illm.domain.errors import IllmError
from illm.reference_rollback import PHASES, PREVIOUS_COMMIT, PREVIOUS_TAG, _verify_commands, portable_rollback_inventory, verify_quarantined_validation, verify_reference_rollback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import rollback_v01


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "a/../../escape", "a\\escape", "a/./file", "a//file", "NUL", "folder./file"])
def test_rollback_archive_rejects_unsafe_names_before_creating_target(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(name, b"unsafe")
    if "\\" in name:
        # Python's Windows ZIP writer normalizes separators. Restore the raw
        # attacker-controlled header name in both ZIP directory records.
        archive.write_bytes(archive.read_bytes().replace(name.replace("\\", "/").encode("ascii"), name.encode("ascii")))
    target = tmp_path / "new-target"
    with pytest.raises(IllmError, match="unsafe"):
        rollback_v01.extract_archive_safely(archive, target)
    assert not target.exists()
    assert not (tmp_path / "escape").exists()


def test_rollback_archive_rejects_symlinks_and_case_aliases(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(entry, "../outside")
    with pytest.raises(IllmError, match="nonregular"):
        rollback_v01.extract_archive_safely(archive, tmp_path / "link-target")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("same", b"one")
        handle.writestr("SAME", b"two")
    with pytest.raises(IllmError, match="duplicate"):
        rollback_v01.extract_archive_safely(archive, tmp_path / "case-target")


def test_rollback_archive_rejects_file_directory_alias_and_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "alias.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("node", b"one")
        handle.writestr("node/child", b"two")
    with pytest.raises(IllmError, match="aliases"):
        rollback_v01.extract_archive_safely(archive, tmp_path / "alias-target")
    monkeypatch.setattr(rollback_v01, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(IllmError, match="byte budget"):
        rollback_v01.extract_archive_safely(archive, tmp_path / "budget-target")


def test_rollback_archive_extracts_bytes_without_reusing_existing_target(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sub/file.txt", b"known bytes\n")
    target = tmp_path / "extracted"
    record = rollback_v01.extract_archive_safely(archive, target)
    assert record["file_count"] == 1
    assert (target / "sub" / "file.txt").read_bytes() == b"known bytes\n"
    with pytest.raises(IllmError, match="new target"):
        rollback_v01.extract_archive_safely(archive, target)


def test_rollback_inventory_excludes_only_regenerable_local_work(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "actual.txt").write_bytes(b"actual")
    (tmp_path / "local-only").mkdir()
    (tmp_path / "local-only" / "never-publish-weights.bin").write_bytes(b"not portable")
    (tmp_path / "INCOMPLETE").write_text("not sealed", encoding="ascii")
    assert set(portable_rollback_inventory(tmp_path)) == {"logs/actual.txt"}


def test_rollback_reader_never_treats_missing_execution_as_success(tmp_path: Path) -> None:
    with pytest.raises(IllmError, match="incomplete"):
        verify_reference_rollback(tmp_path)
    report: dict[str, Any] = {"schema_version": 1, "status": "pass", "previous_tag": "v0.0.0", "previous_commit": "invented"}
    report["report_digest"] = sha256_json(report)
    (tmp_path / "rollback-report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "COMPLETE").write_text(report["report_digest"] + "\n", encoding="ascii")
    with pytest.raises(IllmError, match="immutable V0.0"):
        verify_reference_rollback(tmp_path)


def test_quarantined_validation_requires_actual_linked_cases_not_pass_label(tmp_path: Path) -> None:
    report: dict[str, Any] = {"schema_version": 1, "status": "pass", "fixed_input_count": 20}
    report["report_digest"] = sha256_json(report)
    (tmp_path / "validation-report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(IllmError, match="unreadable"):
        verify_quarantined_validation(tmp_path)


def test_rollback_output_cannot_replace_user_checkout_or_existing_evidence(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve", encoding="ascii")
    arguments = ["--output", str(tmp_path), "--v01-capture", str(tmp_path),
                 "--v01-toolchain-manifest", str(tmp_path / "manifest.json"), "--validation-directory", str(tmp_path)]
    with pytest.raises(IllmError, match="new directory"):
        rollback_v01.run_rollback(arguments)
    assert sentinel.read_text(encoding="ascii") == "preserve"


def test_rollback_semantic_reader_rejects_a_forged_bootstrap_command() -> None:
    phases: list[dict[str, Any]] = [{"phase": name, "argv": []} for name in PHASES]
    phases[0]["argv"] = ["git", "rev-parse", f"refs/tags/{PREVIOUS_TAG}", f"refs/tags/{PREVIOUS_TAG}^{{commit}}"]
    phases[1]["argv"] = ["git", "archive", "--format=zip", "--output", "/work/out/r/local-only/v0.0.0.zip", PREVIOUS_COMMIT]
    phases[2]["argv"] = ["python", "-c", "print('fabricated pass')"]
    with pytest.raises(IllmError, match="minimal old source"):
        _verify_commands(phases, {"project_root": "/work/out/r/local-only/v00"})


def test_quarantine_copy_is_independent_and_keeps_original_writable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = source / "fact.json"
    original.write_bytes(b'{"fact": true}\n')
    mode_before = original.stat().st_mode
    target = tmp_path / "quarantine"
    rollback_v01._copy_tree(source, target)
    assert rollback_v01._readonly(target / "fact.json")
    assert original.stat().st_mode == mode_before
    assert original.read_bytes() == (target / "fact.json").read_bytes()
    assert not original.samefile(target / "fact.json")

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from _toolchain_common import bootstrap_digest, lock_digest, project_digest
from illm.canonical import canonical_json_bytes, sha256_file, sha256_json
from illm.release import (
    verify_release_bundle,
    verify_release_manifest,
    write_verified_release_state,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and freeze the V0.0 release before tagging")
    parser.add_argument("--release-dir", type=Path, default=Path("release/v0.0.0"))
    args = parser.parse_args(argv)
    root = args.release_dir.resolve()
    expected = Path(__file__).resolve().parents[1] / "release" / "v0.0.0"
    if root != expected.resolve():
        parser.error("promotion must target this checkout's release/v0.0.0")
    manifest_path = root / "manifest.json"
    project_root = Path(__file__).resolve().parents[1]
    source_identity = {
        "bootstrap_digest": bootstrap_digest(project_root),
        "lock_digest": lock_digest(project_root),
        "project_root_digest": project_digest(project_root),
    }
    checked = verify_release_bundle(manifest_path, expected_source_identity=source_identity)
    if checked["qualification_tier"] != "functionally_qualified":
        raise SystemExit("V0.0 promotion requires all functional profiles")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spec_status"] = "released"
    manifest["release_stage"] = "released"
    manifest["manifest_digest"] = sha256_json({
        key: value for key, value in manifest.items() if key != "manifest_digest"
    })
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    state = verify_release_manifest(manifest_path)
    write_verified_release_state(state, root / "qualification" / "release-state.json")
    sums = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "SHA256SUMS.json"
    }
    (root / "SHA256SUMS.json").write_bytes(canonical_json_bytes(sums) + b"\n")
    verified = verify_release_bundle(manifest_path, expected_source_identity=source_identity)
    print(json.dumps(verified, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

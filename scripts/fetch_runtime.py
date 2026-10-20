"""Explicit online population of the hash-addressed runtime artifact cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from _runtime_artifacts import ARTIFACT_HOSTS, artifact_path, filesystem_path, load_runtime_lock, verify_artifact
from _toolchain_common import ToolchainFailure, sha256_file


def _check_download_url(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in ARTIFACT_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in {None, 443} or parsed.fragment):
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime download redirected to an untrusted host")


class RuntimeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Request | None:
        _check_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_artifacts(project_root: Path, lock: dict[str, Any], cache_root: Path, max_download_bytes: int) -> dict[str, Any]:
    cache_root = cache_root.resolve()
    missing: list[dict[str, Any]] = []
    reused = 0
    for artifact in lock["artifacts"]:
        destination = filesystem_path(artifact_path(cache_root, artifact))
        if destination.exists():
            verify_artifact(destination, artifact)
            reused += 1
        else:
            missing.append(artifact)
    planned = sum(int(artifact["size_bytes"]) for artifact in missing)
    if planned > max_download_bytes:
        raise ToolchainFailure("INVALID_ARGUMENT", "runtime artifact byte budget exceeded", required_bytes=planned, budget=max_download_bytes)
    cache_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache_root).free < planned:
        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "insufficient free space for runtime artifact cache", required_bytes=planned)
    downloaded = 0
    vendor_reused = 0
    opener = build_opener(RuntimeRedirectHandler())
    for artifact in missing:
        destination = filesystem_path(artifact_path(cache_root, artifact))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".fetch-{uuid.uuid4().hex}.partial")
        vendor = project_root / "vendor" / "wheels" / str(artifact["filename"])
        try:
            if vendor.is_file() and sha256_file(vendor) == artifact["sha256"]:
                shutil.copyfile(vendor, temporary)
                vendor_reused += 1
            else:
                print(json.dumps({"event": "runtime.fetch.started", "filename": artifact["filename"], "size_bytes": artifact["size_bytes"]}), flush=True)
                _check_download_url(str(artifact["url"]))
                request = Request(str(artifact["url"]), headers={"User-Agent": "illm-runtime-fetch/1"})
                with opener.open(request, timeout=120) as response:
                    _check_download_url(response.url)
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != artifact["size_bytes"]:
                        raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime download content length mismatch")
                    count = 0
                    with temporary.open("xb") as handle:
                        while chunk := response.read(1024 * 1024):
                            count += len(chunk)
                            if count > artifact["size_bytes"]:
                                raise ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime download exceeded locked artifact size")
                            handle.write(chunk)
                    downloaded += count
            verify_artifact(temporary, artifact)
            temporary.replace(destination)
            print(json.dumps({"event": "runtime.fetch.verified", "filename": artifact["filename"], "sha256": artifact["sha256"]}), flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    return {"runtime_profile_id": lock["profile_id"], "runtime_lock_digest": lock["lock_digest"], "cache_root": str(cache_root), "artifacts": len(lock["artifacts"]), "reused": reused, "vendor_reused": vendor_reused, "downloaded_bytes": downloaded, "status": "verified"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--max-download-bytes", type=int, default=4_500_000_000)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        lock = load_runtime_lock(root, args.runtime_profile)
        cache_root = (args.artifact_cache or root / ".illm" / "cache" / "runtime").resolve()
        if args.max_download_bytes < 0:
            raise ToolchainFailure("INVALID_ARGUMENT", "max download bytes must be nonnegative")
        result = fetch_artifacts(root, lock, cache_root, args.max_download_bytes)
        print(json.dumps(result, sort_keys=True))
        return 0
    except ToolchainFailure as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 64 if exc.code == "INVALID_ARGUMENT" else 70
    except Exception as exc:
        failure = ToolchainFailure("TOOLCHAIN_MISMATCH", "runtime artifact fetch failed", exception_type=type(exc).__name__, reason=str(exc))
        print(json.dumps(failure.as_dict(), sort_keys=True), file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicitly fetch the pinned V0.1 model; model loading never accesses a network."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence, cast
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from _toolchain_common import ToolchainFailure, atomic_write_json, canonical_bytes, is_within, sha256_bytes


MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
MODEL_HOSTS = {
    "huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co",
}
PROFILE_FIELDS = {
    "schema_version", "model_id", "model_revision", "tokenizer_revision", "assets_directory",
    "runtime_dtype", "max_context_tokens", "minimum_gpu_memory_bytes", "minimum_compute_capability",
    "tokenizer_kind", "artifacts",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ToolchainFailure("INVALID_ARGUMENT", "model profile contains a duplicate JSON field", field=key)
        value[key] = item
    return value


def _check_download_url(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname not in MODEL_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in {None, 443} or parsed.fragment):
        raise ToolchainFailure("MODEL_ARTIFACT_MISSING", "model download host is outside the approved HTTPS artifact hosts")


class ModelRedirectHandler(HTTPRedirectHandler):
    """Check the destination before following a redirect, not after contacting it."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Request | None:
        _check_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load_profile(project_root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ToolchainFailure("INVALID_ARGUMENT", "model profile is unreadable") from exc
    if (not isinstance(value, dict) or set(value) != PROFILE_FIELDS
            or type(value.get("schema_version")) is not int or value.get("schema_version") != 1
            or value.get("model_id") != MODEL_ID or value.get("model_revision") != REVISION
            or value.get("tokenizer_revision") != REVISION):
        raise ToolchainFailure("INVALID_ARGUMENT", "only the immutable V0.1 Qwen2.5-0.5B profile is supported")
    if (value.get("runtime_dtype") not in {"bf16", "fp16", "fp32"}
            or value.get("tokenizer_kind") != "hf-tokenizers-json"
            or type(value.get("max_context_tokens")) is not int or value["max_context_tokens"] <= 0
            or type(value.get("minimum_gpu_memory_bytes")) is not int or value["minimum_gpu_memory_bytes"] < 0
            or not isinstance(value.get("minimum_compute_capability"), list)
            or len(value["minimum_compute_capability"]) != 2
            or any(type(item) is not int or item < 0 for item in value["minimum_compute_capability"])):
        raise ToolchainFailure("INVALID_ARGUMENT", "model profile runtime and device metadata are invalid")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(FILES):
        raise ToolchainFailure("INVALID_ARGUMENT", "model profile must contain the exact four pinned artifacts")
    names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) not in (
            {"path", "bytes", "sha256"}, {"path", "bytes", "git_blob_sha1"},
        ):
            raise ToolchainFailure("INVALID_ARGUMENT", "model artifact identity fields are invalid")
        name = artifact["path"]
        if not isinstance(name, str) or name not in FILES or name in names:
            raise ToolchainFailure("INVALID_ARGUMENT", "model artifact paths must be unique approved basenames")
        if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
            raise ToolchainFailure("INVALID_ARGUMENT", "model artifact byte size must be positive")
        key, length = ("sha256", 64) if "sha256" in artifact else ("git_blob_sha1", 40)
        if not isinstance(artifact[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", artifact[key]):
            raise ToolchainFailure("INVALID_ARGUMENT", "model artifact hash is invalid")
        names.add(name)
    directory = value.get("assets_directory")
    if not isinstance(directory, str) or not directory:
        raise ToolchainFailure("INVALID_ARGUMENT", "model artifact directory must be a nonempty path")
    destination = (project_root / directory).resolve()
    managed = (project_root / ".illm" / "models").resolve()
    if destination == managed or not is_within(destination, managed):
        raise ToolchainFailure("INVALID_ARGUMENT", "model artifact directory must remain inside .illm/models")
    return cast(dict[str, Any], value)


def verify_model_artifact(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        raise ToolchainFailure("INVALID_WEIGHTS", "model artifact is missing or has an unexpected byte size", artifact=artifact["path"])
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(artifact["bytes"]).encode("ascii") + b"\0", usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    expected_key = "sha256" if "sha256" in artifact else "git_blob_sha1"
    actual = sha256.hexdigest() if expected_key == "sha256" else blob.hexdigest()
    if actual != artifact[expected_key]:
        raise ToolchainFailure("INVALID_WEIGHTS", "model artifact checksum differs from the immutable profile", artifact=artifact["path"])
    return {"path": artifact["path"], "bytes": artifact["bytes"], "sha256": sha256.hexdigest(), "authority_kind": expected_key,
            "authority_digest": artifact[expected_key]}


def fetch_model(project_root: Path, profile: dict[str, Any], max_download_bytes: int) -> dict[str, Any]:
    destination = (project_root / profile["assets_directory"]).resolve()
    managed = (project_root / ".illm" / "models").resolve()
    if destination == managed or not is_within(destination, managed):
        raise ToolchainFailure("INVALID_ARGUMENT", "model artifact directory must remain inside .illm/models")
    missing: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for artifact in profile["artifacts"]:
        path = destination / artifact["path"]
        if path.exists():
            verified.append(verify_model_artifact(path, artifact))
        else:
            missing.append(artifact)
    planned = sum(artifact["bytes"] for artifact in missing)
    if type(max_download_bytes) is not int or max_download_bytes < 0 or planned > max_download_bytes:
        raise ToolchainFailure("INVALID_ARGUMENT", "model artifact byte budget exceeded", required_bytes=planned, budget=max_download_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < planned:
        raise ToolchainFailure("MODEL_ARTIFACT_MISSING", "insufficient disk space for pinned model artifacts", required_bytes=planned)
    opener = build_opener(ModelRedirectHandler())
    downloaded_bytes = 0
    for artifact in missing:
        path = destination / artifact["path"]
        temporary = destination / f".fetch-{uuid.uuid4().hex}.partial"
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{quote(artifact['path'])}"
        _check_download_url(url)
        print(json.dumps({"event": "model.fetch.started", "path": artifact["path"], "bytes": artifact["bytes"]}), flush=True)
        try:
            request = Request(url, headers={"User-Agent": "illm-model-fetch/1"})
            with opener.open(request, timeout=120) as response:
                _check_download_url(response.url)
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != artifact["bytes"]:
                    raise ToolchainFailure("INVALID_WEIGHTS", "model response length differs from pinned artifact", artifact=artifact["path"])
                count = 0
                next_progress = 32 * 1024 * 1024
                with temporary.open("xb") as handle:
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > artifact["bytes"]:
                            raise ToolchainFailure("INVALID_WEIGHTS", "model response exceeded pinned byte size", artifact=artifact["path"])
                        handle.write(chunk)
                        if count >= next_progress:
                            print(json.dumps({"event": "model.fetch.progress", "path": artifact["path"], "received_bytes": count,
                                              "total_bytes": artifact["bytes"]}), flush=True)
                            next_progress += 32 * 1024 * 1024
            record = verify_model_artifact(temporary, artifact)
            if path.exists():
                # Another explicit fetch may have completed concurrently; never replace it blindly.
                verify_model_artifact(path, artifact)
            else:
                temporary.replace(path)
            verified.append(record)
            downloaded_bytes += count
            print(json.dumps({"event": "model.fetch.verified", "path": artifact["path"], "sha256": record["sha256"]}), flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    receipt: dict[str, Any] = {"schema_version": 1, "kind": "illm-model-fetch-receipt", "model_id": MODEL_ID,
                               "model_revision": REVISION, "profile_digest": sha256_bytes(canonical_bytes(profile)),
                               "artifacts": sorted(verified, key=lambda artifact: artifact["path"])}
    receipt["receipt_digest"] = sha256_bytes(canonical_bytes(receipt))
    atomic_write_json(destination / "FETCH-COMPLETE.json", receipt)
    return {"status": "verified", "assets_directory": str(destination), "model_revision": REVISION,
            "artifacts": len(verified), "reused": len(verified) - len(missing), "downloaded_bytes": downloaded_bytes,
            "receipt_digest": receipt["receipt_digest"]}


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", type=Path, default=root / "configs" / "models" / "v0_1.yaml")
    parser.add_argument("--max-download-bytes", type=int, default=1_100_000_000)
    args = parser.parse_args(argv)
    try:
        result = fetch_model(root, load_profile(root, args.model_profile), args.max_download_bytes)
        print(json.dumps(result, sort_keys=True))
        return 0
    except ToolchainFailure as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True), file=sys.stderr)
        return 64 if exc.code == "INVALID_ARGUMENT" else 70
    except HTTPError as exc:
        failure = ToolchainFailure("MODEL_ARTIFACT_MISSING", "model server rejected the explicit download; no automatic bypass or retry", status=exc.code)
        print(json.dumps(failure.as_dict(), sort_keys=True), file=sys.stderr)
        return 70
    except Exception as exc:
        failure = ToolchainFailure("MODEL_ARTIFACT_MISSING", "explicit model download failed", exception_type=type(exc).__name__)
        print(json.dumps(failure.as_dict(), sort_keys=True), file=sys.stderr)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())

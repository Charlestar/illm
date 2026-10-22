from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, NoReturn, Sequence, cast


REPORT_KIND = "illm-policy-capture-portable-development-v1"
EVIDENCE_STAGE = "DEVELOPMENT_ONLY"
QUALIFICATION_STATUS = "unqualified"
POLICY_CATALOG = "illm-stateful-v02-policy-v1"
PROTOCOL_SHA256 = "c9fd588a72e0e5b039c43585ef949346db5c98f71aa198172ebd4aaa96e719d5"
CAPTURE_FILES = frozenset({"events.jsonl", "checksums.json", "capture-manifest.json", "COMPLETE"})
REPORT_FILE = "report.json"
COMPLETE_FILE = "COMPLETE"
INCOMPLETE_FILE = "INCOMPLETE"
_DIGEST_CHARS = frozenset("0123456789abcdef")

_CHILD_PROGRAM = r'''
import builtins
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
forbidden = ("torch", "transformers", "tokenizers", "safetensors")
attempted = []
network_events = []
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level == 0 and root in forbidden:
        attempted.append(name)
        raise RuntimeError("tensor/runtime package import is forbidden")
    return original_import(name, globals, locals, fromlist, level)

def audit(event, args):
    if event.startswith("socket."):
        network_events.append(event)
        raise RuntimeError("network access is forbidden")

builtins.__import__ = guarded_import
sys.addaudithook(audit)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

reader_root = Path(sys.argv[1]).resolve()
capture_root = Path(sys.argv[2])
sys.path.insert(0, str(reader_root))
from illm.capture.policy_streaming import PolicyStreamingCaptureBundle

bundle = PolicyStreamingCaptureBundle.open(capture_root)
story = bundle.request_story(bundle.manifest["request_id"])
module_files = {}
for module_name, module in sorted(sys.modules.items()):
    if module_name != "illm" and not module_name.startswith("illm."):
        continue
    filename = getattr(module, "__file__", None)
    if not isinstance(filename, str):
        raise RuntimeError("loaded illm module has no plain source file: " + module_name)
    source = Path(filename).resolve()
    if source.suffix != ".py":
        raise RuntimeError("loaded illm dependency is not a .py source: " + module_name)
    try:
        relative = source.relative_to(reader_root).as_posix()
    except ValueError as exc:
        raise RuntimeError("loaded illm dependency escaped reader root: " + module_name) from exc
    module_files[module_name] = relative

loaded_forbidden = sorted(
    name for name in sys.modules
    if name.split(".", 1)[0] in forbidden
)
if attempted or loaded_forbidden or network_events:
    raise RuntimeError("portable reader touched a forbidden dependency or network")

manifest = bundle.manifest
binding = manifest["execution_binding"]
result = {
    "schema_version": manifest["schema_version"],
    "manifest_digest": manifest["manifest_digest"],
    "event_catalog": manifest["event_catalog"],
    "capabilities": manifest["capabilities"],
    "protocol_sha256": manifest["protocol_sha256"],
    "numerical_policy_digest": manifest["numerical_policy_digest"],
    "memory_plan_digest": manifest["memory_plan_digest"],
    "execution_binding_digest": manifest["execution_binding_digest"],
    "implementation_source_digest": binding["implementation_source_digest"],
    "environment_digest": binding["environment_digest"],
    "toolchain": binding["toolchain"],
    "observation_profile": manifest["observation_profile"],
    "request_id": manifest["request_id"],
    "event_count": manifest["event_count"],
    "logical_terminal_outcome": manifest["logical_terminal_outcome"],
    "terminal_outcome": manifest["terminal_outcome"],
    "story_capability": story["capability"],
    "module_files": module_files,
    "forbidden_import_attempts": attempted,
    "forbidden_loaded_modules": loaded_forbidden,
    "network_events": network_events,
}
print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
'''


class PortableError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise PortableError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise PortableError("report contains non-canonical JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PortableError(f"artifact is unreadable: {path}") from exc
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _DIGEST_CHARS for character in value)
    )


def _decode_json(text: str, *, artifact: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON field in {artifact}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(f"non-finite JSON constant in {artifact}: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except PortableError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise PortableError(f"malformed JSON: {artifact}") from exc


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PortableError(f"JSON artifact is unreadable: {path}") from exc
    return _decode_json(text, artifact=str(path))


def _is_linklike(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & 0x400)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _check_existing_ancestors(path: Path) -> None:
    for candidate in (path, *path.parents):
        if _is_linklike(candidate):
            _fail(f"path contains a symbolic link or reparse point: {candidate}")


def _plain_directory(path: Path, *, name: str) -> Path:
    absolute = _absolute(path)
    _check_existing_ancestors(absolute)
    if not absolute.is_dir() or _is_linklike(absolute):
        _fail(f"{name} must be an ordinary directory: {absolute}")
    return absolute


def _new_output(path: Path) -> Path:
    absolute = _absolute(path)
    _check_existing_ancestors(absolute)
    if absolute.exists() or _is_linklike(absolute):
        _fail("portable proof output must not already exist")
    try:
        absolute.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PortableError("portable proof output could not be created") from exc
    _check_existing_ancestors(absolute)
    return absolute


def _safe_relative(value: str, *, suffix: str | None = None) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        _fail("portable package contains an invalid relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        _fail("portable package relative path escapes its root")
    if suffix is not None and relative.suffix != suffix:
        _fail("portable reader dependency is not a Python source file")
    return relative


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PortableError(f"artifact could not be written exclusively: {path}") from exc


def _copy_plain(source: Path, destination: Path) -> str:
    _check_existing_ancestors(source)
    if _is_linklike(source) or not source.is_file():
        _fail(f"copy source is not an ordinary file: {source}")
    try:
        data = source.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PortableError(f"copy preparation failed: {source}") from exc
    _check_existing_ancestors(destination.parent)
    _write_exclusive(destination, data)
    return _sha256_bytes(data)


def _run_child(*, python: Path, reader_root: Path, capture: Path, cwd: Path) -> dict[str, Any]:
    argv = [
        str(python),
        "-I",
        "-S",
        "-c",
        _CHILD_PROGRAM,
        str(reader_root),
        str(capture),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise PortableError("isolated portable-reader child could not complete") from exc
    record: dict[str, Any] = {
        "argv": argv,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode("utf-8")),
        "result": None,
    }
    if completed.returncode != 0:
        raise PortableError(
            "isolated portable-reader child failed: "
            f"rc={completed.returncode}, stderr={completed.stderr.strip()!r}"
        )
    result = _decode_json(completed.stdout, artifact="isolated-child-stdout")
    if not isinstance(result, dict):
        _fail("isolated portable-reader child did not return an object")
    _validate_child_result(cast(dict[str, Any], result))
    record["result"] = result
    return record


_CHILD_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_digest",
        "event_catalog",
        "capabilities",
        "protocol_sha256",
        "numerical_policy_digest",
        "memory_plan_digest",
        "execution_binding_digest",
        "implementation_source_digest",
        "environment_digest",
        "toolchain",
        "observation_profile",
        "request_id",
        "event_count",
        "logical_terminal_outcome",
        "terminal_outcome",
        "story_capability",
        "module_files",
        "forbidden_import_attempts",
        "forbidden_loaded_modules",
        "network_events",
    }
)


def _validate_digest_map(value: object, *, name: str, exact: frozenset[str] | None = None) -> dict[str, str]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        _fail(f"{name} must be a digest object")
    if exact is not None and set(value) != exact:
        _fail(f"{name} inventory differs from the closed contract")
    result: dict[str, str] = {}
    folded: set[str] = set()
    for key, digest in value.items():
        relative = _safe_relative(key)
        folded_name = relative.as_posix().casefold()
        if folded_name in folded:
            _fail(f"{name} contains a case-colliding path")
        folded.add(folded_name)
        if not _is_digest(digest):
            _fail(f"{name} contains an invalid SHA-256 digest")
        result[relative.as_posix()] = cast(str, digest)
    if not result:
        _fail(f"{name} must not be empty")
    return result


def _validate_module_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        _fail("reader module map must be a nonempty object")
    result: dict[str, str] = {}
    for module_name, relative_name in value.items():
        if (
            type(module_name) is not str
            or (module_name != "illm" and not module_name.startswith("illm."))
            or type(relative_name) is not str
        ):
            _fail("reader module map contains an invalid module identity")
        relative = _safe_relative(relative_name, suffix=".py")
        if not relative.parts or relative.parts[0] != "illm":
            _fail("reader module source is outside the illm package")
        result[module_name] = relative.as_posix()
    return result


def _validate_toolchain(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("embedded toolchain binding must be an object")
    binding = value.get("binding")
    if binding == "unbound-development":
        if set(value) != {"binding", "manifest_digest"} or value["manifest_digest"] is not None:
            _fail("unbound embedded toolchain binding is malformed")
    elif binding == "source-bound-manifest":
        if set(value) != {"binding", "manifest_digest", "environment_id", "lock_digest"}:
            _fail("source-bound embedded toolchain binding fields are malformed")
        if (
            not _is_digest(value["manifest_digest"])
            or type(value["environment_id"]) is not str
            or not value["environment_id"]
            or not _is_digest(value["lock_digest"])
        ):
            _fail("source-bound embedded toolchain binding is malformed")
    else:
        _fail("embedded toolchain binding mode is unsupported")
    return cast(dict[str, Any], value)


def _validate_child_result(value: dict[str, Any]) -> None:
    if set(value) != _CHILD_RESULT_FIELDS:
        _fail("isolated child result fields differ from the closed contract")
    if type(value["schema_version"]) is not int or value["schema_version"] != 3:
        _fail("isolated child did not validate schema3")
    for field in (
        "manifest_digest",
        "protocol_sha256",
        "numerical_policy_digest",
        "memory_plan_digest",
        "execution_binding_digest",
        "implementation_source_digest",
        "environment_digest",
    ):
        if not _is_digest(value[field]):
            _fail(f"isolated child returned an invalid digest: {field}")
    if value["protocol_sha256"] != PROTOCOL_SHA256:
        _fail("isolated child protocol identity is unsupported")
    if value["event_catalog"] != POLICY_CATALOG or value["capabilities"] != ["R0"]:
        _fail("isolated child catalog/capability is unsupported")
    if value["story_capability"] != "R0":
        _fail("isolated child request story exceeded or omitted R0")
    if value["observation_profile"] not in {"off", "metrics", "request", "ops"}:
        _fail("isolated child observation profile is unsupported")
    if type(value["request_id"]) is not str or not value["request_id"]:
        _fail("isolated child request identity is invalid")
    if type(value["event_count"]) is not int or value["event_count"] < 1:
        _fail("isolated child event count is invalid")
    if (
        value["logical_terminal_outcome"] not in {"FINISHED", "CANCELLED", "FAILED"}
        or value["terminal_outcome"] != value["logical_terminal_outcome"]
    ):
        _fail("isolated child terminal facts are invalid")
    _validate_toolchain(value["toolchain"])
    _validate_module_map(value["module_files"])
    for field in ("forbidden_import_attempts", "forbidden_loaded_modules", "network_events"):
        if value[field] != []:
            _fail(f"isolated child reported forbidden activity: {field}")


def _validate_command_record(value: object, *, name: str) -> dict[str, Any]:
    fields = {
        "argv",
        "cwd",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
        "result",
    }
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{name} command record fields are invalid")
    if (
        not isinstance(value["argv"], list)
        or len(value["argv"]) != 7
        or any(type(item) is not str for item in value["argv"])
        or type(value["cwd"]) is not str
        or not value["cwd"]
        or type(value["returncode"]) is not int
        or value["returncode"] != 0
        or type(value["stdout"]) is not str
        or type(value["stderr"]) is not str
    ):
        _fail(f"{name} command execution facts are invalid")
    argv = cast(list[str], value["argv"])
    if argv[1:4] != ["-I", "-S", "-c"] or not all(argv[index] for index in (0, 4, 5, 6)):
        _fail(f"{name} did not record the required isolated child command")
    if (
        value["stdout_sha256"] != _sha256_bytes(value["stdout"].encode("utf-8"))
        or value["stderr_sha256"] != _sha256_bytes(value["stderr"].encode("utf-8"))
    ):
        _fail(f"{name} command stream digest mismatch")
    parsed = _decode_json(value["stdout"], artifact=f"{name}.stdout")
    if not isinstance(parsed, dict) or parsed != value["result"]:
        _fail(f"{name} command stdout/result binding mismatch")
    _validate_child_result(cast(dict[str, Any], parsed))
    return cast(dict[str, Any], value)


def _tree_inventory(root: Path) -> tuple[set[str], set[str]]:
    root = _plain_directory(root, name="portable tree")
    files: set[str] = set()
    directories: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise PortableError(f"portable tree is unreadable: {directory}") from exc
        for child in children:
            if _is_linklike(child):
                _fail(f"portable tree contains a symbolic link or reparse point: {child}")
            relative = child.relative_to(root).as_posix()
            if child.is_dir():
                directories.add(relative)
                pending.append(child)
            elif child.is_file():
                files.add(relative)
            else:
                _fail(f"portable tree contains a non-file artifact: {child}")
    return files, directories


def _verify_hashed_tree(root: Path, expected: Mapping[str, str], *, name: str) -> None:
    normalized = _validate_digest_map(dict(expected), name=name)
    actual_files, actual_directories = _tree_inventory(root)
    expected_files = set(normalized)
    expected_directories: set[str] = set()
    for filename in expected_files:
        parent = PurePosixPath(filename).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_files != expected_files or actual_directories != expected_directories:
        _fail(f"{name} exact file/directory inventory mismatch")
    for filename, digest in normalized.items():
        path = root.joinpath(*PurePosixPath(filename).parts)
        if _sha256_file(path) != digest:
            _fail(f"{name} file digest mismatch: {filename}")


_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "report_kind",
        "status",
        "evidence_stage",
        "qualification_status",
        "evidence_scope",
        "limitations",
        "input_capture_path",
        "input_capture_manifest",
        "source_dependency_snapshot",
        "reader_modules",
        "reader_files",
        "capture_files",
        "commands",
    }
)
_MANIFEST_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_digest",
        "event_catalog",
        "capabilities",
        "protocol_sha256",
        "numerical_policy_digest",
        "memory_plan_digest",
        "execution_binding_digest",
        "implementation_source_digest",
        "environment_digest",
        "toolchain",
        "observation_profile",
        "request_id",
        "event_count",
        "logical_terminal_outcome",
        "terminal_outcome",
    }
)
_LIMITATIONS = [
    "DEVELOPMENT_ONLY; this report is not qualification or release evidence",
    "copied source hashes prove internal byte consistency, not independent source truth",
    "R0 metadata replay only; no model execution, R1 state, or numerical recomputation",
]


def _manifest_identity(result: Mapping[str, Any]) -> dict[str, Any]:
    return {field: result[field] for field in _MANIFEST_IDENTITY_FIELDS}


def _validate_manifest_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MANIFEST_IDENTITY_FIELDS:
        _fail("embedded input capture identity fields are invalid")
    synthetic = {
        **value,
        "story_capability": "R0",
        "module_files": {"illm": "illm/__init__.py"},
        "forbidden_import_attempts": [],
        "forbidden_loaded_modules": [],
        "network_events": [],
    }
    _validate_child_result(cast(dict[str, Any], synthetic))
    return cast(dict[str, Any], value)


def _validate_report(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _REPORT_FIELDS:
        _fail("portable report fields differ from the closed contract")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["report_kind"] != REPORT_KIND
        or value["status"] != "pass"
        or value["evidence_stage"] != EVIDENCE_STAGE
        or value["qualification_status"] != QUALIFICATION_STATUS
        or value["evidence_scope"]
        != "copied-minimal-reader-and-copied-schema3-r0-capture-internal-consistency"
        or value["limitations"] != _LIMITATIONS
        or type(value["input_capture_path"]) is not str
        or not value["input_capture_path"]
    ):
        _fail("portable report status/scope claims are invalid")
    manifest_identity = _validate_manifest_identity(value["input_capture_manifest"])

    snapshot = value["source_dependency_snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != {"scope", "files"}:
        _fail("portable source dependency snapshot fields are invalid")
    if snapshot["scope"] != "integrity-only-copied-current-reader-source;not-independent-source-truth":
        _fail("portable source dependency snapshot overclaims authority")
    source_files = _validate_digest_map(snapshot["files"], name="source_dependency_snapshot.files")
    reader_files = _validate_digest_map(value["reader_files"], name="reader_files")
    if source_files != reader_files:
        _fail("copied reader bytes differ from the recorded source dependency snapshot")
    capture_files = _validate_digest_map(value["capture_files"], name="capture_files", exact=CAPTURE_FILES)
    modules = _validate_module_map(value["reader_modules"])
    if set(modules.values()) != set(reader_files):
        _fail("reader file inventory is not exactly the observed import dependency closure")

    commands = value["commands"]
    if not isinstance(commands, dict) or set(commands) != {"source_validation", "copied_validation"}:
        _fail("portable report command fields are invalid")
    source_command = _validate_command_record(commands["source_validation"], name="source_validation")
    copied_command = _validate_command_record(commands["copied_validation"], name="copied_validation")
    source_result = cast(dict[str, Any], source_command["result"])
    copied_result = cast(dict[str, Any], copied_command["result"])
    if source_result != copied_result:
        _fail("source and copied reader results differ")
    if source_result["module_files"] != modules:
        _fail("report reader module map differs from observed imports")
    if _manifest_identity(source_result) != manifest_identity:
        _fail("report capture identity differs from isolated reader result")
    value["source_dependency_snapshot"] = {"scope": snapshot["scope"], "files": source_files}
    value["reader_files"] = reader_files
    value["capture_files"] = capture_files
    value["reader_modules"] = modules
    return cast(dict[str, Any], value)


def _copy_reader_closure(
    *,
    source_root: Path,
    reader_root: Path,
    modules: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    normalized_modules = _validate_module_map(dict(modules))
    relative_names = sorted(set(normalized_modules.values()))
    folded = [name.casefold() for name in relative_names]
    if len(folded) != len(set(folded)):
        _fail("reader dependency closure contains case-colliding source paths")
    source_hashes: dict[str, str] = {}
    copied_hashes: dict[str, str] = {}
    for relative_name in relative_names:
        relative = _safe_relative(relative_name, suffix=".py")
        source = source_root.joinpath(*relative.parts)
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise PortableError("reader dependency escaped repository src") from exc
        if resolved != source:
            _fail("reader dependency path is not a plain repository source path")
        destination = reader_root.joinpath(*relative.parts)
        source_digest = _sha256_file(source)
        copied_digest = _copy_plain(source, destination)
        if copied_digest != source_digest:
            _fail("copied reader dependency bytes changed during copy")
        source_hashes[relative.as_posix()] = source_digest
        copied_hashes[relative.as_posix()] = copied_digest
    return source_hashes, copied_hashes


def _copy_capture(source: Path, destination: Path) -> dict[str, str]:
    source = _plain_directory(source, name="input policy capture")
    source_files, source_directories = _tree_inventory(source)
    if source_files != set(CAPTURE_FILES) or source_directories:
        _fail("input policy capture inventory changed after isolated validation")
    result: dict[str, str] = {}
    for name in sorted(CAPTURE_FILES):
        result[name] = _copy_plain(source / name, destination / name)
    return result


def build_portable_proof(*, capture: Path, output: Path, python: Path) -> dict[str, Any]:
    script = _absolute(Path(__file__))
    repository = _plain_directory(script.parent.parent, name="repository root")
    source_root = _plain_directory(repository / "src", name="repository src")
    capture_root = _plain_directory(capture, name="input policy capture")
    output_absolute = _absolute(output)
    if output_absolute.is_relative_to(source_root):
        _fail("portable proof output cannot be created inside repository src")
    if output_absolute.is_relative_to(capture_root) or capture_root.is_relative_to(output_absolute):
        _fail("portable proof output and input capture cannot contain one another")
    output_root = _new_output(output_absolute)
    _write_exclusive(output_root / INCOMPLETE_FILE, b"portable_policy_capture_proof_not_complete\n")
    try:
        source_command = _run_child(
            python=python,
            reader_root=source_root,
            capture=capture_root,
            cwd=repository,
        )
        source_result = cast(dict[str, Any], source_command["result"])
        reader_root = output_root / "reader"
        capture_copy = output_root / "capture"
        reader_root.mkdir()
        capture_copy.mkdir()
        modules = _validate_module_map(source_result["module_files"])
        source_hashes, reader_hashes = _copy_reader_closure(
            source_root=source_root,
            reader_root=reader_root,
            modules=modules,
        )
        capture_hashes = _copy_capture(capture_root, capture_copy)
        copied_command = _run_child(
            python=python,
            reader_root=reader_root,
            capture=capture_copy,
            cwd=output_root,
        )
        copied_result = cast(dict[str, Any], copied_command["result"])
        if copied_result != source_result:
            _fail("copied minimal reader result differs from source reader result")

        report: dict[str, Any] = {
            "schema_version": 1,
            "report_kind": REPORT_KIND,
            "status": "pass",
            "evidence_stage": EVIDENCE_STAGE,
            "qualification_status": QUALIFICATION_STATUS,
            "evidence_scope": "copied-minimal-reader-and-copied-schema3-r0-capture-internal-consistency",
            "limitations": list(_LIMITATIONS),
            "input_capture_path": str(capture_root),
            "input_capture_manifest": _manifest_identity(source_result),
            "source_dependency_snapshot": {
                "scope": "integrity-only-copied-current-reader-source;not-independent-source-truth",
                "files": source_hashes,
            },
            "reader_modules": modules,
            "reader_files": reader_hashes,
            "capture_files": capture_hashes,
            "commands": {
                "source_validation": source_command,
                "copied_validation": copied_command,
            },
        }
        normalized_report = _validate_report(report)
        report_bytes = _canonical_bytes(normalized_report) + b"\n"
        _write_exclusive(output_root / REPORT_FILE, report_bytes)
        report_digest = _sha256_bytes(report_bytes)
        _write_exclusive(output_root / COMPLETE_FILE, (report_digest + "\n").encode("ascii"))
        (output_root / INCOMPLETE_FILE).unlink()
        return {
            "report_kind": REPORT_KIND,
            "status": "pass",
            "evidence_stage": EVIDENCE_STAGE,
            "qualification_status": QUALIFICATION_STATUS,
            "report_sha256": report_digest,
            "manifest_digest": source_result["manifest_digest"],
            "output": str(output_root),
        }
    except Exception:
        # INCOMPLETE deliberately remains beside any partial bytes for diagnosis.
        raise


def verify_existing_proof(*, output: Path, python: Path) -> dict[str, Any]:
    output_root = _plain_directory(output, name="portable proof output")
    try:
        children = list(output_root.iterdir())
    except OSError as exc:
        raise PortableError("portable proof output is unreadable") from exc
    expected_names = {"reader", "capture", REPORT_FILE, COMPLETE_FILE}
    if {child.name for child in children} != expected_names:
        _fail("portable proof top-level inventory is not exact or remains incomplete")
    for child in children:
        if _is_linklike(child):
            _fail("portable proof contains a symbolic link or reparse point")
    if not (output_root / "reader").is_dir() or not (output_root / "capture").is_dir():
        _fail("portable proof reader/capture roots are not ordinary directories")
    if not (output_root / REPORT_FILE).is_file() or not (output_root / COMPLETE_FILE).is_file():
        _fail("portable proof report/completion marker is missing")

    report = _validate_report(_read_json(output_root / REPORT_FILE))
    report_bytes = (output_root / REPORT_FILE).read_bytes()
    report_digest = _sha256_bytes(report_bytes)
    try:
        completion = (output_root / COMPLETE_FILE).read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise PortableError("portable proof completion marker is unreadable") from exc
    if completion != report_digest + "\n":
        _fail("portable proof completion marker does not bind report bytes")

    _verify_hashed_tree(output_root / "reader", report["reader_files"], name="reader_files")
    _verify_hashed_tree(output_root / "capture", report["capture_files"], name="capture_files")
    verification_command = _run_child(
        python=python,
        reader_root=output_root / "reader",
        capture=output_root / "capture",
        cwd=output_root,
    )
    verification_result = cast(dict[str, Any], verification_command["result"])
    recorded = cast(dict[str, Any], report["commands"])["copied_validation"]
    recorded_result = cast(dict[str, Any], cast(dict[str, Any], recorded)["result"])
    if verification_result != recorded_result:
        _fail("fresh copied-reader verification differs from the sealed report")
    if _manifest_identity(verification_result) != report["input_capture_manifest"]:
        _fail("fresh copied-reader capture identity differs from the sealed report")
    return {
        "report_kind": REPORT_KIND,
        "status": "pass",
        "evidence_stage": EVIDENCE_STAGE,
        "qualification_status": QUALIFICATION_STATUS,
        "report_sha256": report_digest,
        "manifest_digest": verification_result["manifest_digest"],
        "output": str(output_root),
        "source_capture_argument_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify a DEVELOPMENT_ONLY copied schema3 policy-capture reader proof",
    )
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_existing:
            result = verify_existing_proof(output=args.output, python=Path(sys.executable))
        else:
            result = build_portable_proof(
                capture=args.capture,
                output=args.output,
                python=Path(sys.executable),
            )
    except (PortableError, OSError, ValueError) as exc:
        print(f"portable policy capture check failed: {exc}", file=sys.stderr)
        return 1
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

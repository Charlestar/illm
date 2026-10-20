from __future__ import annotations

import ast
import json
from pathlib import Path

from illm.domain.errors import EXIT_CODE, ErrorCode
from illm.domain.events import EventKind


ROOT = Path(__file__).resolve().parents[2]


def test_registry_json_files_are_parseable_and_match_code() -> None:
    event_registry = json.loads((ROOT / "registries" / "events-v0.json").read_text(encoding="utf-8"))
    registered_events = {(item["code"], item["name"]) for item in event_registry["events"]}
    code_events = {(int(kind), kind.wire_name) for kind in EventKind}
    assert registered_events == code_events

    error_registry = json.loads((ROOT / "registries" / "errors-v0.json").read_text(encoding="utf-8"))
    registered_errors = {(item["code"], item["exit_code"]) for item in error_registry["errors"]}
    extension = json.loads((ROOT / "registries" / "errors-v1.json").read_text(encoding="utf-8"))
    assert extension["extends"] == "errors-v0.json"
    extra_errors = {(item["code"], item["exit_code"]) for item in extension["errors"]}
    assert not {code for code, _ in registered_errors} & {code for code, _ in extra_errors}
    registered_errors |= extra_errors
    streaming = json.loads((ROOT / "registries" / "errors-v2.json").read_text(encoding="utf-8"))
    assert streaming["extends"] == "errors-v1.json"
    streaming_errors = {(item["code"], item["exit_code"]) for item in streaming["errors"]}
    assert not {code for code, _ in registered_errors} & {code for code, _ in streaming_errors}
    registered_errors |= streaming_errors
    code_errors = {(code.value, EXIT_CODE[code]) for code in ErrorCode}
    assert registered_errors == code_errors

    metrics = json.loads((ROOT / "registries" / "metrics-v0.json").read_text(encoding="utf-8"))
    ids = {item["id"] for item in metrics["metrics"]}
    assert "illm.toy.scheduler.completed_steps.v1" in ids
    assert "illm.replay.state_digest.mismatch_count.v1" in ids


def test_all_v0_json_schemas_are_parseable() -> None:
    schemas = sorted((ROOT / "schemas").glob("*-v0.schema.json"))
    assert {path.name for path in schemas} == {
        "capture-manifest-v0.schema.json",
        "domain-event-v0.schema.json",
        "release-manifest-v0.schema.json",
        "run-manifest-v0.schema.json",
        "toolchain-manifest-v0.schema.json",
    }
    for schema in schemas:
        value = json.loads(schema.read_text(encoding="utf-8"))
        assert value["$schema"].endswith("2020-12/schema")
        assert value["type"] == "object"


def test_domain_and_scheduler_do_not_depend_on_outer_layers() -> None:
    forbidden = {"illm.cli", "illm.capture", "illm.observability", "illm.release", "illm.replay"}
    for root in (ROOT / "src" / "illm" / "domain", ROOT / "src" / "illm" / "scheduler"):
        for source in root.glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
            assert not any(any(name == item or name.startswith(item + ".") for item in forbidden) for name in imports)

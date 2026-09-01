from __future__ import annotations

import ast
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _annotation_issues(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for argument in arguments:
            if argument.arg in {"self", "cls"}:
                continue
            if argument.annotation is None:
                issues.append(f"{path}:{node.lineno}: public function {node.name} argument {argument.arg} lacks a type")
        if node.returns is None:
            issues.append(f"{path}:{node.lineno}: public function {node.name} lacks a return type")
    return issues


def check(project_root: Path) -> list[str]:
    issues: list[str] = []
    python_files = sorted((project_root / "src" / "illm").rglob("*.py"))
    for path in python_files:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{path}: is not UTF-8")
            continue
        if "\t" in text:
            issues.append(f"{path}: contains a tab")
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.rstrip() != line:
                issues.append(f"{path}:{line_number}: trailing whitespace")
        try:
            issues.extend(_annotation_issues(path))
        except SyntaxError as exc:
            issues.append(f"{path}:{exc.lineno}: syntax error: {exc.msg}")
    return issues


def _run_mypy(project_root: Path) -> tuple[dict[str, object], list[str]]:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "src/illm",
        "scripts",
        "tests/cpu",
    ]
    process = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output_lines = [
        line
        for line in (process.stdout + process.stderr).splitlines()
        if line.strip()
    ]
    try:
        version = importlib.metadata.version("mypy")
    except importlib.metadata.PackageNotFoundError:
        version = None
    result: dict[str, object] = {
        "tool": "mypy",
        "version": version,
        "returncode": process.returncode,
        "status": "pass" if process.returncode == 0 and version else "fail",
        "output": output_lines,
    }
    issues = [] if result["status"] == "pass" else [f"mypy: {line}" for line in output_lines]
    if version is None:
        issues.append("mypy: locked type checker is not installed")
    return result, issues


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V0.0 static quality and annotation gate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    issues = check(root)
    type_check, type_issues = _run_mypy(root)
    issues.extend(type_issues)
    result = {
        "schema_version": 0,
        "checked_python_files": len(list((root / "src" / "illm").rglob("*.py"))),
        "issues": issues,
        "type_check": type_check,
        "status": "pass" if not issues else "fail",
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(encoded)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

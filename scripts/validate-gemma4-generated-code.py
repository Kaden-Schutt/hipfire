#!/usr/bin/env python3
"""Audit code artifacts emitted by the Gemma4 E-series quality suite."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path


FENCE_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\n(?P<body>.*?)(?P<close>```|\Z)", re.S)


class StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.scripts: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if tag == "script":
            self._script = []

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self.scripts.append("".join(self._script))
            self._script = None


def extract_code(text: str, language: str) -> tuple[str, bool]:
    matches = [match for match in FENCE_RE.finditer(text) if match.group("language").lower() == language]
    if not matches:
        return "", False
    match = max(matches, key=lambda item: len(item.group("body")))
    return match.group("body"), match.group("close") == "```"


def validate_python(source: str, closed: bool) -> dict:
    errors: list[str] = []
    try:
        ast.parse(source)
        compile(source, "<generated>", "exec")
    except SyntaxError as exc:
        errors.append(f"syntax: {exc.msg} at line {exc.lineno}")
    required = {
        "pygame_import": bool(re.search(r"^\s*(?:import|from)\s+pygame\b", source, re.M)),
        "main_guard": 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source,
        "event_loop": "pygame.event.get" in source,
        "display": "pygame.display" in source,
    }
    if not closed:
        errors.append("unclosed fenced code block (likely truncated)")
    return {
        "language": "python",
        "closed_fence": closed,
        "syntax_ok": not any(error.startswith("syntax:") for error in errors),
        "required": required,
        "runtime_smoke": "skipped: pygame is not installed in the validation environment",
        "errors": errors,
    }


def validate_html(source: str, closed: bool, extracted_dir: Path, stem: str) -> dict:
    errors: list[str] = []
    parser = StructureParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception as exc:  # HTMLParser is permissive; retain unexpected failures.
        errors.append(f"html parser: {exc}")
    required = {
        "doctype": source.lstrip().lower().startswith("<!doctype html>"),
        "html": "html" in parser.tags,
        "head": "head" in parser.tags,
        "body": "body" in parser.tags,
        "style": "style" in parser.tags,
        "script": "script" in parser.tags,
        "local_storage": "localStorage" in source,
        "viewport": "viewport" in source,
    }
    js_checks: list[dict] = []
    node = shutil.which("node")
    for index, script in enumerate(parser.scripts):
        js_path = extracted_dir / f"{stem}-{index}.js"
        js_path.write_text(script)
        if node:
            checked = subprocess.run(
                [node, "--check", str(js_path)], capture_output=True, text=True, timeout=30
            )
            js_checks.append({
                "path": str(js_path),
                "ok": checked.returncode == 0,
                "stderr": checked.stderr.strip(),
            })
            if checked.returncode != 0:
                errors.append(f"JavaScript syntax failed: {js_path.name}")
    if not closed:
        errors.append("unclosed fenced code block (likely truncated)")
    return {
        "language": "html",
        "closed_fence": closed,
        "required": required,
        "javascript": js_checks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    extracted_dir = args.report.parent / "extracted-code"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for model_dir in sorted(args.root.glob("*-longdecode")):
        output_dir = model_dir / "outputs"
        model_report: dict[str, dict] = {}
        for task, language in (
            ("snake-python", "python"),
            ("tetris-python", "python"),
            ("responsive-webpage", "html"),
        ):
            path = output_dir / f"{task}.md"
            if not path.exists():
                model_report[task] = {
                    "source": str(path),
                    "skipped": "artifact not present in this filtered run",
                    "errors": [],
                }
                continue
            text = path.read_text()
            source, closed = extract_code(text, language)
            suffix = ".py" if language == "python" else ".html"
            extracted = extracted_dir / f"{model_dir.name}-{task}{suffix}"
            extracted.write_text(source)
            result = (
                validate_python(source, closed)
                if language == "python"
                else validate_html(source, closed, extracted_dir, f"{model_dir.name}-{task}")
            )
            result["source"] = str(path)
            result["extracted"] = str(extracted)
            result["bytes"] = len(source.encode())
            model_report[task] = result
        report[model_dir.name] = model_report

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if any(
        result["errors"]
        for model in report.values()
        for result in model.values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())

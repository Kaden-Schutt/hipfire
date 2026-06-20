#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "env-vars.md"
RUST_MODULE = ROOT / "crates" / "hipfire-runtime" / "src" / "env_docs.rs"
EXCLUDE_PREFIXES = (
    ROOT / "third_party",
    ROOT / "target",
)

COMMENT_PREFIXES = (
    "//!",
    "///",
    "//",
    "#",
)

ENV_READ_RE = re.compile(
    r"""
    (?:
        std::env::var(?:_os)?\(\s*["']([A-Z][A-Z0-9_]+)["']\s*\)|
        env::var(?:_os)?\(\s*["']([A-Z][A-Z0-9_]+)["']\s*\)|
        std::env::set_var\(\s*["']([A-Z][A-Z0-9_]+)["']\s*,|
        env::set_var\(\s*["']([A-Z][A-Z0-9_]+)["']\s*,|
        process\.env\.([A-Z][A-Z0-9_]+)|
        process\.env\[[\"']([A-Z][A-Z0-9_]+)[\"']\]
    )
    """,
    re.VERBOSE,
)

COMMENT_RE = re.compile(r"^\s*(?://!|///|//|#)\s?(.*)$")

USAGE_HINT_RE = [
    (
        re.compile(r"as_deref\(\)\s*==\s*Some\(\"1\"\)", re.IGNORECASE),
        "Enabled when set to 1.",
    ),
    (
        re.compile(r"as_deref\(\)\s*!=\s*Some\(\"0\"\)", re.IGNORECASE),
        "Enabled by default; set to 0 to disable.",
    ),
    (
        re.compile(r"as_deref\(\)\s*==\s*Some\(\"0\"\)", re.IGNORECASE),
        "Disabled when set to 0.",
    ),
    (
        re.compile(r"as_deref\(\)\.unwrap_or\(\"([^\"]+)\"\)", re.IGNORECASE),
        "Defaults to {} when unset.",
    ),
    (
        re.compile(r"unwrap_or_else\(\|.*\|\s*(\"[^\"]+\")", re.IGNORECASE),
        "Defaults to {} when unset.",
    ),
    (
        re.compile(
            r"parse::<\w+>\(\)\.unwrap_or\(([^\)]+)\)",
            re.IGNORECASE,
        ),
        "Parsed with fallback default {}.",
    ),
    (
        re.compile(r"match\s+std::env::var", re.IGNORECASE),
        "Selects behavior from recognized values.",
    ),
    (
        re.compile(r"\bparse::<(?:u\d+|usize|bool|f\d+|String|u8|u16|u32|u64)\>", re.IGNORECASE),
        "Parsed into numeric or typed runtime setting.",
    ),
    (
        re.compile(r"\bSome\((\"[^\"]+\"|'[^']+')\)|\bOk\((\"[^\"]+\"|'[^']+')\)", re.IGNORECASE),
        "Environment toggle value controls runtime behavior.",
    ),
    (
        re.compile(r"as_deref\(\)\.is_ok\(\)", re.IGNORECASE),
        "Optional toggle; presence may enable feature behavior.",
    ),
    (
        re.compile(r"Some\(\"true\"\)|Some\(\"1\"\)|Some\(\"yes\"\)|Some\(\"on\"\)", re.IGNORECASE),
        "Boolean-style toggle env var.",
    ),
]

UNHELPFUL_DESCRIPTIONS = {
    "Behavioral use is defined in source; add a dedicated env-doc entry.",
    "`HIPFIRE_`",
    "HIPFIRE_*",
    "HIPFIRE",
}


def escape_rust_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class EnvUsage:
    name: str
    source: str
    line: int


@dataclass
class EnvDoc:
    name: str
    description: str
    source: str


def tracked_sources() -> List[Path]:
    cmd = ["git", "ls-files"]
    proc = subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)
    files = []
    for line in proc.stdout.splitlines():
        p = Path(line)
        if p.suffix not in {".rs", ".ts"}:
            continue
        if any(str(p).startswith(prefix.as_posix() + "/") for prefix in EXCLUDE_PREFIXES):
            continue
        files.append(ROOT / p)
    return files


def normalize_comment_line(line: str) -> str:
    line = line.strip()
    for prefix in COMMENT_PREFIXES:
        if line.startswith(prefix):
            line = line[len(prefix) :].lstrip()
            break
    line = line.strip()
    if line.startswith("`") and "`" in line[1:]:
        line = line[1:]
        end = line.find("`")
        if end > 0:
            line = line[end + 1 :].lstrip(" :—-")
    return re.sub(r"\s+", " ", line).strip()


def strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and ((text.startswith("\"") and text.endswith("\"")) or (text.startswith("'") and text.endswith("'"))):
        return text[1:-1]
    return text


def normalize_description(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(" .")).replace("`", "\"")


def is_helpful_description(value: str) -> bool:
    text = normalize_description(value)
    if not text:
        return False
    if text in UNHELPFUL_DESCRIPTIONS:
        return False
    low = text.lower()
    if "behavioral use is defined" in low:
        return False
    if len(text) < 18:
        return False
    if re.fullmatch(r"[A-Z0-9\-_\s]+", text):
        return False
    if len(text.split()) <= 2:
        return False
    return True


def score_comment(text: str, var: str) -> int:
    lower = text.lower()
    score = len(text)
    if var in text:
        score += 40
    if any(tok in lower for tok in ("default", "defaults", "set", "enable", "disable", "opt", "file", "path", "directory", "mode", "timeout", "rate", "batch", "budget", "token", "kv", "draft", "loop", "spec", "ddtree", "gpu", "dump", "log")):
        score += 20
    return score


def extract_comment_descriptions(path: Path, line_idx: int, lines: List[str]) -> List[str]:
    candidates: List[str] = []
    idx = line_idx - 1
    if idx < 0 or idx >= len(lines):
        return candidates

    raw_line = lines[idx]
    comment_idx = raw_line.find("//")
    if comment_idx >= 0:
        c = normalize_comment_line(raw_line[comment_idx + 2 :])
        if c:
            candidates.append(c)

    # Backward pass: contiguous doc comments directly above the usage.
    for back in range(1, 18):
        cur = idx - back
        if cur < 0:
            break
        txt = lines[cur].strip()
        if not txt:
            break
        m = COMMENT_RE.match(txt)
        if not m:
            break
        clean = normalize_comment_line(m.group(1))
        if clean:
            candidates.append(clean)

    # Forward pass: keep a small tail in case inline docs are written after code.
    for fwd in range(1, 8):
        cur = idx + fwd
        if cur >= len(lines):
            break
        txt = lines[cur].strip()
        if not txt:
            break
        m = COMMENT_RE.match(txt)
        if not m:
            continue
        clean = normalize_comment_line(m.group(1))
        if clean:
            candidates.append(clean)

    return list(dict.fromkeys(normalize_description(c) for c in candidates if c))


def infer_from_expression(line: str, var: str, line_ctx: List[str], line_idx: int) -> Optional[str]:
    line_text = line.strip()
    if line_text:
        for pattern, template in USAGE_HINT_RE:
            match = pattern.search(line_text)
            if match:
                groups = match.groups()
                if groups:
                    cleaned = [strip_wrapping_quotes(v) for v in groups if v is not None]
                    if cleaned:
                        return normalize_description(template.format(*cleaned))
                return normalize_description(template)

        lower = line_text.lower()
        if "parse().ok()" in lower or "parse::<" in lower:
            parse_type = "value"
            if "parse::<usize>()" in lower:
                parse_type = "usize/integer"
            elif "parse::<u32>()" in lower:
                parse_type = "u32"
            elif "parse::<u16>()" in lower:
                parse_type = "u16"
            elif "parse::<bool>()" in lower:
                parse_type = "boolean"
            elif "parse::<f32>()" in lower or "parse::<f64>()" in lower:
                parse_type = "floating-point"
            return f"Parsed as {parse_type} configuration from environment value."

        if "match std::env::var" in lower or "match env::var" in lower:
            return f"Reads `{var}` and branches runtime behavior by recognized values."

        if ".set_var(" in lower and var in lower:
            return f"Sets `{var}` for runtime or child process configuration."

    window_start = max(0, line_idx - 4)
    window_end = min(len(line_ctx), line_idx + 4)
    local_text = " ".join(line_ctx[window_start:window_end]).lower()
    if "parse::<" in local_text and "unwrap_or" in local_text:
        return f"Parses `{var}` with fallback defaults."
    if "match" in local_text and "as_deref" in local_text:
        return f"Interprets `{var}` from environment to select behavior."
    if "set_var" in local_text:
        return f"Used to configure runtime execution by explicitly setting `{var}`."

    return None


def infer_name_from_var(var: str, source: str) -> str:
    label = var.removeprefix("HIPFIRE_")
    label = label.replace("_", " ").replace("-", " ").lower()
    if "ddtree" in label:
        label = label.replace("ddtree", "DDTree")
    label = label.replace("mtp", "MTP").replace("kv", "KV").replace("q8", "Q8").replace("q4", "Q4")
    label = re.sub(r"\s+", " ", label).strip()
    if label:
        return f"Runtime variable controlling {label} in hipfire."
    return f"Runtime control variable `{var}` defined in {Path(source).name}."


def collect_existing_descs() -> Dict[str, str]:
    if not DOC.exists():
        return {}
    existing: Dict[str, str] = {}
    for line in DOC.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^\| `([A-Z0-9_]+)` \| (.*?) \|", line)
        if m:
            existing[m.group(1)] = m.group(2).strip()
    return existing


def infer_default(
    existing_desc: Optional[str],
    var: str,
    cands: List[str],
    line: str,
    lines: List[str],
    line_idx: int,
    usage_source: str,
) -> str:
    if existing_desc and is_helpful_description(existing_desc):
        return normalize_description(existing_desc)

    ranked: List[Tuple[int, str]] = []
    for c in cands:
        if not c:
            continue
        if is_helpful_description(c):
            ranked.append((score_comment(c, var), c))
    if ranked:
        ranked.sort(reverse=True, key=lambda v: v[0])
        return ranked[0][1]

    inferred = infer_from_expression(line, var, lines, line_idx)
    if inferred:
        return normalize_description(inferred)

    return infer_name_from_var(var, usage_source)


def extract_env_usages(path: Path) -> List[EnvUsage]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[EnvUsage] = []
    for i, line in enumerate(text, start=1):
        for match in ENV_READ_RE.finditer(line):
            for grp in match.groups():
                if grp:
                    out.append(EnvUsage(grp, f"{path}:{i}", i))
                    break
    return out


def collect_env_data() -> Tuple[Dict[str, EnvDoc], Set[str]]:
    usages: Dict[str, List[EnvUsage]] = {}
    raw_lines: Dict[Path, List[str]] = {}
    for path in tracked_sources():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        raw_lines[path] = lines
        for usage in extract_env_usages(path):
            usages.setdefault(usage.name, []).append(usage)

    existing_descs = collect_existing_descs()
    docs: Dict[str, EnvDoc] = {}
    for name, usage_list in usages.items():
        best_desc = ""
        best_source = usage_list[0].source
        best_usage_line = None
        best_doc: Optional[EnvDoc] = None
        for usage in usage_list:
            path_str, line_no_str = usage.source.rsplit(":", 1)
            path = Path(path_str)
            lines = raw_lines[path]
            line_no = int(line_no_str)
            line = lines[line_no - 1]
            cands = extract_comment_descriptions(path, line_no, lines)
            usage_desc = infer_default(
                existing_descs.get(name),
                name,
                cands,
                line,
                lines,
                line_no - 1,
                usage.source,
            )
            if not best_doc or is_helpful_description(usage_desc):
                best_desc = usage_desc
                best_source = usage.source
                best_usage_line = usage
                best_doc = EnvDoc(name=name, description=usage_desc, source=usage.source)
            if is_helpful_description(usage_desc) and is_helpful_description(best_desc) and len(usage_desc) > len(best_desc):
                best_doc = EnvDoc(name=name, description=usage_desc, source=usage.source)
                best_desc = usage_desc
                best_source = usage.source
                best_usage_line = usage

        docs[name] = best_doc if best_doc else EnvDoc(name=name, description=infer_name_from_var(name, best_source), source=best_source)

    return docs, set(usages)


def render_markdown(env_docs: Dict[str, EnvDoc]) -> str:
    total = len(env_docs)
    hipfire = [k for k in env_docs if k.startswith("HIPFIRE_")]
    non = [k for k in env_docs if not k.startswith("HIPFIRE_")]
    lines = [
        "# hipfire environment variables — canonical reference",
        "",
        "Generated automatically from source and inline comments by `scripts/gen-env-docs.py`.",
        "",
        "| Variable | Description | Defined at |",
        "|---|---|---|",
    ]
    for name in sorted(env_docs):
        doc = env_docs[name]
        desc = doc.description.replace("|", "\\|")
        lines.append(f"| `{doc.name}` | {desc} | `{doc.source}` |")
    lines.extend(
        [
            "",
            f"- Total env vars: **{total}**",
            f"- `HIPFIRE_*` vars: **{len(hipfire)}**",
            f"- non-`HIPFIRE_*` vars: **{len(non)}**",
        ]
    )
    return "\n".join(lines) + "\n"


def rust_identifier(env: str) -> str:
    return "ENV_" + env


def render_rust_module(env_docs: Dict[str, EnvDoc]) -> str:
    lines = [
        "#![allow(dead_code)]",
        "",
        "// SPDX-License-Identifier: Apache-2.0",
        "//",
        "// Generated automatically from source env usage by scripts/gen-env-docs.py.",
        "// Do not hand-edit. Re-run `./scripts/regen-env-vars-doc.sh`.",
        "",
        "/// Canonical environment-variable documentation registry.",
        "///",
        "/// Each entry is sourced from inline comments or generated defaults.",
        "pub struct EnvVarDoc {",
        "    pub name: &'static str,",
        "    pub description: &'static str,",
        "    pub source: &'static str,",
        "}",
        "",
        "impl EnvVarDoc {",
        "    pub const fn name(&self) -> &'static str {",
        "        self.name",
        "    }",
        "}",
        "",
    ]

    for name in sorted(env_docs):
        doc = env_docs[name]
        lines.extend(
            [
                f"/// `{doc.name}` — {doc.description}",
                f"pub const {rust_identifier(name)}: EnvVarDoc = EnvVarDoc {{",
                f'    name: "{doc.name}",',
                f'    description: "{escape_rust_str(doc.description)}",',
                f'    source: "{doc.source}",',
                "};",
                "",
            ]
        )

    lines.extend(
        [
            "/// All documented environment variables in deterministic order.",
            "pub const ALL_ENV_VARS: &[EnvVarDoc] = &[",
        ]
    )
    for name in sorted(env_docs):
        lines.append(f"    {rust_identifier(name)},")
    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    env_docs, _ = collect_env_data()
    DOC.write_text(render_markdown(env_docs), encoding="utf-8")
    RUST_MODULE.write_text(render_rust_module(env_docs), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

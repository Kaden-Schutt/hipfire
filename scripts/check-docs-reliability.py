#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Documentation reliability snapshot checker.

Validates a git tree (explicit target ref, or the current index via --staged)
against documentation reliability rules. Stdlib only.

CLI (exactly one mode):
  --target-ref REF --base-ref REF
  --staged --base-ref REF
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]

# Active rewrite / routing surfaces (greenfield active layer).
ACTIVE_MARKDOWN = frozenset(
    {
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
        "CLAUDE.md",
        "docs/INDEX.md",
        "docs/VALIDATION.md",
        "docs/GETTING_STARTED.md",
        "docs/CLI.md",
        "docs/MODELS.md",
        "docs/CONFIG.md",
        "docs/SERVE.md",
        "docs/CHAT.md",
        "docs/env-vars.md",
        "docs/CONTAINER.md",
        "docs/NIXOS.md",
        "docs/ARCHITECTURE.md",
        "docs/architecture-ids.md",
        "docs/QUANTIZATION.md",
        "docs/QUANTIZE.md",
        "docs/multi-gpu.md",
        "docs/BENCHMARKS.md",
        "docs/REDLINE.md",
        "docs/methodology/perf-benchmarking.md",
        "docs/methodology/bench-suite.md",
        "docs/methodology/perf-arch-discipline.md",
        "docs/methodology/arch-port-validation.md",
        "docs/methodology/kernel-atlas.md",
        "docs/methodology/kernel-atlas-architecture.md",
    }
)

ENV_REFERENCE_DOCS = ("AGENTS.md", "README.md", "CONTRIBUTING.md")
ENV_CANONICAL = "docs/env-vars.md"
ADMISSIONS_PATH = "docs/admissions.yml"
INDEX_PATH = "docs/INDEX.md"
CHECKPOINT_PREFIX = "docs/perf-checkpoints/"

# Accept current authority spelling and the alias pair.
OWNERSHIP_HEADER_VARIANTS = (
    ("Concern", "Canonical owner", "State", "Notes"),
    ("Concern", "Canonical owner", "Truth state", "Scope/limits"),
)

TRUTH_STATES = frozenset(
    {
        "shipped / ref-pinned",
        "shipped / integration-ref-pinned",
        "branch-implemented",
        "branch-only",
        "measured",
        "planned",
        "historical",
        "blocked",
        "superseded",
        "rejected",
        "unknown",
        "planned / historical",
        "branch-implemented / transitional",
        "superseded / rejected",
    }
)

BRANCHISH_TRUTH = frozenset(
    {
        "branch-implemented",
        "branch-only",
        "branch-implemented / transitional",
    }
)

HISTORICALISH_TRUTH = frozenset(
    {
        "historical",
        "measured",
        "planned",
        "planned / historical",
        "superseded",
        "rejected",
        "superseded / rejected",
    }
)

COHERENCE_ACCEPTANCE_RE = re.compile(
    r"(?is)coherence-gate"
    r".{0,120}?"
    r"(?:\b(?:acceptance|accepted|mandatory|required|canonical|promotion|merge bar|current gate)\b"
    r"|\bas\s+acceptance\b"
    r"|\bvalid acceptance\b)"
    r"|"
    r"(?:\b(?:acceptance|accepted|mandatory|required|canonical|promotion|merge bar)\b"
    r".{0,120}?"
    r"coherence-gate)"
)

# Markdown links and images: ![alt](tgt) or [text](tgt)
MD_LINK_OR_IMAGE_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
HIPFIRE_ENV_RE = re.compile(r"\bHIPFIRE_[A-Z0-9_]+\b")
MD_OWNER_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FULL_SHA_RE = re.compile(r"\b[0-9a-fA-F]{40}\b")

# Concrete path-like tokens that may appear inside code spans (including commands).
PATH_TOKEN_RE = re.compile(
    r"(?:"
    r"(?:\./)?(?:docs|scripts|crates|tests|cli|kernels|\.agents|\.github|\.githooks)/[^\s`]+"
    r"|(?:\./)?(?:AGENTS|README|CONTRIBUTING|CLAUDE|PRIOR-ART)\.md"
    r"|(?:\./)?NOTICE"
    r"|(?:\./)?\.skills/[^\s`]+"
    r"|(?:\./)?\.skills"
    r")"
)

EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "ftp://")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    path: str
    message: str

    def format(self) -> str:
        return f"{self.code}: {self.path}: {self.message}"


class GitSnapshot:
    """Read-only view of a git treeish or the current index."""

    def __init__(self, root: Path, treeish: str) -> None:
        self.root = root
        self.treeish = treeish
        self._paths: set[str] | None = None
        self._cache: dict[str, bytes | None] = {}

    def _run(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            check=False,
        )

    @classmethod
    def from_ref(cls, root: Path, ref: str) -> "GitSnapshot":
        # Require a commit object so tree/blob SHAs cannot masquerade as refs.
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"error: invalid commit ref {ref!r}: {proc.stderr.strip()}"
            )
        return cls(root, proc.stdout.strip())

    @classmethod
    def from_index(cls, root: Path) -> "GitSnapshot":
        proc = subprocess.run(
            ["git", "write-tree"],
            cwd=root,
            capture_output=True,
            check=False,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"error: git write-tree failed: {proc.stderr.strip()}")
        return cls(root, proc.stdout.strip())

    def list_paths(self) -> set[str]:
        if self._paths is None:
            proc = self._run("ls-tree", "-r", "--name-only", self.treeish)
            if proc.returncode != 0:
                raise SystemExit(
                    f"error: git ls-tree failed for {self.treeish}: "
                    f"{proc.stderr.decode('utf-8', 'replace').strip()}"
                )
            self._paths = {
                line.decode("utf-8", "replace")
                for line in proc.stdout.splitlines()
                if line.strip()
            }
        return self._paths

    def exists(self, path: str) -> bool:
        return path in self.list_paths()

    def read_bytes(self, path: str) -> bytes | None:
        if path in self._cache:
            return self._cache[path]
        if path not in self.list_paths():
            self._cache[path] = None
            return None
        proc = self._run("show", f"{self.treeish}:{path}")
        if proc.returncode != 0:
            self._cache[path] = None
            return None
        self._cache[path] = proc.stdout
        return proc.stdout

    def read_text(self, path: str) -> str | None:
        data = self.read_bytes(path)
        if data is None:
            return None
        return data.decode("utf-8", "replace")

    def paths_with_prefix(self, prefix: str) -> list[str]:
        return sorted(p for p in self.list_paths() if p.startswith(prefix))


def github_anchor(heading: str) -> str:
    """GitHub-like slug for Markdown headings."""
    text = heading.strip().lower()
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = unquote(text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text


def collect_anchors(md: str) -> set[str]:
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for match in HEADING_RE.finditer(md):
        base = github_anchor(match.group(2))
        if not base:
            continue
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def is_external_link(target: str) -> bool:
    t = target.strip()
    if t.startswith("#"):
        return False
    if t.startswith(EXTERNAL_SCHEMES):
        return True
    parsed = urlparse(t)
    return bool(parsed.scheme and parsed.scheme not in ("", "file"))


def split_link_target(raw: str) -> tuple[str, str | None]:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target:
        path_part, maybe_title = target.split(" ", 1)
        if maybe_title[:1] in "\"'(":
            target = path_part
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")].strip()
    if "#" in target:
        path, frag = target.split("#", 1)
        return path, frag
    return target, None


def strip_leading_dot_slash(token: str) -> str:
    """Remove only a literal leading './' prefix; keep '.agents/...' intact."""
    if token.startswith("./"):
        return token[2:]
    return token


def strip_shell_punctuation(token: str) -> str:
    """Trim common shell/markdown punctuation wrapped around path tokens."""
    return token.strip().strip("`\"'").rstrip(".,;:)!?]>")


def extract_path_tokens(span: str) -> list[str]:
    """Extract concrete repo-root path tokens from a code span (may contain spaces)."""
    found: list[str] = []
    for match in PATH_TOKEN_RE.finditer(span):
        token = strip_shell_punctuation(match.group(0))
        # Drop trailing anchor fragments for existence checks.
        token = token.split("#", 1)[0]
        token = strip_shell_punctuation(token)
        if token:
            found.append(token)
    return found


def normalize_rel(base_file: str, rel: str) -> tuple[str | None, str | None]:
    """Resolve rel against base_file.

    Returns (resolved_posix, error). error is set when the link escapes the repo.
    """
    if rel == "":
        return base_file, None
    base_dir = Path(base_file).parent
    parts: list[str] = []
    for part in base_dir.as_posix().split("/"):
        if part in ("", "."):
            continue
        parts.append(part)
    for part in Path(rel).as_posix().split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None, "link escapes repository root"
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts), None


def parse_markdown_table(section: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.rstrip() for ln in section.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return [], []

    def cells(line: str) -> list[str]:
        raw = line.strip().strip("|")
        return [c.strip() for c in raw.split("|")]

    headers = cells(lines[0])
    rows: list[list[str]] = []
    for line in lines[2:]:
        row = cells(line)
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        rows.append(row[: len(headers)])
    return headers, rows


def extract_section(md: str, heading_substr: str) -> str | None:
    matches = list(HEADING_RE.finditer(md))
    for i, match in enumerate(matches):
        if heading_substr.lower() in match.group(2).lower():
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
            return md[start:end]
    return None


def normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", h).strip()


def strip_md(cell: str) -> str:
    return re.sub(r"\*+", "", cell).strip()


def truth_key(cell: str) -> str:
    return strip_md(cell).lower()


def is_blocked_truth(key: str) -> bool:
    return key == "blocked" or key.startswith("blocked")


def owner_is_blocked_marker(owner: str) -> bool:
    text = strip_md(owner)
    if not text or text in {"—", "-", "–", "−"}:
        return True
    if text.upper() == "BLOCKED":
        return True
    return False


def extract_owner_link_targets(owner_cell: str) -> list[str]:
    return [m.group(1).strip() for m in MD_OWNER_LINK_RE.finditer(owner_cell)]


def parse_index_metadata(text: str) -> dict[str, str]:
    """Pull Working branch / Audited source ref / Comparison base values."""
    meta: dict[str, str] = {}
    patterns = {
        "branch": re.compile(
            r"(?im)^\|\s*Working branch\s*\|\s*(.*?)\s*\|"
            r"|^\*\*Working branch\*\*\s*[:=]\s*(.+)$"
            r"|^Working branch\s*[:=]\s*(.+)$"
        ),
        "source": re.compile(
            r"(?im)^\|\s*Audited source ref\s*\|\s*(.*?)\s*\|"
            r"|^\*\*Audited (?:source )?ref\*\*\s*[:=]\s*(.+)$"
            r"|^Audited (?:source )?ref\s*[:=]\s*(.+)$"
        ),
        "base": re.compile(
            r"(?im)^\|\s*Comparison base\s*\|\s*(.*?)\s*\|"
            r"|^\*\*Comparison base\*\*\s*[:=]\s*(.+)$"
            r"|^Comparison base\s*[:=]\s*(.+)$"
        ),
    }
    for key, cre in patterns.items():
        m = cre.search(text)
        if not m:
            continue
        val = next((g for g in m.groups() if g is not None), "").strip()
        meta[key] = val
    return meta


def nonempty_branch_name(value: str) -> str | None:
    ticks = re.findall(r"`([^`]+)`", value)
    for t in ticks:
        t = t.strip()
        if t and not FULL_SHA_RE.fullmatch(t):
            return t.split("@", 1)[0].strip() or None
    cleaned = value.strip().strip("`")
    if not cleaned:
        return None
    token = cleaned.split()[0].strip("`")
    if FULL_SHA_RE.fullmatch(token):
        return None
    return token or None


def require_full_sha(value: str) -> str | None:
    m = FULL_SHA_RE.search(value or "")
    return m.group(0).lower() if m else None


def classify_path_from_index(
    path: str,
    page_states: dict[str, str],
    collection_states: dict[str, str],
) -> str | None:
    if path in page_states:
        return page_states[path]
    if path.startswith("docs/"):
        rest = path[len("docs/") :]
        if "/" in rest:
            coll = rest.split("/", 1)[0] + "/"
            return collection_states.get(coll)
    return None


def is_historicalish_state(state: str | None) -> bool:
    if state is None:
        return False
    key = state.lower()
    if key in HISTORICALISH_TRUTH:
        return True
    if "historical" in key or key == "measured" or key.startswith("planned"):
        return True
    if "superseded" in key or "rejected" in key:
        return True
    return False


def is_allowed_lifecycle_label(state: str) -> bool:
    key = truth_key(state)
    if not key:
        return False
    # unknown is lack-of-authority — fail closed in classification cells.
    if key == "unknown":
        return False
    if is_blocked_truth(key):
        return True
    if key in {s.lower() for s in TRUTH_STATES if s.lower() != "unknown"}:
        return True
    return False


def is_branchish_label(state: str) -> bool:
    key = truth_key(state)
    return key in BRANCHISH_TRUTH or "branch-implemented" in key or key == "branch-only"


def is_executable_docs_skills_target(path: str) -> bool:
    """True for executable skill entrypoints under docs/skills/."""
    if not path.startswith("docs/skills/"):
        return False
    name = Path(path).name.lower()
    if name in {"skill.md", "skill.json"}:
        return True
    if name.endswith((".sh", ".py", ".bash", ".zsh", ".js", ".ts", ".mjs")):
        return True
    return False


def owner_link_is_external(raw: str) -> bool:
    target, _frag = split_link_target(raw)
    if not target:
        return False
    return is_external_link(target)


class Checker:
    def __init__(self, target: GitSnapshot, base: GitSnapshot) -> None:
        self.target = target
        self.base = base
        self.diags: list[Diagnostic] = []
        self.page_states: dict[str, str] = {}
        self.collection_states: dict[str, str] = {}
        self.historical_prefixes: list[str] = []

    def add(self, code: str, path: str, message: str) -> None:
        self.diags.append(Diagnostic(code, path, message))

    def run(self) -> list[Diagnostic]:
        self.check_admissions()
        self.check_index()
        self.check_skill_roots()
        self.check_links_and_paths()
        self.check_coherence_acceptance()
        self.check_env_coverage()
        self.check_checkpoints()
        self.diags.sort(key=lambda d: (d.code, d.path, d.message))
        return self.diags

    def is_historical_path(self, path: str) -> bool:
        state = classify_path_from_index(path, self.page_states, self.collection_states)
        if is_historicalish_state(state):
            return True
        for prefix in self.historical_prefixes:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                return True
        return False

    def is_active_markdown(self, path: str) -> bool:
        if path in ACTIVE_MARKDOWN:
            return True
        if path.startswith(".agents/skills/") and path.endswith((".md", ".MD")):
            return True
        # Non-historical docs/**/*.md are active for scans.
        if (
            path.startswith("docs/")
            and path.endswith((".md", ".MD"))
            and not self.is_historical_path(path)
        ):
            return True
        # Root routing markdown always active when present.
        if path in {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md"}:
            return True
        return False

    def active_markdown_paths(self) -> list[str]:
        """Central active Markdown set for links, coherence, and skill-ref scans."""
        paths = self.target.list_paths()
        out: list[str] = []
        for p in sorted(paths):
            if not p.endswith((".md", ".MD")):
                continue
            if self.is_active_markdown(p):
                out.append(p)
        for p in sorted(ACTIVE_MARKDOWN):
            if p in paths and p not in out:
                out.append(p)
        return out

    # --- admissions -----------------------------------------------------

    def check_admissions(self) -> None:
        path = ADMISSIONS_PATH
        text = self.target.read_text(path)
        if text is None:
            self.add("admissions", path, "missing admissions registry")
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            self.add("admissions", path, f"not JSON-compatible: {exc}")
            return
        if not isinstance(data, dict):
            self.add("admissions", path, "root must be a JSON object")
            return
        expected_keys = {"schema_version", "records"}
        if set(data.keys()) != expected_keys:
            self.add(
                "admissions",
                path,
                f"exact keys required {sorted(expected_keys)}; got {sorted(data.keys())}",
            )
            return
        if data.get("schema_version") != 1:
            self.add(
                "admissions",
                path,
                f"schema_version must be 1; got {data.get('schema_version')!r}",
            )
        records = data.get("records")
        if not isinstance(records, list):
            self.add("admissions", path, "records must be a list")
            return
        if records != []:
            self.add(
                "admissions",
                path,
                "fail closed: records must be [] until earned admissions exist",
            )

    # --- INDEX ownership / lifecycle ------------------------------------

    def check_index(self) -> None:
        path = INDEX_PATH
        text = self.target.read_text(path)
        if text is None:
            self.add("index", path, "missing documentation index")
            return

        meta = parse_index_metadata(text)
        branch = nonempty_branch_name(meta.get("branch", ""))
        source_sha = require_full_sha(meta.get("source", ""))
        base_sha = require_full_sha(meta.get("base", ""))

        if "branch" not in meta:
            self.add("index-meta", path, "missing Working branch metadata")
        elif not branch:
            self.add(
                "index-meta",
                path,
                "Working branch metadata must carry a nonempty branch name",
            )

        if "source" not in meta:
            self.add("index-meta", path, "missing Audited source ref metadata")
        elif not source_sha:
            self.add(
                "index-meta",
                path,
                "Audited source ref must include a full 40-hex commit",
            )

        if "base" not in meta:
            self.add("index-meta", path, "missing Comparison base metadata")
        elif not base_sha:
            self.add(
                "index-meta",
                path,
                "Comparison base must include a full 40-hex commit",
            )

        ownership = extract_section(text, "Ownership")
        if ownership is None:
            self.add("index", path, "missing Ownership section")
            return
        headers, rows = parse_markdown_table(ownership)
        norm_headers = tuple(normalize_header(h) for h in headers)
        if norm_headers not in OWNERSHIP_HEADER_VARIANTS:
            self.add(
                "index-schema",
                path,
                "ownership table headers must be "
                "'Concern | Canonical owner | State | Notes' or "
                "'Concern | Canonical owner | Truth state | Scope/limits'; "
                f"got {' | '.join(norm_headers) if norm_headers else '(none)'}",
            )
            return

        concerns: dict[str, int] = defaultdict(int)
        branch_rows = 0
        for row in rows:
            concern, owner, truth, _notes = (row + ["", "", "", ""])[:4]
            if not concern:
                continue
            concerns[concern] += 1
            tkey = truth_key(truth)
            if tkey in BRANCHISH_TRUTH or "branch-implemented" in tkey or tkey == "branch-only":
                branch_rows += 1
            if tkey not in {s.lower() for s in TRUTH_STATES} and not is_blocked_truth(tkey):
                self.add(
                    "index-truth",
                    path,
                    f"concern {concern!r}: unrecognized truth/state {strip_md(truth)!r}",
                )

            links = extract_owner_link_targets(owner)
            blocked = is_blocked_truth(tkey) or owner_is_blocked_marker(owner)
            if blocked:
                if len(links) > 1:
                    self.add(
                        "index-owner",
                        path,
                        f"concern {concern!r}: blocked row must not list multiple owner links",
                    )
            else:
                if len(links) != 1:
                    self.add(
                        "index-owner",
                        path,
                        f"concern {concern!r}: need exactly one Markdown owner link "
                        f"or explicit BLOCKED (found {len(links)} link(s))",
                    )
                elif owner_link_is_external(links[0]):
                    self.add(
                        "index-owner",
                        path,
                        f"concern {concern!r}: canonical owner link must be local, not external",
                    )

        for concern, count in sorted(concerns.items()):
            if count > 1:
                self.add(
                    "index-owner",
                    path,
                    f"concern {concern!r}: multiple ownership rows ({count})",
                )

        if branch_rows and not (branch and source_sha and base_sha):
            self.add(
                "branch-meta",
                path,
                "branch-only ownership rows require nonempty branch name and "
                "full 40-hex source/base commits on INDEX",
            )

        # Page classification: every top-level docs page exactly once.
        page_sec = extract_section(text, "Top-level page classification")
        if page_sec is None:
            self.add("lifecycle", path, "missing Top-level page classification section")
            page_rows: list[list[str]] = []
        else:
            _ph, page_rows = parse_markdown_table(page_sec)

        classified_pages: dict[str, str] = {}
        page_branchish = 0
        for row in page_rows:
            page_cell, state = (row + ["", ""])[:2]
            state_clean = strip_md(state)
            if not is_allowed_lifecycle_label(state_clean):
                self.add(
                    "lifecycle",
                    path,
                    f"unrecognized/unknown page lifecycle label {state_clean!r}",
                )
            if state_clean and is_branchish_label(state_clean):
                page_branchish += 1
            links = extract_owner_link_targets(page_cell)
            if not links:
                ticks = re.findall(r"`([^`]+)`", page_cell)
                cand = ticks[0] if ticks else strip_md(page_cell)
            else:
                if owner_link_is_external(links[0]):
                    self.add(
                        "lifecycle",
                        path,
                        f"page classification owner link must be local: {page_cell!r}",
                    )
                cand = split_link_target(links[0])[0]
            if not cand:
                continue
            resolved, err = normalize_rel("docs/INDEX.md", cand)
            if err or not resolved:
                self.add(
                    "lifecycle",
                    path,
                    f"page classification entry escapes repo or is empty: {page_cell!r}",
                )
                continue
            if resolved in classified_pages:
                self.add(
                    "lifecycle",
                    path,
                    f"top-level page classified more than once: {resolved}",
                )
            classified_pages[resolved] = state_clean

        # Collection classification: every top-level docs collection exactly once.
        coll_sec = extract_section(text, "Collection classification")
        if coll_sec is None:
            self.add("lifecycle", path, "missing Collection classification section")
            coll_rows: list[list[str]] = []
        else:
            _ch, coll_rows = parse_markdown_table(coll_sec)

        classified_colls: dict[str, str] = {}
        coll_branchish = 0
        for row in coll_rows:
            coll_cell, state = (row + ["", ""])[:2]
            state_clean = strip_md(state)
            if not is_allowed_lifecycle_label(state_clean):
                self.add(
                    "lifecycle",
                    path,
                    f"unrecognized/unknown collection lifecycle label {state_clean!r}",
                )
            if state_clean and is_branchish_label(state_clean):
                coll_branchish += 1
            links = extract_owner_link_targets(coll_cell)
            if links:
                if owner_link_is_external(links[0]):
                    self.add(
                        "lifecycle",
                        path,
                        f"collection classification link must be local: {coll_cell!r}",
                    )
                cand = split_link_target(links[0])[0]
            else:
                ticks = re.findall(r"`([^`]+)`", coll_cell)
                cand = ticks[0] if ticks else strip_md(coll_cell)
            if not cand:
                continue
            resolved, err = normalize_rel("docs/INDEX.md", cand)
            if err or resolved is None:
                self.add(
                    "lifecycle",
                    path,
                    f"collection classification entry escapes repo: {coll_cell!r}",
                )
                continue
            name = resolved
            if name.startswith("docs/"):
                name = name[len("docs/") :]
            name = name.strip("/") + "/"
            if name in classified_colls:
                self.add(
                    "lifecycle",
                    path,
                    f"collection classified more than once: {name}",
                )
            classified_colls[name] = state_clean

        self.page_states = classified_pages
        self.collection_states = classified_colls
        self.historical_prefixes = []
        for coll, state in classified_colls.items():
            if is_historicalish_state(state):
                self.historical_prefixes.append(f"docs/{coll}")
        for page, state in classified_pages.items():
            if is_historicalish_state(state):
                self.historical_prefixes.append(page)

        top_pages = sorted(
            p
            for p in self.target.list_paths()
            if p.startswith("docs/") and p.count("/") == 1 and not p.endswith("/")
        )
        for p in top_pages:
            if p not in classified_pages:
                self.add(
                    "lifecycle",
                    path,
                    f"top-level docs page not lifecycle-classified exactly once: {p}",
                )

        coll_names: set[str] = set()
        for p in self.target.list_paths():
            if not p.startswith("docs/"):
                continue
            rest = p[len("docs/") :]
            if "/" not in rest:
                continue
            coll_names.add(rest.split("/", 1)[0] + "/")

        for coll in sorted(coll_names):
            if coll not in classified_colls:
                self.add(
                    "lifecycle",
                    path,
                    f"top-level docs collection not lifecycle-classified exactly once: {coll}",
                )

        if (page_branchish or coll_branchish) and not (branch and source_sha and base_sha):
            self.add(
                "branch-meta",
                path,
                "branch lifecycle rows require nonempty branch name and "
                "full 40-hex source/base commits on INDEX",
            )

    # --- skill roots ----------------------------------------------------

    def check_skill_roots(self) -> None:
        paths = self.target.list_paths()
        skill_paths = [p for p in paths if p.startswith(".agents/skills/")]
        if not skill_paths:
            self.add("skills-root", ".agents/skills/", "sole executable skill root is missing")
        else:
            skill_ids: set[str] = set()
            for p in skill_paths:
                rest = p[len(".agents/skills/") :]
                if not rest or rest.endswith("/"):
                    continue
                skill_id = rest.split("/", 1)[0]
                if skill_id:
                    skill_ids.add(skill_id)
            for skill_id in sorted(skill_ids):
                skill_md = f".agents/skills/{skill_id}/SKILL.md"
                if not self.target.exists(skill_md):
                    self.add(
                        "skills-root",
                        f".agents/skills/{skill_id}/",
                        "tracked skill subtree must contain SKILL.md",
                    )

        for p in sorted(p for p in paths if p == ".skills" or p.startswith(".skills/")):
            self.add("skills-root", p, ".skills/ must not exist; use .agents/skills/")

        for p in sorted(paths):
            if is_executable_docs_skills_target(p):
                self.add(
                    "skills-root",
                    p,
                    "executable skill definition under docs/skills/ is prohibited",
                )

        # Scan the same centralized active Markdown set used elsewhere.
        for path in self.active_markdown_paths():
            text = self.target.read_text(path)
            if text is None:
                continue
            if re.search(r"(?<![\w./-])\.skills/", text) or "`.skills/" in text:
                self.add("skills-root", path, "active document references .skills/ path")
            # Flag active references to executable docs/skills entrypoints.
            for match in MD_LINK_OR_IMAGE_RE.finditer(text):
                raw = match.group(1).strip()
                if not raw or is_external_link(raw.split()[0]):
                    continue
                link_path, _frag = split_link_target(raw)
                if not link_path or link_path.startswith("#"):
                    continue
                candidate = unquote(link_path)
                if candidate.startswith("//"):
                    continue
                if candidate.startswith("/"):
                    resolved = candidate.lstrip("/")
                    err = None
                else:
                    resolved, err = normalize_rel(path, candidate)
                if err or resolved is None:
                    continue
                if is_executable_docs_skills_target(resolved):
                    self.add(
                        "skills-root",
                        path,
                        f"active document references executable docs/skills path {resolved}",
                    )
            for match in BACKTICK_RE.finditer(text):
                for token in extract_path_tokens(match.group(1)):
                    resolved = strip_leading_dot_slash(token)
                    if is_executable_docs_skills_target(resolved):
                        self.add(
                            "skills-root",
                            path,
                            f"active document references executable docs/skills path {resolved}",
                        )

    # --- links, anchors, backticked paths --------------------------------

    def check_links_and_paths(self) -> None:
        md_paths = self.active_markdown_paths()
        anchors_cache: dict[str, set[str]] = {}

        def anchors_for(path: str) -> set[str]:
            if path not in anchors_cache:
                anchors_cache[path] = collect_anchors(self.target.read_text(path) or "")
            return anchors_cache[path]

        for path in md_paths:
            text = self.target.read_text(path)
            if text is None:
                continue
            self._check_md_links(path, text, anchors_for)
            self._check_backticked_paths(path, text)

    def _check_md_links(self, path: str, text: str, anchors_for) -> None:
        for match in MD_LINK_OR_IMAGE_RE.finditer(text):
            raw = match.group(1).strip()
            if not raw or raw.startswith("?"):
                continue
            first = raw.split()[0] if raw else raw
            if is_external_link(first):
                continue
            link_path, frag = split_link_target(raw)
            if link_path.startswith(EXTERNAL_SCHEMES):
                continue
            if link_path == "" and frag is not None:
                if frag and frag not in anchors_for(path):
                    self.add("anchor", path, f"missing anchor #{frag}")
                continue
            if link_path.startswith("#"):
                frag = link_path[1:]
                if frag and frag not in anchors_for(path):
                    self.add("anchor", path, f"missing anchor #{frag}")
                continue
            if "${" in link_path or link_path.startswith("mailto:"):
                continue

            candidate = unquote(link_path)
            if candidate.startswith("/"):
                if candidate.startswith("//"):
                    continue
                # Repo-root absolute markdown path: resolve from repo root, not referrer.
                resolved = candidate.lstrip("/")
                err = None
                if ".." in Path(resolved).parts:
                    err = "link escapes repository root"
            else:
                resolved, err = normalize_rel(path, candidate)
            if err:
                self.add("link", path, f"repo-escaping local link {raw!r}: {err}")
                continue
            assert resolved is not None

            if resolved.endswith("/"):
                prefix = resolved
                if not any(
                    p == resolved.rstrip("/") or p.startswith(prefix)
                    for p in self.target.list_paths()
                ):
                    self.add(
                        "link",
                        path,
                        f"broken local directory link {raw!r} -> {resolved}",
                    )
                continue
            if not self.target.exists(resolved):
                if not any(p.startswith(resolved + "/") for p in self.target.list_paths()):
                    self.add("link", path, f"broken local link {raw!r} -> {resolved}")
                continue
            if frag and (resolved.endswith(".md") or resolved.endswith(".MD")):
                if frag not in anchors_for(resolved):
                    self.add("anchor", path, f"missing anchor {resolved}#{frag}")

    def _check_backticked_paths(self, path: str, text: str) -> None:
        for match in BACKTICK_RE.finditer(text):
            span = match.group(1)
            for token in extract_path_tokens(span):
                if any(ch in token for ch in "*?["):
                    continue
                if token.startswith(EXTERNAL_SCHEMES):
                    continue
                resolved = strip_leading_dot_slash(token)
                if resolved.startswith(".skills/") or resolved == ".skills":
                    self.add("skills-root", path, f"backticked path references {token}")
                    continue
                if resolved == ".." or resolved.startswith("../"):
                    self.add("path", path, f"repo-escaping backticked path {token}")
                    continue
                if not self.target.exists(resolved):
                    if not any(
                        p.startswith(resolved.rstrip("/") + "/")
                        for p in self.target.list_paths()
                    ):
                        self.add("path", path, f"missing backticked path {token}")

    # --- coherence-gate acceptance --------------------------------------

    def check_coherence_acceptance(self) -> None:
        for path in self.active_markdown_paths():
            text = self.target.read_text(path)
            if text is None:
                continue
            for m in COHERENCE_ACCEPTANCE_RE.finditer(text):
                snippet = " ".join(m.group(0).split())
                window_start = max(0, m.start() - 80)
                window_end = min(len(text), m.end() + 80)
                window = text[window_start:window_end].lower()
                if any(
                    key in window
                    for key in (
                        "retired",
                        "historical reproduction",
                        "never promotion",
                        "not acceptance",
                        "must not be required",
                        "do not treat",
                        "rejected",
                    )
                ):
                    continue
                self.add(
                    "coherence-gate",
                    path,
                    f"active acceptance claim involving coherence-gate: {snippet[:160]}",
                )

    # --- env coverage (same contract as check-env-docs.py) --------------

    def check_env_coverage(self) -> None:
        canonical = self.target.read_text(ENV_CANONICAL)
        if canonical is None:
            self.add("env", ENV_CANONICAL, "missing canonical env docs")
            return
        canon_vars = set(HIPFIRE_ENV_RE.findall(canonical))
        for rel in ENV_REFERENCE_DOCS:
            text = self.target.read_text(rel)
            if text is None:
                continue
            for name in sorted(set(HIPFIRE_ENV_RE.findall(text))):
                if name not in canon_vars:
                    self.add(
                        "env",
                        rel,
                        f"{name} not documented in {ENV_CANONICAL}",
                    )

    # --- perf-checkpoints immutability ----------------------------------

    def check_checkpoints(self) -> None:
        base_files = {
            p: self.base.read_bytes(p)
            for p in self.base.paths_with_prefix(CHECKPOINT_PREFIX)
            if not p.endswith("/")
        }
        for path, base_bytes in sorted(base_files.items()):
            if base_bytes is None:
                continue
            if not self.target.exists(path):
                self.add(
                    "checkpoint",
                    path,
                    "pre-existing perf-checkpoint deleted versus base",
                )
                continue
            target_bytes = self.target.read_bytes(path)
            if target_bytes != base_bytes:
                self.add(
                    "checkpoint",
                    path,
                    "pre-existing perf-checkpoint bytes changed versus base",
                )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-ref", help="git ref for the target snapshot")
    mode.add_argument(
        "--staged",
        action="store_true",
        help="use the current index (git write-tree) as the target snapshot",
    )
    p.add_argument("--base-ref", required=True, help="git ref for the comparison base")
    p.add_argument(
        "--root",
        default=str(ROOT),
        help="repository root (default: parent of scripts/)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        print(f"error: {root} is not a git work tree", file=sys.stderr)
        return 2

    base = GitSnapshot.from_ref(root, args.base_ref)
    if args.staged:
        target = GitSnapshot.from_index(root)
    else:
        if not args.target_ref:
            print("error: --target-ref is required unless --staged", file=sys.stderr)
            return 2
        target = GitSnapshot.from_ref(root, args.target_ref)

    diags = Checker(target, base).run()
    if diags:
        for d in diags:
            print(d.format())
        print(f"docs-reliability: {len(diags)} issue(s)", file=sys.stderr)
        return 1
    print("docs-reliability: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

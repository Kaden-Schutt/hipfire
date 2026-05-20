#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Apply SPDX + copyright headers to first-party hipfire source files.

hipfire is dual-licensed under MIT OR Apache-2.0 (Rust ecosystem
norm; see LICENSE, LICENSE-MIT, LICENSE-APACHE, and NOTICE at the
repo root). The per-file SPDX-License-Identifier reflects ACTUAL
authorship, not a project-wide blanket relicense:

    * Apache-2.0       — file's substantive authors (>= secondary
                         threshold, default 30 %) are all in the
                         {Kaden Schutt, Kevin Read} set, i.e. authors
                         who have opted into Apache-2.0 going forward.
    * MIT              — file's substantive authors are all OUTSIDE
                         the {Kaden, Kevin} set. We respect their
                         original MIT grant rather than relicensing
                         in absentia; if they later opt into dual,
                         re-run --rewrite-spdx to refresh.
    * MIT OR Apache-2.0 — mixed substantive authorship across the
                         opt-in / not-yet-opted-in boundary. The dual
                         tag is the conservative position until the
                         non-K/K author opts in via the tracking
                         issue.

Files newly created by Kaden in this relicense work (e.g. this very
script) are explicitly Apache-2.0 via APACHE_OVERRIDE_FILES below.

Sweep scope:

    crates/**/*.rs
    kernels/src/**/*.{hip,cuh}
    scripts/**/*.py
    scripts/**/*.sh

Skips:
    * any path under target/, .git/, .hipfire_kernels/, node_modules/,
      .claude/worktrees/, .worktrees/
    * any path under a vendored/ / third_party/ / third-party/ dir
    * Markdown, TOML, JSON, YAML, dotfiles (out of scope)
    * binaries (gguf, safetensors, hfq, bin)

Modes:

    # Dry-run scan for unheadered files (default):
    python3 scripts/governance/apply_spdx_headers.py

    # Write headers to all unheadered files:
    python3 scripts/governance/apply_spdx_headers.py --apply

    # Rewrite the SPDX-License-Identifier line on already-headered
    # files to match current authorship. Used by the dual-licensing
    # course correction (2026-05-19). Copyright lines beneath the
    # SPDX line are preserved unchanged.
    python3 scripts/governance/apply_spdx_headers.py --rewrite-spdx
    python3 scripts/governance/apply_spdx_headers.py --rewrite-spdx --apply

    # Write a structured report to disk:
    python3 scripts/governance/apply_spdx_headers.py --rewrite-spdx \\
        --apply --report-path /tmp/hipfire-spdx-correction-report.txt

Identities are collapsed by an explicit email -> canonical-name map
(see IDENTITY_MAP below) before share computation, so Kaden's three
git configs and Kevin's two collapse to one entry each.

The script is committed alongside the relicense so future
contributors can (a) bring a new file into compliance with `--apply`
after adding it, and (b) re-audit attribution after a substantial
refactor or after an opted-in dual-licensing batch using
`--rewrite-spdx`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Comment-style by file extension. `style` is the per-line prefix
# (including trailing space). `shebang_aware` means the header is
# inserted AFTER a `#!`-style first line if present, with a blank line
# separating shebang -> SPDX -> existing content.
EXTENSIONS = {
    ".rs":  {"style": "// ", "shebang_aware": False},
    ".hip": {"style": "// ", "shebang_aware": False},
    ".cuh": {"style": "// ", "shebang_aware": False},
    ".py":  {"style": "# ",  "shebang_aware": True},
    ".sh":  {"style": "# ",  "shebang_aware": True},
}

# Path components that disqualify any descendant from header sweep.
EXCLUDE_DIR_NAMES = {
    "target",
    ".git",
    ".hipfire_kernels",
    "node_modules",
    "vendored",
    "third_party",
    "third-party",
    "thirdparty",
    ".worktrees",
}

# Top-level include roots; the sweep does not descend outside these.
INCLUDE_ROOTS = ("crates", "kernels/src", "scripts")

# Collapse git identities (by email) onto canonical names. Mirrors the
# contributor analysis in docs/governance/relicense-2026-05.md so that
# Kaden's three configs and Kevin's two work tree as one author each.
IDENTITY_MAP: dict[str, str] = {
    "k9crypto@protonmail.com": "Kaden Schutt",
    "151092359+kaden-schutt@users.noreply.github.com": "Kaden Schutt",
    "kaden@k9lin.local": "Kaden Schutt",
    "me@kevin-read.com": "Kevin Read",
    "kevin.read@schleifenbauer.eu": "Kevin Read",
    "alpineq@protonmail.com": "alpineq",
    "37345260+alpineq@users.noreply.github.com": "alpineq",
    "kotdath@yandex.ru": "Daniil Markevich",
    "97358902+kotdath@users.noreply.github.com": "Daniil Markevich",
    "tgutierrezlet@gmail.com": "Tomás Gutiérrez L.",
}

# Default project-copyright fallback used when git blame produces no
# author share for a file (e.g. brand-new untracked file). Override
# with --fallback-author "Your Name".
DEFAULT_FALLBACK_AUTHOR = "Kaden Schutt"

# Default copyright year. Override with --year YYYY for re-attribution
# passes in later years (do NOT modify existing header lines; this only
# affects newly-written headers).
DEFAULT_COPYRIGHT_YEAR = 2026

# Trailing project pointer line shared by every header.
PROJECT_POINTER_LINE = "hipfire — see LICENSE and NOTICE in the project root."

# How many lines from the top to scan for an existing SPDX marker
# before deciding the file is unheadered.
SPDX_SCAN_LINES = 25

# Authors who have opted into Apache-2.0 going forward. A file whose
# substantive authors (>= secondary threshold) are entirely within
# this set is tagged Apache-2.0. A file whose substantive authors are
# entirely outside this set is tagged MIT (respecting their original
# MIT grant). Mixed files are tagged "MIT OR Apache-2.0".
KADEN_OR_KEVIN: set[str] = {"Kaden Schutt", "Kevin Read"}

# Paths (repo-root-relative) that are explicitly Apache-2.0 regardless
# of git blame. Use for files newly created in this session by Kaden
# (so that even before they have meaningful blame history they carry
# the intended license tag).
APACHE_OVERRIDE_FILES: set[str] = {
    "scripts/governance/apply_spdx_headers.py",
}

# SPDX-License-Identifier values produced by `classify_license`.
LICENSE_APACHE = "Apache-2.0"
LICENSE_MIT = "MIT"
LICENSE_DUAL = "MIT OR Apache-2.0"


def classify_license(authors: list[str], rel_path: str) -> str:
    """Return the SPDX-License-Identifier value for a file.

    `authors` is the same selected-author list that gets rendered as
    copyright lines (descending by share, secondary-threshold-filtered).
    The classification rule is:

        rel_path in APACHE_OVERRIDE_FILES  -> Apache-2.0
        every author in KADEN_OR_KEVIN     -> Apache-2.0
        no author in KADEN_OR_KEVIN        -> MIT
        otherwise (mixed)                   -> MIT OR Apache-2.0

    An empty `authors` list (genuinely unknown authorship) falls back
    to Apache-2.0 — this only happens when git blame returned nothing
    AND the file is brand-new, in which case the author is whoever is
    running --apply, who is by definition contributing under the
    project's default (Apache-2.0 per CONTRIBUTING.md).
    """
    if rel_path in APACHE_OVERRIDE_FILES:
        return LICENSE_APACHE
    if not authors:
        return LICENSE_APACHE
    has_kk = any(a in KADEN_OR_KEVIN for a in authors)
    has_other = any(a not in KADEN_OR_KEVIN for a in authors)
    if has_kk and not has_other:
        return LICENSE_APACHE
    if has_other and not has_kk:
        return LICENSE_MIT
    return LICENSE_DUAL


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------


@dataclass
class FileReport:
    """Per-file outcome from the sweep."""

    path: Path
    action: str  # "would-add", "added", "skip-existing", "skip-empty",
                 # "skip-no-blame", "skip-error"
    authors_added: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class SweepStats:
    eligible: int = 0
    skipped_existing: int = 0
    skipped_empty: int = 0
    skipped_no_blame: int = 0
    skipped_error: int = 0
    would_add_or_added: int = 0
    multi_author_headers: int = 0
    files_per_secondary: dict[str, int] = field(default_factory=dict)
    files_by_license: dict[str, int] = field(default_factory=dict)


@dataclass
class RewriteStats:
    """Stats for --rewrite-spdx mode."""

    eligible: int = 0
    skipped_unheadered: int = 0
    skipped_no_blame: int = 0
    skipped_error: int = 0
    skipped_no_change: int = 0
    rewritten: int = 0
    transitions: dict[tuple[str, str], int] = field(default_factory=dict)
    files_apache_to_mit: list[Path] = field(default_factory=list)
    files_apache_to_dual: list[Path] = field(default_factory=list)
    files_kept_apache: int = 0
    files_kept_mit: int = 0
    files_kept_dual: int = 0
    multi_author_files: list[tuple[Path, str, list[str]]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------


def is_excluded_path(path: Path, repo_root: Path) -> bool:
    """True if any part of `path` (under repo_root) is in EXCLUDE_DIR_NAMES."""
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return True
    parts = set(rel.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    # .claude/worktrees specifically (not just any .claude dir, in case
    # we ever ship .claude/skills/ etc. inside the repo).
    if ".claude" in rel.parts and "worktrees" in rel.parts:
        return True
    return False


def discover_eligible_files(repo_root: Path) -> list[Path]:
    """Walk INCLUDE_ROOTS and return paths matching EXTENSIONS, sorted."""
    eligible: list[Path] = []
    for root_name in INCLUDE_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories in-place so os.walk does not
            # descend into them.
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            for fname in filenames:
                p = Path(dirpath) / fname
                ext = p.suffix.lower()
                if ext not in EXTENSIONS:
                    continue
                if is_excluded_path(p, repo_root):
                    continue
                eligible.append(p)
    return sorted(eligible)


# ---------------------------------------------------------------------
# Header detection / construction
# ---------------------------------------------------------------------


def has_existing_spdx(file_path: Path) -> bool:
    """Look for `SPDX-License-Identifier:` in the first SPDX_SCAN_LINES."""
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= SPDX_SCAN_LINES:
                    break
                if "SPDX-License-Identifier:" in line:
                    return True
    except OSError:
        return False
    return False


def build_header_lines(
    style: str,
    authors: list[str],
    year: int,
    license_id: str,
) -> list[str]:
    """Construct the SPDX header block as a list of lines (no trailing \\n)."""
    lines = [f"{style}SPDX-License-Identifier: {license_id}"]
    for author in authors:
        lines.append(f"{style}Copyright (c) {year} {author}")
    lines.append(f"{style}{PROJECT_POINTER_LINE}")
    return lines


# ---------------------------------------------------------------------
# Authorship analysis (git blame)
# ---------------------------------------------------------------------


def canonical_author(name: str, email: str) -> Optional[str]:
    """Resolve a (name, email) pair to its canonical author name.

    Returns None for synthetic git author strings that don't represent
    a real contributor: "Not Committed Yet" (uncommitted working-tree
    edits), "External file" (git blame's stand-in when the file is
    outside the repo). Callers should drop None entries from the
    share calc.
    """
    nm = name.strip()
    if nm in {"Not Committed Yet", "External file (--contents)"}:
        return None
    key = email.strip().lower()
    if key in IDENTITY_MAP:
        return IDENTITY_MAP[key]
    return nm


def author_shares(
    file_path: Path,
    repo_root: Path,
    rev: Optional[str] = None,
) -> Optional[dict[str, int]]:
    """Return {canonical_author: line_count} per `git blame`, or None.

    None indicates that git blame failed (file is untracked at `rev`,
    binary, or repository is unavailable). An empty dict means the
    file is tracked but has zero lines of blame output.

    If `rev` is given, blame uses that revision (e.g. `d46f81b6` for
    pre-relicense authorship). Otherwise blame uses the working tree.
    """
    rel = file_path.relative_to(repo_root)
    cmd = ["git", "blame", "--line-porcelain"]
    if rev is not None:
        cmd.append(rev)
    cmd.extend(["--", str(rel)])
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None

    counts: dict[str, int] = {}
    current_name: Optional[str] = None
    current_email: Optional[str] = None
    for line in result.stdout.splitlines():
        if line.startswith("author "):
            current_name = line[len("author "):].strip()
        elif line.startswith("author-mail "):
            email = line[len("author-mail "):].strip()
            if email.startswith("<") and email.endswith(">"):
                email = email[1:-1]
            current_email = email
        elif line.startswith("\t"):
            # Commit content line (one per blamed source line). Use the
            # most-recent author/email seen for this commit chunk.
            if current_name is not None:
                ca = canonical_author(current_name, current_email or "")
                if ca is not None:
                    counts[ca] = counts.get(ca, 0) + 1
    return counts


def select_authors(
    counts: dict[str, int],
    secondary_threshold: float,
    fallback_author: str,
) -> list[str]:
    """Choose which author copyright lines belong on this file.

    Rule:
    * If no authorship is detectable, return [fallback_author].
    * Otherwise, sort authors by line count descending. Always include
      the top author. Additionally include every secondary author
      whose share crosses the threshold.

    The result is the ordered list of names to render as copyright
    lines (in descending-share order, ties broken alphabetically).
    """
    if not counts:
        return [fallback_author]

    total = sum(counts.values())
    if total <= 0:
        return [fallback_author]

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_name, _top_count = ordered[0]
    chosen = [top_name]
    for name, count in ordered[1:]:
        if count / total >= secondary_threshold and name not in chosen:
            chosen.append(name)
    return chosen


# ---------------------------------------------------------------------
# File rewrite
# ---------------------------------------------------------------------


def insert_header(
    file_path: Path,
    header_lines: list[str],
    shebang_aware: bool,
    apply: bool,
) -> None:
    """Write `header_lines` into `file_path` at the correct offset.

    For non-shebang-aware files, the header is prepended at the very
    top, followed by a blank line, then the existing content.

    For shebang-aware files (.py, .sh): if line 1 starts with `#!`,
    the header is inserted between the shebang and the rest of the
    content, with a blank line on each side. Otherwise it is
    prepended.
    """
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        original = f.read()

    lines = original.splitlines(keepends=True)
    if shebang_aware and lines and lines[0].startswith("#!"):
        shebang = lines[0]
        rest = "".join(lines[1:])
        # Trim leading whitespace from `rest` so we don't end up with
        # blank-line drift if there was already a gap after the shebang.
        rest_stripped = rest.lstrip("\n")
        new_content = (
            shebang
            + "\n"
            + "\n".join(header_lines)
            + "\n"
            + ("\n" + rest_stripped if rest_stripped else "")
        )
    else:
        new_content = (
            "\n".join(header_lines)
            + "\n"
            + ("\n" + original.lstrip("\n") if original else "")
        )

    if apply:
        with file_path.open("w", encoding="utf-8") as f:
            f.write(new_content)


# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------


def run_sweep(
    repo_root: Path,
    apply: bool,
    secondary_threshold: float,
    fallback_author: str,
    year: int,
    verbose: bool,
) -> tuple[list[FileReport], SweepStats]:
    eligible = discover_eligible_files(repo_root)
    stats = SweepStats(eligible=len(eligible))
    reports: list[FileReport] = []

    for path in eligible:
        ext = path.suffix.lower()
        spec = EXTENSIONS[ext]

        # Idempotency check.
        if has_existing_spdx(path):
            stats.skipped_existing += 1
            reports.append(FileReport(path=path, action="skip-existing"))
            continue

        # Authorship analysis.
        try:
            blame = author_shares(path, repo_root)
        except Exception as exc:  # pragma: no cover - defensive
            stats.skipped_error += 1
            reports.append(
                FileReport(path=path, action="skip-error", note=str(exc))
            )
            continue

        if blame is None:
            # Tracked but blame failed (often: brand-new file with no
            # committed history yet). Use fallback author.
            blame = {}
            stats.skipped_no_blame += 1

        # Empty-file guard. Authors may add empty files with `touch`;
        # treat as a no-op so we don't introduce a header that is the
        # entire file content.
        try:
            if path.stat().st_size == 0:
                stats.skipped_empty += 1
                reports.append(FileReport(path=path, action="skip-empty"))
                continue
        except OSError:
            pass

        authors = select_authors(blame, secondary_threshold, fallback_author)
        rel_str = str(path.relative_to(repo_root))
        license_id = classify_license(authors, rel_str)
        header_lines = build_header_lines(
            spec["style"], authors, year, license_id
        )

        stats.files_by_license[license_id] = (
            stats.files_by_license.get(license_id, 0) + 1
        )
        if len(authors) > 1:
            stats.multi_author_headers += 1
            for secondary in authors[1:]:
                stats.files_per_secondary[secondary] = (
                    stats.files_per_secondary.get(secondary, 0) + 1
                )

        try:
            insert_header(
                path,
                header_lines,
                shebang_aware=spec["shebang_aware"],
                apply=apply,
            )
        except OSError as exc:
            stats.skipped_error += 1
            reports.append(
                FileReport(path=path, action="skip-error", note=str(exc))
            )
            continue

        stats.would_add_or_added += 1
        reports.append(
            FileReport(
                path=path,
                action="added" if apply else "would-add",
                authors_added=authors,
            )
        )

        if verbose:
            rel = path.relative_to(repo_root)
            verb = "added" if apply else "would-add"
            print(f"  {verb}: {rel}  ({', '.join(authors)})")

    return reports, stats


def _extract_existing_spdx(file_path: Path) -> Optional[tuple[int, str, str]]:
    """Locate the existing `SPDX-License-Identifier: X` line.

    Returns (line_index, full_line_including_newline, current_identifier)
    or None if no such line is found in the first SPDX_SCAN_LINES.
    The returned `current_identifier` is the value to the right of the
    colon, stripped.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    for idx, line in enumerate(lines[:SPDX_SCAN_LINES]):
        marker = "SPDX-License-Identifier:"
        if marker in line:
            after = line.split(marker, 1)[1].strip()
            # Drop trailing comment delimiters if any (e.g. " */" for
            # C-block style, though we don't emit that form ourselves).
            for trail in (" -->", " */"):
                if after.endswith(trail):
                    after = after[: -len(trail)].rstrip()
            return idx, line, after
    return None


def _find_pointer_line(
    file_lines: list[str], spdx_idx: int, search_window: int = 10
) -> Optional[int]:
    """Find the trailing pointer line (`hipfire — see LICENSE ...`) that
    closes a header block opened by an SPDX line at `spdx_idx`.

    Returns its index in `file_lines`, or None if the header doesn't
    use the project's standard layout.
    """
    end = min(spdx_idx + search_window, len(file_lines))
    for i in range(spdx_idx, end):
        if "hipfire — see LICENSE" in file_lines[i]:
            return i
    return None


def run_rewrite(
    repo_root: Path,
    apply: bool,
    secondary_threshold: float,
    fallback_author: str,
    blame_rev: Optional[str],
    year: int,
    verbose: bool,
) -> tuple[RewriteStats, list[FileReport]]:
    """Rewrite SPDX header blocks to match current dual-license rule.

    Walks the same scope as run_sweep but operates only on files that
    already carry an SPDX-License-Identifier line. For each such file
    the authorship is recomputed (at `blame_rev` if given — usually
    pre-relicense state to avoid the SPDX-administration commit
    skewing the share calculation), the license is reclassified, and
    the entire header block (SPDX + copyright lines + pointer line) is
    rewritten in place.

    Lines above the SPDX block (shebang, if any) and lines below the
    pointer line (the actual file content) are not touched.
    """
    eligible = discover_eligible_files(repo_root)
    stats = RewriteStats(eligible=len(eligible))
    reports: list[FileReport] = []

    for path in eligible:
        rel_str = str(path.relative_to(repo_root))

        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                file_lines = f.readlines()
        except OSError as exc:
            stats.skipped_error += 1
            reports.append(
                FileReport(path=path, action="skip-error", note=str(exc))
            )
            continue

        existing = _extract_existing_spdx(path)
        if existing is None:
            stats.skipped_unheadered += 1
            reports.append(FileReport(path=path, action="skip-unheadered"))
            continue
        spdx_idx, original_line, current_id = existing

        pointer_idx = _find_pointer_line(file_lines, spdx_idx)
        if pointer_idx is None:
            stats.skipped_error += 1
            reports.append(
                FileReport(
                    path=path,
                    action="skip-error",
                    note="non-standard header layout (no pointer line)",
                )
            )
            continue

        # Compute authorship from pre-relicense blame so that the
        # SPDX-administration commit (which adds 3-5 lines of K-authored
        # comment to every file) doesn't shift the percentage calc.
        # Fall back to working-tree blame for files that didn't exist
        # at blame_rev (e.g. files created in this very session).
        blame = None
        if blame_rev is not None:
            blame = author_shares(path, repo_root, rev=blame_rev)
        if blame is None:
            blame = author_shares(path, repo_root, rev=None)
        if blame is None:
            stats.skipped_no_blame += 1
            blame = {}

        authors = select_authors(blame, secondary_threshold, fallback_author)
        new_id = classify_license(authors, rel_str)

        # Comment-style prefix is taken from the existing SPDX line so
        # we preserve whatever the prior sweep wrote (e.g. `// `, `# `).
        marker_pos = original_line.find("SPDX-License-Identifier:")
        prefix = original_line[:marker_pos]

        new_header = build_header_lines(prefix, authors, year, new_id)
        new_header_block = [line + "\n" for line in new_header]
        old_header_block = file_lines[spdx_idx:pointer_idx + 1]

        if new_header_block == old_header_block:
            stats.skipped_no_change += 1
            if new_id == LICENSE_APACHE:
                stats.files_kept_apache += 1
            elif new_id == LICENSE_MIT:
                stats.files_kept_mit += 1
            elif new_id == LICENSE_DUAL:
                stats.files_kept_dual += 1
            reports.append(
                FileReport(path=path, action="skip-no-change",
                           authors_added=authors, note=new_id)
            )
            continue

        if apply:
            new_file_lines = (
                file_lines[:spdx_idx]
                + new_header_block
                + file_lines[pointer_idx + 1:]
            )
            try:
                with path.open("w", encoding="utf-8") as f:
                    f.writelines(new_file_lines)
            except OSError as exc:
                stats.skipped_error += 1
                reports.append(
                    FileReport(path=path, action="skip-error", note=str(exc))
                )
                continue

        stats.rewritten += 1
        transition = (current_id, new_id)
        stats.transitions[transition] = stats.transitions.get(transition, 0) + 1
        if current_id == LICENSE_APACHE and new_id == LICENSE_MIT:
            stats.files_apache_to_mit.append(path)
        elif current_id == LICENSE_APACHE and new_id == LICENSE_DUAL:
            stats.files_apache_to_dual.append(path)

        if len(authors) > 1:
            stats.multi_author_files.append((path, new_id, authors))

        reports.append(
            FileReport(
                path=path,
                action="rewritten" if apply else "would-rewrite",
                authors_added=authors,
                note=f"{current_id} -> {new_id}",
            )
        )

        if verbose:
            verb = "rewrote" if apply else "would-rewrite"
            print(f"  {verb}: {rel_str}  ({current_id} -> {new_id}; "
                  f"{', '.join(authors)})")

    return stats, reports


def write_report(stats: RewriteStats, report_path: Path, apply: bool,
                 repo_root: Path) -> None:
    """Write the structured SPDX-correction report."""
    lines: list[str] = []
    lines.append(
        f"# hipfire SPDX correction report "
        f"({'APPLIED' if apply else 'DRY RUN'})"
    )
    lines.append("")
    lines.append(f"Eligible files in scope:    {stats.eligible}")
    lines.append(f"Skipped (no SPDX header):   {stats.skipped_unheadered}")
    lines.append(f"Skipped (errors):           {stats.skipped_error}")
    lines.append(f"Skipped (no blame):         {stats.skipped_no_blame}")
    lines.append(f"No change required:         {stats.skipped_no_change}")
    lines.append(f"  ... kept Apache-2.0:      {stats.files_kept_apache}")
    lines.append(f"  ... kept MIT:             {stats.files_kept_mit}")
    lines.append(f"  ... kept MIT OR Apache:   {stats.files_kept_dual}")
    lines.append(f"Rewritten:                  {stats.rewritten}")
    if stats.transitions:
        lines.append("")
        lines.append("## License transitions")
        lines.append("")
        for (old, new), count in sorted(stats.transitions.items()):
            lines.append(f"  {old:<22} -> {new:<22} {count}")

    lines.append("")
    lines.append("## Files reassigned Apache-2.0 -> MIT "
                 f"({len(stats.files_apache_to_mit)})")
    lines.append("")
    for p in sorted(stats.files_apache_to_mit):
        lines.append(f"  {p.relative_to(repo_root)}")

    lines.append("")
    lines.append("## Files reassigned Apache-2.0 -> MIT OR Apache-2.0 "
                 f"({len(stats.files_apache_to_dual)})")
    lines.append("")
    for p in sorted(stats.files_apache_to_dual):
        lines.append(f"  {p.relative_to(repo_root)}")

    lines.append("")
    lines.append(f"## Files kept at Apache-2.0:  {stats.files_kept_apache}")
    lines.append(f"## Files kept at MIT:         {stats.files_kept_mit}")
    lines.append(
        f"## Files kept at MIT OR Apache-2.0: {stats.files_kept_dual}"
    )

    if stats.multi_author_files:
        lines.append("")
        lines.append("## Multi-author files (corrected SPDX + authors)")
        lines.append("")
        for path, new_id, authors in sorted(
            stats.multi_author_files,
            key=lambda t: str(t[0])
        ):
            rel = path.relative_to(repo_root)
            lines.append(f"  {rel}")
            lines.append(f"    SPDX:    {new_id}")
            lines.append(f"    authors: {', '.join(authors)}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_rewrite_summary(stats: RewriteStats, apply: bool) -> None:
    print()
    print("=" * 60)
    print(f"  SPDX REWRITE — {'APPLIED' if apply else 'DRY RUN'}")
    print("=" * 60)
    print(f"  Eligible files in scope:        {stats.eligible}")
    print(f"  Skipped (no SPDX header):       {stats.skipped_unheadered}")
    print(f"  Skipped (errors):               {stats.skipped_error}")
    print(f"  No change required:             {stats.skipped_no_change}")
    print(f"     ... kept Apache-2.0:         {stats.files_kept_apache}")
    print(f"     ... kept MIT:                {stats.files_kept_mit}")
    print(f"     ... kept MIT OR Apache:      {stats.files_kept_dual}")
    if apply:
        print(f"  Rewritten:                      {stats.rewritten}")
    else:
        print(f"  Would rewrite:                  {stats.rewritten}")
    if stats.transitions:
        print()
        print("  License transitions:")
        for (old, new), count in sorted(stats.transitions.items()):
            print(f"    {old:<22} -> {new:<22} {count}")
    print("=" * 60)


def print_summary(stats: SweepStats, apply: bool) -> None:
    print()
    print("=" * 60)
    print(f"  SPDX HEADER SWEEP — {'APPLIED' if apply else 'DRY RUN'}")
    print("=" * 60)
    print(f"  Eligible files in scope:        {stats.eligible}")
    print(f"  Already headered (skipped):     {stats.skipped_existing}")
    print(f"  Empty files (skipped):          {stats.skipped_empty}")
    print(f"  Untracked (used fallback):      {stats.skipped_no_blame}")
    print(f"  Errors (skipped):               {stats.skipped_error}")
    if apply:
        print(f"  Headers WRITTEN:                {stats.would_add_or_added}")
    else:
        print(f"  Headers PROPOSED:               {stats.would_add_or_added}")
    print(f"  Multi-author headers (>=2):     {stats.multi_author_headers}")
    if stats.files_by_license:
        print()
        print("  Per-license breakdown:")
        for license_id, count in sorted(stats.files_by_license.items()):
            print(f"    {license_id:<22} {count}")
    if stats.files_per_secondary:
        print()
        print("  Files attributing a non-primary author "
              "(>= secondary threshold):")
        for name in sorted(
            stats.files_per_secondary,
            key=lambda n: (-stats.files_per_secondary[n], n),
        ):
            print(f"    {stats.files_per_secondary[name]:4}  {name}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Apache-2.0 SPDX + copyright headers to "
                    "hipfire first-party source files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the script is a dry "
             "run (default).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: cwd's nearest git toplevel).",
    )
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        default=0.30,
        help="Line-share threshold (0..1) above which a non-primary "
             "author gets a copyright line on a file. Default 0.30.",
    )
    parser.add_argument(
        "--fallback-author",
        default=DEFAULT_FALLBACK_AUTHOR,
        help="Author name used when git blame produces no result "
             f"(default: {DEFAULT_FALLBACK_AUTHOR}).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_COPYRIGHT_YEAR,
        help=f"Copyright year on newly-written lines "
             f"(default: {DEFAULT_COPYRIGHT_YEAR}). Existing headers "
             "are never modified.",
    )
    parser.add_argument(
        "--rewrite-spdx",
        action="store_true",
        help="Rewrite the SPDX-License-Identifier line on already-"
             "headered files to match current authorship per the "
             "dual-license classification rule. Leaves copyright "
             "lines untouched.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="If set, write a structured per-file report (transitions "
             "+ Apache->MIT list + Apache->dual list + multi-author "
             "headers) to this path. Only used by --rewrite-spdx.",
    )
    parser.add_argument(
        "--blame-rev",
        default=None,
        help="Revision to use for blame in --rewrite-spdx mode. "
             "Default None = use the working tree (which counts the "
             "SPDX-administration commit's own added lines toward "
             "Kaden's share, skewing authorship calc). For the "
             "course-correction sweep on the relicense-apache2 branch "
             "use --blame-rev d46f81b6 (the pre-relicense commit) so "
             "authorship reflects the actual code under license.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print one line per file as it is processed.",
    )
    return parser.parse_args()


def resolve_repo_root(arg: Optional[Path]) -> Path:
    if arg is not None:
        return arg.resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd().resolve()


def main() -> int:
    args = parse_args()
    repo_root = resolve_repo_root(args.root)
    if not repo_root.is_dir():
        print(f"error: repo root {repo_root} is not a directory",
              file=sys.stderr)
        return 2

    print(f"hipfire SPDX header sweep")
    print(f"  repo:     {repo_root}")
    print(f"  mode:     "
          f"{'REWRITE-SPDX' if args.rewrite_spdx else 'SWEEP'}"
          f" / {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  threshold for secondary author: "
          f"{args.secondary_threshold:.2f}")

    if args.rewrite_spdx:
        if args.blame_rev:
            print(f"  blame rev:                      {args.blame_rev}")
        rstats, _reports = run_rewrite(
            repo_root=repo_root,
            apply=args.apply,
            secondary_threshold=args.secondary_threshold,
            fallback_author=args.fallback_author,
            blame_rev=args.blame_rev,
            year=args.year,
            verbose=args.verbose,
        )
        print_rewrite_summary(rstats, args.apply)
        if args.report_path is not None:
            write_report(rstats, args.report_path, args.apply, repo_root)
            print(f"\nReport written to {args.report_path}")
        if not args.apply and rstats.rewritten > 0:
            print("\nDry run only. Re-run with --apply to write SPDX changes.")
        return 0

    print(f"  year on new headers:            {args.year}")

    _reports, stats = run_sweep(
        repo_root=repo_root,
        apply=args.apply,
        secondary_threshold=args.secondary_threshold,
        fallback_author=args.fallback_author,
        year=args.year,
        verbose=args.verbose,
    )
    print_summary(stats, args.apply)

    if not args.apply and stats.would_add_or_added > 0:
        print("\nDry run only. Re-run with --apply to write headers.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

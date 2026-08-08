#!/usr/bin/env python3
"""PreToolUse(Bash) guard: block workspace-wide Rust reformatting.

WHY: repos like hipfire carry historical rustfmt debt, so CI only checks
CHANGED files (scripts/ci-rustfmt-changed.sh). A bare `cargo fmt` rewrites
every file in the workspace (200+ files), burying the actual change and making
review impossible. The correct tool is scripts/fmt-changed.sh.

SCOPE: self-scoping. Only denies when the current repo has
scripts/fmt-changed.sh at its root -- i.e. a repo that has explicitly opted
into the changed-files-only formatting workflow. No-op everywhere else, and
resolves per-worktree so it follows the repo into .claude/worktrees/*.

ESCAPE HATCH (humans only; deliberately NOT named in the deny message so
agents don't reach for it): prefix the command with HIPFIRE_ALLOW_FMT=1.

stdlib only -- this machine has no jq, and a guard that fails open on a
missing dependency is not a guard.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

FIX = (
    "Use `scripts/fmt-changed.sh` instead - it formats ONLY the Rust files this "
    "branch touches, with the exact flags CI checks "
    "(--edition 2021 --config skip_children=true)."
)

# Command boundaries: start/end, whitespace, or a shell operator. A subshell's
# closing `)` is a terminator too -- `(cargo fmt)` must not slip through.
# Quotes are deliberately NOT boundaries here: `grep -rn 'cargo fmt' CLAUDE.md`
# is a legitimate search, not an invocation. Quoted payloads are handled by the
# exec-wrapper rescan below instead.
_L = r"(?:^|[;&|(]|\s)"
_R = r"(?:$|[;&|)]|\s)"

# Shells and remote-exec wrappers take their real command as a quoted string:
#   bash -lc 'cargo fmt --all'      ssh hipx 'cd hipfire && cargo fmt'
# The ssh form is the worst case -- it reformats a REMOTE checkout. When one of
# these is present we rescan with quotes flattened to whitespace.
EXEC_WRAPPER = re.compile(r"(?:^|[;&|(]|\s)(?:ba|z|d|k)?sh\s+-[a-z]*c(?:\s|$)|(?:^|\s)ssh\s+\S")

# `cargo fmt`, `cargo +nightly fmt`, `cargo-fmt` -- anywhere in a compound command.
CARGO_FMT = re.compile(_L + r"cargo(?:\s+\+\S+)?\s+fmt" + _R)
CARGO_FMT_BIN = re.compile(_L + r"cargo-fmt" + _R)
RUSTFMT = re.compile(_L + r"rustfmt" + _R)
READ_ONLY = re.compile(r"--(?:check|print-config)(?:\s|=|$)")
# recursive / glob / find / xargs / command-substitution => unbounded file set
UNBOUNDED = re.compile(r"(?:\s-r(?:\s|$)|--recursive|\*\.rs|\bfind\b|\bxargs\b|\$\(|`)")
HAS_RS_TARGET = re.compile(r"\S+\.rs" + _R)
# Must accept a newline as a leading boundary: Bash tool commands are commonly
# multi-line scripts, where the hatch sits at the start of a LINE, not of the
# string.
ESCAPE_HATCH = re.compile(r"(?:^|[;&|\n])\s*HIPFIRE_ALLOW_FMT=1")


def allow() -> None:
    sys.exit(0)


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        allow()

    cmd = (payload.get("tool_input") or {}).get("command") or ""
    if not cmd:
        allow()

    # --- scope: only guard repos using the changed-files formatter ----------
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        allow()
    if not root or not (Path(root) / "scripts" / "fmt-changed.sh").is_file():
        allow()

    if ESCAPE_HATCH.search(cmd):
        allow()

    # The sanctioned wrappers call rustfmt internally -- let them through.
    if re.search(r"scripts/(?:fmt-changed|ci-rustfmt-changed)\.sh", cmd):
        allow()

    # A shell/ssh wrapper hides its payload inside quotes; flatten them so the
    # same rules apply to `bash -lc '...'` and `ssh host '...'`.
    scan = cmd
    if EXEC_WRAPPER.search(cmd):
        scan = cmd.replace("'", " ").replace('"', " ")

    if CARGO_FMT.search(scan) or CARGO_FMT_BIN.search(scan):
        deny(
            "BLOCKED: `cargo fmt` reformats the ENTIRE workspace, not your change. "
            "This repo carries historical rustfmt debt and CI only gates CHANGED "
            "files, so a bare cargo fmt produces a 200+ file diff that buries the "
            "real work and makes review impossible. " + FIX
        )

    if RUSTFMT.search(scan):
        if READ_ONLY.search(scan):
            allow()
        if UNBOUNDED.search(scan):
            deny(
                "BLOCKED: this runs rustfmt over an unbounded set of files "
                "(glob/find/xargs/recursive), which mass-reformats the workspace. "
                + FIX
            )
        if not HAS_RS_TARGET.search(scan):
            deny(
                "BLOCKED: rustfmt with no explicit .rs file arguments walks the "
                "whole crate tree from the manifest root. " + FIX
            )

    allow()


if __name__ == "__main__":
    main()

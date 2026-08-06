#!/usr/bin/env bash
set -uo pipefail

# Detect a workspace-wide Rust reformat in the STAGED diff, whatever produced
# it — `cargo fmt`, an IDE format-on-save, a non-Claude agent, muscle memory.
#
# Why this exists: the repo carries historical rustfmt debt, so CI only checks
# CHANGED files (scripts/ci-rustfmt-changed.sh). A bare `cargo fmt` rewrites
# every file in the workspace, burying the real change under a 200+ file diff
# and making review impossible. scripts/fmt-changed.sh is the correct tool.
#
# Detection: a file whose staged content is byte-identical to HEAD once ALL
# whitespace is stripped has had only its formatting changed. One or two of
# those is normal (you touched a file carrying pre-existing format debt and
# fmt-changed.sh cleaned it). Dozens means the workspace got rewritten.
#
# Usage:
#   scripts/check-fmt-bomb.sh              # check the staged diff
#   FMT_BOMB_THRESHOLD=50 scripts/check-fmt-bomb.sh
#   SKIP_FMT_BOMB_CHECK=1 scripts/check-fmt-bomb.sh   # no-op
#
# Exit 0 = clean, exit 1 = fmt bomb detected.

if [[ -n "${SKIP_FMT_BOMB_CHECK:-}" ]]; then
    exit 0
fi

threshold="${FMT_BOMB_THRESHOLD:-25}"

mapfile -t staged < <(git diff --cached --name-only --diff-filter=ACM -- '*.rs')

# Cheap exit: a bomb needs more staged .rs files than the threshold.
if [[ "${#staged[@]}" -le "$threshold" ]]; then
    exit 0
fi

whitespace_only=0
for f in "${staged[@]}"; do
    # Newly added files have no HEAD version — nothing to compare against.
    git cat-file -e "HEAD:$f" 2>/dev/null || continue
    old=$(git show "HEAD:$f" 2>/dev/null | tr -d '[:space:]')
    new=$(git show ":$f" 2>/dev/null | tr -d '[:space:]')
    if [[ "$old" == "$new" ]]; then
        whitespace_only=$((whitespace_only + 1))
    fi
done

if [[ "$whitespace_only" -le "$threshold" ]]; then
    exit 0
fi

cat >&2 <<EOF

✗ fmt bomb: $whitespace_only staged Rust files changed ONLY in whitespace
  (out of ${#staged[@]} staged .rs files, threshold $threshold).

  This is the signature of a workspace-wide 'cargo fmt'. The repo carries
  historical format debt and CI only gates CHANGED files, so a full-workspace
  reformat buries the real change and makes review impossible.

  Inspect: git diff --cached --stat -- '*.rs'
  Undo:    git restore --staged --worktree -- '*.rs'   # DISCARDS those changes
  Format:  scripts/fmt-changed.sh   # only the files this branch touches
  Bypass:  SKIP_FMT_BOMB_CHECK=1 git commit ...   # deliberate sweep only
EOF
exit 1

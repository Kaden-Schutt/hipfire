#!/usr/bin/env bash
set -euo pipefail

# Format ONLY the Rust files this branch touches, matching CI's rules.
#
# Why this exists: the repo carries historical rustfmt debt (most files are
# not formatted), so CI only checks CHANGED files — see
# scripts/ci-rustfmt-changed.sh. Running bare `cargo fmt` rewrites the whole
# workspace's debt (100+ files) and buries your actual change. DO NOT run
# `cargo fmt` here. Run this instead — it formats only what you changed, with
# the same flags CI checks (`--edition 2021 --config skip_children=true`).
#
# Files considered = union of:
#   - committed changes vs the base ref (default origin/master; override BASE_REF)
#   - staged changes
#   - unstaged working-tree changes
#
# Usage:
#   scripts/fmt-changed.sh                 # format files changed vs origin/master
#   BASE_REF=origin/integration/foo scripts/fmt-changed.sh

base_ref="${BASE_REF:-origin/master}"

# Best-effort: make sure the base ref exists locally so the diff range works.
if ! git rev-parse --verify --quiet "${base_ref}" >/dev/null; then
  remote="${base_ref%%/*}"
  branch="${base_ref#*/}"
  git fetch --no-tags "${remote}" "${branch}:refs/remotes/${base_ref}" >/dev/null 2>&1 || true
fi

collect() {
  if git rev-parse --verify --quiet "${base_ref}" >/dev/null; then
    git diff --name-only --diff-filter=ACMRT "${base_ref}...HEAD" -- '*.rs'
  fi
  git diff --name-only --diff-filter=ACMRT -- '*.rs'          # unstaged
  git diff --cached --name-only --diff-filter=ACMRT -- '*.rs' # staged
}

mapfile -t files < <(collect | sort -u)

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No changed Rust files to format."
  exit 0
fi

printf 'rustfmt-formatting %d changed Rust file(s):\n' "${#files[@]}"
printf '  %s\n' "${files[@]}"
rustfmt --edition 2021 --config skip_children=true "${files[@]}"
echo "Done. Review the diff (it may include pre-existing format debt in files you touched — that is what CI's changed-file gate checks)."

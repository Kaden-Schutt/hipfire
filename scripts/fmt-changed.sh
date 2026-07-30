#!/usr/bin/env bash
set -euo pipefail

# rustfmt needs the edition of the crate OWNING each file. The workspace is
# edition 2021 but redline-rocr and redline-dispatch are 2024, and 2024-only
# syntax (let chains) is a PARSE ERROR under --edition 2021 — the gate then
# reports a formatting failure for a file it never managed to read. Walk up to
# the nearest Cargo.toml and use its edition, falling back to the workspace.
file_edition() {
  local dir; dir="$(dirname "$1")"
  while [ "$dir" != "/" ] && [ "$dir" != "." ]; do
    if [ -f "$dir/Cargo.toml" ]; then
      local ed
      ed="$(sed -n 's/^edition[[:space:]]*=[[:space:]]*"\([0-9]*\)".*/\1/p' "$dir/Cargo.toml" | head -1)"
      if [ -n "$ed" ]; then echo "$ed"; return; fi
      # edition.workspace = true -> fall through to the workspace root
      break
    fi
    dir="$(dirname "$dir")"
  done
  sed -n 's/^edition[[:space:]]*=[[:space:]]*"\([0-9]*\)".*/\1/p' Cargo.toml | head -1
}

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
#   - committed changes vs the base ref (REQUIRED — you must pick it)
#   - staged changes
#   - unstaged working-tree changes
#
# THE BASE REF IS MANDATORY. There is deliberately no default. A branch cut
# from origin/beta diffs against origin/master by every commit beta is ahead
# of master, so defaulting to master silently drags in the whole divergence
# and reformats hundreds of files you never touched — the exact fmt-bomb
# scripts/check-fmt-bomb.sh exists to block. Pick the ref your branch was
# actually cut from.
#
# Usage:
#   scripts/fmt-changed.sh master          # branch cut from origin/master
#   scripts/fmt-changed.sh beta            # branch cut from origin/beta
#   BASE_REF=origin/integration/foo scripts/fmt-changed.sh   # other bases
#
# Run with no argument to see how many files each base would format.

ensure_ref() {
  if ! git rev-parse --verify --quiet "$1" >/dev/null; then
    git fetch --no-tags "${1%%/*}" "${1#*/}:refs/remotes/$1" >/dev/null 2>&1 || true
  fi
  git rev-parse --verify --quiet "$1" >/dev/null
}

collect_for() {
  if ensure_ref "$1"; then
    git diff --name-only --diff-filter=ACMRT "$1...HEAD" -- '*.rs'
  fi
  git diff --name-only --diff-filter=ACMRT -- '*.rs'          # unstaged
  git diff --cached --name-only --diff-filter=ACMRT -- '*.rs' # staged
}

count_for() { collect_for "$1" | sort -u | wc -l; }

selection="${1:-${BASE_REF:-}}"
case "${selection}" in
  master|origin/master) base_ref="origin/master" ;;
  beta|origin/beta)     base_ref="origin/beta" ;;
  "")
    echo "error: no base ref selected — this script has no default on purpose." >&2
    echo >&2
    echo "  Pick the ref THIS branch was cut from. Choosing wrong reformats every" >&2
    echo "  file that differs between the two bases, burying your real change." >&2
    echo >&2
    printf '    scripts/fmt-changed.sh master   -> would format %s file(s)\n' "$(count_for origin/master)" >&2
    printf '    scripts/fmt-changed.sh beta     -> would format %s file(s)\n' "$(count_for origin/beta)" >&2
    echo >&2
    echo "  Other base: BASE_REF=origin/<branch> scripts/fmt-changed.sh" >&2
    exit 2
    ;;
  *)
    if [ -n "${BASE_REF:-}" ] && [ "${selection}" = "${BASE_REF}" ]; then
      base_ref="${BASE_REF}"   # explicit non-standard base, deliberately chosen
    else
      echo "error: unrecognized base '${selection}' — expected 'master' or 'beta'." >&2
      echo "       For any other base, set it explicitly:" >&2
      echo "         BASE_REF=origin/<branch> scripts/fmt-changed.sh" >&2
      exit 2
    fi
    ;;
esac

if ! ensure_ref "${base_ref}"; then
  echo "error: base ref '${base_ref}' not found locally and could not be fetched." >&2
  exit 2
fi

mapfile -t files < <(collect_for "${base_ref}" | sort -u)

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No changed Rust files to format."
  exit 0
fi

printf 'base %s: rustfmt-formatting %d changed Rust file(s):\n' "${base_ref}" "${#files[@]}"
printf '  %s\n' "${files[@]}"
for file in "${files[@]}"; do
  rustfmt --edition "$(file_edition "$file")" --config skip_children=true "$file"
done
echo "Done. Review the diff (it may include pre-existing format debt in files you touched — that is what CI's changed-file gate checks)."

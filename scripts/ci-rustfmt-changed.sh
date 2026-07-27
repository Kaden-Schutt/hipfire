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

# The repository still has historical rustfmt debt. Keep PR/push lint useful by
# enforcing rustfmt on the Rust files changed by this branch or push, without
# turning untouched legacy formatting into a permanent red check.

if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
  mapfile -t files < <(git diff --name-only --diff-filter=ACMRT -- '*.rs' | sort)
elif [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
  base_ref="${GITHUB_BASE_REF:?GITHUB_BASE_REF is required for pull_request events}"
  git fetch --no-tags origin "${base_ref}:refs/remotes/origin/${base_ref}"
  range="origin/${base_ref}...HEAD"
  mapfile -t files < <(git diff --name-only --diff-filter=ACMRT "${range}" -- '*.rs' | sort)
elif [[ -n "${GITHUB_EVENT_BEFORE:-}" && "${GITHUB_EVENT_BEFORE}" != "0000000000000000000000000000000000000000" ]]; then
  range="${GITHUB_EVENT_BEFORE}...HEAD"
  mapfile -t files < <(git diff --name-only --diff-filter=ACMRT "${range}" -- '*.rs' | sort)
else
  base_ref="${BASE_REF:-origin/master}"
  git fetch --no-tags origin "master:refs/remotes/origin/master"
  range="${base_ref}...HEAD"
  mapfile -t files < <(git diff --name-only --diff-filter=ACMRT "${range}" -- '*.rs' | sort)
fi

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No changed Rust files to rustfmt-check."
  exit 0
fi

printf 'rustfmt-checking %d changed Rust files:\n' "${#files[@]}"
printf '  %s\n' "${files[@]}"
status=0
for file in "${files[@]}"; do
  rustfmt --edition "$(file_edition "$file")" --check --config skip_children=true "$file" || status=1
done
exit "$status"

#!/usr/bin/env bash
set -euo pipefail

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
rustfmt --edition 2021 --check --config skip_children=true "${files[@]}"

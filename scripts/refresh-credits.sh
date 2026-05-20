#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# refresh-credits.sh
#
# Regenerate the Contributors block in /CREDITS.md from `gh pr list`.
# Idempotent: re-running with no new merged PRs produces a no-op diff.
#
# The block is delimited by:
#   <!-- contributors:auto-start -->
#   ...
#   <!-- contributors:auto-end -->
#
# Everything outside that block is left untouched.

set -uo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
CREDITS_FILE="${REPO_ROOT}/CREDITS.md"

START_MARK="<!-- contributors:auto-start -->"
END_MARK="<!-- contributors:auto-end -->"

OWNER_LOGIN="Kaden-Schutt"

die() {
  echo "refresh-credits: $*" >&2
  exit 1
}

warn() {
  echo "refresh-credits: $*" >&2
}

if ! command -v gh >/dev/null 2>&1; then
  warn "the 'gh' CLI is not installed."
  warn "install from https://cli.github.com/, then re-run."
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  warn "the 'jq' CLI is not installed."
  warn "install from https://stedolan.github.io/jq/, then re-run."
  exit 0
fi

if [[ ! -f "$CREDITS_FILE" ]]; then
  die "CREDITS.md not found at $CREDITS_FILE"
fi

if ! grep -qF "$START_MARK" "$CREDITS_FILE"; then
  die "start sentinel not found in CREDITS.md: $START_MARK"
fi

if ! grep -qF "$END_MARK" "$CREDITS_FILE"; then
  die "end sentinel not found in CREDITS.md: $END_MARK"
fi

# Make sure we are inside a repo with a remote gh can reach.
if ! gh auth status >/dev/null 2>&1; then
  warn "'gh auth status' failed. log in with 'gh auth login' and re-run."
  exit 0
fi

TMP_PRS="$( mktemp -t hipfire-credits-prs.XXXXXX )"
TMP_BLOCK="$( mktemp -t hipfire-credits-block.XXXXXX )"
TMP_OUT="$( mktemp -t hipfire-credits-out.XXXXXX )"
trap 'rm -f "$TMP_PRS" "$TMP_BLOCK" "$TMP_OUT"' EXIT

if ! gh pr list \
    --state merged \
    --base master \
    --limit 200 \
    --json number,headRefName,author,title,mergedAt \
    > "$TMP_PRS"; then
  die "gh pr list failed (non-zero exit). aborting refresh."
fi

if [[ ! -s "$TMP_PRS" ]]; then
  die "gh pr list returned empty output. aborting refresh."
fi

# Build the contributors block. Authors sorted by total PR count desc;
# within an author, PRs sorted by mergedAt desc. Project owner is dropped.
#
# jq runs into a separate file first so we can verify success and content
# before assembling the spliced block. A silent jq failure inside a
# brace-block would otherwise emit only the two sentinels, which the
# splice below would then write back, wiping the contributors section.
TMP_JQ="$( mktemp -t hipfire-credits-jq.XXXXXX )"
trap 'rm -f "$TMP_PRS" "$TMP_BLOCK" "$TMP_OUT" "$TMP_JQ"' EXIT

if ! jq -r --arg owner "$OWNER_LOGIN" '
    map(select(.author.login != $owner))
    | group_by(.author.login)
    | map({
        login: .[0].author.login,
        name: .[0].author.name,
        count: length,
        prs: (sort_by(.mergedAt) | reverse)
      })
    | sort_by([-.count, .login])
    | .[]
    | (
        "### " + (if (.name // "") != "" then .name else .login end)
        + " ([@" + .login + "](https://github.com/" + .login + ")) - "
        + (.count | tostring) + " PR" + (if .count > 1 then "s" else "" end) + "\n"
        + "\n"
        + (.prs | map("- #" + (.number | tostring) + ": " + .title) | join("\n"))
        + "\n"
      )
  ' "$TMP_PRS" > "$TMP_JQ"; then
  die "jq failed while building the contributors block. CREDITS.md unchanged."
fi

# Sanity check: every refresh expects at least one `### Author` heading
# in the generated block. Empty output means jq parsed but emitted
# nothing (auth scope mismatch, filter typo, owner-only repo). Refuse
# to splice in that case so we never wipe the existing block.
if ! grep -q '^### ' "$TMP_JQ"; then
  die "generated block has no '### Author' headings. CREDITS.md unchanged."
fi

{
  printf '%s\n' "$START_MARK"
  cat "$TMP_JQ"
  printf '%s\n' "$END_MARK"
} > "$TMP_BLOCK"

# Splice the new block into CREDITS.md, replacing the old block.
awk -v start="$START_MARK" -v end="$END_MARK" -v block_file="$TMP_BLOCK" '
  BEGIN { in_block = 0; spliced = 0 }
  {
    if ($0 == start) {
      in_block = 1
      while ((getline line < block_file) > 0) print line
      close(block_file)
      spliced = 1
      next
    }
    if (in_block) {
      if ($0 == end) in_block = 0
      next
    }
    print
  }
  END {
    if (!spliced) {
      exit 2
    }
  }
' "$CREDITS_FILE" > "$TMP_OUT"

awk_status=$?
if [[ $awk_status -ne 0 ]]; then
  die "splice failed: start sentinel was not encountered (awk exit $awk_status)"
fi

if cmp -s "$CREDITS_FILE" "$TMP_OUT"; then
  echo "no changes: CREDITS.md is up to date."
  exit 0
fi

if command -v diff >/dev/null 2>&1; then
  echo "diff (current -> regenerated):"
  diff -u "$CREDITS_FILE" "$TMP_OUT" || true
fi

mv "$TMP_OUT" "$CREDITS_FILE"
echo "refresh-credits: CREDITS.md updated."

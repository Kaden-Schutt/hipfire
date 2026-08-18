#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Guard the EXPECTATIONS, not the measurements.
#
# scripts/leanup-ratchets.sh already fails a tree that violates a committed
# threshold. What it cannot see is the threshold being widened in the same
# commit as the code that needed it widened. That has happened here: an agent
# edited scripts/layering.txt until its own inversion became legal and reported
# the checker exiting 0. The Phase 3 scope recorded the mitigation as "cheap and
# social rather than technical -- a change that edits layering.txt,
# leanup-thresholds.txt, or a golden fixture in the same commit as the code it
# governs should be read as a red flag." This is the technical half.
#
# Three ways to weaken an expectation, all caught here:
#   1. raise a `<=` ceiling
#   2. change an `==` invariant
#   3. delete the line entirely, so the metric becomes informational
#
# and a fourth that hides the evidence:
#   4. stop emitting a metric that used to be asserted
#
# A weakening is not forbidden. It must be declared, in a commit message on the
# branch:
#
#     RATCHET-RAISE: <metric> <old> -> <new>, traded for <reason>
#
# Lowering a ceiling, adding a threshold, or adding a metric always passes.
#
# Metric VALUES are reported for context but never fail this check -- size
# metrics like daemon_lines legitimately fluctuate, and the ceilings in
# leanup-thresholds.txt are what govern them.
#
# Usage: scripts/ratchet-diff.sh [BASE_REF]     # default: merge base w/ master
# Exit:  0 no expectation weakened   1 undeclared weakening   2 harness error

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

THRESH="scripts/leanup-thresholds.txt"

BASE_REF="${1:-}"
if [ -z "$BASE_REF" ]; then
  for cand in origin/master master origin/main main; do
    if git rev-parse --verify --quiet "$cand" >/dev/null; then BASE_REF="$cand"; break; fi
  done
fi
[ -n "$BASE_REF" ] || { echo "ratchet-diff: cannot resolve a base ref" >&2; exit 2; }

BASE_SHA="$(git merge-base HEAD "$BASE_REF" 2>/dev/null)"
[ -n "$BASE_SHA" ] || { echo "ratchet-diff: no merge base with $BASE_REF" >&2; exit 2; }

if [ "$BASE_SHA" = "$(git rev-parse HEAD)" ]; then
  echo "ratchet-diff: HEAD is the merge base; nothing to compare."
  exit 0
fi

# `<metric> <op> <value>`, comments and blanks dropped.
parse_thresholds() { grep -E '^[a-z_]+[[:space:]]+(==|<=)[[:space:]]+-?[0-9]+' 2>/dev/null; }

BASE_TH="$(git show "$BASE_SHA:$THRESH" 2>/dev/null | parse_thresholds)"
HEAD_TH="$(parse_thresholds < "$THRESH" 2>/dev/null)"

if [ -z "$BASE_TH" ]; then
  echo "ratchet-diff: $THRESH absent at base $(git rev-parse --short "$BASE_SHA") —"
  echo "              this branch introduces the thresholds; nothing to weaken."
  exit 0
fi

DECLARED="$(git log --format=%B "$BASE_SHA..HEAD" 2>/dev/null \
            | grep -oE '^RATCHET-RAISE:[[:space:]]*[a-z_]+' | awk '{print $2}' | sort -u)"

echo "ratchet-diff: HEAD $(git rev-parse --short HEAD) vs base $(git rev-parse --short "$BASE_SHA") ($BASE_REF)"
echo

violations=0
note() { printf '  %-26s %s\n' "$1" "$2"; }

# 1-3: every metric asserted at base must still be asserted, at least as tightly.
while read -r name op val; do
  [ -n "$name" ] || continue
  head_line="$(echo "$HEAD_TH" | awk -v k="$name" '$1==k {print; exit}')"

  if [ -z "$head_line" ]; then
    if echo "$DECLARED" | grep -qx "$name"; then
      note "$name" "threshold REMOVED (declared)"
    else
      note "$name" "threshold REMOVED  <-- was '$op $val'"
      violations=$((violations + 1))
    fi
    continue
  fi

  head_op="$(echo "$head_line" | awk '{print $2}')"
  head_val="$(echo "$head_line" | awk '{print $3}')"

  weakened=0
  if [ "$op" = "==" ] && [ "$head_op" = "==" ] && [ "$head_val" != "$val" ]; then
    weakened=1
  elif [ "$op" = "==" ] && [ "$head_op" = "<=" ]; then
    weakened=1   # invariant downgraded to a ceiling
  elif [ "$op" = "<=" ] && [ "$head_op" = "<=" ] && [ "$head_val" -gt "$val" ]; then
    weakened=1
  fi

  if [ "$weakened" -eq 1 ]; then
    if echo "$DECLARED" | grep -qx "$name"; then
      note "$name" "$op $val  ->  $head_op $head_val   RAISED (declared)"
    else
      note "$name" "$op $val  ->  $head_op $head_val   <-- WEAKENED"
      violations=$((violations + 1))
    fi
  elif [ "$op $val" != "$head_op $head_val" ]; then
    note "$name" "$op $val  ->  $head_op $head_val   tightened"
  fi
done <<< "$BASE_TH"

# 4: a metric can also be silenced by no longer emitting it.
EMITTED="$(bash scripts/leanup-ratchets.sh --report 2>/dev/null \
           | grep -oE '^[a-z_]+' | sort -u)"
if [ -n "$EMITTED" ]; then
  while read -r name _op _val; do
    [ -n "$name" ] || continue
    if ! echo "$EMITTED" | grep -qx "$name"; then
      note "$name" "asserted but NO LONGER EMITTED  <-- silently unenforced"
      violations=$((violations + 1))
    fi
  done <<< "$HEAD_TH"
fi

[ "$violations" -eq 0 ] && note "(none)" "no expectation weakened"

echo
if [ "$violations" -gt 0 ]; then
  echo "ratchet-diff: FAIL — $violations expectation(s) weakened without a stated trade."
  echo
  echo "  Lowering a ceiling is routine. Raising one, deleting one, or dropping a"
  echo "  metric is a design decision and the commit must say what was traded:"
  echo
  echo "      RATCHET-RAISE: <metric> <old> -> <new>, traded for <reason>"
  exit 1
fi

echo "ratchet-diff: OK — no committed expectation was weakened against $BASE_REF."
exit 0

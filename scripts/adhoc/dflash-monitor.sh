#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# dflash-monitor.sh — poll the worker on origin/dflash and summarize progress.
# Designed to run from the master checkout (or anywhere git can reach origin).
#
# Writes:
#   /tmp/dflash-monitor-state.json   — delta tracking between checks
#   stdout — human-readable summary for the monitor agent to report
#
# Exit codes:
#   0  — worker active, progressing
#   1  — worker stalled (no commits / no progress-file updates in >90 min)
#   2  — worker completed (DFLASH_MORNING_REPORT.md exists)
#   3  — worker blocked (DFLASH_BLOCKED.md exists)
#   10 — repo access error

set -u
STATE="/tmp/dflash-monitor-state.json"
REPO_DIR="${REPO_DIR:-/home/kaden/ClaudeCode/autorocm/hipfire}"
WORKTREE="${WORKTREE:-$REPO_DIR/.worktrees/dflash}"

cd "$REPO_DIR" 2>/dev/null || { echo "repo dir missing: $REPO_DIR"; exit 10; }

# Fetch latest without touching local branches
git fetch origin dflash 2>&1 | tail -2

# What's on origin/dflash
LATEST_SHA=$(git rev-parse origin/dflash 2>/dev/null || echo none)
LATEST_COMMIT=$(git log -1 --format='%h %s' origin/dflash 2>/dev/null || echo none)
COMMITS_AHEAD=$(git rev-list --count origin/master..origin/dflash 2>/dev/null || echo 0)
LAST_COMMIT_TS=$(git log -1 --format='%ct' origin/dflash 2>/dev/null || echo 0)
NOW=$(date +%s)
MINUTES_SINCE_COMMIT=$(( (NOW - LAST_COMMIT_TS) / 60 ))

# Progress file from origin/dflash
PROGRESS=$(git show origin/dflash:docs/DFLASH_PROGRESS.md 2>/dev/null || echo "")
PROGRESS_LEN=$(echo -n "$PROGRESS" | wc -c)
ARCHITECTURE=$(git show origin/dflash:docs/DFLASH_ARCHITECTURE.md 2>/dev/null | wc -c)
BLOCKED_EXISTS=$(git show origin/dflash:docs/DFLASH_BLOCKED.md 2>/dev/null | wc -c)
MORNING_EXISTS=$(git show origin/dflash:docs/DFLASH_MORNING_REPORT.md 2>/dev/null | wc -c)
INJECTIONS=$(git show origin/dflash:docs/DFLASH_INJECTIONS.md 2>/dev/null || echo "")

# Phase status — count how many of Phase 1..8 have been committed
PHASES_DONE=$(git log origin/master..origin/dflash --oneline 2>/dev/null | grep -ciE 'Phase [1-8]' || echo 0)

# New commits since last check (using state file)
if [ -f "$STATE" ]; then
    LAST_SEEN=$(grep -oE '"last_sha":\s*"[^"]+"' "$STATE" | head -1 | cut -d'"' -f4)
else
    LAST_SEEN="$(git rev-parse origin/master)"
fi
NEW_COMMITS=$(git log "${LAST_SEEN}..origin/dflash" --oneline 2>/dev/null | head -20)

# Write state for next check
cat > "$STATE" <<EOF
{
  "timestamp": $NOW,
  "last_sha": "$LATEST_SHA",
  "commits_ahead_of_master": $COMMITS_AHEAD,
  "phases_done": $PHASES_DONE,
  "minutes_since_last_commit": $MINUTES_SINCE_COMMIT,
  "progress_bytes": $PROGRESS_LEN,
  "architecture_bytes": $ARCHITECTURE,
  "blocked_file_exists": $([ "$BLOCKED_EXISTS" -gt 0 ] && echo true || echo false),
  "morning_report_exists": $([ "$MORNING_EXISTS" -gt 0 ] && echo true || echo false)
}
EOF

# Human-readable output
echo "=== dflash monitor @ $(date -u +'%Y-%m-%dT%H:%M:%SZ') ==="
echo "origin/dflash: $LATEST_COMMIT"
echo "commits ahead of master: $COMMITS_AHEAD"
echo "phases touched (Phase N in commits): $PHASES_DONE"
echo "last commit: $MINUTES_SINCE_COMMIT min ago"
echo "DFLASH_PROGRESS.md: $PROGRESS_LEN bytes"
echo "DFLASH_ARCHITECTURE.md: $ARCHITECTURE bytes"

if [ "$MORNING_EXISTS" -gt 0 ]; then
    echo ""
    echo "*** WORKER COMPLETED — morning report exists ***"
    echo ""
    git show origin/dflash:docs/DFLASH_MORNING_REPORT.md 2>/dev/null | head -30
    exit 2
fi

if [ "$BLOCKED_EXISTS" -gt 0 ]; then
    echo ""
    echo "*** WORKER BLOCKED — DFLASH_BLOCKED.md exists ***"
    echo ""
    git show origin/dflash:docs/DFLASH_BLOCKED.md 2>/dev/null | head -40
    exit 3
fi

if [ -n "$NEW_COMMITS" ]; then
    echo ""
    echo "--- NEW COMMITS SINCE LAST CHECK ---"
    echo "$NEW_COMMITS"
fi

if [ -n "$INJECTIONS" ]; then
    echo ""
    echo "--- CURRENT INJECTIONS ---"
    echo "$INJECTIONS" | head -20
fi

if [ "$PROGRESS_LEN" -gt 0 ]; then
    echo ""
    echo "--- LATEST PROGRESS LOG (tail 30) ---"
    git show origin/dflash:docs/DFLASH_PROGRESS.md 2>/dev/null | tail -30
fi

# Stall detection: no commits in 90+ minutes is suspicious
if [ "$MINUTES_SINCE_COMMIT" -gt 90 ]; then
    echo ""
    echo "!!! STALL WARNING: no commits in $MINUTES_SINCE_COMMIT minutes !!!"
    exit 1
fi

exit 0

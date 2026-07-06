#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# test-gpu-lock.sh — regression test for scripts/gpu-lock.sh (no GPU required).
# Covers the failure modes that motivated the flock rewrite:
#   1. stale self-heal   — a kill -9'd holder must NOT block the next acquirer
#   2. busy -> timeout   — a live holder makes the waiter exit non-zero, not hang
#   3. reentrancy        — a nested gate no-ops and never releases the ancestor's lock
#   4. no-rm invariant   — release leaves the file in place; status reports free

set -uo pipefail
cd "$(dirname "$0")/.."
LOCKSH="$PWD/scripts/gpu-lock.sh"

# Isolated lockfile so we never touch a real agent's /tmp/hipfire-gpu.lock.
export HIPFIRE_GPU_LOCKFILE="/tmp/test-gpu-lock.$$"
trap 'rm -f "$HIPFIRE_GPU_LOCKFILE"' EXIT

pass=0 fail=0
ok()  { echo "  PASS: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }

# Wait (up to ~5s) for a holder to actually acquire — gpu_acquire writes the
# agent name into the lockfile only AFTER flock succeeds, so this confirms the
# lock is genuinely held before we try to race/kill it. Avoids a vacuous pass
# on a loaded runner where a fixed sleep fires before acquisition.
await_held() {
    local name="$1" i
    for i in $(seq 1 50); do
        grep -q "$name" "$HIPFIRE_GPU_LOCKFILE" 2>/dev/null && return 0
        sleep 0.1
    done
    return 1
}

# ── 1. stale self-heal ────────────────────────────────────────────────────
echo "[1] stale self-heal (kill -9 holder, next acquire must succeed fast)"
# setsid → own process group so kill -9 takes the holder AND any children
# (otherwise an inherited fd in a child would keep the flock alive).
setsid bash -c "source '$LOCKSH'; gpu_acquire holder1 >/dev/null; sleep 60" &
holder_pid=$!
await_held holder1 || bad "holder1 never acquired (test setup race)"
kill -9 -"$holder_pid" 2>/dev/null || kill -9 "$holder_pid" 2>/dev/null
sleep 1
if timeout 8 bash -c "source '$LOCKSH'; GPU_LOCK_TIMEOUT=5 gpu_acquire holder2 >/dev/null 2>&1; gpu_release >/dev/null 2>&1"; then
    ok "acquired immediately after holder was killed"
else
    bad "second acquire hung/failed — stale lock not self-healed"
fi

# ── 2. busy -> timeout ────────────────────────────────────────────────────
echo "[2] genuine busy holder -> waiter times out with non-zero exit"
setsid bash -c "source '$LOCKSH'; gpu_acquire busy1 >/dev/null; sleep 60" &
busy_pid=$!
await_held busy1 || bad "busy1 never acquired (test setup race)"
timeout 12 bash -c "source '$LOCKSH'; GPU_LOCK_TIMEOUT=2 GPU_POLL_INTERVAL=1 gpu_acquire waiter >/dev/null 2>&1"
rc=$?
kill -9 -"$busy_pid" 2>/dev/null || kill -9 "$busy_pid" 2>/dev/null
if [ "$rc" -eq 2 ]; then
    ok "waiter returned 2 (timeout) instead of hanging"
else
    bad "waiter returned $rc (expected 2)"
fi
sleep 1

# ── 3. reentrancy ─────────────────────────────────────────────────────────
echo "[3] reentrant nested acquire no-ops; child release leaves ancestor lock held"
reentrancy_check() {
    source "$LOCKSH"
    gpu_acquire parent >/dev/null || { echo "OUTER_ACQUIRE_FAILED"; return; }
    # child inherits exported HIPFIRE_GPU_LOCK_OWNER → must no-op + not release
    bash -c "source '$LOCKSH'; gpu_acquire child >/dev/null; rc=\$?; gpu_release >/dev/null; echo CHILD_RC=\$rc"
    echo "STATUS_AFTER_CHILD=$(gpu_status)"
    gpu_release >/dev/null
    echo "STATUS_AFTER_PARENT=$(gpu_status)"
}
out="$(reentrancy_check)"
echo "$out" | sed 's/^/    /'
if echo "$out" | grep -q "CHILD_RC=0" \
   && echo "$out" | grep -q "STATUS_AFTER_CHILD=gpu BUSY" \
   && echo "$out" | grep -q "STATUS_AFTER_PARENT=gpu is free"; then
    ok "child reentrant no-op; ancestor stayed locked; parent released cleanly"
else
    bad "reentrancy semantics wrong"
fi

# ── 4. no-rm invariant ────────────────────────────────────────────────────
echo "[4] release does not delete the lockfile; status reports free"
bash -c "source '$LOCKSH'; gpu_acquire t4 >/dev/null; gpu_release >/dev/null"
if [ -e "$HIPFIRE_GPU_LOCKFILE" ]; then
    st="$(bash -c "source '$LOCKSH'; gpu_status")"
    [ "$st" = "gpu is free" ] && ok "file persists, status='gpu is free'" \
        || bad "file persists but status='$st'"
else
    bad "lockfile was deleted on release (breaks flock mutual exclusion)"
fi

echo
echo "===== $pass passed, $fail failed ====="
[ "$fail" -eq 0 ]

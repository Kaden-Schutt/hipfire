#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gpu-lock.sh — thin shell adapter over `hipfire gpu-lock`.
# Source this in an agent session:  source scripts/gpu-lock.sh
# Then:  gpu_acquire "model-ingestion" && { run tests; gpu_release; }
#
# The lock itself lives in the engine now (crates/hipfire-cli/.../gpu_lock.rs):
# a flock(2)-backed mutex held by a detached holder that watches this shell's
# pid, so the kernel releases on ANY death (kill -9, crash, terminal close) —
# stale locks are structurally impossible. This file only maps the historical
# gpu_acquire/gpu_release/gpu_status shell functions onto the binary and keeps
# process-tree reentrancy (so nested gates don't deadlock on their parent).

# Resolve the hipfire binary: explicit override, then release build, then PATH.
_gpu_lock_bin() {
    if [ -n "${HIPFIRE_BIN:-}" ] && [ -x "${HIPFIRE_BIN}" ]; then
        echo "$HIPFIRE_BIN"; return 0
    fi
    local repo; repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if [ -x "$repo/target/release/hipfire" ]; then
        echo "$repo/target/release/hipfire"; return 0
    fi
    if command -v hipfire >/dev/null 2>&1; then
        echo hipfire; return 0
    fi
    return 1
}

gpu_acquire() {
    local agent_name="${1:?usage: gpu_acquire <agent-name>}"

    # Reentrancy: an ancestor in this process tree already holds the lock.
    if [ -n "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        echo "[gpu-lock] reentrant: already held by ancestor pid=${HIPFIRE_GPU_LOCK_OWNER}"
        return 0
    fi

    local bin; bin="$(_gpu_lock_bin)" || {
        echo "[gpu-lock] FATAL: hipfire binary not found — build with 'cargo build --release -p hipfire-cli' or set HIPFIRE_BIN" >&2
        return 3
    }

    # Verbose contention warning: if the lock is already held by ANOTHER holder,
    # announce it loudly (with the current holder + how to clear a stale lock)
    # BEFORE we block waiting — silent blocking made contention/stale-lock stalls
    # hard to diagnose.
    local lockfile="${HIPFIRE_GPU_LOCKFILE:-/tmp/hipfire-gpu.lock}"
    local waited_from=""
    if [ -s "$lockfile" ]; then
        local holder; holder="$(cat "$lockfile" 2>/dev/null)"
        echo "[gpu-lock] ⚠️  GPU BUSY — '${agent_name}' is WAITING for the lock." >&2
        echo "[gpu-lock] ⚠️  held by: ${holder}" >&2
        echo "[gpu-lock] ⚠️  poll=${GPU_POLL_INTERVAL:-5}s timeout=${GPU_LOCK_TIMEOUT:-default}s." >&2
        echo "[gpu-lock] ⚠️  if this is STALE (holder dead), clear it: rm -f ${lockfile}" >&2
        waited_from="$SECONDS"
    fi

    # The holder watches THIS shell ($$); it auto-releases when this shell exits.
    # GPU_POLL_INTERVAL / GPU_LOCK_TIMEOUT are honored by the binary's defaults.
    "$bin" gpu-lock acquire "$agent_name" --watch-pid "$$" || return $?
    if [ -n "$waited_from" ]; then
        echo "[gpu-lock] ✓ '${agent_name}' acquired after $((SECONDS - waited_from))s wait" >&2
    fi
    export HIPFIRE_GPU_LOCK_OWNER="$$"
    return 0
}

gpu_release() {
    if [ -z "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        echo "[gpu-lock] no lock held"
        return 0
    fi
    if [ "$HIPFIRE_GPU_LOCK_OWNER" != "$$" ]; then
        return 0   # reentrant child — the ancestor owns it, leave it be
    fi
    local bin; bin="$(_gpu_lock_bin)" || return 0
    "$bin" gpu-lock release
    unset HIPFIRE_GPU_LOCK_OWNER
}

gpu_status() {
    local bin; bin="$(_gpu_lock_bin)" || { echo "gpu is free"; return 0; }
    "$bin" gpu-lock status
}

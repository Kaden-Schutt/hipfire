#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gpu-lock.sh — GPU mutex for multi-agent coordination
# Source this in an agent session:  source scripts/gpu-lock.sh
# Then:  gpu_acquire "model-ingestion" && { run tests; gpu_release; }
#
# Backed by flock(1): the lock is held on an open file descriptor, so the
# Linux kernel releases it automatically when the holding process dies for
# ANY reason — kill -9, crash, OOM, agent cancellation, terminal close.
# Stale locks are therefore structurally impossible: there is nothing to
# clean up by hand.
#
# IMPORTANT: never `rm` the lockfile. Unlinking a file that a holder has
# flock'd lets the next acquirer create a *different* inode and lock that
# too — yielding two simultaneous holders. The file is a permanent fixture;
# recovery happens in the kernel, not by deleting the file.

LOCKFILE="${HIPFIRE_GPU_LOCKFILE:-/tmp/hipfire-gpu.lock}"
POLL_INTERVAL="${GPU_POLL_INTERVAL:-5}"          # cadence of "busy" messages (s)
(( POLL_INTERVAL < 1 )) && POLL_INTERVAL=1        # 0 would make flock -w non-blocking → spin
GPU_LOCK_TIMEOUT="${GPU_LOCK_TIMEOUT:-1800}"     # hard cap (s); 0 = wait forever
GPU_LOCK_FD=""                                   # set by gpu_acquire (auto-alloc)

gpu_acquire() {
    local agent_name="${1:?usage: gpu_acquire <agent-name>}"

    if ! command -v flock >/dev/null 2>&1; then
        echo "[gpu-lock] FATAL: flock(1) not found — refusing to run unlocked" >&2
        return 3
    fi

    # Reentrancy: an ancestor in this process tree already holds the lock.
    # Recognise it and no-op so nested gates (e.g. pp-gate) don't deadlock on
    # their own parent's reservation.
    if [ -n "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        echo "[gpu-lock] reentrant: already held by ancestor pid=${HIPFIRE_GPU_LOCK_OWNER}"
        return 0
    fi

    # Open (create if missing) WITHOUT truncating, so we don't clobber a live
    # holder's metadata while we wait. Bash auto-allocates a free fd into
    # GPU_LOCK_FD; it persists in this shell until gpu_release closes it.
    exec {GPU_LOCK_FD}<>"$LOCKFILE"

    local waited=0
    until flock -w "$POLL_INTERVAL" "$GPU_LOCK_FD"; do
        waited=$(( waited + POLL_INTERVAL ))
        local holder
        holder=$(cat "$LOCKFILE" 2>/dev/null || echo unknown)
        if [ "$GPU_LOCK_TIMEOUT" -gt 0 ] && [ "$waited" -ge "$GPU_LOCK_TIMEOUT" ]; then
            echo "[gpu-lock] TIMEOUT after ${waited}s; holder still alive: ${holder}" >&2
            exec {GPU_LOCK_FD}>&-
            GPU_LOCK_FD=""
            return 2
        fi
        echo "[gpu-lock] busy: ${holder} — waited ${waited}s, still waiting…"
    done

    # We hold it. Record metadata (truncate + write via a separate redirect;
    # this does not disturb the flock held on $GPU_LOCK_FD).
    printf '%s pid=%s host=%s acquired=%s\n' \
        "$agent_name" "$$" "$(hostname)" "$(date -Is 2>/dev/null || date)" > "$LOCKFILE"
    export HIPFIRE_GPU_LOCK_OWNER="$$"
    echo "[gpu-lock] acquired by ${agent_name}"
    return 0
}

gpu_release() {
    # Only the process that actually acquired may release. A child that merely
    # inherited HIPFIRE_GPU_LOCK_OWNER from an ancestor must not release it.
    if [ -z "${HIPFIRE_GPU_LOCK_OWNER:-}" ]; then
        echo "[gpu-lock] no lock held"
        return 0
    fi
    if [ "$HIPFIRE_GPU_LOCK_OWNER" != "$$" ]; then
        return 0   # reentrant child — the ancestor owns it, leave it be
    fi

    if [ -n "${GPU_LOCK_FD:-}" ]; then
        flock -u "$GPU_LOCK_FD" 2>/dev/null
        exec {GPU_LOCK_FD}>&- 2>/dev/null
        GPU_LOCK_FD=""
    fi
    unset HIPFIRE_GPU_LOCK_OWNER
    # NB: never rm "$LOCKFILE" — see header.
    echo "[gpu-lock] released"
}

gpu_status() {
    if [ ! -e "$LOCKFILE" ]; then
        echo "gpu is free"
        return 0
    fi
    # Probe with a non-blocking flock on a scratch fd. If we can take it,
    # nobody is holding it — any leftover file content is cosmetic/stale.
    local probe_fd
    exec {probe_fd}<>"$LOCKFILE"
    if flock -n "$probe_fd"; then
        flock -u "$probe_fd"
        exec {probe_fd}>&-
        echo "gpu is free"
    else
        exec {probe_fd}>&-
        echo "gpu BUSY: $(cat "$LOCKFILE" 2>/dev/null)"
    fi
}

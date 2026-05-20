#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gpu-lock.sh — GPU mutex for multi-agent coordination
# Source this in both agent sessions: source gpu-lock.sh
# Then use: gpu_acquire "model-ingestion" && { run tests; gpu_release; }

LOCKFILE="/tmp/hipfire-gpu.lock"
POLL_INTERVAL="${GPU_POLL_INTERVAL:-5}"

gpu_acquire() {
    local agent_name="${1:?usage: gpu_acquire <agent-name>}"

    while true; do
        # Atomic lock attempt: create file only if it doesn't exist
        if (set -o noclobber; echo "${agent_name} agent is using the gpu" > "$LOCKFILE") 2>/dev/null; then
            echo "[gpu-lock] acquired by ${agent_name}"
            return 0
        fi

        # Lock held by someone else — report and wait
        local holder
        holder=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
        echo "[gpu-lock] busy: ${holder} — retrying in ${POLL_INTERVAL}s"
        sleep "$POLL_INTERVAL"
    done
}

gpu_release() {
    if [ -f "$LOCKFILE" ]; then
        local holder
        holder=$(cat "$LOCKFILE" 2>/dev/null)
        rm -f "$LOCKFILE"
        echo "[gpu-lock] released (was: ${holder})"
    else
        echo "[gpu-lock] no lock held"
    fi
}

gpu_status() {
    if [ -f "$LOCKFILE" ]; then
        cat "$LOCKFILE"
    else
        echo "gpu is free"
    fi
}

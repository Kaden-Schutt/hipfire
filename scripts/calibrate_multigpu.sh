#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# calibrate_multigpu.sh — fan out TriAttention sidecar calibration across
# all visible GPUs. Each job owns one GPU via HIP_VISIBLE_DEVICES. If there
# are more models than GPUs, the extras queue up and start as earlier ones
# finish.
#
# Usage:
#   bash calibrate_multigpu.sh \
#       --models model1.mq4,model2.mq4,... \
#       (--corpus FILE | --recipe NAME) \
#       [--max-tokens 1000000] \
#       [--chunk-len 1024] \
#       [--suffix .triattn.bin] \
#       [--sidecar-dir /root/models] \
#       [--log-dir /root/calib_logs]
#
# --recipe NAME auto-builds a corpus via fetch_calibration_corpus.sh.
# Recipes: agentic | agentic_xl | reasoning | chat | blended | all.
# Corpus is cached at /root/calib_corpus_<NAME>.txt for reuse across
# back-to-back calibration waves.
#
# Each model M gets calibrated against --corpus, writing:
#   <sidecar-dir>/$(basename M)<suffix>
#
# Stdout/stderr go to <log-dir>/$(basename M).log per model.
# Exit codes are non-zero if any job fails; summary prints at the end.

set -uo pipefail

MODELS=""
CORPUS=""
RECIPE=""
MAX_TOKENS=1000000
CHUNK_LEN=1024
SUFFIX=".triattn.bin"
SIDECAR_DIR=""
LOG_DIR=/root/calib_logs
BIN="${HIPFIRE_BIN:-/root/hipfire/target/release/examples/triattn_validate}"

while [ $# -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --recipe) RECIPE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --chunk-len) CHUNK_LEN="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --sidecar-dir) SIDECAR_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --help|-h) sed -n '1,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log() { printf '[calibrate-mg] %s\n' "$*"; }

# --recipe NAME auto-builds a corpus via fetch_calibration_corpus.sh into
# /root/calib_corpus_<recipe>.txt and uses it as --corpus. Saves the
# two-step "fetch + then calibrate" gotcha. Mutually exclusive with
# --corpus (explicit path wins; refuse if both given to surface user
# intent).
if [ -n "$RECIPE" ]; then
    if [ -n "$CORPUS" ]; then
        echo "ERROR: --recipe and --corpus are mutually exclusive" >&2
        exit 2
    fi
    CORPUS="/root/calib_corpus_${RECIPE}.txt"
    if [ ! -f "$CORPUS" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
        FETCH="${SCRIPT_DIR}/fetch_calibration_corpus.sh"
        if [ ! -x "$FETCH" ]; then
            echo "ERROR: --recipe given but $FETCH not found/executable" >&2
            exit 2
        fi
        log "recipe=$RECIPE — fetching corpus to $CORPUS via $FETCH"
        "$FETCH" "$CORPUS" --recipe "$RECIPE" || {
            echo "ERROR: fetch_calibration_corpus.sh --recipe $RECIPE failed" >&2
            exit 2
        }
    else
        log "recipe=$RECIPE — reusing cached corpus at $CORPUS"
    fi
fi

if [ -z "$MODELS" ] || [ -z "$CORPUS" ]; then
    echo "ERROR: --models and (--corpus FILE | --recipe NAME) are required" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "ERROR: calibration binary not found at $BIN (override with --bin)" >&2
    exit 2
fi
if [ ! -f "$CORPUS" ]; then
    echo "ERROR: corpus file not found: $CORPUS" >&2
    exit 2
fi

# Detect visible GPUs (rocminfo lists one Agent per GPU + one for the CPU).
N_GPUS=$(/opt/rocm/bin/rocminfo 2>/dev/null | grep -c "Agent " || echo 1)
N_GPUS=$((N_GPUS - 1))
[ "$N_GPUS" -lt 1 ] && N_GPUS=1
log "GPUs visible: $N_GPUS"

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
N_MODELS=${#MODEL_ARR[@]}
log "models queued: $N_MODELS"
log "corpus:        $CORPUS ($(wc -c < "$CORPUS" | awk '{printf "%.1f MB", $1/1024/1024}'))"
log "max-tokens:    $MAX_TOKENS"
log "chunk-len:     $CHUNK_LEN"
log "suffix:        $SUFFIX"

mkdir -p "$LOG_DIR"

# Launch one background job per model, each pinned to GPU (index % N_GPUS).
# GNU Parallel would be nicer but keeping it dependency-free.
# Explicit `=()` initializers — `set -u` rejects expansion of declared-
# but-uninitialized arrays under bash 4.x (unbound variable on
# `${arr[@]}` / `${#arr[@]}`). Bash 5+ usually tolerates it but the
# explicit form is portable and the bug is silent: the launch loop
# never runs, the wait loop never runs, FAILED stays empty, and the
# script reports "all N calibrations finished ok" while having
# launched zero jobs.
declare -A PID_TO_MODEL=()
declare -a RUNNING_PIDS=()
declare -a FAILED=()

launch() {
    local idx=$1
    local model=$2
    local gpu=$((idx % N_GPUS))
    local name
    name=$(basename "$model")
    local sidecar
    if [ -n "$SIDECAR_DIR" ]; then
        sidecar="${SIDECAR_DIR}/${name}${SUFFIX}"
    else
        sidecar="${model}${SUFFIX}"
    fi
    local logf="${LOG_DIR}/${name}.log"
    log "launch [#$idx, GPU=$gpu] $name → $sidecar (log: $logf)"
    HIP_VISIBLE_DEVICES=$gpu HIPFIRE_ROCBLAS_OFF=1 \
        "$BIN" "$model" \
            --corpus "$CORPUS" \
            --max-tokens "$MAX_TOKENS" \
            --chunk-len "$CHUNK_LEN" \
            --sidecar "$sidecar" \
        > "$logf" 2>&1 &
    local pid=$!
    PID_TO_MODEL[$pid]=$name
    RUNNING_PIDS+=($pid)
}

# Launch up to N_GPUS at once; queue the rest as earlier ones finish.
pending_idx=0
while [ $pending_idx -lt $N_MODELS ] && [ ${#RUNNING_PIDS[@]} -lt $N_GPUS ]; do
    launch "$pending_idx" "${MODEL_ARR[$pending_idx]}"
    pending_idx=$((pending_idx + 1))
done

# Drain + refill as jobs complete. (FAILED already initialized above.)
while [ ${#RUNNING_PIDS[@]} -gt 0 ]; do
    # wait -n returns as soon as ANY background job finishes.
    if ! wait -n; then
        :  # swallow; we'll detect per-pid below
    fi
    # Rebuild RUNNING_PIDS with only still-alive procs; record status of
    # freshly-dead ones.
    NEW_RUNNING=()
    for pid in "${RUNNING_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            NEW_RUNNING+=("$pid")
        else
            # reap + capture exit status
            if wait "$pid" 2>/dev/null; then
                log "done    ${PID_TO_MODEL[$pid]} (pid $pid, ok)"
            else
                rc=$?
                log "FAILED  ${PID_TO_MODEL[$pid]} (pid $pid, rc=$rc)"
                FAILED+=("${PID_TO_MODEL[$pid]}")
            fi
            unset PID_TO_MODEL[$pid]
        fi
    done
    RUNNING_PIDS=("${NEW_RUNNING[@]}")
    # Backfill from queue onto freed GPU slots.
    while [ $pending_idx -lt $N_MODELS ] && [ ${#RUNNING_PIDS[@]} -lt $N_GPUS ]; do
        launch "$pending_idx" "${MODEL_ARR[$pending_idx]}"
        pending_idx=$((pending_idx + 1))
    done
done

# Final summary.
log "────────────────────────────────────────────────────"
if [ ${#FAILED[@]} -eq 0 ]; then
    log "all $N_MODELS calibrations finished ok"
    exit 0
else
    log "FAILURES: ${FAILED[*]}"
    log "logs: $LOG_DIR"
    exit 1
fi

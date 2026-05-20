#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# mi300_chain_runner.sh — sequential auto-chain for MI300X training jobs.
#
# Runs jobs back-to-back so the GPU isn't idle between training runs.
# Each job block:
#   - logs to /root/chain_logs/${name}.log
#   - writes status marker /root/chain_status/${name}.{started,done,failed}
#   - on failure, the chain stops (next jobs NOT started)
#
# Usage (on MI300X):
#   nohup bash /root/hipfire/scripts/mi300_chain_runner.sh \
#     > /root/chain_runner.log 2>&1 &
#
# Set SKIP_UNTIL=<name> to resume after a manual kill (skip completed jobs).
#
# After editing this file on the controller, just scp/git-pull + start fresh.

set -uo pipefail
cd /root/hipfire

LOG_DIR=/root/chain_logs
STATUS_DIR=/root/chain_status
mkdir -p "$LOG_DIR" "$STATUS_DIR"

PY=/root/pytorch_env/bin/python3
CORPUS=/root/agentic_corpus.txt
SKIP_UNTIL="${SKIP_UNTIL:-}"

SKIPPING=1
[ -z "$SKIP_UNTIL" ] && SKIPPING=0

# ── jobs ─────────────────────────────────────────────────────────────
# Each job function should:
#   - return 0 on success, non-zero on failure
#   - print [chain-runner] markers for progress

run_if_pending() {
    local name=$1; shift
    if [ "$SKIPPING" = "1" ]; then
        if [ "$name" = "$SKIP_UNTIL" ]; then
            SKIPPING=0
            echo "[chain-runner] resuming at: $name"
        else
            echo "[chain-runner] skipping: $name (--skip-until=$SKIP_UNTIL)"
            return 0
        fi
    fi
    if [ -f "$STATUS_DIR/$name.done" ]; then
        echo "[chain-runner] already done: $name"
        return 0
    fi
    echo "[chain-runner] starting: $name"
    date -Is > "$STATUS_DIR/$name.started"
    rm -f "$STATUS_DIR/$name.failed"
    "$@" > "$LOG_DIR/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        date -Is > "$STATUS_DIR/$name.done"
        echo "[chain-runner] finished: $name (rc=0)"
    else
        date -Is > "$STATUS_DIR/$name.failed"
        echo "[chain-runner] FAILED:   $name (rc=$rc)"
        echo "[chain-runner] tail of $LOG_DIR/$name.log:"
        tail -20 "$LOG_DIR/$name.log"
        return $rc
    fi
}

# ── job definitions ──────────────────────────────────────────────────

job_9b_scratch_25k() {
    # 9B target is ~2.4× 4B compute → ~0.7 step/s → 25k steps ≈ 10hr.
    # Grad ckpt on target to fit 9B bf16 + 5-layer draft + activations.
    PYTHONUNBUFFERED=1 "$PY" -u scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-9B \
        --corpus "$CORPUS" \
        --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
        --steps 25000 --ckpt-every 2500 --log-every 250 \
        --lr 5e-5 --warmup 500 \
        --loss-gamma 3.0 \
        --match-zlab-arch \
        --grad-ckpt-target \
        --out /root/dflash_9b_scratch_25k
}

job_9b_scratch_convert() {
    ./target/release/dflash_convert \
        --input /root/dflash_9b_scratch_25k \
        --output /root/dflash_9b_scratch_25k.hfq \
        --mq4
}

job_4b_scratch_25k() {
    # 4B is faster (~1.7 step/s) so 25k steps ≈ 4hr. Keep steps parity with 9B
    # to avoid biased comparison.
    PYTHONUNBUFFERED=1 "$PY" -u scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-4B \
        --corpus "$CORPUS" \
        --seq-len 4096 --batch-size 1 --masked-blocks-per-seq 4 \
        --steps 25000 --ckpt-every 2500 --log-every 250 \
        --lr 5e-5 --warmup 500 \
        --loss-gamma 3.0 \
        --match-zlab-arch \
        --out /root/dflash_4b_scratch_25k
}

job_4b_scratch_convert() {
    ./target/release/dflash_convert \
        --input /root/dflash_4b_scratch_25k \
        --output /root/dflash_4b_scratch_25k.hfq \
        --mq4
}

# ── sidecar cal jobs (gated on draft success — require MQ4 target to exist) ──

sidecar_cal() {
    local tgt=$1 sc_out=$2
    # triattn_validate takes the model as a POSITIONAL arg, not --model.
    # Output filename is controlled via --sidecar.
    /root/hipfire/target/release/examples/triattn_validate \
        "$tgt" \
        --sidecar "$sc_out" \
        --corpus "$CORPUS" \
        --max-tokens 1000000 \
        --chunk-len 1024
}

job_4b_sidecar_cal() {
    # Uses the existing MQ4 target (not the new draft — sidecars are for
    # the TARGET's attention, not the draft). Produces
    # qwen3.5-4b.mq4.triattn.bin which pairs with any 4B draft.
    local tgt=/root/models/qwen3.5-4b.mq4
    [ -f "$tgt" ] || { echo "no target at $tgt — stage with stage_models.sh first" >&2; return 3; }
    sidecar_cal "$tgt" "${tgt}.triattn.agentic.bin"
}

job_9b_sidecar_cal() {
    local tgt=/root/models/qwen3.5-9b.mq4
    [ -f "$tgt" ] || { echo "no target at $tgt — stage with stage_models.sh first" >&2; return 3; }
    sidecar_cal "$tgt" "${tgt}.triattn.agentic.bin"
}

# ── main chain ───────────────────────────────────────────────────────

echo "[chain-runner] starting chain at $(date -Is)"
echo "[chain-runner] skip_until='${SKIP_UNTIL:-<none>}'"

run_if_pending 9b_scratch_25k           job_9b_scratch_25k           || exit 1
run_if_pending 9b_scratch_convert       job_9b_scratch_convert       || exit 1
run_if_pending 9b_sidecar_cal           job_9b_sidecar_cal           || exit 1

# PAUSE POINT 2026-04-19: user wants to test 9B draft+sidecar before running
# 4B. If testing validates the pipeline, the plan is to spin up an 8× cluster
# for Qwen3.6-A3B and others — making a 1× 4B scratch run redundant.
# Uncomment the 4B block below only if 9B test fails and we want a 4B fallback.
# run_if_pending 4b_scratch_25k           job_4b_scratch_25k           || exit 1
# run_if_pending 4b_scratch_convert       job_4b_scratch_convert       || exit 1
# run_if_pending 4b_sidecar_cal           job_4b_sidecar_cal           || exit 1

echo "[chain-runner] ALL JOBS DONE at $(date -Is)"

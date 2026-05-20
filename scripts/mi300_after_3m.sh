#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# After the current after-36 chain finishes (3M extended cals), validate
# the draft-training POC end-to-end, then start a real multi-hour training
# run so MI300X stays saturated.

set -uo pipefail

export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/root/.cargo/bin:$PATH
export HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIPFIRE_FP16=0
export HF_HOME=/root/hf_cache

cd /root/hipfire

log() {
    echo "[$(date -u +%FT%TZ)] after-3m: $*" | tee -a /root/chain_status.log
}

wait_file_ready() {
    while [ ! -f "$1" ]; do sleep 30; done
}

# Hand off after the last step in the 3M chain
log "start — waiting for 3.6-A3B 3M sidecar"
wait_file_ready /root/models/qwen3.6-35b-a3b.mq4.triattn.3m.bin
log "all cal chain work done; moving to draft training POC"

source /root/pytorch_env/bin/activate

# ── POC smoke (100 steps on 4B target) ──────────────────────────────
log "POC smoke — 100 steps, 4B target, draft_layers=2 batch=1 seq=512"
python3 scripts/dflash_train_poc.py \
    --target-repo Qwen/Qwen3.5-4B \
    --draft-layers 2 --block-size 16 \
    --seq-len 512 --batch-size 1 \
    --lr 3e-4 --steps 100 --warmup 10 \
    --corpus /root/wikitext_calib.txt \
    --out /root/poc_smoke \
    --ckpt-every 50 --log-every 10 \
    > /root/poc_smoke.log 2>&1

if grep -Eq "^\[done\]" /root/poc_smoke.log; then
    log "POC smoke PASS — starting real draft training (10K steps)"

    # ── Real training — 5-layer draft matching the reference default ──
    python3 scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-4B \
        --draft-layers 5 --block-size 16 \
        --seq-len 1024 --batch-size 4 \
        --lr 3e-4 --steps 10000 --warmup 500 \
        --corpus /root/wikitext_calib.txt \
        --out /root/draft_b16_4b_10k \
        --ckpt-every 1000 --log-every 50 \
        > /root/draft_b16_4b_10k.log 2>&1 || log "WARN: real training returned non-zero"
    log "real B=16 training run finished"

    # ── B=32 variant to unlock task #121 ──────────────────────────────
    python3 scripts/dflash_train_poc.py \
        --target-repo Qwen/Qwen3.5-4B \
        --draft-layers 5 --block-size 32 \
        --seq-len 2048 --batch-size 2 \
        --lr 3e-4 --steps 10000 --warmup 500 \
        --corpus /root/wikitext_calib.txt \
        --out /root/draft_b32_4b_10k \
        --ckpt-every 1000 --log-every 50 \
        > /root/draft_b32_4b_10k.log 2>&1 || log "WARN: B=32 training returned non-zero"
    log "real B=32 training run finished"
else
    log "POC smoke FAILED — see /root/poc_smoke.log; skipping long runs"
    tail -30 /root/poc_smoke.log | tee -a /root/chain_status.log
fi

log "after-3m chain complete — MI300X again idle, needs more work"

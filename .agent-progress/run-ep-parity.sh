#!/usr/bin/env bash
set -u
cd /home/bjoern/hipfire/.claude/worktrees/feature+device-mesh || exit 2
LOG=.agent-progress/ep-parity.log; : > "$LOG"; exec >>"$LOG" 2>&1
echo "== ep parity (mesh-driven EP vs production) start $(date -Is) HEAD $(git rev-parse --short HEAD) =="
source scripts/gpu-lock.sh
export GPU_LOCK_TIMEOUT=2400
gpu_acquire "device-mesh-ep-parity" || { echo "lock FAIL"; exit 3; }
echo "-- lock acquired $(date -Is) --"
# tp=1 in-process anchor: runs BOTH production forward_scratch AND the mesh-driven
# EP executor over the same prompt, asserts per-step argmax parity. 16 steps.
HIP_VISIBLE_DEVICES=0 nix develop -c cargo run --release --features deltanet -p hipfire-runtime \
    --example ep_decode_parity -- \
    ~/.hipfire/models/qwen3.6-35b-a3b.mq4 1 16 "The capital of France is"
echo "PARITY exit: $?"
gpu_release
echo "== done $(date -Is) =="

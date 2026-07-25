#!/usr/bin/env bash
set -euo pipefail

CANDIDATE="${1:?usage: certify_head.sh <candidate.mtp> [output-dir]}"
OUT="${2:-$HOME/.hipfire/training/fastmtp-qwen36-a3b-v1/certification}"
TRUNK="${TRUNK:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
SESSION="${SESSION:-$HOME/mv/session_coding.json}"
TAG="${TAG:-qwen3.6:35b-a3b-mq4r}"
GPU="${GPU:-0}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"

if ! command -v hipcc >/dev/null 2>&1; then
    for rocm_bin in /opt/rocm/bin /opt/rocm/core-*/bin; do
        if [[ -x "$rocm_bin/hipcc" ]]; then
            export PATH="$rocm_bin:$PATH"
            break
        fi
    done
fi
command -v hipcc >/dev/null 2>&1 || {
    echo "hipcc was not found in PATH or an installed ROCm bin directory" >&2
    exit 2
}

for path in "$CANDIDATE" "$TRUNK" "$STOCK_MTP" "$SESSION"; do
    [[ -s "$path" ]] || { echo "missing certification input: $path" >&2; exit 2; }
done
mkdir -p "$OUT/stock" "$OUT/candidate"
mkdir -p "$LOCK_ROOT"
export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-${GPU}.lock"
source scripts/gpu-lock.sh
gpu_acquire "fastmtp-certify-gpu${GPU}"
trap gpu_release EXIT
ln -sfn "$TRUNK" "$OUT/stock/model.mq4r"
ln -sfn "$STOCK_MTP" "$OUT/stock/model.mtp"
ln -sfn "$TRUNK" "$OUT/candidate/model.mq4r"
ln -sfn "$CANDIDATE" "$OUT/candidate/model.mtp"

run_serve() {
    local label="$1"
    local model="$2"
    local mtp="$3"
    HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU" \
        python3 scripts/serve_harness.py \
        --model "$model" \
        --tag "$TAG" \
        --kv q8 \
        --mtp "$mtp" \
        --thinking med \
        --max-tokens 4096 \
        --max-seq 32768 \
        --sampling registry \
        --mode session \
        --session "$SESSION" \
        --port 11520 \
        --home "$OUT/home-$label" \
        --serve-log "$OUT/$label.serve.log" \
        --out "$OUT/$label.json"
}

# Same eight sampled multi-turn requests, with AR as the quality/perf floor.
run_serve ar "$OUT/stock/model.mq4r" off
run_serve stock-mtp "$OUT/stock/model.mq4r" on
run_serve candidate-mtp "$OUT/candidate/model.mq4r" on

# The trained sidecar must not perturb the already-certified trunk PM4 route.
HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU" \
HIPFIRE_REPLAY_MANUAL_CAPTURE=1 HIPFIRE_REPLAY_BACKEND=shadow \
    python3 scripts/redline_daemon_harness.py \
    --model "$OUT/candidate/model.mq4r" \
    --skip-prefill \
    --pm4 \
    --kv-mode q8 \
    --shadow-iterations 15 \
    --out "$OUT/redline-shadow.json" \
    --log "$OUT/redline-shadow.log"

python3 scripts/mtp_train/fastmtp35/evaluate_certification.py "$OUT"

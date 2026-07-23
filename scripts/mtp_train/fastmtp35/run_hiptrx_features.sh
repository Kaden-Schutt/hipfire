#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
BIN="${BIN:-target/release/examples/qwen35_mtp_features}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/features}"
TARGET_ROWS_PER_GPU="${TARGET_ROWS_PER_GPU:-25000000}"
WINDOW_ROWS="${WINDOW_ROWS:-128}"
WINDOWS_PER_RECORD="${WINDOWS_PER_RECORD:-2}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-262144}"

[[ -s "$ROOT/clean/train.jsonl" ]] || {
    echo "missing finalized train split: $ROOT/clean/train.jsonl" >&2
    exit 2
}
[[ -s "$ROOT/clean/validation.jsonl" ]] || {
    echo "missing finalized validation split: $ROOT/clean/validation.jsonl" >&2
    exit 2
}
[[ -s "$ROOT/clean/manifest.json" && -s "$MODEL" ]] || {
    echo "clean manifest or deployed MQ4R trunk is missing" >&2
    exit 2
}

mkdir -p "$FEATURE_ROOT/train" "$FEATURE_ROOT/validation" "$ROOT/logs" "$LOCK_ROOT"
cargo build --release -p hipfire-arch-qwen35 --example qwen35_mtp_features

TRUNK_SHA256="${TRUNK_SHA256:-$(sha256sum "$MODEL" | awk '{print $1}')}"
SOURCE_SHA256="${SOURCE_SHA256:-$(sha256sum "$ROOT/clean/manifest.json" | awk '{print $1}')}"
PRODUCER_COMMIT="${PRODUCER_COMMIT:-$(git rev-parse HEAD)}"

run_split() {
    local split="$1"
    local target_rows="$2"
    local pids=()
    for gpu in 0 1 2 3; do
        (
            export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-${gpu}.lock"
            source scripts/gpu-lock.sh
            gpu_acquire "fastmtp-features-${split}-gpu${gpu}"
            trap gpu_release EXIT
            HIP_VISIBLE_DEVICES="$gpu" ROCR_VISIBLE_DEVICES="$gpu" \
                "$BIN" \
                --input "$ROOT/clean/$split.jsonl" \
                --output "$FEATURE_ROOT/$split" \
                --model "$MODEL" \
                --split "$split" \
                --partition-index "$gpu" \
                --partition-count 4 \
                --max-seq 4096 \
                --recursive-steps 3 \
                --window-rows "$WINDOW_ROWS" \
                --windows-per-record "$WINDOWS_PER_RECORD" \
                --rows-per-shard "$ROWS_PER_SHARD" \
                --target-rows "$target_rows" \
                --trunk-sha256 "$TRUNK_SHA256" \
                --source-manifest-sha256 "$SOURCE_SHA256" \
                --producer-git-commit "$PRODUCER_COMMIT" \
                >"$ROOT/logs/features-${split}.gpu${gpu}.stdout" \
                2>"$ROOT/logs/features-${split}.gpu${gpu}.stderr"
        ) &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 ))
}

# Training is explicitly bounded (~100M hidden rows / ~410 GB at dim=2048).
# Validation is small enough to exhaust rather than sample.
run_split train "$TARGET_ROWS_PER_GPU"
run_split validation 1000000000

#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
CONFIG="${CONFIG:-docs/configs/batched-redline-pm4-product.toml}"
BIN="${BIN:-target/release/examples/qwen35_batch_generate}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"

mkdir -p "$ROOT/completions" "$ROOT/logs" "$LOCK_ROOT"

if [[ ! -x "$BIN" ]]; then
    cargo build --release -p hipfire-runtime --example qwen35_batch_generate
fi

run_job() {
    local bucket="$1"
    local profile="$2"
    local max_seq batch temperature top_p top_k presence
    case "$bucket" in
        short) max_seq=1024; batch=100 ;;
        medium|long) max_seq=4096; batch=96 ;;
        *) echo "unknown bucket: $bucket" >&2; return 2 ;;
    esac
    case "$profile" in
        serve) temperature=1.0; top_p=0.95; top_k=20; presence=1.5 ;;
        fastmtp) temperature=0.6; top_p=0.95; top_k=20; presence=0.0 ;;
        greedy) temperature=0.0; top_p=1.0; top_k=0; presence=0.0 ;;
        *) echo "unknown profile: $profile" >&2; return 2 ;;
    esac

    local stem="${bucket}-${profile}"
    local input="$ROOT/jobs/$stem.jsonl"
    [[ -s "$input" ]] || { echo "missing input: $input" >&2; return 2; }

    local pids=()
    for gpu in 0 1 2 3; do
        (
            export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-${gpu}.lock"
            source scripts/gpu-lock.sh
            gpu_acquire "fastmtp-${stem}-gpu${gpu}"
            trap gpu_release EXIT
            "$BIN" "$MODEL" \
                --input "$input" \
                --output "$ROOT/completions/$stem.gpu$gpu.jsonl" \
                --config "$CONFIG" \
                --device "$gpu" \
                --shard-index "$gpu" \
                --shard-count 4 \
                --batch "$batch" \
                --max-seq "$max_seq" \
                --max-new "$(python3 -c \
                    'import json,sys; print(json.loads(open(sys.argv[1]).readline())["max_new_tokens"])' \
                    "$input")" \
                --temperature "$temperature" \
                --top-p "$top_p" \
                --top-k "$top_k" \
                --presence-penalty "$presence" \
                --repeat-penalty 1.0 \
                --frequency-penalty 0.0 \
                --repeat-window 128 \
                --wave-refill \
                >"$ROOT/logs/$stem.gpu$gpu.stdout" \
                2>"$ROOT/logs/$stem.gpu$gpu.stderr"
        ) &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 )) || return 1
}

for bucket in short medium long; do
    for profile in serve fastmtp greedy; do
        run_job "$bucket" "$profile"
    done
done

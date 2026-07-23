#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
CONFIG="${CONFIG:-docs/configs/batched-redline-pm4-product.toml}"
BIN="${BIN:-target/release/examples/qwen35_batch_generate}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"
MAX_JOB_ATTEMPTS="${MAX_JOB_ATTEMPTS:-3}"

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

mkdir -p "$ROOT/completions" "$ROOT/logs" "$LOCK_ROOT"

# Incremental when unchanged, and guarantees `--resume` support after a pull.
cargo build --release -p hipfire-runtime --example qwen35_batch_generate

warm_kernel_cache() {
    local input="$ROOT/jobs/short-greedy.jsonl"
    local warm_input="$ROOT/.kernel-warmup-input.jsonl"
    local warm_output="$ROOT/.kernel-warmup-output.jsonl"
    python3 - "$input" "$warm_input" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as reader, open(
    destination, "w", encoding="utf-8"
) as writer:
    for _, line in zip(range(100), reader):
        row = json.loads(line)
        row["max_new_tokens"] = 1
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
    (
        export HIPFIRE_GPU_LOCKFILE="$LOCK_ROOT/gpu-0.lock"
        source scripts/gpu-lock.sh
        gpu_acquire "fastmtp-kernel-warmup-gpu0"
        trap 'gpu_release; rm -f "$warm_input" "$warm_output"' EXIT
        "$BIN" "$MODEL" \
            --input "$warm_input" \
            --output "$warm_output" \
            --config "$CONFIG" \
            --device 0 \
            --batch 100 \
            --max-seq 1024 \
            --max-new 1 \
            --temperature 0.0 \
            --top-p 1.0 \
            --top-k 0 \
            --presence-penalty 0.0 \
            --repeat-penalty 1.0 \
            --frequency-penalty 0.0 \
            --repeat-window 128 \
            --wave-refill \
            >"$ROOT/logs/kernel-warmup.stdout" \
            2>"$ROOT/logs/kernel-warmup.stderr"
    )
}

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
                --resume \
                >"$ROOT/logs/$stem.gpu$gpu.stdout" \
                2>>"$ROOT/logs/$stem.gpu$gpu.stderr"
        ) &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 )) || return 1
}

warm_kernel_cache

for bucket in short medium long; do
    for profile in serve fastmtp greedy; do
        attempt=1
        until run_job "$bucket" "$profile"; do
            if (( attempt >= MAX_JOB_ATTEMPTS )); then
                echo "job ${bucket}-${profile} failed after $attempt attempts" >&2
                exit 1
            fi
            attempt=$((attempt + 1))
            echo "retrying ${bucket}-${profile} from durable output rows (attempt $attempt/$MAX_JOB_ATTEMPTS)" >&2
        done
    done
done

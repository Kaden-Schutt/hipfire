#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1}"
if (( $# > 0 )); then
    shift
fi
HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
OUTPUT="${OUTPUT:-$HOME/.hipfire/training/fastmtp-qwen36-a3b-v1}"
VENV="${VENV:-$PWD/.venv-rocm}"
LOCK_ROOT="${XDG_RUNTIME_DIR:-$HOME/.cache/hipfire}/hipfire-locks"
VOCAB_MAP="${VOCAB_MAP:-$ROOT/features/vocab-map.json}"
# Never allow a host user-site CPU-only torch to shadow the distro ROCm build.
export PYTHONNOUSERSITE=1

[[ -d "$ROOT/features/train" && -d "$ROOT/features/validation" ]] || {
    echo "Stage 2 features are missing under $ROOT/features" >&2
    exit 2
}
[[ -d "$HF_MODEL" && -s "$STOCK_MTP" ]] || {
    echo "official HF checkpoint or deployed stock .mtp is missing" >&2
    exit 2
}
"$VENV/bin/python" -c 'import torch.distributed.run' || {
    echo "ROCm PyTorch venv cannot import torch.distributed.run: $VENV" >&2
    echo "Bootstrap the ROCm-supported PyTorch environment before Stage 3." >&2
    exit 2
}

mkdir -p "$OUTPUT" "$LOCK_ROOT"
if [[ ! -s "$VOCAB_MAP" ]]; then
    cargo run --release -p hipfire-arch-qwen35 --example mtp_vocab_dump -- \
        "$STOCK_MTP" "$VOCAB_MAP"
fi

# Hold all four physical-GPU locks in this parent while torchrun owns the four
# visible devices. Dynamic FDs remain open through the complete DDP run.
LOCK_FDS=()
for gpu in 0 1 2 3; do
    lock="$LOCK_ROOT/gpu-${gpu}.lock"
    exec {fd}>>"$lock"
    flock -w "${HIPFIRE_GPU_LOCK_TIMEOUT:-3600}" "$fd" || {
        echo "timed out waiting for GPU $gpu lock: $lock" >&2
        exit 1
    }
    printf 'fastmtp-train pid=%s gpu=%s started=%s\n' "$$" "$gpu" "$(date -Is)" >"$lock"
    LOCK_FDS+=("$fd")
done

export PYTHONPATH="$PWD/scripts/mtp_train${PYTHONPATH:+:$PYTHONPATH}"
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ARGS=("$@")
if [[ "${HIPFIRE_FASTMTP_AUTO_RESUME:-1}" != "0" ]]; then
    explicit_resume=0
    for arg in "${TRAIN_ARGS[@]}"; do
        case "$arg" in
            --resume-optimizer|--resume-optimizer=*|--resume-weights|--resume-weights=*)
                explicit_resume=1
                ;;
        esac
    done
    if (( explicit_resume == 0 )); then
        shopt -s nullglob
        optimizer_checkpoints=("$OUTPUT"/step-*.optimizer.pt)
        shopt -u nullglob
        if (( ${#optimizer_checkpoints[@]} > 0 )); then
            latest_optimizer="${optimizer_checkpoints[-1]}"
            latest_weights="${latest_optimizer%.optimizer.pt}.safetensors"
            if [[ ! -s "$latest_weights" ]]; then
                echo "latest optimizer checkpoint has no matching weights: $latest_optimizer" >&2
                exit 2
            fi
            echo "resuming FastMTP training from $latest_weights"
            TRAIN_ARGS+=(
                --resume-weights "$latest_weights"
                --resume-optimizer "$latest_optimizer"
            )
        fi
    fi
fi

"$VENV/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/mtp_train/fastmtp35/train_head.py \
    --features "$ROOT/features/train" \
    --validation-features "$ROOT/features/validation" \
    --hf-model "$HF_MODEL" \
    --vocab-map "$VOCAB_MAP" \
    --output "$OUTPUT" \
    --epochs 3 \
    --micro-batch-size 128 \
    --global-batch-size 512 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.05 \
    --checkpoint-every 500 \
    --eval-every 250 \
    "${TRAIN_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${1:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
OUTPUT="${2:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
VOCAB_MAP="${3:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map.json}"
HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B}"
OVERRIDE="$TRAINING_DIR/final.safetensors"

[[ -s "$OVERRIDE" && -s "$TRAINING_DIR/training-manifest.json" ]] || {
    echo "final checkpoint or training manifest missing under $TRAINING_DIR" >&2
    exit 2
}
[[ -s "$VOCAB_MAP" ]] || {
    echo "compressed-vocabulary map is missing: $VOCAB_MAP" >&2
    exit 2
}
SNAPSHOT="$(find "$HF_MODEL/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
[[ -n "$SNAPSHOT" ]] || SNAPSHOT="$HF_MODEL"

cargo run --release -p hipfire-quantize --bin mtp_extract -- \
    --hf-dir "$SNAPSHOT" \
    --mtp-override "$OVERRIDE" \
    --vocab-sidecar "$VOCAB_MAP" \
    --quant mq4 \
    --output "$OUTPUT" \
    --verbose

sha256sum "$OVERRIDE" "$VOCAB_MAP" "$OUTPUT" >"$OUTPUT.sha256"

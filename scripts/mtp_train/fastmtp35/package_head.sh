#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${1:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
OUTPUT="${2:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
VOCAB_MAP="${3:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map.json}"
HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
OVERRIDE="$TRAINING_DIR/final.safetensors"

[[ -s "$OVERRIDE" && -s "$TRAINING_DIR/training-manifest.json" ]] || {
    echo "final checkpoint or training manifest missing under $TRAINING_DIR" >&2
    exit 2
}
[[ -s "$VOCAB_MAP" ]] || {
    echo "compressed-vocabulary map is missing: $VOCAB_MAP" >&2
    exit 2
}
if [[ "$(realpath -m "$OUTPUT")" == "$(realpath -m "$STOCK_MTP")" ]]; then
    echo "refusing to overwrite the deployed stock sidecar: $STOCK_MTP" >&2
    exit 2
fi
SNAPSHOT="$HF_MODEL"
if [[ -s "$HF_MODEL/refs/main" ]]; then
    candidate="$HF_MODEL/snapshots/$(<"$HF_MODEL/refs/main")"
    [[ -d "$candidate" ]] && SNAPSHOT="$candidate"
else
    shopt -s nullglob
    snapshots=("$HF_MODEL"/snapshots/*)
    shopt -u nullglob
    (( ${#snapshots[@]} == 0 )) || SNAPSHOT="${snapshots[0]}"
fi

cargo run --release -p hipfire-quantize --bin mtp_extract -- \
    --hf-dir "$SNAPSHOT" \
    --mtp-override "$OVERRIDE" \
    --vocab-sidecar "$VOCAB_MAP" \
    --quant mq4 \
    --output "$OUTPUT" \
    --verbose

sha256sum \
    "$OVERRIDE" \
    "$TRAINING_DIR/training-manifest.json" \
    "$VOCAB_MAP" \
    "$OUTPUT" >"$OUTPUT.sha256"

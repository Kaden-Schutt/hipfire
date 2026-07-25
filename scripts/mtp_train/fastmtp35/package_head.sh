#!/usr/bin/env bash
set -euo pipefail

TRAINING_DIR="${1:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
OUTPUT="${2:?usage: package_head.sh <training-dir> <output.mtp> [vocab-map.json]}"
VOCAB_MAP="${3:-$HOME/.hipfire/datasets/fastmtp-qwen36-a3b-v1/features/vocab-map.json}"
HF_MODEL="${HF_MODEL:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B}"
STOCK_MTP="${STOCK_MTP:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mtp}"
MTP_QUANT="${MTP_QUANT:-mixed}"
OVERRIDE="$TRAINING_DIR/final.safetensors"

[[ -s "$OVERRIDE" && -s "$TRAINING_DIR/training-manifest.json" ]] || {
    echo "final checkpoint or training manifest missing under $TRAINING_DIR" >&2
    exit 2
}
[[ -s "$VOCAB_MAP" ]] || {
    echo "compressed-vocabulary map is missing: $VOCAB_MAP" >&2
    exit 2
}
case "$MTP_QUANT" in
    mixed|mq4) ;;
    q8)
        echo "full-Q8 sidecars are not runtime-compatible with routed-MoE MTP; use mixed" >&2
        exit 2
        ;;
    *)
        echo "unsupported MTP_QUANT=$MTP_QUANT (expected mixed or mq4)" >&2
        exit 2
        ;;
esac
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
    --quant "$MTP_QUANT" \
    --output "$OUTPUT" \
    --verbose

sha256sum \
    "$OVERRIDE" \
    "$TRAINING_DIR/training-manifest.json" \
    "$VOCAB_MAP" \
    "$OUTPUT" >"$OUTPUT.sha256"

{
    printf 'schema_version=1\n'
    printf 'quant=%s\n' "$MTP_QUANT"
    printf 'producer_git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'hf_snapshot=%s\n' "$(realpath "$SNAPSHOT")"
    printf 'training_checkpoint=%s\n' "$(realpath "$OVERRIDE")"
    printf 'training_checkpoint_sha256=%s\n' "$(sha256sum "$OVERRIDE" | cut -d " " -f 1)"
    printf 'training_manifest_sha256=%s\n' \
        "$(sha256sum "$TRAINING_DIR/training-manifest.json" | cut -d " " -f 1)"
    printf 'vocab_map_sha256=%s\n' "$(sha256sum "$VOCAB_MAP" | cut -d " " -f 1)"
    printf 'sidecar_sha256=%s\n' "$(sha256sum "$OUTPUT" | cut -d " " -f 1)"
} >"$OUTPUT.provenance"

#!/usr/bin/env bash
# Build one source-derived DeepSeek-V4-Flash dense E8 surgery overlay.
#
# Usage:
#   scripts/reap/build_deepseek4_e8_layer_overlay.sh <layer 0..42>
#
# Environment:
#   CAMPAIGN   campaign root on NAS
#   SOURCE_REV pinned deepseek-ai/DeepSeek-V4-Flash revision
#   QUANT_BIN  release hipfire-quantize binary
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
SOURCE_REV=${SOURCE_REV:-60d8d70770c6776ff598c94bb586a859a38244f1}
QUANT_BIN=${QUANT_BIN:-"$ROOT/target/release/hipfire-quantize"}
LAYER=${1:?usage: $0 <layer 0..42>}

if ! [[ "$LAYER" =~ ^[0-9]+$ ]] || ((LAYER < 0 || LAYER > 42)); then
    echo "layer must be an integer in 0..42, got '$LAYER'" >&2
    exit 2
fi
if [[ ! -x "$QUANT_BIN" ]]; then
    echo "quantizer not found at $QUANT_BIN; build it with:" >&2
    echo "  cargo build --release -p hipfire-quantize --bin hipfire-quantize" >&2
    exit 2
fi

printf -v LAYER_TAG '%02d' "$LAYER"
printf -v SHARD_TAG '%05d' "$((LAYER + 2))"
SOURCE_DIR="$CAMPAIGN/source"
CANDIDATE_DIR="$CAMPAIGN/candidates/layer-$LAYER_TAG"
INPUT_DIR="$CANDIDATE_DIR/source"
SHARD="model-$SHARD_TAG-of-00046.safetensors"

mkdir -p "$SOURCE_DIR" "$CANDIDATE_DIR" "$INPUT_DIR"

hf download deepseek-ai/DeepSeek-V4-Flash \
    config.json tokenizer.json tokenizer_config.json generation_config.json \
    model.safetensors.index.json "$SHARD" \
    --revision "$SOURCE_REV" --local-dir "$SOURCE_DIR"

for metadata in \
    config.json tokenizer.json tokenizer_config.json generation_config.json \
    model.safetensors.index.json; do
    ln -sfn "$SOURCE_DIR/$metadata" "$INPUT_DIR/$metadata"
done
ln -sfn "$SOURCE_DIR/$SHARD" "$INPUT_DIR/$SHARD"

cat >"$CANDIDATE_DIR/reap_plan.json" <<EOF
{
  "model_arch": "deepseek4",
  "num_layers": 43,
  "original_experts": 256,
  "quant_overrides": [
    {
      "layer": $LAYER,
      "role": "attention",
      "tensors": [
        "layers.$LAYER.attn.wq_a.weight",
        "layers.$LAYER.attn.wq_b.weight",
        "layers.$LAYER.attn.wkv.weight",
        "layers.$LAYER.attn.wo_a.weight",
        "layers.$LAYER.attn.wo_b.weight"
      ],
      "tier": "mfp4e8soa"
    },
    {
      "layer": $LAYER,
      "role": "shared_expert",
      "tensors": [
        "layers.$LAYER.ffn.shared_experts.w1.weight",
        "layers.$LAYER.ffn.shared_experts.w2.weight",
        "layers.$LAYER.ffn.shared_experts.w3.weight"
      ],
      "tier": "mfp4e8soa"
    }
  ]
}
EOF

"$QUANT_BIN" \
    --input "$INPUT_DIR" \
    --output "$CANDIDATE_DIR/unused.hfq" \
    --reap-overlay "$CANDIDATE_DIR" \
    --reap-out "$CANDIDATE_DIR/overlay.hfq" \
    --reap-arch deepseek4

sha256sum "$SOURCE_DIR/$SHARD" "$CANDIDATE_DIR/reap_plan.json" \
    "$CANDIDATE_DIR/overlay.hfq"

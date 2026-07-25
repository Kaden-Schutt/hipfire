#!/usr/bin/env bash
# Build one source-derived DeepSeek-V4-Flash MQ2R P2/P3 E8 bucket overlay.
#
# Usage:
#   build_deepseek4_e8_bucket_overlay.sh p2 <layer 2..42>
#   build_deepseek4_e8_bucket_overlay.sh p3-router <layer 0..42>
#   build_deepseek4_e8_bucket_overlay.sh p3-head
#
# Each output contains only the requested incremental bucket. Cumulative
# P2/P3 candidates are assembled deterministically with hfq_overlay_merge.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
SOURCE_REV=${SOURCE_REV:-60d8d70770c6776ff598c94bb586a859a38244f1}
QUANT_BIN=${QUANT_BIN:-"$ROOT/target/release/hipfire-quantize"}
E8_TIER=${E8_TIER:-mfp4e8soa}
CANDIDATE_SUFFIX=${CANDIDATE_SUFFIX:-}
E8_IMATRIX=${E8_IMATRIX:-}
E8_HESSIAN_DIR=${E8_HESSIAN_DIR:-}
BUCKET=${1:?usage: $0 <p2|p3-router|p3-head> [layer]}
LAYER=${2:-}

if [[ ! -x "$QUANT_BIN" ]]; then
    echo "quantizer not found at $QUANT_BIN; build it with:" >&2
    echo "  cargo build --release -p hipfire-quantize --bin hipfire-quantize" >&2
    exit 2
fi

case "$BUCKET" in
    p2)
        if ! [[ "$LAYER" =~ ^[0-9]+$ ]] || ((LAYER < 2 || LAYER > 42)); then
            echo "p2 layer must be an integer in 2..42, got '$LAYER'" >&2
            exit 2
        fi
        printf -v LAYER_TAG '%02d' "$LAYER"
        printf -v SHARD_TAG '%05d' "$((LAYER + 2))"
        CANDIDATE_DIR="$CAMPAIGN/candidates/p2-layer-$LAYER_TAG$CANDIDATE_SUFFIX"
        SHARD="model-$SHARD_TAG-of-00046.safetensors"
        ;;
    p3-router)
        if ! [[ "$LAYER" =~ ^[0-9]+$ ]] || ((LAYER < 0 || LAYER > 42)); then
            echo "p3-router layer must be an integer in 0..42, got '$LAYER'" >&2
            exit 2
        fi
        printf -v LAYER_TAG '%02d' "$LAYER"
        printf -v SHARD_TAG '%05d' "$((LAYER + 2))"
        CANDIDATE_DIR="$CAMPAIGN/candidates/p3-router-layer-$LAYER_TAG$CANDIDATE_SUFFIX"
        SHARD="model-$SHARD_TAG-of-00046.safetensors"
        ;;
    p3-head)
        if [[ -n "$LAYER" ]]; then
            echo "p3-head does not take a layer argument" >&2
            exit 2
        fi
        CANDIDATE_DIR="$CAMPAIGN/candidates/p3-head$CANDIDATE_SUFFIX"
        SHARD="model-00045-of-00046.safetensors"
        ;;
    *)
        echo "unknown bucket '$BUCKET'; expected p2, p3-router, or p3-head" >&2
        exit 2
        ;;
esac

SOURCE_DIR="$CAMPAIGN/source"
INPUT_DIR="$CANDIDATE_DIR/source"
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

case "$BUCKET" in
    p2)
        INDEXER_TENSORS=
        if ((LAYER % 2 == 0)); then
            INDEXER_TENSORS=$(cat <<EOF
,
        "layers.$LAYER.attn.indexer.wq_b.weight",
        "layers.$LAYER.attn.indexer.weights_proj.weight",
        "layers.$LAYER.attn.indexer.compressor.wkv.weight",
        "layers.$LAYER.attn.indexer.compressor.wgate.weight"
EOF
)
        fi
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
        "layers.$LAYER.attn.compressor.wkv.weight",
        "layers.$LAYER.attn.compressor.wgate.weight"$INDEXER_TENSORS
      ],
      "tier": "$E8_TIER"
    }
  ]
}
EOF
        ;;
    p3-router)
        cat >"$CANDIDATE_DIR/reap_plan.json" <<EOF
{
  "model_arch": "deepseek4",
  "num_layers": 43,
  "original_experts": 256,
  "quant_overrides": [
    {
      "layer": $LAYER,
      "role": "router",
      "tensors": ["layers.$LAYER.ffn.gate.weight"],
      "tier": "$E8_TIER"
    }
  ]
}
EOF
        ;;
    p3-head)
        cat >"$CANDIDATE_DIR/reap_plan.json" <<EOF
{
  "model_arch": "deepseek4",
  "num_layers": 43,
  "original_experts": 256,
  "quant_overrides": [
    {
      "layer": 0,
      "role": "lm_head",
      "tensors": ["head.weight"],
      "tier": "$E8_TIER"
    }
  ]
}
EOF
        ;;
esac

if [[ -n "$E8_IMATRIX" ]]; then
    export HIPFIRE_E8_IMATRIX="$E8_IMATRIX"
fi
if [[ -n "$E8_HESSIAN_DIR" ]]; then
    export HIPFIRE_E8_HESSIAN_DIR="$E8_HESSIAN_DIR"
fi

"$QUANT_BIN" \
    --input "$INPUT_DIR" \
    --output "$CANDIDATE_DIR/unused.hfq" \
    --reap-overlay "$CANDIDATE_DIR" \
    --reap-out "$CANDIDATE_DIR/overlay.hfq" \
    --reap-arch deepseek4

sha256sum "$SOURCE_DIR/$SHARD" "$CANDIDATE_DIR/reap_plan.json" \
    "$CANDIDATE_DIR/overlay.hfq"

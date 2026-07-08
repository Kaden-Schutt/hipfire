#!/usr/bin/env bash
# Build the DeepSeek-V4 DSpark draft-module sidecar HFQ for hipfire.
#
# The released `deepseek-ai/DeepSeek-V4-Flash-DSpark` checkpoint ships the DSpark
# draft as a 3-stage MTP chain (tensor prefixes mtp.0/mtp.1/mtp.2) attached to the
# V4 weights. The three stages live ENTIRELY in shards 46,47,48 (those shards are
# 100% mtp tensors) — so only ~11 GB needs downloading, not the full 167 GB.
#
# Stage 0 carries main_proj/main_norm (ingests target hiddens concat of layers
# [40,41,42]); stage 2 carries hc_head/markov_head/confidence_head. block_size=5,
# markov_rank=256, noise_token_id=128799.
#
# The existing deepseek4-q8-mtp quant path handles every DSpark tensor by name
# (main_proj/markov/confidence -> Q8F16, norms/hc -> F16, experts -> MQ2-Lloyd),
# so NO quantizer code change is needed — this is purely a recipe.
#
# Usage: scripts/quantize-dspark.sh [<ckpt_dir>] [<out_file>] [<trunk_file>]
#
# hipfire AUTO-DISCOVERS this sidecar only if it sits next to the deepseek4
# trunk model named `<trunk_stem>-dspark.<trunk_ext>` — e.g. trunk
# `~/.hipfire/models/deepseek-v4-flash.mq2lloyd` -> sidecar
# `~/.hipfire/models/deepseek-v4-flash-dspark.mq2lloyd`. The default <out_file>
# below matches a trunk with that exact name/location; if yours differs, pass
# its path as <trunk_file> and the correct sidecar path is derived for you
# (mirrors `load_dspark` in crates/hipfire-arch-deepseek4/src/arch.rs).
set -euo pipefail

REPO="${HIPFIRE_REPO:-$HOME/hipfire}"
CKPT="${1:-$HOME/dspark-work/ckpt}"
OUT="${2:-$HOME/.hipfire/models/deepseek-v4-flash-dspark.mq2lloyd}"
TRUNK="${3:-}"

# Derive the discovery-matching sidecar path from the trunk model, if given.
if [ -n "$TRUNK" ]; then
  trunk_dir=$(dirname "$TRUNK")
  trunk_base=$(basename "$TRUNK")
  OUT="$trunk_dir/${trunk_base%.*}-dspark.${trunk_base##*.}"
fi

if [ ! -d "$CKPT" ]; then
  echo "checkpoint dir $CKPT not found — download the 3 mtp shards first:"
  echo "  hf download deepseek-ai/DeepSeek-V4-Flash-DSpark \\"
  echo "     model-00046-of-00048.safetensors model-00047-of-00048.safetensors \\"
  echo "     model-00048-of-00048.safetensors config.json tokenizer.json \\"
  echo "     tokenizer_config.json generation_config.json --local-dir $CKPT"
  exit 1
fi

cargo build --release -p hipfire-quantize --bin hipfire-quantize --manifest-path "$REPO/Cargo.toml"

"$REPO/target/release/hipfire-quantize" \
  --input "$CKPT" \
  --output "$OUT" \
  --format deepseek4-q8-mtp \
  --include-prefix mtp. \
  --allow-mq2-lloyd

echo "wrote $OUT"
echo "discovery: place this next to the trunk as <trunk_stem>-dspark.<trunk_ext> so hipfire auto-loads it"
# Verify: cargo run --release -p hipfire-quantize --example hfq_dump -- "$OUT" mtp.2.markov

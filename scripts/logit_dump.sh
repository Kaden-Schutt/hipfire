#!/bin/bash
# Logit divergence diagnostic: run on each GPU, diff the outputs.
# Usage: ./scripts/logit_dump.sh <model.hfq> [output_dir]
#
# Generates 100 tokens with greedy sampling and dumps:
#   - token_sequence.txt: one token ID per line
#   - logit_stats.txt: per-step top-5 logits, entropy, argmax
#
# Run on 5700 XT:  ./scripts/logit_dump.sh models/qwen3.5-9b.q4.hfq logit_dump_gfx1010
# Swap GPU.
# Run on 7900 XTX: ./scripts/logit_dump.sh models/qwen3.5-9b.q4.hfq logit_dump_gfx1100
# Then: diff logit_dump_gfx1010/token_sequence.txt logit_dump_gfx1100/token_sequence.txt

set -euo pipefail

MODEL="${1:?Usage: logit_dump.sh <model.hfq> [output_dir]}"
OUTDIR="${2:-logit_dump_$(date +%s)}"
mkdir -p "$OUTDIR"

echo "=== Logit Dump Diagnostic ==="
echo "Model: $MODEL"
echo "Output: $OUTDIR/"

# Build the diagnostic example if needed
cargo build --release -p hipfire-runtime --features deltanet --example logit_dump 2>&1 | tail -1

# Run it
cargo run --release -p hipfire-runtime --features deltanet --example logit_dump -- \
    "$MODEL" "$OUTDIR" 2>"$OUTDIR/stderr.log"

echo "Done. Files in $OUTDIR/"
echo "To compare: diff -u logit_dump_gfx1010/token_sequence.txt logit_dump_gfx1100/token_sequence.txt"

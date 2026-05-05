#!/bin/bash
# Run MoE expert heatmap profiling on a representative corpus.
#
# Phase 0 of MoE eGPU offload (docs/plans/moe-egpu-offload.prd):
# generate 1k+ tokens across diverse prompts to characterize expert
# hit-rate. Output: a single heatmap CSV in
# /tmp/hipfire-moe-debug/dumps/, plus the analyzer report.
#
# Usage: scripts/run_moe_heatmap_corpus.sh <model.mq4> [tokens_per_prompt=200]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL=${1:?usage: $0 <model.mq4> [tokens_per_prompt]}
TOKS=${2:-200}

if [[ ! -f $MODEL ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

if [[ ! -x ./target/release/examples/daemon ]]; then
  echo "daemon not built; run: cargo build --release -p hipfire-runtime --example daemon" >&2
  exit 1
fi

OUT=/tmp/hipfire-moe-debug/heatmap-corpus-$(date +%s)
mkdir -p "$OUT"
JSONL=$OUT/corpus.jsonl

cat > "$JSONL" << EOF
{"type":"load","model":"$MODEL","params":{"max_seq":4096,"kv_mode":"asym3"}}
{"type":"generate","id":"prose","prompt":"Write a detailed essay about the history and significance of the Roman Empire, covering its founding, expansion, peak, decline, and lasting impact on Western civilization.","temperature":0,"max_tokens":$TOKS}
{"type":"generate","id":"code","prompt":"Implement a Python function that parses a CSV file with mixed types, handles missing values, and groups rows by category. Include error handling and unit tests.","temperature":0,"max_tokens":$TOKS}
{"type":"generate","id":"math","prompt":"Solve step by step: A train leaves city A at 80 km/h heading east. Two hours later, a second train leaves city A at 120 km/h heading east on a parallel track. When and where will the second train catch up?","temperature":0,"max_tokens":$TOKS}
{"type":"generate","id":"chat","prompt":"Explain quantum entanglement to a curious 12-year-old who already knows what an atom is. Use everyday analogies and avoid technical jargon.","temperature":0,"max_tokens":$TOKS}
{"type":"generate","id":"summary","prompt":"Summarize in three paragraphs the key arguments of any book you choose, then offer one specific critique that a reader from a different cultural background might raise.","temperature":0,"max_tokens":$TOKS}
{"type":"unload"}
EOF

echo "Running heatmap corpus on $MODEL (5 prompts × $TOKS tokens)..."
HIPFIRE_MOE_EXPERT_HEATMAP=1 \
  ./target/release/examples/daemon < "$JSONL" \
  > "$OUT/corpus.out" 2> "$OUT/corpus.err"

DUMP=$(grep -oE "/tmp/hipfire-moe-debug/dumps/heatmap-[^ ]+\.csv" "$OUT/corpus.err" | tail -1)
if [[ -z $DUMP || ! -f $DUMP ]]; then
  echo "ERROR: heatmap dump missing — see $OUT/corpus.err" >&2
  tail -20 "$OUT/corpus.err"
  exit 1
fi

echo
echo "Heatmap: $DUMP"
echo "Run dir: $OUT"
echo
python3 scripts/analyze_moe_heatmap.py "$DUMP"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Overnight on-paper H-Neurons run (arXiv 2512.01797) on Llama-3.1-8B.
#
# The resident model generates its OWN answer to each TriviaQA question; each
# answer is gold-labeled (correct vs hallucinated) via normalized_aliases; the
# classes are balanced; only the kept set is CETT-captured; the hipfire-hneurons
# L1 probe is fit and the H-Neuron set reported. This is the on-paper protocol
# (self-generated answers), unlike the cross-model shipped-response demo.
#
# Manual launch (defaults target ~300/class, ~5 h on gfx1103 release build):
#   ./scripts/hneurons_overnight.sh
#
# Override via env, e.g. a quick self-test:
#   HN_GEN_LIMIT=12 HN_LIMIT=3 HN_GEN_MAX_TOKENS=16 ./scripts/hneurons_overnight.sh
#
# Prereqs (persistent artifacts, produced once):
#   python3 scripts/hneurons_colnorms.py <hf_model_dir> \
#       ~/.hipfire/hneurons/llama31-8b-down-colnorms.bin
#   python3 scripts/hneurons_triviaqa_prep.py --triviaqa-dir <rc.nocontext> \
#       --split train      --limit 4000 --out ~/.hipfire/hneurons/tqa_train.jsonl
#   python3 scripts/hneurons_triviaqa_prep.py --triviaqa-dir <rc.nocontext> \
#       --split validation --limit 4000 --out ~/.hipfire/hneurons/tqa_test.jsonl
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HN_DIR="${HN_DIR:-$HOME/.hipfire/hneurons}"
MODEL="${HN_MODEL:-$HOME/.hipfire/models/Llama-3.1-8B-Instruct.q8f16.hfq}"
COLNORMS="${HN_COLNORMS:-$HN_DIR/llama31-8b-down-colnorms.bin}"
GEN_TRAIN="${HN_GEN_TRAIN:-$HN_DIR/tqa_train.jsonl}"
GEN_TEST="${HN_GEN_TEST:-$HN_DIR/tqa_test.jsonl}"
GEN_LIMIT="${HN_GEN_LIMIT:-900}"
LIMIT="${HN_LIMIT:-300}"
GEN_MAX_TOKENS="${HN_GEN_MAX_TOKENS:-16}"
L1="${HN_L1:-1e-3}"
OUT_DIR="${HN_OUT_DIR:-$HN_DIR/runs}"
DAEMON="${HN_DAEMON:-$ROOT/target/release/hipfire-daemon}"
PROBE="${HN_PROBE:-$ROOT/target/release/hipfire-hneurons-probe}"

die() { echo "hneurons_overnight: $*" >&2; exit 1; }

for f in "$MODEL" "$COLNORMS" "$GEN_TRAIN" "$GEN_TEST"; do
    [ -f "$f" ] || die "missing required file: $f (see prereqs in this script's header)"
done

# Release binaries (much faster host-side capture loop than debug).
if [ ! -x "$DAEMON" ]; then
    echo "== building release hipfire-daemon =="
    cargo build --release -p hipfire-daemon --bin hipfire-daemon
fi
if [ ! -x "$PROBE" ]; then
    echo "== building release hipfire-hneurons-probe =="
    cargo build --release -p hipfire-steer-harness --bin hipfire-hneurons-probe
fi

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/hneurons-$STAMP.log"

echo "== H-Neurons overnight run =="
echo "  model:     $MODEL"
echo "  colnorms:  $COLNORMS"
echo "  gen:       train=$GEN_TRAIN test=$GEN_TEST"
echo "  params:    gen_limit=$GEN_LIMIT limit(per-class)=$LIMIT gen_max_tokens=$GEN_MAX_TOKENS l1=$L1"
echo "  log:       $LOG"
echo "  (the probe spawns its own daemon, which takes the GPU lock; ensure the"
echo "   GPU is free — 'hipfire lock status')"
echo

"$PROBE" \
    --daemon "$DAEMON" \
    --model "$MODEL" \
    --colnorms "$COLNORMS" \
    --gen-train "$GEN_TRAIN" \
    --gen-test "$GEN_TEST" \
    --gen-limit "$GEN_LIMIT" \
    --limit "$LIMIT" \
    --gen-max-tokens "$GEN_MAX_TOKENS" \
    --l1 "$L1" \
    2>&1 | tee "$LOG"

echo
echo "== done: results in $LOG =="

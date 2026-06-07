#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# run_hfim_blend_ab.sh — the hipfire-native "agentic-KLD" harness.
#
# Proves (or falsifies) whether a proper hipfire-native blend beats unsloth on a
# DEPLOYMENT-REPRESENTATIVE, calibration-NEUTRAL metric — not wikitext-KLD (which
# under-sells a chat/agentic blend) and not mq4 task accuracy (near-ceiling on
# robust tasks). The metric here is KLD vs the f32 oracle on a HELD-OUT agentic
# slice (disjoint from calib by construction; see build_hfim_corpora.sh).
#
# Pipeline (all reuse existing binaries — no new kernels):
#   1. build_kld_ref_native  on the held-out agentic eval  -> agentic kldref
#   2. collect_imatrix_native on the blend calib (capped)   -> blend HFIM
#   3. hipfire-quantize --awq --imatrix-hfim                -> quant E (blend)
#   4. eval_hipfire E / D(bartowski) / A(unsloth) vs the agentic kldref
#
# D and A are the quants from the earlier barto run (already on box). E is new.
# The decision: does E (blend HFIM) beat A (unsloth) on agentic KLD, and by how
# much vs D (raw bartowski HFIM)?
#
# Usage:
#   bash scripts/run_hfim_blend_ab.sh            (run on mi300, GPU 0)
#   env knobs: ORACLE, BF16, CALIB, EVAL, NCTX(512), CALIB_CHUNKS(256),
#              EVAL_CHUNKS(32), ALPHA(0.5)
set -u
. /root/.cargo/env 2>/dev/null
cd /root/hipfire

ORACLE="${ORACLE:-/workspace/qwen3.6-27b-oracle.hfq}"
BF16="${BF16:-/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
CALIB="${CALIB:-/workspace/hfim-blend-calib.txt}"
EVAL="${EVAL:-/workspace/hfim-agentic-eval.txt}"
NCTX="${NCTX:-512}"
CALIB_CHUNKS="${CALIB_CHUNKS:-256}"
EVAL_CHUNKS="${EVAL_CHUNKS:-32}"
ALPHA="${ALPHA:-0.5}"

KLDREF=/workspace/agentic-eval-27b.kldref.bin
HFIM=/workspace/qwen3.6-27b-blend.hfim
QE=/workspace/qwen3.6-27b.mq4-awq-blend
QD=/workspace/qwen3.6-27b.mq4-awq-barto      # from the barto run
QA=/workspace/qwen3.6-27b.mq4-awq-ggufim     # unsloth GGUF imatrix

Q=./target/release/hipfire-quantize
COLL=./target/release/examples/collect_imatrix_native
KB=./target/release/examples/build_kld_ref_native
EV=./target/release/examples/eval_hipfire
L=/workspace/hfim_blend; mkdir -p "$L"
ts(){ date +%H:%M:%S; }
echo "BLEND AB START $(date)" > "$L/run.log"
md5sum "$CALIB" "$EVAL" >> "$L/run.log" 2>/dev/null

echo "[$(ts)] STEP1 agentic kldref (held-out agentic eval, ${EVAL_CHUNKS}ch)" | tee -a "$L/run.log"
$KB --model "$ORACLE" --slice "$EVAL" --output "$KLDREF" --n-ctx "$NCTX" --max-chunks "$EVAL_CHUNKS" > "$L/1_kldref.log" 2>&1
echo "[$(ts)] kldref rc=$? size=$(stat -c%s "$KLDREF" 2>/dev/null)" | tee -a "$L/run.log"

echo "[$(ts)] STEP2 blend HFIM gen (calib, ${CALIB_CHUNKS}ch)" | tee -a "$L/run.log"
$COLL --model "$ORACLE" --slice "$CALIB" --output "$HFIM" --n-ctx "$NCTX" --max-chunks "$CALIB_CHUNKS" > "$L/2_hfim.log" 2>&1
echo "[$(ts)] hfim rc=$? $(tail -1 "$L/2_hfim.log")" | tee -a "$L/run.log"

echo "[$(ts)] STEP3 quant E (blend HFIM)" | tee -a "$L/run.log"
$Q --input "$BF16" --output "$QE" --format mq4 --awq --awq-alpha "$ALPHA" --imatrix-hfim "$HFIM" > "$L/3_quantE.log" 2>&1
echo "[$(ts)] quantE rc=$? size=$(stat -c%s "$QE" 2>/dev/null) awq=$(grep -c '    AWQ:' "$L/3_quantE.log")" | tee -a "$L/run.log"

eval_one(){ # tag quant
  local TAG="$1" M="$2"
  [ -f "$M" ] || { echo "[$(ts)] $TAG MISSING $M" | tee -a "$L/run.log"; return; }
  $EV --model "$M" --ref "$KLDREF" --output "$L/eval_${TAG}.kldseq" --kv-mode f32 --scoring-mode per-token --max-chunks "$EVAL_CHUNKS" > "$L/4_eval_${TAG}.log" 2>&1
  echo "[$(ts)] eval $TAG: $(grep 'slice-mean KLD' "$L/4_eval_${TAG}.log" | tail -1)" | tee -a "$L/run.log"
}
echo "[$(ts)] STEP4 eval E/D/A vs agentic kldref" | tee -a "$L/run.log"
eval_one E "$QE"
eval_one D "$QD"
eval_one A "$QA"

echo "===== AGENTIC-KLD RESULT (vs f32 oracle, held-out agentic) =====" | tee -a "$L/run.log"
echo "E blend-HFIM : $(grep 'slice-mean KLD' "$L/4_eval_E.log" 2>/dev/null | tail -1)" | tee -a "$L/run.log"
echo "D barto-HFIM : $(grep 'slice-mean KLD' "$L/4_eval_D.log" 2>/dev/null | tail -1)" | tee -a "$L/run.log"
echo "A unsloth    : $(grep 'slice-mean KLD' "$L/4_eval_A.log" 2>/dev/null | tail -1)" | tee -a "$L/run.log"
echo "BLEND AB DONE $(date)" | tee -a "$L/run.log"
touch "$L/DONE"

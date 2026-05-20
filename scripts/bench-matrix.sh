#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Full TurboQuant benchmark matrix.
# Runs all KV configs on Qwen3-8B, produces publication-quality table.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=${1:-/tmp/qwen3-8b-hfq4.hfq}
INFER=$REPO/target/release/examples/infer_hfq
OUT=$REPO/bench/results/turbo-matrix.txt

# GPU banner detection — replaces the prior hardcoded "RX 5700 XT".
. "$(dirname "$0")/_detect-gpu.sh"

mkdir -p "$REPO/bench/results"

SHORT="Hello"
HARD="Explain the three laws of thermodynamics with mathematical formulations and real-world examples"

run() {
    local label=$1 flags=$2 prompt=$3 maxgen=${4:-91}
    local R
    R=$(timeout 60 $INFER "$MODEL" $flags --maxgen "$maxgen" "$prompt" 2>&1) || true
    local GEN=$(echo "$R" | grep "=== Done:" | grep -oP '[\d.]+(?= tok/s)')
    local NTOK=$(echo "$R" | grep "=== Done:" | grep -oP '\d+(?= tokens in)')
    local TEXT=$(echo "$R" | sed -n '/^Generating/,/^===/p' | grep -v "^Generating\|^===" | tr '\n' ' ')
    local REPEAT=$(echo "${TEXT:0:200}" | tr ' ' '\n' | sort | uniq -c | sort -rn | head -1 | awk '{print $1}')
    local COH="ok"; [ -z "$GEN" ] && COH="FAIL"; [ "${REPEAT:-0}" -gt 10 ] && COH="BAD"
    echo "| $label | ${GEN:-FAIL} | ${NTOK:-0} | $COH |"
}

echo "=== TurboQuant Benchmark Matrix ===" | tee "$OUT"
echo "Model: $(basename $MODEL)" | tee -a "$OUT"
echo "GPU: $(hipfire_gpu_banner)" | tee -a "$OUT"
echo "Date: $(date '+%Y-%m-%d %H:%M')" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# Warm all kernel caches
echo "Warming kernels..." >&2
for f in "--q8kv" "--fp32kv" "--turbo2" "--turbo3" "--turbo4" "--turbo2 --adaptive" "--turbo4 --adaptive"; do
    timeout 30 $INFER "$MODEL" $f --maxgen 3 "Hi" >/dev/null 2>&1 || true
done
echo "Kernels warm." >&2

echo "### Short generation (\"Hello\", 91 tokens)" | tee -a "$OUT"
echo "| Config | tok/s | tokens | quality |" | tee -a "$OUT"
echo "|--------|-------|--------|---------|" | tee -a "$OUT"
run "Q8 KV" "--q8kv" "$SHORT" 91 | tee -a "$OUT"
run "FP32 KV" "--fp32kv" "$SHORT" 91 | tee -a "$OUT"
run "turbo2" "--turbo2" "$SHORT" 91 | tee -a "$OUT"
run "turbo2+adaptive" "--turbo2 --adaptive" "$SHORT" 91 | tee -a "$OUT"
run "turbo3" "--turbo3" "$SHORT" 91 | tee -a "$OUT"
run "turbo4" "--turbo4" "$SHORT" 91 | tee -a "$OUT"
run "turbo4+adaptive" "--turbo4 --adaptive" "$SHORT" 91 | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "### Hard prompt (thermodynamics, 128 tokens)" | tee -a "$OUT"
echo "| Config | tok/s | tokens | quality |" | tee -a "$OUT"
echo "|--------|-------|--------|---------|" | tee -a "$OUT"
run "Q8 KV" "--q8kv" "$HARD" 128 | tee -a "$OUT"
run "turbo2" "--turbo2" "$HARD" 128 | tee -a "$OUT"
run "turbo3" "--turbo3" "$HARD" 128 | tee -a "$OUT"
run "turbo4" "--turbo4" "$HARD" 128 | tee -a "$OUT"
run "turbo4+adaptive" "--turbo4 --adaptive" "$HARD" 128 | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "### Summary" | tee -a "$OUT"
echo "| Config | Compression | Short tok/s | Hard tok/s | Quality |" | tee -a "$OUT"
echo "|--------|-------------|-------------|------------|---------|" | tee -a "$OUT"
echo "(manual: fill from above)" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "=== Done ===" | tee -a "$OUT"

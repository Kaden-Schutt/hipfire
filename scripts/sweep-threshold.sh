#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Björn Bösel
# hipfire — see LICENSE and NOTICE in the project root.

# Sweep MMQ screening thresholds and compare token output vs WMMA baseline.
# Reports which thresholds produce byte-identical token sequences.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/rocm-env.sh"

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
EXE="target/release/examples/daemon"

SYSTEM=$(python3 -c "import json; print(json.dumps(open('benchmarks/prompts/tool_call_system.txt').read()))")
PROMPT=$(python3 -c "import json; print(json.dumps('You are helping me debug a C program. The file /tmp/fibonacci.c contains a recursive implementation of the Fibonacci sequence, but it has a bug that causes incorrect results for inputs greater than 10. I need you to read the file, identify the bug, explain what is wrong, and suggest a fix. Please also consider whether the implementation could be improved for performance using memoization or an iterative approach. Start by reading the file contents.'))")

INPUT_FILE="/tmp/sweep_input_$$.jsonl"
cat > "$INPUT_FILE" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${PROMPT},"temperature":0.0,"max_tokens":180,"repeat_penalty":1.05,"system":${SYSTEM}}
{"type":"unload"}
JL

extract_tokens() {
    grep -a '"type":"token"' | python3 -c '
import sys, json
toks = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        if d.get("type") == "token":
            toks.append(d.get("text",""))
    except: pass
print("".join(toks), end="")
'
}

echo "Model: $MODEL"
echo ""

# Baseline: WMMA only
echo -n "Baseline (WMMA): "
HIPFIRE_MMQ=0 timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | extract_tokens > /tmp/sweep_baseline_$$.txt
BASELINE_MD5=$(md5sum /tmp/sweep_baseline_$$.txt | cut -d' ' -f1)
BASELINE_LEN=$(wc -c < /tmp/sweep_baseline_$$.txt)
echo "md5=$BASELINE_MD5 len=${BASELINE_LEN}B"
echo "  text: $(head -c 150 /tmp/sweep_baseline_$$.txt)"
echo ""

# MMQ no screening
echo -n "MMQ (no screen): "
HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=0 timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | extract_tokens > /tmp/sweep_mmq_raw_$$.txt
RAW_MD5=$(md5sum /tmp/sweep_mmq_raw_$$.txt | cut -d' ' -f1)
RAW_LEN=$(wc -c < /tmp/sweep_mmq_raw_$$.txt)
MATCH="NO"
if [ "$RAW_MD5" = "$BASELINE_MD5" ]; then MATCH="YES"; fi
echo "md5=$RAW_MD5 len=${RAW_LEN}B match=$MATCH"
echo ""

# Sweep thresholds
echo "=== Threshold sweep ==="
echo ""
printf "%-12s  %-34s  %-6s  %s\n" "threshold" "md5" "match" "len"
echo "-------------------------------------------------------------------"

for T in 0.20 0.15 0.12 0.10 0.08 0.06 0.05 0.04 0.03 0.02 0.01; do
    HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN_THRESHOLD=$T timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | extract_tokens > /tmp/sweep_t${T}_$$.txt
    T_MD5=$(md5sum /tmp/sweep_t${T}_$$.txt | cut -d' ' -f1)
    T_LEN=$(wc -c < /tmp/sweep_t${T}_$$.txt)
    MATCH="NO"
    if [ "$T_MD5" = "$BASELINE_MD5" ]; then MATCH="YES"; fi
    printf "%-12s  %-34s  %-6s  %s\n" "$T" "$T_MD5" "$MATCH" "${T_LEN}B"
done

echo ""
echo "Baseline md5: $BASELINE_MD5"

# Cleanup
rm -f /tmp/sweep_*_$$.txt "$INPUT_FILE"

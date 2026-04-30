#!/usr/bin/env bash
# Compare WMMA-only vs MMQ vs MMQ+screening on a tool-call prompt.
set -uo pipefail

export LD_LIBRARY_PATH="/nix/store/b1k6ma9wh4qrncw3vdbf34iayx7hfwdw-clr-7.2.2/lib:${LD_LIBRARY_PATH:-}"

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
EXE="target/release/examples/daemon"
ANALYZE="scripts/analyze_daemon_output.py"

SYSTEM=$(python3 -c "import json; print(json.dumps(open('benchmarks/prompts/tool_call_system.txt').read()))")
PROMPT=$(python3 -c "import json; print(json.dumps('You are helping me debug a C program. The file /tmp/fibonacci.c contains a recursive implementation of the Fibonacci sequence, but it has a bug that causes incorrect results for inputs greater than 10. I need you to read the file, identify the bug, explain what is wrong, and suggest a fix. Please also consider whether the implementation could be improved for performance using memoization or an iterative approach. Start by reading the file contents.'))")

INPUT_FILE="/tmp/mmq_compare_input_$$.jsonl"
cat > "$INPUT_FILE" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${PROMPT},"temperature":0.0,"max_tokens":180,"repeat_penalty":1.05,"system":${SYSTEM}}
{"type":"unload"}
JL

echo "Model: $MODEL"
echo ""

echo "=== 1. Baseline (WMMA only) ==="
HIPFIRE_MMQ=0 timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | python3 "$ANALYZE"
echo ""

echo "=== 2. MMQ (no screening) ==="
HIPFIRE_MMQ=1 timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | python3 "$ANALYZE"
echo ""

echo "=== 3. MMQ + screening ==="
HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1 timeout 120 "$EXE" < "$INPUT_FILE" 2>/dev/null | python3 "$ANALYZE"
echo ""

rm -f "$INPUT_FILE"

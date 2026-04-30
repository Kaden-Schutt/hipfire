#!/usr/bin/env bash
# Check whether MMQ kernels are actually dispatched during inference.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/rocm-env.sh"
export HIPFIRE_MMQ=1
export HIPFIRE_PROFILE=1

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
EXE="target/release/examples/daemon"
SYSTEM_FILE="benchmarks/prompts/tool_call_system.txt"

system_text=$(python3 -c "import sys,json; print(json.dumps(open(sys.argv[1]).read()))" "$SYSTEM_FILE")

# Use a long prompt to force prefill batch >= 128
LONG_PROMPT="You are helping me debug a C program. The file /tmp/fibonacci.c contains a recursive implementation of the Fibonacci sequence, but it has a bug that causes incorrect results for inputs greater than 10. I need you to read the file, identify the bug, explain what is wrong, and suggest a fix. Please also consider whether the implementation could be improved for performance using memoization or an iterative approach. Start by reading the file contents."

prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$LONG_PROMPT")

cat <<JL | timeout 120 "$EXE" 2>&1 | tee /tmp/mmq_check_out.log | grep -iE "mmq|batch|prefill|quantize_q8" | head -30
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":60,"repeat_penalty":1.05,"system":${system_text}}
{"type":"unload"}
JL

echo ""
echo "--- prefill info ---"
grep '"type":"done"' /tmp/mmq_check_out.log || echo "(no done line)"
echo "--- done ---"

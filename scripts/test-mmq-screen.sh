#!/usr/bin/env bash
# Test MMQ screening fix end-to-end on a tool-call prompt.
# Runs three daemon invocations and checks for <|im_start|> leakage.
set -uo pipefail

export LD_LIBRARY_PATH="/nix/store/b1k6ma9wh4qrncw3vdbf34iayx7hfwdw-clr-7.2.2/lib:${LD_LIBRARY_PATH:-}"

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
EXE="target/release/examples/daemon"
SYSTEM_FILE="benchmarks/prompts/tool_call_system.txt"

if [ ! -f "$EXE" ]; then
    echo "Building daemon..."
    cargo build --release --features deltanet --example daemon 2>/dev/null
fi

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    exit 1
fi

system_text=$(python3 -c "import sys,json; print(json.dumps(open(sys.argv[1]).read()))" "$SYSTEM_FILE")

# Long prompt to force prefill batch >= 128 tokens
PROMPT="You are helping me debug a C program. The file /tmp/fibonacci.c contains a recursive implementation of the Fibonacci sequence, but it has a bug that causes incorrect results for inputs greater than 10. I need you to read the file, identify the bug, explain what is wrong, and suggest a fix. Please also consider whether the implementation could be improved for performance using memoization or an iterative approach. Start by reading the file contents."

prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")

run_test() {
    local label="$1"
    shift
    local out_file="/tmp/mmq_screen_test_out_$$.log"

    cat > /tmp/mmq_screen_input_$$.jsonl <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":180,"repeat_penalty":1.05,"system":${system_text}}
{"type":"unload"}
JL

    echo "=== $label ==="
    echo "  env: $*"
    env "$@" timeout 120 "$EXE" < /tmp/mmq_screen_input_$$.jsonl > "$out_file" 2>&1
    local ec=$?

    # Extract generated text
    local text
    text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))' 2>/dev/null || echo "(no tokens)")

    local n_tokens
    n_tokens=$(grep -ac '"type":"token"' "$out_file" 2>/dev/null || echo 0)

    # Get prefill info
    local done_line
    done_line=$(grep -a '"type":"done"' "$out_file" | head -1)
    local pp_tokens pp_toks
    pp_tokens=$(echo "$done_line" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(d.get("prefill_tokens","?"))' 2>/dev/null || echo "?")
    pp_toks=$(echo "$done_line" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print(f"{d.get(\"prefill_tok_s\",0):.1f}")' 2>/dev/null || echo "?")

    local im_start_leaks
    im_start_leaks=$(printf '%s' "$text" | grep -oE '<\|im_start\|>' | wc -l | tr -d ' ')

    local has_tool_call="no"
    if printf '%s' "$text" | grep -qE '<tool_call>'; then
        has_tool_call="yes"
    fi

    # Check for MMQ screen messages
    local screen_msgs
    screen_msgs=$(grep -c "MMQ screen:" "$out_file" 2>/dev/null || echo 0)

    echo "  prefill: ${pp_tokens} tokens @ ${pp_toks} tok/s"
    echo "  output: ${n_tokens} tokens"
    echo "  <|im_start|> leaks: $im_start_leaks"
    echo "  <tool_call> emitted: $has_tool_call"
    echo "  MMQ screen messages: $screen_msgs"
    echo "  text: ${text:0:300}"

    if [ "${im_start_leaks:-0}" -gt 0 ]; then
        echo "  RESULT: FAIL — ChatML corruption detected"
    else
        echo "  RESULT: OK"
    fi
    echo
    rm -f /tmp/mmq_screen_input_$$.jsonl "$out_file"
}

echo "Model: $MODEL"
echo "Prompt: (long debugging prompt, ~100 words)"
echo

# 1. Baseline: no MMQ (WMMA only)
run_test "Baseline (WMMA only)" HIPFIRE_MMQ=0

# 2. MMQ enabled, no screening
run_test "MMQ (no screening)" HIPFIRE_MMQ=1

# 3. MMQ enabled, with screening
run_test "MMQ + screening" HIPFIRE_MMQ=1 HIPFIRE_MMQ_SCREEN=1

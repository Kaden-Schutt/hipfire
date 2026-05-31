#!/usr/bin/env bash
# Soak test for the qwen35 grammar-guided decoder.
#
# Runs many tool-call requests against the live daemon and verifies
# EVERY response carries a parseable JSON tool_call. The Pi turn-12
# attractor (`<|im_start|>` inside the `<tool_call>` body) is rare on
# short sessions but accumulates probability with long context — this
# script tries to surface any latent regression by hammering the path.
#
# Companion to:
#   - cargo test -p hipfire-arch-qwen35 grammar   (42 unit tests, including
#     direct Pi turn-12 attractor reproduction with synthetic logits)
#   - scripts/test-qwen35-grammar.sh              (single-shot integration)
#
# Usage:
#   bash scripts/test-qwen35-grammar-soak.sh [iterations]
# Default iterations: 30. Each iteration is one short tool-call request.
#
# What's checked per iteration:
#   1. HTTP request returns 200
#   2. .choices[0].finish_reason == "tool_calls"
#   3. .choices[0].message.tool_calls[0].function.name == "bash"
#   4. .choices[0].message.tool_calls[0].function.arguments parses as JSON
#   5. Parsed arguments has the expected `command` field
#
# Failure mode this catches: model emits the Pi turn-12 attractor body
# (NOT JSON) anywhere across the iterations. Pass = all iterations OK.

set -uo pipefail
PORT=11435
MODEL="${MODEL:-qwen3.6-35b-a3b.mq4}"
ITERS="${1:-30}"
TOOLS='[{"type":"function","function":{"name":"bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'

fail=0
ok=0
declare -a FAIL_REASONS
declare -a OK_ARGS
declare -a PROMPTS=(
    "Use the bash tool to echo Hello"
    "Run via bash: ls /tmp"
    "Bash tool: echo \$HOME"
    "Call bash with: uname -a"
    "Via bash: date"
    "Bash: cat /etc/hostname"
    "Run bash to print pwd"
    "Use bash: whoami"
)

# Verify daemon is reachable.
if ! ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
    echo "FAIL: daemon not listening on :$PORT"
    exit 2
fi

echo "soak: $ITERS iterations on $MODEL (port $PORT)"
echo ""

for i in $(seq 1 "$ITERS"); do
    prompt="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
    body=$(jq -c -n --argjson t "$TOOLS" --arg p "$prompt" '{
        model: "'$MODEL'", messages: [{role:"user", content:$p}], tools: $t,
        stream: false, max_tokens: 80, temperature: 0,
        chat_template_kwargs: {enable_thinking: false}
    }')
    res=$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' -d "$body" 2>&1) || {
        FAIL_REASONS+=("iter $i: curl failed")
        fail=$((fail + 1))
        continue
    }

    fr=$(echo "$res" | jq -r '.choices[0].finish_reason // "missing"')
    if [ "$fr" != "tool_calls" ]; then
        FAIL_REASONS+=("iter $i: finish_reason=$fr (expected tool_calls)")
        fail=$((fail + 1))
        continue
    fi
    n_tc=$(echo "$res" | jq -r '.choices[0].message.tool_calls | length // 0')
    if [ "$n_tc" -lt 1 ]; then
        FAIL_REASONS+=("iter $i: no tool_calls in response")
        fail=$((fail + 1))
        continue
    fi
    tc_name=$(echo "$res" | jq -r '.choices[0].message.tool_calls[0].function.name // "null"')
    if [ "$tc_name" != "bash" ]; then
        FAIL_REASONS+=("iter $i: tool name=$tc_name (expected bash)")
        fail=$((fail + 1))
        continue
    fi
    tc_args=$(echo "$res" | jq -r '.choices[0].message.tool_calls[0].function.arguments // "null"')
    if ! echo "$tc_args" | jq -e . >/dev/null 2>&1; then
        # The classic Pi turn-12 failure mode lands here.
        FAIL_REASONS+=("iter $i: arguments not JSON: $tc_args")
        fail=$((fail + 1))
        continue
    fi
    cmd=$(echo "$tc_args" | jq -r '.command // "null"')
    if [ "$cmd" = "null" ] || [ -z "$cmd" ]; then
        FAIL_REASONS+=("iter $i: arguments has no .command field: $tc_args")
        fail=$((fail + 1))
        continue
    fi
    OK_ARGS+=("$tc_args")
    ok=$((ok + 1))
    printf "."
    if (( i % 50 == 0 )); then echo ""; fi
done
echo ""
echo ""
echo "soak result: $ok / $((ok + fail)) passed"

if [ "$fail" -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for r in "${FAIL_REASONS[@]}"; do
        echo "  $r"
    done
    exit 1
fi
echo "ALL OK"

#!/usr/bin/env bash
# Integration test for the qwen35 grammar-guided decoder.
#
# Verifies the grammar matcher (crates/hipfire-arch-qwen35/src/grammar.rs)
# is active and constraining qwen35's `<tool_call>` body when tools are
# present in the request, on BOTH the non-dflash AR path and the
# dflash spec-decode path.
#
# Companion to the in-crate unit tests at
# `crates/hipfire-arch-qwen35/src/grammar.rs` mod tests — those directly
# exercise the Pi turn-12 attractor and the mask logic that prevents it:
#
#   - reproduces_pi_turn_12_attractor_without_mask     (proof of failure)
#   - grammar_mask_blocks_pi_turn_12_attractor          (proof of fix)
#   - grammar_mask_prevents_full_pi_attractor_sequence  (full sequence)
#   - dflash_path_post_validation_catches_attractor     (dflash strategy)
#   - full_legal_tool_call_round_trip                   (happy-path control)
#   - grammar_disabled_skips_constraint                 (off-switch control)
#
# Run those with: cargo test -p hipfire-arch-qwen35 grammar
#
# Prereqs:
#   - Daemon launched with model loadable for tool-call requests.
#   - Default model qwen3.6-35b-a3b.mq4 covers the non-dflash path.
#   - qwen3.6-27b.mq4 + its dflash sidecar covers the dflash path
#     (requires dflash_mode=on in ~/.hipfire/config.json).
#   - Grammar is on by default; disable with HIPFIRE_QWEN35_GRAMMAR=0
#     in the daemon env at launch time for A/B.
#
# What's checked:
#   T1: non-dflash request — model emits valid <tool_call>{json}</tool_call>;
#       parsed tool_calls is non-empty; finish_reason="tool_calls"; the
#       arguments string parses as JSON (the Pi turn-12 failure produces
#       NON-JSON `<|im_start|>assistant "..."}}` here).
#   T2: dflash request (qwen3.6-27b.mq4) — same checks. Verifies the
#       dflash post-acceptance grammar validation either lets a valid
#       tool_call through OR catches a violation + force-resets.

set -uo pipefail
PORT=11435
TOOLS='[{"type":"function","function":{"name":"bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'

run_check() {
    local model="$1"
    local label="$2"
    echo "=== $label (model=$model) ==="
    local res
    res=$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d "$(jq -c -n --argjson t "$TOOLS" '{
            model: "'$model'", messages: [{role:"user", content:"Use the bash tool to run: echo Hello"}], tools: $t,
            stream: false, max_tokens: 80, temperature: 0,
            chat_template_kwargs: {enable_thinking: false}
        }')")

    local fr n_tc tc_name tc_args
    fr=$(echo "$res" | jq -r '.choices[0].finish_reason')
    n_tc=$(echo "$res" | jq -r '.choices[0].message.tool_calls | length // 0')
    tc_name=$(echo "$res" | jq -r '.choices[0].message.tool_calls[0].function.name // "null"')
    tc_args=$(echo "$res" | jq -r '.choices[0].message.tool_calls[0].function.arguments // "null"')

    echo "  finish_reason=$fr n_tool_calls=$n_tc name=$tc_name args=$tc_args"

    [ "$fr" = "tool_calls" ] || { echo "  FAIL: finish_reason=$fr"; return 1; }
    [ "$n_tc" -ge 1 ] || { echo "  FAIL: expected >=1 tool_call, got $n_tc"; return 1; }
    [ "$tc_name" = "bash" ] || { echo "  FAIL: tool name=$tc_name"; return 1; }
    if ! echo "$tc_args" | jq -e . >/dev/null 2>&1; then
        echo "  FAIL: tool arguments not valid JSON: $tc_args"
        return 1
    fi
    echo "  PASS"
    return 0
}

# T1: non-dflash path (35B-A3B doesn't have a dflash sidecar locally).
run_check "qwen3.6-35b-a3b.mq4" "T1 non-dflash path"

# T2: dflash path. Requires dflash_mode=on globally and the 27b dflash
# sidecar (qwen36-27b-dflash-mq4.hfq) in ~/.hipfire/models/. The CLI's
# `draft` auto-discovery wires it up at model load.
DFLASH_MODE=$(jq -r '.dflash_mode // "off"' ~/.hipfire/config.json 2>/dev/null)
if [ "$DFLASH_MODE" = "on" ]; then
    run_check "qwen3.6-27b.mq4" "T2 dflash path"
else
    echo "=== T2 dflash path - skipped (dflash_mode=$DFLASH_MODE in config) ==="
    echo "  hint: jq '. + {dflash_mode: \"on\"}' ~/.hipfire/config.json > /tmp/c && mv /tmp/c ~/.hipfire/config.json"
fi

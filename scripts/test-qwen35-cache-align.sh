#!/usr/bin/env bash
# Integration test for the qwen35 + dflash cache alignment work
# (limitations #1-3 from the parser-alignment review).
#
# What it verifies:
#   T1 (dflash on, temp=0):
#     - dflash decode emits a structured `tool_calls` JSONL event
#       (CLI relays as OpenAI `tool_calls` SSE chunks, surface as
#       `message.tool_calls`).
#     - dflash writes the asst turn to `asst_turn_cache` (visible as
#       `[qwen-cache store dflash]` in serve.log when
#       `HIPFIRE_QWEN_CACHE_TRACE=1` is set).
#     - dflash done event carries `finish_reason: "tool_calls"`.
#     - `conversation_tokens` baked with FULL prompt+decode (so the
#       next non-dflash turn can LCP off it).
#   T2 (routed through qwen35 NON-dflash via temp>1e-6, which skips
#       the `temp <= 1e-6` dflash gate):
#     - asst-turn cache lookup must hit the fingerprint dflash stored
#       on T1 — proves the store is byte-compatible with what the
#       qwen35 lookup hashes (parsers are aligned end-to-end).
#
# Prereqs:
#   - Daemon must be loadable with qwen3.6-27b.mq4 (its dflash sidecar
#     qwen36-27b-dflash-mq4.hfq is auto-discovered next to it).
#   - dflash must be enabled. Either set `dflash_mode: "on"` in
#     ~/.hipfire/config.json globally, or per-model. Default is "off".
#   - Daemon launched with HIPFIRE_QWEN_CACHE_TRACE=1 so the dflash
#     cache-store log line is visible (the script greps it for proof).
#
# Quick setup:
#   jq '. + {dflash_mode: "on"}' ~/.hipfire/config.json > /tmp/c && mv /tmp/c ~/.hipfire/config.json
#   scripts/serve-restart.sh 11435 --kill-only
#   HIPFIRE_QWEN_CACHE_TRACE=1 setsid bun cli/index.ts serve 0.0.0.0 11435 >~/.hipfire/serve.log 2>&1 & disown
#   bash scripts/test-qwen35-cache-align.sh
set -uo pipefail
PORT=11435
MODEL="qwen3.6-27b.mq4"
TOOLS='[{"type":"function","function":{"name":"bash","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]'
LOG=~/.hipfire/serve.log

echo "=== T1 (dflash on, temp=0) ==="
RES1=$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "$(jq -c -n --argjson t "$TOOLS" '{
      model: "'$MODEL'", messages: [{role:"user", content:"Use the bash tool to run: echo Hello"}], tools: $t,
      stream: false, max_tokens: 80, temperature: 0,
      chat_template_kwargs: {enable_thinking: false}
  }')")
echo "$RES1" | jq -c '{usage: .usage, finish_reason: .choices[0].finish_reason, content: (.choices[0].message.content // null), n_tool_calls: (.choices[0].message.tool_calls | length // 0), first_tool: (.choices[0].message.tool_calls[0].function // null)}'

# 1a. Check dflash actually ran (daemon trace).
if ! grep -q '\[qwen-cache store dflash\]' "$LOG"; then
  echo "FAIL: no [qwen-cache store dflash] entry — dflash store didn't run"
  exit 1
fi
echo "OK: dflash cache store fired"

# 1b. Check finish_reason in T1.
FR1=$(echo "$RES1" | jq -r '.choices[0].finish_reason')
[ "$FR1" = "tool_calls" ] || { echo "FAIL: T1 finish_reason=$FR1 (expected tool_calls)"; exit 1; }
echo "OK: T1 finish_reason=tool_calls"

# 1c. Check structured tool_calls were emitted.
TC=$(echo "$RES1" | jq -c '.choices[0].message.tool_calls[0]')
if [ -z "$TC" ] || [ "$TC" = "null" ]; then
  echo "FAIL: no tool_call in T1 response"
  exit 1
fi
echo "OK: T1 emitted structured tool_calls"

# Build T2 history with the echoed-back asst turn + tool result.
TC_ID=$(echo "$TC" | jq -r '.id')
T2_MSGS=$(jq -c -n --arg c "$(echo "$RES1" | jq -r '.choices[0].message.content // ""')" --argjson tc "$TC" --arg tid "$TC_ID" '[
  {role:"user", content:"Use the bash tool to run: echo Hello"},
  {role:"assistant", content:($c // null), tool_calls:[$tc]},
  {role:"tool", tool_call_id:$tid, content:"Hello"}
]')

echo "=== T2 (non-dflash route via temp>1e-6, expect HIT on dflash-stored fp) ==="
RES2=$(curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d "$(jq -c -n --argjson m "$T2_MSGS" --argjson t "$TOOLS" '{
      model: "'$MODEL'", messages: $m, tools: $t,
      stream: false, max_tokens: 30, temperature: 0.0001,
      chat_template_kwargs: {enable_thinking: false}
  }')")
echo "$RES2" | jq -c '{usage: .usage, finish_reason: .choices[0].finish_reason, content: (.choices[0].message.content // null)[:60]}'

CACHED=$(echo "$RES2" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0')
if [ "$CACHED" -gt 0 ]; then
  echo "PASS: T2 cached_tokens=$CACHED — dflash store hit on the qwen35 lookup"
else
  echo "FAIL: T2 cache miss (cached_tokens=$CACHED)"
  exit 1
fi

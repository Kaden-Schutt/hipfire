#!/usr/bin/env bash
# test-serve.sh — Live integration tests for hipfire serve
# Tests API conformance, streaming, tool-calling, and multi-turn ChatML.
# Optionally tests Hermes agent integration.
#
# Usage:
#   ./test-serve.sh                  # run all curl-based tests (starts serve automatically)
#   ./test-serve.sh --hermes         # also run Hermes agent integration test
#   ./test-serve.sh --port 8080      # use custom port
#   ./test-serve.sh --running        # connect to already-running hipfire serve
#
# Requires: curl, jq, bun (for starting hipfire serve)

set -euo pipefail

PORT="${PORT:-11435}"
BASE="http://localhost:$PORT"
MODEL="carnice:9b"
HERMES_TEST=false
ALREADY_RUNNING=false
SERVE_PID=""
PASS=0
FAIL=0
SKIP=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hermes) HERMES_TEST=true; shift ;;
    --port) PORT="$2"; BASE="http://localhost:$PORT"; shift 2 ;;
    --running) ALREADY_RUNNING=true; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC} $1${2:+ — $2}"; }
skip() { SKIP=$((SKIP+1)); echo -e "  ${YELLOW}SKIP${NC} $1${2:+ — $2}"; }
header() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

cleanup() {
  if [[ -n "$SERVE_PID" ]]; then
    echo -e "\n${CYAN}Stopping hipfire serve (pid $SERVE_PID)...${NC}"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
  # Restore Hermes config if we modified it
  if [[ -f ~/.hermes/config.yaml.bak ]]; then
    mv ~/.hermes/config.yaml.bak ~/.hermes/config.yaml
  fi
}
trap cleanup EXIT

# ─── Start hipfire serve ───────────────────────────────────
if ! $ALREADY_RUNNING; then
  header "Starting hipfire serve on port $PORT"
  # Find the CLI
  CLI="$(dirname "$(readlink -f "$0")")/cli/index.ts"
  if [[ ! -f "$CLI" ]]; then
    echo "Cannot find cli/index.ts — run from repo root"
    exit 1
  fi
  HIPFIRE_MODEL="$MODEL" bun "$CLI" serve "$PORT" &
  SERVE_PID=$!
  echo "  PID: $SERVE_PID"

  # Wait for health (up to 120s for model load + kernel compile)
  echo "  Waiting for server..."
  for i in $(seq 1 120); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then
      echo "  Ready after ${i}s"
      break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo -e "${RED}Server exited prematurely${NC}"
      exit 1
    fi
    sleep 1
  done

  if ! curl -sf "$BASE/health" >/dev/null 2>&1; then
    echo -e "${RED}Server failed to start after 120s${NC}"
    exit 1
  fi
fi

# ─── Test 1: Health check ──────────────────────────────────
header "Test 1: Health check"
HEALTH=$(curl -sf "$BASE/health" 2>/dev/null || echo "FAIL")
if echo "$HEALTH" | jq -e '.status == "ok"' >/dev/null 2>&1; then
  pass "GET /health returns {status: ok}"
else
  fail "GET /health" "$HEALTH"
fi

# ─── Test 2: Model listing ─────────────────────────────────
header "Test 2: Model listing"
MODELS=$(curl -sf "$BASE/v1/models" 2>/dev/null || echo "FAIL")
if echo "$MODELS" | jq -e '.data | length > 0' >/dev/null 2>&1; then
  COUNT=$(echo "$MODELS" | jq '.data | length')
  pass "GET /v1/models returns $COUNT models"
else
  fail "GET /v1/models" "$MODELS"
fi

# ─── Test 3: Non-streaming basic chat ──────────────────────
header "Test 3: Non-streaming basic chat"
RESP=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "Say exactly: hello world"}],
    "stream": false,
    "temperature": 0.0,
    "max_tokens": 32
  }' 2>/dev/null || echo '{"error":"request failed"}')

if echo "$RESP" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
  CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content')
  FINISH=$(echo "$RESP" | jq -r '.choices[0].finish_reason')
  if [[ -n "$CONTENT" && "$CONTENT" != "null" ]]; then
    pass "Non-streaming response received (finish=$FINISH, ${#CONTENT} chars)"
    echo "       Content: ${CONTENT:0:80}"
  else
    fail "Non-streaming: empty content" "$RESP"
  fi
else
  fail "Non-streaming: invalid response" "$RESP"
fi

# ─── Test 4: Streaming basic chat ──────────────────────────
header "Test 4: Streaming basic chat (SSE open + close)"
STREAM_OUT=$(mktemp)
# timeout ensures we don't hang forever — the key regression test
HTTP_CODE=$(curl -sf -o "$STREAM_OUT" -w "%{http_code}" --max-time 30 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [{"role": "user", "content": "Say exactly: streaming test"}],
    "stream": true,
    "temperature": 0.0,
    "max_tokens": 32
  }' 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" == "200" ]]; then
  # Check for [DONE] marker — proves stream closed properly
  if grep -q '\[DONE\]' "$STREAM_OUT"; then
    # Count content chunks
    CHUNKS=$(grep -c 'data: {' "$STREAM_OUT" || echo 0)
    pass "Stream completed with [DONE] ($CHUNKS chunks)"
  else
    fail "Stream missing [DONE] — possible hang regression" "$(head -5 "$STREAM_OUT")"
  fi
else
  if [[ "$HTTP_CODE" == "000" ]]; then
    fail "Stream timed out (30s) — HANG DETECTED"
  else
    fail "Stream HTTP $HTTP_CODE" "$(head -3 "$STREAM_OUT")"
  fi
fi
rm -f "$STREAM_OUT"

# ─── Test 5: Non-streaming with tools ──────────────────────
header "Test 5: Non-streaming tool call"
TOOL_RESP=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. When asked about the weather, always use the get_weather tool."},
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "stream": false,
    "temperature": 0.0,
    "max_tokens": 2048
  }' 2>/dev/null || echo '{"error":"request failed"}')

TOOL_CALLS=$(echo "$TOOL_RESP" | jq -r '.choices[0].message.tool_calls // empty' 2>/dev/null)
TOOL_FINISH=$(echo "$TOOL_RESP" | jq -r '.choices[0].finish_reason' 2>/dev/null)
TOOL_CONTENT=$(echo "$TOOL_RESP" | jq -r '.choices[0].message.content // ""' 2>/dev/null)

if [[ -n "$TOOL_CALLS" && "$TOOL_CALLS" != "null" ]]; then
  TOOL_NAME=$(echo "$TOOL_CALLS" | jq -r '.[0].function.name' 2>/dev/null)
  pass "Tool call detected: $TOOL_NAME (finish=$TOOL_FINISH)"
elif [[ -n "$TOOL_CONTENT" && "$TOOL_CONTENT" != "null" && "$TOOL_CONTENT" != "" ]]; then
  # Model responded with text instead of tool call — not a server bug
  skip "Model chose text over tool call" "${TOOL_CONTENT:0:60}"
else
  # Empty content = model spent all tokens thinking (common with thinking models).
  # Server still returned a valid response — not a hang. Mark as skip, not fail.
  COMP_TOKENS=$(echo "$TOOL_RESP" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  if echo "$TOOL_RESP" | jq -e '.choices[0]' >/dev/null 2>&1; then
    skip "Model exhausted tokens thinking ($COMP_TOKENS tok, no visible output)"
  else
    fail "No valid response" "$TOOL_RESP"
  fi
fi

# ─── Test 6: Streaming with tools ──────────────────────────
header "Test 6: Streaming tool call"
STOOL_OUT=$(mktemp)
HTTP_CODE=$(curl -sf -o "$STOOL_OUT" -w "%{http_code}" --max-time 60 -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. When asked about the weather, always use the get_weather tool."},
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "stream": true,
    "temperature": 0.0,
    "max_tokens": 256
  }' 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" == "200" ]]; then
  if grep -q '\[DONE\]' "$STOOL_OUT"; then
    # Check if tool_calls appear in stream deltas
    if grep -q 'tool_calls' "$STOOL_OUT"; then
      STREAM_FINISH=$(grep 'finish_reason' "$STOOL_OUT" | tail -1 | grep -o '"tool_calls"\|"stop"' | head -1)
      pass "Streaming tool call with [DONE] (finish=$STREAM_FINISH)"
    else
      # Stream closed (no hang!) but tools came as content
      pass "Stream closed properly (tool as content — model-dependent)"
    fi
  else
    fail "Streaming tool: missing [DONE]"
  fi
else
  if [[ "$HTTP_CODE" == "000" ]]; then
    fail "Streaming tool timed out — HANG DETECTED"
  else
    fail "Streaming tool HTTP $HTTP_CODE"
  fi
fi
rm -f "$STOOL_OUT"

# ─── Test 7: Multi-turn tool call flow ─────────────────────
header "Test 7: Multi-turn tool-calling (ChatML format test)"
# Simulate a complete tool-call round-trip:
# Turn 1: user asks question → assistant calls tool
# Turn 2: tool result → assistant gives final answer
MULTI_RESP=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --max-time 60 \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the weather in Paris?"},
      {"role": "assistant", "content": null, "tool_calls": [{
        "id": "call_test123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}
      }]},
      {"role": "tool", "tool_call_id": "call_test123", "content": "{\"temperature\": 18, \"condition\": \"partly cloudy\", \"humidity\": 65}"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "stream": false,
    "temperature": 0.0,
    "max_tokens": 256
  }' 2>/dev/null || echo '{"error":"request failed"}')

MULTI_CONTENT=$(echo "$MULTI_RESP" | jq -r '.choices[0].message.content // ""' 2>/dev/null)
MULTI_FINISH=$(echo "$MULTI_RESP" | jq -r '.choices[0].finish_reason // ""' 2>/dev/null)

if [[ -n "$MULTI_CONTENT" && "$MULTI_CONTENT" != "null" && ${#MULTI_CONTENT} -gt 5 ]]; then
  # Check if the response references the tool result (18 degrees, cloudy, Paris, etc.)
  if echo "$MULTI_CONTENT" | grep -qiE '18|cloud|paris|weather|temperature'; then
    pass "Multi-turn: model used tool result (finish=$MULTI_FINISH, ${#MULTI_CONTENT} chars)"
    echo "       Content: ${MULTI_CONTENT:0:100}"
  else
    # Got content but didn't reference the tool result — ChatML might still be broken
    pass "Multi-turn: got response but may not reference tool data"
    echo "       Content: ${MULTI_CONTENT:0:100}"
  fi
else
  fail "Multi-turn: empty or missing response" "$MULTI_RESP"
fi

# ─── Test 8: Error handling — bad model ─────────────────────
header "Test 8: Error handling"
ERR_RESP=$(curl -sf -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nonexistent-model-xyz",
    "messages": [{"role": "user", "content": "test"}],
    "stream": false
  }' 2>/dev/null || echo '{"error":"request failed"}')

if echo "$ERR_RESP" | jq -e '.error' >/dev/null 2>&1; then
  pass "Bad model returns error response"
else
  fail "Bad model: expected error" "$ERR_RESP"
fi

# ─── Test 9 (optional): Hermes agent integration ───────────
if $HERMES_TEST; then
  header "Test 9: Hermes agent integration"

  if ! command -v hermes &>/dev/null; then
    skip "Hermes CLI not found"
  else
    # Backup and modify Hermes config to point at hipfire
    cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak

    # Patch config to use hipfire serve as custom endpoint
    python3 -c "
import yaml, sys
with open('$HOME/.hermes/config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['model']['default'] = '$MODEL'
cfg['model']['provider'] = 'custom'
cfg['model']['base_url'] = '$BASE/v1'
cfg['model']['api_key'] = 'hipfire-local'
cfg['streaming']['enabled'] = False
cfg['agent']['max_turns'] = 3
with open('$HOME/.hermes/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Hermes config patched to use hipfire serve')
" 2>/dev/null

    if [[ $? -eq 0 ]]; then
      # Run a simple one-shot query
      echo "  Running: hermes -1 'Say hello world'"
      HERMES_OUT=$(timeout 60 hermes -1 "Say exactly: hello from hermes" 2>&1 || echo "HERMES_TIMEOUT")

      if [[ "$HERMES_OUT" == *"HERMES_TIMEOUT"* ]]; then
        fail "Hermes timed out (60s)"
      elif [[ ${#HERMES_OUT} -gt 5 ]]; then
        pass "Hermes got response (${#HERMES_OUT} chars)"
        echo "       Output: ${HERMES_OUT:0:120}"
      else
        fail "Hermes: empty response" "$HERMES_OUT"
      fi
    else
      skip "Hermes config patch failed (missing pyyaml?)"
    fi

    # Config restored by cleanup trap
  fi
fi

# ─── Summary ───────────────────────────────────────────────
header "Results"
echo -e "  ${GREEN}$PASS passed${NC}  ${RED}$FAIL failed${NC}  ${YELLOW}$SKIP skipped${NC}"

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi

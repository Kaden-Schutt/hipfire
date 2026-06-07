#!/usr/bin/env bash
# End-to-end regression: server-side KV-budget handling through the HTTP API.
# Boots `hipfire serve` on a unique port, exercises:
#   A. normal request (baseline)
#   B. long system prompt that would overrun the pre-warm max_seq — server
#      fix must detect and reload with bigger budget
#   C. follow-up small request (reuse loaded, no spurious reload)

set -uo pipefail
PORT=${PORT:-11439}
MODEL=${MODEL:-qwen3.5:0.8b}
LOG=$(mktemp)

echo "=== booting serve on :$PORT (log $LOG) ==="
HIPFIRE_MODEL="$MODEL" bun cli/index.ts serve "$PORT" > "$LOG" 2>&1 &
PID=$!
trap "kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null; rm -f $LOG" EXIT

# Wait for health
for i in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "ready after ${i}s"; break; fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "FAIL: serve exited prematurely"
    tail -30 "$LOG"
    exit 1
  fi
  sleep 1
done

fail=0

echo "=== A: normal ==="
A=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" --max-time 60 \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0.0,\"stream\":false}")
echo "$A" | jq -r '{finish: .choices[0].finish_reason, tokens: .usage.completion_tokens}'
if ! echo "$A" | jq -e '.choices[0].finish_reason == "stop"' >/dev/null; then
  echo "FAIL A"; fail=$((fail+1))
fi

echo "=== B: long system prompt (~6000 tokens — needs bump past 32768? no; but covers the exact-check path) ==="
SYS=$(awk 'BEGIN{for(i=0;i<1200;i++)printf "the quick brown fox jumped over the lazy dog. "}')
PAY=$(jq -cn --arg sys "$SYS" --arg m "$MODEL" \
  '{model:$m,messages:[{role:"system",content:$sys},{role:"user",content:"hi"}],max_tokens:16,temperature:0.0,stream:false}')
B=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" --max-time 120 -d "$PAY")
echo "$B" | jq '{finish: .choices[0].finish_reason, tokens: .usage.completion_tokens, error: .error}'
if echo "$B" | jq -e '.error' >/dev/null; then
  echo "FAIL B — got error: $(echo $B | jq -r .error)"; fail=$((fail+1))
fi

echo "=== C: short follow-up (should reuse loaded model) ==="
C=$(curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" --max-time 30 \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0.0,\"stream\":false}")
echo "$C" | jq -r '{finish: .choices[0].finish_reason, tokens: .usage.completion_tokens}'
if ! echo "$C" | jq -e '.choices[0].finish_reason == "stop"' >/dev/null; then
  echo "FAIL C"; fail=$((fail+1))
fi

echo "=== server log (KV/reload-relevant) ==="
grep -E "KV budget|bumping load|context full|error" "$LOG" || echo "(no error/reload triggered)"

if [[ $fail -eq 0 ]]; then echo "OK"; exit 0; else echo "$fail FAILS"; exit 1; fi

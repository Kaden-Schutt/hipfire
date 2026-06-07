#!/usr/bin/env bash
# Verify server-side reload on oversized max_tokens still works after the daemon fix.
set -uo pipefail
PORT=${PORT:-11440}
MODEL=${MODEL:-qwen3.5:0.8b}
LOG=$(mktemp)
HIPFIRE_MODEL="$MODEL" bun cli/index.ts serve "$PORT" > "$LOG" 2>&1 &
PID=$!
trap "kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null; rm -f $LOG" EXIT

for i in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "serve died"; tail "$LOG"; exit 1; fi
  sleep 1
done

curl -sf -X POST "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" --max-time 120 \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":40000,\"temperature\":0.0,\"stream\":false}" \
  > /dev/null

echo "=== reload/error lines ==="
if grep -qE "bumping load" "$LOG"; then
  echo "PASS: server bumped load for oversized max_tokens"
  grep -E "bumping load" "$LOG"
  exit 0
else
  echo "FAIL: expected 'bumping load' not found"
  grep -E "KV budget|error|exceeds" "$LOG" || echo "(no error either)"
  exit 1
fi

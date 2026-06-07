#!/usr/bin/env bash
# Verify HTTP clients receive a proper error (not a silent success) when the
# daemon rejects a request for KV-budget overrun. Covers both:
#   N) non-streaming → HTTP 400 with {error:{message,...}}
#   S) streaming     → SSE chunk with {"error":...} before [DONE]
#
# Strategy: load with a tiny max_seq (via config.max_seq=1024, max_tokens=16)
# so the pre-warm load only allocates max_seq=1040. Then send a request with
# max_tokens=900 (fits server-side budget since 900+1024=1924 < default 32768
# so no reload gets triggered, BUT actually we need a scenario where the
# server cannot bump in time).
#
# Simpler: use a request whose max_tokens requires a larger max_seq than the
# daemon can allocate. We can't easily force OOM, so instead we simulate the
# failure by forcing a huge max_tokens AFTER the server-side reload.
# Actually cleanest: use HIPFIRE_MODEL with a preconfigured max_seq=1024 via
# a temporary config file, then send max_tokens>1024.

set -uo pipefail
PORT=${PORT:-11441}
MODEL=${MODEL:-qwen3.5:0.8b}
LOG=$(mktemp)
TMPCFG=$(mktemp -d)

# Isolate HOME so we don't clobber the user's real ~/.hipfire/config.json.
# Only the config file differs — models/bin are SYMLINKED, never copied.
mkdir -p "$TMPCFG/.hipfire"
ln -sfn "$HOME/.hipfire/models" "$TMPCFG/.hipfire/models"
ln -sfn "$HOME/.hipfire/bin"    "$TMPCFG/.hipfire/bin"
# Tight config: tiny max_seq + default max_tokens so a moderate prompt tips
# over the KV budget and the daemon must reject.
cat > "$TMPCFG/.hipfire/config.json" <<'JSON'
{"max_seq": 1024, "max_tokens": 16, "default_model": "qwen3.5:0.8b"}
JSON

HOME="$TMPCFG" HIPFIRE_MODEL="$MODEL" bun cli/index.ts serve "$PORT" > "$LOG" 2>&1 &
PID=$!
trap "kill -TERM $PID 2>/dev/null; wait $PID 2>/dev/null; rm -rf $TMPCFG $LOG /tmp/qg_N.json" EXIT

for i in $(seq 1 90); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "serve died"; tail "$LOG"; exit 1; fi
  sleep 1
done

fails=0

# The server-side fix bumps max_seq to max(config.max_seq, max_tokens+1024) on
# load, so to trigger the DAEMON rejection we need `new_tokens.len()` itself
# (which the server can't predict without tokenizing) to overshoot the bumped
# budget. A huge system prompt does that: bumped max_seq = max(1024, 500+1024)
# = 1524, but a ~1800-token system prompt → prefill ~= 1810, + max_tokens 500
# = 2310 > 1524 → daemon emits the KV-budget error.
HUGE_SYS=$(awk 'BEGIN{for(i=0;i<450;i++)printf "the quick brown fox jumped over the lazy dog. "}')
PAY_N=$(jq -cn --arg s "$HUGE_SYS" --arg m "$MODEL" \
  '{model:$m, messages:[{role:"system",content:$s},{role:"user",content:"hi"}], max_tokens:500, temperature:0.0, stream:false}')
echo "=== N: non-streaming (expect 400 + error.message) ==="
HTTP_N=$(curl -s -o /tmp/qg_N.json -w "%{http_code}" --max-time 60 -X POST \
  "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$PAY_N")
echo "HTTP $HTTP_N"
cat /tmp/qg_N.json | jq '.' 2>/dev/null || cat /tmp/qg_N.json
if [[ "$HTTP_N" == "400" ]] && jq -e '.error.message | test("KV budget"; "i")' /tmp/qg_N.json >/dev/null 2>&1; then
  echo "PASS N"
else
  echo "FAIL N"; fails=$((fails+1))
fi
rm -f /tmp/qg_N.json

# S) Streaming rejection. Same payload, stream=true. Must emit SSE chunk with
# top-level "error" field before [DONE], NOT a silent finish_reason=stop.
PAY_S=$(echo "$PAY_N" | jq -c '.stream = true')
echo "=== S: streaming (expect data: {\"error\":...} before [DONE]) ==="
OUT_S=$(mktemp)
curl -sN --max-time 60 -X POST "http://localhost:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" -d "$PAY_S" > "$OUT_S"
cat "$OUT_S"
if grep -q '"error"' "$OUT_S" && grep -q 'KV budget' "$OUT_S" && grep -q '\[DONE\]' "$OUT_S"; then
  echo "PASS S"
else
  echo "FAIL S"; fails=$((fails+1))
fi
rm -f "$OUT_S"

if [[ $fails -eq 0 ]]; then echo "ALL PASS"; exit 0; else echo "$fails FAILS"; exit 1; fi

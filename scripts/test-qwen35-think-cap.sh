#!/usr/bin/env bash
# Regression test: the total-think cap bounds unbounded thinking.
#
# Bug: 35b-a3b re-opens <think> after the one-shot force-answer and thinks
# unbounded until the client times out (the per-block max_think_tokens resets on
# each re-open). Fix: a re-arm-proof TOTAL-think cap (HIPFIRE_MAX_TOTAL_THINK_TOKENS)
# that latches force-answer at the cap (force-close + block + re-close re-opens)
# and hard-EOS a margin past it, plus a persistent force-answer latch.
#
# Requires the daemon running with the cap enabled + cache trace. Launch e.g.:
#   HIPFIRE_MAX_TOTAL_THINK_TOKENS=800 HIPFIRE_FORCE_ANSWER_SECS=999 \
#     HIPFIRE_QWEN35_GRAMMAR=1 HIPFIRE_QWEN_CACHE_TRACE=1 \
#     HIPFIRE_MODEL=qwen3.6-35b-a3b.mq4 bash scripts/serve-restart.sh 11435
#
# (FORCE_ANSWER_SECS high isolates the cap; set it low to test the time path.)
set -uo pipefail
PORT=${1:-11435}
MODEL=${HIPFIRE_TEST_MODEL:-qwen3.6-35b-a3b.mq4}
LOG=${HIPFIRE_SERVE_LOG:-$HOME/.hipfire/serve.log}
EP="http://127.0.0.1:$PORT/v1/chat/completions"
command -v jq >/dev/null || { echo "need jq"; exit 2; }
curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "daemon not responding on :$PORT — start it with the cap enabled first"; exit 2; }

L0=$(wc -l < "$LOG"); START=$(date +%s)
# A hard reasoning prompt with thinking ON — would think to max_tokens (minutes)
# without the cap.
curl -fsS -N -m 240 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Reason at exhaustive length, double-checking every state, through the 3-gallon/5-gallon water-pouring puzzle to measure exactly 4 gallons, then give the final answer."}],stream:true,max_tokens:6000,temperature:0,chat_template_kwargs:{enable_thinking:true}}')" \
  > /tmp/think_cap.sse 2>/dev/null || true
ELAPSED=$(($(date +%s)-START))
echo "elapsed: ${ELAPSED}s (max_tokens=6000 would be minutes if thinking were unbounded)"
NEW=$(tail -n +$((L0+1)) "$LOG" 2>/dev/null)

# (1) the cap / latch must have engaged
echo "$NEW" | grep -qiE 'think-cap|re-closing a re-opened|force-answer' \
  && echo "  OK: total-think bound engaged (latch / re-close / EOS fired)" \
  || { echo "FAIL: think bound never engaged"; echo "$NEW" | grep -iE 'think|force' | tail; exit 1; }

# (2) the turn must terminate server-side, NOT hang to the client timeout
if echo "$NEW" | grep -qiE 'stream client cancelled'; then
  echo "FAIL: turn ran to the client timeout (thinking not bounded)"; exit 1
fi
echo "  OK: turn terminated server-side (no client-timeout cancel)"

# (3) a clean finish_reason should be present in the stream
grep -qE '"finish_reason":"(stop|length|tool_calls)"' /tmp/think_cap.sse \
  && echo "PASS: thinking was bounded and the turn finished cleanly" \
  || { echo "FAIL: no finish_reason in stream (see /tmp/think_cap.sse)"; exit 1; }

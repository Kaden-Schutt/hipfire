#!/usr/bin/env bash
# Regression test for fix/deltanet-truncation-resume-guard.
#
# Bug: an early-truncated turn (client cancels mid-decode) used to leave the
# NON-REVERSIBLE DeltaNet recurrent state and `conversation_tokens` dirty with
# UNCOMMITTED tokens — the AR decode-abort path returned without resetting
# (unlike the DFlash + AR prefill-abort paths). The next turn then resumed the
# prompt cache off that poisoned prior, drifting the recurrent state
# off-distribution -> whitespace garbage / premature EOS, worse on each retry
# (observed live on qwen3.6-35b-a3b: prior_tail decayed real-code -> "   \tt   "
# -> EOS across three retries).
#
# Fix: the AR decode-abort path now full-resets state. After a cancel, the prior
# is empty, so the next turn cold-prefills from a clean state.
#
# DETERMINISTIC ASSERTION: after a real mid-decode abort, the next turn's
# `[qwen-cache GEN-ENTRY] conv_tok=0` — i.e. the daemon dropped the uncommitted
# tokens. WITHOUT the fix, conv_tok would carry Turn A's partial generation
# (poison), and the retry would `[qwen-cache resume] rewound` off it.
#
# Requires the daemon running with HIPFIRE_QWEN_CACHE_TRACE=1 (for GEN-ENTRY /
# cache logs). Launch e.g.:
#   HIPFIRE_QWEN_CACHE_TRACE=1 HIPFIRE_MODEL=qwen3.6-35b-a3b.mq4 \
#     bash scripts/serve-restart.sh 11435
set -uo pipefail
PORT=${1:-11435}
MODEL=${HIPFIRE_TEST_MODEL:-qwen3.6-35b-a3b.mq4}
LOG=${HIPFIRE_SERVE_LOG:-$HOME/.hipfire/serve.log}
EP="http://127.0.0.1:$PORT/v1/chat/completions"
# Forced-long, thinking-off prompt so Turn A is guaranteed to STILL BE DECODING
# when we close the socket at 6s (can't finish 3000 numbers in 6s, and won't
# hit a natural stop). This is what makes the cancel land in decode, not after.
PROMPT="Reason step by step in exhaustive detail, then write a comprehensive multi-section analysis comparing B-trees, LSM-trees, hash indexes, and tries for high-ingest time-series workloads. Cover concurrency control, cache behavior, and write amplification thoroughly."

command -v jq >/dev/null || { echo "need jq"; exit 2; }
curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || { echo "daemon not responding on :$PORT — start it with $MODEL loaded first"; exit 2; }

LINES0=$(wc -l < "$LOG" 2>/dev/null || echo 0)

echo "=== Turn A: stream with thinking ON, close the socket mid-decode (curl -m 5) ==="
# Thinking ON + a hard prompt guarantees Turn A is still in <think> decode at 5s
# (35b-a3b reasons for minutes). curl -m 5 closes the socket -> bun detects the
# disconnect -> sends {type:abort} -> AR decode-abort fires. `|| true`: -m exits 28.
curl -sS -N -m 5 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" --arg p "$PROMPT" '{model:$m,messages:[{role:"user",content:$p}],max_tokens:4000,temperature:0,stream:true}')" \
  >/tmp/abort_turnA.sse 2>/dev/null || true
BYTES=$(wc -c </tmp/abort_turnA.sse 2>/dev/null || echo 0)
echo "  Turn A socket closed at ~5s ($BYTES bytes streamed)"
[ "$BYTES" -gt 0 ] || { echo "FAIL: Turn A produced no output — never reached decode"; exit 1; }
sleep 3   # let the daemon's stdin reader process the abort + state reset

echo "=== Turn B: fresh request (must start from RESET state, thinking off) ==="
RESB=$(curl -fsS -m 120 "$EP" -H 'content-type: application/json' \
  -d "$(jq -cn --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Reply with exactly: the quick brown fox"}],max_tokens:48,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')")
CONTENT=$(echo "$RESB" | jq -r '.choices[0].message.content // ""')
echo "  Turn B content: $(printf '%s' "$CONTENT" | head -c 160)"

NEW=$(tail -n +$((LINES0+1)) "$LOG" 2>/dev/null)

# (1) the cancel must have propagated to a real abort
echo "$NEW" | grep -qE 'daemon-abort|stream client cancelled|"type":"aborted"' \
  && echo "  OK: cancel propagated to a decode-abort" \
  || { echo "FAIL: no abort fired (Turn A finished before cancel, or disconnect not detected)"; echo "--- new log ---"; echo "$NEW" | grep -E 'qwen-cache|abort|cancel'; exit 1; }

# (2) DETERMINISTIC fix detector: the turn AFTER the abort must enter with a
#     RESET conversation (conv_tok=0). Last GEN-ENTRY in the new log = Turn B.
LAST_CONV=$(echo "$NEW" | grep -oE '\[qwen-cache GEN-ENTRY\] conv_tok=[0-9]+' | tail -1 | grep -oE '[0-9]+$')
echo "  Turn B entered with conv_tok=${LAST_CONV:-?}"
[ "${LAST_CONV:-1}" = "0" ] \
  && echo "  OK: decode-abort reset the recurrent-state bookkeeping (conv_tok=0)" \
  || { echo "FAIL: conv_tok=${LAST_CONV} after abort — uncommitted tokens NOT reset (regression)"; exit 1; }

# (3) and it must not have resumed off a (now-impossible) poisoned prior
echo "$NEW" | grep -qE '\[qwen-cache resume\] rewound' \
  && { echo "FAIL: checkpoint-resume after a cancel — state was not reset"; exit 1; } \
  || echo "  OK: no poisoned checkpoint-resume after cancel"

# (4) coherence sanity on Turn B (catches whitespace-attractor / empty garbage)
LEN=${#CONTENT}; NONSPACE=$(printf '%s' "$CONTENT" | tr -d '[:space:]' | wc -c)
UNIQ=$(printf '%s' "$CONTENT" | fold -w1 | sort -u | wc -l)
echo "  Turn B coherence: len=$LEN nonspace=$NONSPACE uniq_chars=$UNIQ"
{ [ "$NONSPACE" -ge 5 ] && [ "$UNIQ" -ge 6 ]; } \
  && echo "PASS: early truncation did not poison the next turn" \
  || { echo "FAIL: Turn B degenerate/empty after a cancel"; exit 1; }

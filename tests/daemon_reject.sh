#!/usr/bin/env bash
# Direct-daemon probe: verify the KV-budget rejection path covers both
# failure modes the exact post-build guard has to catch:
#   R1. max_tokens on its own exceeds max_seq
#   R2. a long system prompt tips new_tokens.len() + max_tokens past max_seq
#       even when a tokenize-the-user-prompt estimate says it fits
# NOT expected: silent overrun / crash.

set -euo pipefail
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
DAEMON="target/release/daemon"

if [[ ! -f "$MODEL" ]]; then echo "missing model: $MODEL"; exit 2; fi
if [[ ! -x "$DAEMON" ]]; then echo "missing daemon: $DAEMON"; exit 2; fi

run_case() {
  local label="$1"; shift
  local generate_json="$1"; shift
  local max_seq="$1"; shift

  local tmp
  tmp=$(mktemp)
  {
    printf '{"type":"ping"}\n'
    printf '{"type":"load","model":"%s","params":{"max_seq":%d}}\n' "$MODEL" "$max_seq"
    printf '%s\n' "$generate_json"
    sleep 4
  } | "$DAEMON" > "$tmp" 2>/dev/null || true

  echo "=== [$label] daemon responses ==="
  cat "$tmp"
  echo
  if grep -q '"type":"error"' "$tmp" && grep -q 'exceeds loaded KV budget' "$tmp"; then
    echo "PASS [$label]: daemon emitted capacity-rejection error"
    rm -f "$tmp"
    return 0
  else
    echo "FAIL [$label]: did not see expected rejection error"
    rm -f "$tmp"
    return 1
  fi
}

fails=0

# R1: max_tokens alone overruns. Short prompt, no system, small max_seq.
run_case "R1 max_tokens overrun" \
  '{"type":"generate","id":"r1","prompt":"hi","max_tokens":3000}' \
  2048 \
  || fails=$((fails+1))

# R2: long system prompt tips prefill past max_seq. Build a system prompt
# that tokenizes to ~1500 tokens so that user_prompt_est (+20 headroom) would
# say the request fits but new_tokens.len() (which includes the system prompt)
# plus max_tokens does not. This is the false-negative the stop hook flagged.
SYS=""
for _ in $(seq 1 500); do SYS="$SYS the quick brown fox jumped over the lazy dog."; done
# Build JSON with a python-free one-liner (jq handles the escaping)
PAYLOAD=$(jq -cn --arg sys "$SYS" '{type:"generate",id:"r2",prompt:"hi",system:$sys,max_tokens:1000}')
run_case "R2 long system prompt overrun" \
  "$PAYLOAD" \
  2048 \
  || fails=$((fails+1))

# R3: exact-boundary overrun via the terminal \n write. max_seq = 2048,
# prefill ~9 (user-turn framing for "hi"), max_tokens = 2039 → sums to
# exactly 2048. Without reserving nl.len() for the post-generation ChatML
# trailer, a natural im_end termination would silently overflow by 1+
# slots. The guard must count the trailer.
run_case "R3 terminal-newline boundary overrun" \
  '{"type":"generate","id":"r3","prompt":"hi","max_tokens":2039}' \
  2048 \
  || fails=$((fails+1))

echo
if [[ "$fails" -eq 0 ]]; then
  echo "ALL PASS"
  exit 0
else
  echo "$fails FAILED"
  exit 1
fi

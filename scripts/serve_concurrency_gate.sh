#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.
#
# SP7 integration gate: `hipfire serve` serves several agents at once.
#
# Everything else in this programme is tested in-process. This is the only
# test that exercises what an agent actually touches: the real binary, the
# real config path (serve.multi_slot), real HTTP, real OpenAI-compatible
# JSON, and the admission gate in front of the engine.
#
# It asserts three things, and each one has failed for real during
# development:
#
#   * every request returns HTTP 200 with non-empty content
#       (the terminal callback was once never invoked -> "generation worker
#        disconnected" on every request)
#   * the answers are DISTINCT
#       (DeltaNet state was once not reset on slot reuse, so every request
#        after the first echoed the previous conversation)
#   * concurrent is materially faster than sequential
#       (the HTTP admission gate was a single busy flag, so requests
#        serialised even though the engine could take four)
#
# Harness rules encoded here, learned the hard way:
#   * liveness is `ss -ltn`, never `pgrep -f` — a `pgrep -f` pattern that
#     appears in the checking command's own argv matches itself
#   * concurrent waits use explicit PIDs, never bare `wait`, which would
#     also wait on the backgrounded server
#   * teardown kills the server's CHILDREN, not just the run-bounded
#     wrapper — killing the wrapper leaves the model resident and every
#     later run is refused by the memory gate
#
# Usage: scripts/serve_concurrency_gate.sh [model_path] [port]

set -uo pipefail

MODEL="${1:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
PORT="${2:-11477}"
SLOTS="${SERVE_GATE_SLOTS:-4}"
MAX_TOKENS="${SERVE_GATE_MAX_TOKENS:-48}"
MIN_SPEEDUP="${SERVE_GATE_MIN_SPEEDUP:-1.30}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
SRV_PID=""

cleanup() {
  # Kill the children, not just the wrapper: the wrapper exiting leaves the
  # model resident (~46 GiB of GTT observed), after which run-bounded refuses
  # every subsequent run.
  local kids
  kids="$(pgrep -P "${SRV_PID:-0}" 2>/dev/null || true)"
  for p in $kids $SRV_PID; do
    [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true
  done
  sleep 3
  for p in $kids $SRV_PID; do
    [ -n "$p" ] && kill -KILL "$p" 2>/dev/null || true
  done
  # Keep the server log: a gate that deletes its only diagnostic on failure
  # forces every investigation to start by reproducing the failure.
  [ -f "$WORK/serve.log" ] && cp "$WORK/serve.log" /tmp/serve-gate.log 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$ROOT/target/release/hipfire" ] || fail "build first: cargo build --release -p hipfire-cli"
DAEMON="$ROOT/target/release/daemon"
[ -x "$DAEMON" ] || fail "build first: cargo build --release -p hipfire-daemon"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  fail "port $PORT already in use"
fi

echo "=== serve concurrency gate ==="
echo "  model $MODEL"
echo "  $SLOTS slots, $MAX_TOKENS tokens/request, port $PORT"

# The installed ~/.hipfire/bin/daemon is frequently protocol-stale against a
# development branch, so pin the daemon this tree built.
# Defaulted, NOT forced: hardcoding this would override a caller trying to
# run the multi_slot=0 negative control, and the "control" would silently
# test the same arm as the positive run.
HIPFIRE_DAEMON_BIN="$DAEMON" \
HIPFIRE_SERVE_MULTI_SLOT="${HIPFIRE_SERVE_MULTI_SLOT:-1}" \
HIPFIRE_SLOT_TRACE="${HIPFIRE_SLOT_TRACE:-1}" \
HIPFIRE_MEM_CAP="${HIPFIRE_MEM_CAP:-34G}" \
  "$ROOT/scripts/run-bounded.sh" "$ROOT/target/release/hipfire" serve \
    --model "$MODEL" --no-prewarm "$PORT" > "$WORK/serve.log" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 200); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " && break
  sleep 3
done
ss -ltn 2>/dev/null | grep -q ":$PORT " || {
  tail -20 "$WORK/serve.log" >&2
  fail "server never listened on $PORT"
}
grep -q "multi-slot backend up" "$WORK/serve.log" || {
  tail -20 "$WORK/serve.log" >&2
  fail "multi-slot backend did not start — serve.multi_slot was not honoured"
}
echo "  listener up, multi-slot backend confirmed"

URL="http://127.0.0.1:$PORT/v1/chat/completions"
Q=("What is the capital of France?"
   "What does gradient descent do?"
   "How do you make a cup of tea?"
   "Who described the laws of motion?")

req() { # req <prompt> <outfile>
  curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$MAX_TOKENS}" \
    -o "$2" -w '%{http_code}'
}

echo "[phase] sequential"
echo "--- ${#Q[@]} requests sequentially ---"
S=$(date +%s.%N)
for i in "${!Q[@]}"; do
  code="$(req "${Q[$i]}" "$WORK/seq$i.json")"
  [ "$code" = "200" ] || fail "sequential request $i returned HTTP $code"
done
E=$(date +%s.%N)
SEQ=$(echo "$E - $S" | bc)
echo "  sequential: ${SEQ}s"

echo "[phase] concurrent"
echo "--- the same ${#Q[@]} concurrently ---"
S=$(date +%s.%N)
PIDS=()
for i in "${!Q[@]}"; do
  ( req "${Q[$i]}" "$WORK/con$i.json" > "$WORK/code$i" ) &
  PIDS+=($!)
done
# Explicit PIDs: a bare `wait` would also wait on the server started above.
for p in "${PIDS[@]}"; do wait "$p"; done
E=$(date +%s.%N)
CON=$(echo "$E - $S" | bc)
echo "  concurrent: ${CON}s"

for i in "${!Q[@]}"; do
  code="$(cat "$WORK/code$i")"
  [ "$code" = "200" ] || fail "concurrent request $i returned HTTP $code"
done

python3 - "$WORK" "${#Q[@]}" "$SEQ" "$CON" "$MIN_SPEEDUP" <<'PY' || exit 1
import json, sys
work, n, seq, con, min_speedup = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
outs = []
for i in range(n):
    with open(f"{work}/con{i}.json") as fh:
        d = json.load(fh)
    m = d["choices"][0]["message"]
    # Either channel counts. A thinking model on a small token budget spends
    # it all reasoning, so `content` is legitimately empty while
    # `reasoning_content` carries the text -- that is a served request, not a
    # failure. What this phase actually tests is per-client isolation.
    c = (m.get("content") or "") + (m.get("reasoning_content") or "")
    if not c.strip():
        print(f"FAIL: concurrent request {i} produced no text in EITHER channel",
              file=sys.stderr)
        sys.exit(1)
    outs.append(c)
    print(f"  c{i}: {c[-46:]!r}")

if len(set(outs)) < 2:
    print("FAIL: every client produced identical content — sessions are not "
          "isolated (DeltaNet state is not reset on slot reuse?)", file=sys.stderr)
    sys.exit(1)
print(f"  distinct answers: {len(set(outs))}/{n}")

speedup = seq / con if con > 0 else 0.0
print(f"  SPEEDUP: {speedup:.2f}x  (sequential {seq:.2f}s vs concurrent {con:.2f}s)")
if speedup < min_speedup:
    print(f"FAIL: speedup {speedup:.2f}x below the {min_speedup:.2f}x floor — "
          "requests are serialising somewhere (admission gate? runtime mutex?)",
          file=sys.stderr)
    sys.exit(1)
PY

# ---- reasoning channel. A thinking model's <think> span must land in
# `reasoning_content`, not in the visible answer. Before this was routed, a
# 4B reply arrived as ~2 KB of "Thinking Process: 1. **Analyze the
# Request**..." in `content` with `reasoning_content` empty -- technically a
# 200, entirely unusable as an answer.
echo "[phase] reasoning channel"
echo "--- reasoning channel ---"
code="$(curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":600}" \
  -o "$WORK/reason.json" -w '%{http_code}')"
[ "$code" = "200" ] || fail "reasoning-channel request returned HTTP $code"
python3 - "$WORK/reason.json" <<'PY2' || exit 1
import json, sys
d = json.load(open(sys.argv[1]))
m = d["choices"][0]["message"]
content = m.get("content") or ""
reasoning = m.get("reasoning_content") or ""
print(f"  content {len(content)} chars, reasoning_content {len(reasoning)} chars")
for marker in ("<think>", "</think>"):
    if marker in content:
        print(f"FAIL: {marker!r} leaked into the visible answer", file=sys.stderr)
        sys.exit(1)
if not reasoning.strip():
    print("FAIL: reasoning_content is empty -- the think span was not routed, "
          "so it is either being dropped or left in the visible answer",
          file=sys.stderr)
    sys.exit(1)
# 600 tokens is enough for this model to finish reasoning AND answer, so an
# empty answer here means the split swallowed the visible text.
if not content.strip():
    print("FAIL: content is empty with 600 tokens -- the visible answer was "
          "routed into reasoning or dropped", file=sys.stderr)
    sys.exit(1)
print(f"  answer: {content.strip()[:60]!r}")
PY2

# ---- streaming (`"stream": true`) goes through respond_streaming, a
# different function from the non-streaming path above. Agents commonly use
# it, so it gets its own assertions: SSE framing, at least one content
# delta, and the [DONE] sentinel.
echo "[phase] streaming"
echo "--- streaming request ---"
code="$(curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":$MAX_TOKENS,\"stream\":true}" \
  -o "$WORK/stream.sse" -w '%{http_code}')"
[ "$code" = "200" ] || fail "streaming request returned HTTP $code"

python3 - "$WORK/stream.sse" <<'PY2' || exit 1
import json, sys
path = sys.argv[1]
lines = [l for l in open(path).read().splitlines() if l.startswith("data: ")]
if not lines:
    print("FAIL: streaming response contained no SSE `data:` frames", file=sys.stderr)
    sys.exit(1)
if lines[-1].strip() != "data: [DONE]":
    print(f"FAIL: stream did not end with [DONE], last frame: {lines[-1][:80]!r}", file=sys.stderr)
    sys.exit(1)
text = ""
for l in lines[:-1]:
    try:
        obj = json.loads(l[6:])
    except json.JSONDecodeError:
        print(f"FAIL: SSE frame is not valid JSON: {l[:80]!r}", file=sys.stderr)
        sys.exit(1)
    if obj.get("object") != "chat.completion.chunk":
        print(f"FAIL: unexpected SSE object {obj.get('object')!r}", file=sys.stderr)
        sys.exit(1)
    delta = obj["choices"][0].get("delta", {})
    # Reasoning streams as a `reasoning_content` delta. Counting only
    # `content` would call a correctly-routed thinking stream empty.
    text += delta.get("content", "") or ""
    text += delta.get("reasoning_content", "") or ""
if not text.strip():
    print("FAIL: stream delivered frames but no content or reasoning deltas",
          file=sys.stderr)
    sys.exit(1)
print(f"  {len(lines)} SSE frames, {len(text)} chars across both channels, [DONE] present")
print(f"  stream: {text[-46:]!r}")
PY2

# ---- multi-turn prefix reuse. OpenAI completions are stateless, so a
# follow-up turn resends the WHOLE conversation: [user, assistant, user].
# That is what makes turn 2's rendered tokens a strict extension of turn 1's;
# merely appending text to a single user message is NOT, because the chat
# frame puts the assistant opener at the end.
echo "[phase] multi-turn"
echo "--- multi-turn prefix reuse ---"
LONG="$(python3 -c 'print("Explain the history of the Roman empire in detail. " * 40, end="")')"

S=$(date +%s.%N)
code="$(curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"m","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2])}))' "$LONG" "$MAX_TOKENS")" \
  -o "$WORK/t1.json" -w '%{http_code}')"
E=$(date +%s.%N)
[ "$code" = "200" ] || fail "turn 1 returned HTTP $code"
T1=$(echo "$E - $S" | bc)

REPLY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$WORK/t1.json")"

# Push turn 1's session out of its slot before continuing it, so turn 2 has to
# RESTORE from the swap store rather than finding it resident. Without this the
# restore path is wired but never exercised -- the same "evictions=0 proves
# nothing" trap this gate exists to avoid.
echo "  evicting turn 1's session with $((SLOTS + 1)) unrelated conversations..."
for e in $(seq 0 "$SLOTS"); do
  curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Unrelated question number $e about geology?\"}],\"max_tokens\":8}" \
    -o /dev/null || fail "eviction filler request $e failed"
done
S=$(date +%s.%N)
code="$(curl -s -m 300 -X POST "$URL" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"m","messages":[{"role":"user","content":sys.argv[1]},{"role":"assistant","content":sys.argv[2]},{"role":"user","content":"And what caused its decline?"}],"max_tokens":int(sys.argv[3])}))' "$LONG" "$REPLY" "$MAX_TOKENS")" \
  -o "$WORK/t2.json" -w '%{http_code}')"
E=$(date +%s.%N)
[ "$code" = "200" ] || fail "turn 2 returned HTTP $code"
T2=$(echo "$E - $S" | bc)
echo "  turn 1 (cold): ${T1}s   turn 2 (continues turn 1, after eviction): ${T2}s"

# Reuse is asserted on the ENGINE EVENT, not on latency. Latency alone cannot
# carry this claim: turn 1 pays one-off warm-up, so turn 2 comes in ~20-30%
# cheaper even when the continuation MISSES and re-prefills from cold. A run
# that reported "reuse saved 23%" was doing exactly that -- the trace said
# `continuation MISS`. The timing figure stays below as information only.
if [ "${SERVE_GATE_REQUIRE_REUSE:-1}" = "1" ]; then
  HITS=$(grep -c 'continuation HIT' "$WORK/serve.log" || true)
  if [ "$HITS" -eq 0 ]; then
    echo "FAIL: no 'continuation HIT' in serve.log -- prefix reuse did not fire." >&2
    grep -E '\[slot-trace\]' "$WORK/serve.log" | tail -20 >&2
    fail "prefix reuse did not fire (0 continuation HITs)"
  fi
  echo "  continuation HIT x$HITS (engine reused the session's KV)"
fi

# Asserted by default. Reuse works: the engine matches the conversation by its
# USER turns and then APPENDS `continuation_suffix` to the session's exact
# stored tokens, rather than re-rendering the history. Re-rendering could never
# match, for two independent reasons: the generated turn began after an
# OpenThink opener that history rendering does not replay, and re-encoding the
# decoded reply is a detokenise/retokenise round trip that is not guaranteed to
# be the identity. Appending sidesteps both.
python3 - "$T1" "$T2" "0" <<'PY3' || exit 1
import sys
t1, t2, require = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3] == "1"
ratio = t2 / t1 if t1 > 0 else 1.0
print(f"  turn2/turn1 = {ratio:.2f}")
if ratio >= 0.90:
    msg = ("turn 2 was not cheaper than turn 1 -- prefix reuse did not fire. "
           "Either the conversation key stopped matching (SessionTable::"
           "find_continuation), or finished sessions are being closed instead "
           "of kept resident, or the continuation suffix no longer lines up "
           "with the stored tokens.")
    if require:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  KNOWN GAP: {msg}")
else:
    print(f"  reuse saved {100*(1-ratio):.0f}% of turn-2 latency")
PY3

echo "ALL CHECKS PASS"

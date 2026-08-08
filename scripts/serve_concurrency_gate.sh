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
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$ROOT/target/release/hipfire" ] || fail "build first: cargo build --release -p hipfire-cli"
DAEMON="$ROOT/target/release/examples/daemon"
[ -x "$DAEMON" ] || fail "build first: cargo build --release -p hipfire-runtime --features deltanet,arch-qwen35 --example daemon"

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

echo "--- ${#Q[@]} requests sequentially ---"
S=$(date +%s.%N)
for i in "${!Q[@]}"; do
  code="$(req "${Q[$i]}" "$WORK/seq$i.json")"
  [ "$code" = "200" ] || fail "sequential request $i returned HTTP $code"
done
E=$(date +%s.%N)
SEQ=$(echo "$E - $S" | bc)
echo "  sequential: ${SEQ}s"

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
    c = d["choices"][0]["message"]["content"]
    if not c.strip():
        print(f"FAIL: concurrent request {i} returned empty content", file=sys.stderr)
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

echo "ALL CHECKS PASS"

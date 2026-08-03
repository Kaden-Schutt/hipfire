#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

# Output-neutrality gate for DeepSeek V4 routed-expert paging.
#
# Paging is pure memory management over read-only weights, so it must not
# change what the model computes. Greedy ds4 decode is deterministic, which
# makes that directly assertable: run the same prompt fully resident, with a
# cache big enough to hold everything it touches, and with a cache small
# enough to thrash, and require identical committed token IDs.
#
# Compares token IDs rather than text because BPE can mask a divergence — two
# different token sequences can render to the same string.
#
# WHY GRAPHS ARE PINNED OFF: paging does a d2h of the routing, file I/O, and
# an h2d of the pointer table inside the layer body, so it forces HIP graph
# capture off (a captured graph would bake one token's residency and replay
# it). A resident run on gfx11xx/gfx12xx takes the graph path by default, so
# comparing it against a paged run compares graph replay against direct
# dispatch — which on this box is NOT bit-identical: measured 2026-08-03 on
# gfx1151, resident-with-graphs and resident-without-graphs diverge at token
# 42 of 192 under greedy decode. That is a property of graph capture, not of
# paging, so pin it off everywhere and let this gate measure one thing.
#
# WHY THE KERNEL GATE IS PINNED: with paging on, the prefill MoE runs over token
# windows sized to the slot pool rather than over the whole chunk. The MoE
# family picks grouped-GEMM vs scalar K4 on `batch_size >= GROUPED_GATE`
# (default 128), so a paged run could take a different kernel than a resident
# run and differ in the low bits — a kernel difference, not a paging bug. We
# pin GROUPED_GATE=0 for EVERY arm so all three take the grouped path and the
# comparison isolates paging. Set PAGING_GATE_PIN_KERNEL=0 to unpin and see
# the unpinned behaviour.
#
# Exit 0 when paged output is bit-identical to resident output, 1 otherwise.

set -u
cd "$(dirname "$0")/.."

MODEL="${HIPFIRE_DS4_MODEL:-$HOME/.hipfire/models/deepseek-v4-flash-0731.mq2lloyd}"
EXE=./target/release/examples/daemon
# A prompt that actually generates. A two-word answer proves almost nothing:
# the more tokens committed, the more expert routings the comparison covers,
# and the more chances an eviction bug has to surface.
PROMPT="${PAGING_GATE_PROMPT:-Explain how a hash table works, step by step, and describe what happens on a collision.}"
MAX_TOKENS="${PAGING_GATE_MAX_TOKENS:-192}"
LARGE_GB="${PAGING_GATE_LARGE_GB:-70}"
SMALL_GB="${PAGING_GATE_SMALL_GB:-8}"
PIN_KERNEL="${PAGING_GATE_PIN_KERNEL:-1}"

# FAST MODE. Loading 72 GiB of routed experts dominates every arm — generation
# is seconds, the load is minutes — so the full-fidelity run is a poor
# iteration loop. `HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END=N` gives routed experts
# to the first N layers only (the rest fall back to shared-only FFN), which
# shrinks EVERY arm by the same factor and leaves the paging code paths —
# pool allocation, catalog, decode paging, prefill windows, eviction —
# fully exercised. Budgets are scaled to match so the thrashing arm still
# thrashes. Both arms use the same layer_end, so the comparison stays valid;
# what it stops being is a statement about the whole model.
#
#   PAGING_GATE_FAST=1                 # ~6 layers of experts
#   PAGING_GATE_FAST=1 PAGING_GATE_LAYERS=12
FAST="${PAGING_GATE_FAST:-0}"
LAYERS="${PAGING_GATE_LAYERS:-6}"
if [ "$FAST" = 1 ]; then
  # Per-slot bytes scale linearly with paged layers, so scale the budgets by
  # LAYERS/43 to keep the same slots-per-blob the full run would have had.
  LARGE_GB="${PAGING_GATE_LARGE_GB:-$(( (70 * LAYERS + 42) / 43 ))}"
  SMALL_GB="${PAGING_GATE_SMALL_GB:-1}"
  MAX_TOKENS="${PAGING_GATE_MAX_TOKENS:-64}"
fi
OUT_DIR="${PAGING_GATE_OUT_DIR:-$(mktemp -d)}"
mkdir -p "$OUT_DIR" || { echo "FAIL: cannot create $OUT_DIR"; exit 1; }

if [ ! -x "$EXE" ]; then
  echo "FAIL: $EXE not built. Run: cargo build --release --example daemon"
  exit 1
fi
if [ ! -e "$MODEL" ]; then
  echo "FAIL: model not found: $MODEL"
  exit 1
fi

echo "model:   $MODEL"
echo "logs:    $OUT_DIR"
echo "kernel:  $([ "$PIN_KERNEL" = 1 ] && echo 'grouped pinned (GROUPED_GATE=0)' || echo 'unpinned')"
echo "graphs:  off on all arms (paging forces it; see header)"
if [ "$FAST" = 1 ]; then
  echo "mode:    FAST — routed experts on layers 0..$LAYERS only, ${MAX_TOKENS} tokens"
  echo "         (exercises every paging path; NOT a whole-model statement)"
else
  echo "mode:    full — all layers, ${MAX_TOKENS} tokens"
fi
echo

# $1 = arm name, $2 = cache GB ("" = fully resident), $3 = graphs (0/1, default 0)
#
# Wall time is recorded alongside the daemon's own `done` event so load cost is
# visible: a paged arm allocates a small pool instead of reading ~72 GiB of
# experts, which is most of what "time to first token of the process" means.
run() {
  local name="$1" gb="$2" graphs="${3:-0}"
  local log="$OUT_DIR/$name.jsonl"
  local -a env_args=(HIPFIRE_EMIT_TOKEN_IDS=1 "HIPFIRE_DEEPSEEK4_GRAPH=$graphs")
  [ "$FAST" = 1 ] && env_args+=("HIPFIRE_DEEPSEEK4_EXPERT_LAYER_END=$LAYERS")
  [ -n "$gb" ] && env_args+=("HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=$gb")
  [ "$PIN_KERNEL" = 1 ] && env_args+=(HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE=0)

  local t0 t1
  t0=$(date +%s%3N)
  printf '%s\n%s\n%s\n' \
    "{\"type\":\"load\",\"model\":\"$MODEL\",\"params\":{\"max_seq\":4096,\"dspark_mode\":\"off\",\"mtp_mode\":\"off\"}}" \
    "{\"type\":\"generate\",\"id\":\"g\",\"prompt\":\"$PROMPT\",\"temperature\":0.0,\"max_tokens\":$MAX_TOKENS,\"repeat_penalty\":1.0}" \
    '{"type":"unload"}' \
  | env "${env_args[@]}" "$EXE" >"$log" 2>"$OUT_DIR/$name.err"
  local rc=$?
  t1=$(date +%s%3N)
  echo "$((t1 - t0))" > "$OUT_DIR/$name.wall_ms"
  # Committed token IDs, in order. One `committed` event per committed token.
  grep -a '"type":"committed"' "$log" \
    | sed 's/.*"tok_id":[[:space:]]*\([0-9]*\).*/\1/' | tr '\n' ' '
  return $rc
}

# Per-arm one-liner from the daemon's `done` event plus measured wall time.
perf_line() {
  local name="$1"
  python3 - "$OUT_DIR" "$name" <<'PYEOF'
import json, sys, os
out_dir, name = sys.argv[1], sys.argv[2]
done = None
try:
    for line in open(os.path.join(out_dir, name + ".jsonl")):
        if '"type":"done"' in line:
            done = json.loads(line)
except OSError:
    pass
try:
    wall = int(open(os.path.join(out_dir, name + ".wall_ms")).read().strip())
except (OSError, ValueError):
    wall = 0
if not done:
    print(f"  {name:<16} (no done event)")
    sys.exit()
# Load+teardown is whatever wall time the generate did not account for.
overhead = max(0, wall - done["total_ms"])
print(f'  {name:<16} {done["tok_s"]:>6.2f} tok/s   prefill {done["prefill_ms"]:>6} ms   '
      f'gen {done["total_ms"]:>6} ms   load+unload {overhead/1000:>6.1f} s')
PYEOF
}

echo "== [1/4] resident (paging off) =="
A=$(run resident "" 0); rc_a=$?
echo "  tokens: $(echo "$A" | wc -w)  rc=$rc_a"
echo "  $(echo "$A" | head -c 160)"

echo "== [2/4] paged, ${LARGE_GB} GB cache =="
B=$(run paged_large "$LARGE_GB" 0); rc_b=$?
echo "  tokens: $(echo "$B" | wc -w)  rc=$rc_b"
echo "  $(echo "$B" | head -c 160)"

echo "== [3/4] paged, ${SMALL_GB} GB cache (thrashing) =="
C=$(run paged_small "$SMALL_GB" 0); rc_c=$?
echo "  tokens: $(echo "$C" | wc -w)  rc=$rc_c"
echo "  $(echo "$C" | head -c 160)"

# Perf reference ONLY — excluded from the correctness comparison on purpose.
# Resident with graphs ON is the configuration a normal run uses, so it is the
# honest baseline to quote paging's cost against. It is also the arm that
# reveals what graph capture is actually worth. Its token IDs are expected to
# differ from the graphs-off arms (see header); that is not a failure here.
if [ "${PAGING_GATE_PERF_REF:-1}" = 1 ]; then
  echo "== [4/4] resident, graphs ON (perf reference, not compared) =="
  R=$(run resident_graphs "" 1); rc_r=$?
  echo "  tokens: $(echo "$R" | wc -w)  rc=$rc_r"
  if [ "$R" = "$A" ]; then
    echo "  note: matches graphs-off resident"
  else
    echo "  note: DIFFERS from graphs-off resident — graph replay is not"
    echo "        byte-equivalent to direct dispatch on this box"
  fi
fi
echo

echo "-- performance --"
perf_line resident
perf_line paged_large
perf_line paged_small
[ "${PAGING_GATE_PERF_REF:-1}" = 1 ] && perf_line resident_graphs
echo

# Hit rate, if the runtime logged one — the budget/throughput curve should be
# empirical, not asserted.
echo "-- cache --"
grep -ah "expert paging ON\|expert cache" "$OUT_DIR"/*.err 2>/dev/null | sort -u | sed 's/^/  /'
echo

rc=0
[ -n "$A" ]   || { echo "FAIL: resident run produced no tokens (see $OUT_DIR/resident.err)"; rc=1; }
[ $rc_a -eq 0 ] || { echo "FAIL: resident daemon exited $rc_a"; rc=1; }
[ $rc_b -eq 0 ] || { echo "FAIL: large-cache daemon exited $rc_b"; rc=1; }
[ $rc_c -eq 0 ] || { echo "FAIL: thrashing-cache daemon exited $rc_c"; rc=1; }
[ "$A" = "$B" ] || { echo "FAIL: large-cache paged output differs from resident"; rc=1; }
[ "$A" = "$C" ] || { echo "FAIL: thrashing-cache paged output differs from resident"; rc=1; }

# A large cache matching while a thrashing one diverges localises the bug:
# the read path is fine, eviction or pointer-table patching is not.
if [ "$A" = "$B" ] && [ "$A" != "$C" ]; then
  echo "HINT: large matches, thrashing does not — suspect eviction or"
  echo "      pointer-table patching (expert_pager.rs), not the read path."
fi

[ $rc -eq 0 ] && echo "PASS: paging is output-neutral"
exit $rc

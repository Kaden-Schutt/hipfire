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
# WHY THE PINNED ENV BELOW: with paging on, the prefill MoE runs over token
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
PROMPT="List three primary colours, comma separated."
MAX_TOKENS="${PAGING_GATE_MAX_TOKENS:-64}"
LARGE_GB="${PAGING_GATE_LARGE_GB:-70}"
SMALL_GB="${PAGING_GATE_SMALL_GB:-8}"
PIN_KERNEL="${PAGING_GATE_PIN_KERNEL:-1}"
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
echo

# $1 = arm name, $2 = cache GB ("" = fully resident)
run() {
  local name="$1" gb="$2"
  local log="$OUT_DIR/$name.jsonl"
  local -a env_args=(HIPFIRE_EMIT_TOKEN_IDS=1)
  [ -n "$gb" ] && env_args+=("HIPFIRE_DEEPSEEK4_EXPERT_CACHE_GB=$gb")
  [ "$PIN_KERNEL" = 1 ] && env_args+=(HIPFIRE_DEEPSEEK4_MOE_GROUPED_GATE=0)

  printf '%s\n%s\n%s\n' \
    "{\"type\":\"load\",\"model\":\"$MODEL\",\"params\":{\"max_seq\":4096,\"dspark_mode\":\"off\",\"mtp_mode\":\"off\"}}" \
    "{\"type\":\"generate\",\"id\":\"g\",\"prompt\":\"$PROMPT\",\"temperature\":0.0,\"max_tokens\":$MAX_TOKENS,\"repeat_penalty\":1.0}" \
    '{"type":"unload"}' \
  | env "${env_args[@]}" "$EXE" >"$log" 2>"$OUT_DIR/$name.err"
  local rc=$?
  # Committed token IDs, in order. One `committed` event per committed token.
  grep -a '"type":"committed"' "$log" \
    | sed 's/.*"tok_id":[[:space:]]*\([0-9]*\).*/\1/' | tr '\n' ' '
  return $rc
}

echo "== [1/3] resident (paging off) =="
A=$(run resident ""); rc_a=$?
echo "  tokens: $(echo "$A" | wc -w)  rc=$rc_a"
echo "  $(echo "$A" | head -c 160)"

echo "== [2/3] paged, ${LARGE_GB} GB cache =="
B=$(run paged_large "$LARGE_GB"); rc_b=$?
echo "  tokens: $(echo "$B" | wc -w)  rc=$rc_b"
echo "  $(echo "$B" | head -c 160)"

echo "== [3/3] paged, ${SMALL_GB} GB cache (thrashing) =="
C=$(run paged_small "$SMALL_GB"); rc_c=$?
echo "  tokens: $(echo "$C" | wc -w)  rc=$rc_c"
echo "  $(echo "$C" | head -c 160)"
echo

# Hit rate, if the runtime logged one — the budget/throughput curve should be
# empirical, not asserted.
grep -ah "expert paging ON\|expert cache" "$OUT_DIR"/*.err 2>/dev/null | sort -u | sed 's/^/  /'

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

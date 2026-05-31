#!/usr/bin/env bash
# Prompt-cache + DFlash bench MATRIX orchestrator.
#
# Runs the 4-cell matrix {dflash on|off} × {thinking on|off} over the scenarios
# in bench_cache_runner.ts (short / multiturn / divergent / toolcalls) and emits
# a markdown report demonstrating cache effectiveness + DFlash performance.
#
#   dflash on/off  → daemon-level: HIPFIRE_DFLASH_CHAT unset (on) vs =0 (force AR)
#   thinking on/off → per-request: reasoning.effort=medium vs enable_thinking:false
#   (thinking-on always routes to AR regardless of the dflash setting — the
#    matrix makes that explicit.)
#
# Usage:
#   scripts/bench_cache_matrix.sh [--model qwen3.6-27b.mq4] [--trials 2] [--fast]
#                                 [--out /tmp/bench_matrix.md]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="qwen3.6-27b.mq4"; TRIALS="2"; FAST=""; OUT="/tmp/bench_matrix_$(date +%Y%m%d-%H%M%S).md"
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;; --trials) TRIALS="$2"; shift 2;;
  --fast) FAST="--fast"; shift;; --out) OUT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 1;; esac; done

RESULTS="$(mktemp /tmp/bench_results_XXXX.jsonl)"; : > "$RESULTS"
PORT=11435

restart_daemon() { # $1 = dflash on|off
  ~/.hipfire/bin/hipfire stop >/dev/null 2>&1; sleep 2
  pid=$(cat ~/.hipfire/daemon.pid 2>/dev/null); [ -n "${pid:-}" ] && kill -TERM "$pid" 2>/dev/null
  rm -f ~/.hipfire/daemon.pid; sleep 2
  local env="HIPFIRE_MODEL=$MODEL"
  [ "$1" = "off" ] && env="$env HIPFIRE_DFLASH_CHAT=0"
  echo "[matrix] starting daemon (dflash=$1): $env" >&2
  env $env ~/.hipfire/bin/hipfire serve -d >/dev/null 2>&1
  local t=0
  until { grep -qiE "warm-up complete|chat/completions" ~/.hipfire/serve.log 2>/dev/null && ss -tlnp 2>/dev/null | grep -q "$PORT"; }; do
    sleep 3; t=$((t+3)); [ "$t" -gt 240 ] && { echo "[matrix] daemon failed to start" >&2; return 1; }
  done
  sleep 2; echo "[matrix] daemon ready (dflash=$1)" >&2
}

echo "[matrix] model=$MODEL trials=$TRIALS fast=${FAST:-no} results=$RESULTS out=$OUT" >&2
for dflash in on off; do
  restart_daemon "$dflash" || exit 1
  for think in off on; do
    label="dflash=$dflash think=$think"
    log="$(mktemp /tmp/bench_cell_XXXX.log)"
    echo "[matrix] === running cell: $label ===" >&2
    timeout 2400 bun scripts/bench_cache_runner.ts --port "$PORT" --model "$MODEL" \
      --think "$think" --label "$label" --trials "$TRIALS" $FAST --out "$RESULTS" >"$log" 2>&1
    rc=$?
    sed -n '/^=== /,$p' "$log"   # echo the cell's human summary
    [ "$rc" -ne 0 ] && echo "[matrix] cell '$label' exited rc=$rc (partial results kept)" >&2
  done
done

echo "" >&2; echo "[matrix] generating report → $OUT" >&2
bun scripts/bench_cache_report.ts "$RESULTS" | tee "$OUT"
echo "[matrix] raw results: $RESULTS" >&2
echo "[matrix] report:      $OUT" >&2

# Restore production daemon (dflash on, default).
restart_daemon on >/dev/null 2>&1 && echo "[matrix] production daemon restored (dflash on)" >&2

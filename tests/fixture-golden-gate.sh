#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny-fixture golden tripwire (GPU).
#
# For each supported arch: emit a tiny random-init fixture → quantize → run the
# forward (fixture_golden) and capture the logit_hash. Two checks:
#   1. DETERMINISM (universal): run twice, hashes must match. Catches a kernel
#      that started producing nondeterministic output.
#   2. BASELINE (per gpu-arch × model-arch): compare to the committed hash. A
#      byte-exact logit_hash is GPU-arch + build specific, so baselines are keyed
#      by both; a missing entry is recorded (soft pass), a mismatch FAILS.
#
# A failure here is a TRIPWIRE — escalate to the 35B behavioral golden
# (tests/agentic-gate.sh) before rebaselining. See TODO.md "Tiny random-init
# fixtures + golden-output tripwire".
#
#   ./tests/fixture-golden-gate.sh            # check vs baselines
#   ./tests/fixture-golden-gate.sh --record   # (re)write baselines for this GPU
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

# shellcheck source=scripts/gpu-lock.sh
source ./scripts/gpu-lock.sh

BASELINES="tests/fixture-golden-baselines.txt"
ARCHS=(qwen3_5 qwen3_5_moe)
# Format axis: each runtime quant format has its own dequant kernel path, so a
# regression in one shows up as that cell's hash drifting. Support is
# arch-conditional (e.g. mq3 is gfx11+), so an unrunnable cell is a soft SKIP,
# not a failure — only a runnable-but-drifted cell fails.
FORMATS=(mq4 mq3 mq6 q8f16)
LEN=16
WARMUP=2
SEED=42

echo "fixture-golden-gate: building..."
cargo build --release -p hipfire-quantize --example fixture_golden -p hipfire-runtime >/dev/null || exit 2
Q="$ROOT/target/release/hipfire-quantize"
GOLD="$ROOT/target/release/examples/fixture_golden"

gpu_acquire "fixture-golden-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
trap 'gpu_release 2>/dev/null || true' EXIT

TMP="$(mktemp -d)"
trap 'gpu_release 2>/dev/null || true; rm -rf "$TMP"' EXIT

run_golden() { # arch hfq -> prints "<gpu_arch> <logit_hash>"
    local hfq="$1"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
        "$GOLD" "$hfq" --len "$LEN" --warmup "$WARMUP" 2>"$TMP/err" >"$TMP/out"
    local ga hh
    ga="$(grep -oE 'gfx[0-9a-f]+' "$TMP/err" | head -1)"
    hh="$(grep -oE 'logit_hash: 0x[0-9a-f]+' "$TMP/out" | grep -oE '0x[0-9a-f]+')"
    echo "${ga:-unknown} ${hh:-MISSING}"
}

lookup_baseline() { # gpu_arch model_arch format
    [ -f "$BASELINES" ] || return 0
    awk -v g="$1" -v m="$2" -v f="$3" '$1==g && $2==m && $3==f {print $4}' "$BASELINES" | head -1
}

fail=0          # drift or nondeterminism — a real regression signal
matched=0       # runnable cell == its committed baseline for this gpu-arch
nobaseline=0    # runnable cell, but no committed baseline for this gpu-arch
declare -a RECORDED=()
for arch in "${ARCHS[@]}"; do
    echo "== golden: $arch =="
    "$Q" --emit-fixture "$arch" --out "$TMP/$arch" --seed "$SEED" >/dev/null 2>&1
    for fmt in "${FORMATS[@]}"; do
        hfq="$TMP/$arch-$fmt.hfq"
        if ! "$Q" --input "$TMP/$arch" --output "$hfq" --format "$fmt" >/dev/null 2>&1; then
            echo "  SKIP $fmt: quantize unsupported on this build/arch"; continue
        fi

        read -r gpu_arch h1 <<<"$(run_golden "$hfq")"
        read -r _        h2 <<<"$(run_golden "$hfq")"
        if [ "$h1" = "MISSING" ]; then echo "  SKIP $fmt: not runnable here (no logit_hash)"; continue; fi

        # 1. Determinism (hard — a runnable cell that flips is a real bug).
        if [ "$h1" != "$h2" ]; then
            echo "  FAIL determinism $fmt: $h1 != $h2 (nondeterministic kernel)"
            fail=1; continue
        fi
        RECORDED+=("$gpu_arch $arch $fmt $h1")

        # 2. Baseline (hard — a runnable cell that drifts from committed fails).
        base="$(lookup_baseline "$gpu_arch" "$arch" "$fmt")"
        if [ "$RECORD" = 1 ]; then
            echo "  $fmt: $h1 ($gpu_arch) [record]"
        elif [ -z "$base" ]; then
            echo "  NOTE $fmt: no baseline for $gpu_arch — observed $h1 (run --record to add)"
            nobaseline=$((nobaseline + 1))
        elif [ "$base" != "$h1" ]; then
            echo "  FAIL drift $fmt: $h1 != baseline $base → escalate to 35B golden (agentic-gate.sh)"
            fail=1
        else
            echo "  OK $fmt: matches baseline ($h1)"
            matched=$((matched + 1))
        fi
    done
done

if [ "$RECORD" = 1 ]; then
    {
        echo "# gpu_arch  model_arch  format  logit_hash  (len=$LEN warmup=$WARMUP seed=$SEED mode=tf)"
        printf '%s\n' "${RECORDED[@]}"
    } >"$BASELINES"
    echo "fixture-golden-gate: wrote ${#RECORDED[@]} baselines to $BASELINES"
    exit 0
fi

# Exit-code contract (consumed by .githooks/pre-commit's two-tier wiring):
#   0  CONFIRMED unchanged — every runnable cell matched a committed baseline
#      for this gpu-arch (>=1 match, no drift, no missing baseline). The covered
#      forward paths did not change → safe to skip the heavy coherence battery.
#   1  DRIFT / nondeterminism — a covered kernel changed output. Escalate.
#   3  INCONCLUSIVE — cells ran but ≥1 has no baseline for this gpu-arch
#      (e.g. fleet box not yet recorded). Do NOT skip the heavy battery.
#   2  could not run anything (build/GPU). Do NOT skip the heavy battery.
if [ "$fail" != 0 ]; then
    echo "fixture-golden-gate: FAIL (drift/nondeterminism) → escalate to coherence-gate.sh + rebaseline if intended"
    exit 1
fi
if [ "$nobaseline" -gt 0 ]; then
    echo "fixture-golden-gate: INCONCLUSIVE ($nobaseline cell(s) with no baseline for this gpu-arch; matched $matched) — run --record"
    exit 3
fi
if [ "$matched" -gt 0 ]; then
    echo "fixture-golden-gate: PASS ($matched cell(s) match committed baselines)"
    exit 0
fi
echo "fixture-golden-gate: INCONCLUSIVE (no cells ran on this box)"
exit 2

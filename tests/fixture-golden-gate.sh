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

lookup_baseline() { # gpu_arch model_arch
    [ -f "$BASELINES" ] || return 0
    awk -v g="$1" -v m="$2" '$1==g && $2==m {print $3}' "$BASELINES" | head -1
}

fail=0
declare -a RECORDED=()
for arch in "${ARCHS[@]}"; do
    echo "== golden: $arch =="
    "$Q" --emit-fixture "$arch" --out "$TMP/$arch" --seed "$SEED" >/dev/null 2>&1
    "$Q" --input "$TMP/$arch" --output "$TMP/$arch.hfq" --format mq4 >/dev/null 2>&1 \
        || { echo "  FAIL: quantize $arch" >&2; fail=1; continue; }

    read -r gpu_arch h1 <<<"$(run_golden "$TMP/$arch.hfq")"
    read -r _        h2 <<<"$(run_golden "$TMP/$arch.hfq")"
    if [ "$h1" = "MISSING" ]; then echo "  FAIL: no logit_hash ($arch)"; fail=1; continue; fi

    # 1. Determinism.
    if [ "$h1" != "$h2" ]; then
        echo "  FAIL determinism: $h1 != $h2 (nondeterministic kernel on $arch)"
        fail=1; continue
    fi
    echo "  deterministic: $h1 ($gpu_arch)"
    RECORDED+=("$gpu_arch $arch $h1")

    # 2. Baseline.
    base="$(lookup_baseline "$gpu_arch" "$arch")"
    if [ "$RECORD" = 1 ]; then
        :  # baselines written below
    elif [ -z "$base" ]; then
        echo "  NOTE: no baseline for $gpu_arch/$arch — observed $h1 (run --record to add)"
    elif [ "$base" != "$h1" ]; then
        echo "  FAIL drift: $h1 != baseline $base → escalate to 35B golden (agentic-gate.sh)"
        fail=1
    else
        echo "  OK: matches baseline"
    fi
done

if [ "$RECORD" = 1 ]; then
    {
        echo "# gpu_arch  model_arch  logit_hash  (len=$LEN warmup=$WARMUP seed=$SEED)"
        printf '%s\n' "${RECORDED[@]}"
    } >"$BASELINES"
    echo "fixture-golden-gate: wrote ${#RECORDED[@]} baselines to $BASELINES"
    exit 0
fi

[ "$fail" = 0 ] && echo "fixture-golden-gate: PASS" || echo "fixture-golden-gate: FAIL"
exit "$fail"

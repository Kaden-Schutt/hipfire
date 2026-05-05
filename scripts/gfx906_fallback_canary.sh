#!/usr/bin/env bash
# gfx906 fallback regression canary.
#
# Addresses PR #127 review point 4: the wave64 FP16 hybrid path is the new
# default on gfx906, but the wave32 fallback (selected via HIPFIRE_FP16=0)
# must keep working — a future change could regress fallback behaviour
# without anyone noticing because the default-on path masks it.
#
# This script runs the canonical 512-tok prefill bench TWICE on the same
# binary: once on the default path (wave64 FP16 hybrid), once with
# HIPFIRE_FP16=0 (wave32 scalar fallback). It prints both numbers so a
# human (or CI on real hardware) can spot a regression on either side.
#
# Hardware requirement: gfx906 (MI50). On other arches the toggle is a
# no-op and both runs are identical.
#
# Usage:
#   scripts/gfx906_fallback_canary.sh [path/to/qwen3.5-9b.mq4]
#
# Defaults to ~/.hipfire/models/qwen3.5-9b.mq4 if unset.

set -u
MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
PREFILL_LEN="${HIPFIRE_CANARY_PREFILL:-512}"
PREFILL_RUNS="${HIPFIRE_CANARY_RUNS:-3}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "       set first arg or HIPFIRE_CANARY_MODEL to override" >&2
    exit 1
fi

# Build once — both runs share the binary.
cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_mq4 \
    >/tmp/gfx906_canary_build.log 2>&1 || {
    echo "BUILD FAILED — see /tmp/gfx906_canary_build.log" >&2
    exit 1
}

run_bench() {
    local label="$1"; shift
    # Capture last "tok/s:" line emitted by the prefill phase.
    local out
    out=$("$@" target/release/examples/bench_qwen35_mq4 "$MODEL" \
        --prefill "$PREFILL_LEN" --prefill-runs "$PREFILL_RUNS" \
        --warmup 1 --gen 0 2>&1) || {
        echo "  $label: BENCH FAILED" >&2
        echo "$out" | tail -20 >&2
        return 1
    }
    # SUMMARY line always emits prefill_tok_s=N.N regardless of --gen value.
    local toks
    toks=$(echo "$out" | grep -oE 'prefill_tok_s=[0-9.]+' | tail -1 | sed 's/prefill_tok_s=//')
    if [ -z "$toks" ]; then
        echo "  $label: could not parse tok/s" >&2
        echo "$out" | tail -20 >&2
        return 1
    fi
    printf '  %-32s %8s tok/s\n' "$label" "$toks"
}

echo "=== gfx906 prefill fallback canary ==="
echo "  model:         $MODEL"
echo "  prefill_len:   $PREFILL_LEN"
echo "  prefill_runs:  $PREFILL_RUNS"
echo
echo "  (last tok/s is the most-recent run; first run includes JIT cost)"
echo

run_bench "default (wave64 FP16 hybrid)" \
    env HIPFIRE_KV_MODE=asym3
run_bench "fallback (HIPFIRE_FP16=0)" \
    env HIPFIRE_KV_MODE=asym3 HIPFIRE_FP16=0

cat <<'EOF'

Expected on gfx906 (MI50, Qwen 3.5 9B, prefill_len=512):
  default                          ~141 tok/s   (PR #127, wave64 FP16 hybrid)
  fallback (HIPFIRE_FP16=0)        ~22  tok/s   (legacy wave64 scalar)

A drop on the default line means the wave64 hybrid path regressed.
A drop on the fallback line means the legacy wave64 scalar fallback regressed.
Either is a signal worth bisecting.
EOF

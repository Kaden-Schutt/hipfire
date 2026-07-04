#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny-quant matrix gate (GPU).
#
# Per model family: emit a seeded tiny random-init fixture → quantize to that
# family's loader-supported formats (+ a calibrated cell) → build a near-full-
# precision anchor → generate a tiny Hessian (collect) → score each candidate's
# KL divergence vs the anchor over a fixed synthetic token stream. Exercises the
# whole quant pipeline (quantizer → loader → kernels → output) without real
# checkpoints or a daemon.
#
# The Rust `tiny_quant` battery owns the pipeline + the per-cell verdict (it
# reads tests/tiny-quant-baselines.txt and drift-checks each gpu_arch×family×
# format KLD). This wrapper just holds the GPU lock and maps row status → exit.
#
#   ./tests/tiny-quant-gate.sh            # check vs committed baselines
#   ./tests/tiny-quant-gate.sh --record   # (re)write baselines for THIS gpu
#
# Exit: 0 = all cells pass, 1 = a cell failed (crash / non-finite / KLD drift),
#       2 = could not run (build / GPU lock / binary missing),
#       3 = cells ran but at least one was skipped/inconclusive.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECORD=0
[ "${1:-}" = "--record" ] && RECORD=1

HIPFIRE_GPULOCK_BIN="${HIPFIRE_BIN:-$(command -v hipfire 2>/dev/null || echo ./target/release/hipfire)}"
EVAL_BIN="${HIPFIRE_EVAL_BIN:-./target/release/hipfire-eval}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo "tiny-quant-gate: building..."
cargo build --release \
    -p hipfire-quantize --bin hipfire-quantize \
    -p hipfire-eval --bin hipfire-eval \
    -p hipfire-serving-core --example tiny_quant_probe >/dev/null || {
    echo "build failed" >&2
    exit 2
}

"$HIPFIRE_GPULOCK_BIN" gpu-lock acquire "tiny-quant-gate" --watch-pid "$$" || {
    echo "could not acquire GPU lock" >&2
    exit 2
}
OUT="$(mktemp -d)"
trap '"$HIPFIRE_GPULOCK_BIN" gpu-lock release 2>/dev/null || true; rm -rf "$OUT"' EXIT

# Record mode rewrites tests/tiny-quant-baselines.txt from observed KLDs.
[ "$RECORD" = 1 ] && export HIPFIRE_TINYQUANT_RECORD=1

# --no-cache: a tripwire must always run fresh.
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}" \
    "$EVAL_BIN" --battery tiny_quant --no-cache --out "$OUT" || {
    echo "eval run failed" >&2
    exit 2
}

RES="$OUT/results.jsonl"
[ -f "$RES" ] || {
    echo "no results.jsonl produced" >&2
    exit 2
}

# Summarize each cell; count fails.
fail=0
skip=0
while IFS= read -r line; do
    status="$(grep -oE '"status":"[a-z]+"' <<<"$line" | head -1 | cut -d'"' -f4)"
    case_id="$(grep -oE '"case_id":"[^"]+"' <<<"$line" | head -1 | cut -d'"' -f4)"
    reason="$(grep -oE '"reason":"[^"]*"' <<<"$line" | head -1 | cut -d'"' -f4)"
    printf '  %-6s %s%s\n' "$status" "$case_id" "${reason:+  — $reason}"
    [ "$status" = "fail" ] && fail=$((fail + 1))
    [ "$status" = "skip" ] && skip=$((skip + 1))
done <"$RES"

if [ "$RECORD" = 1 ]; then
    echo "tiny-quant-gate: recorded baselines for this GPU → tests/tiny-quant-baselines.txt"
    exit 0
fi
if [ "$fail" -gt 0 ]; then
    echo "tiny-quant-gate: FAIL ($fail cell(s) drifted/crashed)"
    exit 1
fi
if [ "$skip" -gt 0 ]; then
    echo "tiny-quant-gate: INCONCLUSIVE ($skip skipped cell(s))"
    exit 3
fi
echo "tiny-quant-gate: PASS"
exit 0

#!/usr/bin/env bash
# gfx906 wave64 FP16 hybrid vs wave32 fallback: logit divergence test.
#
# Addresses PR #127 review point 2: the coherence gate hard-fails only on
# panics/zero-tokens/timeouts; output divergence isn't a hard fail. This
# script gives a stronger correctness signal by comparing actual logit
# vectors from both dispatch paths.
#
# Runs `dump_logits_qwen35` twice on the SAME deterministic fake prompt:
#   - default path (wave64 FP16 hybrid on gfx906)
#   - HIPFIRE_FP16=0 (forces the wave32 scalar fallback)
# Then computes max-abs and mean-abs diff over the dumped f32 logits.
#
# Hardware requirement: gfx906 (MI50). On other arches HIPFIRE_FP16 still
# routes a different path on the FP16 fast-path families, so the script
# is a useful sanity check there too — but the divergence numbers below
# are calibrated against gfx906.
#
# Usage:
#   scripts/gfx906_logit_divergence.sh [path/to/qwen3.5-9b.mq4]
#
# Defaults to ~/.hipfire/models/qwen3.5-9b.mq4. Tunables:
#   HIPFIRE_DIVERGENCE_PREFILL=64    # prompt length
#   HIPFIRE_DIVERGENCE_TOL=1e-2      # max-abs-diff threshold (advisory)

set -u
MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
PREFILL_LEN="${HIPFIRE_DIVERGENCE_PREFILL:-64}"
TOL="${HIPFIRE_DIVERGENCE_TOL:-1e-2}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    exit 1
fi

# Build once.
cargo build --release --features deltanet -p hipfire-runtime --example dump_logits_qwen35 \
    >/tmp/divergence_build.log 2>&1 || {
    echo "BUILD FAILED — see /tmp/divergence_build.log" >&2
    exit 1
}

TMP_DIR=$(mktemp -d -t gfx906_divergence_XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

DUMP_DEFAULT="$TMP_DIR/logits_default.f32"
DUMP_FALLBACK="$TMP_DIR/logits_fallback.f32"

echo "=== gfx906 logit divergence ==="
echo "  model:        $MODEL"
echo "  prefill_len:  $PREFILL_LEN"
echo

echo "  [1/2] running default path (wave64 FP16 hybrid on gfx906)..."
HIPFIRE_KV_MODE=asym3 \
    target/release/examples/dump_logits_qwen35 "$MODEL" "$DUMP_DEFAULT" \
    --prefill "$PREFILL_LEN" >/tmp/divergence_default.log 2>&1 || {
    echo "  default path FAILED — see /tmp/divergence_default.log" >&2
    exit 1
}

echo "  [2/2] running fallback path (HIPFIRE_FP16=0)..."
HIPFIRE_KV_MODE=asym3 HIPFIRE_FP16=0 \
    target/release/examples/dump_logits_qwen35 "$MODEL" "$DUMP_FALLBACK" \
    --prefill "$PREFILL_LEN" >/tmp/divergence_fallback.log 2>&1 || {
    echo "  fallback path FAILED — see /tmp/divergence_fallback.log" >&2
    exit 1
}

# Diff the two f32 dumps. Use Python (always available) — Rust would need
# another crate. The numerics here are deliberately simple so the script
# stays portable; if you want fancier stats add them downstream.
python3 - "$DUMP_DEFAULT" "$DUMP_FALLBACK" "$TOL" <<'PY'
import sys, struct
default_path, fallback_path, tol_str = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(tol_str)

def load_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data), n

a, na = load_f32(default_path)
b, nb = load_f32(fallback_path)

if na != nb:
    print(f"  MISMATCH: default has {na} floats, fallback has {nb}")
    sys.exit(2)

max_abs = 0.0
sum_abs = 0.0
max_idx = 0
for i, (x, y) in enumerate(zip(a, b)):
    d = abs(x - y)
    if d > max_abs:
        max_abs = d
        max_idx = i
    sum_abs += d
mean_abs = sum_abs / na

# argmax stability: same top-1 token?
am_a = max(range(na), key=lambda i: a[i])
am_b = max(range(nb), key=lambda i: b[i])

print(f"  vocab elements:   {na}")
print(f"  max |a-b|:        {max_abs:.6e}  (at index {max_idx})")
print(f"  mean |a-b|:       {mean_abs:.6e}")
print(f"  default argmax:   {am_a}")
print(f"  fallback argmax:  {am_b}  ({'MATCH' if am_a == am_b else 'DIFFERENT'})")
print()

# Advisory thresholds, not hard fails — FP16 vs FP32 paths are expected
# to diverge by ~1e-3 .. 1e-2 in absolute logit space without that
# implying a correctness bug. The argmax-match check is the stronger
# signal: a top-1 token swap on a deterministic prompt is suspicious.
status = 0
if max_abs > tol:
    print(f"  ADVISORY: max |a-b|={max_abs:.6e} exceeds tol={tol:.0e}")
    print(f"            review whether the divergence is structural or just FP16 noise")
    status = 1
if am_a != am_b:
    print(f"  WARNING: argmax differs between paths — investigate")
    status = 1
sys.exit(status)
PY

rc=$?
echo
if [ $rc -eq 0 ]; then
    echo "  PASS: divergence within advisory threshold and argmax matches"
else
    echo "  CHECK: see advisory/warning above"
fi
exit $rc

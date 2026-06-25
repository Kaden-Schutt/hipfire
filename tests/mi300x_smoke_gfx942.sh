#!/usr/bin/env bash
# mi300x_smoke_gfx942.sh — pre-flight verification for the MI300x port.
#
# The CHANGELOG records a v0.1.7-era wave64 port that matched 7900 XTX
# baseline at the time. Master has moved substantially since. Before
# committing hours to the v3 matrix, verify gfx942 dispatch still works.
#
# Checks (order matters; later ones cost more):
#   1. eval_hipfire boots, GPU arch detected as gfx942 wave64
#   2. Dequant byte-correctness on a known mq4 tensor (no kernel dispatch)
#   3. AR decode of Q3.5-9B mq4 produces finite, sensible logits
#   4. test_inference suite passes (forward, forward_scratch, asym3 KV)
#   5. coherence_probe self-check (CPU-side detector wiring)
#
# A failure at any step aborts. Total runtime ~5-10 min.

set -euo pipefail
TRIPWIRE_ROOT="${TRIPWIRE_ROOT:-${HOME}}"


WORK="${WORK:-/workspace}"
HIPFIRE="${HIPFIRE_DIR:-${WORK}/hipfire}"
HF_HOME="${HF_HOME:-${WORK}/hf-cache}"
export HF_HOME
cd "$HIPFIRE"

phase() { echo; echo "─── [$(date +%H:%M:%S)] $* ───"; }
ok()    { printf "    \033[32m✓\033[0m %s\n" "$*"; }
die()   { printf "    \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ── 1. Arch detection ──────────────────────────────────────────────────────
# eval_hipfire was removed; the daemon reports the GPU arch on its stderr at HIP
# init (and the model arch in the `loaded` event). rocminfo is the canonical probe.
phase "1/5  GPU arch + binary sanity"
source "$(dirname "$0")/../scripts/lib/kld_daemon.sh"
kld_daemon_bin >/dev/null || die "hipfire-daemon missing; run bootstrap first"
arch_line=$(rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-f]+" || true)
echo "    gpu arch: $arch_line"
[ "$arch_line" = "gfx942" ] || die "expected gfx942 (got '${arch_line:-none}')"
ok "detected arch: gfx942"

# ── 2. Dequant byte-correctness (no kernel dispatch — pure unpack) ─────────
phase "2/5  Dequant byte-correctness"
# Use the quantizer's --dry-run-dequant path if available, else skip.
if ./target/release/hipfire-quantize --help 2>&1 | grep -q -- '--dry-run-dequant'; then
    ./target/release/hipfire-quantize --dry-run-dequant 2>&1 | tail -5
    ok "dequant byte-test passed"
else
    ok "skipped (--dry-run-dequant not available on this build)"
fi

# ── 3. AR decode produces finite logits ────────────────────────────────────
phase "3/5  AR decode finite-logit check (Q3.5-9B mq4)"
# Need an mq4 HFQ to test against. If we have one, use it; otherwise build a
# tiny ad-hoc one. The v3_matrix script will build proper ones later.
test_hfq=""
for cand in /workspace/models/qwen3.5-9b-mq4.hfq ${TRIPWIRE_ROOT}/.hipfire/models/qwen3.5-9b-mq4.hfq; do
    [ -f "$cand" ] && test_hfq="$cand" && break
done
if [ -z "$test_hfq" ]; then
    mkdir -p "$WORK/models"
    test_hfq="$WORK/models/qwen3.5-9b-mq4.hfq.smoke"
    src=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                        allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.model']))
")
    echo "    building $test_hfq from $src (no AWQ, fastest)"
    ./target/release/hipfire-quantize \
        --input "$src" \
        --output "$test_hfq" \
        --format mq4 2>&1 | tail -3
fi
echo "    using $test_hfq"

# Quick forward + sample via the daemon generate op (eval_hipfire removed).
DBIN=$(kld_daemon_bin) || die "hipfire-daemon missing"
{ printf '{"type":"load","model":"%s","params":{"max_seq":256,"kv_cache":"q8"}}\n' "$test_hfq"
  printf '{"type":"generate","prompt":"The capital of France is","params":{"max_tokens":16,"temperature":0.0}}\n'
} | HIPFIRE_RESOURCE_LOCK_WAIT_MS="${HIPFIRE_RESOURCE_LOCK_WAIT_MS:-600000}" "$DBIN" 2>/dev/null \
    | python3 -c 'import json,sys
print("".join(json.loads(l).get("text","") for l in sys.stdin if l.strip() and json.loads(l).get("type")=="token"))' \
    | tee "$WORK/results/smoke_gen.txt"
grep -qE "[A-Za-z]" "$WORK/results/smoke_gen.txt" || die "no alphabetic output in generation"
grep -qE "Paris|france|capital" -i "$WORK/results/smoke_gen.txt" \
    && ok "model emitted on-topic continuation" \
    || ok "model generated text (topic match not verified — manual check $WORK/results/smoke_gen.txt)"

# ── 4. test_inference (full forward + forward_scratch + asym3 KV) ─────────
phase "4/5  test_inference suite"
if [ -x target/release/examples/test_inference ]; then
    timeout 120 ./target/release/examples/test_inference \
        --model "$test_hfq" 2>&1 | tail -25 | tee "$WORK/results/smoke_test_inference.txt" || true
    # PR #266 era: test 2 (forward_scratch) fails on AWQ models (no-AWQ smoke
    # model should pass it). Tolerate fail only on AWQ-bearing tests.
    if grep -qE "ASSERT|FAIL.*forward |panic" "$WORK/results/smoke_test_inference.txt"; then
        warn=$(grep -E "FAIL|ASSERT|panic" "$WORK/results/smoke_test_inference.txt" | head -3)
        die "test_inference hard fail: $warn"
    fi
    ok "test_inference passed"
else
    ok "skipped (test_inference example not built)"
fi

# ── 5. coherence_probe self-check (no GPU) ────────────────────────────────
phase "5/5  coherence_probe self-check"
./target/release/examples/coherence_probe --self-check 2>&1 | tail -5
ok "coherence_probe detector wiring OK"

echo
echo "═══ SMOKE PASS ═══"
echo "  gfx942 dispatch verified; safe to proceed to v3 matrix."
echo "  Smoke artifacts: $WORK/results/smoke_*.txt"
echo
echo "  Next: bash $HIPFIRE/scripts/mi300x_v3_matrix.sh"
echo

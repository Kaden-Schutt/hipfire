#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kevin Read
# hipfire — see LICENSE and NOTICE in the project root.

# Pipeline-parallel coherence gate. Sister to coherence-gate.sh; exercises
# the pp>1 paths so the Stage 2b unification refactor has a regression
# baseline. Currently focused on qwen3.6-27b (the only model we've
# validated under PP on this hetero box: gfx906 + gfx1031).
#
# Three test cases:
#   1. PP=1 + MTP baseline   — single-gpu MTP τ snapshot. Reference τ
#                              that PP+MTP must match within 5%.
#   2. PP=2 + AR             — verifies pp=2 trunk path works,
#                              coherent text emitted.
#   3. PP=2 + MTP-head-load  — Stage 2a foundation: verifies head load
#                              succeeds and tokens emit. Before Stage 2b
#                              the daemon dispatches PP+MTP through the
#                              AR path (no spec); after Stage 2b lands
#                              it will run the hetero spec function.
#                              The gate records what each step produces.
#
# Required envs (auto-set if missing):
#   HIPFIRE_ALLOW_MIXED_ARCH=1   — gfx906 + gfx1031 mixed-arch override
#   HIPFIRE_PP_LAYERS=48,16      — manual layer split (no uniform mode
#                                  for 32GB/12GB VRAM disparity)
#
# Exit codes:
#   0  battery ran clean — review report
#   1  any test hit a hard error (panic / zero tokens / load OOM / τ regression)
#   2  build or env error
#
# Report destination: /tmp/coherence-pp-<timestamp>.md
#                     (or $HIPFIRE_COHERENCE_PP_OUT)

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
MODEL="${HIPFIRE_PP_MODEL:-/local/hipfire/qwen3.6-27b.mq4}"
MTP_HEAD="${HIPFIRE_PP_MTP_HEAD:-/data/hipfire/qwen3.6-27b-cvs16384.mtp}"
OUT="${HIPFIRE_COHERENCE_PP_OUT:-/tmp/coherence-pp-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
PROMPT="${HIPFIRE_PP_PROMPT:-Write a Python function to compute the sum of an array using a loop.}"
MAX_TOKENS="${HIPFIRE_PP_MAX_TOKENS:-64}"
TAU_REGRESSION_THRESHOLD="${HIPFIRE_PP_TAU_REGRESSION:-0.05}"  # fraction; 0.05 = 5%

# ── Rebuild daemon if relevant source is newer than the binary ────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs \
               crates/hipfire-arch-qwen35/src/mtp_spec.rs \
               crates/hipfire-arch-qwen35/src/mtp_head.rs \
               crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/multi_gpu.rs \
               crates/hipfire-runtime/src/mtp_mirror.rs \
               crates/hipfire-runtime/examples/daemon.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-pp: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "coherence-gate-pp: build failed" >&2
        exit 2
    fi
fi

# ── Required files present? ───────────────────────────────────────────
if [ ! -f "$MODEL" ]; then
    echo "coherence-gate-pp: model not found: $MODEL" >&2
    echo "  override via HIPFIRE_PP_MODEL=<path>" >&2
    exit 2
fi
if [ ! -f "$MTP_HEAD" ]; then
    echo "coherence-gate-pp: MTP head not found: $MTP_HEAD" >&2
    echo "  override via HIPFIRE_PP_MTP_HEAD=<path>" >&2
    exit 2
fi

# ── GPU lock ──────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-pp" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Init report ───────────────────────────────────────────────────────
prompt_md5=$(printf '%s' "$PROMPT" | md5sum | awk '{print $1}')
{
    echo "# Coherence gate PP — $(date -Iseconds)"
    echo
    echo "**Model:** \`$MODEL\`"
    echo "**MTP head:** \`$MTP_HEAD\`"
    echo "**Prompt:** \"$PROMPT\""
    echo "**Prompt md5:** \`$prompt_md5\`"
    echo "**Max tokens:** $MAX_TOKENS"
    echo "**τ regression threshold:** $TAU_REGRESSION_THRESHOLD"
    echo
    echo "---"
    echo
} > "$OUT"

hard_errors=0
baseline_tau=""  # filled in by test 1

# ── Test runner ───────────────────────────────────────────────────────
# Args: $1=test_id, $2=pp, $3=use_mtp(0/1), $4..N=extra "key":"val" pairs to splice into load params
run_test() {
    local test_id="$1" pp="$2" use_mtp="$3"
    shift 3
    local extra_params="${1:-}"  # already-quoted JSON fragments, comma-prefixed

    local mtp_param=""
    if [ "$use_mtp" -eq 1 ]; then
        mtp_param=",\"mtp_head\":\"$MTP_HEAD\""
    fi

    local in_file out_file
    in_file=$(mktemp -t "coh-pp-in-$test_id.XXXXXX.jsonl")
    out_file=$(mktemp -t "coh-pp-out-$test_id.XXXXXX.log")
    local prompt_json
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$PROMPT")
    # CRITICAL: repeat_penalty MUST be 1.0 for MTP tests. The daemon's
    # generate dispatch (daemon.rs:4703) bypasses MTP to AR when
    # repeat_penalty != 1.0 because the lossless MTP verify can't honor
    # a penalty without diverging from the trunk's argmax. With penalty
    # 1.05 (or anything > 1.0), MTP silently falls back to AR — making
    # the test useless for τ regression detection.
    cat > "$in_file" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096,"pp":$pp,"kv_mode":"asym3"$mtp_param$extra_params}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$MAX_TOKENS,"repeat_penalty":1.0}
{"type":"unload"}
JL

    echo "== $test_id (pp=$pp mtp=$use_mtp) =="
    local t0 t1 wall ec
    t0=$(date +%s.%N)
    env HIPFIRE_ALLOW_MIXED_ARCH=1 HIPFIRE_PP_LAYERS=48,16 \
        timeout 360 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    local done_line n_tokens panic status
    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|"type":"error"' "$out_file" | head -1)
    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    fi

    # Extract τ from the done event (generate_mtp's done emits it inline).
    # Two valid MTP dispatch labels in done events:
    #   "spec_path":"mtp"     — single-gpu MTP (generate_mtp / SpecPath::Mtp)
    #   "spec_path":"pp-mtp"  — Stage 2b PP+MTP (generate_pp_mtp / SpecPath::PpMtp)
    # If the daemon dispatched to AR instead, the done event won't contain
    # spec_path at all — record AR_FALLBACK so the gate doesn't silently
    # lose MTP coverage.
    local tau=""
    local mtp_dispatched="-"
    if [ "$use_mtp" -eq 1 ]; then
        if printf '%s' "$done_line" | grep -q '"spec_path":"pp-mtp"'; then
            mtp_dispatched="pp-mtp"
            tau=$(printf '%s' "$done_line" | grep -oE '"tau":[0-9.]+' | head -1 | cut -d: -f2)
        elif printf '%s' "$done_line" | grep -q '"spec_path":"mtp"'; then
            mtp_dispatched="mtp"
            tau=$(printf '%s' "$done_line" | grep -oE '"tau":[0-9.]+' | head -1 | cut -d: -f2)
        else
            mtp_dispatched="AR_FALLBACK"
            # Don't hard-fail; PP+MTP requests load fine but may bypass to
            # AR on configs PpMtp doesn't yet support (sampling, full-vocab
            # head). The test still validates that load succeeded and
            # tokens emitted.
        fi
    fi

    # Capture VRAM-after-load lines (added in Stage 2a daemon)
    local vram_lines
    vram_lines=$(grep -aE '^  dev [0-9]+ \(' "$out_file" | head -4)

    # Extract emitted text
    local text
    text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')

    # τ regression check — only meaningful when we have a baseline and this run produced τ
    if [ -n "$baseline_tau" ] && [ -n "$tau" ]; then
        local delta_pct
        delta_pct=$(python3 -c "b=$baseline_tau; t=$tau; print(f'{abs(b-t)/b:.4f}')" 2>/dev/null || echo "0")
        local thr_exceeded
        thr_exceeded=$(python3 -c "print(1 if $delta_pct > $TAU_REGRESSION_THRESHOLD else 0)")
        if [ "$thr_exceeded" -eq 1 ]; then
            status="$status; TAU_REGRESSION (baseline=$baseline_tau, this=$tau, delta=$(python3 -c "print(f'{$delta_pct*100:.1f}')")%)"
            hard_errors=$((hard_errors + 1))
        fi
    fi

    # Record baseline τ on test 1
    if [ -z "$baseline_tau" ] && [ -n "$tau" ]; then
        baseline_tau="$tau"
    fi

    {
        echo "## $test_id"
        echo
        echo "- pp: $pp, mtp: $use_mtp, wall: ${wall}s, status: **$status**"
        if [ "$use_mtp" -eq 1 ]; then
            echo "- mtp dispatch: \`$mtp_dispatched\`"
        fi
        if [ -n "$done_line" ]; then
            echo "- done event: \`$done_line\`"
        fi
        if [ -n "$tau" ]; then
            echo "- τ: \`$tau\`"
        fi
        if [ -n "$vram_lines" ]; then
            echo
            echo "**VRAM:**"
            echo
            echo '```'
            echo "$vram_lines"
            echo '```'
        fi
        echo
        if [ -n "$panic" ]; then
            echo '**PANIC/ERROR DETECTED:**'
            echo
            echo '```'
            echo "$panic"
            echo '```'
            echo
        fi
        echo '**Output:**'
        echo
        echo '```'
        echo "$text"
        echo '```'
        echo
        echo '---'
        echo
    } >> "$OUT"

    rm -f "$in_file" "$out_file"
}

# ── Run the matrix ────────────────────────────────────────────────────
# Test 1: pp=1 + mtp — establishes baseline τ for the regression check
run_test "pp1-mtp-baseline" 1 1

# Test 2: pp=2 + AR (no mtp head)
run_test "pp2-ar" 2 0

# Test 3: pp=2 + mtp-load (Stage 2a; pre-Stage 2b dispatch is AR-only)
run_test "pp2-mtp-load" 2 1

# ── Finish ────────────────────────────────────────────────────────────
{
    echo "## Summary"
    echo
    echo "- baseline τ (pp=1 + mtp): \`$baseline_tau\`"
    echo "- hard errors: $hard_errors"
} >> "$OUT"

echo
echo "coherence-pp report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"

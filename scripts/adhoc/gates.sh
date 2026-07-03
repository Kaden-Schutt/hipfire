#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# gates.sh — unified ship gate: coherence (correctness) + perf (A/B) +
# optional DFlash spec-decode battery, in ONE command with ONE summary.
#
# Semantics (per CLAUDE.md):
#   * Coherence is a HARD gate — panics / zero tokens / timeouts / pflash fail
#     → gate FAILS (exit 1). Wraps scripts/coherence-gate.sh.
#   * DFlash spec-decode battery is a HARD gate when it runs (auto when
#     spec-decode files changed vs the baseline, or forced with --dflash).
#     Wraps scripts/coherence-gate-dflash.sh.
#   * Perf is ADVISORY by default — a |Δ| ≥ 5% vs the baseline prints an
#     INVESTIGATE banner ("Δ ≥ 5% investigation rule") but does NOT fail the
#     gate. Pass --perf-strict to make a ≥5% *regression* fail (exit 3).
#     Wraps scripts/probe_commits.sh.
#
# Usage:
#   scripts/gates.sh                  # coherence + perf(vs HEAD~1) + auto-dflash
#   scripts/gates.sh --coherence-only # fast: coherence only, no perf/dflash
#   scripts/gates.sh --perf <ref>     # override perf baseline (default HEAD~1)
#   scripts/gates.sh --no-perf | --dflash | --no-dflash | --full | --perf-strict
#
# Env passes through unchanged to the wrapped scripts (HIPFIRE_MODELS_DIR,
# BENCH_MODEL, HIPFIRE_KV_MODE, HIPFIRE_PFLASH_*, HIPFIRE_DFLASH_*, …).
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)" || { echo "gates.sh: not a git repo" >&2; exit 2; }
cd "$ROOT"
G="$ROOT/scripts"

FULL=""; PERF=1; PERF_STRICT=0; PERF_BASE="HEAD~1"; DFLASH="auto"; PERF_THRESH=5.0

while [ $# -gt 0 ]; do
    case "$1" in
        --full)            FULL="--full" ;;
        --coherence-only)  PERF=0; DFLASH="off" ;;
        --no-perf)         PERF=0 ;;
        --perf)            PERF=1; if [ $# -ge 2 ] && [ "${2#-}" = "$2" ]; then PERF_BASE="$2"; shift; fi ;;
        --perf-strict)     PERF=1; PERF_STRICT=1 ;;
        --dflash)          DFLASH="on" ;;
        --no-dflash)       DFLASH="off" ;;
        -h|--help)         sed -n '7,30p' "$0"; exit 0 ;;
        *) echo "gates.sh: unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# Resolve the perf baseline to a SHA up front so a bad ref errors clearly.
PERF_BASE_SHA=""
if [ "$PERF" -eq 1 ]; then
    if ! PERF_BASE_SHA=$(git rev-parse --verify --quiet "${PERF_BASE}^{commit}"); then
        echo "gates.sh: perf baseline '$PERF_BASE' not found — use --no-perf or --perf <ref>" >&2
        exit 2
    fi
fi
HEAD_SHA=$(git rev-parse HEAD)
HEAD_SHORT=$(git rev-parse --short HEAD)

# Auto-detect spec-decode/DFlash changes → run the dflash battery too.
if [ "$DFLASH" = "auto" ]; then
    if git diff --name-only "${PERF_BASE_SHA:-HEAD~1}" HEAD 2>/dev/null \
        | grep -qiE 'speculative|dflash|spec_step|ddtree|mtp_|pflash'; then
        DFLASH="on"
    else
        DFLASH="off"
    fi
fi

echo "=== hipfire gates.sh ==="
echo "head=$HEAD_SHORT  perf=$([ $PERF -eq 1 ] && echo "on(base ${PERF_BASE})" || echo off)  dflash=$DFLASH  full=${FULL:-no}  perf_strict=$PERF_STRICT"
echo

COH_RC=0; DFL_RC="skip"; PERF_NOTE="skipped"; OVERALL=0

# ── [1] Coherence — HARD gate ──────────────────────────────────────────
echo "── coherence-gate.sh ${FULL} ──"
"$G/coherence-gate.sh" $FULL; COH_RC=$?
[ "$COH_RC" -ne 0 ] && OVERALL=1
echo "coherence: rc=$COH_RC $([ "$COH_RC" -eq 0 ] && echo PASS || echo FAIL)"
echo

# ── [2] DFlash spec-decode — HARD gate (conditional) ───────────────────
if [ "$DFLASH" = "on" ]; then
    echo "── coherence-gate-dflash.sh ${FULL} ──"
    "$G/coherence-gate-dflash.sh" $FULL; DFL_RC=$?
    [ "$DFL_RC" -ne 0 ] && OVERALL=1
    echo "dflash: rc=$DFL_RC $([ "$DFL_RC" -eq 0 ] && echo PASS || echo FAIL)"
    echo
fi

# ── [3] Perf A/B — ADVISORY (unless --perf-strict) ─────────────────────
if [ "$PERF" -eq 1 ]; then
    echo "── probe_commits.sh ${PERF_BASE_SHA:0:9} ${HEAD_SHORT} ──"
    PROBE_OUT=$("$G/probe_commits.sh" "$PERF_BASE_SHA" "$HEAD_SHA" 2>&1)
    echo "$PROBE_OUT"
    mapfile -t TOKS < <(echo "$PROBE_OUT" | grep -oE '[0-9.]+ tok/s' | grep -oE '^[0-9.]+')
    if [ "${#TOKS[@]}" -ge 2 ]; then
        BASE_TOK="${TOKS[0]}"; HEAD_TOK="${TOKS[1]}"
        DELTA=$(awk -v a="$BASE_TOK" -v b="$HEAD_TOK" 'BEGIN{ if(a>0) printf "%.2f",(b-a)/a*100; else print "nan" }')
        PERF_NOTE="base=${BASE_TOK} head=${HEAD_TOK} Δ=${DELTA}%"
        ABS_GE=$(awk -v d="$DELTA" -v t="$PERF_THRESH" 'BEGIN{ dd=(d<0)?-d:d; print (dd>=t)?1:0 }')
        REG_GE=$(awk -v d="$DELTA" -v t="$PERF_THRESH" 'BEGIN{ print (d<=-t)?1:0 }')
        if [ "$ABS_GE" -eq 1 ]; then
            echo "perf: ⚠ Δ=${DELTA}% crosses ±${PERF_THRESH}% — INVESTIGATE (CLAUDE.md Δ≥5% rule):"
            echo "      warm cache+DPM? median of 3–5 fresh-process runs? byte-identical prompt (md5)? kernel occupancy?"
            if [ "$PERF_STRICT" -eq 1 ] && [ "$REG_GE" -eq 1 ]; then
                echo "perf: --perf-strict → regression ≥${PERF_THRESH}% FAILS the gate"
                OVERALL=3
            fi
        else
            echo "perf: Δ=${DELTA}% within ±${PERF_THRESH}% band — OK"
        fi
    else
        PERF_NOTE="UNPARSEABLE — check probe output (build/bench fail?)"
        echo "perf: could not parse two tok/s numbers — advisory, not blocking; review above"
    fi
    echo
fi

# ── Summary ────────────────────────────────────────────────────────────
echo "=== gates summary (head $HEAD_SHORT) ==="
echo "  coherence : $([ "$COH_RC" -eq 0 ] && echo PASS || echo "FAIL(rc=$COH_RC)")"
echo "  dflash    : $DFL_RC"
echo "  perf      : $PERF_NOTE (advisory$([ "$PERF_STRICT" -eq 1 ] && echo ', strict'))"
[ "$OVERALL" -eq 0 ] && echo "  GATES: PASS" || echo "  GATES: FAIL (exit $OVERALL)"
exit "$OVERALL"

#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery — DFlash + DDTree variant.
#
# Sibling to coherence-gate.sh (which only exercises target-only AR decode
# via the daemon binary). This battery exercises the SPECULATIVE decode
# code paths — spec_step_dflash and spec_step_ddtree_batched — which the
# AR-only gate misses entirely.
#
# Why a separate gate exists: Path A (DDTree slow-path-kill, 2026-04-23,
# reverted in 6c84b13) shipped to a smoke that LOOKED great on stats
# (+120% tok/s, +79% τ, sd=0.15) but actually produced "numbers(numbers
# (numbers(..." forever — a degenerate token attractor where 100% draft
# acceptance comes from the model being stuck on a single token. Pure-stat
# gates (speed-gate, τ-gate, even short-prompt PPL on AR) DO NOT catch
# this. The token-distribution check below does.
#
# Hard-fail conditions (block commit):
#   - dflash_spec_demo non-zero exit / panic / zero emitted tokens
#   - max_token_frequency / total > 0.40 in the first 256 emitted tokens
#     (single-token attractor — Path A's failure mode)
#   - unique_token_count / total < 0.30 (low-entropy loop)
#
# Soft fail (write to report, don't block):
#   - any other output change. Reviewer reads the report before committing.
#
# Exit codes:
#   0  battery ran clean
#   1  hard error (panic / zero tokens / token attractor detected)
#   2  build or environment error
#
# Modes:
#   ./scripts/coherence-gate-dflash.sh          # short — 4 tests, ~2-3 min
#   ./scripts/coherence-gate-dflash.sh --fast   # 2 tests (1 prose + 1 code, dflash only) — ~1 min
#   ./scripts/coherence-gate-dflash.sh --full   # add ddtree b22-k4 + b8-k2 — ~6-8 min
#
# --fast is for pre-commit on $SPEC_HOTSPOT match (down from full short battery).
# Force-full via HIPFIRE_FORCE_SPEC_GATE=1.

set -u
cd "$(dirname "$0")/.."

FULL=0
FAST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --fast) FAST=1 ;;
        -h|--help) sed -n '3,32p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$FAST" -eq 1 ] && [ "$FULL" -eq 1 ]; then
    echo "coherence-gate-dflash: --fast and --full are mutually exclusive" >&2
    exit 2
fi

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
# Target/draft resolution. Honor explicit env overrides; otherwise probe the
# qwen3.5 names (historical) then the qwen3.6 names (current) — dflash spec
# decode works on either generation, the gate just needs a target+draft pair.
# The hardcoded qwen3.5-only paths used to leave the gate SKIPPED on machines
# that only ship the qwen3.6 27B (e.g. this one), which is why DFlash changes
# couldn't be attractor-validated.
TARGET_27B="${HIPFIRE_DFLASH_TARGET:-}"
DRAFT_27B="${HIPFIRE_DFLASH_DRAFT:-}"
if [ -z "$TARGET_27B" ]; then
    for cand in qwen3.5-27b.mq4 qwen3.6-27b.mq4; do
        if [ -f "$MODELS_DIR/$cand" ]; then TARGET_27B="$MODELS_DIR/$cand"; break; fi
    done
    [ -z "$TARGET_27B" ] && TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
fi
if [ -z "$DRAFT_27B" ]; then
    for cand in qwen35-27b-dflash.mq4 qwen35-27b-dflash-mq4.hfq qwen36-27b-dflash-mq4.hfq qwen36-27b-dflash-mq4.hf4; do
        if [ -f "$MODELS_DIR/$cand" ]; then DRAFT_27B="$MODELS_DIR/$cand"; break; fi
    done
    [ -z "$DRAFT_27B" ] && DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
fi
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-dflash-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-240}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild dflash_spec_demo if any relevant source is newer ──────────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/dflash.rs crates/hipfire-arch-qwen35/src/speculative.rs \
               crates/hipfire-runtime/src/ddtree.rs crates/hipfire-runtime/examples/dflash_spec_demo.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-dflash: rebuilding dflash_spec_demo..."
    if ! cargo build --release --example dflash_spec_demo --features deltanet >&2; then
        echo "coherence-gate-dflash: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-dflash" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Prompt fixtures ───────────────────────────────────────────────────────
PROSE_PROMPT="The Roman Empire, at its height, stretched from the windswept moors of northern Britain to the sands of the Arabian peninsula. Its decline was not a single event but a long slow unraveling that took centuries. Several factors contributed to this gradual collapse. The first and perhaps most important was"

CODE_PROMPT='from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'

# ── Test matrix ───────────────────────────────────────────────────────────
# Format: "label|mode|prompt_var|max_tokens|extra_args"
#   mode = ar | dflash | ddtree-b12-k2 | ddtree-b22-k4 | ddtree-b8-k2
#   prompt_var = PROSE_PROMPT | CODE_PROMPT
SHORT_TESTS=(
    "27b-dflash-prose|dflash|PROSE_PROMPT|192"
    "27b-dflash-code|dflash|CODE_PROMPT|128"
    "27b-ddtree-b12-prose|ddtree-b12-k2|PROSE_PROMPT|192"
    "27b-ddtree-b12-code|ddtree-b12-k2|CODE_PROMPT|128"
)
# Fast mode drops the ddtree variants; dflash alone catches the Path A
# single-token attractor regression class. ~1 min wall vs ~3 min for short.
FAST_TESTS=(
    "27b-dflash-prose|dflash|PROSE_PROMPT|192"
    "27b-dflash-code|dflash|CODE_PROMPT|128"
)
FULL_EXTRA=(
    "27b-ddtree-b22-prose|ddtree-b22-k4|PROSE_PROMPT|192"
    "27b-ddtree-b8-prose|ddtree-b8-k2|PROSE_PROMPT|192"
)
if [ "$FAST" -eq 1 ]; then
    tests=("${FAST_TESTS[@]}")
else
    tests=("${SHORT_TESTS[@]}")
    [ "$FULL" -eq 1 ] && tests+=("${FULL_EXTRA[@]}")
fi

# ── Run ───────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Coherence battery — DFlash / DDTree"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
    echo "- kv_mode: q8"
    echo "- target: $TARGET_27B"
    echo "- draft:  $DRAFT_27B"
    echo
    echo "Hard-fail thresholds (three-tier, see CLAUDE.md DFlash Coherence Gate):"
    echo "  Tier 1 (first 128): unique_token_ratio < 0.15 OR max_single_token_frequency > 0.50"
    echo "  Tier 2 (last 128):  unique_token_ratio < 0.30 OR max_single_token_frequency > 0.50"
    echo "  Tier 3 (full, SOFT flag — human eyeball, not commit-blocking):"
    echo "          consecutive-3gram repetition density > 0.50 in final half"
    echo "          OR full-output unique_token_ratio < 0.10"
    echo "Plus: zero tokens / panic. (Path A failure mode — see commit 6c84b13.)"
    echo
} > "$OUT"

# Skip everything if 27B model + draft aren't both present.
if [ ! -f "$TARGET_27B" ] || [ ! -f "$DRAFT_27B" ]; then
    {
        echo "## SKIPPED — 27B target or draft model not found"
        echo
        echo "- target present: $( [ -f "$TARGET_27B" ] && echo yes || echo no )"
        echo "- draft present:  $( [ -f "$DRAFT_27B" ] && echo yes || echo no )"
        echo
        echo "DFlash/DDTree coherence skipped. Re-stage models or set"
        echo "\`HIPFIRE_MODELS_DIR\` and re-run."
    } >> "$OUT"
    echo "coherence-gate-dflash: 27B models not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# Token-attractor detector. Implements the three-tier intent documented in
# CLAUDE.md ("DFlash Coherence Gate"). Attractors manifest in two forms:
# (1) single-token loops visible in the FIRST 128 tokens, and (2) block-level
# structural loops (5+ token sequences repeating) that only appear LATER in
# generation. The detector trims at the first end-of-text token, then applies:
#
#   Tier 1 — FIRST 128 tokens (HARD fail):
#     unique_token_ratio < 0.15  OR  max_single_token_frequency > 0.50
#     Catches Path-A-class single-token attractors ("numbers(numbers(...",
#     unique≈0.05/max≈0.60) without false-positiving sentence-level repetition.
#
#   Tier 2 — LAST 128 tokens (HARD fail):
#     unique_token_ratio < 0.30  OR  max_single_token_frequency > 0.50
#     Catches block-level structural loops that pass Tier 1 (diverse early
#     window) but collapse later — e.g. the m-fold drift case in CLAUDE.md
#     (τ=8.98 passed first-128 but emitted a 47-token-vocab tail).
#
#   Tier 3 — FULL output (SOFT flag, human eyeball, NOT commit-blocking):
#     consecutive-3gram repetition density > 0.50 in the final half
#     OR full-output unique_token_ratio < 0.10
#     Flags structural code loops even when both hard windows pass.
#
# A legacy "soft_warn" (paragraph-level repetition in the first window) is
# retained so existing report wording stays meaningful.
#
# Qwen3.5 EOT token IDs: 248044 (<|endoftext|>) + 248046 (<|im_end|>).
DETECT_PY=$(cat <<'PYEOF'
import sys, re, json, collections
EOT_IDS = {248044, 248046}

def ratios(window):
    """unique_ratio, max_freq for a token window."""
    counter = collections.Counter(window)
    total = len(window)
    unique = len(counter)
    max_tok, max_count = counter.most_common(1)[0]
    return unique / total, max_count / total, unique, total, max_tok, max_count

def trigram_repeat_density(seq):
    """Fraction of consecutive overlapping 3-grams that repeat a previously
    seen 3-gram. Captures block-level structural loops (5+ token blocks show
    up as many repeating 3-grams)."""
    if len(seq) < 6:
        return 0.0
    grams = [tuple(seq[i:i+3]) for i in range(len(seq) - 2)]
    seen = set()
    repeats = 0
    for g in grams:
        if g in seen:
            repeats += 1
        else:
            seen.add(g)
    return repeats / len(grams)

out = sys.stdin.read()
m = re.search(r"DFlash tokens: \[([^\]]+)\]", out)
ar_m = re.search(r"AR tokens: \[([^\]]+)\]", out)
src = m or ar_m
if not src:
    print(json.dumps({"ok": False, "reason": "no_tokens_line"}))
    sys.exit(0)
toks = [int(x) for x in src.group(1).split(",") if x.strip()]
if not toks:
    print(json.dumps({"ok": False, "reason": "zero_tokens"}))
    sys.exit(0)
# Trim at first EOT.
trimmed = toks
for i, t in enumerate(toks):
    if t in EOT_IDS:
        trimmed = toks[:i]
        break
# Tier 1 — first 128 of the pre-EOT window.
window = trimmed[:128]
if len(window) < 16:
    # Too short to judge; accept as OK (clean early termination is fine).
    print(json.dumps({
        "ok": True, "total": len(window), "reason": "short_window_ok",
    }))
    sys.exit(0)
u1, f1, unique, total, max_tok, max_count = ratios(window)
# Tier 1 hard fail: Path-A-class single-token attractors.
t1_hard = f1 > 0.50 or u1 < 0.15
# Legacy soft warn (paragraph-level repetition in the first window).
soft_warn = (f1 > 0.40 or u1 < 0.30) and not t1_hard

# Tier 2 — LAST 128 of the pre-EOT window (block-level attractors that pass
# the early window). Only evaluated when there are enough tokens to judge.
t2_hard = False
t2 = None
if len(trimmed) >= 16:
    last = trimmed[-128:]
    u2, f2, _, t2n, _, _ = ratios(last)
    t2_hard = u2 < 0.30 or f2 > 0.50
    t2 = {"total": t2n, "unique_ratio": round(u2, 3), "max_freq": round(f2, 3)}

# Tier 3 — FULL output structural loop (SOFT flag, human eyeball only).
#   3gram repeat density > 0.50 over the FINAL HALF, OR full-output unique < 0.10
second_half = trimmed[len(trimmed) // 2:]
gram_density = trigram_repeat_density(second_half)
full_u, _, _, _, _, _ = ratios(trimmed) if trimmed else (1.0, 0, 0, 0, 0, 0)
tier3_warn = gram_density > 0.50 or full_u < 0.10

hard_fail = t1_hard or t2_hard
print(json.dumps({
    "ok": not hard_fail,
    "t1_hard": t1_hard,
    "t2_hard": t2_hard,
    "soft_warn": soft_warn,
    "tier3_warn": tier3_warn,
    "total": total, "unique": unique,
    "unique_ratio": round(u1, 3),
    "max_freq": round(f1, 3),
    "max_tok": max_tok, "max_count": max_count,
    "tier2": t2,
    "gram_density": round(gram_density, 3),
    "full_unique_ratio": round(full_u, 3),
}))
PYEOF
)

for entry in "${tests[@]}"; do
    IFS='|' read -r label mode prompt_var max_tok <<< "$entry"
    case "$prompt_var" in
        PROSE_PROMPT) prompt="$PROSE_PROMPT" ;;
        CODE_PROMPT)  prompt="$CODE_PROMPT" ;;
        *) echo "unknown prompt_var: $prompt_var" >&2; exit 2 ;;
    esac
    case "$mode" in
        ar)            extra=(--ar-baseline) ;;
        dflash)        extra=() ;;
        ddtree-b12-k2) extra=(--ddtree-batched --ddtree-budget 12 --ddtree-topk 2) ;;
        ddtree-b22-k4) extra=(--ddtree-batched --ddtree-budget 22 --ddtree-topk 4) ;;
        ddtree-b8-k2)  extra=(--ddtree-batched --ddtree-budget  8 --ddtree-topk 2) ;;
        *) echo "unknown mode: $mode" >&2; exit 2 ;;
    esac

    echo "== $label =="
    out_file="/tmp/cohdf_out_$$.log"
    t0=$(date +%s.%N)
    timeout "$CASE_TIMEOUT" "$EXE" \
        --target "$TARGET_27B" --draft "$DRAFT_27B" \
        --prompt "$prompt" --max "$max_tok" --ctx 2048 \
        --kv-mode q8 --no-chatml \
        --state-quant "${HIPFIRE_GATE_STATE_QUANT:-q8}" \
        "${extra[@]}" \
        > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    detect=$(python3 -c "$DETECT_PY" < "$out_file")
    detect_ok=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ok',False))")
    detect_warn=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('soft_warn',False))")
    detect_t3=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('tier3_warn',False))")

    status="OK"
    if [ "$ec" -ne 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_ok" != "True" ]; then
        # Tier 1 (first 128) or Tier 2 (last 128) hard fail.
        status="HARD_ERROR (token attractor: $detect)"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_t3" = "True" ]; then
        # Tier 3 (full-output structural loop) — soft flag, human eyeball.
        status="FLAG (Tier-3 structural 3gram loop — soft, needs human eyeball)"
    elif [ "$detect_warn" = "True" ]; then
        status="WARN (paragraph-level repetition — soft, not blocking)"
    fi

    # Pull stats lines (emitted/τ/cycles/accept_rate) for the report.
    stats=$(grep -aE '^emitted:|^cycles:|^accept_rate:' "$out_file" | head -3)

    {
        echo "## $label ($mode)"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        echo "- detector: \`$detect\`"
        if [ -n "$stats" ]; then
            echo "- stats:"
            echo '  ```'
            echo "$stats" | sed 's/^/  /'
            echo '  ```'
        fi
        if [ -n "$panic" ]; then
            echo
            echo '**PANIC/ERROR:**'
            echo
            echo '```'
            echo "$panic"
            echo '```'
        fi
        echo
        echo '**Output:**'
        echo
        echo '```'
        sed -n '/--- OUTPUT ---/,/-------------/p' "$out_file" \
            | sed '1d;$d' \
            | head -40
        echo '```'
        echo
    } >> "$OUT"

    rm -f "$out_file"
done

echo
echo "coherence report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"
exit 0

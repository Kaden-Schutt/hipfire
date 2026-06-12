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
#   ./tests/coherence-gate-dflash.sh          # short — 4 tests, ~2-3 min
#   ./tests/coherence-gate-dflash.sh --fast   # 2 tests (1 prose + 1 code, dflash only) — ~1 min
#   ./tests/coherence-gate-dflash.sh --full   # add ddtree b22-k4 + b8-k2 — ~6-8 min
#
# --fast is for pre-commit on $SPEC_HOTSPOT match (down from full short battery).
# Force-full via HIPFIRE_FORCE_SPEC_GATE=1.
#
# Set HIPFIRE_DFLASH_AR_PARITY=1 to rerun DFlash cases through --ar-baseline
# and hard-fail on token-list mismatch. This is stricter than the default
# coherence detector and is used while proving rollback parity.

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
HIPFIRE_HOME="${HIPFIRE_DIR:-$HOME/.hipfire}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HIPFIRE_HOME/models}"
DFLASH_DIR="${HIPFIRE_DFLASH_DIR:-$HIPFIRE_HOME/drafts}"
TARGET_27B=""
DRAFT_27B=""
PAIR_27B=""
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-dflash-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-240}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

try_27b_pair() {
    local label="$1"
    local target="$2"
    local draft="$3"
    if [ -f "$target" ] && [ -f "$draft" ]; then
        PAIR_27B="$label"
        TARGET_27B="$target"
        DRAFT_27B="$draft"
        return 0
    fi
    return 1
}

select_27b_pair() {
    try_27b_pair "qwen3.6-27b-mq4+dflash" \
        "$MODELS_DIR/qwen3.6-27b-mq4.hfq" \
        "$DFLASH_DIR/qwen3.6-27b-mq4.dflash.hfq" && return 0
    try_27b_pair "qwen3.5-27b-mq4+dflash" \
        "$MODELS_DIR/qwen3.5-27b-mq4.hfq" \
        "$DFLASH_DIR/qwen3.5-27b-mq4.dflash.hfq" && return 0
    try_27b_pair "qwen3.6-27b-mq3+dflash" \
        "$MODELS_DIR/qwen3.6-27b-mq3.hfq" \
        "$DFLASH_DIR/qwen3.6-27b-mq3.dflash.hfq" && return 0
    try_27b_pair "qwen3.5-27b-mq3+dflash" \
        "$MODELS_DIR/qwen3.5-27b-mq3.hfq" \
        "$DFLASH_DIR/qwen3.5-27b-mq3.dflash.hfq" && return 0
    return 1
}

select_27b_pair || true

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
AR_PARITY="${HIPFIRE_DFLASH_AR_PARITY:-0}"

{
    echo "# Coherence battery — DFlash / DDTree"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
    echo "- kv_mode: q8"
    echo "- ar_parity: $AR_PARITY"
    echo "- pair:   ${PAIR_27B:-not found}"
    echo "- models_dir: $MODELS_DIR"
    echo "- dflash_dir: $DFLASH_DIR"
    echo "- target: $TARGET_27B"
    echo "- draft:  $DRAFT_27B"
    echo
    echo "Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.40,"
    echo "unique_token_ratio < 0.30 (token-attractor detection — see Path A"
    echo "failure mode in commit 6c84b13)."
    echo
} > "$OUT"

# Skip everything if 27B model + draft aren't both present.
if [ ! -f "$TARGET_27B" ] || [ ! -f "$DRAFT_27B" ]; then
    {
        echo "## SKIPPED — 27B target or draft model not found"
        echo
        echo "- target present: $( [ -f "$TARGET_27B" ] && echo yes || echo no )"
        echo "- draft present:  $( [ -f "$DRAFT_27B" ] && echo yes || echo no )"
        echo "- models_dir: $MODELS_DIR"
        echo "- dflash_dir: $DFLASH_DIR"
        echo
        echo "DFlash/DDTree coherence skipped. Re-stage target models under"
        echo "\`HIPFIRE_MODELS_DIR\` and DFlash sidecars under \`HIPFIRE_DFLASH_DIR\`,"
        echo "or use the default \`$HIPFIRE_HOME/models\` and \`$HIPFIRE_HOME/drafts\`."
    } >> "$OUT"
    echo "coherence-gate-dflash: 27B models not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# Token-attractor detector. Targets the Path A failure mode specifically:
# single-token attractor mid-generation that would otherwise pass τ/tok-s
# stat gates. Looks at the FIRST 128 tokens up to (but not including) the
# first end-of-text token. Post-EOT output (model spamming "#" after a
# clean function close) is degenerate but NOT the failure class we're
# guarding against — generation has already finished by that point.
#
# Thresholds are calibrated to catch Path A's "numbers(numbers(..." (where
# unique_ratio ≈ 0.05, max_freq ≈ 0.60) without false-positiving sentence-
# level repetition (where the early window is still diverse).
#
# Qwen3.5 EOT token IDs: 248044 (<|endoftext|>) + 248046 (<|im_end|>).
DETECT_PY=$(cat <<'PYEOF'
import sys, re, json, collections
EOT_IDS = {248044, 248046}
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
# Apply detector to the first 128 tokens of the pre-EOT window.
window = trimmed[:128]
if len(window) < 16:
    # Too short to judge; accept as OK (clean early termination is fine).
    print(json.dumps({
        "ok": True, "total": len(window), "reason": "short_window_ok",
    }))
    sys.exit(0)
counter = collections.Counter(window)
unique = len(counter)
total = len(window)
unique_ratio = unique / total
max_tok, max_count = counter.most_common(1)[0]
max_freq = max_count / total
# Two-tier thresholds:
#   Hard block (commit-blocking): only on Path-A-class single-token
#     attractors (max_freq > 0.50 OR unique_ratio < 0.15). These are
#     unrecoverable — same token >50% of generation = degenerate loop.
#   Soft warn (printed in report, no exit code): paragraph-level
#     repetition (unique_ratio < 0.30 or max_freq > 0.40). Pre-existing
#     DDTree-b12 prose has this and isn't caused by Path B work.
hard_fail = max_freq > 0.50 or unique_ratio < 0.15
soft_warn = (max_freq > 0.40 or unique_ratio < 0.30) and not hard_fail
print(json.dumps({
    "ok": not hard_fail,
    "soft_warn": soft_warn,
    "total": total, "unique": unique,
    "unique_ratio": round(unique_ratio, 3),
    "max_freq": round(max_freq, 3),
    "max_tok": max_tok, "max_count": max_count,
}))
PYEOF
)

PARITY_PY=$(cat <<'PYEOF'
import sys, re, json
if len(sys.argv) != 3:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
def tokens(path, label):
    out = open(path, "rb").read().decode("utf-8", "replace")
    m = re.search(rf"{label} tokens: \[([^\]]+)\]", out)
    if not m:
        return None
    return [int(x) for x in m.group(1).split(",") if x.strip()]
ar = tokens(sys.argv[1], "AR")
df = tokens(sys.argv[2], "DFlash")
if ar is None:
    print(json.dumps({"ok": False, "reason": "missing_ar_tokens"}))
elif df is None:
    print(json.dumps({"ok": False, "reason": "missing_dflash_tokens"}))
elif ar == df:
    print(json.dumps({"ok": True, "tokens": len(df)}))
else:
    first = next((i for i, (a, d) in enumerate(zip(ar, df)) if a != d), min(len(ar), len(df)))
    lo = max(0, first - 4)
    hi = min(max(len(ar), len(df)), first + 5)
    print(json.dumps({
        "ok": False,
        "reason": "token_mismatch",
        "first_mismatch": first,
        "ar_len": len(ar),
        "dflash_len": len(df),
        "ar_token": ar[first] if first < len(ar) else None,
        "dflash_token": df[first] if first < len(df) else None,
        "window_start": lo,
        "ar_window": ar[lo:min(hi, len(ar))],
        "dflash_window": df[lo:min(hi, len(df))],
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
        "${extra[@]}" \
        > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    detect=$(python3 -c "$DETECT_PY" < "$out_file")
    detect_ok=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ok',False))")
    detect_warn=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('soft_warn',False))")
    parity="not_requested"
    ar_out_file=""
    if [ "$AR_PARITY" = "1" ] && [ "$mode" = "dflash" ]; then
        ar_out_file="/tmp/cohdf_ar_$$.log"
        timeout "$CASE_TIMEOUT" "$EXE" \
            --target "$TARGET_27B" --draft "$DRAFT_27B" \
            --prompt "$prompt" --max "$max_tok" --ctx 2048 \
            --kv-mode q8 --no-chatml \
            --ar-baseline \
            > "$ar_out_file" 2>&1
        ar_ec=$?
        ar_panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$ar_out_file" | head -1)
        if [ "$ar_ec" -ne 0 ] || [ -n "$ar_panic" ]; then
            parity="{\"ok\": false, \"reason\": \"ar_run_failed\", \"exit\": $ar_ec}"
        else
            parity=$(python3 -c "$PARITY_PY" "$ar_out_file" "$out_file")
        fi
    fi

    status="OK"
    if [ "$ec" -ne 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_ok" != "True" ]; then
        status="HARD_ERROR (token attractor: $detect)"
        hard_errors=$((hard_errors + 1))
    elif [ "$detect_warn" = "True" ]; then
        status="WARN (paragraph-level repetition — soft, not blocking)"
    fi
    parity_ok=$(echo "$parity" | python3 -c "import sys,json; s=sys.stdin.read().strip(); print('skip' if s == 'not_requested' else json.loads(s).get('ok', False))")
    if [ "$parity_ok" = "False" ]; then
        status="HARD_ERROR (AR parity: $parity)"
        hard_errors=$((hard_errors + 1))
    fi

    # Pull stats lines (emitted/τ/cycles/rollback/accept_rate) for the report.
    stats=$(grep -aE '^emitted:|^cycles:|^rollback_parity:|^accept_rate:' "$out_file" | head -4)

    {
        echo "## $label ($mode)"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        echo "- detector: \`$detect\`"
        echo "- ar_parity: \`$parity\`"
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
    [ -n "$ar_out_file" ] && rm -f "$ar_out_file"
done

echo
echo "coherence report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"
exit 0

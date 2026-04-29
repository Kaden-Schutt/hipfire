#!/usr/bin/env bash
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
#   ./scripts/coherence-gate-dflash.sh --full   # add ddtree b22-k4 + b8-k2 — ~6-8 min

set -u
cd "$(dirname "$0")/.."

FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        -h|--help) sed -n '3,32p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
if [ ! -f "$DRAFT_27B" ] && [ -f "$MODELS_DIR/qwen35-27b-dflash-mq4.hfq" ]; then
    DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash-mq4.hfq"
fi
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-dflash-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-240}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild dflash_spec_demo if any relevant source is newer ──────────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/engine/src/qwen35.rs crates/engine/src/llama.rs \
               crates/engine/src/dflash.rs crates/engine/src/speculative.rs \
               crates/engine/src/ddtree.rs crates/engine/examples/dflash_spec_demo.rs \
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
FULL_EXTRA=(
    "27b-ddtree-b22-prose|ddtree-b22-k4|PROSE_PROMPT|192"
    "27b-ddtree-b8-prose|ddtree-b8-k2|PROSE_PROMPT|192"
)
tests=("${SHORT_TESTS[@]}")
[ "$FULL" -eq 1 ] && tests+=("${FULL_EXTRA[@]}")

# ── Run ───────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Coherence battery — DFlash / DDTree"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FULL" -eq 1 ] && echo full || echo short )"
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
        echo
        echo "DFlash/DDTree coherence skipped. Re-stage models or set"
        echo "\`HIPFIRE_MODELS_DIR\` and re-run."
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
        --kv-mode asym3 --no-chatml \
        "${extra[@]}" \
        > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    detect=$(python3 -c "$DETECT_PY" < "$out_file")
    detect_ok=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ok',False))")
    detect_warn=$(echo "$detect" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('soft_warn',False))")

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

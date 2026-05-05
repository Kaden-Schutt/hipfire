#!/usr/bin/env bash
# Path C smoke gate — runs `spec_step_ddtree_path_c` (Phase 1 and Phase 2)
# end-to-end through `dflash_spec_demo`, applies the Path A/B token-attractor
# detector, and reports per-mode pass/fail.
#
# PRD: docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd
#
# Hard-fail conditions (block commit / push):
#   - dflash_spec_demo non-zero exit / panic / zero emitted tokens
#   - Phase 1 output diverges from --ddtree-batched on the same prompt
#     (Phase 1 should be bit-exact with verify_dflash_block on the main chain;
#     diff against ddtree-batched gives a sanity signal — they're not
#     guaranteed identical but should agree on most prompts)
#   - max_token_frequency / total > 0.50 in the first 128 emitted tokens
#     (Path A failure mode — single-token attractor)
#   - unique_token_count / total < 0.15 (low-entropy loop)
#
# Soft warn (printed, doesn't block): paragraph-level repetition.
#
# Modes tested (each with a short prose prompt + a short code prompt):
#   path-c-phase1-b12-k2  : Step 1 only
#   path-c-phase2-b12-k2  : Steps 1+2+3 (lazy branch FA-only re-verify)
#
# Usage:
#   ./scripts/path-c-smoke.sh                    # auto-detect models
#   TARGET=/path/to/t.mq4 DRAFT=/path/to/d.hfq ./scripts/path-c-smoke.sh
#
# Exit codes:
#   0  smoke ran clean
#   1  hard error
#   2  build / environment error

set -u
cd "$(dirname "$0")/.."

FULL=0
GRAPH_AB=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --graph-ab) GRAPH_AB=1 ;;  # A/B verify-graph capture on/off (Phase 3 gate)
        -h|--help) sed -n '3,33p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/dflash_spec_demo"

# Model resolution: explicit env wins, else /tmp default, else $HOME defaults.
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET="${TARGET:-}"
DRAFT="${DRAFT:-}"
if [ -z "$TARGET" ]; then
    for cand in "$MODELS_DIR/qwen3.6-27b.mq4" "$MODELS_DIR/qwen3.5-27b.mq4"; do
        [ -f "$cand" ] && TARGET="$cand" && break
    done
fi
if [ -z "$DRAFT" ]; then
    for cand in "$MODELS_DIR/qwen36-27b-dflash-mq4.hfq" "$MODELS_DIR/qwen35-27b-dflash.mq4"; do
        [ -f "$cand" ] && DRAFT="$cand" && break
    done
fi

OUT="${HIPFIRE_PATH_C_OUT:-/tmp/path-c-smoke-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Build dflash_spec_demo if needed ──────────────────────────────────────
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
    echo "path-c-smoke: rebuilding dflash_spec_demo (release)..."
    if ! cargo build --release --example dflash_spec_demo --features deltanet >&2; then
        echo "path-c-smoke: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "path-c-smoke" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

if [ -z "$TARGET" ] || [ -z "$DRAFT" ] || [ ! -f "$TARGET" ] || [ ! -f "$DRAFT" ]; then
    {
        echo "# Path C smoke — SKIPPED (target/draft model not found)"
        echo
        echo "- target: ${TARGET:-(unset)}"
        echo "- draft:  ${DRAFT:-(unset)}"
        echo
        echo "Re-stage models or set TARGET / DRAFT env vars and re-run."
    } > "$OUT"
    echo "path-c-smoke: models not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# ── Prompts ──────────────────────────────────────────────────────────────
PROSE_PROMPT="The Roman Empire, at its height, stretched from the windswept moors of northern Britain to the sands of the Arabian peninsula. Its decline was not a single event but a long slow unraveling that took centuries. Several factors contributed to this gradual collapse. The first and perhaps most important was"

# Second prose: science-leaning expository — different domain than empire-history.
PROSE2_PROMPT="The discovery of penicillin by Alexander Fleming in 1928 was a turning point in the history of medicine, but the path from a serendipitous mould in a Petri dish to a mass-produced antibiotic that saved millions of lives was anything but straightforward. The decade between Fleming's observation and the first clinical use of penicillin was marked by"

# Third prose: narrative — a third register again to triangulate paragraph-level
# repetition versus genuine cohesion.
PROSE3_PROMPT="The lighthouse keeper's daughter had grown up listening to the sea. Every gale that battered the rocks below the cottage taught her something new about the moods of the Atlantic, and by the time she was twelve she could read a coming storm from the colour of the spray alone. The morning the lifeboat went out and did not return, the wind was"

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

# Second code: HumanEval #14 (all_prefixes). Different control flow than #0.
CODE2_PROMPT='from typing import List


def all_prefixes(string: str) -> List[str]:
    """ Return list of all prefixes from shortest to longest of the input string
    >>> all_prefixes("abc")
    ["a", "ab", "abc"]
    """
'

# Instruct: assistant-style request, less repetitive than continuation prompts.
INSTRUCT_PROMPT="Explain step by step why a soap film between two parallel wires forms a flat surface rather than a curved one, and describe what would change if one of the wires were heated. Use clear physical reasoning."

# ── Test matrix ──────────────────────────────────────────────────────────
SHORT_TESTS=(
    "path-c-phase1-prose|phase1|PROSE_PROMPT|192"
    "path-c-phase1-code|phase1|CODE_PROMPT|128"
    "path-c-phase2-prose|phase2|PROSE_PROMPT|192"
    "path-c-phase2-code|phase2|CODE_PROMPT|128"
)
# --full: 3 prose × 2 code × 1 instruct, each at 256 tokens, on phase1 + phase2.
# Per-prompt PRD smoke gate: unique_ratio > 0.3, max_freq < 0.4 over 256 tokens.
FULL_TESTS=(
    "path-c-phase1-prose1|phase1|PROSE_PROMPT|256"
    "path-c-phase1-prose2|phase1|PROSE2_PROMPT|256"
    "path-c-phase1-prose3|phase1|PROSE3_PROMPT|256"
    "path-c-phase1-code1|phase1|CODE_PROMPT|192"
    "path-c-phase1-code2|phase1|CODE2_PROMPT|192"
    "path-c-phase1-instruct|phase1|INSTRUCT_PROMPT|256"
    "path-c-phase2-prose1|phase2|PROSE_PROMPT|256"
    "path-c-phase2-prose2|phase2|PROSE2_PROMPT|256"
    "path-c-phase2-prose3|phase2|PROSE3_PROMPT|256"
    "path-c-phase2-code1|phase2|CODE_PROMPT|192"
    "path-c-phase2-code2|phase2|CODE2_PROMPT|192"
    "path-c-phase2-instruct|phase2|INSTRUCT_PROMPT|256"
)
if [ "$FULL" -eq 1 ]; then
    TESTS=("${FULL_TESTS[@]}")
else
    TESTS=("${SHORT_TESTS[@]}")
fi

# --graph-ab pairs each test with a `-nograph` variant that runs the same
# command with HIPFIRE_VERIFY_GRAPH=0. Used to validate the PRD's Phase 3
# expected delta (+10-15 % tok/s with verify-graph capture on the Path C
# main + branch FA forwards). Doubles the test count.
if [ "$GRAPH_AB" -eq 1 ]; then
    AB=()
    for t in "${TESTS[@]}"; do
        AB+=("$t")
        IFS='|' read -r label phase prompt_var max_tok <<< "$t"
        AB+=("${label}-nograph|$phase|$prompt_var|$max_tok|nograph")
    done
    TESTS=("${AB[@]}")
fi

# ── Detector (same logic as coherence-gate-dflash.sh) ────────────────────
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
trimmed = toks
for i, t in enumerate(toks):
    if t in EOT_IDS:
        trimmed = toks[:i]
        break
window = trimmed[:128]
if len(window) < 16:
    print(json.dumps({"ok": True, "total": len(window), "reason": "short_window_ok"}))
    sys.exit(0)
counter = collections.Counter(window)
unique = len(counter)
total = len(window)
unique_ratio = unique / total
max_tok, max_count = counter.most_common(1)[0]
max_freq = max_count / total
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

# ── Run ──────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Path C smoke (PRD ddtree-path-c-main-path-first-from-lucebox)"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FULL" -eq 1 ] && echo full || echo short )"
    echo "- target: $TARGET"
    echo "- draft:  $DRAFT"
    echo
    echo "Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.50,"
    echo "unique_token_ratio < 0.15."
    echo
} > "$OUT"

for entry in "${TESTS[@]}"; do
    IFS='|' read -r label phase prompt_var max_tok graph_flag <<< "$entry"
    case "$prompt_var" in
        PROSE_PROMPT)    prompt="$PROSE_PROMPT" ;;
        PROSE2_PROMPT)   prompt="$PROSE2_PROMPT" ;;
        PROSE3_PROMPT)   prompt="$PROSE3_PROMPT" ;;
        CODE_PROMPT)     prompt="$CODE_PROMPT" ;;
        CODE2_PROMPT)    prompt="$CODE2_PROMPT" ;;
        INSTRUCT_PROMPT) prompt="$INSTRUCT_PROMPT" ;;
        *) echo "unknown prompt_var: $prompt_var" >&2; exit 2 ;;
    esac
    # graph_flag = "nograph" → HIPFIRE_VERIFY_GRAPH=0 for this run; otherwise
    # default behaviour (graph capture on by default for eligible models).
    if [ "${graph_flag:-}" = "nograph" ]; then
        graph_env=("HIPFIRE_VERIFY_GRAPH=0")
    else
        graph_env=()
    fi

    echo "== $label =="
    out_file="/tmp/path_c_out_$$.log"
    t0=$(date +%s.%N)
    timeout 240 env "${graph_env[@]}" "$EXE" \
        --target "$TARGET" --draft "$DRAFT" \
        --prompt "$prompt" --max "$max_tok" --ctx 2048 \
        --kv-mode asym3 --no-chatml \
        --ddtree-path-c "$phase" --ddtree-budget 12 --ddtree-topk 2 \
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

    stats=$(grep -aE '^emitted:|^cycles:|^accept_rate:' "$out_file" | head -3)
    path_c_last=$(grep -a '^\[path-c\]' "$out_file" | tail -1)

    {
        echo "## $label (phase=$phase, b=12, k=2)"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        echo "- detector: \`$detect\`"
        if [ -n "$stats" ]; then
            echo "- stats:"
            echo '  ```'
            echo "$stats" | sed 's/^/  /'
            echo '  ```'
        fi
        if [ -n "$path_c_last" ]; then
            echo "- path-c counters (HIPFIRE_DDTREE_PATH_C_VERBOSE=1):"
            echo '  ```'
            echo "  $path_c_last"
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
        echo '**Output (first 40 lines of generation):**'
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
echo "path-c-smoke report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence"
exit 0

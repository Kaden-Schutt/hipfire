#!/usr/bin/env bash
# Smoke test for `dflash_spec_demo --prompts-file` — issue #173.
#
# Asserts the resident-bench harness loads (target, drafter) once and
# runs multiple prompts against the same loaded pair, with proper
# state reset between rows. Specifically checks:
#
# 1. Exit 0
# 2. Both `@@@ ROW 0 END @@@` and `@@@ ROW 1 END @@@` appear in stderr
# 3. Both rows emit non-empty `DFlash tokens: [...]` lines
# 4. Determinism: when both rows use the same prompt under --temp 0,
#    the `DFlash tokens` lists must match byte-for-byte. Failing this
#    means a per-row state reset is missing — the canonical row-2
#    correctness check.
#
# Usage:
#   ./scripts/dflash_bench_resident_smoke.sh           # use staged drafter
#   ./scripts/dflash_bench_resident_smoke.sh --build   # cargo build first
#
# Skips cleanly (exit 0) if no DFlash drafter is staged.

set -u
cd "$(dirname "$0")/.."

DO_BUILD=0
while [ $# -gt 0 ]; do
    case "$1" in
        --build) DO_BUILD=1 ;;
        -h|--help) sed -n '3,21p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"

if [ "$DO_BUILD" -eq 1 ] || [ ! -x "$EXE" ]; then
    echo "smoke: building dflash_spec_demo..."
    if ! cargo build --release --example dflash_spec_demo --features deltanet >&2; then
        echo "smoke: build failed" >&2
        exit 2
    fi
fi

# Pick the smallest staged DFlash drafter and its matching target.
# Drafter naming convention (from existing benches): `*-dflash*.hfq` or
# `*-dflash*.mq4`. Match by model size prefix (e.g. qwen35-27b-dflash → qwen3.5-27b).
DRAFT=""
TARGET=""
for d in \
    "$MODELS_DIR/qwen35-9b-dflash-mq4.hfq" \
    "$MODELS_DIR/qwen35-9b-dflash.mq4" \
    "$MODELS_DIR/qwen35-27b-dflash-mq4.hfq" \
    "$MODELS_DIR/qwen35-27b-dflash.mq4" \
    "$MODELS_DIR/qwen36-27b-dflash-mq4.hfq"; do
    if [ -f "$d" ]; then DRAFT="$d"; break; fi
done
if [ -z "$DRAFT" ]; then
    echo "smoke: no DFlash drafter staged in $MODELS_DIR — skipping"
    echo "       (looked for qwen35-9b-dflash, qwen35-27b-dflash, qwen36-27b-dflash)"
    exit 0
fi
case "$DRAFT" in
    *qwen35-9b-dflash*)  TARGET="$MODELS_DIR/qwen3.5-9b.mq4" ;;
    *qwen35-27b-dflash*) TARGET="$MODELS_DIR/qwen3.5-27b.mq4" ;;
    *qwen36-27b-dflash*) TARGET="$MODELS_DIR/qwen3.6-27b.mq4" ;;
esac
if [ ! -f "$TARGET" ]; then
    echo "smoke: target $TARGET not found (drafter $DRAFT staged but pair incomplete) — skipping"
    exit 0
fi

echo "smoke: target=$TARGET"
echo "smoke: drafter=$DRAFT"

# GPU lock — same pattern as coherence-gate-dflash.sh.
LOCK_SCRIPT="./scripts/gpu-lock.sh"
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "dflash-bench-resident-smoke" || { echo "smoke: could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# Two-row manifest with IDENTICAL prompts. Under --temp 0 the
# committed token streams must match byte-for-byte; mismatch means a
# state reset is missing between rows.
MANIFEST=$(mktemp /tmp/dflash_smoke_manifest.XXXXXX.jsonl)
OUT=$(mktemp /tmp/dflash_smoke_out.XXXXXX.log)
trap 'rm -f "$MANIFEST" "$OUT"; gpu_release 2>/dev/null || true' EXIT

# Per CLAUDE.md "Prompt-structure τ sensitivity" and
# docs/methodology/perf-benchmarking.md: the prompt must be a
# committed fixture, not a heredoc, to keep the prompt md5 stable
# across editor/agent reformats. The smoke is a correctness test
# (assertion: identical prompts → identical token streams) so this
# is convention-following more than load-bearing, but worth modeling
# for future bench scripts that adopt --prompts-file.
PROMPT_FILE="benchmarks/prompts/dflash_resident_smoke.txt"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "smoke: fixture missing: $PROMPT_FILE" >&2
    exit 2
fi
python3 - "$PROMPT_FILE" "$MANIFEST" <<'PYEOF'
import json, sys
prompt = open(sys.argv[1], "rb").read().decode("utf-8")
with open(sys.argv[2], "w") as f:
    for label in ("row0", "row1"):
        f.write(json.dumps({"label": label, "prompt": prompt, "max": 16}) + "\n")
PYEOF

echo "smoke: running 2-row manifest (~60s for one model load)..."
t0=$(date +%s)
if ! timeout 240 "$EXE" \
    --target "$TARGET" --draft "$DRAFT" \
    --prompts-file "$MANIFEST" \
    --ctx 1024 --kv-mode q8 --no-chatml \
    > "$OUT" 2>&1; then
    echo "smoke: FAIL — dflash_spec_demo exited non-zero"
    tail -40 "$OUT"
    exit 1
fi
t1=$(date +%s)
echo "smoke: elapsed $((t1-t0))s"

# Assertion 1: both row END markers present
fail=0
for marker in "@@@ ROW 0 END @@@" "@@@ ROW 1 END @@@"; do
    if ! grep -qF "$marker" "$OUT"; then
        echo "smoke: FAIL — missing marker: $marker"
        fail=1
    fi
done

# Assertion 2: both rows emit non-empty DFlash tokens lists
n_token_lines=$(grep -cE 'DFlash tokens: \[[^]]+\]' "$OUT" || true)
if [ "$n_token_lines" -ne 2 ]; then
    echo "smoke: FAIL — expected 2 'DFlash tokens: [...]' lines, got $n_token_lines"
    fail=1
fi

# Assertion 3: identical prompts → identical token streams (state reset)
tokens_row0=$(awk '/@@@ ROW 0:/,/@@@ ROW 0 END/' "$OUT" | grep -oE 'DFlash tokens: \[[^]]+\]' | head -1)
tokens_row1=$(awk '/@@@ ROW 1:/,/@@@ ROW 1 END/' "$OUT" | grep -oE 'DFlash tokens: \[[^]]+\]' | head -1)
if [ -z "$tokens_row0" ] || [ -z "$tokens_row1" ]; then
    echo "smoke: FAIL — could not extract per-row DFlash tokens lines"
    fail=1
elif [ "$tokens_row0" != "$tokens_row1" ]; then
    echo "smoke: FAIL — row 0 and row 1 token streams differ under --temp 0"
    echo "  row 0: $tokens_row0"
    echo "  row 1: $tokens_row1"
    echo "  This indicates a per-row state reset is missing. See"
    echo "  docs/plans/173-bench-residents-prompts.md 'Per-row reset' table."
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo
    echo "smoke output: $OUT"
    exit 1
fi
echo "smoke: PASS"
exit 0

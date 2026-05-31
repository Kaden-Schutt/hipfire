#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery — DeepSeek V4 Flash MTP spec-decode variant.
#
# Sibling to coherence-gate-dflash.sh (qwen35 DDtree spec-decode). DeepSeek V4's
# speculative decode goes through a different path:
#   - same model, no separate draft (MTP layer is opt-in `mtp.0.*` weights)
#   - speculative_decode_step_with_pbs in `crates/hipfire-arch-deepseek4/src/spec_decode.rs`
#   - driven through the daemon's JSONL protocol with
#     HIPFIRE_DEEPSEEK4_SPEC_DECODE=1 + HIPFIRE_DEEPSEEK4_SPEC_K={2,3}
#
# So we can't share dflash_spec_demo as the driver. Instead we drive the
# daemon directly and apply the SAME token-attractor detector from
# coherence-gate-dflash.sh to the emitted token stream.
#
# Hard-fail conditions (block commit, exit 1):
#   - daemon non-zero exit / panic / zero tokens
#   - max_token_frequency > 0.50 in the first 128 emitted tokens
#   - unique_token_ratio < 0.15 in the first 128 emitted tokens
#
# Soft warn (printed in report, no exit code):
#   - max_freq > 0.40 OR unique_ratio < 0.30 (paragraph-level repetition,
#     not the unrecoverable Path-A class)
#
# Exit codes:
#   0  battery ran clean
#   1  hard error (panic / zero tokens / token attractor)
#   2  build or environment error
#
# Modes:
#   ./scripts/coherence-gate-deepseek4-mtp.sh         # short — 2 tests, ~1-2 min
#   ./scripts/coherence-gate-deepseek4-mtp.sh --fast  # 1 test (code) — <1 min
#   ./scripts/coherence-gate-deepseek4-mtp.sh --full  # 4 tests (cap + reason + code + math)

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
    echo "coherence-gate-deepseek4-mtp: --fast and --full are mutually exclusive" >&2
    exit 2
fi

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
V4F_MODEL="$MODELS_DIR/deepseek-v4-flash.mq2lloyd"
V4F_ADDON="$MODELS_DIR/deepseek-v4-flash-mtp.mq2lloyd"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-deepseek4-mtp-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-240}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild daemon if any DeepSeek V4 / spec_decode source is newer than the binary ─
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-deepseek4/src/arch.rs \
               crates/hipfire-arch-deepseek4/src/deepseek4.rs \
               crates/hipfire-arch-deepseek4/src/forward.rs \
               crates/hipfire-arch-deepseek4/src/spec_decode.rs \
               crates/hipfire-runtime/examples/daemon.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-deepseek4-mtp: rebuilding daemon..."
    if ! cargo build --release --example daemon --features deltanet >&2; then
        echo "coherence-gate-deepseek4-mtp: build failed" >&2
        exit 2
    fi
fi

# Skip everything if the DeepSeek V4 base model isn't present.
if [ ! -f "$V4F_MODEL" ]; then
    {
        echo "# Coherence battery — DeepSeek V4 MTP spec-decode"
        echo
        echo "## SKIPPED — DeepSeek V4 base model not found at $V4F_MODEL"
        echo
        echo "Symlink to enable:"
        echo "  ln -s /data/hipfire-models/deepseek-v4-flash.mq2lloyd \\"
        echo "        \"\$HOME/.hipfire/models/deepseek-v4-flash.mq2lloyd\""
        echo "  ln -s /data/hipfire-models/deepseek-v4-flash-mtp.mq2lloyd \\"
        echo "        \"\$HOME/.hipfire/models/deepseek-v4-flash-mtp.mq2lloyd\""
    } > "$OUT"
    echo "coherence-gate-deepseek4-mtp: DeepSeek V4 model not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# ── GPU lock ──────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-deepseek4-mtp" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Prompts ───────────────────────────────────────────────────────────────
CAP_PROMPT='What is the capital of France? Answer in one short sentence.'
CODE_PROMPT='Write a Python function `truncate_number(number: float) -> float` that returns the decimal part of a positive floating point number.'
REASON_PROMPT='A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.'
MATH_PROMPT='A rectangular swimming pool is 25 meters long and 12 meters wide. The pool has a depth that varies linearly from 1 meter at the shallow end to 3 meters at the deep end. Calculate the total volume of water.'

# ── Test matrix ───────────────────────────────────────────────────────────
# Format: "label|prompt_var|max_tokens|spec_k"
# Memory: K=2 has highest accept; K=3 has highest TG on code/math but lower
# on prose/qa. We hard-fail on attractors at either K (Path-A class), and
# soft-warn on paragraph-level repetition.
SHORT_TESTS=(
    "deepseek4-mtp-code-k2|CODE_PROMPT|80|2"
    "deepseek4-mtp-prose-k2|REASON_PROMPT|80|2"
)
FAST_TESTS=(
    "deepseek4-mtp-code-k2|CODE_PROMPT|80|2"
)
FULL_EXTRA=(
    "deepseek4-mtp-cap-k2|CAP_PROMPT|40|2"
    "deepseek4-mtp-math-k2|MATH_PROMPT|80|2"
    "deepseek4-mtp-code-k3|CODE_PROMPT|80|3"
    "deepseek4-mtp-prose-k3|REASON_PROMPT|80|3"
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
    echo "# Coherence battery — DeepSeek V4 MTP spec-decode"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
    echo "- model:  $V4F_MODEL"
    echo "- addon:  $V4F_ADDON $([ -f "$V4F_ADDON" ] && echo "(present)" || echo "(missing — MTP off)")"
    echo
    echo "Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.50 in"
    echo "first 128 tokens, unique_token_ratio < 0.15. Soft warn at 0.40/0.30."
    echo
} > "$OUT"

DETECT_PY=$(cat <<'PYEOF'
import sys, json, collections
EOT_IDS = {1, 128001, 128009}  # DeepSeek family: end-of-sentence + EOT
toks = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('{'): continue
    try:
        ev = json.loads(line)
    except Exception:
        continue
    if ev.get("type") == "committed" and "tok_id" in ev:
        toks.append(int(ev["tok_id"]))
if not toks:
    print(json.dumps({"ok": False, "reason": "no_committed_ids"}))
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

# Daemon emits per-committed-token events when HIPFIRE_EMIT_TOKEN_IDS=1.
# That env var unblocks the detector — without it, only `text` events
# arrive and we'd need to re-tokenize (lossy on BPE merges).
export HIPFIRE_EMIT_TOKEN_IDS=1
if [ -f "$V4F_ADDON" ]; then
    export HIPFIRE_DEEPSEEK4_MTP_ADDON="$V4F_ADDON"
fi

for entry in "${tests[@]}"; do
    IFS='|' read -r label prompt_var max_tok spec_k <<< "$entry"
    prompt="${!prompt_var}"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    in_file="/tmp/coh_deepseek4_in_$$.jsonl"
    out_file="/tmp/coh_deepseek4_out_$$.log"
    cat > "$in_file" <<JL
{"type":"load","model":"$V4F_MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"$label","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.0}
{"type":"unload"}
JL

    echo "== $label =="
    t0=$(date +%s.%N)
    HIPFIRE_DEEPSEEK4_SPEC_DECODE=1 HIPFIRE_DEEPSEEK4_SPEC_K="$spec_k" \
        timeout "$CASE_TIMEOUT" "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    detector=$(python3 -c "$DETECT_PY" < "$out_file" 2>/dev/null || echo '{"ok":false,"reason":"detector_crash"}')
    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ] \
        || ! echo "$detector" | python3 -c "import sys,json; sys.exit(0 if json.loads(sys.stdin.read()).get('ok') else 1)" 2>/dev/null; then
        status="HARD-FAIL"
        hard_errors=$((hard_errors+1))
    fi

    {
        echo "## $label"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        if [ -n "$done_line" ]; then
            echo "- stats: \`$done_line\`"
        fi
        echo "- detector: \`$detector\`"
        echo "- prompt: $(echo "$prompt" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))")"
        echo
        echo "**Output:**"
        echo
        echo '```'
        grep -aE '"type":"token"' "$out_file" | python3 -c "
import sys, json
text = ''
for line in sys.stdin:
    try: text += json.loads(line)['text']
    except: pass
print(text)
" 2>/dev/null
        echo '```'
        echo
    } >> "$OUT"
    rm -f "$in_file" "$out_file"
done

echo
if [ "$hard_errors" -gt 0 ]; then
    echo "coherence-gate-deepseek4-mtp: $hard_errors HARD ERROR(S) — see $OUT" >&2
    echo "report: $OUT"
    exit 1
fi
echo "coherence-gate-deepseek4-mtp: no hard errors — review $OUT for coherence"
echo "report: $OUT"
exit 0

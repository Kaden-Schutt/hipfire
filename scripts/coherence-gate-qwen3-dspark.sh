#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Bjoern Boesel
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery — Qwen3-8B DSpark spec-decode variant.
#
# Drives the Qwen3-8B target with the `-dspark.mq4` sidecar auto-attached
# (loader auto-discovers `<stem>-dspark.mq4` alongside the trunk) through
# the daemon's JSONL protocol.  DSpark is greedy spec-decode (temp=0.0), so
# the same token-attractor thresholds used for DFlash greedy apply here.
#
# Hard-fail conditions (exit 1):
#   - daemon non-zero exit / panic / zero tokens
#   - max_token_frequency > 0.50 in the first 128 emitted tokens  (Tier 1)
#   - unique_token_ratio  < 0.15 in the first 128 emitted tokens  (Tier 1)
#   - max_token_frequency > 0.50 in the last  128 emitted tokens  (Tier 2)
#   - unique_token_ratio  < 0.30 in the last  128 emitted tokens  (Tier 2)
#
# Soft warn (printed in report, no exit-code impact):
#   - 3-gram density > 50% in second half of output (Tier 3 block attractor)
#
# Exit codes:
#   0  battery ran clean — inspect the report for fluency
#   1  hard error (panic / zero tokens / token attractor)
#   2  build or environment error
#
# Modes:
#   ./scripts/coherence-gate-qwen3-dspark.sh           # short (4 rows) ~2-4 min
#   ./scripts/coherence-gate-qwen3-dspark.sh --fast    # 1 row (capital) ~<1 min
#   ./scripts/coherence-gate-qwen3-dspark.sh --full    # 4 rows + long-context ~4-8 min

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
    echo "coherence-gate-qwen3-dspark: --fast and --full are mutually exclusive" >&2
    exit 2
fi

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
MODEL="$MODELS_DIR/qwen3-8b.mq4"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-qwen3-dspark-$(date +%Y%m%d-%H%M%S).md}"
CASE_TIMEOUT="${HIPFIRE_COHERENCE_TIMEOUT:-300}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# ── Rebuild daemon if qwen3/dspark-relevant source is newer than the binary ──
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in \
        crates/hipfire-arch-llama/src/dspark_body.rs \
        crates/hipfire-arch-llama/src/spec_impl.rs \
        crates/hipfire-arch-llama/src/carrier.rs \
        crates/hipfire-runtime/src/dspark_core.rs \
        crates/hipfire-daemon/src/main.rs \
        crates/hipfire-loader/src/carriers.rs \
        crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-qwen3-dspark: rebuilding daemon..."
 if ! cargo build --release -p hipfire-daemon >&2; then
        echo "coherence-gate-qwen3-dspark: build failed" >&2
        exit 2
    fi
fi

# Skip gracefully if the model isn't present.
if [ ! -f "$MODEL" ]; then
    {
        echo "# Coherence battery — Qwen3-8B DSpark spec-decode"
        echo
        echo "## SKIPPED — model not found at $MODEL"
        echo
        echo "Symlink to enable:"
        echo "  ln -s /data/hipfire-models/qwen3-8b.mq4 \\"
        echo "        \"\$HOME/.hipfire/models/qwen3-8b.mq4\""
    } > "$OUT"
    echo "coherence-gate-qwen3-dspark: model not present, skipping (no hard error)"
    echo "report: $OUT"
    exit 0
fi

# ── GPU lock ──────────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-qwen3-dspark" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Prompts ───────────────────────────────────────────────────────────────────
CAP_PROMPT='What is the capital of France? Answer in one short sentence.'
SHEEP_PROMPT='A farmer has 17 sheep. All but 9 die. How many sheep are left? Show brief reasoning then state the final number.'
CODE_PROMPT='Write a one-line Python function named square that takes n and returns n*n.'
LONG_PROMPT='Explain in detail how garbage collection works in modern programming languages. Cover at minimum: reference counting, mark-and-sweep, generational collection, and incremental GC. Discuss tradeoffs between throughput and latency, and give an example of a real-world language runtime that uses each approach.'

# ── Test matrix ───────────────────────────────────────────────────────────────
# Format: "label|prompt_var|max_tokens"
# --fast: just capital (fastest, <1 min)
# default short: cap + sheep + code (2-4 min each)
# --full: adds long-context row
FAST_TESTS=(
    "qwen3-dspark-cap|CAP_PROMPT|60"
)
SHORT_TESTS=(
    "qwen3-dspark-cap|CAP_PROMPT|60"
    "qwen3-dspark-sheep|SHEEP_PROMPT|100"
    "qwen3-dspark-code|CODE_PROMPT|80"
)
FULL_EXTRA=(
    "qwen3-dspark-long|LONG_PROMPT|200"
)

if [ "$FAST" -eq 1 ]; then
    tests=("${FAST_TESTS[@]}")
else
    tests=("${SHORT_TESTS[@]}")
    [ "$FULL" -eq 1 ] && tests+=("${FULL_EXTRA[@]}")
fi

# ── Attractor detector (Tier 1 + Tier 2 hard fail, Tier 3 soft) ─────────────
# Applied to the committed token stream (HIPFIRE_EMIT_TOKEN_IDS=1).
DETECT_PY=$(cat <<'PYEOF'
import sys, json, collections, math

EOT_IDS = {1, 2, 151643, 151645, 151649}  # Qwen3 EOS/EOT family

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

# Trim trailing EOS/EOT tokens.
trimmed = toks
for i, t in enumerate(toks):
    if t in EOT_IDS:
        trimmed = toks[:i]
        break

total_all = len(trimmed)

def check_window(window, label, hard_uniq_lo, hard_freq_hi, soft_uniq_lo, soft_freq_hi):
    if len(window) < 16:
        return {"label": label, "ok": True, "reason": "short_window_ok", "n": len(window)}
    c = collections.Counter(window)
    unique = len(c)
    n = len(window)
    unique_ratio = unique / n
    max_tok, max_count = c.most_common(1)[0]
    max_freq = max_count / n
    hard_fail = max_freq > hard_freq_hi or unique_ratio < hard_uniq_lo
    soft_warn = (max_freq > soft_freq_hi or unique_ratio < soft_uniq_lo) and not hard_fail
    return {
        "label": label,
        "ok": not hard_fail,
        "soft_warn": soft_warn,
        "n": n,
        "unique": unique,
        "unique_ratio": round(unique_ratio, 3),
        "max_freq": round(max_freq, 3),
        "max_tok": max_tok,
    }

t1 = check_window(trimmed[:128],  "tier1_first128", 0.15, 0.50, 0.25, 0.40)
t2 = check_window(trimmed[-128:], "tier2_last128",  0.30, 0.50, 0.40, 0.45)

# Tier 3: 3-gram density in second half (soft flag only)
tier3_flag = False
if total_all >= 32:
    second_half = trimmed[total_all // 2:]
    if len(second_half) >= 6:
        three_grams = [tuple(second_half[i:i+3]) for i in range(len(second_half)-2)]
        if three_grams:
            most_freq_3g = max(collections.Counter(three_grams).values()) if three_grams else 0
            density = most_freq_3g / len(three_grams)
            tier3_flag = density > 0.50

overall_ok = t1["ok"] and t2["ok"]
print(json.dumps({
    "ok": overall_ok,
    "total": total_all,
    "tier1": t1,
    "tier2": t2,
    "tier3_3gram_flag": tier3_flag,
}))
PYEOF
)

# ── Run ───────────────────────────────────────────────────────────────────────
hard_errors=0

{
    echo "# Coherence battery — Qwen3-8B DSpark spec-decode"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- mode:   $( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
    echo "- model:  $MODEL"
    echo
    echo "Hard-fail thresholds (greedy spec-decode = DFlash attractor class):"
    echo "  Tier 1 (first 128): unique_ratio < 0.15 FAIL, max_freq > 0.50 FAIL"
    echo "  Tier 2 (last 128):  unique_ratio < 0.30 FAIL, max_freq > 0.50 FAIL"
    echo "  Tier 3 (2nd half):  3-gram density > 0.50 → soft FLAG (human eyeball)"
    echo
} > "$OUT"

export HIPFIRE_EMIT_TOKEN_IDS=1

for entry in "${tests[@]}"; do
    IFS='|' read -r label prompt_var max_tok <<< "$entry"
    prompt="${!prompt_var}"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    in_file="/tmp/coh_qwen3dspark_in_$$.jsonl"
    out_file="/tmp/coh_qwen3dspark_out_$$.log"
    cat > "$in_file" <<JL
{"type":"load","model":"$MODEL","params":{"max_seq":4096}}
{"type":"generate","id":"$label","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.0}
{"type":"unload"}
JL

    echo "== $label =="
    t0=$(date +%s.%N)
    timeout "$CASE_TIMEOUT" "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file" || true)
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL' "$out_file" | head -1 || true)
    error_ev=$(grep -aE '"type":"error"' "$out_file" | head -1 || true)
    detector=$(python3 -c "$DETECT_PY" < "$out_file" 2>/dev/null || echo '{"ok":false,"reason":"detector_crash"}')

    # Check for DSpark speculator being engaged (log line)
    dspark_active=$(grep -a "DSpark speculator enabled" "$out_file" | head -1 || true)

    status="OK"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ] || [ -n "$error_ev" ] \
        || ! echo "$detector" | python3 -c "import sys,json; sys.exit(0 if json.loads(sys.stdin.read()).get('ok') else 1)" 2>/dev/null; then
        status="HARD-FAIL"
        hard_errors=$((hard_errors+1))
    fi
    # Hard-fail if the DSpark speculator was not engaged: a missing sidecar silently
    # falls back to AR, producing false-green results for the gate.
    if [ -z "$dspark_active" ]; then
        echo "HARD FAIL: DSpark speculator not engaged for row $label" >&2
        status="HARD-FAIL(dspark-not-engaged)"
        hard_errors=$((hard_errors+1))
    fi
    tier3_flag=$(echo "$detector" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('FLAG' if d.get('tier3_3gram_flag') else 'ok')" 2>/dev/null || echo "?")
    if [ "$status" = "OK" ] && [ "$tier3_flag" = "FLAG" ]; then
        status="OK(tier3-soft-flag)"
    fi

    {
        echo "## $label"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        if [ -n "$done_line" ]; then
            echo "- stats: \`$done_line\`"
        fi
        echo "- detector: \`$detector\`"
        echo "- dspark_active: \`${dspark_active:-NOT FOUND in log}\`"
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
    echo "coherence-gate-qwen3-dspark: $hard_errors HARD ERROR(S) — see $OUT" >&2
    echo "report: $OUT"
    exit 1
fi
echo "coherence-gate-qwen3-dspark: no hard errors — review $OUT for fluency"
echo "report: $OUT"
exit 0

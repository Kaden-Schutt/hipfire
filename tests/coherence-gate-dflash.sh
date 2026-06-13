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
# coherence detector and is used while proving rollback parity. AR parity also
# requires conservative serial/full-prefill rollback replay; fast GDN-tape
# replay remains diagnostic-only until it passes the same token parity gate.

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
TRACE_JSON_OUT="${HIPFIRE_COHERENCE_DFLASH_TRACE_JSON:-${OUT%.md}.dflash_trace.json}"
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
run_mode="$( [ "$FAST" -eq 1 ] && echo fast || ( [ "$FULL" -eq 1 ] && echo full || echo short ) )"
run_config_json=$(python3 - "$run_mode" <<'PYEOF'
import json
import os
import sys

keys = [
    "HIPFIRE_DFLASH_AR_PARITY",
    "HIPFIRE_DFLASH_ROLLBACK_COMPARE",
    "HIPFIRE_DFLASH_ROLLBACK_LOGIT_COMPARE_STEPS",
    "HIPFIRE_DFLASH_TRACE_POSITION",
    "HIPFIRE_DFLASH_TRACE_EXPECTED_TOKEN",
    "HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY",
    "HIPFIRE_DFLASH_ROLLBACK_PREFIX_VERIFY",
    "HIPFIRE_VERIFY_GRAPH",
    "HIPFIRE_DFLASH_ROLLBACK_FA_RAW_ATOL",
    "HIPFIRE_DFLASH_ROLLBACK_X_IN_ATOL",
    "HIPFIRE_HFQ4_QKV_FAST",
    "HIPFIRE_HFQ4_QKVZA_FAST",
    "HIPFIRE_HFQ4_GATE_UP_FAST",
    "HIPFIRE_HFQ4_RESIDUAL_FAST",
    "HIPFIRE_Q8_FA_ATTENTION_ROW_LOOP",
    "HIPFIRE_Q8_FA_ATTENTION_IGNORE_TREE_BIAS",
    "HIPFIRE_Q8_FA_ATTENTION_SCALAR_LOOP",
    "HIPFIRE_Q8_FA_ATTENTION_SERIAL_KV_LOOP",
]

print(json.dumps({
    "mode": sys.argv[1],
    "env": {key: os.environ[key] for key in keys if key in os.environ},
}, sort_keys=True))
PYEOF
)
printf '{"kind":"dflash_trace","run_config":%s,"records":[\n' "$run_config_json" > "$TRACE_JSON_OUT"
trace_json_records=0

append_trace_json_record() {
    if [ "$trace_json_records" -gt 0 ]; then
        printf ',\n' >> "$TRACE_JSON_OUT"
    fi
    printf '%s' "$1" >> "$TRACE_JSON_OUT"
    trace_json_records=$((trace_json_records + 1))
}

# Skip everything if 27B model + draft aren't both present.
if [ ! -f "$TARGET_27B" ] || [ ! -f "$DRAFT_27B" ]; then
    printf '\n]}\n' >> "$TRACE_JSON_OUT"
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
    echo "dflash trace json: $TRACE_JSON_OUT"
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

ROLLBACK_REPLAY_PY=$(cat <<'PYEOF'
import sys, re, json
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
m = re.search(
    r"^rollback_parity: .*replay_gdn_tape=(\d+)(?: replay_batched_prefill=(\d+))? replay_full_prefill=(\d+)(?: replay_prefix_verify=(\d+))?(?: replay_verify_complete=(\d+))?",
    out,
    re.MULTILINE,
)
if not m:
    print(json.dumps({"ok": False, "reason": "missing_rollback_parity_stats"}))
    sys.exit(0)
gdn = int(m.group(1))
batched = int(m.group(2) or 0)
full = int(m.group(3))
prefix_verify = int(m.group(4) or 0)
verify_complete = int(m.group(5) or 0)
if gdn != 0:
    print(json.dumps({
        "ok": False,
        "reason": "fast_gdn_tape_replay_is_diagnostic_only",
        "replay_gdn_tape": gdn,
        "replay_batched_prefill": batched,
        "replay_full_prefill": full,
        "replay_prefix_verify": prefix_verify,
        "replay_verify_complete": verify_complete,
    }))
elif batched + full + prefix_verify + verify_complete <= 0:
    print(json.dumps({
        "ok": False,
        "reason": "missing_admitted_rollback_replay",
        "replay_gdn_tape": gdn,
        "replay_batched_prefill": batched,
        "replay_full_prefill": full,
        "replay_prefix_verify": prefix_verify,
        "replay_verify_complete": verify_complete,
    }))
else:
    print(json.dumps({
        "ok": True,
        "replay_gdn_tape": gdn,
        "replay_batched_prefill": batched,
        "replay_full_prefill": full,
        "replay_prefix_verify": prefix_verify,
        "replay_verify_complete": verify_complete,
    }))
PYEOF
)

VERIFY_GRAPH_PY=$(cat <<'PYEOF'
import sys, re, json
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
m = re.search(
    r"^verify_graph: direct=(\d+) warmup=(\d+) capture=(\d+) replay=(\d+) not_applicable=(\d+)",
    out,
    re.MULTILINE,
)
if not m:
    print(json.dumps({"ok": False, "reason": "missing_verify_graph_stats"}))
    sys.exit(0)
direct, warmup, capture, replay, not_applicable = [int(x) for x in m.groups()]
if direct != 0:
    print(json.dumps({
        "ok": False,
        "reason": "direct_verify_is_diagnostic_only_for_dflash_ar_parity",
        "direct": direct,
        "warmup": warmup,
        "capture": capture,
        "replay": replay,
        "not_applicable": not_applicable,
    }))
elif capture + replay <= 0:
    print(json.dumps({
        "ok": False,
        "reason": "missing_verify_graph_capture_or_replay",
        "direct": direct,
        "warmup": warmup,
        "capture": capture,
        "replay": replay,
        "not_applicable": not_applicable,
    }))
else:
    print(json.dumps({
        "ok": True,
        "direct": direct,
        "warmup": warmup,
        "capture": capture,
        "replay": replay,
        "not_applicable": not_applicable,
    }))
PYEOF
)

ROLLBACK_LOGIT_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
pattern = re.compile(
    r"^\[dflash-rollback-next-logit-compare\] "
    r"pos=(\d+) step=(\d+) compare_steps=(\d+) token_pos=(\d+) token=(\d+) "
    r"serial_argmax=(\d+) fast_argmax=(\d+) "
    r"serial_token_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"fast_token_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"f32_words=(\d+) f32_bit_diff_words=(\d+) "
    r"max_abs=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"mean_abs=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"max_rel=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_argmax_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_runner_up_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_argmax_margin=([+\-0-9.eE]+|NaN|nan|inf|-inf)$",
    re.MULTILINE,
)
rows = []
for m in pattern.finditer(out):
    max_abs = float(m.group(12))
    mean_abs = float(m.group(13))
    max_rel = float(m.group(14))
    serial_argmax_margin = float(m.group(17))
    rows.append({
        "pos": int(m.group(1)),
        "step": int(m.group(2)),
        "compare_steps": int(m.group(3)),
        "token_pos": int(m.group(4)),
        "token": int(m.group(5)),
        "serial_argmax": int(m.group(6)),
        "fast_argmax": int(m.group(7)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "serial_argmax_logit": float(m.group(15)),
        "serial_runner_up_logit": float(m.group(16)),
        "serial_argmax_margin": serial_argmax_margin,
        "max_abs_over_margin": (
            max_abs / serial_argmax_margin
            if math.isfinite(max_abs) and math.isfinite(serial_argmax_margin) and serial_argmax_margin > 0
            else None
        ),
    })
if not rows:
    print(json.dumps({"ok": True, "checked": 0}))
    sys.exit(0)
mismatches = [r for r in rows if r["serial_argmax"] != r["fast_argmax"]]
finite_max_abs = [r["max_abs"] for r in rows if math.isfinite(r["max_abs"])]
finite_mean_abs = [r["mean_abs"] for r in rows if math.isfinite(r["mean_abs"])]
finite_max_rel = [r["max_rel"] for r in rows if math.isfinite(r["max_rel"])]
finite_margins = [r["serial_argmax_margin"] for r in rows if math.isfinite(r["serial_argmax_margin"])]
finite_abs_over_margin = [
    r["max_abs_over_margin"]
    for r in rows
    if r["max_abs_over_margin"] is not None and math.isfinite(r["max_abs_over_margin"])
]
payload = {
    "ok": not mismatches,
    "checked": len(rows),
    "argmax_mismatches": len(mismatches),
    "positions": sorted(set(r["pos"] for r in rows)),
    "max_compare_steps": max(r["compare_steps"] for r in rows),
    "max_abs": max(finite_max_abs) if finite_max_abs else None,
    "max_mean_abs": max(finite_mean_abs) if finite_mean_abs else None,
    "max_rel": max(finite_max_rel) if finite_max_rel else None,
    "min_serial_argmax_margin": min(finite_margins) if finite_margins else None,
    "max_abs_over_margin": max(finite_abs_over_margin) if finite_abs_over_margin else None,
}
if finite_abs_over_margin:
    worst = max(
        (r for r in rows if r["max_abs_over_margin"] is not None and math.isfinite(r["max_abs_over_margin"])),
        key=lambda r: r["max_abs_over_margin"],
    )
    payload["max_abs_over_margin_row"] = {
        "pos": worst["pos"],
        "step": worst["step"],
        "token_pos": worst["token_pos"],
        "token": worst["token"],
        "serial_argmax": worst["serial_argmax"],
        "fast_argmax": worst["fast_argmax"],
        "max_abs": worst["max_abs"],
        "serial_argmax_margin": worst["serial_argmax_margin"],
        "max_abs_over_margin": worst["max_abs_over_margin"],
    }
if mismatches:
    first = mismatches[0]
    payload["reason"] = "fast_replay_next_logit_argmax_mismatch"
    payload["first_mismatch"] = {
        "pos": first["pos"],
        "step": first["step"],
        "token_pos": first["token_pos"],
        "token": first["token"],
        "serial_argmax": first["serial_argmax"],
        "fast_argmax": first["fast_argmax"],
    }
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_PREFILL_LOGIT_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
pattern = re.compile(
    r"^\[dflash-rollback-prefill-next-logit-compare\] "
    r"pos=(\d+) step=(\d+) compare_steps=(\d+) token_pos=(\d+) token=(\d+) "
    r"serial_argmax=(\d+) prefill_argmax=(\d+) "
    r"serial_token_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"prefill_token_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"f32_words=(\d+) f32_bit_diff_words=(\d+) "
    r"max_abs=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"mean_abs=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"max_rel=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_argmax_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_runner_up_logit=([+\-0-9.eE]+|NaN|nan|inf|-inf) "
    r"serial_argmax_margin=([+\-0-9.eE]+|NaN|nan|inf|-inf)$",
    re.MULTILINE,
)
rows = []
for m in pattern.finditer(out):
    max_abs = float(m.group(12))
    mean_abs = float(m.group(13))
    max_rel = float(m.group(14))
    serial_argmax_margin = float(m.group(17))
    rows.append({
        "pos": int(m.group(1)),
        "step": int(m.group(2)),
        "compare_steps": int(m.group(3)),
        "token_pos": int(m.group(4)),
        "token": int(m.group(5)),
        "serial_argmax": int(m.group(6)),
        "prefill_argmax": int(m.group(7)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "serial_argmax_logit": float(m.group(15)),
        "serial_runner_up_logit": float(m.group(16)),
        "serial_argmax_margin": serial_argmax_margin,
        "max_abs_over_margin": (
            max_abs / serial_argmax_margin
            if math.isfinite(max_abs) and math.isfinite(serial_argmax_margin) and serial_argmax_margin > 0
            else None
        ),
    })
if not rows:
    print(json.dumps({"ok": True, "checked": 0}))
    sys.exit(0)
mismatches = [r for r in rows if r["serial_argmax"] != r["prefill_argmax"]]
finite_max_abs = [r["max_abs"] for r in rows if math.isfinite(r["max_abs"])]
finite_mean_abs = [r["mean_abs"] for r in rows if math.isfinite(r["mean_abs"])]
finite_max_rel = [r["max_rel"] for r in rows if math.isfinite(r["max_rel"])]
finite_margins = [r["serial_argmax_margin"] for r in rows if math.isfinite(r["serial_argmax_margin"])]
finite_abs_over_margin = [
    r["max_abs_over_margin"]
    for r in rows
    if r["max_abs_over_margin"] is not None and math.isfinite(r["max_abs_over_margin"])
]
payload = {
    "ok": not mismatches,
    "checked": len(rows),
    "argmax_mismatches": len(mismatches),
    "positions": sorted(set(r["pos"] for r in rows)),
    "max_compare_steps": max(r["compare_steps"] for r in rows),
    "max_abs": max(finite_max_abs) if finite_max_abs else None,
    "max_mean_abs": max(finite_mean_abs) if finite_mean_abs else None,
    "max_rel": max(finite_max_rel) if finite_max_rel else None,
    "min_serial_argmax_margin": min(finite_margins) if finite_margins else None,
    "max_abs_over_margin": max(finite_abs_over_margin) if finite_abs_over_margin else None,
}
if finite_abs_over_margin:
    worst = max(
        (r for r in rows if r["max_abs_over_margin"] is not None and math.isfinite(r["max_abs_over_margin"])),
        key=lambda r: r["max_abs_over_margin"],
    )
    payload["max_abs_over_margin_row"] = {
        "pos": worst["pos"],
        "step": worst["step"],
        "token_pos": worst["token_pos"],
        "token": worst["token"],
        "serial_argmax": worst["serial_argmax"],
        "prefill_argmax": worst["prefill_argmax"],
        "max_abs": worst["max_abs"],
        "serial_argmax_margin": worst["serial_argmax_margin"],
        "max_abs_over_margin": worst["max_abs_over_margin"],
    }
if mismatches:
    first = mismatches[0]
    payload["reason"] = "prefill_replay_next_logit_argmax_mismatch"
    payload["first_mismatch"] = {
        "pos": first["pos"],
        "step": first["step"],
        "token_pos": first["token_pos"],
        "token": first["token"],
        "serial_argmax": first["serial_argmax"],
        "prefill_argmax": first["prefill_argmax"],
    }
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_STATE_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")
mismatch_pattern = re.compile(
    r"^\[dflash-rollback-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) forced_serial=1 "
    r"mismatch family=([A-Za-z0-9_]+) index=(\d+) bytes=(\d+) "
    r"differing_bytes=(\d+) first_offset=(\d+) serial_byte=(\d+) gdn_byte=(\d+)(.*)$",
    re.MULTILINE,
)
match_pattern = re.compile(
    r"^\[dflash-rollback-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) forced_serial=1 match$",
    re.MULTILINE,
)

def parse_context(context):
    parsed = {}
    if not context:
        return parsed
    for key, raw in re.findall(r"\b([A-Za-z0-9_]+)=([+\-0-9.eE]+|NaN|nan|inf|-inf)", context):
        if key in {"f32_words", "f32_bit_diff_words"}:
            parsed[key] = int(raw)
        else:
            value = float(raw)
            parsed[key] = value if math.isfinite(value) else raw
    return parsed

matches = [
    {
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
    }
    for m in match_pattern.finditer(out)
]
mismatches = []
for m in mismatch_pattern.finditer(out):
    context = m.group(11).strip() or None
    mismatches.append({
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
        "family": m.group(4),
        "index": int(m.group(5)),
        "bytes": int(m.group(6)),
        "differing_bytes": int(m.group(7)),
        "first_offset": int(m.group(8)),
        "serial_byte": int(m.group(9)),
        "gdn_byte": int(m.group(10)),
        "context": context,
        "stats": parse_context(context),
    })
checked = len(matches) + len(mismatches)
if checked == 0:
    print(json.dumps({"ok": True, "checked": 0}))
    sys.exit(0)
all_rows = matches + mismatches
single_step_matches = [r for r in matches if r["replay_steps"] == 1]
single_step_mismatches = [r for r in mismatches if r["replay_steps"] == 1]
multi_step_matches = [r for r in matches if r["replay_steps"] > 1]
multi_step_mismatches = [r for r in mismatches if r["replay_steps"] > 1]
replay_step_counts = {}
for r in all_rows:
    key = str(r["replay_steps"])
    replay_step_counts[key] = replay_step_counts.get(key, 0) + 1
mismatch_replay_step_counts = {}
for r in mismatches:
    key = str(r["replay_steps"])
    mismatch_replay_step_counts[key] = mismatch_replay_step_counts.get(key, 0) + 1
payload = {
    "ok": not mismatches,
    "checked": checked,
    "matches": len(matches),
    "mismatches": len(mismatches),
    "positions": sorted(set([r["pos"] for r in matches] + [r["pos"] for r in mismatches])),
    "min_replay_steps": min(r["replay_steps"] for r in all_rows),
    "max_replay_steps": max(r["replay_steps"] for r in all_rows),
    "replay_step_counts": replay_step_counts,
    "mismatch_replay_step_counts": mismatch_replay_step_counts,
    "single_step_checked": len(single_step_matches) + len(single_step_mismatches),
    "single_step_mismatches": len(single_step_mismatches),
    "multi_step_checked": len(multi_step_matches) + len(multi_step_mismatches),
    "multi_step_mismatches": len(multi_step_mismatches),
}
if mismatches:
    first = mismatches[0]
    payload["reason"] = "fast_replay_recurrent_state_mismatch"
    payload["first_mismatch"] = {
        "pos": first["pos"],
        "accepted": first["accepted"],
        "replay_steps": first["replay_steps"],
        "family": first["family"],
        "index": first["index"],
        "differing_bytes": first["differing_bytes"],
        "first_offset": first["first_offset"],
        "serial_byte": first["serial_byte"],
        "gdn_byte": first["gdn_byte"],
        "context": first["context"],
        "stats": first["stats"],
    }
    if single_step_mismatches:
        first_single = single_step_mismatches[0]
        payload["first_single_step_mismatch"] = {
            "pos": first_single["pos"],
            "accepted": first_single["accepted"],
            "replay_steps": first_single["replay_steps"],
            "family": first_single["family"],
            "index": first_single["index"],
            "differing_bytes": first_single["differing_bytes"],
            "first_offset": first_single["first_offset"],
            "serial_byte": first_single["serial_byte"],
            "gdn_byte": first_single["gdn_byte"],
            "context": first_single["context"],
            "stats": first_single["stats"],
        }
    if multi_step_mismatches:
        first_multi = multi_step_mismatches[0]
        payload["first_multi_step_mismatch"] = {
            "pos": first_multi["pos"],
            "accepted": first_multi["accepted"],
            "replay_steps": first_multi["replay_steps"],
            "family": first_multi["family"],
            "index": first_multi["index"],
            "differing_bytes": first_multi["differing_bytes"],
            "first_offset": first_multi["first_offset"],
            "serial_byte": first_multi["serial_byte"],
            "gdn_byte": first_multi["gdn_byte"],
            "context": first_multi["context"],
            "stats": first_multi["stats"],
        }
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_PREFILL_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")

frame_pattern = re.compile(
    r"^\[dflash-rollback-prefill-frame-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) forced_serial=1 "
    r"serial_start=(\d+) serial_end=(\d+) prefill_end=(\d+)$",
    re.MULTILINE,
)
match_pattern = re.compile(
    r"^\[dflash-rollback-prefill-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) forced_serial=1 match$",
    re.MULTILINE,
)
mismatch_pattern = re.compile(
    r"^\[dflash-rollback-prefill-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) forced_serial=1 "
    r"mismatch family=([A-Za-z0-9_]+) index=(\d+) bytes=(\d+) "
    r"differing_bytes=(\d+) first_offset=(\d+) serial_byte=(\d+) prefill_byte=(\d+)(.*)$",
    re.MULTILINE,
)

def parse_context(context):
    parsed = {}
    if not context:
        return parsed
    for key, raw in re.findall(r"\b([A-Za-z0-9_]+)=([+\-0-9.eE]+|NaN|nan|inf|-inf)", context):
        if key in {"f32_words", "f32_bit_diff_words"}:
            parsed[key] = int(raw)
        else:
            value = float(raw)
            parsed[key] = value if math.isfinite(value) else raw
    return parsed

frames = [
    {
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
        "serial_start": int(m.group(4)),
        "serial_end": int(m.group(5)),
        "prefill_end": int(m.group(6)),
    }
    for m in frame_pattern.finditer(out)
]
matches = [
    {
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
    }
    for m in match_pattern.finditer(out)
]
mismatches = []
for m in mismatch_pattern.finditer(out):
    context = m.group(11).strip() or None
    mismatches.append({
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
        "family": m.group(4),
        "index": int(m.group(5)),
        "bytes": int(m.group(6)),
        "differing_bytes": int(m.group(7)),
        "first_offset": int(m.group(8)),
        "serial_byte": int(m.group(9)),
        "prefill_byte": int(m.group(10)),
        "context": context,
        "stats": parse_context(context),
    })
checked = len(matches) + len(mismatches)
payload = {
    "ok": not mismatches,
    "checked": checked,
    "frames": frames,
    "matches": len(matches),
    "mismatches": len(mismatches),
}
if mismatches:
    payload["reason"] = "prefill_replay_recurrent_state_mismatch"
    payload["first_mismatch"] = mismatches[0]
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_INPUT_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")

frame_pattern = re.compile(
    r"^\[dflash-rollback-forced-serial-input-frame-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) "
    r"serial_start=(\d+) serial_end=(\d+) capture_end=(\d+)$",
    re.MULTILINE,
)
match_pattern = re.compile(
    r"^\[(dflash-rollback-(?:input|gdn-input-all)-compare)\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) (?:forced_serial=1 )?match$",
    re.MULTILINE,
)
mismatch_pattern = re.compile(
    r"^\[(dflash-rollback-(?:input|gdn-input-all)-compare)\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) (?:forced_serial=1 )?"
    r"mismatch family=([A-Za-z0-9_]+) index=(\d+) bytes=(\d+) "
    r"differing_bytes=(\d+) first_offset=(\d+) serial_byte=(\d+) gdn_byte=(\d+)(.*)$",
    re.MULTILINE,
)

def parse_context(context):
    parsed = {}
    if not context:
        return parsed
    for key, raw in re.findall(r"\b([A-Za-z0-9_]+)=([+\-0-9.eE]+|NaN|nan|inf|-inf)", context):
        if key in {"f32_words", "f32_bit_diff_words"}:
            parsed[key] = int(raw)
        else:
            value = float(raw)
            parsed[key] = value if math.isfinite(value) else raw
    return parsed

events = []
checks = {}
for m in match_pattern.finditer(out):
    label = m.group(1).removeprefix("dflash-rollback-").removesuffix("-compare")
    check = {
        "ok": True,
        "pos": int(m.group(2)),
        "accepted": int(m.group(3)),
        "replay_steps": int(m.group(4)),
    }
    events.append((m.start(), label, check))
    checks[label] = check

for m in mismatch_pattern.finditer(out):
    label = m.group(1).removeprefix("dflash-rollback-").removesuffix("-compare")
    context = m.group(12).strip() or None
    check = {
        "ok": False,
        "pos": int(m.group(2)),
        "accepted": int(m.group(3)),
        "replay_steps": int(m.group(4)),
        "family": m.group(5),
        "index": int(m.group(6)),
        "bytes": int(m.group(7)),
        "differing_bytes": int(m.group(8)),
        "first_offset": int(m.group(9)),
        "serial_byte": int(m.group(10)),
        "gdn_byte": int(m.group(11)),
        "context": context,
        "stats": parse_context(context),
    }
    events.append((m.start(), label, check))
    checks[label] = check

frames = [
    {
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
        "serial_start": int(m.group(4)),
        "serial_end": int(m.group(5)),
        "capture_end": int(m.group(6)),
    }
    for m in frame_pattern.finditer(out)
]

events.sort(key=lambda event: event[0])
mismatch_counts = {}
for _, label, check in events:
    if not check.get("ok", False):
        mismatch_counts[label] = mismatch_counts.get(label, 0) + 1

payload = {
    "ok": not mismatch_counts,
    "checked": len(events),
    "matches": sum(1 for _, _, check in events if check.get("ok", False)),
    "mismatches": sum(1 for _, _, check in events if not check.get("ok", False)),
    "mismatch_counts": mismatch_counts,
    "frames": frames,
    "checks": checks,
}
for _, label, check in events:
    if not check.get("ok", False):
        payload["reason"] = f"rollback_{label}_mismatch"
        first = dict(check)
        first["check"] = label
        payload["first_mismatch"] = first
        if label == "input" and check.get("family") == "qkv":
            payload["blocker_class"] = "projection_family_mismatch"
            payload["projection_split"] = {
                "serial_route": "forward_scratch_capture_gdn_tape:fused_qkvza",
                "verify_route": "verify_dflash_block:gemm_qkvza",
                "matched_prior_boundary": "x_in",
                "first_mismatch_family": "qkv",
            }
        break
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_FAST_TOKEN_MAJOR_COMPARE_PY=$(cat <<'PYEOF'
import sys, re, json, math
if len(sys.argv) != 2:
    print(json.dumps({"ok": False, "reason": "usage"}))
    sys.exit(0)
out = open(sys.argv[1], "rb").read().decode("utf-8", "replace")

frame_pattern = re.compile(
    r"^\[dflash-rollback-fast-token-major-frame-compare\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) (?:forced_serial=1 )?"
    r"serial_start=(\d+) serial_end=(\d+) replay_end=(\d+)$",
    re.MULTILINE,
)
match_pattern = re.compile(
    r"^\[(dflash-rollback-fast-token-major-(?:attn-out|serial|vs-production)-compare)\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) (?:forced_serial=1 )?match$",
    re.MULTILINE,
)
mismatch_pattern = re.compile(
    r"^\[(dflash-rollback-fast-token-major-(?:attn-out|serial|vs-production)-compare)\] "
    r"pos=(\d+) accepted=(\d+) replay_steps=(\d+) (?:forced_serial=1 )?"
    r"(?:step=(\d+) )?mismatch family=([A-Za-z0-9_]+) index=(\d+) bytes=(\d+) "
    r"differing_bytes=(\d+) first_offset=(\d+) "
    r"(?:replay_byte|serial_byte|fast_token_major_byte)=(\d+) "
    r"(?:tape_byte|fast_token_major_byte|production_byte)=(\d+)(.*)$",
    re.MULTILINE,
)

def parse_context(context):
    parsed = {}
    if not context:
        return parsed
    for key, raw in re.findall(r"\b([A-Za-z0-9_]+)=([+\-0-9.eE]+|NaN|nan|inf|-inf)", context):
        if key in {"f32_words", "f32_bit_diff_words"}:
            parsed[key] = int(raw)
        else:
            value = float(raw)
            parsed[key] = value if math.isfinite(value) else raw
    return parsed

events = []
checks = {}
for m in match_pattern.finditer(out):
    label = m.group(1).removeprefix("dflash-rollback-fast-token-major-").removesuffix("-compare")
    check = {
        "ok": True,
        "pos": int(m.group(2)),
        "accepted": int(m.group(3)),
        "replay_steps": int(m.group(4)),
    }
    events.append((m.start(), label, check))
    checks[label] = check

for m in mismatch_pattern.finditer(out):
    label = m.group(1).removeprefix("dflash-rollback-fast-token-major-").removesuffix("-compare")
    context = m.group(13).strip() or None
    check = {
        "ok": False,
        "pos": int(m.group(2)),
        "accepted": int(m.group(3)),
        "replay_steps": int(m.group(4)),
        "step": None if m.group(5) is None else int(m.group(5)),
        "family": m.group(6),
        "index": int(m.group(7)),
        "bytes": int(m.group(8)),
        "differing_bytes": int(m.group(9)),
        "first_offset": int(m.group(10)),
        "actual_byte": int(m.group(11)),
        "expected_byte": int(m.group(12)),
        "context": context,
        "stats": parse_context(context),
    }
    events.append((m.start(), label, check))
    checks[label] = check

frames = [
    {
        "pos": int(m.group(1)),
        "accepted": int(m.group(2)),
        "replay_steps": int(m.group(3)),
        "serial_start": int(m.group(4)),
        "serial_end": int(m.group(5)),
        "replay_end": int(m.group(6)),
    }
    for m in frame_pattern.finditer(out)
]

events.sort(key=lambda event: event[0])
checked = len(events)
mismatch_counts = {}
for _, label, check in events:
    if not check.get("ok", False):
        mismatch_counts[label] = mismatch_counts.get(label, 0) + 1
payload = {
    "ok": checked > 0 and not mismatch_counts,
    "checked": checked,
    "matches": sum(1 for _, _, check in events if check.get("ok", False)),
    "mismatches": sum(1 for _, _, check in events if not check.get("ok", False)),
    "mismatch_counts": mismatch_counts,
    "frames": frames,
    "checks": checks,
}
if checked == 0:
    payload["ok"] = True
if checked and not payload["ok"]:
    first_label, first = next((label, check) for _, label, check in events if not check.get("ok", False))
    payload["reason"] = f"fast_token_major_{first_label}_mismatch"
    payload["first_mismatch"] = {"check": first_label, **first}
print(json.dumps(payload, sort_keys=True))
PYEOF
)

ROLLBACK_FAST_REPLAY_ADMISSION_PY=$(cat <<'PYEOF'
import sys, json
if len(sys.argv) not in (3, 4):
    print(json.dumps({"verdict": "not_evaluated", "reason": "usage"}))
    sys.exit(0)
try:
    logit = json.loads(sys.argv[1])
    state = json.loads(sys.argv[2])
    input_compare = json.loads(sys.argv[3]) if len(sys.argv) == 4 else {}
except json.JSONDecodeError as exc:
    print(json.dumps({"verdict": "not_evaluated", "reason": f"json_decode_error:{exc}"}))
    sys.exit(0)

blockers = []
logit_checked = int(logit.get("checked", 0) or 0)
state_checked = int(state.get("checked", 0) or 0)
input_checked = int(input_compare.get("checked", 0) or 0)

if logit_checked <= 0:
    blockers.append("missing_logit_compare")
elif not logit.get("ok", False):
    blockers.append(logit.get("reason", "logit_argmax_mismatch"))
elif (logit.get("max_abs_over_margin") is not None) and logit["max_abs_over_margin"] >= 1.0:
    blockers.append("logit_margin_overrun")

if state_checked <= 0:
    blockers.append("missing_recurrent_state_compare")
elif not state.get("ok", False):
    blockers.append(state.get("reason", "recurrent_state_mismatch"))
    if int(state.get("single_step_mismatches", 0) or 0) > 0:
        blockers.append("single_step_recurrent_state_mismatch")

if input_checked > 0 and not input_compare.get("ok", False):
    blockers.append(input_compare.get("reason", "captured_input_mismatch"))
    if input_compare.get("blocker_class") == "projection_family_mismatch":
        blockers.append("projection_family_mismatch")

payload = {
    "verdict": "not_evaluated" if logit_checked <= 0 and state_checked <= 0 else ("admitted" if not blockers else "rejected"),
    "blockers": blockers,
    "logit_checked": logit_checked,
    "state_checked": state_checked,
    "input_checked": input_checked,
}
shape_keys = [
    "min_replay_steps",
    "max_replay_steps",
    "replay_step_counts",
    "mismatch_replay_step_counts",
    "single_step_checked",
    "single_step_mismatches",
    "multi_step_checked",
    "multi_step_mismatches",
]
shape = {key: state[key] for key in shape_keys if key in state}
if shape:
    payload["replay_shape"] = shape
if state.get("first_mismatch"):
    payload["first_state_mismatch"] = state["first_mismatch"]
if state.get("first_single_step_mismatch"):
    payload["first_single_step_state_mismatch"] = state["first_single_step_mismatch"]
if state.get("first_multi_step_mismatch"):
    payload["first_multi_step_state_mismatch"] = state["first_multi_step_mismatch"]
if logit.get("first_mismatch"):
    payload["first_logit_mismatch"] = logit["first_mismatch"]
if input_compare.get("first_mismatch"):
    payload["first_input_mismatch"] = input_compare["first_mismatch"]
if input_compare.get("blocker_class"):
    payload["input_blocker_class"] = input_compare["blocker_class"]
if input_compare.get("projection_split"):
    payload["projection_split"] = input_compare["projection_split"]
if input_compare.get("mismatch_counts"):
    payload["input_mismatch_counts"] = input_compare["mismatch_counts"]
if logit.get("max_abs_over_margin") is not None:
    payload["max_abs_over_margin"] = logit["max_abs_over_margin"]
if logit.get("max_abs_over_margin_row"):
    payload["max_abs_over_margin_row"] = logit["max_abs_over_margin_row"]
print(json.dumps(payload, sort_keys=True))
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

    # Pull stats lines (emitted/τ/cycles/rollback/accept_rate) for the report.
    stats=$(grep -aE '^emitted:|^cycles:|^rollback_parity:|^verify_graph:|^accept_rate:' "$out_file" | head -5)

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

    rollback_replay="not_checked"
    verify_graph="not_checked"
    rollback_logit_compare="not_checked"
    rollback_prefill_logit_compare="not_checked"
    rollback_state_compare="not_checked"
    rollback_prefill_compare="not_checked"
    rollback_input_compare="not_checked"
    rollback_fast_token_major_compare="not_checked"
    rollback_fast_replay_admission="not_checked"
    if [ "$AR_PARITY" = "1" ] && [ "$mode" = "dflash" ]; then
        rollback_replay=$(python3 -c "$ROLLBACK_REPLAY_PY" "$out_file")
        rollback_replay_ok=$(echo "$rollback_replay" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok', False))")
        if [ "$rollback_replay_ok" != "True" ]; then
            status="HARD_ERROR (rollback replay: $rollback_replay)"
            hard_errors=$((hard_errors + 1))
        fi
        verify_graph=$(python3 -c "$VERIFY_GRAPH_PY" "$out_file")
        verify_graph_ok=$(echo "$verify_graph" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok', False))")
        if [ "$verify_graph_ok" != "True" ]; then
            status="HARD_ERROR (verify graph: $verify_graph)"
            hard_errors=$((hard_errors + 1))
        fi
        rollback_logit_compare=$(python3 -c "$ROLLBACK_LOGIT_COMPARE_PY" "$out_file")
        rollback_logit_compare_ok=$(echo "$rollback_logit_compare" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ok', False))")
        if [ "$rollback_logit_compare_ok" != "True" ]; then
            status="HARD_ERROR (rollback logit compare: $rollback_logit_compare)"
            hard_errors=$((hard_errors + 1))
        fi
        rollback_prefill_logit_compare=$(python3 -c "$ROLLBACK_PREFILL_LOGIT_COMPARE_PY" "$out_file")
        rollback_state_compare=$(python3 -c "$ROLLBACK_STATE_COMPARE_PY" "$out_file")
        rollback_prefill_compare=$(python3 -c "$ROLLBACK_PREFILL_COMPARE_PY" "$out_file")
        rollback_input_compare=$(python3 -c "$ROLLBACK_INPUT_COMPARE_PY" "$out_file")
        rollback_fast_token_major_compare=$(python3 -c "$ROLLBACK_FAST_TOKEN_MAJOR_COMPARE_PY" "$out_file")
        rollback_fast_replay_admission=$(python3 -c "$ROLLBACK_FAST_REPLAY_ADMISSION_PY" "$rollback_logit_compare" "$rollback_state_compare" "$rollback_input_compare")
    fi

    if [ "$mode" = "dflash" ]; then
        record_json=$(python3 - "$label" "$mode" "$status" "$detect" "$parity" "$rollback_replay" "$rollback_logit_compare" "$rollback_prefill_logit_compare" "$rollback_state_compare" "$rollback_prefill_compare" "$rollback_input_compare" "$rollback_fast_token_major_compare" "$rollback_fast_replay_admission" "$verify_graph" "$wall" <<'PYEOF'
import json, sys

label, mode, status, detect, parity, rollback_replay, rollback_logit, rollback_prefill_logit, rollback_state, rollback_prefill, rollback_input, token_major, admission, verify_graph, wall = sys.argv[1:]

def decode(value):
    if value in {"not_checked", "not_requested"}:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

metrics = {
    "mode": mode,
    "wall_s": float(wall),
    "detector": decode(detect),
    "ar_parity": decode(parity),
    "rollback_replay": decode(rollback_replay),
    "rollback_logit_compare": decode(rollback_logit),
    "rollback_prefill_logit_compare": decode(rollback_prefill_logit),
    "rollback_state_compare": decode(rollback_state),
    "rollback_prefill_compare": decode(rollback_prefill),
    "rollback_input_compare": decode(rollback_input),
    "rollback_fast_token_major_compare": decode(token_major),
    "rollback_fast_replay_admission": decode(admission),
    "verify_graph": decode(verify_graph),
}
print(json.dumps({
    "battery": "dflash",
    "case_id": label,
    "status": status,
    "metrics": metrics,
}, sort_keys=True))
PYEOF
)
        append_trace_json_record "$record_json"
    fi

    {
        echo "## $label ($mode)"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        echo "- detector: \`$detect\`"
        echo "- ar_parity: \`$parity\`"
        echo "- rollback_replay: \`$rollback_replay\`"
        echo "- rollback_logit_compare: \`$rollback_logit_compare\`"
        echo "- rollback_prefill_logit_compare: \`$rollback_prefill_logit_compare\`"
        echo "- rollback_state_compare: \`$rollback_state_compare\`"
        echo "- rollback_prefill_compare: \`$rollback_prefill_compare\`"
        echo "- rollback_input_compare: \`$rollback_input_compare\`"
        echo "- rollback_fast_token_major_compare: \`$rollback_fast_token_major_compare\`"
        echo "- rollback_fast_replay_admission: \`$rollback_fast_replay_admission\`"
        echo "- verify_graph: \`$verify_graph\`"
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
summary_json=$(python3 - "$TRACE_JSON_OUT" <<'PYEOF'
import json
import sys

path = sys.argv[1]
text = open(path, "r", encoding="utf-8").read()
records = json.loads(text + "\n]}")["records"]
dflash_records = [r for r in records if r.get("battery") == "dflash"]
admissions = []
for record in dflash_records:
    metrics = record.get("metrics") or {}
    admission = metrics.get("rollback_fast_replay_admission")
    if isinstance(admission, dict):
        admissions.append((record.get("case_id"), admission))

counts = {"admitted": 0, "rejected": 0, "not_evaluated": 0, "other": 0}
blockers = {}
case_verdicts = []
logit_checked = 0
state_checked = 0
for case_id, admission in admissions:
    verdict = admission.get("verdict", "other")
    if verdict not in counts:
        verdict = "other"
    counts[verdict] += 1
    logit_checked += int(admission.get("logit_checked") or 0)
    state_checked += int(admission.get("state_checked") or 0)
    case_verdicts.append({"case_id": case_id, "verdict": admission.get("verdict", "other")})
    for blocker in admission.get("blockers") or []:
        blockers[blocker] = blockers.get(blocker, 0) + 1

if counts["rejected"]:
    verdict = "rejected"
elif counts["admitted"] and not counts["not_evaluated"] and not counts["other"]:
    verdict = "admitted"
elif admissions:
    verdict = "not_evaluated"
else:
    verdict = "not_checked"

print(json.dumps({
    "battery": "dflash",
    "case_id": "rollback_fast_replay_admission_summary",
    "status": verdict,
    "metrics": {
        "verdict": verdict,
        "case_count": len(dflash_records),
        "admission_count": len(admissions),
        "verdict_counts": counts,
        "blocker_counts": blockers,
        "case_verdicts": case_verdicts,
        "logit_checked": logit_checked,
        "state_checked": state_checked,
    },
}, sort_keys=True))
PYEOF
)
append_trace_json_record "$summary_json"
printf '\n]}\n' >> "$TRACE_JSON_OUT"

echo
echo "coherence report: $OUT"
echo "dflash trace json: $TRACE_JSON_OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"
exit 0

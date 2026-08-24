#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# forward-lowered parity gate — commits the manual hand≡lowered byte A/B.
#
# Rationale: the lowered forward path (SuperOp / lower_variant /
# run_layer_program, `HIPFIRE_FORWARD_LOWERED` default ON since 2026-06-07,
# commit 9afba773) is claimed byte-identical to the legacy hand-rolled
# `forward_scratch_layers` arms, but that claim has only ever rested on a
# manual fleet run cited in a commit message — there is NO committed gate
# asserting it. This script is that gate: for each model it runs the daemon's
# autoregressive decode TWICE — once with `HIPFIRE_FORWARD_LOWERED=0` (hand
# path) and once with the default (lowered path) — captures the committed
# token-id stream from each (via `HIPFIRE_EMIT_TOKEN_IDS=1`), and hard-fails
# if the two streams differ.
#
# Why this is direction-agnostic: whichever way the qwen35 dispatch question
# (ex-N6) is later resolved in N5 (keep SuperOp + delete hand arms, OR make
# the hand arms a `Forward::Custom` survivor + delete SuperOp), one of the two
# paths gets deleted. This gate proves they are equivalent *before* either is
# removed, so the survivor is trustworthy. It also catches any future drift
# that the default-ON flip would otherwise mask.
#
# Determinism: both runs use temperature 0.0 (greedy argmax) with byte-identical
# load/generate params, so identical forward output ⇒ identical committed
# tokens. A and B read the exact same prompt bytes in the same process, so the
# prompt-md5 cross-session rule does not bind here. Only `tok_id` is compared
# (the `t_ms` field in each committed event is wall-clock and nondeterministic).
#
# Exit codes:
#   0  every present model's hand stream == lowered stream
#   1  a parity mismatch or hard error (zero tokens / panic / timeout)
#   2  build or environment error
#
# Report destination: $HIPFIRE_PARITY_OUT or /tmp/forward-lowered-parity-<ts>.md
#
# Usage:
#   ./scripts/forward-lowered-parity.sh          # qwen35 dense + Paro rows
#   ./scripts/forward-lowered-parity.sh --full   # add A3B MoE rows (large)

set -u
cd "$(dirname "$0")/.."

FULL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        -h|--help) sed -n '3,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_PARITY_OUT:-/tmp/forward-lowered-parity-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOK="${HIPFIRE_PARITY_MAX_TOK:-256}"

# ── Build the daemon if missing (deltanet feature = qwen35 DeltaNet path) ──
# Unlike coherence-gate we do not stat every source; a correctness A/B run is
# usually invoked right after a build. Force a rebuild with --release yourself
# (or rm the binary) if you changed forward/dispatch code since the last build.
if [ ! -x "$EXE" ]; then
    echo "forward-lowered-parity: building daemon (--features deltanet)..." >&2
 if ! cargo build --release -p hipfire-daemon >&2; then
        echo "forward-lowered-parity: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ───────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "forward-lowered-parity" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Model matrix ───────────────────────────────────────────────────────────
# Format: "model_file|id|prompt"
# Seeded with qwen35 rows (the arch whose dispatch shape is the open question):
#   - 9b.mq4   : dense MQ4 workhorse — exercises DeltaNet + FullAttn variants
#                + FWHT rotation on the lowered Proj/GateUp bindings.
#   - 9b.q8f16 : Q8_0 path (different GemvResidual / rotation arms).
#   - a3b-paro : Givens (ParoQ4G128) QKVZA + MoE — the highest-risk lowered
#                path (per-weight Givens + Recurrent + Moe super-ops).
# A short instruction that yields a long greedy continuation maximizes the
# number of DECODE steps under test (the toggle only gates decode-side
# `forward_scratch_layers`; prefill is shared and identical across A/B).
# The model list is a single array so the other FORWARD_LOWERED arches
# (qwen2 / lfm2moe / deepseek4 / minimax — same shared toggle) drop in later.
DENSE_PROMPT="Write a short Python function called fib that returns the nth Fibonacci number, then explain how it works in two sentences."
SHEEP_PROMPT="A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

ROWS=(
    # llama-family dense (arch_id 0/1) — N5 Phase A: validates the new llama
    # forward_scratch_layers_lowered (execute_steps projections) vs the legacy
    # hand path. MQ4 → expected byte-identical (the F1 rmsnorm fix only changes
    # the Q4K branch, which MQ4 doesn't take).
    "qwen3-0.6b-llama.mq4|llama-0.6b-mq4|$DENSE_PROMPT"
    # qwen2-family dense (arch_id 7) — N5 Phase B: validates the qwen2 lowered
    # path now routes through the shared dense_forward (was the SuperOp
    # interpreter). Byte-identical by construction (same kernel sequences).
    "qwen25-0.5b-instruct.mq4|qwen25-0.5b-mq4|$DENSE_PROMPT"
    "qwen3.5-9b.mq4|qwen35-9b-mq4|$DENSE_PROMPT"
    "qwen3.5-9b.q8f16|qwen35-9b-q8|$DENSE_PROMPT"
    "qwen3.6-35b-a3b-paro.hfq|qwen36-a3b-paro|$SHEEP_PROMPT"
)
FULL_EXTRA=(
    "qwen3.5-35b-a3b.mq4|qwen35-a3b-mq4|$SHEEP_PROMPT"
    "qwen3.6-35b-a3b.mq4|qwen36-a3b-mq4|$SHEEP_PROMPT"
)
rows=("${ROWS[@]}")
[ "$FULL" -eq 1 ] && rows+=("${FULL_EXTRA[@]}")

# ── Report header ──────────────────────────────────────────────────────────
{
    echo "# forward-lowered parity (hand ≡ lowered)"
    echo
    echo "- commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:    $(date -Iseconds)"
    echo "- max_tok: $MAX_TOK   mode: $( [ "$FULL" -eq 1 ] && echo full || echo short )"
    echo
    echo "Each row runs the daemon twice (HIPFIRE_FORWARD_LOWERED=0 vs default),"
    echo "greedy temp=0, and compares the committed token-id streams. A mismatch"
    echo "means the lowered path diverged from the legacy hand path — a hard fail."
    echo
} > "$OUT"

# Extract the ordered committed token-id stream (tok_id only; t_ms excluded).
committed_ids() {
    grep -a '"type":"committed"' "$1" | python3 -c '
import sys, json
ids = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ids.append(str(json.loads(line)["tok_id"]))
    except Exception:
        pass
print("\n".join(ids))'
}

# Per-token visible text, json-encoded one token per line (newlines/commas in a
# token survive). Used as the parity stream when the daemon emits no `committed`
# events for an arch (e.g. the qwen2 decode loop) — for a same-daemon greedy A/B
# identical token text is equivalent to identical token ids.
token_text_stream() {
    grep -a '"type":"token"' "$1" | python3 -c '
import sys, json
print("\n".join(json.dumps(json.loads(l).get("text","")) for l in sys.stdin if "token" in l))'
}

# Decoded visible text (for the human-eyeball requirement in the report).
visible_text() {
    grep -a '"type":"token"' "$1" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))'
}

# Run one decode pass. $1=model_path $2=prompt $3=lowered(0|1) $4=out_file
run_pass() {
    local model_path="$1" prompt="$2" lowered="$3" out_file="$4"
    local in_file; in_file="$(mktemp /tmp/parity_in_XXXX.jsonl)"
    local prompt_json; prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":4096,"kv_mode":"q8"}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$MAX_TOK}
{"type":"unload"}
JL
    HIPFIRE_EMIT_TOKEN_IDS=1 HIPFIRE_FORWARD_LOWERED="$lowered" \
        timeout 600 "$EXE" < "$in_file" > "$out_file" 2>&1
    local ec=$?
    rm -f "$in_file"
    return $ec
}

hard_errors=0

for entry in "${rows[@]}"; do
    IFS='|' read -r model_file prompt_id prompt <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [ ! -f "$model_path" ]; then
        echo "## $model_file — $prompt_id — SKIPPED (model not present)" >> "$OUT"; echo >> "$OUT"
        continue
    fi

    echo "== $model_file / $prompt_id =="
    hand_out="$(mktemp /tmp/parity_hand_XXXX.log)"
    low_out="$(mktemp /tmp/parity_low_XXXX.log)"

    run_pass "$model_path" "$prompt" "0" "$hand_out"; ec_hand=$?
    run_pass "$model_path" "$prompt" "1" "$low_out";  ec_low=$?

    hand_ids="$(committed_ids "$hand_out")"
    low_ids="$(committed_ids "$low_out")"
    cmp_basis="committed token-id"
    if [ -z "$hand_ids" ] && [ -z "$low_ids" ]; then
        # No committed events emitted for this arch — gate on the token-text
        # stream instead (e.g. qwen2's daemon decode loop doesn't emit them).
        hand_ids="$(token_text_stream "$hand_out")"
        low_ids="$(token_text_stream "$low_out")"
        cmp_basis="token-text"
    fi
    hand_n=$(printf '%s' "$hand_ids" | grep -c .)
    low_n=$(printf '%s' "$low_ids" | grep -c .)
    hand_md5=$(printf '%s' "$hand_ids" | md5sum | awk '{print $1}')
    low_md5=$(printf '%s' "$low_ids"  | md5sum | awk '{print $1}')
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$hand_out" "$low_out" | head -1)

    status="PASS"
    if [ "$ec_hand" -ne 0 ] || [ "$ec_low" -ne 0 ] || [ "$hand_n" -eq 0 ] || [ "$low_n" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit hand=$ec_hand low=$ec_low; tokens hand=$hand_n low=$low_n; panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    elif [ "$hand_md5" != "$low_md5" ]; then
        status="HARD_ERROR (PARITY MISMATCH — hand≠lowered $cmp_basis streams)"
        hard_errors=$((hard_errors + 1))
    fi
    echo "   $status"

    {
        echo "## $model_file — $prompt_id"
        echo
        echo "- status: **$status**"
        echo "- basis: $cmp_basis"
        echo "- hand   (FORWARD_LOWERED=0): $hand_n tokens, md5 \`$hand_md5\`"
        echo "- lowered(default ON):        $low_n tokens, md5 \`$low_md5\`"
        echo
        if [ "$status" != "PASS" ] && [ "$hand_md5" != "$low_md5" ]; then
            echo '**FIRST DIVERGENCE (hand | lowered):**'
            echo
            echo '```'
            paste -d'|' <(printf '%s\n' "$hand_ids") <(printf '%s\n' "$low_ids") \
                | awk -F'|' '$1!=$2{print "pos "NR": hand="$1" lowered="$2; c++} c>=8{exit}'
            echo '```'
            echo
        fi
        if [ -n "$panic" ]; then
            echo '**PANIC/ERROR:**'; echo; echo '```'; echo "$panic"; echo '```'; echo
        fi
        echo '**Visible output (lowered path, human eyeball):**'
        echo; echo '```'; visible_text "$low_out"; echo '```'; echo
    } >> "$OUT"

    rm -f "$hand_out" "$low_out"
done

echo
echo "parity report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors row(s) failed parity or hit hard errors — gate FAILED"
    exit 1
fi
echo "all present models: hand ≡ lowered — gate PASSED"
exit 0

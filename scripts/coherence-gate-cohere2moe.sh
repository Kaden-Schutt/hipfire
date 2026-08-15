#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Coherence battery for Cohere2-MoE / North-Mini-Code (arch_id=12).
#
# The companion to scripts/coherence-gate.sh, which is Qwen/ChatML-specific
# (im_start leak, AWQ "Paris", DFlash). North is a different arch — Cohere
# agentic markers instead of ChatML, an interleaved sliding-window/global
# attention stack, and an HF-recommended temp=1.0 regime — so it gets its own
# gate (same split as coherence-gate-dflash.sh keeps DFlash separate).
#
# Runs a small fixed matrix of prompts through the daemon (greedy, like the
# Qwen gate — deterministic and reproducible) and writes a markdown report a
# human/agent reviews before committing. The gate HARD-FAILS only on
# unambiguous signals:
#   - daemon panic / non-zero exit / timeout / zero tokens emitted
#   - a daemon `{"type":"error"}` event (e.g. the long row tripping the
#     KV-capacity OOB guard)
#   - a Cohere marker leak: any `<|MARKER|>` (START_THINKING / START_TEXT /
#     START_ACTION / START_OF_TURN_TOKEN / CHATBOT_TOKEN / …) in the VISIBLE
#     token stream means the decode-loop marker state machine failed (the
#     North analog of the Qwen `<|im_start|>` leak).
# A low unique-token ratio (possible loop / garbage) is a SOFT flag in the
# report for human eyeball, not a hard fail.
#
# The `long-context` row sends a ~5.7k-token (rendered) prompt to exercise the
# sliding-window flash attention above the 4096 window AND the KV-capacity OOB
# guard; the daemon is loaded with max_seq=8192 so it fits.
#
# Exit codes:
#   0  battery ran clean — open the report and inspect coherence
#   1  a test hit a hard error (panic / error event / marker leak / timeout)
#   2  build or environment error
#
# Report destination: /tmp/coherence-cohere2moe-<timestamp>.md
#   (or $HIPFIRE_COHERENCE_OUT)
#
# Usage:
#   ./scripts/coherence-gate-cohere2moe.sh

set -u
cd "$(dirname "$0")/.."

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) sed -n '3,49p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
OUT="${HIPFIRE_COHERENCE_OUT:-/tmp/coherence-cohere2moe-$(date +%Y%m%d-%H%M%S).md}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_SEQ="${HIPFIRE_COHERENCE_MAX_SEQ:-8192}"  # > longest rendered row AND > the 4096 window

# ── Rebuild daemon if any cohere2moe-relevant source is newer than the binary ─
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-cohere2moe/src/cohere2moe.rs \
               crates/hipfire-arch-cohere2moe/src/forward.rs \
               crates/hipfire-arch-cohere2moe/src/config.rs \
               crates/hipfire-runtime/src/prompt_frame.rs \
               crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-daemon/src/main.rs \
               crates/rdna-compute/src/attention.rs \
               kernels/src/attention_flash_q8_0_tile.hip \
               kernels/src/attention_flash_q8_0_tile_batched.hip \
               kernels/src/attention_flash_asym_reduce_batched.hip; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then rebuild=1; break; fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "coherence-gate-cohere2moe: rebuilding daemon..."
 if ! cargo build --release -p hipfire-daemon >&2; then
        echo "coherence-gate-cohere2moe: build failed" >&2
        exit 2
    fi
fi

# ── GPU lock ─────────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "coherence-gate-cohere2moe" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Test matrix ──────────────────────────────────────────────────────────────
# Format: "model_file|id|prompt|max_tokens"
# `@file` reads the prompt verbatim from benchmarks/prompts/<file> (md5-tracked
# per CLAUDE.md). All rows are north-mini-code.mq4. max_tokens are generous on
# the reasoning row — North is a heavy reasoner and will otherwise hit the cap
# mid-`<think>`; the report still shows coherent reasoning either way.
TESTS=(
    "north-mini-code.mq4.hfq|cap|What is the capital of France? Answer in one short sentence.|200"
    "north-mini-code.mq4.hfq|reason|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|600"
    "north-mini-code.mq4.hfq|code|Write a one-line Python function named square that returns n*n.|256"
    # long-context: ~5.7k rendered tokens — exercises sliding-window flash
    # attention above the 4096 window + the KV-capacity OOB guard.
    #   md5(cohere2moe_long.txt) = 092f8f3205a1c46a9829cd73426e54d6
    "north-mini-code.mq4.hfq|long-context|@cohere2moe_long.txt|260"
)

# Any `<|MARKER|>` special token in visible output = marker state-machine leak.
MARKER_RE='<\|[A-Z_]+\|>'

hard_errors=0

{
    echo "# Coherence battery — Cohere2-MoE / North-Mini-Code"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- sampling: greedy (temp=0, repeat_penalty=1.05), max_seq=$MAX_SEQ"
    echo
    echo "Review each output for coherence (fluent, on-topic, not looping)."
    echo "Hard fails: panic / error event / marker leak / timeout / zero tokens."
    echo "Soft flags (eyeball): low unique-token ratio."
    echo
} > "$OUT"

for entry in "${TESTS[@]}"; do
    IFS='|' read -r model_file prompt_id prompt max_tok <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [ ! -f "$model_path" ]; then
        echo "## $model_file — $prompt_id — SKIPPED (model not present)" >> "$OUT"
        echo >> "$OUT"
        continue
    fi

    prompt_md5=""; prompt_ref=""
    if [ "${prompt:0:1}" = "@" ]; then
        prompt_ref="${prompt:1}"
        prompt_path="benchmarks/prompts/$prompt_ref"
        if [ ! -f "$prompt_path" ]; then
            echo "## $model_file — $prompt_id — SKIPPED (prompt file $prompt_path not found)" >> "$OUT"
            continue
        fi
        prompt=$(cat "$prompt_path")
        prompt_md5=$(md5sum "$prompt_path" | awk '{print $1}')
    fi

    echo "== $model_file / $prompt_id =="
    in_file="/tmp/coh_c2m_in_$$.jsonl"
    out_file="/tmp/coh_c2m_out_$$.log"
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":$MAX_SEQ}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.05}
{"type":"unload"}
JL

    t0=$(date +%s.%N)
    timeout 300 "$EXE" < "$in_file" > "$out_file" 2>&1
    ec=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print(f'{$t1 - $t0:.1f}')")

    done_line=$(grep -aE '"type":"done"' "$out_file" | head -1)
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL|error: ' "$out_file" | head -1)
    err_event=$(grep -aE '"type":"error"' "$out_file" | head -1)

    # Assemble the visible token text (reasoning + content) once.
    text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')
    marker_leaks=$(printf '%s' "$text" | grep -oE "$MARKER_RE" | wc -l | tr -d ' ')
    # Soft: unique whitespace-word ratio (only meaningful past ~40 words).
    uniq_note=$(printf '%s' "$text" | python3 -c '
import sys
w = sys.stdin.read().split()
if len(w) >= 40:
    r = len(set(w)) / len(w)
    if r < 0.15:
        print(f" (soft: low unique-word ratio {r:.2f} — possible loop/garbage)")')

    status="OK${uniq_note}"
    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        hard_errors=$((hard_errors + 1))
    elif [ -n "$err_event" ]; then
        status="HARD_ERROR (daemon error event: $err_event)"
        hard_errors=$((hard_errors + 1))
    elif [ "${marker_leaks:-0}" -gt 0 ]; then
        status="HARD_ERROR (${marker_leaks}× Cohere marker leaked into visible output — decode marker state machine failed)"
        hard_errors=$((hard_errors + 1))
    fi

    {
        echo "## $model_file — $prompt_id"
        echo
        echo "- wall: ${wall}s  status: **$status**"
        [ -n "$done_line" ] && echo "- stats: \`$done_line\`"
        if [ -n "$prompt_md5" ]; then
            echo "- prompt: \`@$prompt_ref\` (md5: \`$prompt_md5\`)"
        else
            echo "- prompt: \"$prompt\""
        fi
        echo
        if [ -n "$panic" ]; then
            echo '**PANIC/ERROR DETECTED:**'; echo; echo '```'; echo "$panic"; echo '```'; echo
        fi
        echo '**Output:**'; echo; echo '```'
        printf '%s\n' "$text"
        echo '```'; echo
    } >> "$OUT"

    rm -f "$in_file" "$out_file"
done

echo
echo "coherence report: $OUT"
if [ "$hard_errors" -gt 0 ]; then
    echo "$hard_errors test(s) hit hard errors — gate FAILED"
    exit 1
fi
echo "no hard errors — review $OUT for coherence, then commit if satisfied"
exit 0

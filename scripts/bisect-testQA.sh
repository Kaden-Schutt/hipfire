#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# QA bisect harness mirror.
# Preserves the original workflow but records explicit command statuses and clearer failure modes.
set -euo pipefail

COMMIT=${1:?Usage: ./scripts/bisect-testQA.sh <commit-hash> [label]}
LABEL=${2:-$COMMIT}

REPO=${REPO:-/home/kaden/ClaudeCode/autorocm/hipfire}
MODEL_06B=${MODEL_06B:-/home/kaden/llama.cpp/models/qwen3-0.6b-fp16}
MODEL_8B=${MODEL_8B:-/home/kaden/llama.cpp/models/qwen3-8b-fp16}
RESULTS_DIR=${RESULTS_DIR:-/tmp/bisect-results}
TIMEOUT_QUANT=${TIMEOUT_QUANT:-180}
TIMEOUT_INFER=${TIMEOUT_INFER:-120}
INFER_BIN=${INFER_BIN:-./target/release/examples/infer_hfq}
QUANT_BIN=${QUANT_BIN:-./target/release/hipfire-quantize}

mkdir -p "$RESULTS_DIR"
RESULT_FILE="$RESULTS_DIR/${COMMIT}-${LABEL//[^a-zA-Z0-9_-]/_}.txt"

log() {
    printf '%s\n' "$*" | tee -a "$RESULT_FILE"
}

classify_rc() {
    local rc=$1
    case "$rc" in
        0) printf 'OK' ;;
        124) printf 'TIMEOUT' ;;
        137) printf 'KILLED' ;;
        *) printf 'FAIL(rc=%s)' "$rc" ;;
    esac
}

run_capture() {
    local timeout_s=$1
    shift
    local output
    set +e
    output=$(timeout "$timeout_s" "$@" 2>&1)
    local rc=$?
    set -e
    printf '%s\n__RC__=%s\n' "$output" "$rc"
}

parse_infer_metrics() {
    local output=$1
    local gen prefill ntok text

    gen=$(printf '%s\n' "$output" | sed -n 's/^=== Done: .* (\([0-9.][0-9.]*\) tok\/s) ===$/\1/p' | tail -1)
    ntok=$(printf '%s\n' "$output" | sed -n 's/^=== Done: \([0-9][0-9]*\) tokens in .*$/\1/p' | tail -1)
    prefill=$(printf '%s\n' "$output" | sed -n 's/^Prompt: .* (.* tokens, \([0-9.][0-9.]*\) tok\/s).*$/\1/p' | tail -1)
    text=$(printf '%s\n' "$output" | sed -n '/^Generating/,/^===/p' | grep -v '^Generating' | grep -v '^===' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')

    printf '%s\n' "${gen:-}" "${prefill:-}" "${ntok:-0}" "${text:0:120}"
}

run_infer() {
    local model=$1
    local label=$2
    local prompt=$3
    shift 3

    local capture rc output metrics gen prefill ntok text status repeat_count coherence
    capture=$(run_capture "$TIMEOUT_INFER" "$INFER_BIN" "$model" --temp 0 "$@" "$prompt")
    rc=$(printf '%s\n' "$capture" | sed -n 's/^__RC__=//p' | tail -1)
    output=$(printf '%s\n' "$capture" | sed '/^__RC__=/d')

    mapfile -t metrics < <(parse_infer_metrics "$output")
    gen=${metrics[0]:-}
    prefill=${metrics[1]:-}
    ntok=${metrics[2]:-0}
    text=${metrics[3]:-}
    status=$(classify_rc "$rc")

    coherence="$status"
    if [[ "$rc" -eq 0 && -n "$gen" ]]; then
        repeat_count=$(printf '%s' "${text:0:200}" | tr ' ' '\n' | sed '/^$/d' | sort | uniq -c | sort -rn | awk 'NR==1 {print $1}')
        if [[ ${repeat_count:-0} -gt 10 ]]; then
            coherence="GARBAGE"
        else
            coherence="OK"
        fi
    elif [[ "$rc" -eq 0 ]]; then
        coherence="PARSE_FAIL"
    fi

    log "RESULT label='$label' status=$status coherence=$coherence gen_tok_s=${gen:-NA} prefill_tok_s=${prefill:-NA} tokens=$ntok preview='${text}'"
    if [[ "$rc" -ne 0 ]]; then
        log "OUTPUT BEGIN [$label]"
        printf '%s\n' "$output" | tee -a "$RESULT_FILE" >/dev/null
        log "OUTPUT END [$label]"
    fi
}

quantize() {
    local input_dir=$1
    local output=$2
    local fmt=$3
    local capture rc cmd size

    cmd=("$QUANT_BIN" --input "$input_dir" --output "$output")
    if [[ "$fmt" != "default" ]]; then
        cmd+=(--format "$fmt")
    fi

    capture=$(run_capture "$TIMEOUT_QUANT" "${cmd[@]}")
    rc=$(printf '%s\n' "$capture" | sed -n 's/^__RC__=//p' | tail -1)

    if [[ "$rc" -eq 0 && -f "$output" ]]; then
        size=$(ls -lh "$output" | awk '{print $5}')
        log "QUANT fmt=$fmt status=OK size=$size"
        return 0
    fi

    log "QUANT fmt=$fmt status=$(classify_rc "$rc")"
    return 1
}

cd "$REPO"
log "=== BISECT QA: $LABEL ($COMMIT) === $(date '+%Y-%m-%d %H:%M:%S')"

git checkout "$COMMIT" --quiet
rm -rf /tmp/hipfire_kernels/ .hipfire_kernels/
log "checked out $COMMIT"

log "building..."
timeout 600 cargo build --release --bin hipfire-quantize
timeout 600 cargo build --release --example infer_hfq

PROMPT_SHORT="The meaning of life is"
PROMPT_LONG="Explain the difference between a compiler and an interpreter. Give one concrete example of each, and describe when you would choose one over the other in a real project."

log ""
log "──── Qwen3-0.6B ────"

log "-- q8 (default) --"
if quantize "$MODEL_06B" /tmp/bisect-06b-q8.hfq default; then
    run_infer /tmp/bisect-06b-q8.hfq "q8 warmup" "$PROMPT_SHORT" >/dev/null || true
    run_infer /tmp/bisect-06b-q8.hfq "q8 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-q8.hfq "q8 defkv long" "$PROMPT_LONG"
    run_infer /tmp/bisect-06b-q8.hfq "q8 fp32kv short" "$PROMPT_SHORT" --fp32kv
fi

log "-- hfq4 (auto) --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4.hfq hfq4; then
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 warmup" "$PROMPT_SHORT" >/dev/null || true
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 defkv long" "$PROMPT_LONG"
    run_infer /tmp/bisect-06b-hfq4.hfq "hfq4 fp32kv short" "$PROMPT_SHORT" --fp32kv
fi

log "-- hfq4g256 --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4g256.hfq hfq4g256; then
    run_infer /tmp/bisect-06b-hfq4g256.hfq "hfq4g256 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-06b-hfq4g256.hfq "hfq4g256 defkv long" "$PROMPT_LONG"
fi

log "-- hfq4g128 --"
if quantize "$MODEL_06B" /tmp/bisect-06b-hfq4g128.hfq hfq4g128; then
    run_infer /tmp/bisect-06b-hfq4g128.hfq "hfq4g128 defkv short" "$PROMPT_SHORT"
fi

log ""
log "──── Qwen3-8B (HFQ4 only) ────"

log "-- hfq4 (auto) --"
if quantize "$MODEL_8B" /tmp/bisect-8b-hfq4.hfq hfq4; then
    run_infer /tmp/bisect-8b-hfq4.hfq "8B hfq4 warmup" "$PROMPT_SHORT" >/dev/null || true
    run_infer /tmp/bisect-8b-hfq4.hfq "8B hfq4 defkv short" "$PROMPT_SHORT"
    run_infer /tmp/bisect-8b-hfq4.hfq "8B hfq4 defkv long" "$PROMPT_LONG"
fi

log "-- hfq4g256 --"
if quantize "$MODEL_8B" /tmp/bisect-8b-hfq4g256.hfq hfq4g256; then
    run_infer /tmp/bisect-8b-hfq4g256.hfq "8B hfq4g256 defkv short" "$PROMPT_SHORT"
fi

log ""
log "=== END QA $LABEL === $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kevin Read (unverbraucht)
# hipfire — see LICENSE and NOTICE in the project root.

# A/B correctness + speed validation: two commits/branches side by side.
#
# Builds both, then runs a fixed battery of (model × prompt × GPU) and
# compares:
#   1. Correctness: token-level output at temp=0 (byte-parity check)
#   2. Speed: decode + prefill tok/s from bench_qwen35_mq4
#   3. GPU-free coverage: hipfire-dispatch-tests on the B side
#
# Usage:
#   ./scripts/ab-dispatch-validation.sh                     # full A/B
#   ./scripts/ab-dispatch-validation.sh --gpu 0             # single GPU
#   ./scripts/ab-dispatch-validation.sh --gpu 1
#   ./scripts/ab-dispatch-validation.sh --correctness-only  # skip speed
#   ./scripts/ab-dispatch-validation.sh --speed-only        # skip correctness
#   ./scripts/ab-dispatch-validation.sh --model qwen3.5-4b.mq4  # single model
#   ./scripts/ab-dispatch-validation.sh --long-prefill      # prefill=512 (default: 16)
#
# Env vars:
#   HIPFIRE_A_REF       branch/commit for side A (default: master)
#   HIPFIRE_B_REF       branch/commit for side B (default: upstream/integration/dispatch-unification)
#   HIPFIRE_KV_MODE     KV mode (default: asym3)
#   HIPFIRE_A_OUT       output directory (default: /tmp/hipfire-ab-<timestamp>)
#   HIPFIRE_MODELS_DIR  models directory (default: ~/.hipfire/models)
#   HIPFIRE_GPUS        space-separated GPU indices (default: "0 1")
#   HIPFIRE_GPU_0_BIG   set to 1 if GPU 0 has >= 32 GB VRAM (enables 9B models)
#   HIPFIRE_GRAPH       graph capture for bench (default: 0)
#
# Exit codes:
#   0  validation ran, results written to report
#   1  hard error (panic / build fail / zero tokens / >5% regression)
#   2  usage / env error

set -uo pipefail

cd "$(dirname "$0")/.."

# ── Configuration ─────────────────────────────────────────────────────────
A_REF="${HIPFIRE_A_REF:-master}"
B_REF="${HIPFIRE_B_REF:-upstream/integration/dispatch-unification}"
KV_MODE="${HIPFIRE_KV_MODE:-asym3}"
OUT="${HIPFIRE_A_OUT:-/tmp/hipfire-ab-$(date +%Y%m%d-%H%M%S)}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
BENCH_EXE="target/release/examples/bench_qwen35_mq4"
DAEMON_EXE="target/release/examples/daemon"
GRAPH="${HIPFIRE_GRAPH:-0}"

# GPU selection
read -ra GPUS_TO_TEST <<< "${HIPFIRE_GPUS:-0 1}"
GPU_BIG="${HIPFIRE_GPU_0_BIG:-0}"

# Models for correctness + speed (must fit on all tested GPUs)
MODELS=("qwen3.5-0.8b.mq4" "qwen3.5-4b.mq4")
# 9B only on GPUs with >= 24 GB VRAM
MODELS_BIG=("qwen3.5-9b.mq4")

# Fixed prompts (byte-identical — AGENTS.md §2 rule 2)
PROMPT_SHORT="What is the capital of France? Answer in one short sentence."
PROMPT_REASON="A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."
PROMPT_CODE="Write a Python function to find the longest substring without repeating characters."

MAX_TOKENS=128
TEMP=0.0
WARMUP=3
GEN=30
PREFILL=16

# ── Parse args ────────────────────────────────────────────────────────────
DO_CORRECTNESS=1
DO_SPEED=1
DO_GPU_FILTER=0
GPU_FILTER=()
DO_MODEL_FILTER=0
MODEL_FILTER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            shift
            DO_GPU_FILTER=1
            GPU_FILTER=("$1")
            ;;
        --correctness-only) DO_SPEED=0 ;;
        --speed-only) DO_CORRECTNESS=0 ;;
        --model)
            shift
            DO_MODEL_FILTER=1
            MODEL_FILTER="$1"
            ;;
        --long-prefill) PREFILL=512 ;;
        -h|--help)
            sed -n '3,30p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$DO_GPU_FILTER" -eq 1 ]; then
    GPUS_TO_TEST=("${GPU_FILTER[@]}")
fi
if [ "$DO_MODEL_FILTER" -eq 1 ]; then
    MODELS=("$MODEL_FILTER")
    MODELS_BIG=()
fi

mkdir -p "$OUT"

# ── Resolve refs to short hashes ─────────────────────────────────────────
A_HASH=$(git rev-parse --short "$A_REF" 2>/dev/null || echo "INVALID")
B_HASH=$(git rev-parse --short "$B_REF" 2>/dev/null || echo "INVALID")

if [ "$A_HASH" = "INVALID" ] || [ "$B_HASH" = "INVALID" ]; then
    echo "ERROR: cannot resolve refs: A=$A_REF B=$B_REF" >&2
    exit 2
fi

# ── Detect GPU names ─────────────────────────────────────────────────────
declare -A GPU_NAMES
for gpu in "${GPUS_TO_TEST[@]}"; do
    gpu_arch=$(ROCR_VISIBLE_DEVICES="$gpu" rocminfo 2>/dev/null \
        | grep 'Name:' | grep 'gfx' | head -1 | sed 's/.*: *//' | tr -d ' ')
    if [ -n "$gpu_arch" ]; then
        GPU_NAMES[$gpu]="$gpu_arch"
    else
        GPU_NAMES[$gpu]="gpu$gpu"
    fi
    echo "  GPU $gpu: ${GPU_NAMES[$gpu]}"
done

echo "=== hipfire A/B dispatch validation ==="
echo "  A: $A_REF ($A_HASH)"
echo "  B: $B_REF ($B_HASH)"
echo "  GPUs: ${GPUS_TO_TEST[*]}"
echo "  KV mode: $KV_MODE"
echo "  Prefill: $PREFILL tokens"
echo "  Output: $OUT"
echo

# ── Helper: detect VRAM for a GPU ────────────────────────────────────────
gpu_vram_bytes() {
    local gpu="$1"
    ROCR_VISIBLE_DEVICES="$gpu" rocm-smi --showmeminfo vram 2>/dev/null \
        | grep "VRAM Total Memory" | head -1 | grep -oE '[0-9]+' | head -1
}

# ── Helper: build at a given ref ─────────────────────────────────────────
build_ref() {
    local ref="$1"
    local label="$2"
    local hash
    hash=$(git rev-parse --short "$ref")

    echo "[build] Checking out $label ($hash)..."
    git checkout -f "$ref" >/dev/null 2>&1 || {
        echo "[build] FAIL: checkout $ref" >&2
        return 1
    }

    cargo build --release --features deltanet \
        --example bench_qwen35_mq4 \
        --example daemon \
        -p hipfire-runtime 2>&1 | tail -3

    if [ ! -x "$BENCH_EXE" ] || [ ! -x "$DAEMON_EXE" ]; then
        echo "[build] FAIL: binaries not found after build" >&2
        return 1
    fi

    # Copy binaries to per-ref paths so they survive checkout
    cp "$BENCH_EXE" "$OUT/bench_${label}"
    cp "$DAEMON_EXE" "$OUT/daemon_${label}"
    echo "[build] $label OK — binary md5s:"
    md5sum "$OUT/bench_${label}" "$OUT/daemon_${label}"
    echo
}

# ── Helper: run bench on a specific GPU + model ──────────────────────────
run_bench() {
    local gpu="$1"
    local model="$2"
    local bench_bin="$3"
    local log="$4"

    local model_path="$MODELS_DIR/$model"

    if [ ! -f "$model_path" ] && [ ! -L "$model_path" ]; then
        echo "  SKIP (model not found: $model_path)" >> "$log"
        echo "SKIP"
        return
    fi

    ROCR_VISIBLE_DEVICES="$gpu" \
    HIPFIRE_KV_MODE="$KV_MODE" \
    HIPFIRE_GRAPH="$GRAPH" \
    "$bench_bin" "$model_path" \
        --prefill "$PREFILL" --warmup "$WARMUP" --gen "$GEN" \
        > "$log.stdout" 2> "$log.stderr"

    local ec=$?
    if [ $ec -ne 0 ]; then
        echo "FAIL (exit=$ec)" >> "$log"
        head -5 "$log.stderr" >> "$log"
        echo "FAIL"
        return
    fi

    # bench_qwen35_mq4 prints SUMMARY to stderr
    local tok_s prefill_tok_s
    tok_s=$(cat "$log.stderr" "$log.stdout" 2>/dev/null \
        | grep -oE 'gen_tok_s=[0-9.]+' | head -1 | sed 's/gen_tok_s=//')
    prefill_tok_s=$(cat "$log.stderr" "$log.stdout" 2>/dev/null \
        | grep -oE 'prefill_tok_s=[0-9.]+' | tail -1 | sed 's/prefill_tok_s=//')

    if [ -z "$tok_s" ]; then
        echo "FAIL (no gen_tok_s)" >> "$log"
        echo "FAIL"
        return
    fi

    # Write both metrics to log
    echo "decode=$tok_s prefill=${prefill_tok_s:-N/A}" >> "$log"
    echo "decode=$tok_s prefill=${prefill_tok_s:-N/A}"
}

# ── Helper: run daemon for correctness (temp=0, capture tokens) ──────────
run_correctness() {
    local gpu="$1"
    local model="$2"
    local daemon_bin="$3"
    local prompt="$4"
    local label="$5"
    local out_dir="$6"

    local model_path="$MODELS_DIR/$model"

    if [ ! -f "$model_path" ] && [ ! -L "$model_path" ]; then
        echo "  SKIP (model not found)"
        return 0
    fi

    local in_file out_file
    in_file=$(mktemp /tmp/ab_in_XXXXXX.jsonl)
    out_file="$out_dir/daemon_${label}.out"

    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")

    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":2048}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":${MAX_TOKENS},"repeat_penalty":1.0}
{"type":"unload"}
JL

    ROCR_VISIBLE_DEVICES="$gpu" \
    timeout 300 "$daemon_bin" < "$in_file" > "$out_file" 2>"${out_file}.stderr"
    local ec=$?

    rm -f "$in_file"

    local n_tokens panic
    n_tokens=$(grep -ac '"type":"token"' "$out_file" 2>/dev/null || echo 0)
    panic=$(grep -aE 'panicked|thread.*panicked|FATAL' "$out_file" 2>/dev/null | head -1 || true)

    if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
        echo "  HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
        echo "HARD_ERROR" > "$out_dir/result_${label}"
        # Capture last 5 lines of stderr for diagnostics
        tail -5 "${out_file}.stderr" >> "$out_dir/result_${label}" 2>/dev/null || true
        return 1
    fi

    # Extract token texts for comparison
    grep '"type":"token"' "$out_file" | python3 -c "
import sys, json
texts = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    t = obj.get('text', '')
    texts.append(t)
print(''.join(texts))
" > "$out_dir/tokens_${label}.txt" 2>/dev/null

    echo "$n_tokens tokens" > "$out_dir/result_${label}"
    echo "  OK ($n_tokens tokens)"
    return 0
}

# ── Helper: compute delta% ───────────────────────────────────────────────
delta_pct() {
    python3 -c "
a, b = float('$1'), float('$2')
if a > 0:
    print(f'{((b - a) / a) * 100:+.1f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A"
}

is_above_threshold() {
    python3 -c "exit(0 if abs(float('$1')) > float('$2') else 1)" 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

START_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
echo "[setup] Current branch: $START_BRANCH — will restore on exit"

# Stash dirty state
git stash push -u -m "ab-dispatch-autostash-$$" >/dev/null 2>&1
STASHED=$?

trap '
    echo "[cleanup] Restoring $START_BRANCH..."
    git checkout -f "$START_BRANCH" >/dev/null 2>&1 || true
    [ '"$STASHED"' -eq 0 ] && git stash pop >/dev/null 2>&1 || true
' EXIT

# ── Phase 1: Build both sides ────────────────────────────────────────────
echo "═══ Phase 1: Build ═══════════════════════════════════════════════════"
echo

build_ref "$A_REF" "A" || { echo "FATAL: build A failed" >&2; exit 1; }
build_ref "$B_REF" "B" || { echo "FATAL: build B failed" >&2; exit 1; }

# Restore working tree
git checkout -f "$START_BRANCH" >/dev/null 2>&1 || true
[ $STASHED -eq 0 ] && git stash pop >/dev/null 2>&1 && STASHED=1  # prevent double-pop

# ── Phase 2+3: Correctness + Speed ───────────────────────────────────────
REPORT="$OUT/report.md"
HARD_ERRORS=0

{
echo "# A/B Dispatch Validation Report"
echo
echo "| Field | Value |"
echo "|---|---|"
echo "| A (baseline) | $A_REF ($A_HASH) |"
echo "| B (dispatch) | $B_REF ($B_HASH) |"
echo "| Date | $(date -Iseconds) |"
echo "| GPUs | ${GPUS_TO_TEST[*]} |"
echo "| KV mode | $KV_MODE |"
echo "| Prefill | $PREFILL tokens |"
echo

# ── Correctness ──────────────────────────────────────────────────────────
if [ "$DO_CORRECTNESS" -eq 1 ]; then
    echo "## 1. Correctness (temp=0, byte-parity)"
    echo
    echo "Each (model × prompt × GPU) is run at temp=0 on both sides."
    echo "Token output is compared for exact match. Any mismatch is flagged."
    echo
    echo "| GPU | Model | Prompt | A tokens | B tokens | Match |"
    echo "|-----|-------|--------|----------|----------|-------|"

    for gpu in "${GPUS_TO_TEST[@]}"; do
        gpu_name="${GPU_NAMES[$gpu]:-gpu$gpu}"

        models_this_gpu=("${MODELS[@]}")
        # Add big models only on GPU 0 if flagged, or if explicitly requested
        if [ "$gpu" -eq 0 ] && [ "$GPU_BIG" = "1" ]; then
            models_this_gpu+=("${MODELS_BIG[@]}")
        fi

        for model in "${models_this_gpu[@]}"; do
            for prompt_name in "reason" "code"; do
                case "$prompt_name" in
                    reason) prompt="$PROMPT_REASON" ;;
                    code)   prompt="$PROMPT_CODE" ;;
                esac

                run_id="${gpu_name}|${model}|${prompt_name}"
                run_dir="$OUT/correctness/${gpu_name}/${model}/${prompt_name}"
                mkdir -p "$run_dir"

                echo "  [$run_id] Running correctness..."

                a_result=$(run_correctness "$gpu" "$model" "$OUT/daemon_A" "$prompt" "A" "$run_dir")
                b_result=$(run_correctness "$gpu" "$model" "$OUT/daemon_B" "$prompt" "B" "$run_dir")

                a_tok=$(cat "$run_dir/result_A" 2>/dev/null | head -1 || echo "MISSING")
                b_tok=$(cat "$run_dir/result_B" 2>/dev/null | head -1 || echo "MISSING")

                if echo "$a_result" | grep -q "HARD_ERROR" || echo "$b_result" | grep -q "HARD_ERROR"; then
                    match="**HARD_ERROR**"
                    HARD_ERRORS=$((HARD_ERRORS + 1))
                elif [ ! -f "$run_dir/tokens_A.txt" ] || [ ! -f "$run_dir/tokens_B.txt" ]; then
                    match="SKIP"
                elif diff -q "$run_dir/tokens_A.txt" "$run_dir/tokens_B.txt" >/dev/null 2>&1; then
                    match="✅ match"
                else
                    match="❌ MISMATCH"
                    HARD_ERRORS=$((HARD_ERRORS + 1))
                    diff -u "$run_dir/tokens_A.txt" "$run_dir/tokens_B.txt" \
                        > "$run_dir/diff.txt" 2>/dev/null || true
                fi

                echo "| $gpu_name | $model | $prompt_name | $a_tok | $b_tok | $match |"
            done
        done
    done
    echo
fi

# ── Speed ────────────────────────────────────────────────────────────────
if [ "$DO_SPEED" -eq 1 ]; then
    echo "## 2. Speed (bench_qwen35_mq4, prefill=$PREFILL)"
    echo
    echo "| GPU | Model | A decode | B decode | Δ% | A prefill | B prefill | Δ% | Verdict |"
    echo "|-----|-------|----------|----------|-----|-----------|-----------|-----|---------|"

    for gpu in "${GPUS_TO_TEST[@]}"; do
        gpu_name="${GPU_NAMES[$gpu]:-gpu$gpu}"
        models_this_gpu=("${MODELS[@]}")
        if [ "$gpu" -eq 0 ] && [ "$GPU_BIG" = "1" ]; then
            models_this_gpu+=("${MODELS_BIG[@]}")
        fi

        for model in "${models_this_gpu[@]}"; do
            run_id="${gpu_name}|${model}"
            run_dir="$OUT/speed/${gpu_name}/${model}"
            mkdir -p "$run_dir"

            echo "  [$run_id] Running speed bench..."

            a_result=$(run_bench "$gpu" "$model" "$OUT/bench_A" "$run_dir/a.log")
            b_result=$(run_bench "$gpu" "$model" "$OUT/bench_B" "$run_dir/b.log")

            if [ "$a_result" = "SKIP" ] || [ "$b_result" = "SKIP" ]; then
                echo "| $gpu_name | $model | — | — | — | — | — | — | SKIP |"
                continue
            fi

            if [ "$a_result" = "FAIL" ] || [ "$b_result" = "FAIL" ]; then
                echo "| $gpu_name | $model | $a_result | $b_result | — | — | — | — | **FAIL** |"
                HARD_ERRORS=$((HARD_ERRORS + 1))
                continue
            fi

            # Parse results: "decode=X prefill=Y"
            a_decode=$(echo "$a_result" | grep -oE 'decode=[0-9.]+' | sed 's/decode=//')
            b_decode=$(echo "$b_result" | grep -oE 'decode=[0-9.]+' | sed 's/decode=//')
            a_prefill=$(echo "$a_result" | grep -oE 'prefill=[0-9.]+' | sed 's/prefill=//')
            b_prefill=$(echo "$b_result" | grep -oE 'prefill=[0-9.]+' | sed 's/prefill=//')

            decode_delta=$(delta_pct "$a_decode" "$b_decode")
            prefill_delta=$(delta_pct "$a_prefill" "$b_prefill")

            # Verdict based on decode delta (±5% mandatory investigation from #397)
            verdict="OK"
            if [ "$decode_delta" != "N/A" ]; then
                if is_above_threshold "$decode_delta" "5.0"; then
                    verdict="⚠️ INVESTIGATE (>±5%)"
                    HARD_ERRORS=$((HARD_ERRORS + 1))
                elif is_above_threshold "$decode_delta" "3.0"; then
                    verdict="⚠️ MARGINAL (>±3%)"
                fi
            fi

            a_pf="${a_prefill:--}"
            b_pf="${b_prefill:--}"
            pf_delta="${prefill_delta:--}"

            echo "| $gpu_name | $model | $a_decode | $b_decode | ${decode_delta}% | $a_pf | $b_pf | ${pf_delta}% | $verdict |"
        done
    done
    echo
fi

# ── GPU-free coverage tests ─────────────────────────────────────────────
echo "## 3. GPU-free dispatch coverage (hipfire-dispatch-tests)"
echo
echo "Runs on side B only. Catches missing-arm / arch dead-gate defects without a GPU."
echo

# Check if hipfire-dispatch-tests exists on side B
git checkout -f "$B_REF" >/dev/null 2>&1
if [ -d "crates/hipfire-dispatch-tests" ]; then
    coverage_out=$(cargo test -p hipfire-dispatch-tests 2>&1)
    coverage_ec=$?
    if [ $coverage_ec -eq 0 ]; then
        echo "| Test | Result |"
        echo "|------|--------|"
        echo "| hipfire-dispatch-tests | ✅ pass |"
    else
        echo "| Test | Result |"
        echo "|------|--------|"
        echo "| hipfire-dispatch-tests | ❌ FAIL |"
        echo '```'
        echo "$coverage_out" | tail -20
        echo '```'
        HARD_ERRORS=$((HARD_ERRORS + 1))
    fi
else
    echo "| Test | Result |"
    echo "|------|--------|"
    echo "| hipfire-dispatch-tests | SKIP (crate not present on B) |"
fi

# Also run dispatch crate internal tests if present
if [ -d "crates/hipfire-dispatch" ]; then
    dispatch_out=$(cargo test -p hipfire-dispatch 2>&1)
    dispatch_ec=$?
    if [ $dispatch_ec -eq 0 ]; then
        echo "| hipfire-dispatch (internal) | ✅ pass |"
    else
        echo "| hipfire-dispatch (internal) | ❌ FAIL |"
        HARD_ERRORS=$((HARD_ERRORS + 1))
    fi
fi
echo

git checkout -f "$START_BRANCH" >/dev/null 2>&1 || true

# ── Summary ──────────────────────────────────────────────────────────────
echo "## Summary"
echo
if [ "$HARD_ERRORS" -gt 0 ]; then
    echo "**$HARD_ERRORS hard error(s) found.** See rows marked HARD_ERROR / MISMATCH / FAIL above."
    echo
    echo "Next steps:"
    echo "1. Read the report sections above for details"
    echo "2. Check diff files under $OUT/correctness/"
    echo "3. Check bench logs under $OUT/speed/"
else
    echo "**All checks passed.** No hard errors, no mismatches, no >5% decode regressions."
fi
echo
echo "Full artifacts: \`$OUT\`"

} | tee "$REPORT"

echo
echo "══════════════════════════════════════════════════════════════════════"
echo "Report written to: $REPORT"
echo "Artifacts in: $OUT"

if [ "$HARD_ERRORS" -gt 0 ]; then
    exit 1
fi
exit 0

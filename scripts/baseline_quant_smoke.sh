#!/usr/bin/env bash
# Convert a local safetensors model into unquantized baselines plus quantized
# candidates, then compare greedy_dump_top5 outputs against the F32 baseline.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${HIPFIRE_BASELINE_MODEL:-/home/sadara/hipfire/models/Qwen3.5-0.8B}"
MODEL_NAME="${HIPFIRE_BASELINE_NAME:-qwen3.5-0.8b}"
MODEL_STORE="${HIPFIRE_MODEL_STORE:-$HOME/.hipfire/models}"
OUT="${HIPFIRE_BASELINE_OUT:-benchmarks/results/baseline-quant-$(date +%Y%m%d-%H%M%S)}"
PROMPT_FILE="${HIPFIRE_BASELINE_PROMPT:-benchmarks/prompts/merge_sort_thinking_off.txt}"
PROMPT_TEXT=""
FORMATS="${HIPFIRE_BASELINE_FORMATS:-f32,f16,mq4}"
MAX_GEN="${HIPFIRE_BASELINE_MAX_GEN:-64}"
KV_MODE="${HIPFIRE_BASELINE_KV_MODE:-q8}"
PROMPT_MODE="${HIPFIRE_BASELINE_PROMPT_MODE:-thinking}"
WIDE_MARGIN="${HIPFIRE_BASELINE_WIDE_MARGIN:-1.0}"
FORCE=0
SKIP_BUILD=0

model_path_for_format() {
    local format="$1"
    local hfq="$MODEL_STORE/$MODEL_NAME.$format.hfq"
    local legacy="$MODEL_STORE/$MODEL_NAME.$format"
    if [ "$FORCE" -eq 0 ] && [ -f "$hfq" ]; then
        printf '%s\n' "$hfq"
    elif [ "$FORCE" -eq 0 ] && [ -f "$legacy" ]; then
        printf '%s\n' "$legacy"
    else
        printf '%s\n' "$hfq"
    fi
}

usage() {
    sed -n '2,4p' "$0"
    cat <<USAGE

Options:
  --model-dir PATH      Safetensors model directory (default: $MODEL_DIR)
  --model-name NAME     Output artifact stem (default: $MODEL_NAME)
  --model-store DIR     Where converted .hfq files are stored (default: $MODEL_STORE)
  --formats CSV        Formats to compare; include f32 as baseline (default: $FORMATS)
  --prompt-file PATH    Prompt file to use (default: $PROMPT_FILE)
  --prompt TEXT         Inline prompt text instead of --prompt-file
  --max-gen N           Generated tokens per run (default: $MAX_GEN)
  --kv-mode MODE        Runtime KV mode for all runs (default: $KV_MODE)
  --prompt-mode MODE    greedy_dump_top5 prompt mode (default: $PROMPT_MODE)
  --out DIR             Result directory (default: $OUT)
  --force               Rebuild converted model artifacts
  --skip-build          Do not build converter/example binaries first
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --model-store) MODEL_STORE="$2"; shift 2 ;;
        --formats) FORMATS="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --prompt) PROMPT_TEXT="$2"; shift 2 ;;
        --max-gen) MAX_GEN="$2"; shift 2 ;;
        --kv-mode) KV_MODE="$2"; shift 2 ;;
        --prompt-mode) PROMPT_MODE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "missing safetensors model directory with config.json: $MODEL_DIR" >&2
    exit 2
fi

if [ -n "$PROMPT_TEXT" ]; then
    PROMPT_KIND="inline"
    PROMPT="$PROMPT_TEXT"
    PROMPT_MD5="$(printf '%s' "$PROMPT" | md5sum | awk '{print $1}')"
elif [ -f "$PROMPT_FILE" ]; then
    PROMPT_KIND="file"
    PROMPT="$(cat "$PROMPT_FILE")"
    PROMPT_MD5="$(md5sum "$PROMPT_FILE" | awk '{print $1}')"
else
    echo "missing prompt file: $PROMPT_FILE" >&2
    exit 2
fi

IFS=',' read -r -a FORMAT_ARR <<< "$FORMATS"
if [ "${#FORMAT_ARR[@]}" -lt 2 ]; then
    echo "--formats must contain f32 plus at least one comparison format" >&2
    exit 2
fi
if [ "${FORMAT_ARR[0]}" != "f32" ]; then
    echo "--formats must list f32 first; it is the comparison baseline" >&2
    exit 2
fi

mkdir -p "$MODEL_STORE" "$OUT/logits" "$OUT/prompts"

if [ "$SKIP_BUILD" -eq 0 ]; then
    cargo build --release -p hipfire-quantize --bin hipfire-quantize
    cargo build --release --features deltanet -p hipfire-runtime \
        --example greedy_dump_top5 \
        --example decode_tokens
fi

{
    echo "# Baseline Quant Smoke"
    echo
    echo "- date: $(date -Iseconds)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- model_dir: $MODEL_DIR"
    echo "- model_name: $MODEL_NAME"
    echo "- model_store: $MODEL_STORE"
    echo "- formats: $FORMATS"
    echo "- prompt_kind: $PROMPT_KIND"
    if [ "$PROMPT_KIND" = "file" ]; then echo "- prompt_file: $PROMPT_FILE"; fi
    echo "- prompt_md5: $PROMPT_MD5"
    echo "- max_gen: $MAX_GEN"
    echo "- kv_mode: $KV_MODE"
    echo "- prompt_mode: $PROMPT_MODE"
    echo "- wide_margin: $WIDE_MARGIN"
    echo
} > "$OUT/report.md"

printf '%s' "$PROMPT" > "$OUT/prompts/prompt.txt"

specs=()
for format in "${FORMAT_ARR[@]}"; do
    model="$(model_path_for_format "$format")"
    log="$OUT/$format.convert.log"
    if [ "$FORCE" -eq 1 ] || [ ! -f "$model" ]; then
        echo "== convert $format =="
        target/release/hipfire-quantize \
            --input "$MODEL_DIR" \
            --output "$model" \
            --format "$format" \
            > "$log" 2>&1
    else
        echo "== convert $format: using existing $model =="
        echo "using existing $model" > "$log"
    fi
    if [ ! -f "$model" ]; then
        echo "conversion did not produce $model" >&2
        exit 1
    fi

    prefix="$OUT/logits/$format"
    echo "== run $format =="
    PROMPT_MODE="$PROMPT_MODE" HIPFIRE_KV_MODE="$KV_MODE" \
        target/release/examples/greedy_dump_top5 \
            "$model" "$prefix" --max-gen "$MAX_GEN" "$PROMPT" \
            > "$prefix.stdout" 2> "$prefix.stderr"
    target/release/examples/decode_tokens "$model" "$prefix.tokens" \
        > "$prefix.decoded.txt" 2> "$prefix.decode.stderr"

    {
        echo "## $format"
        echo
        echo "- model: $model"
        echo "- model_md5: $(md5sum "$model" | awk '{print $1}')"
        echo "- model_size_bytes: $(stat -c%s "$model" 2>/dev/null || wc -c < "$model")"
        echo "- tokens: $(wc -l < "$prefix.tokens")"
        echo
        echo '```text'
        sed -n '1,12p' "$prefix.decoded.txt"
        echo '```'
        echo
    } >> "$OUT/report.md"

    if [ "$format" != "f32" ]; then
        specs+=("$format=$prefix")
    fi
done

./scripts/quant_compare_top5.py "$OUT/logits/f32" "${specs[@]}" --wide-margin "$WIDE_MARGIN" \
    > "$OUT/compare.json"

{
    echo "## Top-5 Comparison"
    echo
    echo '```json'
    cat "$OUT/compare.json"
    echo '```'
    echo
} >> "$OUT/report.md"

echo "baseline quant smoke report: $OUT/report.md"

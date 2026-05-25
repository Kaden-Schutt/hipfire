#!/usr/bin/env bash
# Vision quality comparison: hipfire vs llama.cpp on Qwen3.5-9B + mmproj-F16.
# Writes per-(image,prompt,engine) outputs to outputs/<image>/<prompt>__<engine>.txt
# and a summary markdown to comparison-$(date +%F).md.

set -uo pipefail
cd "$(dirname "$0")"

OUT=$HOME/git/hipfire/benchmarks/vision/outputs
mkdir -p "$OUT"

IMG_DIR=$HOME/git/hipfire/benchmarks/vision/images
IMAGES=(barney_cigar.jpg doge.jpeg doge_napping.png scene_1.jpg scene_2.jpg general_qa.jpg)

# Two consistent prompts across all images. `/no_think` suppresses Qwen3.5
# thinking blocks so output budget (n=300) doesn't get eaten by reasoning.
P1='Describe this image in 2-3 sentences. /no_think'
P2='Transcribe any visible text, signs, or written words in this image. If there is no text, say so. /no_think'
declare -A PROMPTS=( [desc]="$P1" [ocr]="$P2" )

LLAMA_BIN=$HOME/git/llm/llama.cpp/build/bin/llama-mtmd-cli
MMPROJ=/data/models/unsloth/Qwen3.5-9B/mmproj-F16.gguf
GGUF_Q4=/data/models/unsloth/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf
GGUF_Q8=/data/models/unsloth/Qwen3.5-9B/Qwen3.5-9B-Q8_0.gguf
HIPFIRE_HFQ=$HOME/models-local/qwen3.5-9b.mq4-q8head-vision-f16-spliced.hfq

source $HOME/git/hipfire/scripts/gpu-lock.sh

run_llama() {
  # $1 = label (e.g. q4km), $2 = gguf path, $3 = image path, $4 = prompt, $5 = out file
  local label=$1 gguf=$2 img=$3 prompt=$4 out=$5
  "$LLAMA_BIN" -m "$gguf" --mmproj "$MMPROJ" --image "$img" \
    -p "$prompt" --temp 0 -n 300 -ngl 99 --no-warmup 2>/dev/null \
    | awk '
        /^<think>/    { in_think=1; next }
        /^<\/think>/  { in_think=0; next }
        /^llama_perf/ { exit }
        /^\[gpu-lock\]/ { next }
        in_think      { next }
        { print }
      ' > "$out" 2>&1
}

run_hipfire_cli() {
  # $1 = image path, $2 = prompt, $3 = out file
  local img=$1 prompt=$2 out=$3
  # `hipfire run --image` self-locks; we MUST NOT wrap in gpu_acquire.
  $HOME/.hipfire/cli/index.ts >/dev/null 2>&1 || true  # ensure CLI is built/runnable
  # Use the prebuilt infer example binary (avoids re-running cargo which would
  # change CWD and break relative output paths).
  timeout 120s $HOME/git/hipfire/target/release/examples/infer \
    "$HIPFIRE_HFQ" --image "$img" --max-tokens 300 --no-think "$prompt" 2>/dev/null \
    | awk '
        /^Prompt:/        { next }
        /^Prefill:/       { next }
        /^=== Done:/      { next }
        /^\[hipGraph\]/   { next }
        /^<\|im_end\|>/   { next }
        /<think>/         { in_think=1; next }
        /<\/think>/       { in_think=0; next }
        in_think          { next }
        { print }
      ' > "$out" 2>&1
}

# === Pass 1: llama.cpp (Q4_K_M and Q8_0) ===
gpu_acquire "vision-bench-llama"
trap 'gpu_release' EXIT

for img in "${IMAGES[@]}"; do
  mkdir -p "$OUT/${img%.*}"
  for pk in desc ocr; do
    prompt="${PROMPTS[$pk]}"
    for label in q4km q80; do
      gguf_var="GGUF_${label^^}"
      gguf="${!gguf_var:-}"
      [ -z "$gguf" ] && case $label in
        q4km) gguf=$GGUF_Q4 ;;
        q80)  gguf=$GGUF_Q8 ;;
      esac
      out="$OUT/${img%.*}/${pk}__llama-${label}.txt"
      echo ">> llama-${label} ${img} ${pk}"
      run_llama "$label" "$gguf" "$IMG_DIR/$img" "$prompt" "$out"
    done
  done
done

gpu_release
trap - EXIT

# === Pass 2: hipfire (q8head-vision-f16-spliced) ===
# `cargo run --example infer` self-locks → no outer gpu_acquire
for img in "${IMAGES[@]}"; do
  for pk in desc ocr; do
    prompt="${PROMPTS[$pk]}"
    out="$OUT/${img%.*}/${pk}__hipfire-q8head.txt"
    echo ">> hipfire-q8head ${img} ${pk}"
    run_hipfire_cli "$IMG_DIR/$img" "$prompt" "$out"
  done
done

echo
echo "Bench complete. Outputs in: $(pwd)/$OUT"

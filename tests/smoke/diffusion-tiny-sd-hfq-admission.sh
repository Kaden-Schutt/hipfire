#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${HIPFIRE_TINY_SD_HFQ:-${TMPDIR:-/tmp}/hipfire-tiny-sd-batch2.hfq}"
SOURCE="${HIPFIRE_TINY_SD_SOURCE:-}"
MODEL_NAME="${HIPFIRE_TINY_SD_MODEL_NAME:-tiny-sd-batch2}"
BATCH_SIZE="${HIPFIRE_DIFFUSION_ADMISSION_BATCH_SIZE:-2}"
WIDTH="${HIPFIRE_DIFFUSION_ADMISSION_WIDTH:-64}"
HEIGHT="${HIPFIRE_DIFFUSION_ADMISSION_HEIGHT:-64}"
STEPS="${HIPFIRE_DIFFUSION_ADMISSION_STEPS:-1}"
CFG_SCALE="${HIPFIRE_DIFFUSION_ADMISSION_CFG_SCALE:-1.0}"
SCHEDULER="${HIPFIRE_DIFFUSION_ADMISSION_SCHEDULER:-Euler}"
SEED="${HIPFIRE_DIFFUSION_ADMISSION_SEED:-101}"
PROMPT="${HIPFIRE_DIFFUSION_ADMISSION_PROMPT:-hipfire Tiny-SD admission smoke}"
OUTPUT_DIR="${HIPFIRE_DIFFUSION_ADMISSION_OUTPUT_DIR:-${TMPDIR:-/tmp}/hipfire-tiny-sd-admission-smoke}"
IMPORT_TIMEOUT="${HIPFIRE_DIFFUSION_ADMISSION_IMPORT_TIMEOUT:-900s}"
SMOKE_TIMEOUT="${HIPFIRE_DIFFUSION_ADMISSION_SMOKE_TIMEOUT:-1200s}"
ROCM_DEVICE_ID="${HIPFIRE_DIFFUSION_ADMISSION_ROCM_DEVICE_ID:-}"
CARGO_FEATURES="${HIPFIRE_DIFFUSION_ADMISSION_CARGO_FEATURES:-}"
TXT2IMG_ONLY="${HIPFIRE_DIFFUSION_ADMISSION_TXT2IMG_ONLY:-0}"
SKIP_MASKED_IMG2IMG="${HIPFIRE_DIFFUSION_ADMISSION_SKIP_MASKED_IMG2IMG:-0}"

# hipfire is HIP/ROCm-first: the GPU path is always compiled, so no cargo
# feature is needed to exercise it. CARGO_FEATURES remains an opt-in passthrough
# for unrelated features.
HIPFIRE_CLI=(cargo run --release -q -p hipfire-cli)
if [[ -n "$CARGO_FEATURES" ]]; then
  HIPFIRE_CLI+=(--features "$CARGO_FEATURES")
fi
HIPFIRE_CLI+=(--)

find_tiny_sd_source() {
  if [[ -n "$SOURCE" ]]; then
    printf '%s\n' "$SOURCE"
    return 0
  fi
  local cache_root="${HIPFIRE_HF_CACHE:-/srv/huggingface}"
  local snapshots_dir="$cache_root/models--segmind--tiny-sd/snapshots"
  if [[ ! -d "$snapshots_dir" ]]; then
    return 1
  fi
  find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

if [[ ! -f "$MODEL" ]]; then
  if [[ "${HIPFIRE_DIFFUSION_ADMISSION_NO_IMPORT:-0}" = "1" ]]; then
    echo "missing Tiny-SD HFQ model: $MODEL" >&2
    echo "unset HIPFIRE_DIFFUSION_ADMISSION_NO_IMPORT or set HIPFIRE_TINY_SD_HFQ" >&2
    exit 2
  fi
  if ! SOURCE="$(find_tiny_sd_source)"; then
    echo "missing Tiny-SD HFQ model: $MODEL" >&2
    echo "set HIPFIRE_TINY_SD_HFQ or HIPFIRE_TINY_SD_SOURCE to a Diffusers Tiny-SD snapshot" >&2
    exit 2
  fi
  echo "importing Tiny-SD diffusion HFQ from $SOURCE -> $MODEL"
  timeout "$IMPORT_TIMEOUT" "${HIPFIRE_CLI[@]}" diffusion import \
    --max-batch "$BATCH_SIZE" \
    --model-name "$MODEL_NAME" \
    --output "$MODEL" \
    "$SOURCE"
fi

ROCM_ARGS=()
if [[ -n "$ROCM_DEVICE_ID" ]]; then
  ROCM_ARGS=(--rocm-device-id "$ROCM_DEVICE_ID")
fi
SMOKE_MODE_ARGS=()
if [[ "$TXT2IMG_ONLY" = "1" ]]; then
  SMOKE_MODE_ARGS+=(--txt2img-only)
fi
if [[ "$SKIP_MASKED_IMG2IMG" = "1" ]]; then
  SMOKE_MODE_ARGS+=(--skip-masked-img2img)
fi

timeout "$SMOKE_TIMEOUT" "${HIPFIRE_CLI[@]}" diffusion smoke \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --cfg-scale "$CFG_SCALE" \
  --scheduler "$SCHEDULER" \
  --prompt "$PROMPT" \
  --seed "$SEED" \
  "${ROCM_ARGS[@]}" \
  "${SMOKE_MODE_ARGS[@]}"

#!/usr/bin/env bash
# Parallel trunk-AR distillation runner.
#
# For each prompt file in --prompts-dir, runs `dflash_spec_demo --ar-baseline`
# pinned to one ROCR_VISIBLE_DEVICES slot. Captures stderr (which contains
# `AR tokens: [...]`) into --output-dir/<prompt>.stderr.txt for the
# aggregator to parse.
#
# On hiptrx (4x R9700) this gets ~4x throughput vs single-card. Tested
# locally on hipx (1 GPU) with --gpus 1 too.
#
# Usage:
#   ./scripts/distill/run_distill_parallel.sh \
#     --target ~/.hipfire/models/qwen3.5-27b.mq4 \
#     --drafter ~/.hipfire/models/qwen3.5-9b.mq4 \
#     --prompts-dir /tmp/distill_prompts \
#     --output-dir /tmp/distill_outputs \
#     --gpus 4 \
#     --max-tokens 300

set -euo pipefail

TARGET=""
DRAFTER=""
PROMPTS_DIR=""
OUTPUT_DIR=""
GPUS=4
MAX_TOKENS=300
CTX_CAPACITY=4096
KV_MODE="q8"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)        TARGET="$2"; shift 2 ;;
    --drafter)       DRAFTER="$2"; shift 2 ;;
    --prompts-dir)   PROMPTS_DIR="$2"; shift 2 ;;
    --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
    --ctx-capacity)  CTX_CAPACITY="$2"; shift 2 ;;
    --kv-mode)       KV_MODE="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TARGET"      ]] || { echo "--target required" >&2; exit 2; }
[[ -n "$DRAFTER"     ]] || { echo "--drafter required (any 9b .hfq; AR-baseline ignores it)" >&2; exit 2; }
[[ -n "$PROMPTS_DIR" ]] || { echo "--prompts-dir required" >&2; exit 2; }
[[ -n "$OUTPUT_DIR"  ]] || { echo "--output-dir required" >&2; exit 2; }
[[ -d "$PROMPTS_DIR" ]] || { echo "prompts dir not found: $PROMPTS_DIR" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"

DEMO="$PWD/target/release/examples/dflash_spec_demo"
if [[ ! -x "$DEMO" ]]; then
  echo "binary not found: $DEMO" >&2
  echo "build first: cargo build --release --example dflash_spec_demo" >&2
  exit 2
fi

mapfile -t PROMPT_FILES < <(find "$PROMPTS_DIR" -maxdepth 1 -name 'prompt_*.txt' | sort)
N_PROMPTS=${#PROMPT_FILES[@]}
[[ $N_PROMPTS -gt 0 ]] || { echo "no prompt_*.txt found in $PROMPTS_DIR" >&2; exit 2; }

echo "=== distill runner ===" >&2
echo "  target:      $TARGET" >&2
echo "  drafter:     $DRAFTER (ignored in --ar-baseline)" >&2
echo "  prompts:     $N_PROMPTS files in $PROMPTS_DIR" >&2
echo "  output:      $OUTPUT_DIR" >&2
echo "  gpus:        $GPUS  (ROCR_VISIBLE_DEVICES=0..$((GPUS-1)))" >&2
echo "  max_tokens:  $MAX_TOKENS" >&2
echo "  kv_mode:     $KV_MODE" >&2
echo "  ctx:         $CTX_CAPACITY" >&2
echo "" >&2

# Run one prompt on one GPU. Captures stderr (carries the `AR tokens: [...]`
# line we care about) and stdout (the decoded text, useful for sanity).
run_one() {
  local gpu="$1"
  local prompt_file="$2"
  local base
  base=$(basename "$prompt_file" .txt)
  local out_stderr="$OUTPUT_DIR/$base.stderr.txt"
  local out_stdout="$OUTPUT_DIR/$base.stdout.txt"

  if [[ -s "$out_stderr" ]] && grep -q "^AR tokens:" "$out_stderr"; then
    return 0
  fi

  # dflash_spec_demo only accepts --prompt <text>, not --prompt-file. Read
  # the file content into a single arg. Truncate to 6000 chars to avoid
  # exceeding the bash arg length limit on huge prompts.
  local prompt_text
  prompt_text=$(head -c 6000 "$prompt_file")

  ROCR_VISIBLE_DEVICES="$gpu" \
  HIPFIRE_NORMALIZE_PROMPT=1 \
    "$DEMO" \
      --target "$TARGET" \
      --draft "$DRAFTER" \
      --prompt "$prompt_text" \
      --max "$MAX_TOKENS" \
      --ctx "$CTX_CAPACITY" \
      --kv-mode "$KV_MODE" \
      --no-chatml \
      --ar-baseline \
      > "$out_stdout" 2> "$out_stderr" || {
    echo "  FAILED gpu=$gpu prompt=$base (rc=$?)" >&2
    # Don't delete on failure — preserve stderr for debugging.
    return 1
  }
}

export -f run_one
export DEMO TARGET DRAFTER OUTPUT_DIR MAX_TOKENS CTX_CAPACITY KV_MODE

if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY RUN — would dispatch $N_PROMPTS prompts across $GPUS GPUs" >&2
  for ((i=0; i<N_PROMPTS && i<GPUS; i++)); do
    gpu=$((i % GPUS))
    echo "  gpu=$gpu  ${PROMPT_FILES[i]}" >&2
  done
  echo "  ... (and $((N_PROMPTS - GPUS)) more)" >&2
  exit 0
fi

START=$(date +%s)
PIDS=()
GPU_BUSY=()
for ((g=0; g<GPUS; g++)); do GPU_BUSY[g]=""; done

i=0
done_count=0
fail_count=0

# Round-robin scheduler: dispatch one prompt per GPU, wait for any to free.
# A GPU is "free" when its background pid completes.
while [[ $i -lt $N_PROMPTS || ${#PIDS[@]} -gt 0 ]]; do
  for ((g=0; g<GPUS; g++)); do
    if [[ -z "${GPU_BUSY[g]}" && $i -lt $N_PROMPTS ]]; then
      pf="${PROMPT_FILES[i]}"
      run_one "$g" "$pf" &
      pid=$!
      GPU_BUSY[g]="$pid:$pf"
      PIDS+=("$pid")
      i=$((i + 1))
    fi
  done

  if [[ ${#PIDS[@]} -gt 0 ]]; then
    if ! wait -n 2>/dev/null; then
      :
    fi
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      else
        for ((g=0; g<GPUS; g++)); do
          if [[ "${GPU_BUSY[g]}" == "$pid:"* ]]; then
            GPU_BUSY[g]=""
            break
          fi
        done
        done_count=$((done_count + 1))
        if (( done_count % 10 == 0 || done_count == N_PROMPTS )); then
          elapsed=$(( $(date +%s) - START ))
          echo "  progress: $done_count/$N_PROMPTS  (${elapsed}s elapsed)" >&2
        fi
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
  fi
done

ELAPSED=$(($(date +%s) - START))
echo "" >&2
echo "DONE: $done_count/$N_PROMPTS prompts in ${ELAPSED}s" >&2
echo "  outputs in $OUTPUT_DIR/" >&2
echo "  next: scripts/distill/aggregate_argmax.py" >&2

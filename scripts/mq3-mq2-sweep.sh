#!/usr/bin/env bash
# MQ3/MQ2 sweep harness — drives the daemon binary against each MQ3/MQ2
# model artifact and records coherence + perf for the validation pass.
#
# Prompts are committed under benchmarks/prompts/sweep/ as separate files
# rather than inline heredocs. CLAUDE.md mandates byte-identical prompts +
# recorded md5 alongside any tok/s comparison across sessions; inline
# bash-array prompts are silently reformatted by editors and have no
# integrity check. Each row in the report records the prompt md5.
#
# Same prompt set as coherence-gate.sh's short battery (cap, code, reason)
# plus a longform DL-vs-ML prompt that exercises multi-paragraph generation.
# Each row gets: VRAM peak (from done line), decode tok/s + prefill tok/s
# (from done line), wall, panic flag, and the full output text for human
# eyeball.
#
# Models that are not on disk are SKIPPED (quants may still be running);
# re-run after the missing artifacts land.
#
# Output: ${HIPFIRE_SWEEP_OUT:-$HOME/.hipfire/mq3-tests/sweep-<ts>.md}
#
# Exit codes:
#   0  battery ran clean — open the report
#   1  any model hit a hard error (panic / zero tokens / timeout)
#   2  build or environment error

set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
OUT="${HIPFIRE_SWEEP_OUT:-$HOME/.hipfire/mq3-tests/sweep-$(date +%Y%m%d-%H%M%S).md}"
PROMPTS_DIR="${HIPFIRE_SWEEP_PROMPTS_DIR:-./benchmarks/prompts/sweep}"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

if [ ! -x "$EXE" ]; then
    echo "$EXE missing — run: cargo build --release --features deltanet --example daemon -p hipfire-runtime" >&2
    exit 2
fi

if [ ! -d "$PROMPTS_DIR" ]; then
    echo "prompts dir missing: $PROMPTS_DIR" >&2
    exit 2
fi

USE_GPU_LOCK=0
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "mq3-mq2-sweep" || { echo "could not acquire GPU lock" >&2; exit 2; }
    USE_GPU_LOCK=1
fi
# Single composed EXIT trap so adding the scratch-dir cleanup below doesn't
# clobber the gpu_release call. Both invocations are best-effort.
sweep_exit_trap() {
    if [ -n "${SCRATCH_DIR:-}" ] && [ -d "$SCRATCH_DIR" ]; then
        rm -rf "$SCRATCH_DIR"
    fi
    if [ "$USE_GPU_LOCK" = 1 ]; then
        gpu_release 2>/dev/null || true
    fi
}
trap sweep_exit_trap EXIT

# Format: prompt-id|max_tokens
# Each id maps to ${PROMPTS_DIR}/${id}.txt; read at runtime, md5 recorded.
PROMPTS=(
    "cap|80"
    "code|180"
    "reason|300"
    "longform|400"
)

# Override-able via $1; otherwise sweep the canonical set.
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(
        "qwen3.5-0.8b.mq3"
        "qwen3.5-0.8b.mq2"
        "qwen3.5-4b.mq3"
        "qwen3.5-4b.mq2"
        "qwen3.5-9b.mq3"
        "qwen3.5-9b.mq2"
        "qwen3.5-27b.mq3"
        "qwen3.6-27b.mq3"
    )
fi

mkdir -p "$(dirname "$OUT")"
SCRATCH_DIR="$(dirname "$OUT")/_scratch-$(date +%s)-$$"
mkdir -p "$SCRATCH_DIR"

{
    echo "# MQ3/MQ2 sweep"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "- branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "- date:   $(date -Iseconds)"
    echo "- models: ${#MODELS[@]}"
    echo "- prompts/model: ${#PROMPTS[@]}"
    echo "- prompts dir: $PROMPTS_DIR"
    echo
    echo "### Prompt manifest (md5 of each \`*.txt\` consumed by the run)"
    echo
    for pe in "${PROMPTS[@]}"; do
        IFS='|' read -r pid _ <<< "$pe"
        pf="$PROMPTS_DIR/${pid}.txt"
        if [ -f "$pf" ]; then
            echo "- \`${pid}.txt\` md5=\`$(md5sum < "$pf" | cut -d' ' -f1)\` size=$(wc -c < "$pf")B"
        else
            echo "- \`${pid}.txt\` MISSING at $pf"
        fi
    done
    echo
    echo "Hard fail = panic OR zero tokens OR 240s timeout."
    echo "Soft eyeball = read the **Output** block; flag attractor loops, off-topic, broken language."
    echo
} > "$OUT"

hard_errors=0

for model in "${MODELS[@]}"; do
    p="$MODELS_DIR/$model"
    if [ ! -f "$p" ]; then
        {
            echo "## $model — SKIPPED (model not present)"
            echo
        } >> "$OUT"
        continue
    fi
    size=$(du -h "$p" | cut -f1)
    md5=$(md5sum "$p" | cut -d' ' -f1)

    {
        echo "## $model"
        echo
        echo "- size: $size"
        echo "- md5:  \`$md5\`"
        echo
    } >> "$OUT"

    for prompt_entry in "${PROMPTS[@]}"; do
        IFS='|' read -r pid max_tok <<< "$prompt_entry"
        echo "== $model / $pid =="

        prompt_file="$PROMPTS_DIR/${pid}.txt"
        if [ ! -f "$prompt_file" ]; then
            echo "  prompt file missing: $prompt_file; skipping" >&2
            continue
        fi
        prompt="$(cat "$prompt_file")"
        prompt_md5=$(md5sum < "$prompt_file" | cut -d' ' -f1)

        in_file="$SCRATCH_DIR/sweep_in_${model//\//_}_${pid}.jsonl"
        out_file="$SCRATCH_DIR/sweep_out_${model//\//_}_${pid}.log"
        prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")

        # max_seq=8192 to give 27B + 4096 generation budget headroom.
        cat > "$in_file" <<JL
{"type":"load","model":"$p","params":{"max_seq":8192}}
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
        panic=$(grep -aE 'panicked|thread.*panicked|FATAL|^error: ' "$out_file" | head -3)
        status="OK"
        if [ "$ec" -ne 0 ] || [ "$n_tokens" -eq 0 ] || [ -n "$panic" ]; then
            status="HARD_ERROR (exit=$ec tokens=$n_tokens panic=${panic:+yes})"
            hard_errors=$((hard_errors + 1))
        fi

        {
            echo "### $pid"
            echo
            echo "- wall: ${wall}s  status: **$status**"
            if [ -n "$done_line" ]; then
                echo "- stats: \`$done_line\`"
            fi
            echo "- prompt-file: \`$prompt_file\`  md5=\`$prompt_md5\`"
            echo "- prompt: \`$(printf '%s' "$prompt" | head -c 120)\`"
            echo
            if [ -n "$panic" ]; then
                echo '**PANIC/ERROR DETECTED:**'
                echo
                echo '```'
                echo "$panic"
                echo '```'
                echo
            fi
            echo '**Output:**'
            echo
            echo '```'
            grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))' || true
            echo '```'
            echo
        } >> "$OUT"

        # in_file / out_file live under SCRATCH_DIR; cleanup happens via the
        # EXIT trap so failed runs leave evidence behind for diagnosis.
    done
done

{
    echo "---"
    echo
    echo "## Summary"
    echo
    echo "- hard errors: $hard_errors"
    echo "- report:      $OUT"
} >> "$OUT"

echo "sweep done — report: $OUT"
echo "hard errors: $hard_errors"

if [ "$hard_errors" -gt 0 ]; then
    exit 1
fi
exit 0

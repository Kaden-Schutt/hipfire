#!/usr/bin/env bash
# A/B eval: baseline MQ4 vs approximate-MQ3 (sim_mq3.py).
# Runs the same prompts through both copies and prints decoded text side-by-side.
#
# IMPORTANT: the "MQ3" file produced by sim_mq3.py is an APPROXIMATION of
# real MQ3, NOT a strict upper bound on its quality cost.
#
# Per-element MSE: simulator ~61% heavier than real MQ3 (1.61×, verified
#   numerically — 37/13500 vs 1/588 for uniform-input). The wide gap in
#   reconstruction values 6/15 → 9/15 doesn't make a wider input bin (it
#   stays 4/30), but it forces every internal bin to be poorly centered
#   on its reconstruction value (offset 1/30 from bin center), inflating
#   E[e²] by about half via the squared-mean term.
# Worst-case error: simulator ~40% heavier (10% of range vs real MQ3's
#   7.1% from the 1/14 max-error of the centered uniform grid).
#
# Probabilistically biased pessimistic at every scale, with larger gaps
# than earlier docs claimed. If this eval shows fluent output, real MQ3
# is likely viable. If it collapses, real MQ3 is probably also worse
# than baseline MQ4 but the gap is too large to bound from this harness.
# See sim_mq3.py docstring for the full derivation.
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/daemon"
SIM_DIR="${SIM_DIR:-/tmp/mq3-sim/models}"
SRC_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
OUT="${OUT:-/tmp/mq3-eval-$(date +%Y%m%d-%H%M%S).md}"

if [ ! -x "$EXE" ]; then
    echo "daemon binary not built: $EXE" >&2
    exit 2
fi

# Test set: model basename | label | prompt | max_tokens
TESTS=(
    "qwen3.5-0.8b.mq4|cap-08|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.5-0.8b.mq4|reason-08|A farmer has 17 sheep. All but 9 die. How many are left?|160"
    "qwen3.5-9b.mq4|cap-9|What is the capital of France? Answer in one short sentence.|80"
    "qwen3.5-9b.mq4|reason-9|A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number.|240"
    "qwen3.5-9b.mq4|code-9|Write a one-line Python function named square that returns n*n.|180"
)

run_one() {
    local model_path="$1" label="$2" prompt="$3" max_tok="$4"
    local prompt_json
    prompt_json=$(python3 -c "import sys,json; print(json.dumps(sys.argv[1]))" "$prompt")
    local in_file="/tmp/mq3eval_in_$$.jsonl"
    local out_file="/tmp/mq3eval_out_$$.log"
    cat > "$in_file" <<JL
{"type":"load","model":"$model_path","params":{"max_seq":4096}}
{"type":"generate","id":"r1","prompt":${prompt_json},"temperature":0.0,"max_tokens":$max_tok,"repeat_penalty":1.05}
{"type":"unload"}
JL
    timeout 180 "$EXE" < "$in_file" > "$out_file" 2>&1
    local ec=$?
    local text
    text=$(grep -a '"type":"token"' "$out_file" | python3 -c '
import sys, json
print("".join(json.loads(l).get("text","") for l in sys.stdin if "token" in l))')
    local n_tokens
    n_tokens=$(grep -ac '"type":"token"' "$out_file")
    rm -f "$in_file" "$out_file"
    echo "ec=$ec tokens=$n_tokens"
    echo "$text"
}

source ./scripts/gpu-lock.sh
gpu_acquire "mq3-sim-eval" || { echo "could not acquire GPU lock" >&2; exit 2; }
trap 'gpu_release 2>/dev/null || true' EXIT

{
    echo "# MQ3 simulation A/B"
    echo
    echo "Source: $SRC_DIR"
    echo "Sim:    $SIM_DIR"
    echo "commit: $(git rev-parse --short HEAD)"
    echo

    for entry in "${TESTS[@]}"; do
        IFS='|' read -r model_file label prompt max_tok <<<"$entry"
        src_path="$SRC_DIR/$model_file"
        sim_path="$SIM_DIR/$model_file"
        if [ ! -f "$src_path" ]; then
            echo "## $label — SKIP (no $src_path)"
            echo
            continue
        fi
        if [ ! -f "$sim_path" ]; then
            echo "## $label — SKIP (no $sim_path — run sim_mq3.py first)"
            echo
            continue
        fi
        echo "## $label  ($model_file)"
        echo
        echo "**Prompt:** \`$prompt\`"
        echo
        echo "### Baseline MQ4"
        echo '```'
        run_one "$src_path" "${label}-base" "$prompt" "$max_tok"
        echo '```'
        echo
        echo "### Simulated MQ3"
        echo '```'
        run_one "$sim_path" "${label}-mq3" "$prompt" "$max_tok"
        echo '```'
        echo
        echo "---"
        echo
    done
} > "$OUT"

echo "wrote $OUT"

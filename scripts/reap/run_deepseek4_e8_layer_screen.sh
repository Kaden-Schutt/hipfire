#!/usr/bin/env bash
# Resume-safe isolated DeepSeek-V4-Flash dense-E8 surgery screen on hipx/gfx1151.
#
# Usage:
#   scripts/reap/run_deepseek4_e8_layer_screen.sh [first-layer [last-layer]]
#
# Each candidate changes exactly one layer. The immutable MQ2-Lloyd base and
# fixed corpus are shared across runs; logits are compared against the frozen
# baseline dump after every successful evaluation.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
REMOTE_HOST=${REMOTE_HOST:-hipx}
REMOTE_REPO=${REMOTE_REPO:-/home/kaden/hipfire-ds4-gfx1151-opt}
REMOTE_BASE=${REMOTE_BASE:-/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd}
CORPUS=${CORPUS:-benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}
HIP_DEVICE=${HIP_DEVICE:-1}
CTX=${CTX:-256}
WARMUP=${WARMUP:-8}
FIRST=${1:-0}
LAST=${2:-42}
KLD_SOURCE="$ROOT/scripts/reap/kld_compare.rs"
KLD_BIN="$ROOT/target/release/deepseek4_kld_compare"

if ! [[ "$FIRST" =~ ^[0-9]+$ && "$LAST" =~ ^[0-9]+$ ]] ||
    ((FIRST < 0 || LAST > 42 || FIRST > LAST)); then
    echo "layer range must satisfy 0 <= first <= last <= 42" >&2
    exit 2
fi

BASELINE="$CAMPAIGN/results/baseline-ctx$CTX.logits"
if [[ ! -s "$BASELINE" ]]; then
    echo "missing baseline logits: $BASELINE" >&2
    exit 2
fi

mkdir -p "$CAMPAIGN/results"
if [[ ! -x "$KLD_BIN" || "$KLD_SOURCE" -nt "$KLD_BIN" ]]; then
    echo "building fast Rust logit comparator"
    rustc -O "$KLD_SOURCE" -o "$KLD_BIN"
fi

for ((layer = FIRST; layer <= LAST; layer++)); do
    printf -v tag '%02d' "$layer"
    candidate="$CAMPAIGN/candidates/layer-$tag"
    log="$CAMPAIGN/results/layer-$tag-ctx$CTX.log"
    logits="$CAMPAIGN/results/layer-$tag-ctx$CTX.logits"
    kld="$CAMPAIGN/results/layer-$tag-vs-baseline-ctx$CTX.kld.txt"

    if [[ -s "$kld" && -s "$logits" ]] &&
        rg -q "Model .*overlay ACTIVE" "$log"; then
        echo "layer $tag: complete; skipping"
        continue
    fi

    partial_log="$log.partial"
    partial_logits="$logits.partial"
    if [[ -s "$partial_log" && -s "$partial_logits" ]] &&
        rg -q "Model .*overlay ACTIVE" "$partial_log"; then
        echo "layer $tag: recovering completed temporary evaluation"
        mv "$partial_log" "$log"
        mv "$partial_logits" "$logits"
    fi

    if [[ ! -s "$log" || ! -s "$logits" ]] ||
        ! rg -q "Model .*overlay ACTIVE" "$log"; then
        echo "layer $tag: building exact eight-tensor overlay"
        "$ROOT/scripts/reap/build_deepseek4_e8_layer_overlay.sh" "$layer"

        rm -f "$partial_log" "$partial_logits"
        remote=(
            env -u HIPFIRE_DEEPSEEK4_REAP_KEEPMAP
            "HIP_VISIBLE_DEVICES=$HIP_DEVICE"
            "HIPFIRE_REAP_PLAN=$candidate"
            HIPFIRE_REQUIRE_REAP_OVERLAY=1
            ./target/release/examples/deepseek4_perplexity
            "$REMOTE_BASE"
            "$CORPUS"
            --ctx "$CTX"
            --warmup "$WARMUP"
            --dump-logits "$partial_logits"
        )
        printf -v remote_command '%q ' "${remote[@]}"

        echo "layer $tag: evaluating on $REMOTE_HOST HIP device $HIP_DEVICE"
        ssh -o BatchMode=yes "$REMOTE_HOST" \
            "cd $(printf '%q' "$REMOTE_REPO") && set -o pipefail && $remote_command 2>&1 | tee $(printf '%q' "$partial_log")"

        if ! rg -q "Model .*overlay ACTIVE" "$partial_log"; then
            echo "layer $tag: evaluator did not confirm active overlay" >&2
            exit 1
        fi
        mv "$partial_log" "$log"
        mv "$partial_logits" "$logits"
    fi

    echo "layer $tag: comparing logits against immutable MQ2-Lloyd baseline"
    "$KLD_BIN" "$BASELINE" "$logits" |
        tee "$kld.partial"
    mv "$kld.partial" "$kld"
done

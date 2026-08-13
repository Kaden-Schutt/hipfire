#!/usr/bin/env bash
# Resume-safe paired PPL/KLD screen for a cumulative or isolated DS4 E8 phase.
#
# Usage:
#   run_deepseek4_e8_phase_quality.sh \
#     <p2|p3|p3-lsq|p3-awls-head|p3-gptq-head|router|head|p2-router|p2-head> \
#     <256|1024> <wikitext|code>
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
REMOTE_HOST=${REMOTE_HOST:-hipx}
REMOTE_REPO=${REMOTE_REPO:-/home/kaden/hipfire-ds4-gfx1151-opt}
REMOTE_BASE=${REMOTE_BASE:-/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2lloyd}
HIP_DEVICE=${HIP_DEVICE:-1}
PHASE=${1:?usage: $0 <p2|p3|p3-lsq|p3-awls-head|p3-gptq-head|router|head|p2-router|p2-head> <256|1024> <wikitext|code>}
CTX=${2:?usage: $0 <p2|p3|router|head> <256|1024> <wikitext|code>}
CORPUS_TAG=${3:?usage: $0 <p2|p3|router|head> <256|1024> <wikitext|code>}
WARMUP=${WARMUP:-8}
KLD_SOURCE="$ROOT/scripts/reap/kld_compare.rs"
KLD_BIN="$ROOT/target/release/deepseek4_kld_compare"

case "$PHASE" in
    p2) CANDIDATE="$CAMPAIGN/candidates/p2-all-layers" ;;
    p3) CANDIDATE="$CAMPAIGN/candidates/p3-all-layers" ;;
    p3-lsq) CANDIDATE="$CAMPAIGN/candidates/p3-all-layers-lsq" ;;
    p3-awls-head) CANDIDATE="$CAMPAIGN/candidates/p3-all-layers-awls-head" ;;
    p3-gptq-head) CANDIDATE="$CAMPAIGN/candidates/p3-all-layers-gptq-head" ;;
    router) CANDIDATE="$CAMPAIGN/candidates/p3-router-bucket" ;;
    head) CANDIDATE="$CAMPAIGN/candidates/p3-head" ;;
    p2-router) CANDIDATE="$CAMPAIGN/candidates/p2-plus-router" ;;
    p2-head) CANDIDATE="$CAMPAIGN/candidates/p2-plus-head" ;;
    *)
        echo "phase must be p2, p3, p3-lsq, p3-awls-head, p3-gptq-head, router, head, p2-router, or p2-head" >&2
        exit 2
        ;;
esac
case "$CTX" in
    256 | 1024) ;;
    *)
        echo "ctx must be 256 or 1024" >&2
        exit 2
        ;;
esac
case "$CORPUS_TAG" in
    wikitext)
        CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
        BASELINE="$CAMPAIGN/results/baseline-wikitext-ctx$CTX.logits"
        if [[ "$CTX" == 256 ]]; then
            BASELINE="$CAMPAIGN/results/baseline-ctx256.logits"
        fi
        ;;
    code)
        CORPUS=benchmarks/prompts/longcode_pflash.jsonl
        BASELINE="$CAMPAIGN/results/baseline-code-ctx$CTX.logits"
        ;;
    *)
        echo "corpus must be wikitext or code" >&2
        exit 2
        ;;
esac

if [[ ! -s "$CANDIDATE/overlay.hfq" || ! -s "$CANDIDATE/reap_plan.json" ]]; then
    echo "candidate is incomplete: $CANDIDATE" >&2
    exit 2
fi
if [[ ! -s "$BASELINE" ]]; then
    echo "missing immutable baseline logits: $BASELINE" >&2
    exit 2
fi
if [[ ! -x "$KLD_BIN" || "$KLD_SOURCE" -nt "$KLD_BIN" ]]; then
    rustc -O "$KLD_SOURCE" -o "$KLD_BIN"
fi

tag="$PHASE-$CORPUS_TAG-ctx$CTX"
log="$CAMPAIGN/results/$tag.log"
logits="$CAMPAIGN/results/$tag.logits"
kld="$CAMPAIGN/results/$tag-vs-baseline.kld.txt"
partial_log="$log.partial"
partial_logits="$logits.partial"

if [[ ! -s "$log" || ! -s "$logits" ]] ||
    ! rg -q "Model .*overlay ACTIVE" "$log"; then
    rm -f "$partial_log" "$partial_logits"
    remote=(
        env -u HIPFIRE_DEEPSEEK4_REAP_KEEPMAP
        "HIP_VISIBLE_DEVICES=$HIP_DEVICE"
        "HIPFIRE_REAP_PLAN=$CANDIDATE"
        HIPFIRE_REQUIRE_REAP_OVERLAY=1
        ./target/release/examples/deepseek4_perplexity
        "$REMOTE_BASE"
        "$CORPUS"
        --ctx "$CTX"
        --warmup "$WARMUP"
        --dump-logits "$partial_logits"
    )
    printf -v remote_command '%q ' "${remote[@]}"
    ssh -o BatchMode=yes "$REMOTE_HOST" \
        "cd $(printf '%q' "$REMOTE_REPO") && set -o pipefail && $remote_command 2>&1 | tee $(printf '%q' "$partial_log")"
    if ! rg -q "Model .*overlay ACTIVE" "$partial_log"; then
        echo "evaluator did not confirm active overlay" >&2
        exit 1
    fi
    mv "$partial_log" "$log"
    mv "$partial_logits" "$logits"
fi

"$KLD_BIN" "$BASELINE" "$logits" | tee "$kld.partial"
mv "$kld.partial" "$kld"

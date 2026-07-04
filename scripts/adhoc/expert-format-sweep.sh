#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quality-vs-bytes sweep for MoE expert quant formats, to pick the decode
# tok/sec operating point before investing in new codecs/kernels.
#
# For one bf16 MoE hfq: builds a calibration sidecar once, quantizes to each
# format, runs single-window perplexity (ppl + top-K KLD, the qwen3.5
# `perplexity` harness), and tabulates ppl/KLD vs bytes-per-256-group (the
# decode bandwidth proxy — decode is memory-bound, so tok/sec ∝ 1/bytes).
#
# Resumable: skips a calib/quant/ppl step whose output already exists.
#
#   scripts/expert-format-sweep.sh [bf16.hfq] [corpus.txt] [out.md]
set -u
cd "$(dirname "$0")/.."

MODELS="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
BF16="${1:-$MODELS/qwen3-coder-30b-a3b-instruct-bf16.hfq}"
CORPUS="${2:-benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}"
STEM="$(basename "$BF16" .hfq)"
WORK="${HIPFIRE_SWEEP_WORK:-$HOME/.hipfire/format-sweep/$STEM}"
OUT="${3:-$WORK/sweep.md}"
CTX="${HIPFIRE_SWEEP_CTX:-2048}"
CALIB_TOKS="${HIPFIRE_SWEEP_CALIB_TOKS:-256}"
LOCK="${HIPFIRE_BIN:-./target/release/hipfire}"
LD="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
mkdir -p "$WORK"

Q=./target/release/hipfire-quantize
PPL=./target/release/examples/perplexity
COLLECT=./target/release/examples/collect_artifacts
for b in "$Q" "$PPL" "$COLLECT"; do
    [ -x "$b" ] || {
        echo "missing $b — build: cargo build --release -p hipfire-quantize --bin hipfire-quantize; cargo build --release -p hipfire-runtime --example perplexity --example collect_artifacts" >&2
        exit 2
    }
done
[ -e "$BF16" ] || {
    echo "bf16 model not found: $BF16" >&2
    exit 2
}

# format -> bytes per 256-group (the decode bandwidth proxy). oq4++ needs the
# Hessian; mq* are RTN (no calib). bf16 is the quality ground truth.
# From a .hfq source the runnable formats are oq4/oq8/mq3/mq4/mq6/qtip3.
# mq3 = plain affine 3-bit (the RTN cliff); qtip3 = advanced 3-bit trellis
# (LDLQ via HIPFIRE_QTIP_HESSIAN) — the fair "good 3-bit" point. mq2/lloyd-*
# need an HF/GGUF source (not testable from the bf16 hfq).
declare -A BYTES=([bf16]=512 [oq8++]=260 [oq4++]=132 [mq4]=136 [mq3]=104 [qtip3]=100)
FORMATS=(bf16 oq8++ oq4++ mq4 mq3 qtip3)

"$LOCK" lock acquire "expert-format-sweep" --watch-pid "$$" || {
    echo "GPU lock busy" >&2
    exit 2
}
trap '"$LOCK" lock release 2>/dev/null || true' EXIT

# Some bf16 hfqs are mis-tagged arch_id=0 but load via the qwen35 backend;
# HIPFIRE_SWEEP_ARCH overrides the arch for the calib collector.
ARCH="${HIPFIRE_SWEEP_ARCH:-}"
CALIB="$WORK/$STEM.calib.hfq"
if [ ! -e "$CALIB" ]; then
    echo "[sweep] calibrating ($CALIB_TOKS tokens${ARCH:+, arch=$ARCH}) → $CALIB"
    LD_LIBRARY_PATH="$LD" "$COLLECT" --model "$BF16" --corpus "$CORPUS" \
        --output "$CALIB" --max-tokens "$CALIB_TOKS" --kldref ${ARCH:+--arch "$ARCH"} \
        >"$WORK/calib.log" 2>&1 || { echo "  calib FAILED (see $WORK/calib.log)"; }
fi

run_ppl() { # model.hfq -> "ppl kld" (parsed from perplexity output)
    local m="$1" log="$2"
    LD_LIBRARY_PATH="$LD" "$PPL" "$m" "$CORPUS" --ctx "$CTX" --warmup 8 >"$log" 2>&1
    local ppl kld
    ppl=$(grep -oiE "ppl[= :]+[0-9.]+" "$log" | grep -oE "[0-9.]+" | tail -1)
    kld=$(grep -oiE "kld[= :]+[0-9.]+" "$log" | grep -oE "[0-9.]+" | tail -1)
    echo "${ppl:-NA} ${kld:-NA}"
}

declare -A RES_PPL RES_KLD RES_SZ
for fmt in "${FORMATS[@]}"; do
    if [ "$fmt" = "bf16" ]; then
        model="$BF16"
    else
        model="$WORK/$STEM.$fmt.hfq"
        if [ ! -e "$model" ]; then
            echo "[sweep] quantize $fmt → $(basename "$model")"
            extra=()
            qenv=()
            case "$fmt" in
                oq4++ | oq8++) [ -e "$CALIB" ] && extra=(--hessian "$CALIB") ;;
                qtip3) [ -e "$CALIB" ] && qenv=(env "HIPFIRE_QTIP_HESSIAN=$CALIB") ;;
                mq2 | lloyd-mq2) extra=(--allow-mq2) ;;
            esac
            "${qenv[@]}" "$Q" --input "$BF16" --output "$model" --format "$fmt" "${extra[@]}" \
                >"$WORK/quant.$fmt.log" 2>&1 || {
                echo "  quant $fmt FAILED (see $WORK/quant.$fmt.log)"
                continue
            }
        fi
    fi
    echo "[sweep] perplexity $fmt"
    read -r p k <<<"$(run_ppl "$model" "$WORK/ppl.$fmt.log")"
    RES_PPL[$fmt]="$p"
    RES_KLD[$fmt]="$k"
    RES_SZ[$fmt]="$(du -h "$model" 2>/dev/null | cut -f1)"
done

{
    echo "# Expert-format quality-vs-bytes sweep — $STEM"
    echo
    echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo ?)  ctx=$CTX  calib_toks=$CALIB_TOKS"
    echo "- corpus: $CORPUS"
    echo "- bytes/group = decode bandwidth proxy (decode is memory-bound → tok/sec ∝ 1/bytes)"
    echo
    printf "| format | B/group | rel tok/s vs oq4 | ppl | top-K KLD | size |\n"
    printf "|---|---|---|---|---|---|\n"
    base=${BYTES[oq4++]}
    for fmt in "${FORMATS[@]}"; do
        b=${BYTES[$fmt]:-?}
        rel=$(awk "BEGIN{ if(\"$b\"==\"?\") print \"?\"; else printf \"%.2fx\", $base/$b }")
        printf "| %s | %s | %s | %s | %s | %s |\n" \
            "$fmt" "$b" "$rel" "${RES_PPL[$fmt]:-NA}" "${RES_KLD[$fmt]:-NA}" "${RES_SZ[$fmt]:-NA}"
    done
} | tee "$OUT"
echo "[sweep] wrote $OUT"

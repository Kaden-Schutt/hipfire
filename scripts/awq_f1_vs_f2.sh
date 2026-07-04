#!/usr/bin/env bash
# awq_f1_vs_f2.sh — Comparison eval between F1 and F2 AWQ whitelists.
#
# F1: AWQ scales emitted for input-side projections only
#     (q/k/v/gate/up/router/in_proj_*) — 184 sidecars on 9B.
# F2: AWQ scales also for output-side projections
#     (o_proj/wo/out_proj/down_proj/w_down) — 248 sidecars on 9B.
#
# Toggle controlled by HIPFIRE_AWQ_F1_ONLY env var at quantize time.
# Runtime is the same binary (F2 dispatch routes via _for helpers; when
# AWQ scale is None, falls through to non-AWQ kernel byte-identically).

set -euo pipefail
cd "$(dirname "$0")/.."

BF16_DIR=/local/hipfire/Qwen3.5-9B-BF16-st
IMATRIX=benchmarks/quality-baselines/refs/qwen3.5-9b-bf16.imatrix.gguf
QUANT_BIN=target/release/hipfire-quantize
QUANT_SLOT=/local/hipfire/qwen3.5-9b-awq-current-mq4.hfq

# KLD reference is a hipfire-self HFKREF, built once by the daemon from a BF16 .hfq
# (the cross-engine `.kldref.bin` format has been removed). Scoring runs through the
# daemon kld_eval op (scripts/lib/kld_daemon.sh).
REF_HFQ=${REF_HFQ:-/local/hipfire/qwen3.5-9b-bf16.hfq}
KLDREF=${KLDREF:-/local/hipfire/qwen3.5-9b-bf16.kldref}
CORPUS=${CORPUS:-benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}
source "$(dirname "$0")/lib/kld_daemon.sh"

ALPHA=${ALPHA:-0.55}
MAX_CHUNKS=${MAX_CHUNKS:-256}
KV_MODE=${KV_MODE:-q8}

OUT_DIR=benchmarks/quality-baselines/results/2026-05-14-f1-vs-f2-n${MAX_CHUNKS}-kv${KV_MODE}-9b-gfx906
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
[ -f "$SUMMARY" ] || printf "variant\tsidecars\tquantize_sec\teval_sec\tmean_kld\tppl\tkldseq_path\n" >"$SUMMARY"

# Build the hipfire-self KLD reference once if not present.
if [ ! -f "$KLDREF" ]; then
    [ -f "$REF_HFQ" ] || {
        echo "FATAL: need REF_HFQ ($REF_HFQ, a BF16 .hfq) or a pre-built KLDREF ($KLDREF)" >&2
        exit 2
    }
    echo "building HFKREF $KLDREF from $REF_HFQ"
    kld_build_ref "$REF_HFQ" "$CORPUS" "$KLDREF" "$MAX_CHUNKS" "$KV_MODE" >/dev/null \
        || {
            echo "FATAL: daemon build_ref failed" >&2
            exit 1
        }
fi

run_one() {
    local tag="$1"    # f1 or f2
    local f1_env="$2" # 1 or 0
    local label="${tag}-a${ALPHA//./_}"
    local qlog="$OUT_DIR/${label}.quantize.log"
    local elog="$OUT_DIR/${label}.eval.log"
    local kld="$OUT_DIR/${label}.kldseq"

    echo "==== $tag (HIPFIRE_AWQ_F1_ONLY=$f1_env, alpha=$ALPHA) ===="
    rm -f "$QUANT_SLOT"

    QSTART=$SECONDS
    HIPFIRE_AWQ_F1_ONLY=$f1_env "$QUANT_BIN" \
        --input "$BF16_DIR" --output "$QUANT_SLOT" \
        --format mq4g256 --imatrix "$IMATRIX" --awq-alpha "$ALPHA" \
        >"$qlog" 2>&1
    QSEC=$((SECONDS - QSTART))

    if ! grep -q "^AWQ pre-scaling: ENABLED" "$qlog"; then
        echo "FATAL: AWQ did not enable" >&2
        tail -5 "$qlog"
        exit 1
    fi
    local sidecars=$(grep -c "^    AWQ:    " "$qlog")
    echo "  $tag sidecars=$sidecars quantize=${QSEC}s"

    ESTART=$SECONDS
    local line
    line=$(KLD_DAEMON_LOG="$elog" kld_score "$QUANT_SLOT" "$KLDREF" "$kld" "$MAX_CHUNKS" "$KV_MODE") \
        || {
            echo "  $tag EVAL FAILED — see $elog" >&2
            echo "$line" >&2
            exit 1
        }
    local mean_kld ppl
    mean_kld=$(kld_field "$line" mean_kld)
    ppl=$(kld_field "$line" ppl)
    ESEC=$((SECONDS - ESTART))
    echo "  $tag eval=${ESEC}s mean_kld=$mean_kld ppl=$ppl → $kld"

    printf "%s\t%d\t%d\t%d\t%s\t%s\t%s\n" "$tag-a${ALPHA}" "$sidecars" "$QSEC" "$ESEC" "$mean_kld" "$ppl" "$kld" >>"$SUMMARY"
}

run_one f1 1
run_one f2 0

echo ""
echo "==== DONE ===="
cat "$SUMMARY"

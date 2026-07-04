#!/usr/bin/env bash
# QTIP low-bit treatments with the fixed binary: 3INST codebook (now threaded
# into LDLQ) and the LQER low-rank weight-error residual (HIPFIRE_LOWRANK_R).
# All LDLQ-on. qtip{2,3,4} × {base, 3inst, lr16, lr32, 3inst+lr32}.
set -u
cd "$(dirname "$0")/.."
HF="${HIPFIRE_QTIP_HF:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
W="${HIPFIRE_QTIP_WORK:-$HOME/.hipfire/format-sweep/qwen3.5-0.8b-bf16}"
CALIB="$W/qwen3.5-0.8b-bf16.calib.hfq"
IM="$HOME/.hipfire/imatrix/qwen3.5-0.8b-bf16.imatrix.gguf"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
OUT="$W/qtip-residual-sweep.md"
Q=./target/release/hipfire-quantize
PPL=./target/release/examples/perplexity
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
./target/release/hipfire lock acquire "qtip-residual-sweep" --watch-pid "$$" || {
    echo "lock busy" >&2
    exit 2
}
trap './target/release/hipfire lock release 2>/dev/null || true' EXIT
declare -A R
for bits in 2 3 4; do
    for tr in base 3inst lr16 lr32 3inst-lr32; do
        env_pfx=(env "HIPFIRE_QTIP_HESSIAN=$CALIB")
        case "$tr" in
            3inst) env_pfx+=("HIPFIRE_QTIP_CODEBOOK=3inst") ;;
            lr16) env_pfx+=("HIPFIRE_LOWRANK_R=16") ;;
            lr32) env_pfx+=("HIPFIRE_LOWRANK_R=32") ;;
            3inst-lr32) env_pfx+=("HIPFIRE_QTIP_CODEBOOK=3inst" "HIPFIRE_LOWRANK_R=32") ;;
        esac
        tag="qtipR-${bits}-${tr}"
        out="$W/$tag.hfq"
        if [ ! -e "$out" ]; then
            echo "[+] $tag"
            "${env_pfx[@]}" "$Q" --input "$HF" --output "$out" --format "qtip${bits}-sim" --imatrix "$IM" >"$W/quant.$tag.log" 2>&1 || {
                echo "  FAIL: $(tail -1 "$W/quant.$tag.log")"
                continue
            }
        fi
        p=$("$PPL" "$out" "$CORPUS" --ctx 2048 --warmup 8 2>/dev/null | grep -oiE "PPL:[ ]+[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
        R[$tag]="${p:-NA}"
        echo "  $tag ppl=${p:-NA}"
    done
done
{
    echo "# QTIP residual treatments (LDLQ-on) — qwen3.5-0.8b, ppl ctx2048, bf16=24.05"
    printf "| bits | base | +3INST | +lr16 | +lr32 | +3INST+lr32 |\n|---|---|---|---|---|---|\n"
    for bits in 2 3 4; do
        printf "| %s | %s | %s | %s | %s | %s |\n" "$bits" "${R[qtipR - ${bits} - base]:-NA}" "${R[qtipR - ${bits} - 3inst]:-NA}" "${R[qtipR - ${bits} - lr16]:-NA}" "${R[qtipR - ${bits} - lr32]:-NA}" "${R[qtipR - ${bits} - 3inst - lr32]:-NA}"
    done
} | tee "$OUT"

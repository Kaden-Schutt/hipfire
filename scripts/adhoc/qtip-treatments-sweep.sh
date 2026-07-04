#!/usr/bin/env bash
# Marginal gain of each low-bit treatment on QTIP, at 2- and 3-bit, all with
# LDLQ on (the strong baseline): 1MAD vs 3INST codebook × off/on BBT-spectral
# influence scaling. All via the bit-parametric sim (emits bf16 → faithful ppl).
#   scripts/qtip-treatments-sweep.sh
set -u
cd "$(dirname "$0")/.."
HF="${HIPFIRE_QTIP_HF:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
W="${HIPFIRE_QTIP_WORK:-$HOME/.hipfire/format-sweep/qwen3.5-0.8b-bf16}"
CALIB="$W/qwen3.5-0.8b-bf16.calib.hfq"
IM="$HOME/.hipfire/imatrix/qwen3.5-0.8b-bf16.imatrix.gguf"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
OUT="$W/qtip-treatments-sweep.md"
BBT_ALPHA="${HIPFIRE_BBT_ALPHA:-0.5}"
Q=./target/release/hipfire-quantize
PPL=./target/release/examples/perplexity
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
./target/release/hipfire lock acquire "qtip-treatments-sweep" --watch-pid "$$" || {
    echo "lock busy" >&2
    exit 2
}
trap './target/release/hipfire lock release 2>/dev/null || true' EXIT

declare -A R
# treatment tag -> extra env (all LDLQ-on via HIPFIRE_QTIP_HESSIAN)
for bits in 2 3; do
    for tr in base 3inst bbt 3inst+bbt; do
        cb=1mad
        bbt=off
        case "$tr" in 3inst) cb=3inst ;; bbt) bbt=on ;; 3inst+bbt)
            cb=3inst
            bbt=on
            ;;
        esac
        tag="qtip${bits}-ldlq-${tr}"
        out="$W/$tag.hfq"
        if [ ! -e "$out" ]; then
            echo "[+] qtip${bits} LDLQ codebook=$cb bbt=$bbt"
            env_pfx=(env "HIPFIRE_QTIP_HESSIAN=$CALIB")
            [ "$cb" = 3inst ] && env_pfx+=("HIPFIRE_QTIP_CODEBOOK=3inst")
            [ "$bbt" = on ] && env_pfx+=("HIPFIRE_QTIP_BBT_ALPHA=$BBT_ALPHA")
            "${env_pfx[@]}" "$Q" --input "$HF" --output "$out" --format "qtip${bits}-sim" --imatrix "$IM" \
                >"$W/quant.$tag.log" 2>&1 || {
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
    echo "# QTIP low-bit treatments (all LDLQ-on) — qwen3.5-0.8b, ppl ctx2048, bf16=24.05, BBT α=$BBT_ALPHA"
    echo
    printf "| bits | LDLQ (1MAD) | +3INST | +BBT | +3INST+BBT |\n|---|---|---|---|---|\n"
    for bits in 2 3; do
        printf "| %s | %s | %s | %s | %s |\n" "$bits" \
            "${R[qtip${bits} - ldlq - base]:-NA}" "${R[qtip${bits} - ldlq - 3inst]:-NA}" \
            "${R[qtip${bits} - ldlq - bbt]:-NA}" "${R[qtip${bits} - ldlq - 3inst + bbt]:-NA}"
    done
} | tee "$OUT"
echo "[treatments-sweep] wrote $OUT"

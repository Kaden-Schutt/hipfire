#!/usr/bin/env bash
# Does sparse outlier protection rescue low-bit QTIP?  Sweeps the
# roughquant4-sim treatment (channel-consistent residual-stream protection +
# QTIP trellis bulk, folds for free, LDLQ via the calib Hessian) across
# bulk bitwidth × protected-fraction. All LDLQ-on (HIPFIRE_QTIP_HESSIAN).
# protect_frac=0 is the plain-qtip baseline (from qtip-bitwidth-sweep.sh).
#   scripts/qtip-outlier-sweep.sh
set -u
cd "$(dirname "$0")/.."
HF="${HIPFIRE_QTIP_HF:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
W="${HIPFIRE_QTIP_WORK:-$HOME/.hipfire/format-sweep/qwen3.5-0.8b-bf16}"
CALIB="$W/qwen3.5-0.8b-bf16.calib.hfq"
IM="$HOME/.hipfire/imatrix/qwen3.5-0.8b-bf16.imatrix.gguf"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
OUT="$W/qtip-outlier-sweep.md"
Q=./target/release/hipfire-quantize; PPL=./target/release/examples/perplexity
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
./target/release/hipfire lock acquire "qtip-outlier-sweep" --watch-pid "$$" || { echo "lock busy" >&2; exit 2; }
trap './target/release/hipfire lock release 2>/dev/null || true' EXIT

declare -A R
BULK=(2 3 4); FRAC=(0.01 0.03 0.06)
for bits in "${BULK[@]}"; do
  for frac in "${FRAC[@]}"; do
    tag="rq4-b${bits}-p${frac}"
    out="$W/$tag.hfq"
    if [ ! -e "$out" ]; then
      echo "[+] roughquant4-sim bulk_bits=$bits protect_frac=$frac (LDLQ on)"
      env HIPFIRE_QTIP_HESSIAN="$CALIB" HIPFIRE_RQ4_BULK_BITS="$bits" HIPFIRE_RQ4_PROTECT_FRAC="$frac" \
        "$Q" --input "$HF" --output "$out" --format roughquant4-sim --imatrix "$IM" \
        > "$W/quant.$tag.log" 2>&1 || { echo "  FAIL: $(tail -1 "$W/quant.$tag.log")"; continue; }
    fi
    p=$("$PPL" "$out" "$CORPUS" --ctx 2048 --warmup 8 2>/dev/null | grep -oiE "PPL:[ ]+[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    R[$tag]="${p:-NA}"; echo "  $tag ppl=${p:-NA}"
  done
done

{
  echo "# QTIP + sparse outlier protection (roughquant4-sim, LDLQ on) — qwen3.5-0.8b"
  echo "# baseline (protect_frac=0) from qtip-bitwidth-sweep.sh LDLQ column."
  echo
  printf "| bulk bits | p=1%% | p=3%% | p=6%% |\n|---|---|---|---|\n"
  for bits in "${BULK[@]}"; do
    printf "| %s | %s | %s | %s |\n" "$bits" "${R[rq4-b${bits}-p0.01]:-NA}" "${R[rq4-b${bits}-p0.03]:-NA}" "${R[rq4-b${bits}-p0.06]:-NA}"
  done
} | tee "$OUT"
echo "[outlier-sweep] wrote $OUT"

#!/usr/bin/env bash
# QTIP trellis quality across bitwidths × LDLQ, via the bit-parametric sim
# (emits bf16 → faithful ppl, no kernel needed). LDLQ on = HIPFIRE_QTIP_HESSIAN
# (the calib Hessian sidecar); off = plain MSE trellis.
#   scripts/qtip-bitwidth-sweep.sh
set -u
cd "$(dirname "$0")/.."
HF="${HIPFIRE_QTIP_HF:-/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17}"
W="${HIPFIRE_QTIP_WORK:-$HOME/.hipfire/format-sweep/qwen3.5-0.8b-bf16}"
CALIB="$W/qwen3.5-0.8b-bf16.calib.hfq"
IM="$HOME/.hipfire/imatrix/qwen3.5-0.8b-bf16.imatrix.gguf"
CORPUS=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
OUT="$W/qtip-sweep.md"
Q=./target/release/hipfire-quantize; PPL=./target/release/examples/perplexity
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/opt/rocm/lib}"
mkdir -p "$W"
./target/release/hipfire lock acquire "qtip-bitwidth-sweep" --watch-pid "$$" || { echo "lock busy" >&2; exit 2; }
trap './target/release/hipfire lock release 2>/dev/null || true' EXIT

declare -A PPL_RES
for bits in 2 3 4 6 8; do
  for ldlq in off on; do
    tag="qtip${bits}-${ldlq}ldlq"
    out="$W/$tag.hfq"
    if [ ! -e "$out" ]; then
      echo "[+] quantize qtip${bits}-sim ldlq=$ldlq"
      env_pfx=(); [ "$ldlq" = on ] && env_pfx=(env "HIPFIRE_QTIP_HESSIAN=$CALIB")
      "${env_pfx[@]}" "$Q" --input "$HF" --output "$out" --format "qtip${bits}-sim" --imatrix "$IM" \
          > "$W/quant.$tag.log" 2>&1 || { echo "  FAIL: $(tail -1 "$W/quant.$tag.log")"; continue; }
    fi
    p=$("$PPL" "$out" "$CORPUS" --ctx 2048 --warmup 8 2>/dev/null | grep -oiE "PPL:[ ]+[0-9.]+" | grep -oE "[0-9.]+" | tail -1)
    PPL_RES[$tag]="${p:-NA}"
    echo "  qtip${bits} ldlq=$ldlq ppl=${p:-NA}"
  done
done

{
  echo "# QTIP bitwidth × LDLQ sweep — qwen3.5-0.8b (sim, ppl ctx=2048, bf16=24.05)"
  echo
  echo "- commit: $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  printf "| bits | B/group | rel tok/s vs oq4 | ppl (no LDLQ) | ppl (LDLQ) |\n|---|---|---|---|---|\n"
  for bits in 2 3 4 6 8; do
    bg=$((bits*32+4)); rel=$(awk "BEGIN{printf \"%.2fx\",132/$bg}")
    printf "| %s | %s | %s | %s | %s |\n" "$bits" "$bg" "$rel" "${PPL_RES[qtip${bits}-offldlq]:-NA}" "${PPL_RES[qtip${bits}-onldlq]:-NA}"
  done
} | tee "$OUT"
echo "[qtip-sweep] wrote $OUT"

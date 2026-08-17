#!/usr/bin/env bash
# SP-E ablation runner (Plan A).
#
# Source & teacher = the existing correct-layout qwen3.6-27b.mq4 (.hfq); the
# unsloth GGUF path is layout-incompatible with hipfire's qwen35 convert
# (spec §2.2), so everything here goes through the `--input *.hfq` requant.
#
# Arms
#   bonsai-ternary  PrismML Q2_0, byte-verbatim passthrough      — the bar
#   bonsai-binary   PrismML Q1_0, byte-verbatim passthrough      — the 1-bit bar
#   spe-tq2-sweep   our PTQ, scale-swept packer, uniform weights — R0
#   spe-tq2-awqim   our PTQ, scale-swept + AWQ-derived imatrix   — R1
#   spe-bq1-sweep / spe-bq1-awqim   same two, 1-bit
#
# The teacher .kldref is built ONCE from the mq4 and cached; every arm is
# scored against that same reference and token stream.
#
# SCORING MODE: per-token. The teacher (build_kld_ref_native) is per-token, and
# TQ2G128/BQ1G128 are not in `is_batchable_la`, so `--scoring-mode prefill`
# drops to a per-token fallback anyway — but through a hidden-state capture path
# that has no parity gate for these dtypes. Per-token is the validated path
# (identity control: mq4 vs its own ref = KLD 0.000000).
#
# Usage: spe_ablation.sh {canary|full} [MAX_CHUNKS] [N_CTX]
set -euo pipefail

REPO=/home/nick/.hipfire/src
cd "$REPO"

MQ4=/home/nick/.hipfire/models/qwen3.6-27b.mq4
MODELS=/data/hipfire-models
SLICE=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
RUN="cargo run --release"

MODE="${1:-canary}"
case "$MODE" in
  canary) MAX_CHUNKS="${2:-4}";  N_CTX="${3:-512}"; OUTDIR=/data/spe-canary ;;
  full)   MAX_CHUNKS="${2:-32}"; N_CTX="${3:-512}"; OUTDIR=/data/spe-ablation ;;
  *) echo "usage: $0 {canary|full} [MAX_CHUNKS] [N_CTX]" >&2; exit 2 ;;
esac
mkdir -p "$OUTDIR"
REF="$OUTDIR/teacher.kldref.bin"

# variant -> model path
declare -A ARMS=(
  [bonsai-ternary]="$MODELS/ternary-bonsai-27b.hfq"
  [bonsai-binary]="$MODELS/binary-bonsai-27b/bb-fix.hfq"
  [spe-tq2-sweep]="$MODELS/spe-tq2-sweep.hfq"
  [spe-tq2-awqim]="$MODELS/spe-tq2-awqim.hfq"
)
[[ "$MODE" == "full" ]] && ARMS+=(
  [spe-bq1-sweep]="$MODELS/spe-bq1-sweep.hfq"
  [spe-bq1-awqim]="$MODELS/spe-bq1-awqim.hfq"
)

echo "=== [1] teacher kldref from mq4 (chunks=$MAX_CHUNKS n-ctx=$N_CTX) ==="; date
if [[ -s "$REF" ]]; then
  echo "  reusing cached $REF ($(stat -c%s "$REF") B)"
else
  $RUN -p hipfire-runtime --example build_kld_ref_native --features deltanet,arch-qwen35 -- \
    --model "$MQ4" --slice "$SLICE" --top-k 256 --n-ctx "$N_CTX" \
    --max-chunks "$MAX_CHUNKS" --output "$REF"
fi

echo "=== [2] provenance of every arm (what are we actually scoring?) ==="
# Not decoration. The 2026-07-16 canary scored a Bonsai ternary built before
# that day's norm-bias fix and reported KLD 6.15 for a model that measures
# 0.61; nothing in the run said so. Print it, every time, before the numbers.
python3 benchmarks/quality-baselines/harness/hfq_provenance.py "$MQ4" "${ARMS[@]}" || true

echo "=== [3] eval arms ==="; date
missing=()
for variant in "${!ARMS[@]}"; do
  model="${ARMS[$variant]}"
  if [[ ! -s "$model" ]]; then
    echo "  SKIP $variant — model not built: $model"
    missing+=("$variant")
    continue
  fi
  out="$OUTDIR/${variant}__qwen36-27b__per-token.kldseq"
  if [[ -s "$out" ]]; then echo "  reusing $out"; continue; fi
  echo "--- $variant ---"; date
  $RUN -p hipfire-runtime --example eval_hipfire --features deltanet -- \
    --model "$model" --ref "$REF" --output "$out" \
    --kv-mode asym3 --scoring-mode per-token --max-chunks "$MAX_CHUNKS"
done

echo "=== [4] reduce ==="; date
python3 benchmarks/quality-baselines/harness/kld_reduce.py \
  --result-dir "$OUTDIR" \
  --out-md "$OUTDIR/result-table.md" \
  --out-json "$OUTDIR/result-data.json"
cat "$OUTDIR/result-table.md"

# Never let a skipped arm read as a complete table.
if (( ${#missing[@]} )); then
  echo
  echo "INCOMPLETE — ${#missing[@]} arm(s) had no model on disk: ${missing[*]}"
  echo "Build them first (hipfire-quantize --input \$MQ4 --format ternary|binary [--awq-imatrix 0.55])."
fi
echo "=== DONE ($MODE) ==="; date

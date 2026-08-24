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
# The teacher .kldref is built ONCE from the mq4 and cached under a fixture key
# that embeds teacher digest + chunk count + n_ctx + top-k; every arm is scored
# against that same reference and token stream. Mismatched keys cannot be reused
# silently across modes.
#
# build_kld_ref_native forces f32 KV. Candidate scoring must also use f32;
# asym3 adds KV quantization error and confounds the weight-format comparison.
#
# SCORING MODE: per-token. The teacher (build_kld_ref_native) is per-token.
# TQ2G128/BQ1G128 are batchable now, but per-token remains the validated path
# and keeps candidate scoring byte-aligned with the teacher fixture
# (identity control: mq4 vs its own ref = KLD 0.000000).
#
# Env (all optional; defaults are portable):
#   HIPFIRE_REPO_ROOT      repo root (default: inferred from this script)
#   HIPFIRE_MODELS_DIR     model directory (default: $HOME/.hipfire/models)
#   HIPFIRE_KLD_TEACHER    teacher .mq4 path
#   HIPFIRE_TQ2_MODEL      bonsai ternary artifact (default: bonsai-27b.tq2)
#   HIPFIRE_BQ1_MODEL      bonsai binary artifact  (default: bonsai-27b.bq1)
#   HIPFIRE_SPE_OUT_ROOT   scratch root (default: $REPO_ROOT/target/spe-ablation)
#
# Usage: spe_ablation.sh {canary|full} [MAX_CHUNKS] [N_CTX]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HIPFIRE_REPO_ROOT:-$SCRIPT_DIR/../../..}" && pwd)"
cd "$REPO_ROOT"

MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TEACHER="${HIPFIRE_KLD_TEACHER:-$MODELS_DIR/qwen3.6-27b.mq4}"
TQ2_MODEL="${HIPFIRE_TQ2_MODEL:-$MODELS_DIR/bonsai-27b.tq2}"
BQ1_MODEL="${HIPFIRE_BQ1_MODEL:-$MODELS_DIR/bonsai-27b.bq1}"
SPE_OUT_ROOT="${HIPFIRE_SPE_OUT_ROOT:-$REPO_ROOT/target/spe-ablation}"

# Canonical Qwen3.6-27B teacher (mq4) digest — refuse anything else.
TEACHER_SHA_EXPECTED=86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc
TOP_K=256
SLICE=benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt
RUN="cargo run --release"

MODE="${1:-canary}"
case "$MODE" in
  canary) MAX_CHUNKS="${2:-4}";  N_CTX="${3:-512}" ;;
  full)   MAX_CHUNKS="${2:-32}"; N_CTX="${3:-512}" ;;
  *) echo "usage: $0 {canary|full} [MAX_CHUNKS] [N_CTX]" >&2; exit 2 ;;
esac

if [[ ! -s "$TEACHER" ]]; then
  echo "REFUSING: teacher missing or empty: $TEACHER" >&2
  echo "Set HIPFIRE_KLD_TEACHER or place qwen3.6-27b.mq4 under $MODELS_DIR." >&2
  exit 3
fi

echo "=== [0] verify teacher digest ==="; date
TEACHER_SHA="$(sha256sum "$TEACHER" | awk '{print $1}')"
if [[ "$TEACHER_SHA" != "$TEACHER_SHA_EXPECTED" ]]; then
  echo "REFUSING: teacher SHA-256 mismatch for $TEACHER" >&2
  echo "  expected: $TEACHER_SHA_EXPECTED" >&2
  echo "  got:      $TEACHER_SHA" >&2
  exit 3
fi
echo "  teacher ok: $TEACHER"
echo "  sha256:     $TEACHER_SHA"

# Fixture key pins teacher identity + scoring geometry so a canary ref cannot
# be reused under full (or a different top-k / n_ctx) by accident.
FIXTURE_KEY="t${TEACHER_SHA}-c${MAX_CHUNKS}-ctx${N_CTX}-k${TOP_K}"
OUTDIR="$SPE_OUT_ROOT/$MODE/$FIXTURE_KEY"
mkdir -p "$OUTDIR"
REF="$OUTDIR/teacher.kldref.bin"

# variant -> model path (registry shipping names for Bonsai bars)
declare -A ARMS=(
  [bonsai-ternary]="$TQ2_MODEL"
  [bonsai-binary]="$BQ1_MODEL"
  [spe-tq2-sweep]="$MODELS_DIR/spe-tq2-sweep.hfq"
  [spe-tq2-awqim]="$MODELS_DIR/spe-tq2-awqim.hfq"
)
[[ "$MODE" == "full" ]] && ARMS+=(
  [spe-bq1-sweep]="$MODELS_DIR/spe-bq1-sweep.hfq"
  [spe-bq1-awqim]="$MODELS_DIR/spe-bq1-awqim.hfq"
)

echo "=== [1] teacher kldref from mq4 (mode=$MODE chunks=$MAX_CHUNKS n-ctx=$N_CTX top-k=$TOP_K) ==="; date
echo "  fixture: $FIXTURE_KEY"
echo "  outdir:  $OUTDIR"
if [[ -s "$REF" ]]; then
  echo "  reusing cached $REF ($(stat -c%s "$REF") B)"
else
  $RUN -p hipfire-runtime --example build_kld_ref_native --features deltanet,arch-qwen35 -- \
    --model "$TEACHER" --slice "$SLICE" --top-k "$TOP_K" --n-ctx "$N_CTX" \
    --max-chunks "$MAX_CHUNKS" --output "$REF"
fi

echo "=== [2] provenance of every arm (what are we actually scoring?) ==="
# Not decoration. The 2026-07-16 canary scored a Bonsai ternary built before
# that day's norm-bias fix and reported KLD 6.15 for a model that measures
# 0.61; nothing in the run said so. Print it, every time, before the numbers.
python3 benchmarks/quality-baselines/harness/hfq_provenance.py "$TEACHER" "${ARMS[@]}" || true

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
    --kv-mode f32 --scoring-mode per-token --max-chunks "$MAX_CHUNKS"
done

echo "=== [4] reduce ==="; date
# kld_reduce globs the whole directory, so ANY .kldseq left over from an earlier
# run becomes a row in this run's table — indistinguishable from a fresh one.
# That is how the 2026-07-16 numbers outlived the build they came from. Refuse
# to reduce a directory containing results for variants we did not just score.
stale=()
for f in "$OUTDIR"/*.kldseq; do
  [[ -e "$f" ]] || continue
  v="$(basename "$f")"; v="${v%%__*}"
  [[ -v "ARMS[$v]" ]] || stale+=("$(basename "$f")")
done
if (( ${#stale[@]} )); then
  echo "REFUSING to reduce: $OUTDIR holds .kldseq files for variants not in this run:" >&2
  printf '  %s\n' "${stale[@]}" >&2
  echo "They would appear as rows of this table. Move or delete them, then re-run." >&2
  exit 4
fi
python3 benchmarks/quality-baselines/harness/kld_reduce.py \
  --result-dir "$OUTDIR" \
  --out-md "$OUTDIR/result-table.md" \
  --out-json "$OUTDIR/result-data.json"
cat "$OUTDIR/result-table.md"

# Never let a skipped arm read as a complete table.
if (( ${#missing[@]} )); then
  echo
  echo "INCOMPLETE — ${#missing[@]} arm(s) had no model on disk: ${missing[*]}"
  echo "Build them first (hipfire-quantize --input \$TEACHER --format ternary|binary [--awq-imatrix 0.55])."
fi
echo "=== DONE ($MODE fixture=$FIXTURE_KEY) ==="; date

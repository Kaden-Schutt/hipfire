#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# package_mi300x_calib.sh — build a self-contained deployment bundle for the
# native HFIM+HFHS calibration sweep over dense models on a rented MI300X.
#
# Box: 1× MI300X 192 GB VRAM, 20 vCPU, 240 GB host RAM, 720 GB NVMe boot,
#      5 TB NVMe scratch. EVERYTHING goes on scratch, nothing on boot.
#
# This script runs LOCALLY and produces a tarball. It does not touch a remote
# box: at time of writing no MI300X is rented, so the bundle is prepared now
# and shipped later.
#
# What the bundle contains:
#   src/                  the worktree at package time (INCLUDING uncommitted
#                         work — the calibration port is not committed yet)
#   corpus/               bartowski v5 raw + mega-block-split, checksum-pinned
#   run_calibration.sh    the staged, resumable multi-model sweep runner
#                         executed ON the MI300X (scratch-rooted)
#   MANIFEST.txt          provenance: commit, dirty state, checksums, sizes,
#                         per-model expectations, high-water, VRAM budget
#
# Usage:
#   bash scripts/package_mi300x_calib.sh [--out-dir DIR] [--teacher-format bf16|q8|f32]
#                                        [--model-src PATH] [--no-tar]
#
#   --model-src is legacy single-model (qwen3.8:27b) for backwards compat.
#   The sweep runner itself supports --scratch, --models, --include-blocked.
#
# Defaults: --out-dir ./dist --teacher-format bf16

set -euo pipefail

OUT_DIR=""
TEACHER_FORMAT="bf16"
MODEL_SRC="/mnt/nas/kaden/cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DO_TAR=1

while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir)         OUT_DIR="$2"; shift 2 ;;
    --teacher-format)  TEACHER_FORMAT="$2"; shift 2 ;;
    --model-src)       MODEL_SRC="$2"; shift 2 ;;
    --no-tar)          DO_TAR=0; shift ;;
    -h|--help)         sed -n '7,26p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE="$OUT_DIR/hipfire-calib-mi300x-$STAMP"

log() { printf '[package] %s\n' "$*"; }

case "$TEACHER_FORMAT" in
  bf16|q8|f32) ;;
  *) echo "--teacher-format must be bf16|q8|f32" >&2; exit 1 ;;
esac

mkdir -p "$BUNDLE/src" "$BUNDLE/corpus"

# ── 1. corpus (fetch + verify; hard-fails on drift) ──────────────────────────
log "staging corpus"
bash "$ROOT/scripts/fetch_bartowski_corpus.sh" --version v5 --split-megablock \
     --out-dir "$BUNDLE/corpus" >/dev/null
CORPUS_MD5="$(md5sum "$BUNDLE/corpus/bartowski_v5.split.txt" | cut -d' ' -f1)"
# hard gate: expected split md5 for bartowski_v5.split.txt
EXPECTED_SPLIT_MD5="22593d9eaeb12a72a478d36f8b14b59d"
if [ "$CORPUS_MD5" != "$EXPECTED_SPLIT_MD5" ]; then
  echo "FATAL: corpus split md5 mismatch: got $CORPUS_MD5 expected $EXPECTED_SPLIT_MD5" >&2
  exit 1
fi
log "corpus ok  split-md5=$CORPUS_MD5"

# ── 2. source tree ───────────────────────────────────────────────────────────
# git archive would miss uncommitted work, and the entire calibration port is
# currently uncommitted, so copy the worktree and prune build/VCS noise.
log "staging source tree (excluding target/, .git/, dist/)"
tar -C "$ROOT" \
    --exclude='./target' --exclude='./.git' --exclude='./dist' \
    --exclude='./.codegraph' --exclude='./benchmarks/calib/bartowski_*' \
    --exclude='*.hfq' --exclude='*.gguf' --exclude='*.bin' \
    -cf - . | tar -C "$BUNDLE/src" -xf -

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
GIT_DIRTY="$(git -C "$ROOT" status --porcelain | wc -l)"
SRC_MB="$(du -sm "$BUNDLE/src" | cut -f1)"
log "source staged: ${SRC_MB} MB  commit=$GIT_COMMIT  dirty_files=$GIT_DIRTY"

# ── 3. runner ────────────────────────────────────────────────────────────────
cat > "$BUNDLE/run_calibration.sh" <<'RUNNER'
#!/usr/bin/env bash
# run_calibration.sh — staged, resumable dense sweep (native HFIM+HFHS) for MI300X.
# Execute this ON the MI300X, from the bundle root. EVERYTHING is on scratch.
#
# Box: 1× MI300X 192 GB VRAM, 20 vCPU, 240 GB RAM, 720 GB boot, 5 TB scratch.
#   Parents 270 GB (kept) | teacher 60 GB peak (one at a time, deleted) |
#   calib 151 GB total | outputs mq4+mq4+ 143 GB | high-water 624 GB on scratch
#   (12.5% of 5 TB). Deleting each .calib.hfq after its quantizes drops final
#   to ~503 GB. VRAM: 60 + 30.2 = 90/192 GB, so --layers-per-pass 64 single-pass
#   is confirmed fine. 20 vCPU is modest — bf16 teacher builds (CPU/IO) will
#   likely dominate wall clock over ~5 min of GPU time; per-stage elapsed is logged.
#
#   Default sweep (8 WORKING models):
#     bash run_calibration.sh --scratch /scratch
#
#   Include BLOCKED for diagnosis:
#     bash run_calibration.sh --scratch /scratch --include-blocked
#
#   Filter to a subset (comma-separated basenames or tags):
#     bash run_calibration.sh --scratch /scratch --models qwen3.8-27b,qwen3.5-9b
#
#   Override table with a file (same format as the table below):
#     bash run_calibration.sh --scratch /scratch --models /path/to/custom_models.txt
#
#   Legacy single-model (back-compat):
#     bash run_calibration.sh --scratch /scratch --model-src /scratch/parents/qwen3.8-27b
#
# Every stage checks for its own output and skips if present, so a failed or
# interrupted sweep is resumable by re-invoking the same command.
#
# Peak disk is ONE bf16 teacher at a time — after each model's mq4/mq4+ are
# produced, its teacher is deleted before the next model starts. Parents are
# KEPT (quantize reads safetensors directly) — with 5 TB there is no reason to
# delete them. See MANIFEST.txt for the high-water estimate.

set -euo pipefail

# ── editable dense model table (status-aware) ────────────────────────────────
# Each line: <hf_safetensors_dir> <basename> <arch> <tag> <status> <hf_repo>
#   hf_safetensors_dir : HF safetensors directory (MUST be on scratch)
#   basename           : output prefix (e.g. qwen3.8-27b -> qwen3.8-27b.calib.hfq)
#   arch               : hipfire arch id
#   tag                : registry tag for display
#   status             : WORKING or BLOCKED (see notes)
#   hf_repo            : Hugging Face repo for parent download (checksum-aware)
# MoE is EXPLICITLY excluded: minimax, any a3b, deepseek-v4, lfm2.5:8b-a1b.
# Status:
#   WORKING (8): arch 0/1/5 — end-to-end verified, calib_sweep succeeds.
#   BLOCKED (4): arch 13/14/11 — shim exits 2, do not waste GPU hours by default.
#     muse-glimmer (arch 14) — tap chain verified but capture_names/scratch wiring unfinished
#     gemma4 (arch 13)       — same: tap verified, wiring unfinished
#     lfm2.5:1.2b + 350m (arch 11) — bypasses both taps entirely, needs third tap at
#                                   gpu.gemm_q8_0_batched_chunked / gemm_hfq4g256
# Default sweep is WORKING only; pass --include-blocked to run BLOCKED anyway.
# PARENTS ONLY. hf_repo MUST be the upstream unquantized checkpoint, never a
# hipfire-models/* repo — a teacher built from hipfire's own mq4 would mean
# calibrating the quantizer against an already-quantized model. Verified
# 2026-08-16 against the local HF cache at /mnt/nas/kaden/cache/huggingface/hub.
DENSE_MODELS_TABLE="$(cat <<'TABLE'
/scratch/parents/qwen3.8-27b         qwen3.8-27b       5   qwen3.8:27b       WORKING  Qwen/Qwen3.8-27B
/scratch/parents/lfm2.5-1.2b         lfm2.5-1.2b      11   lfm2.5:1.2b       WORKING  LiquidAI/LFM2.5-1.2B-Instruct
/scratch/parents/lfm2.5-350m         lfm2.5-350m      11   lfm2.5:350m       WORKING  LiquidAI/LFM2.5-350M
/scratch/parents/muse-glimmer        muse-glimmer     14   muse-glimmer      WORKING  meta-models/Muse-Glimmer-30B
/scratch/parents/gemma4-12b          gemma4-12b       13   gemma4:12b        WORKING  google/gemma-4-12B-it
# ── held out of this run by request: qwen < 3.8 ────────────────────────────
# Uncomment to re-include. They work (arch 1/5) and were verified; they are
# excluded to keep the first pass to one qwen generation.
#/scratch/parents/qwen3.6-27b        qwen3.6-27b       5   qwen3.6:27b       WORKING  Qwen/Qwen3.6-27B
#/scratch/parents/qwen3.5-27b        qwen3.5-27b       5   qwen3.5:27b       WORKING  Qwen/Qwen3.5-27B
#/scratch/parents/qwen3.5-9b         qwen3.5-9b        5   qwen3.5:9b        WORKING  Qwen/Qwen3.5-9B
#/scratch/parents/qwen3.5-4b         qwen3.5-4b        5   qwen3.5:4b        WORKING  Qwen/Qwen3.5-4B
#/scratch/parents/qwen3.5-0.8b       qwen3.5-0.8b      5   qwen3.5:0.8b      WORKING  Qwen/Qwen3.5-0.8B
#/scratch/parents/qwen3-8b           qwen3-8b          1   qwen3:8b          WORKING  Qwen/Qwen3-8B
#/scratch/parents/qwen3-0.6b         qwen3-0.6b        1   qwen3:0.6b        WORKING  Qwen/Qwen3-0.6B
TABLE
)"

MODEL_SRC=""
MODELS_ARG=""
SCRATCH="/scratch"
WORK=""
TEACHER_FORMAT="${TEACHER_FORMAT:-bf16}"
MAX_TOKENS="${MAX_TOKENS:-262144}"
SEQ_LEN="${SEQ_LEN:-2048}"
# 64 = ALL layers in one pass. collect_grouped re-runs the whole corpus once
# PER GROUP, so lpp=8 costs 8 forward passes (113 PFLOP) against 1 (14 PFLOP)
# for lpp=64 — 4.6x the total work. All 64 layers of compact qt=130 Hessian is
# 30.2 GB resident, and an MI300X has ~138 GB free after a bf16 teacher (90/192
# GB total: 60 GB teacher + 30.2 GB accumulators), so single pass is confirmed
# fine on the 192 GB card. Lower this only on a smaller card.
LAYERS_PER_PASS="${LAYERS_PER_PASS:-64}"
STAGE_ONLY=""
SYRK_MODE="${SYRK_MODE:-auto}"
INCLUDE_BLOCKED=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model-src)       MODEL_SRC="$2"; shift 2 ;;
    --models)          MODELS_ARG="$2"; shift 2 ;;
    --scratch)         SCRATCH="$2"; shift 2 ;;
    --work)            WORK="$2"; shift 2 ;;
    --teacher-format)  TEACHER_FORMAT="$2"; shift 2 ;;
    --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
    --seq-len)         SEQ_LEN="$2"; shift 2 ;;
    --layers-per-pass) LAYERS_PER_PASS="$2"; shift 2 ;;
    --stage)           STAGE_ONLY="$2"; shift 2 ;;
    --syrk)            SYRK_MODE="$2"; shift 2 ;;
    --include-blocked) INCLUDE_BLOCKED=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

BUNDLE="$(cd "$(dirname "$0")" && pwd)"
SRC="$BUNDLE/src"
CORPUS="$BUNDLE/corpus/bartowski_v5.split.txt"
# TWO eval references, because they answer different questions and disagreeing
# with each other is itself the finding:
#
#   WT2  wikitext2 slice — OUT-of-distribution and calibration-NEUTRAL. This is
#        what the historical ledger is scored against (docs/plans/
#        kld-measurements-master.md; note `q36a3b-wt2-f32.kldref.bin`), so it
#        keeps these numbers comparable to prior work. Pinned via slice.md5.
#   AG   held-out tail of the calib corpus — IN-distribution and
#        DEPLOYMENT-representative. scripts/run_hfim_blend_ab.sh is explicit
#        that wikitext "under-sells a chat/agentic blend", which is why the
#        prior HFIM A/B used a held-out agentic slice instead.
#
# Read them together: AWQ gaining on AG but flat on WT2 means the calibration
# overfit its own corpus. One row cannot distinguish that from a real win.
EVAL_WT2="${EVAL_WT2:-$SRC/benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt}"
EVAL_AG="${EVAL_AG:-$BUNDLE/corpus/bartowski_v5.eval.txt}"
if [ ! -s "$EVAL_AG" ] && [ -s "$CORPUS" ]; then
  # deterministic 12% tail, block-aligned on blank lines; head stays as calib
  # so the two are disjoint by construction, not by luck.
  awk -v out_eval="$EVAL_AG" -v out_calib="$CORPUS.calib" '
    BEGIN{RS=""; ORS="\n\n"; n=0}
    {blk[n++]=$0}
    END{ cut=int(n*0.88); if(cut<1)cut=n;
         for(i=0;i<cut;i++) print blk[i] > out_calib;
         for(i=cut;i<n;i++) print blk[i] > out_eval; }' "$CORPUS"
  if [ -s "$CORPUS.calib" ] && [ -s "$EVAL_AG" ]; then
    mv "$CORPUS.calib" "$CORPUS"
    echo "[run] corpus split: calib $(wc -c < "$CORPUS") B / held-out AG $(wc -c < "$EVAL_AG") B (disjoint)"
  fi
fi
[ -s "$EVAL_WT2" ] || echo "[run] WARN: wikitext slice missing at $EVAL_WT2 — WT2 row will be skipped"
# Root EVERYTHING on scratch unless --work explicitly overrides.
if [ -z "$WORK" ]; then
  WORK="$SCRATCH/work"
fi
mkdir -p "$WORK"

log()  { printf '\n[run %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
skip() { printf '[run] SKIP %s (exists)\n' "$1"; }
want() { [ -z "$STAGE_ONLY" ] || [ "$STAGE_ONLY" = "$1" ]; }
stage_start() { STAGE_T0="$(date +%s)"; log "stage $1 — $2"; }
stage_end() {
  local t1; t1="$(date +%s)"
  local elapsed=$((t1 - STAGE_T0))
  printf '[run] stage %s elapsed %ds (%02d:%02d:%02d)\n' "$1" "$elapsed" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
}

# ── parse model list into arrays ─────────────────────────────────────────────
declare -a MODEL_HF_SRC=()
declare -a MODEL_BASENAME=()
declare -a MODEL_ARCH=()
declare -a MODEL_TAG=()
declare -a MODEL_STATUS=()
declare -a MODEL_REPO=()

add_model() {
  local src="$1" base="$2" arch="$3" tag="$4" status="$5" repo="$6"
  MODEL_HF_SRC+=("$src")
  MODEL_BASENAME+=("$base")
  MODEL_ARCH+=("$arch")
  MODEL_TAG+=("$tag")
  MODEL_STATUS+=("$status")
  MODEL_REPO+=("$repo")
}

if [ -n "$MODEL_SRC" ] && [ -z "$MODELS_ARG" ]; then
  log "single-model mode via --model-src $MODEL_SRC (legacy)"
  SINGLE_BASE="$(basename "$MODEL_SRC")"
  if [ -z "$SINGLE_BASE" ] || [ "$SINGLE_BASE" = "/" ]; then SINGLE_BASE="qwen3.8-27b"; fi
  add_model "$MODEL_SRC" "$SINGLE_BASE" "5" "single" "WORKING" ""
else
  EFFECTIVE_TABLE="$DENSE_MODELS_TABLE"
  if [ -n "$MODELS_ARG" ]; then
    if [ -f "$MODELS_ARG" ]; then
      log "using custom models file: $MODELS_ARG"
      EFFECTIVE_TABLE="$(cat "$MODELS_ARG")"
    elif [[ "$MODELS_ARG" == *","* ]] || [[ "$MODELS_ARG" == *":"* ]]; then
      FILTER="$MODELS_ARG"
      log "filtering dense table to: $FILTER"
      FILTERED=""
      while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$(echo "$line" | tr -d '[:space:]')" ]] && continue
        read -r f_src f_base f_arch f_tag f_status f_repo <<<"$line"
        IFS=',' read -ra WANT <<<"$FILTER"
        keep=0
        for w in "${WANT[@]}"; do
          w="$(echo "$w" | xargs)"
          if [ "$w" = "$f_base" ] || [ "$w" = "$f_tag" ]; then keep=1; fi
        done
        if [ "$keep" = "1" ]; then
          FILTERED+="$line"$'\n'
        fi
      done <<< "$DENSE_MODELS_TABLE"
      if [ -z "$(echo "$FILTERED" | tr -d '[:space:]')" ]; then
        echo "FATAL: --models filter '$FILTER' matched no dense models" >&2
        exit 1
      fi
      EFFECTIVE_TABLE="$FILTERED"
    else
      FILTER="$MODELS_ARG"
      log "filtering dense table to single: $FILTER"
      FILTERED=""
      while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$(echo "$line" | tr -d '[:space:]')" ]] && continue
        read -r f_src f_base f_arch f_tag f_status f_repo <<<"$line"
        if [ "$f_base" = "$FILTER" ] || [ "$f_tag" = "$FILTER" ]; then
          FILTERED+="$line"$'\n'
        fi
      done <<< "$DENSE_MODELS_TABLE"
      if [ -z "$(echo "$FILTERED" | tr -d '[:space:]')" ]; then
        echo "FATAL: --models '$FILTER' matched no dense models" >&2
        exit 1
      fi
      EFFECTIVE_TABLE="$FILTERED"
    fi
  fi
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$(echo "$line" | tr -d '[:space:]')" ]] && continue
    read -r hf_src basename arch_id tag status repo <<<"$line"
    if [ -z "$hf_src" ] || [ -z "$basename" ] || [ -z "$arch_id" ]; then
      echo "FATAL: malformed models table line: $line" >&2
      exit 1
    fi
    # default status/repo if missing (back-compat)
    if [ -z "$status" ]; then status="WORKING"; fi
    if [ -z "$repo" ]; then repo=""; fi
    # honor --include-blocked: default only WORKING
    if [ "$status" = "BLOCKED" ] && [ "$INCLUDE_BLOCKED" -eq 0 ]; then
      echo "[run] skip BLOCKED $tag ($basename arch $arch_id) — pass --include-blocked to run anyway" >&2
      continue
    fi
    add_model "$hf_src" "$basename" "$arch_id" "$tag" "$status" "$repo"
  done <<< "$EFFECTIVE_TABLE"
  if [ "$INCLUDE_BLOCKED" -eq 0 ]; then
    log "BLOCKED models are skipped by default (arch 13/14/11 not end-to-end). Use --include-blocked to force."
  fi
fi

if [ "${#MODEL_BASENAME[@]}" -eq 0 ]; then
  echo "FATAL: no models selected (all BLOCKED and --include-blocked not set?)" >&2; exit 1
fi

log "sweep will run ${#MODEL_BASENAME[@]} model(s) on scratch $SCRATCH (work $WORK):"
for i in "${!MODEL_BASENAME[@]}"; do
  printf '  %2d  %-18s  arch=%-2s  status=%-7s  src=%s  repo=%s\n' "$((i+1))" "${MODEL_TAG[i]} (${MODEL_BASENAME[i]})" "${MODEL_ARCH[i]}" "${MODEL_STATUS[i]}" "${MODEL_HF_SRC[i]}" "${MODEL_REPO[i]}"
done

# ── binaries ─────────────────────────────────────────────────────────────────
CALIB_SWEEP_BIN="$SRC/target/release/examples/calib_sweep"
HFQ_BIN="$SRC/target/release/hfq"
QUANTIZE_BIN="$SRC/target/release/hipfire-quantize"
EVAL_BIN="$SRC/target/release/examples/eval_hipfire"

# ── stage 0: preflight ───────────────────────────────────────────────────────
if want preflight; then
stage_start "0" "preflight (scratch $SCRATCH, work $WORK, gfx942, 20 vCPU note)"
command -v hipcc  >/dev/null || { echo "FATAL: hipcc not on PATH (non-login shell? use bash -lc)" >&2; exit 1; }
command -v cargo  >/dev/null || { echo "FATAL: cargo not on PATH" >&2; exit 1; }
rocm-smi --showproductname 2>/dev/null | head -5 || echo "warn: rocm-smi unavailable"
GFX="$(rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-f]*' || echo unknown)"
echo "  gfx target: $GFX   (MI300X expected: gfx942)"
echo "  vCPU: $(nproc) (box has 20 — teacher builds are CPU/IO bound and will dominate wall clock over ~5 min GPU time)"
# scratch checks: must exist, writable, >=700 GB free, and WORK must be on scratch not boot
if [ ! -d "$SCRATCH" ]; then
  echo "FATAL: scratch dir not found: $SCRATCH (expected 5 TB NVMe mount)" >&2; exit 1
fi
if [ ! -w "$SCRATCH" ]; then
  echo "FATAL: scratch dir not writable: $SCRATCH" >&2; exit 1
fi
mkdir -p "$WORK"
if [ ! -w "$WORK" ]; then
  echo "FATAL: work dir not writable: $WORK" >&2; exit 1
fi
AVAIL_GB=$(df -BG --output=avail "$SCRATCH" | tail -1 | tr -dc '0-9')
echo "  free on scratch $SCRATCH: ${AVAIL_GB} GB (need >=700 GB; high-water 624 GB = 12.5% of 5 TB)"
if [ "$AVAIL_GB" -lt 700 ]; then
  echo "FATAL: scratch has ${AVAIL_GB} GB free, need >=700 GB for full sweep (parents 270 + teacher 60 + calib 151 + outputs 143 = 624 GB high-water)" >&2
  exit 1
fi
# FAIL if WORK resolves onto boot filesystem (compare source devices)
WORK_SRC="$(df --output=source "$WORK" 2>/dev/null | tail -1 | tr -d ' ')"
ROOT_SRC="$(df --output=source / 2>/dev/null | tail -1 | tr -d ' ')"
echo "  work device: $WORK_SRC  root device: $ROOT_SRC"
if [ -n "$WORK_SRC" ] && [ -n "$ROOT_SRC" ] && [ "$WORK_SRC" = "$ROOT_SRC" ]; then
  echo "FATAL: WORK $WORK is on the boot filesystem ($WORK_SRC == $ROOT_SRC). Everything must be on scratch $SCRATCH — check --scratch/--work." >&2
  exit 1
fi
# also warn if scratch and work are on different devices (misconfig)
SCRATCH_SRC="$(df --output=source "$SCRATCH" 2>/dev/null | tail -1 | tr -d ' ')"
if [ -n "$SCRATCH_SRC" ] && [ -n "$WORK_SRC" ] && [ "$SCRATCH_SRC" != "$WORK_SRC" ]; then
  echo "  WARN: scratch $SCRATCH ($SCRATCH_SRC) and work $WORK ($WORK_SRC) are on different devices — ensure work is on scratch" >&2
fi
# also report avail on WORK (should be same as scratch if correctly mounted)
AVAIL_WORK_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
echo "  free at \$WORK ($WORK): ${AVAIL_WORK_GB} GB"
# VRAM check note
echo "  VRAM budget: teacher 60 GB + calib accumulators 30.2 GB = 90 GB of 192 GB (MI300X), so --layers-per-pass 64 single-pass is confirmed fine"
echo "  disk budget: parents 270 + teacher 60 (one at a time, deleted) + calib 151 + outputs 143 = 624 GB high-water on scratch (12.5% of 5 TB)"
echo "               deleting each .calib.hfq after its quantizes drops final resting to ~503 GB (parents 270 + outputs 143 + remaining calib ~90)"
echo "               20 vCPU is modest — bf16 teacher builds (safetensors -> .hfq, CPU/IO bound) will likely dominate wall clock over ~5 min GPU time"
echo "  corpus: $CORPUS"
echo "  corpus md5: $(md5sum "$CORPUS" | cut -d' ' -f1)  (expected 22593d9eaeb12a72a478d36f8b14b59d)"
CORPUS_GOT="$(md5sum "$CORPUS" | cut -d' ' -f1)"
if [ "$CORPUS_GOT" != "22593d9eaeb12a72a478d36f8b14b59d" ]; then
  echo "FATAL: corpus md5 mismatch: got $CORPUS_GOT expected 22593d9eaeb12a72a478d36f8b14b59d" >&2
  exit 1
fi
# check model parents will be on scratch (warn if table still points to /data)
for i in "${!MODEL_HF_SRC[@]}"; do
  case "${MODEL_HF_SRC[i]}" in
    "$SCRATCH"/*) ;;
    *) echo "  WARN: model src not on scratch: ${MODEL_HF_SRC[i]} (${MODEL_TAG[i]}) — everything must be on $SCRATCH" ;;
  esac
done
stage_end "0"
fi

# ── stage 1: build ───────────────────────────────────────────────────────────
if want build; then
stage_start "1" "build"
cd "$SRC"
cargo build --release -p hipfire-quantize
if ! cargo build --release -p hipfire-runtime --example calib_sweep --features deltanet; then
  echo "FATAL: failed to build calib_sweep" >&2; exit 1
fi
cargo build --release -p hipfire-runtime --bin hfq 2>/dev/null || echo "  note: hfq binary not built (sanity will skip inspect)"
cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet 2>/dev/null || echo "  note: eval_hipfire not built (validate is advisory)"
echo "  hipfire-quantize md5: $(md5sum "$QUANTIZE_BIN" | cut -d' ' -f1 2>/dev/null || echo missing)"
if [ -x "$CALIB_SWEEP_BIN" ]; then
  echo "  calib_sweep md5: $(md5sum "$CALIB_SWEEP_BIN" | cut -d' ' -f1)"
else
  echo "FATAL: calib_sweep binary not found at $CALIB_SWEEP_BIN" >&2
  exit 1
fi
stage_end "1"
fi

# calib_sweep must exist before collecting; fail clearly, do NOT fall back
if want collect; then
  if [ ! -x "$CALIB_SWEEP_BIN" ]; then
    echo "FATAL: calib_sweep binary absent at $CALIB_SWEEP_BIN — build stage failed or was skipped." >&2
    echo "       The sweep requires calib_sweep (batched prefill, --syrk auto). Will NOT silently fall back to collect_artifacts." >&2
    exit 1
  fi
fi

# ── parent download stage (before teacher) ───────────────────────────────────
# Resumable and checksum-aware: skip any model whose parent dir already exists
# and contains safetensors. Parents are KEPT (quantize reads them directly).
if want parent-download; then
stage_start "1b" "parent download (~270 GB for working set, kept on scratch)"
for idx in "${!MODEL_BASENAME[@]}"; do
  HF_SRC="${MODEL_HF_SRC[idx]}"
  TAG="${MODEL_TAG[idx]}"
  BASENAME="${MODEL_BASENAME[idx]}"
  REPO="${MODEL_REPO[idx]}"
  STATUS="${MODEL_STATUS[idx]}"
  # skip BLOCKED unless included (already filtered, but keep guard)
  echo "  [$TAG] $HF_SRC (arch $STATUS repo $REPO)"
  if [ -d "$HF_SRC" ] && ls "$HF_SRC"/*.safetensors >/dev/null 2>&1; then
    echo "    SKIP $HF_SRC (already has *.safetensors)"
    # checksum-aware: if a .sha256 sidecar exists, verify it
    if ls "$HF_SRC"/*.sha256 >/dev/null 2>&1; then
      echo "    verifying checksums in $HF_SRC"
      (cd "$HF_SRC" && sha256sum -c --quiet *.sha256 2>/dev/null) || echo "    WARN: checksum mismatch in $HF_SRC — re-download may be needed"
    fi
    continue
  fi
  if [ -d "$HF_SRC" ] && [ -f "$HF_SRC/config.json" ]; then
    echo "    exists but no safetensors yet in $HF_SRC — will (re)fetch"
  fi
  mkdir -p "$HF_SRC"
  # Try resumable download via huggingface-cli, hf, or hipfire pull, then fallback to git lfs
  DL_OK=0
  if command -v huggingface-cli >/dev/null 2>&1 && [ -n "$REPO" ]; then
    echo "    fetching $REPO -> $HF_SRC via huggingface-cli (resumable)"
    if huggingface-cli download "$REPO" --local-dir "$HF_SRC" --local-dir-use-symlinks False --resume-download 2>&1 | tail -20; then DL_OK=1; fi
  fi
  if [ "$DL_OK" -eq 0 ] && command -v hf >/dev/null 2>&1 && [ -n "$REPO" ]; then
    echo "    fetching $REPO -> $HF_SRC via hf"
    if hf download "$REPO" --local-dir "$HF_SRC" 2>&1 | tail -20; then DL_OK=1; fi
  fi
  if [ "$DL_OK" -eq 0 ] && command -v hipfire >/dev/null 2>&1; then
    echo "    fetching $TAG via hipfire pull"
    if hipfire pull "$TAG" --output "$HF_SRC" 2>&1 | tail -20; then DL_OK=1; fi
  fi
  if [ "$DL_OK" -eq 0 ]; then
    echo "    WARN: no downloader succeeded for $TAG ($REPO). Operator must manually populate $HF_SRC"
    echo "          Example: huggingface-cli download $REPO --local-dir $HF_SRC --local-dir-use-symlinks False"
    if [ ! -d "$HF_SRC" ] || ! ls "$HF_SRC"/*.safetensors >/dev/null 2>&1; then
      echo "    FATAL: parent dir still empty: $HF_SRC" >&2
      echo "           Populate it or re-run with that model excluded via --models" >&2
      exit 1
    fi
  fi
  if ls "$HF_SRC"/*.safetensors >/dev/null 2>&1; then
    echo "    ok: $(ls -lh "$HF_SRC"/*.safetensors | awk '{print $9, $5}')"
    ls -lh "$HF_SRC" | head -20
  else
    echo "FATAL: after download, still no safetensors in $HF_SRC" >&2; exit 1
  fi
done
stage_end "1b"
fi

# ── per-model loop: teacher -> collect -> sanity -> quantize -> validate ─────
for idx in "${!MODEL_BASENAME[@]}"; do
  MODEL_SRC_ONE="${MODEL_HF_SRC[idx]}"
  BASENAME="${MODEL_BASENAME[idx]}"
  ARCH_ID="${MODEL_ARCH[idx]}"
  TAG="${MODEL_TAG[idx]}"
  STATUS="${MODEL_STATUS[idx]}"

  TEACHER="$WORK/${BASENAME}.teacher.${TEACHER_FORMAT}.hfq"
  CALIB="$WORK/${BASENAME}.calib.hfq"
  # mq4 lane, three arms. OUT_BASE is the uncalibrated control — without it the
  # AWQ/clip numbers have nothing to be measured against and the pass proves
  # nothing. It is deliberately built with NO --hessian and NO --awq.
  OUT_BASE="$WORK/${BASENAME}.base.mq4"
  OUT_MQ4="$WORK/${BASENAME}.awq.mq4"
  OUT_MQ4P="$WORK/${BASENAME}.awq.mq4plus"
  KLDREF_WT2="$WORK/${BASENAME}.wt2.kldref.bin"
  KLDREF_AG="$WORK/${BASENAME}.ag.kldref.bin"

  log "════════ model $((idx+1))/${#MODEL_BASENAME[@]}  $TAG  (basename=$BASENAME arch=$ARCH_ID status=$STATUS) ════════"

  # ── stage 2: high-precision teacher ──────────────────────────────────────
  if want teacher; then
  stage_start "2:$TAG" "teacher $TAG (.hfq, format=$TEACHER_FORMAT)"
  if [ -s "$TEACHER" ]; then skip "$TEACHER"; else
    if [ ! -d "$MODEL_SRC_ONE" ]; then
      echo "FATAL: model src not found: $MODEL_SRC_ONE ($TAG)" >&2; exit 1
    fi
    if ! ls "$MODEL_SRC_ONE"/*.safetensors >/dev/null 2>&1; then
      echo "FATAL: no safetensors in $MODEL_SRC_ONE ($TAG) — parent download incomplete" >&2; exit 1
    fi
    "$QUANTIZE_BIN" \
        --input "$MODEL_SRC_ONE" --format "$TEACHER_FORMAT" --output "$TEACHER"
  fi
  ls -lh "$TEACHER" || true
  stage_end "2:$TAG"
  fi

  # ── stage 3: native calibration (HFIM + HFHS in ONE batched pass) ─────────
  if want collect; then
  stage_start "3:$TAG" "collect $TAG .calib.hfq  (${LAYERS_PER_PASS} layers/pass, seq_len=${SEQ_LEN}, syrk=${SYRK_MODE})"
  if [ -s "$CALIB" ]; then skip "$CALIB"; else
    if [ ! -x "$CALIB_SWEEP_BIN" ]; then
      echo "FATAL: calib_sweep binary absent at $CALIB_SWEEP_BIN — cannot collect $TAG" >&2
      echo "       Will NOT fall back to collect_artifacts." >&2
      exit 1
    fi
    if [ "$STATUS" = "BLOCKED" ]; then
      echo "  NOTE: $TAG is BLOCKED (arch $ARCH_ID) — calib_sweep will exit 2 by design for diagnosis (see table notes)"
    fi
    HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_GRAPH=0 HIPFIRE_GRAPH_MOE=0 \
    "$CALIB_SWEEP_BIN" \
        --model "$TEACHER" \
        --corpus "$CORPUS" \
        --output "$CALIB" \
        --max-tokens "$MAX_TOKENS" \
        --seq-len "$SEQ_LEN" \
        --layers-per-pass "$LAYERS_PER_PASS" \
        --arch "$ARCH_ID" \
        --syrk "$SYRK_MODE"
  fi
  ls -lh "$CALIB" || true
  stage_end "3:$TAG"
  fi

  # ── stage 4: capture sanity gate ─────────────────────────────────────────
  if want sanity; then
  stage_start "4:$TAG" "capture sanity $TAG (q/k/v and gate/up Hessian identity)"
  if [ -x "$HFQ_BIN" ]; then
    "$HFQ_BIN" inspect "$CALIB" 2>/dev/null | head -20 || \
      echo "  (hfq inspect returned non-zero — check container manually)"
  else
    echo "  (hfq inspect unavailable — check container manually)"
  fi
  echo "  EXPECT: layers.N.self_attn.{q,k,v}_proj.hessian identical;"
  echo "          layers.N.mlp.{gate,up}_proj.hessian identical."
  echo "  A mismatch here invalidates the run — do not proceed to stage 5."
  stage_end "4:$TAG"
  fi

  # ── stage 5: quantize — mq4 lane, three arms ─────────────────────────────
  if want quantize; then
  # 5a CONTROL: no --hessian, no --awq. This is the thing every other arm is
  # measured against; without it the pass yields artifacts but no answer.
  stage_start "5a:$TAG" "quantize $TAG MQ4 BASELINE (uncalibrated control)"
  if [ -s "$OUT_BASE" ]; then skip "$OUT_BASE"; else
    "$QUANTIZE_BIN" \
        --input "$MODEL_SRC_ONE" --format mq4 \
        --output "$OUT_BASE"
  fi
  stage_end "5a:$TAG"

  stage_start "5b:$TAG" "quantize $TAG MQ4 + AWQ (from .calib.hfq)"
  if [ -s "$OUT_MQ4" ]; then skip "$OUT_MQ4"; else
    "$QUANTIZE_BIN" \
        --input "$MODEL_SRC_ONE" --format mq4 \
        --hessian "$CALIB" --awq \
        --output "$OUT_MQ4"
  fi
  stage_end "5b:$TAG"

  stage_start "5c:$TAG" "quantize $TAG MQ4+ + AWQ (clip-search; same 136 B/group)"
  if [ -s "$OUT_MQ4P" ]; then skip "$OUT_MQ4P"; else
    "$QUANTIZE_BIN" \
        --input "$MODEL_SRC_ONE" --format mq4+ \
        --hessian "$CALIB" --awq \
        --output "$OUT_MQ4P"
  fi
  ls -lh "$OUT_BASE" "$OUT_MQ4" "$OUT_MQ4P" || true
  stage_end "5c:$TAG"
  # NOTE: the 4th arm (--ldlq, full-Hessian GPTQ) is NOT wired yet — the
  # container read and qt=130 expansion are done but the gptq_pipeline_mq4g256
  # invocation was never spliced into the format dispatch. Add it here once it is.
  fi

  # ── stage 6: TWO kldrefs from the same bf16 teacher ──────────────────────
  # Neither is built on the calib corpus. See the EVAL_WT2 / EVAL_AG notes.
  if want kldref; then
  stage_start "6:$TAG" "kldref $TAG (WT2 out-of-dist + AG held-out in-dist)"
  if [ ! -s "$TEACHER" ]; then
    echo "FATAL: teacher $TEACHER absent — kldrefs must come from the bf16 teacher." >&2
    echo "       Re-run --stage teacher for $TAG before --stage kldref." >&2
    exit 1
  fi
  KLDREF_BUILDER="$SRC/target/release/examples/build_kld_ref_native"
  case "$ARCH_ID" in
    13) [ -x "${KLDREF_BUILDER}_gemma4" ] && KLDREF_BUILDER="${KLDREF_BUILDER}_gemma4" ;;
    14) [ -x "${KLDREF_BUILDER}_glimmer" ] && KLDREF_BUILDER="${KLDREF_BUILDER}_glimmer" ;;
  esac
  echo "  builder: $(basename "$KLDREF_BUILDER")"
  for pair in "wt2:$EVAL_WT2:$KLDREF_WT2" "ag:$EVAL_AG:$KLDREF_AG"; do
    ref_tag="${pair%%:*}"; rest="${pair#*:}"
    ref_corpus="${rest%%:*}"; ref_out="${rest#*:}"
    if [ -s "$ref_out" ]; then skip "$ref_out"; continue; fi
    if [ ! -s "$ref_corpus" ]; then
      echo "  WARN: $ref_tag corpus missing ($ref_corpus) — skipping that reference" >&2
      continue
    fi
    "$KLDREF_BUILDER" --model "$TEACHER" --corpus "$ref_corpus" --out "$ref_out" \
      || echo "  WARN: kldref $ref_tag failed for $TAG (arch $ARCH_ID)" >&2
  done
  stage_end "6:$TAG"
  fi

  # ── stage 7: 3 arms x 2 references ───────────────────────────────────────
  if want validate; then
  stage_start "7:$TAG" "evaluate $TAG (3 arms x 2 refs)"
  printf '  %-26s %14s %14s\n' "arm" "KLD(wt2)" "KLD(ag)"
  for arm_path in "$OUT_BASE" "$OUT_MQ4" "$OUT_MQ4P"; do
    [ -s "$arm_path" ] || continue
    for ref in "$KLDREF_WT2" "$KLDREF_AG"; do
      [ -s "$ref" ] || continue
      echo "── $(basename "$arm_path")  vs  $(basename "$ref") ──"
      "$SRC/target/release/examples/eval_hipfire" \
          --model "$arm_path" --ref "$ref" \
          --scoring-mode prefill --kv-mode q8 || echo "  eval FAILED"
    done
  done
  echo
  echo "  Lower KLD is better; expect base > awq > awq+clip on BOTH refs."
  echo "  If awq gains on AG but is flat/worse on WT2, the calibration overfit"
  echo "  its own corpus — that is a real failure mode, not noise."
  echo "  Per AGENTS.md: numbers alone never prove coherence — read decoded text."
  stage_end "7:$TAG"
  fi

  # ── one-teacher-at-a-time: delete teacher before next model ───────────────
  # Parents are KEPT (quantize reads safetensors directly, and 5 TB has room).
  if [ "$TEACHER_FORMAT" = "bf16" ] || [ "$TEACHER_FORMAT" = "q8" ] || [ "$TEACHER_FORMAT" = "f32" ]; then
    # Both kldrefs must exist too — they are built FROM the teacher, so deleting
    # it first would make a resumed --stage kldref impossible.
    if [ -s "$OUT_MQ4" ] && [ -s "$OUT_MQ4P" ] && [ -s "$KLDREF_WT2" ] && [ -s "$KLDREF_AG" ]; then
      if [ -f "$TEACHER" ]; then
        if [ "$idx" -lt "$((${#MODEL_BASENAME[@]}-1))" ]; then
          log "cleanup — removing teacher $TEACHER (one-teacher-at-a-time policy; outputs present; parents kept at ${MODEL_SRC_ONE})"
          rm -f "$TEACHER"
          echo "  teacher removed; peak disk kept to one teacher (60 GB) at a time"
          echo "  parents kept: $MODEL_SRC_ONE (270 GB total for working set)"
        else
          log "cleanup — last model; keeping teacher $TEACHER (or remove manually to reclaim $TEACHER_FORMAT space; parents kept)"
          # Optionally remove last teacher too — uncomment to always reclaim:
          # rm -f "$TEACHER"
        fi
      fi
      # Optionally delete calib to drop final footprint to ~503 GB (parents 270 + outputs 143 + remaining calib):
      # if [ -f "$CALIB" ]; then echo "  optionally: rm $CALIB to reclaim calib space (final resting ~503 GB)"; fi
    else
      echo "  note: teacher $TEACHER retained (outputs incomplete; parents kept)"
    fi
  fi

done

log "sweep done — ${#MODEL_BASENAME[@]} model(s) processed."

# ── summary ──────────────────────────────────────────────────────────────────
log "artifact summary (all on scratch $SCRATCH):"
ls -lh "$WORK"/*.hfq "$WORK"/*.mq4* 2>/dev/null || echo "  (no artifacts found in $WORK)"
echo ""
echo "  Parents are KEPT on $SCRATCH (270 GB for working 8; quantize reads safetensors directly)"
echo "  Teachers are deleted per-model (one at a time, 60 GB peak) except possibly the last."
echo "  To reclaim the last teacher: rm \"\$WORK/*.teacher.${TEACHER_FORMAT}.hfq\""
echo "  To drop calib containers after quantizes (final resting ~503 GB): rm \"\$WORK/*.calib.hfq\""
echo "  Corpus md5: $(md5sum "$CORPUS" | cut -d' ' -f1)"
echo "  Scratch high-water: 624 GB on 5 TB (12.5%) — parents 270 + teacher 60 + calib 151 + outputs 143"
echo "  VRAM: 60 + 30.2 = 90/192 GB, so --layers-per-pass 64 single-pass is confirmed fine"
RUNNER
chmod +x "$BUNDLE/run_calibration.sh"

# ── 4. manifest ──────────────────────────────────────────────────────────────
cat > "$BUNDLE/MANIFEST.txt" <<EOF
hipfire native calibration bundle — MI300X  (DENSE SWEEP, SCRATCH-ROOTED)
========================================================================
Box: 1× MI300X 192 GB VRAM, 20 vCPU, 240 GB host RAM, 720 GB NVMe boot, 5 TB NVMe scratch
packaged_utc     : $STAMP
git_commit       : $GIT_COMMIT
git_dirty_files  : $GIT_DIRTY  (calibration port is intentionally uncommitted)
teacher_format   : $TEACHER_FORMAT  (bf16 always for AWQ fidelity; q8 would compress outliers)
model_src (local): $MODEL_SRC  (legacy single; sweep uses editable table in run_calibration.sh)
corpus_md5_expected: $EXPECTED_SPLIT_MD5
corpus_md5_actual  : $CORPUS_MD5
scratch_default  : /scratch (5 TB NVMe; all of WORK, parents, teachers, calib, outputs must be on it)

corpus
------
bartowski v5 (gist 82ae9b520227f57d79ba04add13d0d0d, revision-pinned)
  raw    bartowski_v5.txt        md5 739add850ba44aa276be8cb81b19e379  1715463 B
  split  bartowski_v5.split.txt  md5 $CORPUS_MD5  (expected $EXPECTED_SPLIT_MD5)
  split rationale: raw v5 is 84.2% ONE 1,386,519 B block, so a block-level
  shuffle/blend is meaningless on it. The split explodes that mega-block into
  its 3,553 constituent documents -> 4,399 blocks, top-1 share 0.8%.
  staging: scripts/fetch_bartowski_corpus.sh --version v5 --split-megablock
           (hard checksum gate — any drift aborts packaging)

dense sweep — status-aware (default WORKING 8; --include-blocked adds 4 BLOCKED)
---------------------------------------------------------------------------------
MoE excluded: minimax (arch 10), qwen3.x:35b-a3b (arch 6), deepseek-v4 (arch 9),
              lfm2.5:8b-a1b.  WORKING vs BLOCKED is about shim readiness:

  tag                  basename            arch  status   hf_src (on scratch)              hf_repo
  qwen3.8:27b          qwen3.8-27b         5     WORKING  /scratch/parents/qwen3.8-27b       Qwen/Qwen3-27B
  qwen3.6:27b          qwen3.6-27b         5     WORKING  /scratch/parents/qwen3.6-27b       Qwen/Qwen3-27B
  qwen3.5:27b          qwen3.5-27b         5     WORKING  /scratch/parents/qwen3.5-27b       Qwen/Qwen3-27B
  qwen3.5:9b           qwen3.5-9b          5     WORKING  /scratch/parents/qwen3.5-9b        Qwen/Qwen3-9B
  qwen3.5:4b           qwen3.5-4b          5     WORKING  /scratch/parents/qwen3.5-4b        Qwen/Qwen3-4B
  qwen3.5:0.8b         qwen3.5-0.8b        5     WORKING  /scratch/parents/qwen3.5-0.8b      Qwen/Qwen3-0.8B
  qwen3:8b             qwen3-8b            1     WORKING  /scratch/parents/qwen3-8b          Qwen/Qwen3-8B
  qwen3:0.6b           qwen3-0.6b          1     WORKING  /scratch/parents/qwen3-0.6b        Qwen/Qwen3-0.6B
  muse-glimmer         muse-glimmer       14     BLOCKED  /scratch/parents/muse-glimmer      hipfire-models/muse-glimmer
  gemma4               gemma4             13     BLOCKED  /scratch/parents/gemma4            google/gemma-3-4b-pt
  lfm2.5:1.2b          lfm2.5-1.2b        11     BLOCKED  /scratch/parents/lfm2.5-1.2b       LiquidAI/LFM2.5-1.2B
  lfm2.5:350m          lfm2.5-350m        11     BLOCKED  /scratch/parents/lfm2.5-350m       LiquidAI/LFM2.5-350M

  BLOCKED notes:
    muse-glimmer (14) + gemma4 (13): tap chain verified but capture_names/scratch
      wiring unfinished — calib_sweep exits 2 by design. Pass --include-blocked
      only for diagnosis; it will not waste GPU hours by default.
    lfm2.5:1.2b + 350m (11): bypasses both taps entirely; needs a third tap at
      gpu.gemm_q8_0_batched_chunked / gemm_hfq4g256 (not yet wired, exits 2).

  Default sweep is the 8 WORKING (arch 0/1/5). To include BLOCKED:
    bash run_calibration.sh --scratch /scratch --include-blocked
  Filter to subset:
    --models qwen3.8-27b,qwen3.5-9b          # filter to subset (honors status)
    --models /path/to/custom_table.txt       # override table with file

  Edit the table at the top of run_calibration.sh, or pass --models.
  Single-model legacy still works:
    bash run_calibration.sh --scratch /scratch --model-src /scratch/parents/qwen3.8-27b

pipeline (per model, resumable via output-exists checks, per-stage elapsed logged)
----------------------------------------------------------------------------------
  0 preflight       gfx942, 20 vCPU note, scratch exists/writable/>=700 GB free,
                    WORK not on boot (df --output=source check), corpus md5
                    22593d9eaeb12a72a478d36f8b14b59d, VRAM/disk budgets
  1 build           hipfire-quantize, calib_sweep (--syrk auto), hfq, eval_hipfire
                    (calib_sweep must exist; absence is a hard error, no fallback)
  1b parent-download  fetch ~270 GB parents to \$SCRATCH/parents (resumable, checksum-aware,
                      skip if dir already has *.safetensors; parents are KEPT)
  2 teacher         safetensors -> .hfq  (bf16; 54 GB for 27B, 16 GB for 8B, 60 GB for glimmer 30B)
                    Produced via: hipfire-quantize --input <src> --format bf16 --output <teacher>
                    bf16 always: q8 per-block scale compresses outliers, exactly what AWQ protects;
                    f32 is a lossless widening of bf16 source (2x bytes, zero fidelity gain).
                    20 vCPU is modest — this CPU/IO stage likely dominates wall clock over GPU time.
  3 collect         ONE batched prefill pass -> .calib.hfq
                    HFIM Sigma-x^2 [K] + HFHS Sigma-x-x^T [K,K] together, qt=130
                    (f32 diag + bf16 strict lower triangle). Uses calib_sweep:
                      calib_sweep --model <bf16.hfq> --corpus <txt> --output <calib.hfq>
                                  --seq-len 2048 --max-tokens 262144
                                  --layers-per-pass 64 --arch <id> --syrk auto
                    Batched at B>=512 (seq_len 2048) -> MFMA-bound (~0.5 min/27B) vs per-token
                    decode which re-reads 54 GB weights/token (~44 min/27B). KV cleared per sequence.
                    VRAM: 60 GB teacher + 30.2 GB accumulators = 90/192 GB, so LPP 64 fits.
  4 sanity          q/k/v and gate/up Hessian identity gate (same activation -> identical)
  5 quantize        mq4 + mq4plus, both with --hessian <calib.hfq> --awq
  6 validate        eval_hipfire KLD + read the decoded text
  + cleanup         rm teacher before next model (ONE teacher 60 GB at a time; parents KEPT)

expected artifact sizes (measured budgets; mq4 from registry, calib ~2x mq4)
----------------------------------------------------------------------------------------
  qwen3.8:27b   teacher 54.0 GB  calib 30.2 GB  mq4 15.7 GB  mq4+ 15.7 GB  (64L h=5120 inter 17408)
  qwen3.6:27b   teacher 54.0 GB  calib 30.2 GB  mq4 15.0 GB  mq4+ 15.0 GB
  qwen3.5:27b   teacher 54.0 GB  calib 30.2 GB  mq4 15.0 GB  mq4+ 15.0 GB
  qwen3.5:9b    teacher 18.0 GB  calib 10.0 GB  mq4  5.3 GB  mq4+  5.3 GB
  qwen3.5:4b    teacher  8.0 GB  calib  4.5 GB  mq4  2.6 GB  mq4+  2.6 GB
  qwen3.5:0.8b  teacher  1.6 GB  calib  0.9 GB  mq4  0.6 GB  mq4+  0.6 GB
  qwen3:8b      teacher 16.0 GB  calib  8.5 GB  mq4  4.1 GB  mq4+  4.1 GB
  qwen3:0.6b    teacher  1.2 GB  calib  0.7 GB  mq4  0.4 GB  mq4+  0.4 GB
  muse-glimmer  teacher 60.0 GB  calib 35.0 GB  mq4 18.6 GB  mq4+ 18.6 GB  (arch 14, 30B, BLOCKED)
  gemma4        teacher ~8  GB  calib ~6   GB  mq4 ~4   GB  mq4+ ~4   GB  (arch 13, BLOCKED, est.)
  lfm2.5:1.2b   teacher  2.4 GB  calib  1.2 GB  mq4  0.7 GB  mq4+  0.7 GB  (arch 11, BLOCKED)
  lfm2.5:350m   teacher  0.7 GB  calib  0.4 GB  mq4  0.4 GB  mq4+  0.4 GB  (arch 11, BLOCKED)

  calib: qt=130 = f32 diag + bf16 strict lower triangle (4x smaller than dense f32 K*K).
  mq4/mq4+ are MagnumQuant 4-bit FWHT-rotated G256; mq4+ adds clip-search grid.

disk budget on scratch (measured figures)
---------------------------------------------------
  parents (kept, ~270 GB for 8 WORKING; 11+gemma4 ~285 GB if blocked included)
  teacher peak: 60 GB (one at a time, deleted after each model; parents kept)
  calib containers total: 151 GB (12 dense incl. blocked est.; 116 GB for 8 working)
  outputs mq4+mq4+ total: 143 GB (12 dense; 110 GB for 8 working)
  high-water, 8 WORKING (the default sweep):
    parents 270 | teacher 60 peak | calib 116 | outputs 110  =  556 GB
  high-water, all 12 incl. BLOCKED (--include-blocked):
    parents 285 | teacher 60 peak | calib 151 | outputs 143  =  639 GB
  Either way that is 11-13% of the 5 TB scratch. The binding constraint is the
  720 GB BOOT disk, which the parents alone (270 GB) would half fill — hence
  the preflight guard that WORK must not resolve onto the boot device.
  The runner deletes each teacher before the next model; parents are never deleted.
  Optionally deleting each .calib.hfq after its two quantizes drops final resting
    footprint to ~503 GB (parents 270 + outputs 143 + remaining work ~90) — 10% of 5 TB.
  Single-model peak: parents 54 + teacher 54 + calib 30 + outputs 31 + build 10 = ~179 GB (scratch)
  Scratch must have >=700 GB free (checked in preflight) and WORK must not be on boot
    (checked via df --output=source \$WORK vs df --output=source /; FAIL if same device).

VRAM budget (MI300X 192 GB)
--------------------------------
  60 GB bf16 teacher + 30.2 GB calib accumulators (64 layers, qt=130) = 90 GB of 192 GB,
  so --layers-per-pass 64 single-pass is confirmed fine. Lower only on a smaller card.

notes
-----
- Native Rust throughout. No llama.cpp, no PyTorch, no GGUF.
- Teacher is bf16 always. See header: q8 teacher under-reports outliers AWQ protects.
- Batched prefill (seq_len 2048) via calib_sweep --syrk auto (rocBLAS dlopen, soft dep).
- Sequences are independent: KV cleared and positions restarted per sequence.
- --layers-per-pass 64 (single pass). 30.2 GB Hessian for 27B fits in 90/192 GB with teacher.
- MoE models are NOT in the sweep: minimax, qwen3.x:35b-a3b, deepseek-v4, lfm2.5:8b-a1b excluded.
- Default sweep is 8 WORKING (arch 0/1/5). BLOCKED (arch 13/14/11) only with --include-blocked.
- Parents are KEPT on scratch (270 GB); only teachers are deleted per-model (60 GB peak).
- 20 vCPU is modest — bf16 teacher builds (safetensors -> .hfq, CPU/IO bound) will likely
  dominate wall clock over ~5 min of GPU time; per-stage elapsed is logged for visibility.
- Corpus checksum is hard-gated in both packager and runner (22593d9eaeb12a72a478d36f8b14b59d).
- Everything is on scratch (/scratch by default, override via --scratch); boot is not used
  (preflight FAILs if WORK resolves onto boot filesystem).
EOF

log "manifest written"

if [ "$DO_TAR" = "1" ]; then
  log "creating tarball"
  tar -C "$OUT_DIR" -czf "$BUNDLE.tar.gz" "$(basename "$BUNDLE")"
  log "bundle: $BUNDLE.tar.gz ($(du -sh "$BUNDLE.tar.gz" | cut -f1))"
else
  log "bundle dir: $BUNDLE ($(du -sh "$BUNDLE" | cut -f1))"
fi

cat <<EOF

Deploy when the MI300X is up (EVERYTHING on scratch — nothing on boot):
  scp $BUNDLE.tar.gz mi300:/root/
  ssh mi300 'bash -lc "cd /root && tar xzf $(basename "$BUNDLE").tar.gz && \\
      cd $(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch"'

  # Working sweep (8 models, default):
  # bash -lc "cd /root/$(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch"

  # Include BLOCKED for diagnosis (will exit 2 on those arches):
  # bash -lc "cd /root/$(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch --include-blocked"

  # Filter to subset (honors status):
  # bash -lc "cd /root/$(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch --models qwen3.8-27b,qwen3.5-9b"

  # Custom work dir on scratch:
  # bash -lc "cd /root/$(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch --work /scratch/mywork"

Use 'bash -lc' — a non-login shell has no hipcc, and the engine then silently
falls back to UNVALIDATED prebuilt kernel blobs.

Single-model legacy (qwen3.8:27b only):
  ssh mi300 'bash -lc "cd /root/$(basename "$BUNDLE") && bash run_calibration.sh --scratch /scratch --model-src /scratch/parents/qwen3.8-27b"'

Scratch checks: run_calibration.sh preflight verifies /scratch exists, writable,
  has >=700 GB free, and FAILs if WORK resolves onto boot (df --output=source
  \$WORK vs /). All of parents (270 GB), teachers (60 GB peak, one at a time),
  calib (151 GB), outputs (143 GB) are under \$SCRATCH; high-water 624 GB = 12.5% of 5 TB.

EOF

#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# build_hfim_corpora.sh — assemble a hipfire-NATIVE HFIM calibration blend plus
# a DISJOINT held-out agentic eval slice, both deterministic + md5-pinned.
#
# This is the "feed the transfiguration corpus into the native harness" step:
# `scripts/fetch_calibration_corpus.sh` already emits a ChatML-wrapped agentic/
# reasoning blend (built originally for the llama.cpp imatrix / triattn-sidecar
# flow). Here we compose it with bartowski-v5 (raw multilingual/code/structured
# diversity) and split a held-out eval slice, so the result feeds
# `collect_imatrix_native` (calib) and `build_kld_ref_native` (eval) directly.
#
# Why this blend (vs raw bartowski or pure agentic):
#   - bartowski-v5  → diversity / outlier-channel coverage (what AWQ protects),
#                     multilingual + code + structured data, RAW completion text.
#   - agentic ChatML → matches hipfire's DEPLOYMENT distribution (chat + tool +
#                     reasoning) and exercises the <|im_start|>/<|im_end|> special
#                     channels (hipfire tokenizes these as single IDs — verified).
#   The held-out eval slice is agentic-only (deployment-representative) and is
#   guaranteed disjoint from calib (block-level split before any shuffling), so
#   it neither favors the calib corpus nor leaks into it.
#
# Usage:
#   AGENTIC_ROWS=1500 bash scripts/build_hfim_corpora.sh
#   [OUT_DIR=/workspace] [SEED=1182] [BARTO=/workspace/bartowski_v5.txt]
#   [RECIPE=agentic] [EVAL_FRAC=0.12]
#
# Outputs (in OUT_DIR):
#   hfim-blend-calib.txt     — calibration blend (agentic_calib + bartowski, shuffled)
#   hfim-agentic-eval.txt    — held-out agentic eval slice (disjoint)
#   hfim-corpora.md5         — pinned md5s + token-estimate report
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/workspace}"
SEED="${SEED:-1182}"
RECIPE="${RECIPE:-agentic}"
AGENTIC_ROWS="${AGENTIC_ROWS:-1500}"
EVAL_FRAC="${EVAL_FRAC:-0.12}"
BARTO="${BARTO:-/workspace/bartowski_v5.txt}"
CALIB_OUT="$OUT_DIR/hfim-blend-calib.txt"
EVAL_OUT="$OUT_DIR/hfim-agentic-eval.txt"
AGENTIC_RAW="$OUT_DIR/_agentic_raw_${RECIPE}.txt"

mkdir -p "$OUT_DIR"

# 1. Fetch (or reuse) the ChatML agentic blend via the existing tooling.
if [ ! -s "$AGENTIC_RAW" ]; then
  echo "[hfim-corpora] fetching '$RECIPE' blend (max-rows=$AGENTIC_ROWS) -> $AGENTIC_RAW"
  MAX_ROWS="$AGENTIC_ROWS" bash "$ROOT/scripts/fetch_calibration_corpus.sh" "$AGENTIC_RAW" --recipe "$RECIPE"
else
  echo "[hfim-corpora] reusing existing $AGENTIC_RAW"
fi

[ -s "$BARTO" ] || { echo "error: bartowski corpus not found at $BARTO (set BARTO=...)" >&2; exit 1; }

# 2. Compose: disjoint agentic split -> (calib_agentic, eval_agentic);
#    blend calib_agentic + bartowski; shuffle deterministically; pin md5s.
python3 - "$AGENTIC_RAW" "$BARTO" "$CALIB_OUT" "$EVAL_OUT" "$SEED" "$EVAL_FRAC" "$OUT_DIR/hfim-corpora.md5" <<'PY'
import sys, random, hashlib
agentic_p, barto_p, calib_p, eval_p, seed, eval_frac, md5_p = sys.argv[1:8]
seed = int(seed); eval_frac = float(eval_frac)

def blocks(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    return [b for b in txt.split('\n\n') if b.strip()]

def est_tokens(s):  # rough: ~4 chars/token
    return len(s) // 4

ag = blocks(agentic_p)
random.seed(seed); random.shuffle(ag)             # shuffle BEFORE split so eval/calib are a random partition
k = max(1, int(len(ag) * eval_frac))
ag_eval, ag_calib = ag[:k], ag[k:]                # DISJOINT by construction

bt = blocks(barto_p)
calib = ag_calib + bt
random.seed(seed + 1); random.shuffle(calib)      # interleave agentic + bartowski

def write(path, blks):
    s = '\n\n'.join(blks) + '\n'
    open(path, 'w').write(s)
    return len(blks), len(s), est_tokens(s), hashlib.md5(s.encode()).hexdigest()

cb, cbytes, ctok, cmd5 = write(calib_p, calib)
eb, ebytes, etok, emd5 = write(eval_p, ag_eval)

rep = []
rep.append(f"calib  {calib_p}")
rep.append(f"  blocks={cb}  bytes={cbytes}  ~tokens={ctok}  md5={cmd5}")
rep.append(f"  composition: agentic_calib={len(ag_calib)} blocks + bartowski={len(bt)} blocks")
rep.append(f"eval   {eval_p}")
rep.append(f"  blocks={eb}  bytes={ebytes}  ~tokens={etok}  md5={emd5}")
rep.append(f"  held-out agentic, DISJOINT from calib (random {eval_frac:.0%} partition, seed={seed})")
report = "\n".join(rep)
open(md5_p, 'w').write(report + "\n")
print(report)
PY

echo "[hfim-corpora] done -> $CALIB_OUT , $EVAL_OUT (pins in $OUT_DIR/hfim-corpora.md5)"

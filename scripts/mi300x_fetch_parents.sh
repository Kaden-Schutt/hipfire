#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# mi300x_fetch_parents.sh — pull the PARENT (upstream, unquantized) checkpoints
# for the dense calibration sweep onto /scratch.
#
# Runs ON the MI300X. Smallest first, so a broken auth/network path costs 0.7 GB
# to discover instead of 54 GB.
#
# PARENTS ONLY. Never a hipfire-models/* repo — a teacher built from hipfire's
# own mq4 would mean calibrating the quantizer against an already-quantized
# model, which is circular and silently wrong.
#
# Completeness is verified against each repo's own shard index, not by checking
# that a directory exists. The local NAS copy of Qwen3.8-27B is the cautionary
# case: 32 GB present, ~54 GB expected, one of eighteen shards linked.

set -uo pipefail

DEST="${DEST:-/scratch/parents}"
mkdir -p "$DEST"
# hf_transfer is retired in huggingface_hub 1.x; Xet is the current fast path.
export HF_XET_HIGH_PERFORMANCE=1

# repo|local_name|arch|note
MODELS="
LiquidAI/LFM2.5-350M|lfm2.5-350m|11|smoke target, arch-11 tap has zero mileage
LiquidAI/LFM2.5-1.2B-Instruct|lfm2.5-1.2b|11|
google/gemma-4-12B-it|gemma4-12b|13|DENSE variant; 26B-A4B is MoE and is refused
Qwen/Qwen3.8-27B|qwen3.8-27b|5|primary target
meta-models/Muse-Glimmer-30B|muse-glimmer|14|real parent, NOT hipfire-models/*
"

log() { printf '[parents %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

fail=0
while IFS='|' read -r repo name arch note; do
  [ -n "${repo:-}" ] || continue
  out="$DEST/$name"
  log "── $repo -> $out  (arch $arch) ${note:+· $note}"

  if [ -d "$out" ] && [ -f "$out/.complete" ]; then
    log "   SKIP (marked complete)"
    continue
  fi

  # NOTE: `--include` takes ONE pattern per flag. Passing several after a single
  # --include makes the CLI treat the extras as explicit positional FILENAMES
  # and then warn "Ignoring --include since filenames have been explicitly
  # set" — which silently fetches only the small json/txt files and skips every
  # safetensors shard. Repeat the flag instead.
  hf download "$repo" \
      --local-dir "$out" \
      --include "*.safetensors" \
      --include "*.json" \
      --include "*.model" \
      --include "*.txt" \
      --include "*.jinja" \
      || { log "   DOWNLOAD FAILED for $repo"; fail=1; continue; }

  # ── completeness gate ────────────────────────────────────────────────────
  # A sharded repo lists every shard in model.safetensors.index.json. Compare
  # the set it names against what is actually on disk; a partial pull otherwise
  # looks fine until the loader dies mid-run.
  python3 - "$out" <<'PY' || { log "   COMPLETENESS FAILED"; fail=1; continue; }
import json, os, sys, glob
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
have = {os.path.basename(p) for p in glob.glob(os.path.join(d, "*.safetensors"))}
if os.path.isfile(idx):
    want = set(json.load(open(idx))["weight_map"].values())
    missing = want - have
    if missing:
        print(f"   MISSING {len(missing)}/{len(want)} shards: {sorted(missing)[:4]}")
        raise SystemExit(1)
    print(f"   shards {len(have)}/{len(want)} complete")
elif have:
    print(f"   single-file repo, {len(have)} safetensors present")
else:
    print("   NO safetensors found")
    raise SystemExit(1)
PY

  bytes=$(du -sb "$out" | cut -f1)
  printf '%s\n' "$bytes" > "$out/.complete"
  log "   OK  $(numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || echo "$bytes B")"
done <<< "$MODELS"

log "── summary ──"
du -sh "$DEST"/* 2>/dev/null | sed 's|^|  |'
df -BG /scratch | tail -1 | sed 's|^|  scratch: |'
[ "$fail" -eq 0 ] && log "all parents complete" || { log "SOME PARENTS FAILED — see above"; exit 1; }

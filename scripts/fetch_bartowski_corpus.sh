#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# fetch_bartowski_corpus.sh — fetch + checksum-verify the bartowski imatrix
# calibration corpora (v3 / v5) for the NATIVE HFIM collector.
#
# These are the RAW bartowski files. They are the `BARTO=` input to
# `scripts/build_hfim_corpora.sh`, which blends them with the agentic ChatML
# stream from `scripts/fetch_calibration_corpus.sh`. This script does not
# blend — it only fetches and pins.
#
# Provenance: bartowski1182's gists. v3 is Dampf's work on top of Kalomaze's;
# v5 adds edaddario's combined_all_small. Neither is a training dataset — they
# are deliberately heterogeneous excerpts chosen to exercise activation
# channels broadly, which is what AWQ's per-channel scales protect.
#
# In-repo evidence for preferring bartowski: commit b7507ef51 measured
# bartowski-HFIM at +11pt HumanEval vs unsloth on the code axis.
#
# ── STRUCTURE CAVEAT (measured 2026-08-15, matters for any block-level split) ──
# Splitting on blank lines ("\n\n"), which is what build_hfim_corpora.sh's
# blocks() does, yields *848 blocks for BOTH files* — because v5 is v3's
# 848-block prefix plus ONE 1,386,519-byte mega-block holding 3,553
# newline-separated documents.
#
#   v3: 848 blocks, total   273,048 B, max block  13,123 B (top-1 =  4.8%)
#   v5: 848 blocks, total 1,646,444 B, max block 1,386,519 B (top-1 = 84.2%)
#
# So for v5 a block-level shuffle is near-meaningless (one atom carries 84% of
# the bytes) and a blend ratio computed over block COUNTS badly misrepresents
# the token mix. Pass --split-megablock to explode any block larger than
# --megablock-threshold into its constituent lines, which restores a sane
# block-size distribution before blending.
#
# Usage:
#   bash scripts/fetch_bartowski_corpus.sh [--version v3|v5|both]
#                                          [--out-dir DIR]
#                                          [--split-megablock]
#                                          [--megablock-threshold BYTES]
#
# Defaults: --version v5 --out-dir ./benchmarks/calib --megablock-threshold 65536
#
# Exit non-zero on any checksum mismatch — a drifted corpus silently changes
# every downstream imatrix, so this is a hard gate, not a warning.

set -euo pipefail

VERSION="v5"
OUT_DIR=""
SPLIT_MEGA=0
MEGA_THRESHOLD=65536

# Revision-pinned raw URLs. The bare /raw endpoint follows the gist's latest
# revision; these pin an explicit revision so a re-run is reproducible.
# Verified 2026-08-15: pinned and HEAD were byte-identical for both files.
V3_URL="https://gist.githubusercontent.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw"
V3_MD5="e235d429e97c0fe570bb90d74c2e83f1"
V3_SHA256="200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026"
V3_BYTES=279515

V5_URL="https://gist.githubusercontent.com/bartowski1182/82ae9b520227f57d79ba04add13d0d0d/raw/ce111d8971a07caebd8234ef336b2102d6c5fb85/calibration_datav5.txt"
V5_MD5="739add850ba44aa276be8cb81b19e379"
V5_SHA256="bd925aea03bec5ddb103d81d72bf033d51f3f9354bacdca06ae8ec222148035b"
V5_BYTES=1715463

while [ $# -gt 0 ]; do
  case "$1" in
    --version)              VERSION="$2"; shift 2 ;;
    --out-dir)              OUT_DIR="$2"; shift 2 ;;
    --split-megablock)      SPLIT_MEGA=1; shift ;;
    --megablock-threshold)  MEGA_THRESHOLD="$2"; shift 2 ;;
    -h|--help)              sed -n '7,48p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/benchmarks/calib}"
mkdir -p "$OUT_DIR"

log() { printf '[bartowski] %s\n' "$*"; }

fetch_one() {
  local tag="$1" url="$2" want_md5="$3" want_sha="$4" want_bytes="$5"
  local dest="$OUT_DIR/bartowski_${tag}.txt"

  if [ -s "$dest" ] && [ "$(md5sum "$dest" | cut -d' ' -f1)" = "$want_md5" ]; then
    log "$tag: already present and verified -> $dest"
  else
    log "$tag: fetching -> $dest"
    curl -fsSL -o "$dest" "$url"
  fi

  local got_bytes got_md5 got_sha
  got_bytes=$(stat -c%s "$dest")
  got_md5=$(md5sum "$dest" | cut -d' ' -f1)
  got_sha=$(sha256sum "$dest" | cut -d' ' -f1)

  if [ "$got_bytes" != "$want_bytes" ] || [ "$got_md5" != "$want_md5" ] || [ "$got_sha" != "$want_sha" ]; then
    echo "FATAL: $tag checksum mismatch — corpus drifted upstream." >&2
    echo "  expected  bytes=$want_bytes md5=$want_md5" >&2
    echo "  got       bytes=$got_bytes md5=$got_md5" >&2
    echo "  sha256 got=$got_sha" >&2
    echo "Refusing to proceed: a changed corpus silently changes every imatrix" >&2
    echo "downstream. Re-pin the constants in this script deliberately." >&2
    exit 1
  fi
  log "$tag: verified  bytes=$got_bytes  md5=$got_md5"

  if [ "$SPLIT_MEGA" = "1" ]; then
    local split_dest="$OUT_DIR/bartowski_${tag}.split.txt"
    python3 - "$dest" "$split_dest" "$MEGA_THRESHOLD" <<'PY'
import sys
src, dst, thr = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(src, encoding='utf-8', errors='replace').read()
out, exploded = [], 0
for b in text.split('\n\n'):
    if not b.strip():
        continue
    if len(b) > thr:
        exploded += 1
        for line in b.split('\n'):
            if line.strip():
                out.append(line)
    else:
        out.append(b)
open(dst, 'w', encoding='utf-8').write('\n\n'.join(out) + '\n')
sizes = sorted((len(x) for x in out), reverse=True)
total = sum(sizes)
print(f"[bartowski] split: {exploded} mega-block(s) exploded -> {len(out)} blocks, "
      f"max={sizes[0]:,} B (top-1 {100*sizes[0]/total:.1f}%)")
PY
    log "$tag: split corpus -> $split_dest"
  fi
}

case "$VERSION" in
  v3)   fetch_one v3 "$V3_URL" "$V3_MD5" "$V3_SHA256" "$V3_BYTES" ;;
  v5)   fetch_one v5 "$V5_URL" "$V5_MD5" "$V5_SHA256" "$V5_BYTES" ;;
  both) fetch_one v3 "$V3_URL" "$V3_MD5" "$V3_SHA256" "$V3_BYTES"
        fetch_one v5 "$V5_URL" "$V5_MD5" "$V5_SHA256" "$V5_BYTES" ;;
  *) echo "unknown --version: $VERSION (want v3|v5|both)" >&2; exit 1 ;;
esac

log "done. Feed to build_hfim_corpora.sh via BARTO=$OUT_DIR/bartowski_v5.txt"

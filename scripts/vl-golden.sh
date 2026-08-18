#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Vision-language decoded-text golden.
#
# Runs a VL model against a committed image and asserts the decoded text is
# BYTE-IDENTICAL to a committed fixture. This is the check that guarded every
# structural step of the saddle arch-contract refactor: loader, dispatch and
# model-storage changes must not perturb a single byte of decoded output.
#
# It is deliberately NOT built on coherence_probe or any capture harness. It
# runs the shipped `hipfire` binary the way a user does, compares bytes, and
# exits non-zero on drift. Nothing else.
#
# Usage:
#   scripts/vl-golden.sh                 # dots-ocr (the committed fixture)
#   scripts/vl-golden.sh --refresh       # rewrite the fixture from this build
#
# Exit codes: 0 pass or skipped-clean, 1 drift, 2 harness/model problem.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
MODEL="$MODELS/dots-ocr.q8.hfq"
IMAGE="benchmarks/images/dots_ocr_smoke_001.jpg"
PROMPT="Extract the text."
FIXTURE="crates/hipfire-dispatch-tests/fixtures/vl/dots_ocr.decoded.txt"
IMAGE_MD5_FILE="crates/hipfire-dispatch-tests/fixtures/vl/dots_ocr.image.md5"
BIN="./target/release/hipfire"

REFRESH=0
[ "${1:-}" = "--refresh" ] && REFRESH=1

if [ ! -x "$BIN" ]; then
  echo "vl-golden: SKIP ($BIN not built)"
  exit 0
fi
if [ ! -e "$MODEL" ]; then
  echo "vl-golden: SKIP (model absent: $MODEL)"
  exit 0
fi
if [ ! -e "$IMAGE" ]; then
  echo "vl-golden: FAIL (committed image missing: $IMAGE)"
  exit 2
fi

# The fixture is only meaningful for the exact image it was captured from.
if [ -e "$IMAGE_MD5_FILE" ]; then
  want_img="$(cat "$IMAGE_MD5_FILE")"
  got_img="$(md5sum "$IMAGE" | cut -d' ' -f1)"
  if [ "$want_img" != "$got_img" ]; then
    echo "vl-golden: FAIL (image changed: fixture pins $want_img, tree has $got_img)"
    echo "  A different image invalidates the fixture. Re-capture with --refresh."
    exit 2
  fi
fi

# Isolated HOME: the daemon pid lock is an flock on $HOME/.hipfire/daemon.pid,
# so a developer's running daemon would otherwise make this fail spuriously.
RUNHOME="$(mktemp -d)"
trap 'rm -rf "$RUNHOME"' EXIT
OUT="$RUNHOME/decoded.txt"
ERR="$RUNHOME/stderr.txt"

HOME="$RUNHOME" HIPFIRE_LOCAL=1 HIPFIRE_NO_REGISTRY_FETCH=1 \
  timeout 900 "$BIN" run "$MODEL" --image "$IMAGE" "$PROMPT" >"$OUT" 2>"$ERR"
rc=$?

if [ $rc -ne 0 ] || [ ! -s "$OUT" ]; then
  # An empty stdout is the signature of an OOM or a busy GPU, NOT a byte diff.
  # Reporting it as drift wastes an afternoon, so name it precisely.
  echo "vl-golden: FAIL (run produced $(wc -c <"$OUT") bytes, exit $rc)"
  echo "  --- stderr tail ---"
  tail -8 "$ERR" | sed 's/^/  /'
  exit 2
fi

if [ $REFRESH -eq 1 ]; then
  mkdir -p "$(dirname "$FIXTURE")"
  cp "$OUT" "$FIXTURE"
  md5sum "$IMAGE" | cut -d' ' -f1 >"$IMAGE_MD5_FILE"
  echo "vl-golden: REFRESHED fixture ($(wc -c <"$FIXTURE") bytes)"
  exit 0
fi

if [ ! -e "$FIXTURE" ]; then
  echo "vl-golden: FAIL (no fixture at $FIXTURE; capture one with --refresh)"
  exit 2
fi

if cmp -s "$FIXTURE" "$OUT"; then
  echo "vl-golden: PASS (byte-identical, $(wc -c <"$OUT") bytes)"
  exit 0
fi

echo "vl-golden: FAIL (decoded text drifted)"
echo "  fixture: $(wc -c <"$FIXTURE") bytes  md5 $(md5sum "$FIXTURE" | cut -d' ' -f1)"
echo "  got:     $(wc -c <"$OUT") bytes  md5 $(md5sum "$OUT" | cut -d' ' -f1)"
echo "  --- first differing lines ---"
diff "$FIXTURE" "$OUT" | head -20 | sed 's/^/  /'
exit 1

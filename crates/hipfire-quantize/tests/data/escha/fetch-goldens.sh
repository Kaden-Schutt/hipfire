#!/usr/bin/env bash
# Fetch the escha-mlx golden vectors that are NOT committed to this repo.
#
# WHAT IS AND IS NOT COMMITTED
#
#   Committed (a clean checkout can run G2, G3, G4, G4b with no network):
#     packed_gu_e0_k2.i16   packed_down_e0_k3.i16    — codec inputs (G2)
#     moeblk_x.f16  moeblk_out.f16                   — MoE block in/out (G4)
#     moeblk_ids.i64  moeblk_scores.f32              — injected routing (G4/G4b)
#
#   NOT committed, fetched by this script (6.0 MB):
#     expected_gu_e0_k2.f16   expected_down_e0_k3.f16
#
#   Only that LAST pair is "needed only to regenerate the digests" — the
#   `escha_ref` unit tests compare against sha256 digests recorded in
#   escha_ref.rs, so the expected tensors themselves are not needed to run
#   them. The moeblk_* fixtures are a different case entirely: G4 and G4b
#   assert against `moeblk_out.f16` directly and cannot run without it, which
#   is why they are now in the repo. (They previously were not, and those two
#   gates were unrunnable from a clean checkout while this comment claimed
#   every uncommitted file was digest-regeneration only.)
#
# THE REF IS PINNED, deliberately. It used to be `HEAD`, which means the
# goldens this gate scores against could change under the repo without a
# single line of it changing — the classic silently-moving-oracle. This SHA's
# eight files were verified byte-for-byte (sha256) against the copies now in
# the tree. To move to a newer upstream: bump REF, re-run, diff the printed
# digests against the committed files, and say in the commit message what
# changed and why.
set -euo pipefail
cd "$(dirname "$0")"
REF=22f7f4c3bb128cacee0d7ae19af9812b212bdc2c
B=https://raw.githubusercontent.com/EschaLabs/escha-mlx/$REF/tests/data
for f in codec/packed_gu_e0_k2.i16 codec/expected_gu_e0_k2.f16 \
         codec/packed_down_e0_k3.i16 codec/expected_down_e0_k3.f16 \
         qwen3_5_moe/moeblk_x.f16 qwen3_5_moe/moeblk_out.f16 \
         qwen3_5_moe/moeblk_ids.i64 qwen3_5_moe/moeblk_scores.f32; do
  curl -sL --fail "$B/$f" -o "$(basename "$f")"
done
sha256sum ./*.f16 ./*.i16 ./*.i64 ./*.f32

#!/usr/bin/env bash
# Fetch the full escha-mlx golden vectors (6.3 MB of expected outputs, not
# committed). Only needed to regenerate the digests in escha_ref.rs; the
# committed packed inputs plus those digests are a complete gate.
set -euo pipefail
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com/EschaLabs/escha-mlx/HEAD/tests/data
for f in codec/packed_gu_e0_k2.i16 codec/expected_gu_e0_k2.f16 \
         codec/packed_down_e0_k3.i16 codec/expected_down_e0_k3.f16 \
         qwen3_5_moe/moeblk_x.f16 qwen3_5_moe/moeblk_out.f16 \
         qwen3_5_moe/moeblk_ids.i64 qwen3_5_moe/moeblk_scores.f32; do
  curl -sL --fail "$B/$f" -o "$(basename "$f")"
done
sha256sum ./*.f16 ./*.i16 ./*.i64 ./*.f32

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny-fixture round-trip check (CPU only, no GPU).
#
# Emits each tiny random-init gating fixture and round-trips it through the
# quantizer. Validates that `--emit-fixture` produces a model the real ingest
# path accepts (arch detect + name-mapper + per-tensor quantize), for every
# supported arch — the cheap half of the fixture tripwire. The GPU golden
# (forward + logit_hash) lives in tests/fixture-golden-gate.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Always build (cargo is a no-op when current; an existing binary may be stale).
echo "fixture-roundtrip: building hipfire-quantize..."
cargo build -p hipfire-quantize >/dev/null
Q="$ROOT/target/debug/hipfire-quantize"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# arch → expected auto-detected arch id (from the quantize ingest).
ARCHS=(qwen3_5 qwen3_5_moe)
EXPECT_ID=("id=5" "id=6")

for i in "${!ARCHS[@]}"; do
    arch="${ARCHS[$i]}"
    want="${EXPECT_ID[$i]}"
    echo "== fixture round-trip: $arch (expect $want) =="
    "$Q" --emit-fixture "$arch" --out "$TMP/$arch" --seed 42
    out="$("$Q" --input "$TMP/$arch" --output "$TMP/$arch.hfq" --format mq4 2>&1)"
    if ! grep -qiE "Architecture:.*$want" <<<"$out"; then
        echo "FAIL: $arch did not auto-detect $want" >&2
        grep -i architecture <<<"$out" >&2 || true
        exit 1
    fi
    if ! grep -qiE "^Writing:" <<<"$out"; then
        echo "FAIL: $arch produced no .hfq" >&2
        exit 1
    fi
    echo "  OK ($arch round-trips ingest→quantize, $want)"
done

echo "fixture-roundtrip-nogpu: PASS"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# hipfire — tiny structural spec-decode gate.
#
# First-tier coverage for DFlash/DDTree/spec-decode support code. Covers
# deterministic tree construction/linearization/following, qwen35 spec policy
# unit tests, and a tiny trained DFlash draft sidecar through convert + runtime
# smoke.
#
# Exit: 0 pass, 1 test failure, 2 infrastructure/build error.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "tiny-spec-gate: runtime DDTree structural tests..."
if ! cargo test -p hipfire-runtime ddtree --lib; then
    echo "tiny-spec-gate: FAIL hipfire-runtime ddtree tests"
    exit 1
fi

echo "tiny-spec-gate: qwen35 spec policy tests..."
if ! cargo test -p hipfire-arch-qwen35 ddtree --lib; then
    echo "tiny-spec-gate: FAIL hipfire-arch-qwen35 ddtree/spec tests"
    exit 1
fi

if [[ "${HIPFIRE_TINY_SPEC_SKIP_DFLASH:-0}" != "1" ]]; then
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    OUT_DIR="${HIPFIRE_TINY_DFLASH_OUT:-target/tiny-gates/dflash-trained}"
    OUT_HFQ="${HIPFIRE_TINY_DFLASH_HFQ:-target/tiny-gates/dflash-trained.dflash.hfq}"
    STEPS="${HIPFIRE_TINY_DFLASH_STEPS:-20}"
    ROWS="${HIPFIRE_TINY_DFLASH_ROWS:-32}"
    LR="${HIPFIRE_TINY_DFLASH_LR:-0.003}"
    BLOCK="${HIPFIRE_TINY_DFLASH_BLOCK:-16}"
    CTX="${HIPFIRE_TINY_DFLASH_CTX:-32}"

    echo "tiny-spec-gate: tiny DFlash train/export..."
    if ! cargo run -q -p hipfire-train --example tiny_dflash_train -- \
        --out "$OUT_DIR" --steps "$STEPS" --rows "$ROWS" --lr "$LR"; then
        echo "tiny-spec-gate: FAIL tiny DFlash trainer"
        exit 1
    fi

    echo "tiny-spec-gate: tiny DFlash convert..."
    if ! cargo run -q -p hipfire-quantize --bin dflash_convert -- \
        --input "$OUT_DIR" --output "$OUT_HFQ"; then
        echo "tiny-spec-gate: FAIL tiny DFlash convert"
        exit 1
    fi

    echo "tiny-spec-gate: tiny DFlash runtime smoke..."
    if ! cargo run -q -p hipfire-runtime --features deltanet --example dflash_smoke -- \
        "$OUT_HFQ" --block "$BLOCK" --ctx "$CTX"; then
        echo "tiny-spec-gate: FAIL tiny DFlash runtime smoke"
        exit 1
    fi
else
    echo "tiny-spec-gate: SKIP tiny DFlash train/export/smoke (HIPFIRE_TINY_SPEC_SKIP_DFLASH=1)"
fi

echo "tiny-spec-gate: PASS"
exit 0

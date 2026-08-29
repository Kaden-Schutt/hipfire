#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Reproduces the Gemma4 E-series path-specific numerical/KV checks used by
# this port. This is not a universal gate, promotion criterion, or admission
# record; see docs/VALIDATION.md for the claim-specific validation routes.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
E2B_MODEL="${E2B_MODEL:-}"
E4B_MODEL="${E4B_MODEL:-}"
DENSE_MODEL="${DENSE_MODEL:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/target/validation/gemma4-eseries/$RUN_ID}"

if [[ -z "$E2B_MODEL" || -z "$E4B_MODEL" ]]; then
    echo "usage: E2B_MODEL=/path/e2b.hfq E4B_MODEL=/path/e4b.hfq [DENSE_MODEL=/path/12b.hfq] [GPU_ID=N] $0" >&2
    exit 2
fi
for model in "$E2B_MODEL" "$E4B_MODEL"; do
    if [[ ! -f "$model" ]]; then
        echo "missing model: $model" >&2
        exit 2
    fi
done
if [[ -n "$DENSE_MODEL" && ! -f "$DENSE_MODEL" ]]; then
    echo "missing dense control: $DENSE_MODEL" >&2
    exit 2
fi

E2B_MODEL="$(realpath "$E2B_MODEL")"
E4B_MODEL="$(realpath "$E4B_MODEL")"
if [[ -n "$DENSE_MODEL" ]]; then
    DENSE_MODEL="$(realpath "$DENSE_MODEL")"
fi

cd "$ROOT"
cargo test -p hipfire-arch-gemma4 batch_window_must_fit_allocated_kv_capacity --lib
cargo build --release --locked --features deltanet -p hipfire-arch-gemma4 \
    --example infer_gemma4 --example verify_batch_gemma4

mkdir -p "$OUT_DIR"
{
    echo "run_id=$RUN_ID"
    echo "git_head=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short
    echo "git_status_end"
    echo "gpu_id=$GPU_ID"
    echo "e2b_model=$E2B_MODEL"
    echo "e4b_model=$E4B_MODEL"
    echo "dense_model=$DENSE_MODEL"
    sha256sum "$E2B_MODEL" "$E4B_MODEL"
    if [[ -n "$DENSE_MODEL" ]]; then
        sha256sum "$DENSE_MODEL"
    fi
    sha256sum ./target/release/examples/infer_gemma4 \
        ./target/release/examples/verify_batch_gemma4
    env | LC_ALL=C sort | sed -n '/^HIPFIRE_/p'
} >"$OUT_DIR/identity.txt"

run_eseries() {
    local label="$1"
    local model="$2"
    local expected_shape="$3"

    env HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_GEMMA4_GRAPH=0 HIPFIRE_GEMMA4_EAGLE=0 \
        ./target/release/examples/infer_gemma4 \
        --model "$model" --route auto --token-ids 2,9259 --max 1 --rep-pen 1.0 \
        2>&1 | tee "$OUT_DIR/$label-top1.log"
    grep -Fq "$expected_shape" "$OUT_DIR/$label-top1.log"
    grep -Fq 'token ids: [236888]' "$OUT_DIR/$label-top1.log"

    env HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_GEMMA4_GRAPH=0 HIPFIRE_GEMMA4_EAGLE=0 \
        ./target/release/examples/verify_batch_gemma4 \
        --model "$model" --token-ids 2,9259,236888,575,106 --bs 1,2,4 \
        2>&1 | tee "$OUT_DIR/$label-batch.log"
    grep -Fq "$expected_shape" "$OUT_DIR/$label-batch.log"
    grep -Fq '=== OVERALL: PASS ===' "$OUT_DIR/$label-batch.log"
}

run_eseries e2b "$E2B_MODEL" 'gemma4 dim=1536 layers=35 sliding=28 full=7'
run_eseries e4b "$E4B_MODEL" 'gemma4 dim=2560 layers=42 sliding=35 full=7'

if [[ -n "$DENSE_MODEL" ]]; then
    env HIP_VISIBLE_DEVICES="$GPU_ID" HIPFIRE_GEMMA4_GRAPH=0 HIPFIRE_GEMMA4_EAGLE=0 \
        ./target/release/examples/infer_gemma4 \
        --model "$DENSE_MODEL" --route eager --token-ids 2,9259 --max 1 --rep-pen 1.0 \
        2>&1 | tee "$OUT_DIR/dense-top1.log"
    grep -Fq 'gemma4 dim=3840 layers=48 sliding=40 full=8' "$OUT_DIR/dense-top1.log"
    grep -Fq 'token ids: [575]' "$OUT_DIR/dense-top1.log"
fi

echo "Gemma4 E-series parity reproduction: PASS (artifacts: $OUT_DIR)"

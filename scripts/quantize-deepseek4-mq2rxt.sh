#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
#
# Build DeepSeek V4 Flash 0731 MQ2RXT directly from the native parent.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <0731-parent-dir> <0731-mq2r-base> <0731-dspark-mq2r-base> <output-dir>" >&2
    exit 2
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PARENT=$1
TRUNK_BASE=$2
DSPARK_BASE=$3
OUTPUT_DIR=$4
TRUNK_BASE_SHA=cbf2bbcfa3f47b1712a071836b2c48232dad7dfb763813a720f7d348a9318cce
DSPARK_BASE_SHA=bc695a000643801d26e5ae96c9f4ac4c222a36d9db40566f4cc1de0e9d3d5d2e
PARENT_REVISION=7872f01b1d1fe23eabc4c98b48bffcef5a386062
QUANT_BIN="$ROOT/target/release/hipfire-quantize"
BAKE_SOURCE="$ROOT/scripts/reap/hfq_overlay_bake.rs"
BAKE_BIN="$ROOT/target/release/hfq_overlay_bake"
TRUNK_OVERLAY="$OUTPUT_DIR/deepseek-v4-flash-0731-mq2rxt-trunk-overlay.hfq"
DSPARK_OVERLAY="$OUTPUT_DIR/deepseek-v4-flash-0731-mq2rxt-dspark-overlay.hfq"
TRUNK_OUTPUT="$OUTPUT_DIR/deepseek-v4-flash-0731.mq2rxt"
DSPARK_OUTPUT="$OUTPUT_DIR/deepseek-v4-flash-0731-dspark.mq2rxt"

for input in "$PARENT/config.json" "$PARENT/model.safetensors.index.json" "$TRUNK_BASE" "$DSPARK_BASE"; do
    if [[ ! -s "$input" ]]; then
        echo "missing required input: $input" >&2
        exit 2
    fi
done

PARENT_METADATA="$PARENT/.cache/huggingface/download/config.json.metadata"
if [[ ! -s "$PARENT_METADATA" ]]; then
    echo "missing Hugging Face revision metadata: $PARENT_METADATA" >&2
    exit 2
fi
IFS= read -r ACTUAL_PARENT_REVISION <"$PARENT_METADATA"
if [[ "$ACTUAL_PARENT_REVISION" != "$PARENT_REVISION" ]]; then
    echo "wrong 0731 parent revision: got $ACTUAL_PARENT_REVISION, expected $PARENT_REVISION" >&2
    exit 2
fi
shopt -s nullglob
PARENT_SHARDS=("$PARENT"/model-*-of-00048.safetensors)
if [[ ${#PARENT_SHARDS[@]} -ne 48 ]]; then
    echo "incomplete 0731 parent: found ${#PARENT_SHARDS[@]} of 48 safetensor shards" >&2
    exit 2
fi
shopt -u nullglob

mkdir -p "$OUTPUT_DIR"
for output in "$TRUNK_OVERLAY" "$DSPARK_OVERLAY" "$TRUNK_OUTPUT" "$DSPARK_OUTPUT"; do
    if [[ -e "$output" ]]; then
        echo "refusing to overwrite existing output: $output" >&2
        exit 2
    fi
done

printf '%s  %s\n' "$TRUNK_BASE_SHA" "$TRUNK_BASE" | sha256sum --check --strict
printf '%s  %s\n' "$DSPARK_BASE_SHA" "$DSPARK_BASE" | sha256sum --check --strict

cargo build --release -p hipfire-quantize --bin hipfire-quantize \
    --manifest-path "$ROOT/Cargo.toml"
rustc --edition=2021 -O "$BAKE_SOURCE" -o "$BAKE_BIN"

"$QUANT_BIN" \
    --input "$PARENT" \
    --output "$TRUNK_OVERLAY" \
    --format deepseek4-mq2rxt-overlay

"$QUANT_BIN" \
    --input "$PARENT" \
    --output "$DSPARK_OVERLAY" \
    --format deepseek4-mq2rxt-overlay \
    --include-prefix mtp.

"$BAKE_BIN" \
    "$TRUNK_OUTPUT" "$TRUNK_BASE" "$TRUNK_OVERLAY" 554 \
    --metadata-from-overlay
"$BAKE_BIN" \
    "$DSPARK_OUTPUT" "$DSPARK_BASE" "$DSPARK_OVERLAY" 24 \
    --metadata-from-overlay

sha256sum "$TRUNK_OVERLAY" "$DSPARK_OVERLAY" "$TRUNK_OUTPUT" "$DSPARK_OUTPUT"

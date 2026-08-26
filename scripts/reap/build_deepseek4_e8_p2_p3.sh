#!/usr/bin/env bash
# Resume-safe assembly of cumulative DeepSeek-V4 MQ2R P2 and P3 overlays.
#
# P2 = frozen P1 + compressor/indexer bucket.
# P3 = P2 + router bucket + output head.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
BUILDER="$ROOT/scripts/reap/build_deepseek4_e8_bucket_overlay.sh"
MERGE_SOURCE="$ROOT/scripts/reap/hfq_overlay_merge.rs"
MERGE_BIN="$ROOT/target/release/hfq_overlay_merge"
PLAN_SOURCE="$ROOT/scripts/reap/deepseek4_e8_plan.rs"
PLAN_BIN="$ROOT/target/release/deepseek4_e8_plan"
P1="$CAMPAIGN/candidates/p1-all-layers/overlay.hfq"

if [[ ! -s "$P1" ]]; then
    echo "missing frozen P1 overlay: $P1" >&2
    exit 2
fi
if [[ ! -x "$MERGE_BIN" || "$MERGE_SOURCE" -nt "$MERGE_BIN" ]]; then
    rustc -O "$MERGE_SOURCE" -o "$MERGE_BIN"
fi
if [[ ! -x "$PLAN_BIN" || "$PLAN_SOURCE" -nt "$PLAN_BIN" ]]; then
    rustc -O "$PLAN_SOURCE" -o "$PLAN_BIN"
fi

p2_inputs=()
for layer in $(seq 2 42); do
    printf -v tag '%02d' "$layer"
    candidate="$CAMPAIGN/candidates/p2-layer-$tag/overlay.hfq"
    if [[ ! -s "$candidate" ]]; then
        "$BUILDER" p2 "$layer"
    fi
    p2_inputs+=("$candidate")
done

p2_bucket="$CAMPAIGN/candidates/p2-bucket"
p2_all="$CAMPAIGN/candidates/p2-all-layers"
mkdir -p "$p2_bucket" "$p2_all"
if [[ ! -s "$p2_bucket/overlay.hfq" ]]; then
    "$MERGE_BIN" "$p2_bucket/overlay.hfq" "${p2_inputs[@]}"
fi
"$PLAN_BIN" p2 "$p2_all/reap_plan.json"
if [[ ! -s "$p2_all/overlay.hfq" ]]; then
    "$MERGE_BIN" "$p2_all/overlay.hfq" "$P1" "$p2_bucket/overlay.hfq"
fi

router_inputs=()
for layer in $(seq 0 42); do
    printf -v tag '%02d' "$layer"
    candidate="$CAMPAIGN/candidates/p3-router-layer-$tag/overlay.hfq"
    if [[ ! -s "$candidate" ]]; then
        "$BUILDER" p3-router "$layer"
    fi
    router_inputs+=("$candidate")
done
head_overlay="$CAMPAIGN/candidates/p3-head/overlay.hfq"
if [[ ! -s "$head_overlay" ]]; then
    "$BUILDER" p3-head
fi

p3_router="$CAMPAIGN/candidates/p3-router-bucket"
p3_bucket="$CAMPAIGN/candidates/p3-bucket"
p3_all="$CAMPAIGN/candidates/p3-all-layers"
mkdir -p "$p3_router" "$p3_bucket" "$p3_all"
if [[ ! -s "$p3_router/overlay.hfq" ]]; then
    "$MERGE_BIN" "$p3_router/overlay.hfq" "${router_inputs[@]}"
fi
"$PLAN_BIN" router "$p3_router/reap_plan.json"
if [[ ! -s "$p3_bucket/overlay.hfq" ]]; then
    "$MERGE_BIN" "$p3_bucket/overlay.hfq" "$p3_router/overlay.hfq" "$head_overlay"
fi
"$PLAN_BIN" p3 "$p3_all/reap_plan.json"
if [[ ! -s "$p3_all/overlay.hfq" ]]; then
    "$MERGE_BIN" \
        "$p3_all/overlay.hfq" \
        "$P1" "$p2_bucket/overlay.hfq" "$p3_bucket/overlay.hfq"
fi

sha256sum \
    "$p2_bucket/overlay.hfq" "$p2_all/reap_plan.json" "$p2_all/overlay.hfq" \
    "$p3_router/overlay.hfq" "$p3_router/reap_plan.json" \
    "$p3_bucket/overlay.hfq" "$p3_all/reap_plan.json" "$p3_all/overlay.hfq"

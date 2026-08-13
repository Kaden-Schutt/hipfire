#!/usr/bin/env bash
# Resume-safe assembly of the DeepSeek-V4 MQ2R P3 candidate with
# least-squares-corrected E8 row scales for the router and output head.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
BUILDER="$ROOT/scripts/reap/build_deepseek4_e8_bucket_overlay.sh"
MERGE_SOURCE="$ROOT/scripts/reap/hfq_overlay_merge.rs"
MERGE_BIN="$ROOT/target/release/hfq_overlay_merge"
PLAN_SOURCE="$ROOT/scripts/reap/deepseek4_e8_plan.rs"
PLAN_BIN="$ROOT/target/release/deepseek4_e8_plan"
P2="$CAMPAIGN/candidates/p2-all-layers/overlay.hfq"

if [[ ! -s "$P2" ]]; then
    echo "missing frozen P2 overlay: $P2" >&2
    exit 2
fi
if [[ ! -x "$MERGE_BIN" || "$MERGE_SOURCE" -nt "$MERGE_BIN" ]]; then
    rustc -O "$MERGE_SOURCE" -o "$MERGE_BIN"
fi
if [[ ! -x "$PLAN_BIN" || "$PLAN_SOURCE" -nt "$PLAN_BIN" ]]; then
    rustc -O "$PLAN_SOURCE" -o "$PLAN_BIN"
fi

router_inputs=()
for layer in $(seq 0 42); do
    printf -v tag '%02d' "$layer"
    candidate="$CAMPAIGN/candidates/p3-router-layer-$tag-lsq/overlay.hfq"
    if [[ ! -s "$candidate" ]]; then
        E8_TIER=mfp4e8soa-lsq CANDIDATE_SUFFIX=-lsq \
            "$BUILDER" p3-router "$layer"
    fi
    router_inputs+=("$candidate")
done

head="$CAMPAIGN/candidates/p3-head-lsq/overlay.hfq"
if [[ ! -s "$head" ]]; then
    E8_TIER=mfp4e8soa-lsq CANDIDATE_SUFFIX=-lsq \
        "$BUILDER" p3-head
fi

router="$CAMPAIGN/candidates/p3-router-bucket-lsq"
candidate="$CAMPAIGN/candidates/p3-all-layers-lsq"
mkdir -p "$router" "$candidate"
if [[ ! -s "$router/overlay.hfq" ]]; then
    "$MERGE_BIN" "$router/overlay.hfq" "${router_inputs[@]}"
fi
"$PLAN_BIN" router "$router/reap_plan.json"
"$PLAN_BIN" p3 "$candidate/reap_plan.json"
if [[ ! -s "$candidate/overlay.hfq" ]]; then
    "$MERGE_BIN" "$candidate/overlay.hfq" \
        "$P2" "$router/overlay.hfq" "$head"
fi

sha256sum \
    "$router/overlay.hfq" "$router/reap_plan.json" \
    "$head" "$candidate/reap_plan.json" "$candidate/overlay.hfq"

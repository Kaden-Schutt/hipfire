#!/usr/bin/env bash
# Resume-safe assembly of the DeepSeek-V4 MQ2R P3 candidate with the
# frozen regular-E8 router and an activation-weighted E8 output head.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CAMPAIGN=${CAMPAIGN:-/mnt/nas/kaden/experiments/deepseek4-mq2r-e8-20260723}
BUILDER="$ROOT/scripts/reap/build_deepseek4_e8_bucket_overlay.sh"
MERGE_SOURCE="$ROOT/scripts/reap/hfq_overlay_merge.rs"
MERGE_BIN="$ROOT/target/release/hfq_overlay_merge"
PLAN_SOURCE="$ROOT/scripts/reap/deepseek4_e8_plan.rs"
PLAN_BIN="$ROOT/target/release/deepseek4_e8_plan"
P2="$CAMPAIGN/candidates/p2-all-layers/overlay.hfq"
ROUTER="$CAMPAIGN/candidates/p3-router-bucket/overlay.hfq"
WIKI_IMATRIX="$CAMPAIGN/calibration/head-wikitext-ctx1024.imatrix"
CODE_IMATRIX="$CAMPAIGN/calibration/head-code-ctx1024.imatrix"
HEAD_DIR="$CAMPAIGN/candidates/p3-head-awls"
CANDIDATE="$CAMPAIGN/candidates/p3-all-layers-awls-head"

for required in "$P2" "$ROUTER" "$WIKI_IMATRIX" "$CODE_IMATRIX"; do
    if [[ ! -s "$required" ]]; then
        echo "missing required input: $required" >&2
        exit 2
    fi
done
if [[ ! -x "$MERGE_BIN" || "$MERGE_SOURCE" -nt "$MERGE_BIN" ]]; then
    rustc -O "$MERGE_SOURCE" -o "$MERGE_BIN"
fi
if [[ ! -x "$PLAN_BIN" || "$PLAN_SOURCE" -nt "$PLAN_BIN" ]]; then
    rustc -O "$PLAN_SOURCE" -o "$PLAN_BIN"
fi

mkdir -p "$HEAD_DIR" "$CANDIDATE"
imatrix_manifest=$(sha256sum "$WIKI_IMATRIX" "$CODE_IMATRIX")
if [[ -s "$HEAD_DIR/imatrix.sha256" ]] &&
    [[ "$(cat "$HEAD_DIR/imatrix.sha256")" != "$imatrix_manifest" ]]; then
    echo "AWLS head exists for different calibration inputs: $HEAD_DIR" >&2
    exit 2
fi
if [[ ! -s "$HEAD_DIR/overlay.hfq" ]]; then
    E8_TIER=mfp4e8soa-awls \
        CANDIDATE_SUFFIX=-awls \
        E8_IMATRIX="$WIKI_IMATRIX:$CODE_IMATRIX" \
        "$BUILDER" p3-head
    printf '%s\n' "$imatrix_manifest" >"$HEAD_DIR/imatrix.sha256"
fi

"$PLAN_BIN" p3 "$CANDIDATE/reap_plan.json"
if [[ ! -s "$CANDIDATE/overlay.hfq" ]]; then
    "$MERGE_BIN" "$CANDIDATE/overlay.hfq" \
        "$P2" "$ROUTER" "$HEAD_DIR/overlay.hfq"
fi

sha256sum \
    "$P2" "$ROUTER" "$HEAD_DIR/overlay.hfq" \
    "$CANDIDATE/reap_plan.json" "$CANDIDATE/overlay.hfq"

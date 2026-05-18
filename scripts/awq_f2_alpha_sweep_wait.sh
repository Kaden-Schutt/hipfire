#!/usr/bin/env bash
# awq_f2_alpha_sweep_wait.sh — wait for any in-flight eval_hipfire, then
# run the F2 alpha sweep at α∈[0.35..0.65] step 0.05, n=100, KV q8.
#
# F2-whitelist is the default in this branch (no HIPFIRE_AWQ_F1_ONLY needed).

set -euo pipefail
cd "$(dirname "$0")/.."

# Wait for the running F1/F2 comparison eval to release the GPU.
until ! pgrep -f "target/release/examples/eval_hipfire" > /dev/null 2>&1; do
    sleep 5
done

# Brief pause to let any cleanup land before we hammer the slot.
sleep 2

MAX_CHUNKS=100 \
KV_MODE=q8 \
RESULTS_LABEL=2026-05-14-f2-alpha-sweep-n100-kvq8-9b-gfx906 \
exec scripts/awq_alpha_sweep.sh 0.35 0.40 0.45 0.50 0.55 0.60 0.65

#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-dispatch-floor" || exit 9
export HIP_VISIBLE_DEVICES=1
cargo build --release -p rdna-compute --example bench_dispatch_floor 2>&1 | grep -E "^error|Finished" | head -3
./target/release/examples/bench_dispatch_floor
gpu_release
echo "DONE"

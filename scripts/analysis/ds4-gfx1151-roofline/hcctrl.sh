#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-hc-ctrl" || exit 9
export HIP_VISIBLE_DEVICES=1
M=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
BIN=./target/release/examples/deepseek4_prefill_bench
cargo build --release -p hipfire-arch-deepseek4 --example deepseek4_prefill_bench 2>&1 | grep -E "^error|Finished" | head -3
run() {  # $1 = flag value
  echo "### HIPFIRE_HC_CTRL_T1024=$1"
  HIPFIRE_HC_CTRL_T1024=$1 $BIN $M --prefix 2048 --ar-ref 25 --tokens 0 \
    --batch 1 --e8-batched 0 --reps 1 --warmup 3 2>&1 | grep -E "^AR-REF"
}
run 0
run 1
run 0
run 1
gpu_release
echo "DONE"

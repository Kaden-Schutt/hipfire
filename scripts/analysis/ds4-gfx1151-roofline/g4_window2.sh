#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-g4-window" || exit 9
export HIP_VISIBLE_DEVICES=1
./target/release/examples/deepseek4_prefill_bench \
  /home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r \
  --prefix 2048 --ar-ref 7 \
  --tokens 0 --batch 1,2,4,6,8 --e8-batched 0,8 --reps 5 --warmup 2
rc=$?
gpu_release
echo "DONE rc=$rc"
exit $rc

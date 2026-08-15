#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-prefill; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
cd $W; source /root/.cargo/env
M=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
cargo build --release -p hipfire-arch-deepseek4 --example deepseek4_prefill_bench > $E/00-build.log 2>&1
echo "build exit=$?" >> $E/progress.txt
export HIP_VISIBLE_DEVICES=0 HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
for n in 256 1024 2048; do
  ./target/release/examples/deepseek4_prefill_bench "$M" \
    --prompt benchmarks/prompts/ds4-gfx942-ar-2048.txt --tokens $n --batch 1024 --warmup 1 --reps 5 \
    > $E/prefill-$n.stdout 2> $E/prefill-$n.stderr
  echo "--- tokens=$n exit=$? ---" >> $E/progress.txt
  grep -iE "tok/s|tokens/s|ms|throughput|prefill" $E/prefill-$n.stdout 2>/dev/null | tail -6 >> $E/progress.txt
done
echo PREFILLDONE >> $E/progress.txt

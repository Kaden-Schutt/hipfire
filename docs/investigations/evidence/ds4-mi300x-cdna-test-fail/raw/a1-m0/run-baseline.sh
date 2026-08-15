#!/bin/bash
set -x
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-base; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_PROMPT="$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)"
for run in warm m1 m2 m3; do
  MT=16; [ "$run" != "warm" ] && MT=510
  HIPFIRE_MAX_TOKENS=$MT HIPFIRE_LABEL=base-$run \
  HIPFIRE_EVENT_LOG=$E/$run-events.jsonl HIPFIRE_TEXT_OUT=$E/$run-text.txt \
  HIPFIRE_DAEMON_STDERR=$E/$run-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$run-driver.out 2> $E/$run-driver.err
  echo "=== $run exit=$? ===" >> $E/progress.txt
  grep -h commit_ready $E/$run-events.jsonl >> $E/progress.txt 2>/dev/null
done
echo ALLDONE >> $E/progress.txt

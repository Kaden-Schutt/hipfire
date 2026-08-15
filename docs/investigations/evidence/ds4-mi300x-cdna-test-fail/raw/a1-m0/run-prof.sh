#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-prof-f1on; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=900 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL=1
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
# 1) warm (unprofiled) so JIT is not in the trace
HIPFIRE_MAX_TOKENS=16 HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=warm HIPFIRE_EVENT_LOG=$E/warm-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/warm.txt HIPFIRE_DAEMON_STDERR=$E/warm.stderr \
  python3 -u $D/04-profile-feed.py > $E/warm.out 2> $E/warm.err
echo "warm done $(grep -ho \"tok_s[^,]*\" $E/warm-events.jsonl|head -1)" >> $E/progress.txt
# 2) profiled decode screen: 2048 prompt / 128 output
HIPFIRE_MAX_TOKENS=128 HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=prof HIPFIRE_EVENT_LOG=$E/prof-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/prof.txt HIPFIRE_DAEMON_STDERR=$E/prof.stderr \
  timeout 2400 rocprofv3 --kernel-trace --stats --output-format csv -d $E/rocprof -o ds4 -- \
  python3 -u $D/04-profile-feed.py > $E/prof.out 2> $E/prof.err
echo "prof exit=$?" >> $E/progress.txt
grep -h commit_ready $E/prof-events.jsonl >> $E/progress.txt 2>/dev/null
find $E/rocprof -name "*stats*" -o -name "*.csv" | head -10 >> $E/progress.txt
echo PROFDONE >> $E/progress.txt

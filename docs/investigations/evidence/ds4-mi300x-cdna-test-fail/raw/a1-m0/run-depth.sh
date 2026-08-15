#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-depth; mkdir -p $E
B=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-base
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
# wait for the baseline job to finish so the GPU stays single-tenant
for i in $(seq 1 400); do grep -q ALLDONE $B/progress.txt 2>/dev/null && break; sleep 15; done
echo "baseline done, starting depth sweep $(date -u)" >> $E/progress.txt
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_MAX_TOKENS=32
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
# token-count-approximate prefixes by character fraction; exact counts come from the event log
for frac in 64 256 1024 2048; do
  case $frac in
    64)   P=$(echo "$FULL" | head -c 300) ;;
    256)  P=$(echo "$FULL" | head -c 1230) ;;
    1024) P=$(echo "$FULL" | head -c 4940) ;;
    2048) P="$FULL" ;;
  esac
  HIPFIRE_PROMPT="$P" HIPFIRE_LABEL=depth-$frac \
  HIPFIRE_EVENT_LOG=$E/d$frac-events.jsonl HIPFIRE_TEXT_OUT=$E/d$frac-text.txt \
  HIPFIRE_DAEMON_STDERR=$E/d$frac-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/d$frac-driver.out 2> $E/d$frac-driver.err
  grep -h commit_ready $E/d$frac-events.jsonl >> $E/progress.txt 2>/dev/null
done
echo DEPTHDONE >> $E/progress.txt

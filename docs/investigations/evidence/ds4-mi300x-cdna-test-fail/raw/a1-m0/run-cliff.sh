#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-cliff; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_MAX_TOKENS=32
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
NEAR=$(echo "$FULL" | head -c 9200)
run() { # $1 label  $2 prompt  $3 mcp
  if [ -n "$3" ]; then export HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS=$3; else unset HIPFIRE_DEEPSEEK4_MAX_COMPRESS_POS; fi
  HIPFIRE_PROMPT="$2" HIPFIRE_LABEL=$1 HIPFIRE_EVENT_LOG=$E/$1-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/$1-text.txt HIPFIRE_DAEMON_STDERR=$E/$1-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$1-driver.out 2> $E/$1-driver.err
  echo "--- $1 mcp=${3:-default} ---" >> $E/progress.txt
  grep -h commit_ready $E/$1-events.jsonl >> $E/progress.txt 2>/dev/null
}
run A-2052-mcp2048-default "$FULL" ""
run B-2052-mcp8192        "$FULL" 8192
run C-under-mcp2048       "$NEAR" ""
echo CLIFFDONE >> $E/progress.txt

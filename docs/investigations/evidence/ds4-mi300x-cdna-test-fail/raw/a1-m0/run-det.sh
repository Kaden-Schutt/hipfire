#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-det; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_GFX942_MQ2_DOWN_ALLRANKS=1
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
run() { if [ "$2" = on ]; then export HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL=1; else unset HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL; fi
  HIPFIRE_MAX_TOKENS=$3 HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=$1 HIPFIRE_EVENT_LOG=$E/$1-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/$1-text.txt HIPFIRE_DAEMON_STDERR=$E/$1-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$1-driver.out 2> $E/$1-driver.err
  echo "--- $1 f1=$2 tok=$3 allranks=1 ---" >> $E/progress.txt
  grep -h commit_ready $E/$1-events.jsonl >> $E/progress.txt 2>/dev/null; }
run warm    on  16
run D1off   off 510
run D2off   off 510
run D3on    on  510
run D4on    on  510
echo "=== md5 ===" >> $E/progress.txt
md5sum $E/D1off-text.txt $E/D2off-text.txt $E/D3on-text.txt $E/D4on-text.txt >> $E/progress.txt 2>/dev/null
cmp -s $E/D1off-text.txt $E/D2off-text.txt && echo "DETERMINISM_off_off=YES" >> $E/progress.txt || echo "DETERMINISM_off_off=NO" >> $E/progress.txt
cmp -s $E/D3on-text.txt $E/D4on-text.txt && echo "DETERMINISM_on_on=YES" >> $E/progress.txt || echo "DETERMINISM_on_on=NO" >> $E/progress.txt
cmp -s $E/D1off-text.txt $E/D3on-text.txt && echo "F1_EXACTNESS=YES" >> $E/progress.txt || echo "F1_EXACTNESS=NO" >> $E/progress.txt
echo DETDONE >> $E/progress.txt

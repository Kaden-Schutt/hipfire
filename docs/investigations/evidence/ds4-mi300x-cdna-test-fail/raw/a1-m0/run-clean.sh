#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-clean; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
T=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/transfer3
cd $W; source /root/.cargo/env
{ git diff HEAD > $E/00-pre.patch
  git checkout HEAD -- $(git diff HEAD --name-only) 2>/dev/null
  git apply $T/campaign-tracked.patch && echo PATCH_OK
  tar xzf $T/campaign-untracked.tgz && echo TAR_OK
  echo "--- gfx1151 kernel must be pristine ---"; git diff HEAD --stat -- kernels/src/indexer_top_k_buf_parallel.gfx1151.hip
  ls -la kernels/src/indexer_top_k_buf_parallel.gfx942.hip
  cargo build --release --features deltanet --example daemon -p hipfire-runtime 2>&1 | tail -1
  sha256sum target/release/examples/daemon; } > $E/00-build.log 2>&1
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
run() { HIPFIRE_MAX_TOKENS=$2 HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=$1 HIPFIRE_EVENT_LOG=$E/$1-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/$1-text.txt HIPFIRE_DAEMON_STDERR=$E/$1-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$1-driver.out 2> $E/$1-driver.err
  echo "--- $1 tok=$2 ---" >> $E/progress.txt
  grep -h commit_ready $E/$1-events.jsonl >> $E/progress.txt 2>/dev/null; }
# ALL LEVERS OFF. Short runs first so we get an A/A answer fast.
run warm 16
run AA1 64
run AA2 64
echo "=== lever line ===" >> $E/progress.txt
grep -h "A2 levers" $E/AA1-daemon.stderr >> $E/progress.txt 2>/dev/null
md5sum $E/AA1-text.txt $E/AA2-text.txt >> $E/progress.txt 2>/dev/null
cmp -s $E/AA1-text.txt $E/AA2-text.txt && echo "CLEAN_AA_64=YES" >> $E/progress.txt || echo "CLEAN_AA_64=NO" >> $E/progress.txt
echo CLEANDONE >> $E/progress.txt

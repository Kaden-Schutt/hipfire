#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-f1; mkdir -p $E
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
T=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/transfer2
cd $W
{
git diff HEAD > $E/00-pre-f1-tree.patch
git checkout HEAD -- $(git diff HEAD --name-only) 2>/dev/null
git apply $T/campaign-tracked.patch && echo "patch applied OK"
tar xzf $T/campaign-untracked.tgz && echo "tarball applied OK"
grep -c INDEXER_TOPK_PARALLEL crates/hipfire-arch-deepseek4/src/forward.rs
source /root/.cargo/env
cargo build --release -p hipfire-cli 2>&1 | tail -2
cargo build --release --features deltanet --example daemon -p hipfire-runtime 2>&1 | tail -2
sha256sum target/release/hipfire target/release/examples/daemon
} > $E/00-build.log 2>&1
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
run() { # label  f1  maxtok
  if [ "$2" = on ]; then export HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL=1; else unset HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL; fi
  HIPFIRE_MAX_TOKENS=$3 HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=$1 HIPFIRE_EVENT_LOG=$E/$1-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/$1-text.txt HIPFIRE_DAEMON_STDERR=$E/$1-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$1-driver.out 2> $E/$1-driver.err
  echo "--- $1 f1=$2 maxtok=$3 ---" >> $E/progress.txt
  grep -h commit_ready $E/$1-events.jsonl >> $E/progress.txt 2>/dev/null
}
run warm       on  16
run f1off-32   off 32
run f1on-32    on  32
run f1off-510  off 510
run f1on-510   on  510
echo "=== greedy 510 byte-identity ===" >> $E/progress.txt
md5sum $E/f1off-510-text.txt $E/f1on-510-text.txt >> $E/progress.txt 2>/dev/null
cmp -s $E/f1off-510-text.txt $E/f1on-510-text.txt && echo "BYTE_IDENTICAL=YES" >> $E/progress.txt || echo "BYTE_IDENTICAL=NO" >> $E/progress.txt
echo F1DONE >> $E/progress.txt

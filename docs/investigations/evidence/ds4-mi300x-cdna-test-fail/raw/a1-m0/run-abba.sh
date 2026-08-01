#!/bin/bash
E=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-abba; mkdir -p $E/pairs
W=/mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx
D=/mnt/scratch/hipfire-evidence/ds4-mi300x-agentmaxx/a1-m0
cd $W
export HIP_VISIBLE_DEVICES=0 HIPFIRE_LOCAL=1 HIPFIRE_SPECULATION=off
export HIPFIRE_DAEMON_BIN=$W/target/release/examples/daemon
export HIPFIRE_MODEL=/mnt/scratch/models/deepseek-v4-flash-mq2r/deepseek-v4-flash.mq2r
export HIPFIRE_KERNEL_CACHE=$W/.hipfire_kernels
export HIPFIRE_TEMP=0 HIPFIRE_PING_TIMEOUT=600 HIPFIRE_LOAD_TIMEOUT=1800 HIPFIRE_GEN_TIMEOUT=3600
export HIPFIRE_MAX_TOKENS=510
FULL=$(cat benchmarks/prompts/ds4-gfx942-ar-2048.txt)
one() { # $1 pair  $2 arm(A=off,B=on)
  if [ "$2" = B ]; then export HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL=1; else unset HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL; fi
  TS=$(date -u +%Y%m%dT%H%M%SZ); L=p$1-$2-$TS
  HIPFIRE_PROMPT="$FULL" HIPFIRE_LABEL=$L HIPFIRE_EVENT_LOG=$E/$L-events.jsonl \
  HIPFIRE_TEXT_OUT=$E/$L-text.txt HIPFIRE_DAEMON_STDERR=$E/$L-daemon.stderr \
  python3 -u $D/04-profile-feed.py > $E/$L-driver.out 2> $E/$L-driver.err
  TS2=$(grep -h commit_ready $E/$L-events.jsonl | python3 -c "
import sys,json
o=json.loads(sys.stdin.readline())
tot=o[\"total_ms\"]; pf=o[\"prefill_ms\"]; n=o[\"tokens\"]
d=n/((tot-pf)/1000.0)
json.dump({\"decode_tok_s\":d,\"prefill_tok_s\":o[\"prefill_tokens\"]/(pf/1000.0),\"tokens\":n,\"total_ms\":tot,\"prefill_ms\":pf,\"finish\":o[\"finish_reason\"],\"drafter\":o[\"drafter\"]},open(\"$E/pairs/pair-$1-$2-$TS.json\",\"w\"))
print(f\"{d:.4f}\")")
  echo "pair=$1 arm=$2 decode_tok_s=$TS2 md5=$(md5sum $E/$L-text.txt 2>/dev/null|cut -d\  -f1)" >> $E/progress.txt
}
# warm every cell once, unrecorded
HIPFIRE_MAX_TOKENS=16 one warm A; HIPFIRE_MAX_TOKENS=16 one warm B
rm -f $E/pairs/pair-warm-* 2>/dev/null
for i in 1 2 3 4 5; do for arm in A B B A; do one $i $arm; done; echo "PAIR$i_DONE" >> $E/progress.txt; done
python3 $W/.omp/ds4_abba_bootstrap.py --glob "$E/pairs/pair-*.json" --metric decode_tok_s --resamples 10000 --seed 20260801 --out $E/bootstrap-decode.json >> $E/progress.txt 2>&1
echo ABBADONE >> $E/progress.txt

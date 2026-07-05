#!/usr/bin/env bash
# swarm_certify.sh <arch> <card> <model> <kernel> <variant_dir> — BUILD-PARALLEL / MEASURE-SERIAL fan-out.
# Rationale: build (cargo, ~minutes, CPU) is the bottleneck; the A/B (GPU, ~seconds) is a fraction and is
# already flock-serialized inside ab_certify_v2. So build ONE shared baseline, then certify every variant in
# variant_dir CONCURRENTLY across a pool of isolated worker worktrees (parallel cargo builds must NOT share
# kernels/src + target), while the GPU run-queue serializes the measures so they never thrash. Best verdict wins.
#
# One-time setup of the worker pool (N = SWARM_NPAR):
#   for i in $(seq 0 $((N-1))); do
#     git -C ~/hipfire worktree add ~/hipfire/.aw/sw_card<card>_w$i <BASELINE_REF>
#     ( cd ~/hipfire/.aw/sw_card<card>_w$i && HOME=$PWD/.swhome CARGO_TARGET_DIR=$PWD/target \
#         cargo build --release --example daemon --features deltanet -p hipfire-runtime )   # warm it (sccache shares artifacts)
#   done
set -u
ARCH=$1 CARD=$2 MODEL=$3 KERNEL=$4 VDIR=$5
MAIN=~/hipfire; BASELINE_REF="${BASELINE_REF:?}"
NPAR="${SWARM_NPAR:-4}"
POOL="${SWARM_POOL_PREFIX:-$MAIN/.aw/sw_card${CARD}_w}"
PRIMARY="$MAIN/.aw/sw_card${CARD}"
log(){ echo "[swarm $(date -u +%T)] $*"; }

VARS=( "$VDIR"/*.hip ); [ -e "${VARS[0]}" ] || { log "no variants in $VDIR"; exit 0; }
log "fanning out ${#VARS[@]} variants over $NPAR isolated worktrees (build-parallel, measure-serial via flock)"

# 1. baseline built ONCE, shared to all workers via PREBUILT_BASE
( cd "$PRIMARY" && export HOME="$PRIMARY/.swhome" CARGO_TARGET_DIR="$PRIMARY/target" PATH="$HOME/.bun/bin:$PATH"
  git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
  cargo build --release --example daemon --features deltanet -p hipfire-runtime >/dev/null 2>&1 ) \
  || { log "baseline build FAILED"; exit 1; }
BASE=/tmp/swarm_base_${ARCH}_c${CARD}; cp "$PRIMARY/target/release/examples/daemon" "$BASE"
log "shared baseline daemon: $BASE ($(stat -c%s "$BASE" 2>/dev/null) bytes)"

# 2. distribute variants round-robin across workers; each worker builds in its OWN worktree (parallel),
#    measures through the shared flock (serial). PREBUILT_BASE skips the redundant baseline rebuild.
worker(){ local wt=$1; shift
  for v in "$@"; do local lbl; lbl=$(basename "$v" .hip)
    WT_OVERRIDE="$wt" PREBUILT_BASE="$BASE" BASELINE_REF="$BASELINE_REF" \
      bash /tmp/ab_certify_v2.sh "$ARCH" "$CARD" "$CARD" "$MODEL" "$KERNEL" "$lbl" "$v" >>/tmp/swarm_${KERNEL}.log 2>&1
  done; }
declare -a BATCH; i=0
for v in "${VARS[@]}"; do BATCH[$((i%NPAR))]+=" $v"; i=$((i+1)); done
for s in $(seq 0 $((NPAR-1))); do
  wt="${POOL}${s}"; [ -d "$wt" ] || { log "WARN $wt missing -> primary (this slot runs serial)"; wt="$PRIMARY"; }
  worker "$wt" ${BATCH[$s]:-} &
done
wait
log "all ${#VARS[@]} variants certified"

# 3. best verdict from the shared ledger (all workers append to the same file)
L="$MAIN/autoresearch/ledger/swarm_${ARCH}_${KERNEL}.jsonl"
python3 -c "
import json
rows=[json.loads(l) for l in open('$L') if l.strip()]
cand=[d for d in rows if d.get('verdict') in ('WIN','INCONCLUSIVE','DEAD') and d.get('delta_pct') is not None]
cand.sort(key=lambda d:-d['delta_pct'])
for d in cand[:6]: print('  %-34s %-6s d=%6.2f f=%s  %s'%(d.get('label','')[:34],d['verdict'],d['delta_pct'],d.get('mwu_dominance'),(d.get('profile_feedback','') or '')[:50]))
print('BEST:',cand[0]['label'],cand[0]['verdict'],'d=%.2f'%cand[0]['delta_pct']) if cand else print('no verdicts')
" 2>/dev/null

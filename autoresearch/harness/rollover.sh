#!/usr/bin/env bash
# rollover.sh <current_round> — homogeneous-multicard baseline rollover for the
# gfx1201 autoresearch loop. Folds the BEST-per-kernel wins that are fresh since
# the last rollover into a new committed baseline; all 4 gfx1201 worktrees advance
# to it in lockstep, so subsequent rounds compound.
#
# SAFETY:
#  - HOMOGENEOUS-ONLY: refuses unless every GPU reports the same gfx arch.
#  - DRY_RUN=1 (default): report pending set + composed A/B + gate, DO NOT advance.
#  - Reversible: advance is a commit on top of the current baseline; `git checkout
#    <orig_sha>` in each worktree rolls back.
#  - Greedy-myopia guard: BEST win per kernel only (same-kernel refinements collapse).
set -u
CUR_ROUND="${1:-0}"
DRY_RUN="${DRY_RUN:-1}"
MAIN="$HOME/hipfire"
LEDGER_DIR="$MAIN/autoresearch/ledger"
MANIFEST=/tmp/baseline_manifest.txt
FOLDED_LIST=/tmp/baseline_folded.txt
WINDIRS="/tmp/wins /tmp/wins_seed"
MODEL=/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r
MIN_NAIVE_SUM="${MIN_NAIVE_SUM:-5.0}"
MIN_COMPOSED="${MIN_COMPOSED:-2.5}"
MIN_ROUNDS_GAP="${MIN_ROUNDS_GAP:-8}"
BRANCH=loop_baseline_gfx1201
log(){ echo "[rollover r$CUR_ROUND $(date -u +%T)] $*"; }

# 0. HOMOGENEOUS GUARD (fail-safe: refuse if arch can't be verified)
ROCMINFO=""; for p in rocminfo /opt/rocm/bin/rocminfo /opt/rocm-*/bin/rocminfo; do command -v "$p" >/dev/null 2>&1 && { ROCMINFO="$p"; break; }; done
if [ -z "$ROCMINFO" ]; then log "rocminfo not found -> cannot verify homogeneity -> REFUSE"; exit 3; fi
UARCH=$($ROCMINFO 2>/dev/null | grep -oiE 'gfx[0-9a-f]{3,}' | sort -u)  # {3,} so the 2-char 'gfx12' generic ISA label doesn't pollute the count
NARCH=$(printf '%s\n' "$UARCH" | grep -c gfx)
if [ "${NARCH:-0}" -ne 1 ]; then log "HETEROGENEOUS ($NARCH arches: $(echo $UARCH)) -> rollover REFUSED (homogeneous-only)"; exit 3; fi
ARCH=$(printf '%s\n' "$UARCH" | head -1)

# 1. anti-thrash + current baseline
CUR_SHA=$(git -C "$MAIN/.aw/sw_card0" rev-parse --short HEAD 2>/dev/null)
LAST_ROLL_ROUND=$(awk -F'rolled_at_round=' '/rolled_at_round=/{split($2,a," ");r=a[1]} END{print r+0}' "$MANIFEST" 2>/dev/null); LAST_ROLL_ROUND="${LAST_ROLL_ROUND:-0}"
LAST_ROLL_EPOCH=$(awk -F'rolled_at_epoch=' '/rolled_at_epoch=/{split($2,a," ");e=a[1]} END{print e+0}' "$MANIFEST" 2>/dev/null); LAST_ROLL_EPOCH="${LAST_ROLL_EPOCH:-0}"
GAP=$((CUR_ROUND - LAST_ROLL_ROUND))
if [ "$GAP" -lt "$MIN_ROUNDS_GAP" ]; then log "gap $GAP<$MIN_ROUNDS_GAP rounds since last rollover -> skip"; exit 0; fi

# 2. PENDING = best-per-kernel WIN whose .hip is fresh since last rollover (mtime > last_roll_epoch)
python3 - "$LEDGER_DIR" "$LAST_ROLL_EPOCH" "$WINDIRS" <<'PY' > /tmp/roll_pending.tsv 2>/tmp/roll_pending.err
import sys,json,glob,os
ld=sys.argv[1]; last=float(sys.argv[2]); windirs=sys.argv[3].split()
win={}  # label -> (kernel, delta)
for f in glob.glob(ld+'/swarm_gfx1201_*.jsonl'):
  for l in open(f):
    try: d=json.loads(l)
    except: continue
    if d.get('verdict')=='WIN': win[d['label']]=(d['kernel'], d.get('delta_pct') or 0)
best={}
for wd in windirs:
  for hp in glob.glob(wd+'/*.hip'):
    lbl=os.path.basename(hp)[:-4]
    if lbl not in win: continue
    if os.path.getmtime(hp) <= last: continue     # already folded / stale
    k,dp=win[lbl]
    if k not in best or dp>best[k][1]: best[k]=(lbl,dp,hp)
tot=0.0
for k,(lbl,dp,hp) in sorted(best.items(), key=lambda x:-x[1][1]):
  print(f"{dp}\t{k}\t{lbl}\t{hp}"); tot+=dp
print(f"NAIVE_SUM={tot:.2f} NKERN={len(best)}", file=sys.stderr)
PY
NAIVE=$(awk '{s+=$1} END{printf "%.2f", s+0}' /tmp/roll_pending.tsv)
NKERN=$(grep -c . /tmp/roll_pending.tsv 2>/dev/null || echo 0)
log "baseline=$CUR_SHA  pending=$NKERN kernels  naive_sum=${NAIVE}%  (pre-filter >= ${MIN_NAIVE_SUM}%)"
awk -F'\t' '{printf "    %+.2f%%  %-36s %s\n",$1,$2,$3}' /tmp/roll_pending.tsv
[ "$NKERN" -ge 1 ] || { log "no fresh pending wins -> skip"; exit 0; }
awk "BEGIN{exit !($NAIVE >= $MIN_NAIVE_SUM)}" || { log "naive_sum below pre-filter -> skip"; exit 0; }

# 3. EXPENSIVE CONFIRM: composed A/B vs current baseline on sw_card0 (idle at checkpoint)
WT="$MAIN/.aw/sw_card0"; export PATH="$HOME/.bun/bin:$PATH"
export HIPFIRE_DAEMON_ID=roll CARGO_TARGET_DIR="$WT/target"; SW="$WT/.swhome"; mkdir -p "$SW/.hipfire"
SCLK=/sys/class/drm/card0/device/pp_dpm_sclk
cd "$WT" || { log "no worktree $WT"; exit 1; }
git checkout -- kernels/src/ 2>/dev/null
DB=target/release/examples/daemon
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
measure(){ local bin=$1 out=/tmp/roll_m.jsonl clkf=/tmp/roll_m.clk
  rm -f .hipfire_kernels/*/*.hsaco "$clkf" 2>/dev/null
  ( for _ in $(seq 1 40); do grep '\*' "$SCLK" 2>/dev/null|grep -oiE "[0-9]+Mhz"|grep -oE "[0-9]+"|head -1; sleep 0.4; done > "$clkf" ) & local s=$!
  HOME="$SW" printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w1","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"w2","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m1","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HOME="$SW" HIP_VISIBLE_DEVICES=0 "$bin" 2>/dev/null > "$out"
  kill "$s" 2>/dev/null; wait "$s" 2>/dev/null
  python3 - "$out" "$clkf" <<'PY'
import json,sys
dec=None;m=""
for l in open(sys.argv[1]):
 try:d=json.loads(l)
 except:continue
 if d.get("type")=="done" and d.get("id")=="m1":dec=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or dec
 if d.get("type")=="token" and d.get("id")=="m1":m+=d.get("text","")
t=m.split();u=(len(set(t))/len(t)) if t else 0
try:c=[int(x) for x in open(sys.argv[2]) if x.strip()]
except:c=[]
print(f"{dec if dec is not None else 0:.2f} {'OK' if (len(t)>15 and u>0.35) else 'BAD'} {max(c) if c else 0}")
PY
}
build || { log "current-baseline build FAIL -> abort"; exit 1; }
cp "$DB" /tmp/roll_base
while IFS=$'\t' read -r dp kern lbl hp; do [ -n "$kern" ] && cp "$hp" "kernels/src/${kern}.hip"; done < /tmp/roll_pending.tsv
if ! build; then git checkout -- kernels/src/; log "COMPOSED build FAIL -> abort rollover (kernels conflict?)"; exit 1; fi
cp "$DB" /tmp/roll_var
git checkout -- kernels/src/
BASE=();VAR=();BC=OK;VC=OK
for r in $(seq 1 6); do
  read d c k < <(measure /tmp/roll_base); BASE+=("$d");[ "$c" = BAD ]&&BC=BAD
  read d c k < <(measure /tmp/roll_var);  VAR+=("$d"); [ "$c" = BAD ]&&VC=BAD
done
read CDELTA CF BMED VMED < <(python3 - "${BASE[*]}" "${VAR[*]}" <<'PY'
import sys,statistics as st
b=[float(x) for x in sys.argv[1].split()];v=[float(x) for x in sys.argv[2].split()]
bm=st.median(b);vm=st.median(v);delta=100*(vm-bm)/bm if bm else 0
n=len(v)*len(b); f=sum((1.0 if x>y else .5 if x==y else 0) for x in v for y in b)/n if n else .5
print(f"{delta:.2f} {f:.3f} {bm:.2f} {vm:.2f}")
PY
)
log "COMPOSED A/B: base=$BMED var=$VMED delta=${CDELTA}% f=$CF coh=$BC/$VC (base_runs=${BASE[*]} var_runs=${VAR[*]})"
GATE=$(awk "BEGIN{print (($CDELTA>=$MIN_COMPOSED)&&(\"$BC\"==\"OK\")&&(\"$VC\"==\"OK\")&&($CF>=0.85))?1:0}")
if [ "$GATE" != 1 ]; then log "GATE FAIL (need delta>=$MIN_COMPOSED, coh OK/OK, f>=0.85) -> NO rollover"; exit 0; fi

if [ "$DRY_RUN" = 1 ]; then log "GATE PASS +${CDELTA}% -> DRY-RUN: would advance to a new baseline folding [$(awk -F'\t' '{printf "%s ",$2}' /tmp/roll_pending.tsv)]. Not advancing."; exit 0; fi

# 4. ADVANCE (DRY_RUN=0): commit CLEAN composed baseline + lockstep-force all worktrees
cd "$WT"; git checkout -- kernels/src/; git clean -fdq kernels/src/   # drop certify-created untracked strays so they don't get committed
while IFS=$'\t' read -r dp kern lbl hp; do [ -n "$kern" ] && cp "$hp" "kernels/src/${kern}.hip"; done < /tmp/roll_pending.tsv
git add kernels/src/
NEWN=$(( $(awk -F'baseline_v' '/baseline_v/{split($2,a," ");n=a[1]} END{print n+0}' "$MANIFEST" 2>/dev/null) + 1 ))
FOLDED_KERNS=$(awk -F'\t' '{printf "%s,",$2}' /tmp/roll_pending.tsv)
git -c user.email=151092359+Kaden-Schutt@users.noreply.github.com -c user.name="Kaden Schutt" commit -q -m "rollover baseline_v$NEWN (+${CDELTA}% composed) [$FOLDED_KERNS] round $CUR_ROUND"
NEW_SHA=$(git rev-parse --short HEAD)
git branch -f "$BRANCH" HEAD 2>/dev/null
# force-align the other 3 (clean strays first so an untracked file can't silently block the checkout)
for c in 1 2 3; do
  git -C "$MAIN/.aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null
  git -C "$MAIN/.aw/sw_card$c" clean -fdq kernels/src/ 2>/dev/null
  git -C "$MAIN/.aw/sw_card$c" checkout -fq "$NEW_SHA" 2>/dev/null
done
# VERIFY all 4 reached NEW_SHA (a silent half-advance mixes baselines across cards)
MISALIGNED=""
for c in 0 1 2 3; do h=$(git -C "$MAIN/.aw/sw_card$c" rev-parse --short HEAD 2>/dev/null); [ "$h" = "$NEW_SHA" ] || { MISALIGNED="$MISALIGNED card$c=$h"; git -C "$MAIN/.aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null; git -C "$MAIN/.aw/sw_card$c" clean -fdq kernels/src/ 2>/dev/null; git -C "$MAIN/.aw/sw_card$c" checkout -fq "$NEW_SHA" 2>/dev/null; }; done
[ -n "$MISALIGNED" ] && log "WARNING: worktrees needed force-fix ($MISALIGNED) -> re-forced to $NEW_SHA"
NOW_EPOCH=$(date +%s)
echo "baseline_v$NEWN sha=$NEW_SHA rolled_at_round=$CUR_ROUND rolled_at_epoch=$NOW_EPOCH composed_delta=$CDELTA new_tok_s=$VMED folded=$FOLDED_KERNS $(date -u +%FT%TZ)" >> "$MANIFEST"
awk -F'\t' '{print $2}' /tmp/roll_pending.tsv >> "$FOLDED_LIST"; sort -u "$FOLDED_LIST" -o "$FOLDED_LIST" 2>/dev/null
log "ROLLED -> baseline_v$NEWN sha=$NEW_SHA new_tok_s=$VMED  folded: $FOLDED_KERNS"

# 5. RE-CENSUS: re-profile the NEW baseline so the bill-of-debt re-ranks toward the
#    shifted bottleneck (folded kernels drop, un-optimized ones rise). Detached sudo -n
#    works; rocprofv3 is user-level. /tmp/roll_var IS the baseline_v1 daemon.
log "re-censusing baseline_v$NEWN -> refreshing bod_gfx1201.json..."
cp /tmp/roll_var "$WT/target/release/examples/daemon" 2>/dev/null
if HIPFIRE_REPO="$WT" timeout 1400 bash /tmp/oracle_profile.sh gfx1201 0 0 "$MODEL" 40 > /tmp/bod_new.json 2>/tmp/bod_census.err \
   && [ -s /tmp/bod_new.json ] && python3 -c "import json,sys;d=json.load(open('/tmp/bod_new.json'));sys.exit(0 if d.get('rows') else 1)" 2>/dev/null; then
  cp /tmp/bod_gfx1201.json "/tmp/bod_gfx1201.v$((NEWN-1)).json" 2>/dev/null
  cp /tmp/bod_new.json /tmp/bod_gfx1201.json
  log "bod refreshed: top-5 now -> $(python3 -c "import json;d=json.load(open('/tmp/bod_gfx1201.json'));print(', '.join(r['kernel'][:22]+'('+str(round(r['wall_pct'],1))+'%)' for r in d['rows'][:5]))" 2>/dev/null)"
else
  log "WARNING: re-census failed -> keeping prior bod ($(head -1 /tmp/bod_census.err 2>/dev/null))"
fi

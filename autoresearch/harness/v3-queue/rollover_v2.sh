#!/usr/bin/env bash
# rollover_v2.sh <current_round> — harness-v2 rollover.
# vs v1: harvests standing WIN-commits from the per-card branches loop/card{0..3}
# (not /tmp/wins), resets those branches on fold, and reports a computed roofline-ceiling
# (accurate per-kernel L2/DRAM/CU mix) so we can see how close to roofline we are.
# Default ADVANCES the fold (composes → A/B-gates → advances BASELINE_REF → re-censuses).
# DRY_RUN=1 is an OPT-IN preview flag (report the gate but do NOT advance). Never default-dry-run:
# a fold that passes its A/B gate should commit — a silent no-advance is a liability, not a safety.
set -u
CUR_ROUND="${1:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAIN="$HOME/hipfire"
LEDGER_DIR="$MAIN/autoresearch/ledger"
MODEL=/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r
MIN_NAIVE_SUM="${MIN_NAIVE_SUM:-3.0}"
MIN_COMPOSED="${MIN_COMPOSED:-2.0}"
MIN_ROUNDS_GAP="${MIN_ROUNDS_GAP:-8}"
BASELINE_REF="${BASELINE_REF:-loop_baseline_gfx1201}"
# arch config (env-overridable; defaults = gfx1201 4-card)
ARCH="${ARCH:-gfx1201}"
CARDS="${CARDS:-0 1 2 3}"
BOD_OUT="${BOD:-/tmp/bod_gfx1201.json}"
SKIP_HOMOGUARD="${SKIP_HOMOGUARD:-0}"
MANIFEST="${MANIFEST:-/tmp/baseline_manifest.txt}"
FOLDED_LIST="${FOLDED:-/tmp/baseline_folded.txt}"
log(){ echo "[rollover_v2 r$CUR_ROUND $(date -u +%T)] $*"; }

# 0. homogeneous guard (skippable when a single-arch campaign is DECLARED on a hetero box:
#    it only ever touches its own card(s), so box-heterogeneity is irrelevant)
if [ "$SKIP_HOMOGUARD" = 1 ]; then
  log "homoguard SKIPPED (declared single-arch campaign: arch=$ARCH cards='$CARDS')"
else
  ROCMINFO=""; for p in rocminfo /opt/rocm/bin/rocminfo /opt/rocm-*/bin/rocminfo; do command -v "$p" >/dev/null 2>&1 && { ROCMINFO="$p"; break; }; done
  [ -z "$ROCMINFO" ] && { log "rocminfo not found -> REFUSE"; exit 3; }
  NARCH=$($ROCMINFO 2>/dev/null | grep -oiE 'gfx[0-9a-f]{3,}' | sort -u | grep -c gfx)
  [ "${NARCH:-0}" -ne 1 ] && { log "HETEROGENEOUS ($NARCH arches) -> rollover REFUSED"; exit 3; }
fi

# 1. anti-thrash + current baseline
CUR_SHA=$(git -C "$MAIN/.aw/sw_card0" rev-parse --short "$BASELINE_REF" 2>/dev/null)
LAST_ROLL_ROUND=$(awk -F'rolled_at_round=' '/rolled_at_round=/{split($2,a," ");r=a[1]} END{print r+0}' "$MANIFEST" 2>/dev/null); LAST_ROLL_ROUND="${LAST_ROLL_ROUND:-0}"
GAP=$((CUR_ROUND - LAST_ROLL_ROUND))
[ "$GAP" -lt "$MIN_ROUNDS_GAP" ] && { log "gap $GAP<$MIN_ROUNDS_GAP -> skip"; exit 0; }

# 2. HARVEST best-per-kernel WIN-commits from the card branches
python3 - "$MAIN" "$BASELINE_REF" "$CARDS" <<'PY' > /tmp/rollv2_pending.tsv 2>/dev/null
import subprocess,re,sys,os
MAIN,REF=sys.argv[1],sys.argv[2]
best={}
for c in sys.argv[3].split():
    wt=f"{MAIN}/.aw/sw_card{c}"
    try: lg=subprocess.check_output(["git","-C",wt,"log",f"{REF}..loop/card{c}","--format=%H %s"],text=True,stderr=subprocess.DEVNULL)
    except Exception: continue
    for line in lg.strip().splitlines():
        if not line.strip(): continue
        h,subj=line.split(" ",1)
        m=re.match(r"WIN\s+(\S+)\s+(\S+)\s+f=([\d.]+)\s+d=([\-\d.]+)%",subj)
        if not m: continue
        label,kern,f,d=m.group(1),m.group(2),float(m.group(3)),float(m.group(4))
        if kern not in best or d>best[kern][0]: best[kern]=(d,h,wt,label)
tot=0.0
for k,(d,h,wt,label) in sorted(best.items(),key=lambda x:-x[1][0]):
    try: src=subprocess.check_output(["git","-C",wt,"show",f"{h}:kernels/src/{k}.hip"],text=True,stderr=subprocess.DEVNULL)
    except Exception: continue
    hp=f"/tmp/rollv2_{k}.hip"; open(hp,"w").write(src)
    print(f"{d}\t{k}\t{label}\t{hp}"); tot+=d
print(f"NAIVE_SUM={tot:.2f} NKERN={len(best)}",file=sys.stderr)
PY
NAIVE=$(awk '{s+=$1} END{printf "%.2f", s+0}' /tmp/rollv2_pending.tsv)
NKERN=$(grep -c . /tmp/rollv2_pending.tsv 2>/dev/null || echo 0)
log "baseline=$CUR_SHA  harvested=$NKERN kernels from card branches  naive_sum=${NAIVE}% (pre-filter >= ${MIN_NAIVE_SUM}%)"
awk -F'\t' '{printf "    +%.2f%%  %-36s %s\n",$1,$2,$3}' /tmp/rollv2_pending.tsv
[ "$NKERN" -ge 1 ] || { log "no standing wins on branches -> skip"; exit 0; }
awk "BEGIN{exit !($NAIVE >= $MIN_NAIVE_SUM)}" || { log "below pre-filter -> skip"; exit 0; }

# 3. COMPOSE + gate (build baseline + composed on sw_card0, 6-round MWU + coherence)
WT="$MAIN/.aw/sw_card0"; export PATH="$HOME/.bun/bin:$PATH"
export HIPFIRE_DAEMON_ID=rollv2 CARGO_TARGET_DIR="$WT/target"; SW="$WT/.swhome"; mkdir -p "$SW/.hipfire"
SCLK=/sys/class/drm/card0/device/pp_dpm_sclk
cd "$WT" || { log "no worktree"; exit 1; }
git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
DB=target/release/examples/daemon
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
measure(){ local bin=$1 out=/tmp/rollv2m.jsonl clkf=/tmp/rollv2m.clk
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
print(f"{dec if dec is not None else 0:.2f} {'OK' if (len(t)>15 and u>0.35) else 'BAD'}")
PY
}
build || { log "baseline build FAIL"; exit 1; }
cp "$DB" /tmp/rollv2_base
while IFS=$'\t' read -r d k lbl hp; do [ -n "$k" ] && cp "$hp" "kernels/src/${k}.hip"; done < /tmp/rollv2_pending.tsv
if ! build; then git checkout "$BASELINE_REF" -- kernels/src/; log "COMPOSED build FAIL -> abort"; exit 1; fi
cp "$DB" /tmp/rollv2_var; git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null
BASE=();VAR=();BC=OK;VC=OK
for r in $(seq 1 6); do
  read d c < <(measure /tmp/rollv2_base); BASE+=("$d");[ "$c" = BAD ]&&BC=BAD
  read d c < <(measure /tmp/rollv2_var);  VAR+=("$d"); [ "$c" = BAD ]&&VC=BAD
done
read CDELTA CF BMED VMED < <(python3 - "${BASE[*]}" "${VAR[*]}" <<'PY'
import sys,statistics as st
b=[float(x) for x in sys.argv[1].split()];v=[float(x) for x in sys.argv[2].split()]
bm=st.median(b);vm=st.median(v);delta=100*(vm-bm)/bm if bm else 0
n=len(v)*len(b); f=sum((1.0 if x>y else .5 if x==y else 0) for x in v for y in b)/n if n else .5
print(f"{delta:.2f} {f:.3f} {bm:.2f} {vm:.2f}")
PY
)
log "COMPOSED A/B: base=$BMED var=$VMED delta=${CDELTA}% f=$CF coh=$BC/$VC"
GATE=$(awk "BEGIN{print (($CDELTA>=$MIN_COMPOSED)&&(\"$BC\"==\"OK\")&&(\"$VC\"==\"OK\")&&($CF>=0.85))?1:0}")
[ "$GATE" != 1 ] && { log "GATE FAIL -> no rollover"; exit 0; }
[ "$DRY_RUN" = 1 ] && { log "GATE PASS +${CDELTA}% -> DRY-RUN, not advancing"; exit 0; }

# 4. ADVANCE: commit fold, advance BASELINE_REF, RESET all card branches to it
cd "$WT"; git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
git checkout -q "$BASELINE_REF" 2>/dev/null
while IFS=$'\t' read -r d k lbl hp; do [ -n "$k" ] && cp "$hp" "kernels/src/${k}.hip"; done < /tmp/rollv2_pending.tsv
git add kernels/src/
NEWN=$(( $(awk -F'baseline_v' '/baseline_v/{split($2,a," ");n=a[1]} END{print n+0}' "$MANIFEST" 2>/dev/null) + 1 ))
FOLDED_KERNS=$(awk -F'\t' '{printf "%s,",$2}' /tmp/rollv2_pending.tsv)
git -c user.email=151092359+Kaden-Schutt@users.noreply.github.com -c user.name="Kaden Schutt" commit -q -m "rollover_v2 baseline_v$NEWN (+${CDELTA}% composed) [$FOLDED_KERNS] round $CUR_ROUND"
NEW_SHA=$(git rev-parse --short HEAD)
git branch -f "$BASELINE_REF" HEAD 2>/dev/null
# reset every card branch to the new baseline (fresh start; provisional wins were harvested)
MIS=""
for c in $CARDS; do
  git -C "$MAIN/.aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null; git -C "$MAIN/.aw/sw_card$c" clean -fdq kernels/src/ 2>/dev/null
  git -C "$MAIN/.aw/sw_card$c" checkout -q -B "loop/card$c" "$NEW_SHA" 2>/dev/null
  h=$(git -C "$MAIN/.aw/sw_card$c" rev-parse --short HEAD); [ "$h" = "$NEW_SHA" ] || MIS="$MIS card$c=$h"
done
[ -n "$MIS" ] && log "WARNING: branch reset mismatch ($MIS)"
NOW_EPOCH=$(date +%s)
echo "baseline_v$NEWN sha=$NEW_SHA rolled_at_round=$CUR_ROUND rolled_at_epoch=$NOW_EPOCH composed_delta=$CDELTA new_tok_s=$VMED folded=$FOLDED_KERNS $(date -u +%FT%TZ)" >> "$MANIFEST"
awk -F'\t' '{print $2}' /tmp/rollv2_pending.tsv >> "$FOLDED_LIST"; sort -u "$FOLDED_LIST" -o "$FOLDED_LIST" 2>/dev/null
log "ROLLED -> baseline_v$NEWN sha=$NEW_SHA new_tok_s=$VMED  folded: $FOLDED_KERNS"

# 5. RE-CENSUS + ROOFLINE-CEILING readout
log "re-censusing baseline_v$NEWN..."
cp /tmp/rollv2_var "$WT/target/release/examples/daemon" 2>/dev/null
if HIPFIRE_REPO="$WT" timeout 1400 bash /tmp/oracle_profile.sh "$ARCH" 0 0 "$MODEL" 40 > /tmp/bod_new.json 2>/dev/null \
   && [ -s /tmp/bod_new.json ] && python3 -c "import json,sys;d=json.load(open('/tmp/bod_new.json'));sys.exit(0 if d.get('rows') else 1)" 2>/dev/null; then
  cp "$BOD_OUT" "${BOD_OUT}.v$((NEWN-1))" 2>/dev/null; cp /tmp/bod_new.json "$BOD_OUT"
  python3 - "$BOD_OUT" "$VMED" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); cur=float(sys.argv[2]); rows=d.get("rows",[])
# computed roofline ceiling: per-kernel min-time from measured utilization (accurate L2/DRAM/CU mix).
# util proxy: mem_busy for mem-bound; occ-scaled otherwise. target = 90% util.
tot_wall=sum(r.get("wall_pct",0) for r in rows) or 1
speedup=0.0
for r in rows:
  w=r.get("wall_pct",0)/tot_wall
  mb=(r.get("mem_busy") or 0)/100.0; occ=(r.get("occ") or 0)/100.0
  util=max(mb, occ, 0.05)                       # whichever binds; floor to avoid div-blowup
  headroom=min(0.90/util, 3.0) if util>0 else 1 # cap 3x per kernel
  speedup += w*headroom
ceiling=cur*speedup
print(f"[rollover_v2] ROOFLINE-CEILING est {ceiling:.0f} tok/s (current {cur:.0f} = {100*cur/ceiling:.0f}% of ceiling); top hot: "+", ".join(f"{x['kernel'][:20]}({round(x.get('wall_pct',0),1)}%)" for x in rows[:4]))
PY
else log "WARNING: re-census failed -> keeping prior bod"; fi

#!/usr/bin/env bash
# ab_certify_v2.sh <arch> <dev> <card> <model> <kernel> <label> <variant.hip>
# Harness-v2 certify. Core change vs v1: ADAPTIVE SAMPLING instead of fixed 4/8 rounds +
# magnitude floor. Sample until the MWU dominance f decisively resolves, so real-but-noisy
# 2-3% marginals get captured and true noise is rejected. Wins commit to a per-card
# wins-only branch (loop/card<N>); everything logs to the JSONL. NO fold until rollover.
#
# Verdict:  f>=0.90 & delta>FLOOR & gated -> WIN
#           f<=0.65 (or delta<-FLOOR & f<=0.35) & gated -> DEAD
#           0.65<f<0.90 at CAP rounds -> INCONCLUSIVE (parked; re-openable on rerun)
#           var_coh=BAD -> COHERENCE_FAIL ;  clock skew -> VOID   (both regardless of f)
set -u
ARCH=$1 DEV=$2 CARD=$3 MODEL=$4 KERNEL=$5 LABEL=$6 VARIANT=$7
MAIN=~/hipfire; WT="${WT_OVERRIDE:-$MAIN/.aw/sw_card${CARD}}"   # WT_OVERRIDE = per-worker worktree for the swarm (parallel builds must not share kernels/src+target)
export PATH=$HOME/.bun/bin:$PATH
export HOME="$WT/.swhome"; mkdir -p "$HOME/.hipfire"
export HIPFIRE_DAEMON_ID="sw_card${CARD}"; ID="$HIPFIRE_DAEMON_ID"
export CARGO_TARGET_DIR="$WT/target"
FLOOR=0.15                # just above clock-noise; f carries the reliability
CAP=16                    # max A/B rounds before parking INCONCLUSIVE
WIN_F=0.90 ; DEAD_F=0.65  # resolve band edges
BASELINE_REF="${BASELINE_REF:-loop_baseline_gfx1201}"   # env-overridable (smoke tests use an isolated ref)
CARD_BRANCH="loop/card${CARD}"
SCLK=/sys/class/drm/card${CARD}/device/pp_dpm_sclk
KSRC="kernels/src/${KERNEL}.hip"
LEDGER="$MAIN/autoresearch/ledger/swarm_${ARCH}_${KERNEL}.jsonl"
cd "$WT" || { echo "{\"label\":\"$LABEL\",\"error\":\"no worktree $WT\"}"; exit 1; }
# reset the target kernel to the CURRENT baseline (branch may carry prior card-wins in history)
git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null || git checkout -- kernels/src/ 2>/dev/null
git clean -fdq kernels/src/ 2>/dev/null
DB=target/release/examples/daemon
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
measure(){ local bin=$1 out=/tmp/v2m_${ID}.jsonl clkf=/tmp/v2m_${ID}.clk
  rm -f .hipfire_kernels/*/"${KERNEL}".hsaco "$clkf" 2>/dev/null
  ( for _ in $(seq 1 40); do grep '\*' "$SCLK" 2>/dev/null|grep -oiE "[0-9]+Mhz"|grep -oE "[0-9]+"|head -1; sleep 0.4; done > "$clkf" ) & local s=$!
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w1","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"w2","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m1","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HIP_VISIBLE_DEVICES=$DEV "$bin" 2>/dev/null > "$out"
  kill "$s" 2>/dev/null; wait "$s" 2>/dev/null
  python3 - "$out" "$clkf" <<'PY'
import json,sys
dec=None;m1=""
for l in open(sys.argv[1]):
 try:d=json.loads(l)
 except:continue
 if d.get("type")=="done" and d.get("id")=="m1":dec=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or dec
 if d.get("type")=="token" and d.get("id")=="m1":m1+=d.get("text","")
t=m1.split();u=(len(set(t))/len(t)) if t else 0
try:c=[int(x) for x in open(sys.argv[2]) if x.strip()]
except:c=[]
print(f"{dec if dec is not None else 0:.2f} {'OK' if (len(t)>15 and u>0.35) else 'BAD'} {max(c) if c else 0}")
PY
}
# build baseline + variant (keep both binaries). PREBUILT_BASE lets the swarm build ONE baseline daemon and
# share it across all parallel variant-certifies (the baseline is identical per campaign) — no N× rebuild.
if [ -n "${PREBUILT_BASE:-}" ] && [ -s "${PREBUILT_BASE}" ]; then
  cp "$PREBUILT_BASE" /tmp/v2_base_$ID
else
  build || { echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"BASELINE_BUILD_FAIL\"}"; exit 0; }
  cp "$DB" /tmp/v2_base_$ID
fi
cp "$VARIANT" "$KSRC"
if ! build; then git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"VARIANT_BUILD_FAIL\"}"; exit 0; fi
cp "$DB" /tmp/v2_var_$ID
git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
# ---- GPU RUN-QUEUE: serialize measure+profile across parallel certifies (flock) so concurrent A/Bs don't
#      thrash the GPU and produce garbage tok/s (the builds above ran UNLOCKED = parallel/CPU, the bottleneck).
#      Reentrant within a process tree + kernel auto-releases on death. This is what makes a build-parallel /
#      measure-serial swarm safe to fan out onto one card.
GPULK="$MAIN/scripts/gpu-lock.sh"; [ -f "$GPULK" ] && { . "$GPULK"; gpu_acquire "cert_${ARCH}_c${CARD}_${LABEL}" >/dev/null 2>&1; }
# ---- PARITY GATE: candidate token-ids MUST equal baseline at temp0 (catches coherent-but-wrong .hip changes) ----
pids(){ rm -f .hipfire_kernels/*/*.hsaco 2>/dev/null; printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"p","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":48}
{"type":"unload"}
' "$MODEL" | HIPFIRE_EMIT_TOKEN_IDS=1 HIP_VISIBLE_DEVICES=$DEV "$1" 2>/dev/null | python3 -c "import json,sys;print(','.join(str(json.loads(l).get('tok_id')) for l in sys.stdin if chr(34)+'committed'+chr(34) in l and chr(34)+'p'+chr(34) in l))"; }
PB=$(pids /tmp/v2_base_$ID); PC=$(pids /tmp/v2_var_$ID)
if [ -z "$PB" ] || [ "$PB" != "$PC" ]; then [ -f "$GPULK" ] && gpu_release >/dev/null 2>&1; echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"PARITY_FAIL\"}"; exit 0; fi
# ---- ADAPTIVE SAMPLING ----
BASE=();VAR=();BK=();VK=();BC=OK;VC=OK
sample(){ local n=$1 r d c k; for r in $(seq 1 "$n"); do
  read d c k < <(measure /tmp/v2_base_$ID); BASE+=("$d");BK+=("$k");[ "$c" = BAD ]&&BC=BAD
  read d c k < <(measure /tmp/v2_var_$ID);  VAR+=("$d"); VK+=("$k");[ "$c" = BAD ]&&VC=BAD
done; }
fstats(){ python3 - "${BASE[*]}" "${VAR[*]}" <<'PY'
import sys,statistics as st
b=[float(x) for x in sys.argv[1].split()];v=[float(x) for x in sys.argv[2].split()]
n=len(v)*len(b); f=sum((1.0 if x>y else .5 if x==y else 0) for x in v for y in b)/n if n else .5
bm=st.median(b);vm=st.median(v);d=100*(vm-bm)/bm if bm else 0
print(f"{f:.4f} {d:.3f} {bm:.2f} {vm:.2f}")
PY
}
sample 4
read F DELTA BM VM < <(fstats)
while awk "BEGIN{exit !($F>$DEAD_F && $F<$WIN_F)}" && [ "${#VAR[@]}" -lt "$CAP" ]; do
  sample 2; read F DELTA BM VM < <(fstats)
done
ROUNDS=${#VAR[@]}
# ---- gating + verdict ----
BCK=$(python3 -c "import statistics as s;a=[$(IFS=,;echo "${BK[*]}")];print(int(s.median(a)) if a else 0)" 2>/dev/null)
VCK=$(python3 -c "import statistics as s;a=[$(IFS=,;echo "${VK[*]}")];print(int(s.median(a)) if a else 0)" 2>/dev/null)
CLK_OK=$(awk "BEGIN{print (($BCK>0 && $VCK>0)? (( ($BCK>$VCK?$BCK-$VCK:$VCK-$BCK)/$BCK*100 < 4.0)?1:0 ) : 1)}")
if [ "$VC" = BAD ] || [ "$BC" = BAD ]; then VERDICT=COHERENCE_FAIL
elif [ "$CLK_OK" != 1 ]; then VERDICT=VOID
elif awk "BEGIN{exit !($F>=$WIN_F && $DELTA>$FLOOR)}"; then VERDICT=WIN
elif awk "BEGIN{exit !($F<=$DEAD_F || ($DELTA< -$FLOOR && $F<=0.35))}"; then VERDICT=DEAD
else VERDICT=INCONCLUSIVE; fi
# ---- WIN -> commit to the per-card wins-only branch ----
COMMITTED=""
if [ "$VERDICT" = WIN ]; then
  git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
  cp "$VARIANT" "$KSRC"; git add "$KSRC" 2>/dev/null
  git -c user.email=151092359+Kaden-Schutt@users.noreply.github.com -c user.name="Kaden Schutt" \
    commit -q -m "WIN $LABEL $KERNEL f=$F d=${DELTA}% rounds=$ROUNDS" 2>/dev/null && COMMITTED=$(git rev-parse --short HEAD)
  git checkout "$BASELINE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
fi
# ---- per-variant PROFILE on EVERY verdict (so the agent sees WHY it won/lost + what to try next,
#      not a bare scalar — this is what stops it flying blind under the 5-run permutation budget) ----
BP="null"; VP="null"
if [ "$VERDICT" != "BASELINE_BUILD_FAIL" ] && [ "$VERDICT" != "VARIANT_BUILD_FAIL" ]; then
  rp(){ cp "$1" "$DB"; HIPFIRE_REPO="$WT" bash /tmp/oracle_profile.sh "$ARCH" "$DEV" "$CARD" "$MODEL" 24 2>/dev/null | tail -1; }
  BPC="/tmp/baseprof_${ARCH}_$(echo "$KERNEL" | tr / _)"   # base profiled ONCE per kernel, cached + reused across its variants
  [ -s "$BPC" ] || rp /tmp/v2_base_$ID > "$BPC"
  BP=$(cat "$BPC"); VP=$(rp /tmp/v2_var_$ID)
fi
[ -f "$GPULK" ] && gpu_release >/dev/null 2>&1   # GPU phase done -> release the run-queue for the next queued certify
rm -f /tmp/v2_base_$ID /tmp/v2_var_$ID
mkdir -p "$(dirname "$LEDGER")"
python3 - "$ARCH" "$KERNEL" "$LABEL" "$(basename "$VARIANT")" "$VERDICT" "$ROUNDS" "$F" "$DELTA" "$BM" "$VM" "$BC" "$VC" "$COMMITTED" "$LEDGER" "${BASE[*]}" "${VAR[*]}" "$BP" "$VP" <<'PY'
import sys,json
(arch,kern,label,variant,verdict,rounds,f,delta,bm,vm,bc,vc,committed,ledger,bs,vs,bp,vp)=sys.argv[1:19]
def prow(p):
  # match the profile row by KERNEL: the arg is the SOURCE-FILE base name, but rocprof reports the RUNTIME
  # SYMBOL name, and they can diverge after the core (gate_up file "..._indexed_batched" vs symbol
  # "..._k8_indexed"). So: exact/prefix-either-direction wins outright; else best LEADING-TOKEN overlap (>=4
  # core tokens, so moe_gate_up != moe_down which split at token 4). This is the "no profile" fix.
  try: d=json.loads(p)
  except: return None
  kt=kern.split("_"); best=None; bestn=0
  for r in d.get("rows",[]):
    rk=r.get("kernel",""); rt=rk.split("_"); n=0
    for a,b in zip(kt,rt):
      if a==b: n+=1
      else: break
    if rk==kern or rk.startswith(kern) or kern.startswith(rk): n=999
    if n>bestn: bestn=n; best=r
  if best and bestn>=4:
    return {x:best.get(x) for x in ("wall_pct","occ","l2_hit_pct","mem_busy","vgpr","lds")}
  return None
b=prow(bp); v=prow(vp)
roof={"target_base":b,"target_var":v} if (b or v) else None
# ANNOTATE per the instrument->lever map (autoresearch/levers/): tell the agent WHY + what to try next
fb=[]
if b and v:
  bo,vo=b.get("occ"),v.get("occ"); bg,vg=b.get("vgpr"),v.get("vgpr"); vl=v.get("l2_hit_pct"); vmb=v.get("mem_busy")
  if bg is not None and vg is not None and vg>bg+2: fb.append(f"VGPR {bg}->{vg}")
  if bo and vo and vo<bo*0.85: fb.append(f"occ {bo}->{vo}% DROPPED-WAVES(you overloaded=the row-reuse regression mode; cut registers/use global_load_lds)")
  if vo is not None and vo<5: fb.append(f"occ {vo}% CATASTROPHIC-UNDEROCC(per-wave MLP ring is the lever here)")
  elif vmb is not None and vo is not None and vmb<70 and vo>=20: fb.append(f"mem_busy {vmb}%+occ {vo}%=TLP already hides latency, NOT MLP-limited")
  if vl is not None and vl<30: fb.append(f"L2-hit {vl}% CACHE-NOT-FIXED(DRAM-thrash; SLC/stream the write-once weights, keep x/KV resident)")
  if vo is not None and vmb is not None and vo>85 and vmb>85: fb.append("SATURATED at roofline-leave it")
feedback="; ".join(fb) if fb else ("no profile" if not v else "no clear lever signal")
rec={"arch":arch,"kernel":kern,"label":label,"variant":variant,"verdict":verdict,
 "WIN":verdict=="WIN","mwu_dominance":round(float(f),3),"rounds":int(rounds),
 "base_decode":round(float(bm),2),"var_decode":round(float(vm),2),"delta_pct":round(float(delta),2),
 "base_coh":bc,"var_coh":vc,"win_commit":committed or None,"roofline":roof,"profile_feedback":feedback,
 "base_runs":[float(x) for x in bs.split()],"var_runs":[float(x) for x in vs.split()]}
open(ledger,"a").write(json.dumps(rec)+"\n")
out={k:rec[k] for k in ("label","verdict","mwu_dominance","delta_pct","rounds","var_coh","win_commit")}
out["profile_feedback"]=feedback; out["target_var"]=v
print(json.dumps(out))
PY

#!/usr/bin/env bash
# ab_certify_swarm.sh <arch> <dev> <card> <model> <kernel> <label> <variant.hip>
# Parallel-safe A/B for the fleet swarm. Worktree-isolated + isolated HOME (daemon.pid
# never collides across cards). auto pre-set by the driver.
# ACCEPTANCE = Mann-Whitney dominance f = P(var>base over all pairs), NOT strict
# min>max (which buried marginal wins like r2lb, f=0.84, on a single overlap):
#   f>=0.90 & delta>floor  -> WIN
#   f<=0.10 & delta<-floor -> LOSS
#   0.75<=f<0.90 & delta>floor -> MARGINAL -> run 4 MORE rounds, re-adjudicate (f>=0.85 -> WIN else NOISE)
#   else -> NOISE
# ROOFLINE (oracle_profile: VGPR/occ/L2/wall) on WIN or confirmed-marginal = the WHY.
# git-reverts the kernel. Emits ONE verdict JSON + appends to the swarm ledger.
set -u
ARCH=$1 DEV=$2 CARD=$3 MODEL=$4 KERNEL=$5 LABEL=$6 VARIANT=$7
MAIN=~/hipfire; WT="$MAIN/.aw/sw_card${CARD}"
export PATH=$HOME/.bun/bin:$PATH
export HOME="$WT/.swhome"; mkdir -p "$HOME/.hipfire"
export HIPFIRE_DAEMON_ID="sw_card${CARD}"; ID="$HIPFIRE_DAEMON_ID"
export CARGO_TARGET_DIR="$WT/target"
NOISE=0.3
SCLK=/sys/class/drm/card${CARD}/device/pp_dpm_sclk
KSRC="kernels/src/${KERNEL}.hip"
LEDGER="$MAIN/autoresearch/ledger/swarm_${ARCH}_${KERNEL}.jsonl"
cd "$WT" || { echo "{\"label\":\"$LABEL\",\"error\":\"no worktree $WT\"}"; exit 1; }
git checkout -- "$KSRC" 2>/dev/null
DB=target/release/examples/daemon
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
measure(){ local bin=$1 out=/tmp/swm_${ID}.jsonl clkf=/tmp/swm_${ID}.clk
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
build || { echo "{\"label\":\"$LABEL\",\"verdict\":\"BASELINE_BUILD_FAIL\"}"; exit 0; }
cp "$DB" /tmp/sw_base_$ID
cp "$VARIANT" "$KSRC"
if ! build; then git checkout -- "$KSRC"; echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"VARIANT_BUILD_FAIL\"}"; exit 0; fi
cp "$DB" /tmp/sw_var_$ID
git checkout -- "$KSRC"
BASE=();VAR=();BK=();VK=();BC=OK;VC=OK
run_rounds(){ local n=$1 r d c k; for r in $(seq 1 "$n"); do
  read d c k < <(measure /tmp/sw_base_$ID); BASE+=("$d");BK+=("$k");[ "$c" = BAD ]&&BC=BAD
  read d c k < <(measure /tmp/sw_var_$ID);  VAR+=("$d"); VK+=("$k");[ "$c" = BAD ]&&VC=BAD
done; }
adjudicate(){ local winf=$1; python3 - "$NOISE" "$winf" "${BASE[*]}" "${VAR[*]}" "${BK[*]}" "${VK[*]}" "$BC" "$VC" <<'PY'
import sys,statistics as st
noise=float(sys.argv[1]); winf=float(sys.argv[2])
base=[float(x) for x in sys.argv[3].split()];var=[float(x) for x in sys.argv[4].split()]
bk=[int(x) for x in sys.argv[5].split() if x];vk=[int(x) for x in sys.argv[6].split() if x]
bc,vc=sys.argv[7],sys.argv[8]
bmed=st.median(base);vmed=st.median(var);delta=100*(vmed-bmed)/bmed if bmed else 0
bck=st.median(bk) if bk else 0;vck=st.median(vk) if vk else 0
clk_ok=not(bck and vck and abs(bck-vck)/max(bck,vck)*100>=4.0)
gated=vc=="OK" and bc=="OK" and clk_ok
n=len(var)*len(base)
f=(sum((1.0 if x>y else 0.5 if x==y else 0.0) for x in var for y in base)/n) if n else 0.5
if f>=winf and delta>noise and gated: print("WIN 0")
elif f<=(1-winf) and delta<-noise and gated: print("LOSS 0")
elif 0.75<=f<winf and delta>noise and gated: print("NOISE 1")   # marginal -> CONFIRM
else: print("NOISE 0")
PY
}
run_rounds 4
read VERDICT MARGINAL < <(adjudicate 0.90)
CONFIRMED=false
if [ "$MARGINAL" = 1 ]; then run_rounds 4; read VERDICT MARGINAL < <(adjudicate 0.85); CONFIRMED=true; fi
ROOF="null"
if [ "$VERDICT" = WIN ] || [ "$CONFIRMED" = true ]; then
  rp(){ cp "$1" "$DB"; HIPFIRE_REPO="$WT" bash /tmp/oracle_profile.sh "$ARCH" "$DEV" "$CARD" "$MODEL" 24 2>/dev/null | tail -1; }
  BP=$(rp /tmp/sw_base_$ID); VP=$(rp /tmp/sw_var_$ID)
  ROOF=$(python3 - "$KERNEL" "$BP" "$VP" <<'PY'
import json,sys
k=sys.argv[1]
def row(p):
  try:d=json.loads(p)
  except:return None
  for r in d.get("rows",[]):
    if r["kernel"].startswith(k):return {x:r.get(x) for x in("wall_pct","occ","l2_hit_pct","mem_busy","vgpr","lds","roofline")}
print(json.dumps({"target_base":row(sys.argv[2]),"target_var":row(sys.argv[3])}))
PY
)
fi
rm -f /tmp/sw_base_$ID /tmp/sw_var_$ID
mkdir -p "$(dirname "$LEDGER")"
python3 - "$ARCH" "$KERNEL" "$LABEL" "$(basename "$VARIANT")" "$VERDICT" "$CONFIRMED" "$BC" "$VC" "$ROOF" "$LEDGER" "${BASE[*]}" "${VAR[*]}" "${BK[*]}" "${VK[*]}" <<'PY'
import sys,json,os,statistics as st
arch,kern,label,variant,verdict,confirmed,bc,vc,roof,ledger,bs,vs,bk,vk=sys.argv[1:15]
base=[float(x) for x in bs.split()];var=[float(x) for x in vs.split()]
bk=[int(x) for x in bk.split() if x];vk=[int(x) for x in vk.split() if x]
bmed=st.median(base);vmed=st.median(var);delta=100*(vmed-bmed)/bmed if bmed else 0
n=len(var)*len(base); f=(sum((1.0 if x>y else 0.5 if x==y else 0.0) for x in var for y in base)/n) if n else 0.5
try: rf=json.loads(roof) if roof!="null" else None
except: rf=None
rec={"arch":arch,"kernel":kern,"label":label,"variant":variant,"verdict":verdict,"WIN":verdict=="WIN",
 "confirmed":confirmed=="true","mwu_dominance":round(f,3),"base_decode":round(bmed,2),"var_decode":round(vmed,2),
 "delta_pct":round(delta,2),"rounds":len(base),"base_runs":base,"var_runs":var,
 "base_sclk":int(st.median(bk)) if bk else 0,"var_sclk":int(st.median(vk)) if vk else 0,
 "base_coh":bc,"var_coh":vc,"roofline":rf}
open(ledger,"a").write(json.dumps(rec)+"\n")
print(json.dumps(rec))
PY

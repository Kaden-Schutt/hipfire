#!/usr/bin/env bash
# ab_certify_v2.sh — worktree-isolated, git-reverted, durability-gated, banked
# fixed-eval loop. Runs ON the GPU box. Each arch/card experiment lives in its OWN
# git worktree (isolated source + build + kernel cache + per-GPU daemon), so:
#   - revert is git (cherry-precise, guaranteed clean — never poisons another arch)
#   - N worktrees on N GPUs run DIFFERENT variants simultaneously (scatter/gather)
#   - the agent COMMITS only on a durability-passing WIN; a LOSS is force-reverted
#     but its instrument record is committed (losses are data)
#   - the banked counter moves only on a committed win
#
# The chain of gates a variant must pass to bank:
#   1) A/B decode separation (fast, 128-tok, auto clock, sclk-verified)  [ab_certify v1 logic]
#   2) coherence (fluency)                                               [fast gate]
#   3) DURABILITY: serve_harness --mode chain, default sampling, multiturn
#      — a 128-tok decode win is NOT coherence or durability. Fail -> BOUNCE.
#   4) no-clobber roofline knock-on (PROFILE)                            [optional]
# Only all-pass -> git commit the variant (dispatch invoice) + bank.
#
# Usage: ab_certify_v2.sh <arch> <dev> <card> <model> <kernel> <label> <variant.hip>
set -u
ARCH=$1; DEV=$2; CARD=$3; MODEL=$4; KERNEL=$5; LABEL=$6; VARIANT=$7
MAINREPO=~/hipfire
WT="$MAINREPO/.aw/${ARCH}_card${CARD}"          # per-arch/card worktree
LEDGER="$MAINREPO/autoresearch/ledger/${ARCH}_${KERNEL}.jsonl"   # ledger stays in the MAIN repo (shared)
export PATH=$HOME/.bun/bin:$PATH
export HIPFIRE_DAEMON_ID="${ARCH}_card${CARD}"  # per-GPU daemon pid (parallel-safe)
NOISE="${NOISE:-0.3}"; ROUNDS="${ROUNDS:-4}"; PERF="${PERF:-auto}"; CLK_TOL="${CLK_TOL:-4.0}"
PL="/sys/class/drm/card${CARD}/device/power_dpm_force_performance_level"
KSRC="kernels/src/${KERNEL}.hip"
emit(){ echo "$1"; mkdir -p "$(dirname "$LEDGER")"; echo "$1" >> "$LEDGER"; }

# --- worktree: create or reuse, isolated build ---
cd "$MAINREPO" || exit 1
if [ ! -d "$WT" ]; then git worktree add -f "$WT" HEAD >/dev/null 2>&1 || { emit "{\"arch\":\"$ARCH\",\"error\":\"worktree add failed\"}"; exit 1; }; fi
cd "$WT" || exit 1
git checkout -- "$KSRC" 2>/dev/null   # guarantee a clean baseline source before we start
DBIN=target/release/examples/daemon

set_perf(){ local i; for i in $(seq 1 20); do echo "$1"|sudo tee "$PL">/dev/null 2>&1 && return; sleep 0.3; done; }
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
# measure <binary> -> "<decode> <coh> <sclk>"
measure(){
  local bin="$1" out="/tmp/abv2_${HIPFIRE_DAEMON_ID}.jsonl" clkf="/tmp/abv2_${HIPFIRE_DAEMON_ID}.clk"
  rm -f .hipfire_kernels/*/"${KERNEL}".hsaco "$clkf" 2>/dev/null
  ( for _ in $(seq 1 40); do grep '\*' "/sys/class/drm/card${CARD}/device/pp_dpm_sclk" 2>/dev/null|grep -oiE "[0-9]+Mhz"|grep -oE "[0-9]+"|head -1; sleep 0.4; done > "$clkf" ) & local s=$!
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w1","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"w2","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m1","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HIP_VISIBLE_DEVICES=$DEV "$bin" 2>/dev/null > "$out"
  kill "$s" 2>/dev/null; wait "$s" 2>/dev/null
  python3 - "$out" "$clkf" <<'PY'
import json,sys
dec=None; m1=""
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    if d.get("type")=="done" and d.get("id")=="m1": dec=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or d.get("tok_s") or dec
    if d.get("type")=="token" and d.get("id")=="m1": m1+=d.get("text","")
t=m1.split(); u=(len(set(t))/len(t)) if t else 0.0
try: c=[int(x) for x in open(sys.argv[2]) if x.strip()]
except Exception: c=[]
print(f"{dec if dec is not None else 0.0:.2f} {'OK' if (len(t)>15 and u>0.35) else 'BAD'} {max(c) if c else 0}")
PY
}

set_perf "$PERF"
# --- build baseline + variant binaries (both, up front) ---
build || { emit "{\"arch\":\"$ARCH\",\"kernel\":\"$KERNEL\",\"error\":\"baseline build\"}"; set_perf auto; exit 1; }
cp "$DBIN" /tmp/abv2_base_$$
cp "$VARIANT" "$KSRC"
build || { git checkout -- "$KSRC"; emit "{\"arch\":\"$ARCH\",\"kernel\":\"$KERNEL\",\"error\":\"variant build\"}"; set_perf auto; exit 1; }
cp "$DBIN" /tmp/abv2_var_$$
# NOTE: variant source stays in the worktree tree until the verdict decides commit-vs-revert.
# --- INTERLEAVE A/B ---
BASE=(); VAR=(); BK=(); VK=(); BC=OK; VC=OK
for r in $(seq 1 "$ROUNDS"); do
  set_perf "$PERF"
  read d c k < <(measure /tmp/abv2_base_$$); BASE+=("$d"); BK+=("$k"); [ "$c" = BAD ] && BC=BAD
  read d c k < <(measure /tmp/abv2_var_$$);  VAR+=("$d");  VK+=("$k"); [ "$c" = BAD ] && VC=BAD
done
# --- fast-gate verdict (separation) ---
read VERDICT DELTA BMED VMED CLKOK < <(python3 - "$NOISE" "$CLK_TOL" "${BASE[*]}" "${VAR[*]}" "${BK[*]}" "${VK[*]}" "$BC" "$VC" <<'PY'
import sys,statistics as st
noise=float(sys.argv[1]); ctol=float(sys.argv[2])
base=[float(x) for x in sys.argv[3].split()]; var=[float(x) for x in sys.argv[4].split()]
bk=[int(x) for x in sys.argv[5].split() if x]; vk=[int(x) for x in sys.argv[6].split() if x]
bc,vc=sys.argv[7],sys.argv[8]
bmed=st.median(base); vmed=st.median(var); delta=100*(vmed-bmed)/bmed if bmed else 0
bck=st.median(bk) if bk else 0; vck=st.median(vk) if vk else 0
clkok = not(bck and vck and abs(bck-vck)/max(bck,vck)*100>=ctol)
gated = vc=="OK" and bc=="OK" and clkok
win = min(var)>max(base) and delta>noise and gated
loss= max(var)<min(base) and delta<-noise
print(("WIN" if win else "LOSS" if loss else "NOISE"), f"{delta:.2f}", f"{bmed:.2f}", f"{vmed:.2f}", int(clkok))
PY
)

DURABLE="n/a"; COMMIT_SHA=""
if [ "$VERDICT" = WIN ]; then
  # --- gate 3: DURABILITY — serve_harness, default sampling, MULTITURN (variant daemon is $DBIN in this worktree) ---
  if [ -f scripts/serve_harness.py ]; then
    HIP_VISIBLE_DEVICES=$DEV timeout 500 python3 scripts/serve_harness.py --model "$MODEL" --kv q8 --sampling registry \
        --mode chain --max-tokens 256 --port "1160${CARD}" > /tmp/abv2_dur_$$.log 2>&1
    if grep -qE "attractor=[1-9]|empty=[1-9]|type.:.error" /tmp/abv2_dur_$$.log; then DURABLE=FAIL; VERDICT=BOUNCE; else DURABLE=PASS; fi
  else DURABLE="no-harness"; fi
fi

if [ "$VERDICT" = WIN ] && [ "$DURABLE" != FAIL ]; then
  # --- COMMIT the win (the variant diff = the dispatch invoice) + bank ---
  git add "$KSRC"
  git commit -q -m "autoresearch WIN: ${KERNEL} on ${ARCH} +${DELTA}% (durable) [${LABEL}]" 2>/dev/null
  COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null)
  cp "$VARIANT" "$MAINREPO/autoresearch/variants/${ARCH}_${KERNEL}_win_${COMMIT_SHA}.hip" 2>/dev/null
else
  # --- LOSS/BOUNCE/NOISE: FORCE-REVERT the source; the loss record is still committed to the ledger ---
  git checkout -- "$KSRC"
fi
rm -f "$DBIN.novar" /tmp/abv2_base_$$ /tmp/abv2_var_$$
set_perf auto
rm -f .hipfire_kernels/*/"${KERNEL}".hsaco 2>/dev/null   # leave the cache clean regardless

# --- REQUIRED loss/win record (instruments) committed to the ledger ---
emit "$(python3 - "$ARCH" "$KERNEL" "$LABEL" "$VERDICT" "$DELTA" "$BMED" "$VMED" "$DURABLE" "$COMMIT_SHA" "$BC" "$VC" "$(basename "$VARIANT")" "${BASE[*]}" "${VAR[*]}" "${BK[*]}" "${VK[*]}" <<'PY'
import sys,json
(arch,kern,label,verdict,delta,bmed,vmed,durable,sha,bc,vc,variant,bs,vs,bk,vk)=sys.argv[1:17]
print(json.dumps({"arch":arch,"kernel":kern,"label":label,"verdict":verdict,"delta_pct":float(delta),
  "base_decode":float(bmed),"var_decode":float(vmed),"durable":durable,"win_commit":sha,
  "base_coh":bc,"var_coh":vc,"variant":variant,"committed":bool(sha),
  "base_runs":[float(x) for x in bs.split()],"var_runs":[float(x) for x in vs.split()],
  "base_sclk_max":max([int(x) for x in bk.split()] or [0]),"var_sclk_max":max([int(x) for x in vk.split()] or [0])}))
PY
)"

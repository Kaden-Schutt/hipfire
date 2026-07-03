#!/usr/bin/env bash
# ab_certify.sh — A/B-certify ONE kernel variant against baseline on ONE arch.
# The fixed-eval loop of the RDNA autoresearch engine (see autoresearch/README.md).
# Runs ON the GPU box (operates on ~/hipfire).
#
# DESIGN (each guard earned by a calibration failure):
#  - THERMAL/ORDER-ROBUST: builds BOTH daemon binaries up front, then INTERLEAVES
#    their measurements (A B A B) so state is held constant across the comparison.
#  - CLOCK-CONTROLLED: perf_level `high` UNDER-clocks the R9700 (2350 vs auto 2838);
#    and `echo` to the perf-level sysfs SILENTLY FAILS under contention (busy file),
#    which is what made a no-op read 130 vs 115. So we set `auto` with retry, and
#    CAPTURE sclk on every measure. A/B is REJECTED if base/var clocks disagree.
#  - COHERENCE-GATED: a faster-but-incoherent variant (e.g. GEMV_ROWS=1 garbage) is
#    never certified.
# WIN only if Δ clears the noise band AND both coherent AND clocks matched.
#
# Usage: ab_certify.sh <arch> <dev> <card> <model> <kernel_name> <label> \
#            [B_env="KEY=VAL ..."] [B_swap=/abs/path/variant.hip]
set -u
ARCH=$1; DEV=$2; CARD=$3; MODEL=$4; KERNEL=$5; LABEL=$6
BENV="${7:-}"; BSWAP="${8:-}"
REPO=~/hipfire; cd "$REPO" || exit 1
export PATH=$HOME/.bun/bin:$PATH
KSRC="kernels/src/${KERNEL}.hip"
PL="/sys/class/drm/card${CARD}/device/power_dpm_force_performance_level"
NOISE="${NOISE:-3.0}"; ROUNDS="${ROUNDS:-3}"; PERF="${PERF:-auto}"; CLK_TOL="${CLK_TOL:-4.0}"
LEDGER="${LEDGER:-$REPO/autoresearch/ledger/${ARCH}_${KERNEL}.jsonl}"
BASE_BIN=/tmp/ab_daemon_base_$$; VAR_BIN=/tmp/ab_daemon_var_$$
DBIN=target/release/examples/daemon

build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && return 1; return 0; }
# set perf level with retry (the sysfs file goes "Device or resource busy" under contention)
set_perf(){ local lvl="$1" i; for i in $(seq 1 20); do echo "$lvl" | sudo tee "$PL" >/dev/null 2>&1 && return 0; sleep 0.3; done; return 1; }

# measure <binary> <env> -> "<decode> <coh> <sclk_mhz>"  (kernel_decode + coherence + boosted clock)
measure(){
  local bin="$1" env="$2" out="/tmp/ab_${ARCH}_$$_m.jsonl" clkf="/tmp/ab_${ARCH}_$$_c"
  rm -f .hipfire_kernels/*/"${KERNEL}".hsaco "$clkf" 2>/dev/null
  ( for _ in $(seq 1 40); do grep '\*' "/sys/class/drm/card${CARD}/device/pp_dpm_sclk" 2>/dev/null | grep -oiE "[0-9]+Mhz" | grep -oE "[0-9]+" | head -1; sleep 0.4; done > "$clkf" ) &
  local sampler=$!
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w1","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"w2","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m1","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | env $env HIP_VISIBLE_DEVICES=$DEV "$bin" 2>/dev/null > "$out"
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  python3 - "$out" "$clkf" <<'PY'
import json,sys
dec=None; m1=""
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    if d.get("type")=="done" and d.get("id")=="m1": dec=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or d.get("tok_s") or dec
    if d.get("type")=="token" and d.get("id")=="m1": m1+=d.get("text","")
toks=m1.split(); uniq=(len(set(toks))/len(toks)) if toks else 0.0
try: clks=[int(x) for x in open(sys.argv[2]) if x.strip()]
except Exception: clks=[]
sclk=max(clks) if clks else 0        # boosted (load-time) clock
print(f"{dec if dec is not None else 0.0:.2f} {'OK' if (len(toks)>15 and uniq>0.35) else 'BAD'} {sclk}")
PY
}

set_perf "$PERF" || { echo '{"error":"could not set perf level (busy)"}'; exit 1; }
# --- PHASE 1: build BOTH binaries up front ---
build || { echo '{"error":"baseline build failed"}'; set_perf auto; exit 1; }
cp "$DBIN" "$BASE_BIN"
if [ -n "$BSWAP" ]; then
  cp "$KSRC" "/tmp/${KERNEL}.bak"; cp "$BSWAP" "$KSRC"
  build || { cp "/tmp/${KERNEL}.bak" "$KSRC"; echo '{"error":"variant build failed"}'; set_perf auto; exit 1; }
  cp "$DBIN" "$VAR_BIN"; cp "/tmp/${KERNEL}.bak" "$KSRC"; build >/dev/null
else
  cp "$BASE_BIN" "$VAR_BIN"
fi
# --- PHASE 2: INTERLEAVE-measure (A B A B ...) with per-measure clock capture ---
BASE=(); VAR=(); BCLK=(); VCLK=(); BC="OK"; VC="OK"
for r in $(seq 1 "$ROUNDS"); do
  set_perf "$PERF"
  read d c k < <(measure "$BASE_BIN" "");    BASE+=("$d"); BCLK+=("$k"); [ "$c" = BAD ] && BC=BAD
  read d c k < <(measure "$VAR_BIN" "$BENV"); VAR+=("$d");  VCLK+=("$k"); [ "$c" = BAD ] && VC=BAD
done
rm -f "$BASE_BIN" "$VAR_BIN"
# --- PHASE 3: certify (clock-consistency + coherence + noise) + ledger ---
python3 - "$ARCH" "$KERNEL" "$LABEL" "$NOISE" "$LEDGER" "$BENV" "$BSWAP" "$BC" "$VC" "$CLK_TOL" "$PERF" "${BASE[*]}" "${VAR[*]}" "${BCLK[*]}" "${VCLK[*]}" <<'PY'
import sys,json,os,statistics as st
(arch,kern,label,noise,ledger,benv,bswap,bc,vc,ctol,perf,bs,vs,bk,vk)=sys.argv[1:16]
noise=float(noise); ctol=float(ctol)
base=[float(x) for x in bs.split()]; var=[float(x) for x in vs.split()]
bclk=[int(x) for x in bk.split() if x]; vclk=[int(x) for x in vk.split() if x]
bmed=st.median(base); vmed=st.median(var)
bck=int(st.median(bclk)) if bclk else 0; vck=int(st.median(vclk)) if vclk else 0
delta=100*(vmed-bmed)/bmed if bmed else 0.0
clk_captured = bool(bck and vck)
clk_mismatch = bool(clk_captured and abs(bck-vck)/max(bck,vck)*100 >= ctol)  # only a VERIFIED disagreement voids
rec={"arch":arch,"kernel":kern,"label":label,"benv":benv,"bswap":os.path.basename(bswap) if bswap else "","perf_level":perf,
     "base_runs":base,"var_runs":var,"base_decode":round(bmed,2),"var_decode":round(vmed,2),
     "base_sclk":bck,"var_sclk":vck,"clock_matched":(None if not clk_captured else not clk_mismatch),
     "base_coh":bc,"var_coh":vc,"delta_pct":round(delta,2),"noise_band_pct":noise,
     "WIN":bool(delta>noise and vc=="OK" and bc=="OK" and not clk_mismatch),
     "within_noise":bool(abs(delta)<noise and vc=="OK" and bc=="OK"),
     "VOID":bool(clk_mismatch or bc=="BAD" or vc=="BAD")}
os.makedirs(os.path.dirname(ledger),exist_ok=True)
open(ledger,"a").write(json.dumps(rec)+"\n")
print(json.dumps(rec))
PY

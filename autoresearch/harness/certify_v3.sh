#!/usr/bin/env bash
# certify_v3.sh — grade the CURRENT worktree (codex's committed candidate) vs a PRE-BUILT cached baseline daemon.
# NO git checkouts (never fights codex's live worktree). Codex commits its candidate (tree=candidate) then calls this.
# Guards: ARCH-BLEED (baseline gemm/gemv block literals must survive in the current tree) + warmed clock-gated
# 4-run A/B vs cached baseline + coherence. Prints ONE JSON verdict.
set -u
MAIN=~/hipfire; WT="$MAIN/.aw/sw_card0"; BASE_REF="${BASE_REF:-loop_baseline_gfx1100}"; DEV=0
MODEL="${MODEL:-/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r}"; BASE_DAEMON="${V3_BASE_DAEMON:-/tmp/v3_base_daemon}"
export HOME="$WT/.swhome"; mkdir -p "$HOME/.hipfire"; export CARGO_TARGET_DIR="$WT/target"; export PATH="$HOME/.bun/bin:$PATH"
cd "$WT" || { echo '{"verdict":"NO_WT"}'; exit 1; }
DB=target/release/examples/daemon; SCLK=/sys/class/drm/card0/device/pp_dpm_sclk; GPULK="$MAIN/scripts/gpu-lock.sh"
[ -s "$BASE_DAEMON" ] || { echo '{"verdict":"NO_BASELINE_DAEMON"}'; exit 1; }

# --- ARCH-BLEED: baseline block/grid literals (from BASE_REF via git show) must all still exist in the CURRENT tree files ---
BLEED=$(python3 - "$BASE_REF" <<'PY'
import subprocess,re,sys
base=sys.argv[1]; FS=("crates/rdna-compute/src/gemm.rs","crates/rdna-compute/src/gemv.rs")
def lits_ref(ref):
  s=set()
  for f in FS:
    try:t=subprocess.check_output(["git","show",f"{ref}:{f}"],text=True,stderr=subprocess.DEVNULL)
    except:continue
    for m in re.findall(r"\[\s*[0-9][0-9*u\s]*u32\s*,\s*1\s*,\s*1\s*\]",t):s.add(re.sub(r"\s","",m))
  return s
def lits_tree():
  s=set()
  for f in FS:
    try:t=open(f).read()
    except:continue
    for m in re.findall(r"\[\s*[0-9][0-9*u\s]*u32\s*,\s*1\s*,\s*1\s*\]",t):s.add(re.sub(r"\s","",m))
  return s
missing=lits_ref(base)-lits_tree()
print("BLEED "+";".join(sorted(missing))[:140] if missing else "OK")
PY
)
[[ "$BLEED" == BLEED* ]] && { echo "{\"verdict\":\"ARCH_BLEED_FAIL\",\"detail\":\"${BLEED#BLEED }\"}"; exit 0; }

# --- build the CURRENT tree (candidate); baseline is the cached daemon ---
cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | grep -qiE "^error|error\[" && { echo '{"verdict":"CANDIDATE_BUILD_FAIL"}'; exit 0; }
cp "$DB" /tmp/v3_cand

measure(){ local bin=$1 out=/tmp/v3m.jsonl clkf=/tmp/v3m.clk
  ( for _ in $(seq 1 30); do grep '\*' "$SCLK" 2>/dev/null|grep -oiE "[0-9]+Mhz"|grep -oE "[0-9]+"|head -1; sleep 0.4; done > "$clkf" ) & local s=$!
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HIP_VISIBLE_DEVICES=$DEV "$bin" 2>/dev/null > "$out"
  kill "$s" 2>/dev/null; wait "$s" 2>/dev/null
  python3 - "$out" "$clkf" <<'PY'
import json,sys
dec=None;m=""
for l in open(sys.argv[1]):
 try:d=json.loads(l)
 except:continue
 if d.get("type")=="done" and d.get("id")=="m":dec=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or dec
 if d.get("type")=="token" and d.get("id")=="m":m+=d.get("text","")
t=m.split();u=(len(set(t))/len(t)) if t else 0
try:c=[int(x) for x in open(sys.argv[2]) if x.strip()]
except:c=[]
print(f"{dec if dec else 0:.2f} {'OK' if (len(t)>15 and u>0.35) else 'BAD'} {max(c) if c else 0}")
PY
}
[ -f "$GPULK" ] && { . "$GPULK"; gpu_acquire "v3" >/dev/null 2>&1; }
measure /tmp/v3_cand >/dev/null; measure "$BASE_DAEMON" >/dev/null
B=();C=();BC=OK;CC=OK;BK=();CK=()
for r in 1 2 3 4; do
  read d c k < <(measure "$BASE_DAEMON"); B+=("$d");BK+=("$k");[ "$c" = BAD ]&&BC=BAD
  read d c k < <(measure /tmp/v3_cand);   C+=("$d");CK+=("$k");[ "$c" = BAD ]&&CC=BAD
done
[ -f "$GPULK" ] && gpu_release >/dev/null 2>&1
python3 - "${B[*]}" "${C[*]}" "$BC" "$CC" "${BK[*]}" "${CK[*]}" <<'PY'
import sys,statistics as st,json
B=[float(x) for x in sys.argv[1].split()];C=[float(x) for x in sys.argv[2].split()]
bc,cc=sys.argv[3],sys.argv[4];bk=[int(x) for x in sys.argv[5].split() if x];ck=[int(x) for x in sys.argv[6].split() if x]
bm,cm=st.median(B),st.median(C);d=100*(cm-bm)/bm if bm else 0
clkok=(not bk or not ck) or (abs(st.median(bk)-st.median(ck))/max(st.median(bk),1)*100<4.0)
v="COHERENCE_FAIL" if cc=="BAD" else ("VOID" if not clkok else ("WIN" if d>1.0 else ("DEAD" if d<-1.0 else "NEUTRAL")))
print(json.dumps({"verdict":v,"base_tok_s":round(bm,2),"cand_tok_s":round(cm,2),"delta_pct":round(d,2),"base_runs":B,"cand_runs":C,"clock_ok":clkok,"cand_coh":cc}))
PY

#!/usr/bin/env bash
# Verify the gfx11 combined stack (loop/card0 = baseline + moe_down + attn Phase-A + Phase-D rings) vs baseline.
cd "$HOME/hipfire/.aw/sw_card0" || { echo "no wt" > /tmp/stack_verify.txt; exit 1; }
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
MODEL=/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r
SCLK=/sys/class/drm/card0/device/pp_dpm_sclk
build_d(){ CARGO_TARGET_DIR="$PWD/target" cargo build --release --example daemon --features deltanet -p hipfire-runtime >/dev/null 2>&1 && cp target/release/examples/daemon "$1"; }
git checkout -q loop_baseline_gfx1100 -- kernels/src/ 2>/dev/null; git checkout -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
git checkout -q loop_baseline_gfx1100 2>/dev/null; build_d /tmp/base_d; echo "base built" >> /tmp/stack_verify.txt
git checkout -q loop/card0 2>/dev/null; build_d /tmp/stack_d; echo "stack built" >> /tmp/stack_verify.txt
git checkout -q loop_baseline_gfx1100 2>/dev/null
measure(){ # $1=daemon -> decode_tok_s (kernel warm)
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w","prompt":"Explain hash maps.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HIP_VISIBLE_DEVICES=0 timeout 220 "$1" 2>/dev/null | grep '"id":"m"' | grep done | python3 -c "import json,sys
for l in sys.stdin:
 d=json.loads(l); print(d.get('decode_tok_s',0))" | tail -1; }
clk(){ grep '\*' "$SCLK" 2>/dev/null | grep -oiE '[0-9]+Mhz' | grep -oE '[0-9]+' | head -1; }
B=(); S=()
for r in $(seq 1 6); do
  bc=$(clk); b=$(measure /tmp/base_d); ba=$(clk)
  sc=$(clk); s=$(measure /tmp/stack_d); saf=$(clk)
  echo "r$r base=$b (clk $bc/$ba) stack=$s (clk $sc/$saf)" >> /tmp/stack_verify.txt
  B+=("$b"); S+=("$s")
done
python3 -c "
import statistics as st
B=[float(x) for x in '${B[*]}'.split() if x]; S=[float(x) for x in '${S[*]}'.split() if x]
if B and S:
  mb,ms=st.median(B),st.median(S)
  f=sum(1 for x in S for y in B if x>y)/(len(S)*len(B))
  print(f'RESULT base={mb:.1f} stack={ms:.1f} delta={100*(ms/mb-1):+.2f}% f={f:.3f} (n={len(B)})')
" >> /tmp/stack_verify.txt
echo "STACK_VERIFY_DONE" >> /tmp/stack_verify.txt

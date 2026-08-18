#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-dtod-dump2" || exit 9
export HIP_VISIBLE_DEVICES=1
HIPFIRE_DTOD_DUMP=1 ./target/release/examples/deepseek4_prefill_bench \
  /home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r \
  --prefix 640 --ar-ref 10 --tokens 0 --batch 1 --e8-batched 0 --reps 1 --warmup 0 \
  > /home/kaden/dtod2_out.txt 2> /home/kaden/dtod2_err.txt
gpu_release
python3 - <<'PY'
import re, collections
lines=open('/home/kaden/dtod2_err.txt', errors='replace').read().splitlines()
# The AR reference loop starts after the "Prefilled" line; each step is
# delimited by an "[ar N]" progress line.
start=None
for i,l in enumerate(lines):
    if l.startswith('[ar '): start=i; break
pre=[l for l in lines[:start] if l.startswith('dtod ')]
post=[l for l in lines[start:] if l.startswith('dtod ')]
nsteps=sum(1 for l in lines if l.startswith('[ar '))
def hist(ls, div, label):
    c=collections.Counter(); b=collections.Counter()
    for l in ls:
        m=re.match(r'dtod bytes=(\d+) at (.+)', l)
        if m: c[m.group(2)]+=1; b[m.group(2)]+=int(m.group(1))
    print(f"\n--- {label} ---")
    print(f"{'calls':>8} {'per step':>9} {'KB/step':>9}  site")
    for s,n in c.most_common():
        print(f"{n:>8} {n/div:>9.1f} {b[s]/1024/div:>9.1f}  {s}")
    print(f"{'TOTAL':>8} {sum(c.values())/div:>9.1f} {sum(b.values())/1024/div:>9.1f}")
print(f"AR steps counted: {nsteps}")
hist(pre,1,"PREFILL (once)")
hist(post,max(nsteps,1),"AR DECODE (per step)")
PY
echo "DONE"

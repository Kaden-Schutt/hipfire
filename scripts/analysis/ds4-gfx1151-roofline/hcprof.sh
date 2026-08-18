#!/usr/bin/env bash
set -uo pipefail
cd /home/kaden/hipfire-ds4-gfx1151-opt
source scripts/gpu-lock.sh
gpu_acquire "ds4-hc-prof" || exit 9
export HIP_VISIBLE_DEVICES=1
M=/home/kaden/.cache/hipfire-surgery/deepseek-v4-flash.mq2r
BIN=./target/release/examples/deepseek4_prefill_bench
OUT=/home/kaden/hcprof; rm -rf $OUT; mkdir -p $OUT
HIPFIRE_HC_CTRL_T1024=1 rocprofv3 --kernel-trace --output-format csv -d $OUT -o t1024 -- \
  $BIN $M --prefix 640 --ar-ref 100 --tokens 0 --batch 1 --e8-batched 0 \
  --reps 1 --warmup 0 > $OUT/run.log 2>&1
gpu_release
python3 - <<'PY'
import csv, collections
tot=collections.Counter(); cnt=collections.Counter(); meta={}
for r in csv.DictReader(open('/home/kaden/hcprof/t1024_kernel_trace.csv')):
    k=r.get("Kernel_Name"); s=r.get("Start_Timestamp")
    if not k or not s: continue
    tot[k]+=int(r["End_Timestamp"])-int(s); cnt[k]+=1
    if k not in meta:
        gx=int(r["Grid_Size_X"]); wx=int(r["Workgroup_Size_X"])
        meta[k]=(max(1,gx//max(wx,1)), wx, int(r["VGPR_Count"]), int(r["LDS_Block_Size"]))
N=100
rows=[(tot[k]/1e6/N, cnt[k]/N, k) for k in tot if cnt[k]>=1000]
rows.sort(reverse=True)
print(f"{'ms/step':>8} {'calls':>7} {'us/call':>8} {'WGs':>6} {'thr':>5} {'VGPR':>5}  kernel")
for ms,c,k in rows[:8]:
    wg,wx,v,l=meta[k]
    print(f"{ms:8.3f} {c:7.1f} {ms*1000/c:8.2f} {wg:6d} {wx:5d} {v:5d}  {k[:46]}")
print(f"\ntotal decode GPU-busy: {sum(r[0] for r in rows):.3f} ms/step  (baseline 33.788)")
for ms,c,k in rows:
    if 'hc_compute_control' in k:
        wg,wx,v,l=meta[k]
        print(f"hc_compute_control: {ms:.3f} ms/step, {ms*1000/c:.2f} us/call, {wg} WGs x {wx}t, VGPR {v}, LDS {l}  (baseline 0.942 ms / 10.95 us)")
PY
echo "DONE"

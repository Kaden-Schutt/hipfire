#!/usr/bin/env bash
cd /tmp
echo "=== compiling mall_probe (gfx1100) ==="
hipcc --offload-arch=gfx1100 -O3 mall_probe.hip -o mall_probe 2>/tmp/mall_build.log || { echo BUILD_FAIL; tail -8 /tmp/mall_build.log; exit 1; }
source ~/hipfire/scripts/gpu-lock.sh 2>/dev/null; gpu_acquire mall_probe >/dev/null 2>&1
echo "device: $(HIP_VISIBLE_DEVICES=0 ./mall_probe 4 2 1 2>&1 >/dev/null | head -1)"
echo "=== BW sweep: steady-state reread GB/s vs buffer MB (L2 edge ~6MB, MALL edge ~96MB, then DRAM ~960) ==="
for mb in 1 2 4 6 8 12 16 24 32 48 64 96 128 160 224; do
  HIP_VISIBLE_DEVICES=0 ./mall_probe $mb 64 2>/dev/null
done
gpu_release >/dev/null 2>&1

#!/usr/bin/env bash
cd /tmp
hipcc --offload-arch=gfx1100 -O3 cu_scale_probe.hip -o cu_scale_probe 2>/tmp/cu_build.log || { echo BUILD_FAIL; tail -8 /tmp/cu_build.log; exit 1; }
source ~/hipfire/scripts/gpu-lock.sh 2>/dev/null; gpu_acquire cu_scale >/dev/null 2>&1
echo "=== gfx1100 CU-saturation sweep: throughput vs grid blocks (knee = useful CU count) ==="
HIP_VISIBLE_DEVICES=0 ./cu_scale_probe
gpu_release >/dev/null 2>&1

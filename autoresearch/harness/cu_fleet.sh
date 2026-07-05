#!/usr/bin/env bash
cd /tmp
OA="$1"; FLAGS=""; for a in $OA; do FLAGS="$FLAGS --offload-arch=$a"; done
hipcc $FLAGS -O3 cu_scale_probe.hip -o cu_scale_probe 2>/tmp/cu_build.log || { echo BUILD_FAIL; tail -8 /tmp/cu_build.log; exit 1; }
source ~/hipfire/scripts/gpu-lock.sh 2>/dev/null
for dev in 0 1 2 3 4 5; do
  out=$( { gpu_acquire cs_$dev >/dev/null 2>&1; HIP_VISIBLE_DEVICES=$dev timeout 90 /tmp/cu_scale_probe 2>&1; gpu_release >/dev/null 2>&1; } )
  echo "$out" | grep -q "arch=gfx" && { echo "===== HIP_VISIBLE_DEVICES=$dev ====="; echo "$out"; }
done

#!/usr/bin/env bash
cd /tmp
OA="$1"; DEVS="${2:-0}"; FLAGS=""; for a in $OA; do FLAGS="$FLAGS --offload-arch=$a"; done
hipcc $FLAGS -O3 gemv_occ_probe.hip -o gemv_occ_probe 2>/tmp/gemv_build.log || { echo BUILD_FAIL; tail -8 /tmp/gemv_build.log; exit 1; }
source ~/hipfire/scripts/gpu-lock.sh 2>/dev/null
for dev in $DEVS; do
  out=$( { gpu_acquire go_$dev >/dev/null 2>&1; HIP_VISIBLE_DEVICES=$dev timeout 90 /tmp/gemv_occ_probe 2>&1; gpu_release >/dev/null 2>&1; } )
  echo "$out" | grep -q "arch=gfx" && { echo "===== dev $dev ====="; echo "$out"; }
done

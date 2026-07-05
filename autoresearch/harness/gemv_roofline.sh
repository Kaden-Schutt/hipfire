#!/usr/bin/env bash
cd /tmp
OA="$1"; DEVS="${2:-0}"; FLAGS=""; for a in $OA; do FLAGS="$FLAGS --offload-arch=$a"; done
hipcc $FLAGS -O3 gemv_roofline_probe.hip -o gemv_roofline_probe 2>/tmp/rfl_build.log || { echo BUILD_FAIL; tail -12 /tmp/rfl_build.log; exit 1; }
source ~/hipfire/scripts/gpu-lock.sh 2>/dev/null
for dev in $DEVS; do
  out=$( { gpu_acquire rfl_$dev >/dev/null 2>&1; HIP_VISIBLE_DEVICES=$dev timeout 120 /tmp/gemv_roofline_probe 2>&1; gpu_release >/dev/null 2>&1; } )
  echo "$out" | grep -q "arch=gfx" && { echo "===== dev $dev ====="; echo "$out"; }
done

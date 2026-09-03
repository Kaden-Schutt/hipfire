#!/usr/bin/env bash
# Measure the escha model's resident footprint as an amdgpu GTT delta.
#
# gfx1151 is a unified-memory APU: "VRAM" is system RAM handed to the GPU
# through GTT, so `mem_info_gtt_used` is the model's true resident cost and
# process RSS is not (the weights are device allocations, not process pages).
# Reports the idle baseline, the peak, and the delta, sampling at 200 ms.
#
# Usage: scripts/escha-gtt-probe.sh <command> [args...]
set -u

CARD="${ESCHA_GTT_CARD:-/sys/class/drm/card1/device/mem_info_gtt_used}"
[ -r "$CARD" ] || { echo "no readable GTT node at $CARD" >&2; exit 2; }

base=$(cat "$CARD")
echo "[gtt] baseline: $base bytes ($(echo "scale=2; $base/1000000000" | bc) GB)"

"$@" &
pid=$!

peak=$base
while kill -0 "$pid" 2>/dev/null; do
  now=$(cat "$CARD" 2>/dev/null || echo 0)
  [ "$now" -gt "$peak" ] && peak=$now
  sleep 0.2
done
wait "$pid"
status=$?

after=$(cat "$CARD")
echo "[gtt] peak:     $peak bytes ($(echo "scale=2; $peak/1000000000" | bc) GB)"
echo "[gtt] delta:    $((peak - base)) bytes ($(echo "scale=2; ($peak-$base)/1000000000" | bc) GB)"
echo "[gtt] after:    $after bytes ($(echo "scale=2; $after/1000000000" | bc) GB)"
echo "[gtt] exit:     $status"
exit $status

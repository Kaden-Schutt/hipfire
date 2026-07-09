#!/usr/bin/env bash
# Fully terminate the autoresearch loop + free the GPU for clean profiling.
for pat in "driver_v3" "agent_exec.sh" "codex exec" "grok -p" "ab_certify_v2" "release/examples/daemon" "loop_round" "gen_digest" "rollover_v2"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 3
echo "remaining: driver=$(pgrep -cf driver_v3) codex=$(pgrep -cf 'codex exec') grok=$(pgrep -cf 'grok -p') certify=$(pgrep -cf ab_certify_v2) daemon=$(pgrep -cf 'release/examples/daemon')"
echo "card0 VRAM used: $(awk '{printf "%.1fGB", $1/1073741824}' /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null)  gpu_busy: $(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)%"

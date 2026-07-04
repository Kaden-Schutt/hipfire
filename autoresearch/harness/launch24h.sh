#!/usr/bin/env bash
# Runs ON hiptrx (invoked by a SHORT ssh, so no inline-nohup FD-hang). Cleans all
# stale state, then detaches the 24h driver via setsid so it survives ssh close.
cd "$HOME/hipfire" || exit 1
# kill stale drivers (exclude THIS script's own pid; our cmdline is launch24h.sh, not loop_driver.sh)
for p in $(pgrep -f "/tmp/loop_driver.sh" 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
pkill -9 -f "release/examples/daemon" 2>/dev/null
pkill -9 -f "cli/index.ts serve" 2>/dev/null
pkill -9 -f "ab_certify_swarm" 2>/dev/null
sleep 2
for c in 0 1 2 3; do
  git -C ".aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null
  echo auto | sudo tee "/sys/class/drm/card$c/device/power_dpm_force_performance_level" >/dev/null 2>&1
done
rm -f /tmp/loop_driver.log /tmp/loop_driver.done /tmp/loop_R*.hip /tmp/loop_progress.log
mkdir -p /tmp/wins; rm -f /tmp/wins/*.hip
# detach the driver (setsid = new session, survives ssh; driver A/B is sudo-free)
setsid bash /tmp/loop_driver.sh 86400 </dev/null >/tmp/loop_driver_main.out 2>&1 &
DPID=$!
disown 2>/dev/null
echo "LAUNCH24H_OK driver_pid=$DPID $(date -u '+%F %T')"

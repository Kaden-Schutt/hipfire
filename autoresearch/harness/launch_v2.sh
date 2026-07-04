#!/usr/bin/env bash
# Swap the running v1 driver -> v2 (rollover-aware) at a clean point. Runs ON hiptrx
# via a short ssh (no inline-nohup FD-hang). Aborts the in-flight round (re-runs
# under v2), cleans all worktrees, relaunches v2 detached for the remaining budget.
cd "$HOME/hipfire" || exit 1
for p in $(pgrep -f "/tmp/loop_driver.sh" 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
pkill -9 -f "codex exec" 2>/dev/null
pkill -9 -f "ab_certify_swarm" 2>/dev/null
pkill -9 -f "release/examples/daemon" 2>/dev/null
sleep 2
# a killed certify may have left a variant applied — restore pristine baseline in all worktrees
for c in 0 1 2 3; do git -C ".aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null; done
echo 1 > /tmp/rollover_dryrun    # start in DRY mode (validate before arming)
# remaining budget until the original 24h deadline
DUR=$(( $(date -d "2026-07-05 01:46:00" +%s) - $(date +%s) )); [ "$DUR" -lt 60 ] && DUR=3600
setsid bash /tmp/loop_driver_v2.sh "$DUR" </dev/null >/tmp/loop_driver_main.out 2>&1 &
disown 2>/dev/null
echo "LAUNCH_V2_OK dur=${DUR}s pid=$! $(date -u '+%F %T')"

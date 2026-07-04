#!/usr/bin/env bash
# Cleanly stop the v1 run, then do ONE final threshold-lowered rollover to bank the
# attention +2.31% win into baseline_v2. Stays STOPPED afterward (harness-v2 next).
cd "$HOME/hipfire" || exit 1
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
for p in $(pgrep -f "loop_driver_v2.sh"); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
pkill -9 -f "codex exec" 2>/dev/null
pkill -9 -f "ab_certify_swarm" 2>/dev/null
pkill -9 -f "release/examples/daemon" 2>/dev/null
sleep 3
# clean all worktrees back to baseline_v1 (kill-mid-round may leave a variant applied)
for c in 0 1 2 3; do
  git -C .aw/sw_card$c checkout -- kernels/src/ 2>/dev/null
  git -C .aw/sw_card$c clean -fdq kernels/src/ 2>/dev/null
  git -C .aw/sw_card$c checkout -fq 35811504 2>/dev/null
done
echo "STOPPED+CLEANED — cards at: $(for c in 0 1 2 3; do git -C .aw/sw_card$c rev-parse --short HEAD; done | sort -u | tr '\n' ' ')"
echo "=== final rollover: bank attention (thresholds lowered for the single win) ==="
MIN_NAIVE_SUM=2 MIN_COMPOSED=2 DRY_RUN=0 bash /tmp/rollover.sh 92
echo "=== post ==="
for c in 0 1 2 3; do echo -n "$(git -C .aw/sw_card$c rev-parse --short HEAD) "; done; echo
tail -1 /tmp/baseline_manifest.txt
echo "FINAL_ROLLOVER_DONE (run left STOPPED)"

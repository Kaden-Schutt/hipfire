#!/usr/bin/env bash
# One-shot: rebuild baseline_v1 CLEAN (drop the certify-created stray
# moe_gate_up_k8_indexed that git-add swept into 376fe217), re-align all 4
# worktrees, patch the manifest, restart driver_v2. Runs ON hiptrx.
cd "$HOME/hipfire" || exit 1
for p in $(pgrep -f "loop_driver_v2.sh" 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
pkill -9 -f "codex exec" 2>/dev/null; pkill -9 -f "ab_certify_swarm" 2>/dev/null; pkill -9 -f "release/examples/daemon" 2>/dev/null
sleep 2
WT=.aw/sw_card0
git -C "$WT" checkout -f 376fe217 2>/dev/null
git -C "$WT" clean -fdq kernels/src/ 2>/dev/null
git -C "$WT" rm --quiet kernels/src/gemv_hfq4g256_moe_gate_up_k8_indexed.hip 2>/dev/null
git -C "$WT" -c user.email=151092359+Kaden-Schutt@users.noreply.github.com -c user.name="Kaden Schutt" commit --amend --no-edit -q
CLEAN=$(git -C "$WT" rev-parse --short HEAD)
git -C "$WT" branch -f loop_baseline_gfx1201 "$CLEAN" 2>/dev/null
for c in 1 2 3; do git -C .aw/sw_card$c checkout -- kernels/src/ 2>/dev/null; git -C .aw/sw_card$c clean -fdq kernels/src/ 2>/dev/null; git -C .aw/sw_card$c checkout -fq "$CLEAN" 2>/dev/null; done
NF=$(git -C "$WT" show --name-only --format= "$CLEAN" 2>/dev/null | grep -c "kernels/src")
echo "clean baseline_v1 = $CLEAN  (files: $NF — expect 7)"
for c in 0 1 2 3; do echo "  sw_card$c=$(git -C .aw/sw_card$c rev-parse --short HEAD)"; done
sed -i "s/sha=376fe217/sha=$CLEAN/" /tmp/baseline_manifest.txt 2>/dev/null
echo "  manifest: $(cat /tmp/baseline_manifest.txt)"
echo 0 > /tmp/rollover_dryrun
DUR=$(( $(date -d "2026-07-05 01:46:00" +%s) - $(date +%s) )); [ "$DUR" -lt 60 ] && DUR=3600
setsid bash /tmp/loop_driver_v2.sh "$DUR" </dev/null >/tmp/loop_driver_main.out 2>&1 &
disown 2>/dev/null
echo "FIX_DONE clean=$CLEAN dur=${DUR}s restarted pid=$!"

#!/usr/bin/env bash
# Queue-driven certify loop (RESUME mode): drain certify_queue.json, drain detected from the PROGRESS LOG (not codex prose).
cd "$HOME/hipfire" || exit 1
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
ARCH="${ARCH:?}"; CARDS="${CARDS:?}"; BASELINE_REF="${BASELINE_REF:?}"; MODEL="${MODEL:?}"; MAXR="${MAXR:-30}"
[ -f /tmp/loop_progress.log ] || : > /tmp/loop_progress.log   # RESUME: keep prior verdicts, skip done items
: > /tmp/loop_driver.log
rm -f /tmp/baseprof_* 2>/dev/null   # fresh baseline profiles for the per-variant profile_feedback (baseline may have advanced)
NC=$(echo $CARDS | wc -w); FIRST=${CARDS%% *}
for c in $CARDS; do git -C "$HOME/hipfire/.aw/sw_card$c" checkout -- kernels/src/ 2>/dev/null; git -C "$HOME/hipfire/.aw/sw_card$c" clean -fdq kernels/src/ 2>/dev/null; git -C "$HOME/hipfire/.aw/sw_card$c" checkout -q -B "loop/card$c" "$BASELINE_REF" 2>/dev/null; done
HDR="BOX: arch=$ARCH cards=[$CARDS] ncards=$NC model=$MODEL baseline_ref=$BASELINE_REF. The queue is PRE-VETTED -- IMPLEMENT, do not brainstorm. Skip any Q<order> already in the progress log. Certify each variant (card=dev=card; up to $NC parallel with & then wait): BASELINE_REF=$BASELINE_REF bash /tmp/ab_certify_v2.sh $ARCH <card> <card> $MODEL <kernel> <Qlabel> <variant.hip>"
drain_check(){ python3 -c "
import json,re
orders=[str(e['order']) for e in json.load(open('/tmp/certify_queue.json'))]
prog=open('/tmp/loop_progress.log').read()
print('1' if all(re.search('Q'+o+'v',prog) for o in orders) else '0')" 2>/dev/null; }
stall=0; r=0
while [ "$r" -lt "$MAXR" ]; do
  r=$((r+1))
  [ "$(drain_check)" = "1" ] && { echo "QUEUE DRAINED (all queue orders present in progress log) r$r $(date -u +%T)" >> /tmp/loop_driver.log; break; }
  before=$(wc -l < /tmp/loop_progress.log 2>/dev/null || echo 0)
  echo "===== QUEUE ROUND $r $(date -u +%T) =====" >> /tmp/loop_driver.log
  OUT=$(cd "$HOME/hipfire/.aw/sw_card$FIRST" && timeout 3400 codex exec --dangerously-bypass-approvals-and-sandbox "ROUND $r. $HDR
$(cat /tmp/loop_round_prompt_queue.txt)" 2>&1)
  printf '%s\n' "$OUT" | tail -30 >> /tmp/loop_driver.log
  after=$(wc -l < /tmp/loop_progress.log 2>/dev/null || echo 0)
  if [ "$after" -le "$before" ]; then stall=$((stall+1)); else stall=0; fi
  [ "$stall" -ge 2 ] && { echo "STALL: no new progress 2 rounds, stopping r$r $(date -u +%T)" >> /tmp/loop_driver.log; break; }
done
# --- AUTONOMOUS FOLD: compound the drained queue's wins into the baseline + re-census the BOD ---
# (this is what I hand-drove last time; the loop now folds itself — rollover_v2 composes the
#  branch wins, A/B-gates the composed fold, advances loop_baseline_$ARCH, and re-censuses bod_$ARCH.json)
echo "===== AUTONOMOUS FOLD (promote wins to arch-forks, A/B, advance) $(date -u +%T) =====" >> /tmp/loop_driver.log
# fold_with_forks: codex promotes each branch win to <k>.<arch>.hip (anti-clobber — writes the fork const +
# dispatch arm, REVERTS the shared <k>.hip), then A/Bs the combined fork state vs baseline (isolate-decode-
# window) and advances loop_baseline_$ARCH + re-censuses the BOD if it holds. Lineage manifest is persistent.
ARCH="$ARCH" CARDS="$CARDS" BASELINE_REF="$BASELINE_REF" MODEL="$MODEL" \
  timeout 3400 bash /tmp/fold_with_forks.sh >> /tmp/loop_driver.log 2>&1 || echo "  [fold] fold_with_forks rc=$?" >> /tmp/loop_driver.log
echo "QUEUE LOOP COMPLETE r$r $(date -u +%T)" >> /tmp/loop_driver.log

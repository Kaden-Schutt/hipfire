#!/usr/bin/env bash
# Queue-driven certify loop (RESUME mode): drain certify_queue.json, drain detected from the PROGRESS LOG (not codex prose).
cd "$HOME/hipfire" || exit 1
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
ARCH="${ARCH:?}"; CARDS="${CARDS:?}"; BASELINE_REF="${BASELINE_REF:?}"; MODEL="${MODEL:?}"; MAXR="${MAXR:-30}"
[ -f /tmp/loop_progress.log ] || : > /tmp/loop_progress.log   # RESUME: keep prior verdicts, skip done items
: > /tmp/loop_driver.log
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
echo "===== AUTONOMOUS FOLD (rollover_v2) $(date -u +%T) =====" >> /tmp/loop_driver.log
# rollover_v2 advances by DEFAULT now (dry-run ripped out). The lineage manifest is PERSISTENT so
# baseline versions CONTINUE (v2->v3->v4...); the anti-thrash gap is bypassed via MIN_ROUNDS_GAP=0 + round 999.
BOD="/tmp/bod_${ARCH}.json" SKIP_HOMOGUARD="${SKIP_HOMOGUARD:-0}" MANIFEST="/tmp/baseline_manifest_${ARCH}.txt" \
  FOLDED="/tmp/queue_folded_${ARCH}.txt" MIN_ROUNDS_GAP=0 MIN_NAIVE_SUM="${MIN_NAIVE_SUM:-2.0}" MIN_COMPOSED="${MIN_COMPOSED:-2.0}" \
  timeout 2600 bash /tmp/rollover_v2.sh 999 >> /tmp/loop_driver.log 2>&1 || echo "  [fold] rollover_v2 rc=$?" >> /tmp/loop_driver.log
echo "QUEUE LOOP COMPLETE r$r $(date -u +%T)" >> /tmp/loop_driver.log

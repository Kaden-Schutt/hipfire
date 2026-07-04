#!/usr/bin/env bash
# driver_v3 — SELF-EXHAUSTING v2 loop. No fixed round/time limit: stops when every
# candidate kernel (BOD wall% >= CAND_WALL, unfolded) has hit K=5 consecutive
# DEAD/INCONCLUSIVE attempts. Uses ab_certify_v2 (adaptive) + rollover_v2 (branch
# harvest). Injects a per-kernel tried-levers digest so Codex avoids re-derivation and
# drives coverage toward exhaustion. SAFETY_CAP is a backstop, not the primary stop.
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
cd "$HOME/hipfire" || exit 1
MAIN="$HOME/hipfire"
K="${K:-5}"; CAND_WALL="${CAND_WALL:-3.0}"; SAFETY_CAP="${1:-500}"
BASELINE_REF=loop_baseline_gfx1201
EXH=/tmp/exhaustion.json
BOD=/tmp/bod_gfx1201.json
FOLDED=/tmp/baseline_folded.txt
[ -f "$EXH" ] || echo "{}" > "$EXH"
[ -f /tmp/loop_progress.log ] || : > /tmp/loop_progress.log
# ensure per-card branches exist at the current baseline
for c in 0 1 2 3; do git -C "$MAIN/.aw/sw_card$c" checkout -q -B "loop/card$c" "$BASELINE_REF" 2>/dev/null; done
round=$(grep -oE 'R[0-9]+c' /tmp/loop_progress.log 2>/dev/null | grep -oE '[0-9]+' | awk '$1<100000{if($1>m)m=$1} END{print m+0}')
echo "===== DRIVER v3 (self-exhausting, K=$K) resuming after round $round $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
while [ "$round" -lt "$SAFETY_CAP" ]; do
  # global stop: all candidates exhausted?
  if python3 /tmp/check_exhausted.py "$EXH" "$BOD" "$CAND_WALL" "$K" "$FOLDED" 2>/dev/null; then
    echo "===== ALL CANDIDATES EXHAUSTED -> SELF-TERMINATE after round $round $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
    break
  fi
  round=$((round+1))
  echo "===== ROUND $round START $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
  DIGEST=$(python3 /tmp/gen_digest.py "$EXH" "$BOD" "$CAND_WALL" "$K" "$FOLDED" 2>/dev/null)
  timeout 3600 codex exec --dangerously-bypass-approvals-and-sandbox -C "$HOME/hipfire" \
    "ROUND $round of a SELF-EXHAUSTING autoresearch loop (adaptive certify + branch wins). ${DIGEST} $(cat /tmp/loop_round_prompt_v2.txt)" \
    >> /tmp/loop_driver.log 2>&1
  echo "===== ROUND $round END rc=$? $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
  # update per-kernel exhaustion counters from this round's verdicts
  python3 /tmp/update_exhaustion.py "$EXH" "$round" "$MAIN" >> /tmp/loop_driver.log 2>&1
  # rollover checkpoint (gap-gated inside rollover_v2)
  RDRY=$(cat /tmp/rollover_dryrun 2>/dev/null || echo 0)
  echo "  [driver] rollover_v2 check (round $round, dry=$RDRY)" >> /tmp/loop_driver.log
  DRY_RUN="$RDRY" timeout 2700 bash /tmp/rollover_v2.sh "$round" >> /tmp/loop_driver.log 2>&1 || echo "  [driver] rollover_v2 rc=$?" >> /tmp/loop_driver.log
done
echo "LOOP v3 COMPLETE ($round rounds) $(date -u '+%F %T')" >> /tmp/loop_driver.log
touch /tmp/loop_driver.done

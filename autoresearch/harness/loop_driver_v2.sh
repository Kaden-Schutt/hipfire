#!/usr/bin/env bash
# loop_driver_v2 — v1 + baseline ROLLOVER between rounds (homogeneous-multicard
# compounding). After each round, rollover.sh checks whether fresh best-per-kernel
# wins compose above threshold and, if armed (/tmp/rollover_dryrun == 0), folds
# them into a new baseline all 4 worktrees advance to. Codex is told what is now
# in-baseline so it re-targets the shifted bottleneck instead of re-finding it.
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
cd "$HOME/hipfire" || exit 1
DURATION="${1:-86400}"
END=$(( $(date +%s) + DURATION ))
mkdir -p /tmp/wins
[ -f /tmp/loop_progress.log ] || : > /tmp/loop_progress.log
# continue v1's round counter (avoid R<N> label collisions in ledger/wins); real rounds are <1000 (date-labels excluded)
round=$(grep -oE 'R[0-9]+c' /tmp/loop_progress.log 2>/dev/null | grep -oE '[0-9]+' | awk '$1<1000{if($1>m)m=$1} END{print m+0}')
echo "===== DRIVER v2 (rollover) resuming after round $round $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
while [ "$(date +%s)" -lt "$END" ]; do
  round=$((round+1))
  echo "===== ROUND $round START $(date -u '+%F %T') (budget ends $(date -u -d @"$END" '+%F %T')) =====" >> /tmp/loop_driver.log
  # inject current baseline state so Codex doesn't re-target folded kernels
  FOLDED=""; [ -s /tmp/baseline_folded.txt ] && FOLDED="BASELINE ALREADY INCLUDES these optimized kernels (their wall% has DROPPED — do NOT re-target them, pick OTHER kernels): $(paste -sd, /tmp/baseline_folded.txt). "
  timeout 3600 codex exec --dangerously-bypass-approvals-and-sandbox -C "$HOME/hipfire" \
    "ROUND $round of an ONGOING autoresearch loop (the ledger accumulates across rounds — read it and never repeat a dead end). ${FOLDED}$(cat /tmp/loop_round_prompt.txt)" \
    >> /tmp/loop_driver.log 2>&1
  echo "===== ROUND $round END rc=$? $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
  wc=$(ls /tmp/wins/*.hip 2>/dev/null | wc -l); echo "  [driver] cumulative win .hips: $wc" >> /tmp/loop_driver.log
  # --- ROLLOVER CHECKPOINT (between rounds; card0 idle) ---
  RDRY=$(cat /tmp/rollover_dryrun 2>/dev/null || echo 1)
  echo "  [driver] rollover check (round $round, dry=$RDRY)..." >> /tmp/loop_driver.log
  DRY_RUN="$RDRY" timeout 2700 bash /tmp/rollover.sh "$round" >> /tmp/loop_driver.log 2>&1 || echo "  [driver] rollover check exited rc=$? (continuing)" >> /tmp/loop_driver.log
done
echo "LOOP COMPLETE ($round rounds) $(date -u '+%F %T')" >> /tmp/loop_driver.log
touch /tmp/loop_driver.done

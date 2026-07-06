#!/usr/bin/env bash
# driver_v3 — SELF-EXHAUSTING, ARCH-CONFIGURABLE autoresearch loop. No fixed round/time
# limit: stops when every candidate kernel (BOD wall% >= CAND_WALL, unfolded) has hit K
# consecutive DEAD/INCONCLUSIVE attempts. Injects a per-kernel tried-levers digest so Codex
# avoids re-derivation and drives coverage toward exhaustion. SAFETY_CAP is a backstop.
#
# ARCH-AGNOSTIC: every arch/model/card knob is an env var (defaults = gfx1201 4-card so an
# existing gfx1201 campaign is untouched). Point PROMPT at an arch-specific prompt that bakes
# in the model/dev/certify — the driver only owns the loop, digest, branch lifecycle, and stop.
#
#   ARCH          arch id, also selects the ledger glob for exhaustion  (gfx1201)
#   BASELINE_REF  branch the loop anchors + certify resets to           (loop_baseline_$ARCH)
#   BOD           census BOD json (candidate kernels + roofline)        (/tmp/bod_$ARCH.json)
#   CARDS         space-sep WORKTREE slot indices (sw_card<N>)          ("0 1 2 3")
#   PROMPT        arch-specific codex round prompt                      (/tmp/loop_round_prompt_v2.txt)
#   EXH           per-arch exhaustion counter json                      (/tmp/exhaustion.json)
#   ROLLOVER      in-loop rollover script ($round arg); noop to skip    (/tmp/rollover_v2.sh)
#   K             consecutive-dead threshold per kernel                 (5)
#   CAND_WALL     min BOD wall% to be a candidate                       (3.0)
#   $1            SAFETY_CAP (max rounds backstop)                       (500)
#
# BRANCH SAFETY (non-negotiable, DECODE_AR_2STAGE_DESIGN.md): the driver OWNS the loop branch.
# It CREATES loop/card<N> from baseline ONCE, then RESUMES it every restart (never `checkout -B`
# on an existing branch — that force-resets to baseline and silently drops every banked win).
# A `loop/card<N>_recovered` safety branch is fast-forwarded each round so a stack can never be gc'd.
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
cd "$HOME/hipfire" || exit 1
MAIN="$HOME/hipfire"
K="${K:-5}"; CAND_WALL="${CAND_WALL:-3.0}"; SAFETY_CAP="${1:-500}"
export ARCH="${ARCH:-gfx1201}"
# --- NON-EPHEMERAL, ARCH-SCOPED STATE (NOT /tmp) ---------------------------------------------------
# Mutable loop state lives in the repo's state dir, suffixed per-arch, so it survives /tmp clears AND
# can NEVER cross-contaminate arches (the gfx1100-seed->gfx1151 bleed class is structurally excluded).
# The durable research record is the git-committed ledger; this is the derived counters + progress.
STATE="${STATE:-$MAIN/autoresearch/state}"; mkdir -p "$STATE"
export BASELINE_REF="${BASELINE_REF:-loop_baseline_gfx1201}"
export BOD="${BOD:-$STATE/bod_$ARCH.json}"
export EXH="${EXH:-$STATE/exhaustion_$ARCH.json}"
export LOOP_PROGRESS="${LOOP_PROGRESS:-$STATE/loop_progress_$ARCH.log}"  # trio + codex prompt all use THIS
export CROSSARCH_SEED="${CROSSARCH_SEED:-}"                              # OPT-IN arch-correct seed; empty = none
export CARDS="${CARDS:-0 1 2 3}"
export FOLDED="${FOLDED:-$STATE/baseline_folded_$ARCH.txt}"
export MANIFEST="${MANIFEST:-$STATE/baseline_manifest_$ARCH.txt}"
export SKIP_HOMOGUARD="${SKIP_HOMOGUARD:-0}"
PROMPT="${PROMPT:-/tmp/loop_round_prompt_v2.txt}"
ROLLOVER="${ROLLOVER:-/tmp/rollover_v2.sh}"
[ -f "$EXH" ] || echo "{}" > "$EXH"
[ -f "$LOOP_PROGRESS" ] || : > "$LOOP_PROGRESS"
# ---- CREATE-OR-RESUME the per-card loop branch (never force-reset an existing branch) ----
for c in $CARDS; do
  wt="$MAIN/.aw/sw_card$c"
  [ -d "$wt" ] || { echo "  [driver] WARN no worktree $wt (skip card $c)" >> /tmp/loop_driver.log; continue; }
  if git -C "$wt" rev-parse --verify -q "loop/card$c" >/dev/null; then
    git -C "$wt" checkout -q "loop/card$c" 2>/dev/null            # RESUME — keep accumulated wins
  else
    git -C "$wt" checkout -q -B "loop/card$c" "$BASELINE_REF" 2>/dev/null   # CREATE once from baseline
  fi
  git -C "$wt" branch -f "loop/card${c}_recovered" "loop/card$c" 2>/dev/null # gc-proof safety branch
done
round=$(grep -oE 'R[0-9]+c' "$LOOP_PROGRESS" 2>/dev/null | grep -oE '[0-9]+' | awk '$1<100000{if($1>m)m=$1} END{print m+0}')
echo "===== DRIVER v3 [$ARCH] (self-exhausting, K=$K, cards='$CARDS') resuming after round $round $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
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
    "ROUND $round of a SELF-EXHAUSTING autoresearch loop (adaptive certify + branch wins). ${DIGEST} $(cat "$PROMPT")" \
    >> /tmp/loop_driver.log 2>&1
  echo "===== ROUND $round END rc=$? $(date -u '+%F %T') =====" >> /tmp/loop_driver.log
  # update per-kernel exhaustion counters from this round's verdicts (ARCH selects the ledger glob)
  python3 /tmp/update_exhaustion.py "$EXH" "$round" "$MAIN" "$ARCH" >> /tmp/loop_driver.log 2>&1
  # keep the gc-proof safety branch current after any banked win this round
  for c in $CARDS; do git -C "$MAIN/.aw/sw_card$c" branch -f "loop/card${c}_recovered" "loop/card$c" 2>/dev/null; done
  # in-loop rollover checkpoint (gap-gated inside the rollover script; point at a noop to skip)
  RDRY=$(cat /tmp/rollover_dryrun 2>/dev/null || echo 0)
  echo "  [driver] rollover check (round $round, dry=$RDRY, script=$ROLLOVER)" >> /tmp/loop_driver.log
  DRY_RUN="$RDRY" timeout 2700 bash "$ROLLOVER" "$round" >> /tmp/loop_driver.log 2>&1 || echo "  [driver] rollover rc=$?" >> /tmp/loop_driver.log
done
echo "LOOP v3 [$ARCH] COMPLETE ($round rounds) $(date -u '+%F %T')" >> /tmp/loop_driver.log
touch /tmp/loop_driver.done

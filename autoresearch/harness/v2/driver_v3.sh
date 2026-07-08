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
#   BOD           census BOD json (candidate kernels + roofline)        (STATE/bod_$ARCH.json)
#   CARDS         space-sep WORKTREE slot indices (sw_card<N>)          ("0 1 2 3")
#   PROMPT        arch-specific codex round prompt                      (HARN/loop_round_prompt_$ARCH.txt)
#   EXH           per-arch exhaustion counter json                      (STATE/exhaustion_$ARCH.json)
#   ROLLOVER      in-loop rollover script ($round arg); noop to skip    (HARN/noop_rollover.sh)
#   STATE/HARN    persistent state + deployed-scripts dirs (NOT /tmp)   ($MAIN/autoresearch/{state,harness})
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
# PERSISTENT deployed scripts (NOT /tmp): the exhaustion trio + rollover live here, survive reboot.
HARN="${HARN:-$MAIN/autoresearch/harness}"; TRIO="$HARN/v2"
LOG="${LOG:-$STATE/loop_driver_$ARCH.log}"   # driver log is arch-scoped + persistent, not /tmp
export BASELINE_REF="${BASELINE_REF:-loop_baseline_gfx1201}"
export BOD="${BOD:-$STATE/bod_$ARCH.json}"
export EXH="${EXH:-$STATE/exhaustion_$ARCH.json}"
export LOOP_PROGRESS="${LOOP_PROGRESS:-$STATE/loop_progress_$ARCH.log}"  # trio + codex prompt all use THIS
export CROSSARCH_SEED="${CROSSARCH_SEED:-}"                              # OPT-IN arch-correct seed; empty = none
export CARDS="${CARDS:-0 1 2 3}"
export FOLDED="${FOLDED:-$STATE/baseline_folded_$ARCH.txt}"
export MANIFEST="${MANIFEST:-$STATE/baseline_manifest_$ARCH.txt}"
export SKIP_HOMOGUARD="${SKIP_HOMOGUARD:-0}"
PROMPT="${PROMPT:-$HARN/loop_round_prompt_$ARCH.txt}"
ROLLOVER="${ROLLOVER:-$HARN/noop_rollover.sh}"
[ -f "$EXH" ] || echo "{}" > "$EXH"
[ -f "$LOOP_PROGRESS" ] || : > "$LOOP_PROGRESS"
# Loop branch resolver. LOOP_BRANCH overrides with an ARCH-keyed name (e.g. loop/gfx1151) — the shared
# branch namespace must be globally unambiguous ("card2" = gfx1201 on hiptrx but gfx1030 on hipx). Only
# valid single-card. Default (unset) = legacy per-card loop/card<N> (the multi-card gfx1201 campaign is
# untouched). The certify wrapper (ab_certify_serve.sh) resets to loop/$ARCH, so set LOOP_BRANCH to match.
lb() { [ -n "${LOOP_BRANCH:-}" ] && echo "$LOOP_BRANCH" || echo "loop/card$1"; }
# ---- CREATE-OR-RESUME the per-card loop branch (never force-reset an existing branch) ----
for c in $CARDS; do
  wt="$MAIN/.aw/sw_card$c"
  [ -d "$wt" ] || { echo "  [driver] WARN no worktree $wt (skip card $c)" >> "$LOG"; continue; }
  lbr="$(lb "$c")"
  if git -C "$wt" rev-parse --verify -q "$lbr" >/dev/null; then
    git -C "$wt" checkout -q "$lbr" 2>/dev/null            # RESUME — keep accumulated wins
  else
    git -C "$wt" checkout -q -B "$lbr" "$BASELINE_REF" 2>/dev/null   # CREATE once from baseline
  fi
  git -C "$wt" branch -f "${lbr}_recovered" "$lbr" 2>/dev/null # gc-proof safety branch
done
round=$(grep -oE 'R[0-9]+c' "$LOOP_PROGRESS" 2>/dev/null | grep -oE '[0-9]+' | awk '$1<100000{if($1>m)m=$1} END{print m+0}')
echo "===== DRIVER v3 [$ARCH] (self-exhausting, K=$K, cards='$CARDS') resuming after round $round $(date -u '+%F %T') =====" >> "$LOG"
while [ "$round" -lt "$SAFETY_CAP" ]; do
  # global stop: all candidates exhausted?
  if python3 "$TRIO/check_exhausted.py" "$EXH" "$BOD" "$CAND_WALL" "$K" "$FOLDED" 2>/dev/null; then
    echo "===== ALL CANDIDATES EXHAUSTED -> SELF-TERMINATE after round $round $(date -u '+%F %T') =====" >> "$LOG"
    break
  fi
  round=$((round+1))
  echo "===== ROUND $round START $(date -u '+%F %T') =====" >> "$LOG"
  DIGEST=$(python3 "$TRIO/gen_digest.py" "$EXH" "$BOD" "$CAND_WALL" "$K" "$FOLDED" 2>/dev/null)
  # codex reads kernels/src from ITS OWN worktree (this worker's advancing baseline) — NOT $HOME/hipfire
  # main, which may be an old/foreign checkout. First CARDS slot = this worker's build worktree.
  CODEX_CWD="${CODEX_CWD:-$MAIN/.aw/sw_card$(set -- $CARDS; echo "$1")}"
  [ -d "$CODEX_CWD/kernels/src" ] || CODEX_CWD="$MAIN"
  timeout 3600 codex exec --dangerously-bypass-approvals-and-sandbox -C "$CODEX_CWD" \
    "ROUND $round of a SELF-EXHAUSTING autoresearch loop (adaptive certify + branch wins). ${DIGEST} $(cat "$PROMPT")" \
    >> "$LOG" 2>&1
  echo "===== ROUND $round END rc=$? $(date -u '+%F %T') =====" >> "$LOG"
  # update per-kernel exhaustion counters from this round's verdicts (ARCH selects the ledger glob)
  python3 "$TRIO/update_exhaustion.py" "$EXH" "$round" "$MAIN" "$ARCH" >> "$LOG" 2>&1
  # keep the gc-proof safety branch current after any banked win this round
  for c in $CARDS; do lbr="$(lb "$c")"; git -C "$MAIN/.aw/sw_card$c" branch -f "${lbr}_recovered" "$lbr" 2>/dev/null; done
  # in-loop rollover checkpoint (gap-gated inside the rollover script; point at a noop to skip)
  RDRY=$(cat /tmp/rollover_dryrun 2>/dev/null || echo 0)
  echo "  [driver] rollover check (round $round, dry=$RDRY, script=$ROLLOVER)" >> "$LOG"
  DRY_RUN="$RDRY" timeout 2700 bash "$ROLLOVER" "$round" >> "$LOG" 2>&1 || echo "  [driver] rollover rc=$?" >> "$LOG"
done
echo "LOOP v3 [$ARCH] COMPLETE ($round rounds) $(date -u '+%F %T')" >> "$LOG"
touch "$STATE/loop_driver_$ARCH.done"

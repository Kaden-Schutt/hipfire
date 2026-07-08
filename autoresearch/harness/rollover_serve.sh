#!/usr/bin/env bash
# rollover_serve.sh — fold the accumulated per-round wins (loop/<arch>) into trunk, GATED BY THE SERVE
# COHERENCE PASS (thinking ON, sampled, multiturn seed-set + validators — the real user path). This is
# the ONLY place the serve/CLI/TS path is used. The per-round screen (ab_certify_v2p) already enforced
# parity + perf on each win as it was banked; here we verify the COMPOSED stack (loop/<arch> tip, which
# is the composition of every banked win) is still COHERENT vs trunk before folding. DRY_RUN=1 (default)
# reports the coherence verdict without advancing.
#   env: ARCH CARD DEV MODEL  LOOP_BRANCH(=loop/<arch>)  TRUNK(fold target)  DRM_CARD  SEEDS  DRY_RUN
set -u
ROUND="${1:-0}"; ROLLOVER_EVERY="${ROLLOVER_EVERY:-5}"   # driver calls this every round; only fold-check every Nth
ARCH="${ARCH:-gfx1151}"; CARD="${CARD:-2}"; DEV="${DEV:-1}"
MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
LOOP_BRANCH="${LOOP_BRANCH:-loop/$ARCH}"; TRUNK="${TRUNK:-loop_baseline_$ARCH}"
DRM_CARD="${DRM_CARD:-$DEV}"; SEEDS="${SEEDS:-8}"; DRY_RUN="${DRY_RUN:-1}"
MAIN="$HOME/hipfire"; WT="$MAIN/.aw/sw_card$CARD"; HARN="$MAIN/autoresearch/harness"
STATE="$MAIN/autoresearch/state"; CACHE="$STATE/cache"; LOG="${LOG:-$STATE/rollover_$ARCH.log}"
GUARD="$HARN/guards/coherence_guard.json"
mkdir -p "$CACHE"
log(){ echo "  [rollover $ARCH] $*" >> "$LOG"; echo "  [rollover $ARCH] $*"; }
cd "$WT" 2>/dev/null || { log "no worktree $WT"; exit 1; }
export HOME_ORIG="$HOME"; export HOME="$WT/.swhome"; mkdir -p "$HOME"
export CARGO_TARGET_DIR="$WT/target"; export PATH="$HOME_ORIG/.bun/bin:$PATH"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME_ORIG/.rustup}" CARGO_HOME="${CARGO_HOME:-$HOME_ORIG/.cargo}"
DB="target/release/examples/daemon"
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 \
         | grep -qiE "^error|error\[" && return 1; return 0; }

# cadence: skip non-fold-check rounds cheaply (the serve coherence pass is ~18 min)
[ "$ROUND" -gt 0 ] && [ $((ROUND % ROLLOVER_EVERY)) -ne 0 ] && exit 0

# anything to fold?
NWINS=$(git rev-list --count "${TRUNK}..${LOOP_BRANCH}" 2>/dev/null || echo 0)
[ "${NWINS:-0}" -gt 0 ] || { log "no wins on $LOOP_BRANCH vs $TRUNK -> nothing to fold"; exit 0; }
log "$NWINS win-commit(s) on $LOOP_BRANCH pending fold vs $TRUNK"

# build trunk daemon + composed (loop/<arch> tip = every banked win) daemon
TRUNK_D="$CACHE/rollover_trunk_$ARCH"; COMP_D="$CACHE/rollover_comp_$ARCH"
git checkout "$TRUNK" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
build || { log "trunk build FAIL"; exit 1; }; cp "$DB" "$TRUNK_D"
git checkout "$LOOP_BRANCH" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
if ! build; then log "composed build FAIL -> abort"; git checkout "$TRUNK" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null; exit 1; fi
cp "$DB" "$COMP_D"
git checkout "$TRUNK" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null

# --- SERVE COHERENCE PASS (composed vs trunk) — the fold gate ---
GPULK="$MAIN/scripts/gpu-lock.sh"; [ -f "$GPULK" ] && { set +u; . "$GPULK"; gpu_acquire "rollover_${ARCH}" >/dev/null 2>&1; set -u; }
ROW=$(HOME="$HOME_ORIG" HIP_VISIBLE_DEVICES=$DEV python3 "$HARN/ab_certify_serve.py" --mode coherence \
        --arch "$ARCH" --dev "$DEV" --card "$DRM_CARD" --kernel "COMPOSED" --label "rollover_${ARCH}" \
        --model "$MODEL" --base-daemon "$TRUNK_D" --var-daemon "$COMP_D" --base-ref "$TRUNK" \
        --seeds "$SEEDS" --prompts-file "$GUARD" 2>"$STATE/rollover_coh_${ARCH}.err")
[ -f "$GPULK" ] && { set +u; gpu_release >/dev/null 2>&1; set -u; }
[ -n "$ROW" ] || { log "coherence pass produced no row (see $STATE/rollover_coh_${ARCH}.err) -> abort"; exit 1; }
VERDICT=$(printf '%s' "$ROW" | python3 -c "import json,sys;print(json.load(sys.stdin).get('verdict','?'))" 2>/dev/null || echo "?")
log "serve coherence verdict: $VERDICT  row=$ROW"

if [ "$VERDICT" != "COHERENT" ]; then
  log "NOT COHERENT ($VERDICT) -> NOT folding (the composed stack regresses output under real sampling)"
  exit 0
fi
if [ "$DRY_RUN" = 1 ]; then
  log "COHERENT -> DRY-RUN, not advancing (set DRY_RUN=0 to fold $NWINS wins into $TRUNK)"
  exit 0
fi

# --- FOLD: advance trunk to the composed stack ---
if git branch -f "$TRUNK" "$LOOP_BRANCH" 2>/dev/null; then
  log "FOLDED: $TRUNK advanced to $LOOP_BRANCH ($NWINS wins now on trunk)"
else
  log "FOLD FAILED (could not advance $TRUNK)"
  exit 1
fi

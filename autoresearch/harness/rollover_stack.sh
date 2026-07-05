#!/usr/bin/env bash
# rollover_stack.sh — verification-gated BRANCH-STACK rollover for the decode-AR explore loop.
#
# The explore loop banks wins as COMMITS on loop/cardN (unlike the gfx1201 rollover.sh, which folds
# best-per-kernel .hip files from /tmp/wins + the ledger). This variant rolls that commit-stack into a
# new baseline: fresh-process cumulative re-measure (DPM-warm, N samples, median) + coherence, then
# promote (fast-forward baseline + rebuild PREBUILT_BASE cache + re-anchor loop) ONLY on --promote AND pass.
#
# Default = DRY-RUN (measure + report, no mutation). Reuses ab_certify_v2p's exact measurement
# (kernel_decode_tok_s on the m1 "done" event, q8 KV, the standard prompt). Builds the stack daemon from
# the DETACHED tip SHA (a branch checked out in sw_cardN can't be checked out in a 2nd worktree).
#
# Usage:
#   rollover_stack.sh --arch gfx1100 --card 0 --dev 0 --baseline-ref loop_baseline_gfx1100 \
#                     --model ~/.hipfire/models/qwen3.6-35b-a3b.mq4r [--samples 5] [--margin 1.0] [--promote]
# Caller MUST pause (pkill) the card's codex loop before --promote to avoid a commit race.
set -uo pipefail
MAIN="$HOME/hipfire"
ARCH= CARD= DEV= BASE= MODEL= SAMPLES=5 MARGIN=1.0 PROMOTE=0
while [ $# -gt 0 ]; do case "$1" in
  --arch) ARCH=$2;shift 2;; --card) CARD=$2;shift 2;; --dev) DEV=$2;shift 2;;
  --baseline-ref) BASE=$2;shift 2;; --model) MODEL=$2;shift 2;;
  --samples) SAMPLES=$2;shift 2;; --margin) MARGIN=$2;shift 2;; --promote) PROMOTE=1;shift;;
  *) echo "unknown arg: $1" >&2; exit 2;; esac; done
for v in ARCH CARD DEV BASE MODEL; do [ -n "${!v}" ] || { echo "missing --${v,,}" >&2; exit 2; }; done

SRC="$MAIN/.aw/sw_card$CARD"
[ -d "$SRC" ] || { echo "no worktree $SRC" >&2; exit 2; }
TIP=$(git -C "$SRC" rev-parse "loop/card$CARD" 2>/dev/null) || { echo "no loop/card$CARD" >&2; exit 2; }
BREF=$(git -C "$SRC" rev-parse "$BASE" 2>/dev/null) || { echo "no ref $BASE" >&2; exit 2; }
NW=$(git -C "$SRC" rev-list --count "$BASE".."loop/card$CARD" 2>/dev/null)
echo "[rollover_stack] arch=$ARCH card=$CARD stack=$NW wins  tip=${TIP:0:8}  base=${BREF:0:8}  promote=$PROMOTE"
[ "${NW:-0}" -gt 0 ] || { echo "[rollover_stack] nothing to roll over"; exit 0; }
[ "$TIP" != "$BREF" ] || { echo "[rollover_stack] tip==base"; exit 0; }

BASED="/tmp/v2b_base_c$CARD"
[ -s "$BASED" ] || { echo "[rollover_stack] FATAL: no baseline daemon cache $BASED (build+cache the baseline first)" >&2; exit 1; }

# ---- build STACK daemon from the DETACHED tip SHA ----
WT="$MAIN/.aw/sw_rollover_c$CARD"
git -C "$MAIN" worktree remove -f "$WT" 2>/dev/null
git -C "$MAIN" worktree add -f --detach "$WT" "$TIP" >/dev/null 2>&1 || { echo "[rollover_stack] worktree add failed" >&2; exit 1; }
STACKD="/tmp/rollover_stack_c$CARD"
( cd "$WT"; export HOME="$PWD/.swhome" CARGO_TARGET_DIR="$PWD/target" PATH="$HOME/.bun/bin:$PATH"
  echo "[rollover_stack] building stack daemon @ ${TIP:0:8} (cold ~2-4min)..."
  if cargo build --release --example daemon --features deltanet -p hipfire-runtime >"/tmp/rollover_build_c$CARD.log" 2>&1 && [ -x target/release/examples/daemon ]; then
    cp target/release/examples/daemon "$STACKD"; echo "[rollover_stack] stack daemon built"
  else echo "[rollover_stack] STACK BUILD FAILED (see /tmp/rollover_build_c$CARD.log)"; exit 3; fi )
RC=$?
git -C "$MAIN" worktree remove -f "$WT" 2>/dev/null
[ $RC -eq 0 ] || { echo "[rollover_stack] abort: stack build failed" >&2; exit 1; }

# ---- warm A/B measure (fresh proc per sample; v2p's m1 decode-tok/s + light coherence) ----
cd "$SRC"
PROMPT='Write a detailed paragraph about the history and future of computing.'
meas(){ # $1=daemon -> "tok_s COH"
  printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}\n{"type":"generate","id":"w1","prompt":"Explain hash maps briefly.","temperature":0.0,"max_tokens":32}\n{"type":"generate","id":"m1","prompt":"%s","temperature":0.0,"max_tokens":128}\n{"type":"unload"}\n' "$MODEL" "$PROMPT" \
    | HIP_VISIBLE_DEVICES=$DEV "$1" 2>/dev/null | python3 -c '
import json,sys
tok=0.0; txt=""
for l in sys.stdin:
    try: d=json.loads(l)
    except: continue
    if d.get("id")!="m1": continue
    if d.get("type")=="done": tok=d.get("kernel_decode_tok_s") or d.get("decode_tok_s") or tok
    if d.get("type")=="token": txt+=d.get("text","")
w=txt.split(); u=(len(set(w))/len(w)) if w else 0
coh="OK" if (len(w)>15 and u>0.35) else ("LIVE" if tok>0 else "BAD")
print(f"{tok} {coh}")'
}
echo "[rollover_stack] warming both daemons..."; meas "$BASED" >/dev/null; meas "$STACKD" >/dev/null
declare -a B S; COH=OK
for i in $(seq 1 "$SAMPLES"); do
  read bt bc < <(meas "$BASED"); read st sc < <(meas "$STACKD")
  B+=("$bt"); S+=("$st")
  { [ "$bc" = BAD ] || [ "$sc" = BAD ]; } && COH=BAD
  printf '  sample %d: base=%s(%s)  stack=%s(%s)\n' "$i" "$bt" "$bc" "$st" "$sc"
done
read BM SM D < <(python3 -c "
import statistics as st
b=[float(x) for x in '${B[*]}'.split() if x]; s=[float(x) for x in '${S[*]}'.split() if x]
bm=st.median(b) if b else 0; sm=st.median(s) if s else 0
print(bm, sm, (100*(sm-bm)/bm) if bm else 0)")
printf '[rollover_stack] baseline_median=%.1f  stack_median=%.1f  cumulative Δ=%.2f%%  coherence=%s  margin_req=%s%%\n' "$BM" "$SM" "$D" "$COH" "$MARGIN"

# ---- GATE ----
PASS=$(python3 -c "print(1 if (${D:-0} >= $MARGIN and '$COH'!='BAD') else 0)")
if [ "$PASS" != 1 ]; then
  echo "[rollover_stack] REJECT — Δ=${D}% (need ≥${MARGIN}%), coherence=$COH. NOT promoting; stack stays on loop/card$CARD (safety-branched)."
  exit 0
fi
echo "[rollover_stack] PASS — cumulative +${D}% and coherent."
if [ "$PROMOTE" != 1 ]; then
  echo "[rollover_stack] DRY-RUN (no --promote). Would: baseline $BASE -> ${TIP:0:8}, rebuild cache, re-anchor loop. Re-run with --promote (after pausing the loop)."
  exit 0
fi

# ---- PROMOTE (caller must have paused the loop's codex) ----
LASTV=$(git -C "$SRC" tag -l "baseline_v*_$ARCH" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1); NEXTV=$(( ${LASTV:-3} + 1 ))
git -C "$SRC" branch -f "loop/card${CARD}_recovered" "$TIP" 2>/dev/null   # preserve stack
git -C "$SRC" branch -f "$BASE" "$TIP"
git -C "$SRC" tag -f "baseline_v${NEXTV}_$ARCH" "$TIP" 2>/dev/null
cp "$STACKD" "$BASED"                                                     # rebuild PREBUILT_BASE cache
git -C "$SRC" branch -f "loop/card$CARD" "$TIP"                           # loop re-anchors at new base (0 new wins)
echo "[rollover_stack] PROMOTED: $BASE -> ${TIP:0:8} (tag baseline_v${NEXTV}_$ARCH); cache rebuilt; loop re-anchored; stack preserved on loop/card${CARD}_recovered."

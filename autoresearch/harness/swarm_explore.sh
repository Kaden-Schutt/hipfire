#!/usr/bin/env bash
# swarm_explore.sh — launch N PARALLEL explore agents for one arch. Each worker is an independent
# driver_v3 (its own worktree, wins-branch loop/<arch>_w<i>, substituted prompt, exhaustion, progress).
# Workers that share a GPU (same dev) share a per-GPU lock so their CERTIFIES serialize (no measurement
# contention) while their BUILDS + codex reasoning overlap; workers on different devs run fully parallel.
# Wins accumulate on the per-worker branches; rollover_serve.sh composes + coherence-folds them later.
#   env: ARCH MODEL K CAP  WORKERS="slot:dev:drmcard slot:dev:drmcard ..."  BASE_TIP(branch to fork)
set -u
ARCH="${ARCH:?ARCH required}"; MODEL="${MODEL:?MODEL required}"
K="${K:-10}"; CAP="${CAP:-50}"; WORKERS="${WORKERS:?WORKERS required}"
MAIN="$HOME/hipfire"; HARN="$MAIN/autoresearch/harness"; STATE="$MAIN/autoresearch/state"
BASE_TIP="${BASE_TIP:-loop/$ARCH}"; BASEP="$HARN/loop_round_prompt_$ARCH.txt"
mkdir -p "$STATE/cache"
[ -f "$BASEP" ] || { echo "no base prompt $BASEP"; exit 1; }
i=0
for spec in $WORKERS; do
  s="${spec%%:*}"; rest="${spec#*:}"; dev="${rest%%:*}"; drm="${rest#*:}"
  WT="$MAIN/.aw/sw_card$s"
  [ -d "$WT" ] || { echo "w$i SKIP: no worktree sw_card$s"; i=$((i+1)); continue; }
  BR="loop/${ARCH}_w${i}"
  git -C "$WT" rev-parse --verify -q "$BR" >/dev/null 2>&1 || git -C "$WT" branch "$BR" "$BASE_TIP" 2>/dev/null
  # per-worker prompt: retarget dev, worktree slot, DRM card, wins-branch, label prefix, progress log
  P="$STATE/prompt_${ARCH}_w${i}.txt"
  sed -e "s#$ARCH 1 2 #$ARCH $dev $s #g" -e "s#$ARCH 0 0 #$ARCH $dev $s #g" \
      -e "s#\.aw/sw_card2#.aw/sw_card$s#g" -e "s#\.aw/sw_card0#.aw/sw_card$s#g" \
      -e "s#drm/card1/#drm/card$drm/#g" -e "s#drm/card0/#drm/card$drm/#g" \
      -e "s#(DRM card1)#(DRM card$drm)#g" -e "s#(DRM card0)#(DRM card$drm)#g" \
      -e "s#HIP dev 1#HIP dev $dev#g" -e "s#HIP dev 0#HIP dev $dev#g" \
      -e "s#BASELINE_REF=loop/$ARCH #BASELINE_REF=$BR #g" \
      -e "s#R<round>c2#R<round>c$s#g" -e "s#R<round>c0#R<round>c$s#g" \
      -e "s#loop_progress_$ARCH.log#loop_progress_${ARCH}_w${i}.log#g" \
      "$BASEP" > "$P"
  rm -f "$STATE/loop_driver_${ARCH}_w${i}.done"
  env ARCH="$ARCH" CARDS="$s" LOOP_BRANCH="$BR" BASELINE_REF="$BR" K="$K" CAND_WALL=3.0 \
      MODEL="$MODEL" EXH="$STATE/exhaustion_${ARCH}_w${i}.json" \
      LOOP_PROGRESS="$STATE/loop_progress_${ARCH}_w${i}.log" PROMPT="$P" \
      HIPFIRE_GPU_LOCKFILE="/tmp/hipfire-gpu-${ARCH}-dev${dev}.lock" \
      LOG="$STATE/loop_driver_${ARCH}_w${i}.log" ROLLOVER="$HARN/noop_rollover.sh" \
      AGENT_HARNESS="${AGENT_HARNESS:-codex}" AGENT_MODEL="${AGENT_MODEL:-}" AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-100}" \
      GROK_BIN="${GROK_BIN:-$HOME/.local/bin/grok}" \
      PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.bun/bin:$PATH" \
      setsid nohup bash "$HARN/v2/driver_v3.sh" "$CAP" >/dev/null 2>&1 </dev/null &
  echo "w$i: slot=$s dev=$dev drm=card$drm branch=$BR pid=$! lock=dev${dev}"
  i=$((i+1))
  sleep 1
done
echo "SWARM $ARCH launched: $i workers (K=$K cap=$CAP)"

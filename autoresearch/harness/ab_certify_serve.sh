#!/usr/bin/env bash
# ab_certify_serve.sh — the v2 DRIVER-CALLABLE certify. Wires the loop to the serve_harness / CLI
# serve path (NO raw-daemon voodoo). Same arg shape as ab_certify_v2p so it's a drop-in for the
# loop prompt. Builds the advancing baseline B_a (= loop/cardN tip) + the variant daemon, runs the
# three-arm gate (ab_certify_serve.py -> serve_runner -> serve_harness), and on WIN advances B_a
# (the winning daemon becomes the next baseline) + commits to loop/cardN. Prints the verdict row.
#
#   args: <arch> <dev> <card> <model> <kernel> <label> <variant.hip>
#   env : SEEDS (default 12), KV (default q8), PROMPTS_FILE (guard set)
set -u
ARCH=$1 DEV=$2 CARD=$3 MODEL=$4 KERNEL=$5 LABEL=$6 VARIANT=$7
MAIN="$HOME/hipfire"
WT="${WT_OVERRIDE:-$MAIN/.aw/sw_card${CARD}}"
STATE="$MAIN/autoresearch/state"; CACHE="$STATE/cache"; HARN="$MAIN/autoresearch/harness"
LEDGER="$MAIN/autoresearch/ledger/swarm_${ARCH}_${KERNEL}.jsonl"
SEEDS="${SEEDS:-12}"; KV="${KV:-q8}"
PF_ARG=""; [ -n "${PROMPTS_FILE:-}" ] && PF_ARG="--prompts-file $PROMPTS_FILE"
mkdir -p "$CACHE"
cd "$WT" 2>/dev/null || { echo "{\"label\":\"$LABEL\",\"verdict\":\"BUILD_FAIL\",\"error\":\"no worktree $WT\"}"; exit 0; }
# per-worker isolated build home (parallel builds must not share kernels/src + target)
export HOME_ORIG="$HOME"; export HOME="$WT/.swhome"; mkdir -p "$HOME"
export CARGO_TARGET_DIR="$WT/target"; export PATH="$HOME_ORIG/.bun/bin:$PATH"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME_ORIG/.rustup}" CARGO_HOME="${CARGO_HOME:-$HOME_ORIG/.cargo}"

# B_a = the agent's ADVANCING baseline = loop/cardN tip (carries its banked wins). The variant is
# B_a's kernels with THIS kernel swapped; the gate measures variant-vs-B_a; a WIN advances B_a.
BASE_REF="loop/card${CARD}"
BASE_SHA=$(git rev-parse --short "$BASE_REF" 2>/dev/null || echo "${ARCH}-base")
KSRC="kernels/src/${KERNEL}.hip"
BA_DAEMON="$CACHE/base_${ARCH}_c${CARD}"
VAR_DAEMON="/tmp/var_serve_c${CARD}"
DB="target/release/examples/daemon"
build(){ cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 \
         | grep -qiE "^error|error\[" && return 1; return 0; }

# reset to B_a's kernels
git checkout "$BASE_REF" -- kernels/src/ 2>/dev/null || git checkout -- kernels/src/ 2>/dev/null
git clean -fdq kernels/src/ 2>/dev/null

# --- B_a daemon: cached, else build it from loop/cardN ---
if [ ! -s "$BA_DAEMON" ]; then
  build || { echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"BASELINE_BUILD_FAIL\"}"; exit 0; }
  cp "$DB" "$BA_DAEMON"
fi
# --- variant daemon: B_a + this kernel swapped ---
cp "$VARIANT" "$KSRC"
if ! build; then
  git checkout "$BASE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
  echo "{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"VARIANT_BUILD_FAIL\"}"; exit 0
fi
cp "$DB" "$VAR_DAEMON"
git checkout "$BASE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null

# --- three-arm gate via serve_harness (GPU-lock serialized) ---
GPULK="$MAIN/scripts/gpu-lock.sh"; [ -f "$GPULK" ] && { . "$GPULK"; gpu_acquire "certserve_${ARCH}_c${CARD}_${LABEL}" >/dev/null 2>&1; }
ROW=$(HIP_VISIBLE_DEVICES=$DEV python3 "$HARN/ab_certify_serve.py" \
        --arch "$ARCH" --dev "$DEV" --kernel "$KERNEL" --label "$LABEL" --model "$MODEL" \
        --base-daemon "$BA_DAEMON" --var-daemon "$VAR_DAEMON" --base-ref "$BASE_SHA" \
        --seeds "$SEEDS" --kv "$KV" $PF_ARG 2>"/tmp/certserve_c${CARD}.err")
[ -f "$GPULK" ] && gpu_release >/dev/null 2>&1
[ -n "$ROW" ] || ROW="{\"arch\":\"$ARCH\",\"label\":\"$LABEL\",\"verdict\":\"INCONCLUSIVE\",\"error\":\"gate produced no row (see /tmp/certserve_c${CARD}.err)\"}"
VERDICT=$(printf '%s' "$ROW" | python3 -c "import json,sys;print(json.load(sys.stdin).get('verdict','?'))" 2>/dev/null || echo "?")

# --- WIN -> commit to loop/cardN + advance B_a (winning daemon becomes the new baseline) ---
COMMITTED=""
if [ "$VERDICT" = "WIN" ]; then
  cp "$VARIANT" "$KSRC"; git add "$KSRC" 2>/dev/null
  git -c user.email=151092359+Kaden-Schutt@users.noreply.github.com -c user.name="Kaden Schutt" \
    commit -q -m "WIN(serve) $LABEL $KERNEL" 2>/dev/null && COMMITTED=$(git rev-parse --short HEAD)
  git checkout "$BASE_REF" -- kernels/src/ 2>/dev/null; git clean -fdq kernels/src/ 2>/dev/null
  cp "$VAR_DAEMON" "$BA_DAEMON"           # ADVANCE B_a
fi
rm -f "$VAR_DAEMON"

# --- ledger + emit (annotate win_commit) ---
mkdir -p "$(dirname "$LEDGER")"
ROW=$(printf '%s' "$ROW" | python3 -c "import json,sys;r=json.load(sys.stdin);r['win_commit']=('$COMMITTED' or None);print(json.dumps(r))" 2>/dev/null || printf '%s' "$ROW")
printf '%s\n' "$ROW" >> "$LEDGER"
printf '%s\n' "$ROW"

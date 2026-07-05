#!/usr/bin/env bash
# Fold-with-forks (anti-clobber): promote the drained queue's branch wins to ARCH-FORKS (<k>.<arch>.hip +
# kernels.rs + dispatch, reverting the shared <k>.hip), A/B the combined fork state vs baseline, and advance
# loop_baseline_$ARCH + re-census if it holds. Replaces the shared-source rollover for the fork model.
cd "$HOME/hipfire" || exit 1
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
ARCH="${ARCH:?}"; CARDS="${CARDS:?}"; BASELINE_REF="${BASELINE_REF:?}"; MODEL="${MODEL:?}"
FIRST=${CARDS%% *}; PW="$HOME/hipfire/.aw/sw_card$FIRST"; DB="$PW/target/release/examples/daemon"
FLOOR="${FOLD_FLOOR:-0.3}"
log(){ echo "[fold-forks $(date -u +%T)] $*" >> /tmp/loop_driver.log; }

# 1. collect the winning variant sources from the branch (one file per won kernel)
rm -f /tmp/promote_var_*.hip
git -C "$PW" log "$BASELINE_REF..loop/card$FIRST" --format='%s|%H' 2>/dev/null | while IFS='|' read s h; do
  case "$s" in "WIN "*) kern=$(echo "$s"|awk '{print $3}'); git -C "$PW" show "$h:kernels/src/$kern.hip" > "/tmp/promote_var_$kern.hip" 2>/dev/null;; esac
done
WINS=$(ls /tmp/promote_var_*.hip 2>/dev/null | wc -l)
[ "$WINS" -gt 0 ] || { log "no wins on branch -> nothing to fold"; exit 0; }
log "promoting $WINS won kernel(s) to arch-forks"

# 2. codex promotes each win to a FORK (revert shared, register fork const + dispatch, build-verify)
( cd "$PW" && timeout 2600 codex exec --dangerously-bypass-approvals-and-sandbox \
  "ARCH=$ARCH BASELINE_REF=$BASELINE_REF. Winning kernel variant sources are in /tmp/promote_var_<kernel>.hip (the KERNEL name is the filename between promote_var_ and .hip). Promote EACH to an arch-specific FORK (or keep shared only if universal per the spec) and REVERT the shared kernels/src/<kernel>.hip to $BASELINE_REF so ONLY the fork carries the change. Verify 'cargo build --release --workspace --all-targets --locked' ONCE at the end. Commit the fork files + rdna-compute/src/kernels.rs changes to this branch.
$(cat /tmp/promote_fork_prompt.txt)" ) >> /tmp/loop_driver.log 2>&1

# 3. build baseline + branch-HEAD (fork) daemons, isolate-decode-window A/B
build_d(){ ( cd "$PW" && git checkout -q "$1" 2>/dev/null; CARGO_TARGET_DIR="$PW/target" cargo build --release --example daemon --features deltanet -p hipfire-runtime >/dev/null 2>&1 && cp "$DB" "$2" ); }
build_d "$BASELINE_REF" /tmp/fold_base_$ARCH
build_d "loop/card$FIRST" /tmp/fold_fork_$ARCH
( cd "$PW" && git checkout -q "$BASELINE_REF" 2>/dev/null )
[ -s /tmp/fold_fork_$ARCH ] || { log "fork daemon build FAILED -> NOT advancing (promote likely broke the build)"; exit 1; }
measure(){ printf '{"type":"load","model":"%s","params":{"max_seq":2048,"kv_mode":"q8"}}
{"type":"generate","id":"w","prompt":"Explain hash maps.","temperature":0.0,"max_tokens":32}
{"type":"generate","id":"m","prompt":"Write a detailed paragraph about the history and future of computing.","temperature":0.0,"max_tokens":128}
{"type":"unload"}
' "$MODEL" | HIP_VISIBLE_DEVICES=$FIRST timeout 220 "$1" 2>/dev/null | grep '"id":"m"' | grep done | python3 -c "import json,sys
for l in sys.stdin: print(json.loads(l).get('decode_tok_s',0))" | tail -1; }
B=(); FK=()
for r in 1 2 3 4 5 6; do B+=("$(measure /tmp/fold_base_$ARCH)"); FK+=("$(measure /tmp/fold_fork_$ARCH)"); done
read DELTA FR < <(python3 -c "
import statistics as st
b=[float(x) for x in '${B[*]}'.split() if x]; f=[float(x) for x in '${FK[*]}'.split() if x]
if b and f:
  mb,mf=st.median(b),st.median(f); ff=sum(1 for x in f for y in b if x>y)/(len(f)*len(b)); print(f'{100*(mf/mb-1):.2f} {ff:.3f}')
else: print('0 0')")
log "combined fork A/B: delta=${DELTA}% f=${FR} (base med vs fork-branch med, isolate-decode-window)"

# 4. advance loop_baseline + re-census if it holds
if awk "BEGIN{exit !($DELTA>$FLOOR && $FR>=0.85)}"; then
  git -C "$PW" branch -f "loop_baseline_$ARCH" "loop/card$FIRST"
  NEWV=$(( $(grep -oE 'baseline_v[0-9]+' "/tmp/baseline_manifest_${ARCH}.txt" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1 || echo 0) + 1 ))
  echo "baseline_v${NEWV} sha=$(git -C "$PW" rev-parse --short loop/card$FIRST) composed_delta=${DELTA} folded=arch-forks $(date -u +%FT%TZ)" >> "/tmp/baseline_manifest_${ARCH}.txt"
  HIPFIRE_REPO="$PW" timeout 1400 bash /tmp/oracle_profile.sh "$ARCH" "$FIRST" "$FIRST" "$MODEL" 40 > "/tmp/bod_${ARCH}.json.new" 2>/dev/null \
    && python3 -c "import json,sys;sys.exit(0 if json.load(open('/tmp/bod_${ARCH}.json.new')).get('rows') else 1)" 2>/dev/null \
    && mv "/tmp/bod_${ARCH}.json.new" "/tmp/bod_${ARCH}.json"
  log "ADVANCED loop_baseline_$ARCH -> baseline_v${NEWV} (+${DELTA}%); BOD re-censused"
else
  log "combined fork A/B below floor (+${DELTA}%, f=${FR}) -> NOT advancing"
fi
rm -f /tmp/fold_base_$ARCH /tmp/fold_fork_$ARCH
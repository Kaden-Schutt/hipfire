#!/usr/bin/env bash
# One-time catch-up: the v1 rollover advanced the baseline BEFORE the census was baked
# into rollover.sh, so /tmp/bod_gfx1201.json still reflects the pre-v1 profile (Codex is
# guessing on stale targets). Stop loop, build the baseline_v1 daemon, re-census, refresh
# the bod, restart. Runs ON hiptrx.
cd "$HOME/hipfire" || exit 1
export PATH="$HOME/.local/bin:$HOME/.bun/bin:$PATH"
for p in $(pgrep -f "loop_driver_v2.sh" 2>/dev/null); do [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null; done
pkill -9 -f "codex exec" 2>/dev/null; pkill -9 -f "ab_certify_swarm" 2>/dev/null; pkill -9 -f "release/examples/daemon" 2>/dev/null
sleep 2
WT="$PWD/.aw/sw_card0"
MODEL=/home/kaden/.hipfire/models/qwen3.6-35b-a3b.mq4r
git -C "$WT" checkout -- kernels/src/ 2>/dev/null; git -C "$WT" clean -fdq kernels/src/ 2>/dev/null
echo "=== building baseline_v1 daemon (sw_card0 @ $(git -C "$WT" rev-parse --short HEAD)) ==="
( cd "$WT" && CARGO_TARGET_DIR="$WT/target" cargo build --release --example daemon --features deltanet -p hipfire-runtime 2>&1 | tail -1 )
echo "=== census against baseline_v1 ==="
HIPFIRE_REPO="$WT" timeout 1400 bash /tmp/oracle_profile.sh gfx1201 0 0 "$MODEL" 40 > /tmp/bod_new.json 2>/tmp/bod_census.err
if [ -s /tmp/bod_new.json ] && python3 -c "import json,sys;d=json.load(open('/tmp/bod_new.json'));sys.exit(0 if d.get('rows') else 1)" 2>/dev/null; then
  cp /tmp/bod_gfx1201.json /tmp/bod_gfx1201.v0.json 2>/dev/null
  cp /tmp/bod_new.json /tmp/bod_gfx1201.json
  echo "BOD REFRESHED (baseline_v1). top-8 hot kernels now:"
  python3 -c "import json;d=json.load(open('/tmp/bod_gfx1201.json'));[print(f'  {round(r[\"wall_pct\"],1)}%  {r[\"kernel\"][:36]:<36} {r.get(\"roofline\",\"\")}') for r in d['rows'][:8]]"
else
  echo "CENSUS FAILED (keeping prior bod): $(head -3 /tmp/bod_census.err)"
fi
echo 0 > /tmp/rollover_dryrun
DUR=$(( $(date -d "2026-07-05 01:46:00" +%s) - $(date +%s) )); [ "$DUR" -lt 60 ] && DUR=3600
setsid bash /tmp/loop_driver_v2.sh "$DUR" </dev/null >/tmp/loop_driver_main.out 2>&1 &
disown 2>/dev/null
echo "CATCHUP_DONE restarted pid=$! dur=${DUR}s"

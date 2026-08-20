#!/bin/bash
# Per-kernel bt2 isolation A/B test.
# Measures the marginal contribution of each bt2 variant by incrementally
# enabling them one at a time, always relative to the full baseline.
#
# Arms:
#   A: baseline (all plain WMMA) — HIPFIRE_BT2_DISABLE=1
#   B: gate_up bt2 only          — HIPFIRE_BT2_DISABLE=1 HIPFIRE_GATE_UP_VARIANT=bt2
#   C: gate_up + qkvza bt2       — + HIPFIRE_QKVZA_BT2_FORCE=1
#   D: all bt2 (gate_up + qkvza + ksplit_det) — HIPFIRE_BT2_DISABLE=0
#
# 3 sessions per arm, 20 prefill runs per session, alternating A/B/A/C/A/D
# to control for thermal drift. 60 samples per arm.
set -euo pipefail

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-4b.mq4}"
OUTPUT="${2:-.codeinsight+research/kernel-tune/deep_ab_isolation.json}"
BENCH="./target/release/examples/bench_qwen35_mq4"
SESSIONS=3
PREFILL_RUNS=20
WARMUP=20
DPM_WARMUP=20

mkdir -p "$(dirname "$OUTPUT")"

MODEL_MD5=$(md5sum "$MODEL" | awk '{print $1}')
BENCH_MD5=$(md5sum "$BENCH" | awk '{print $1}')
GIT_COMMIT=$(git rev-parse HEAD)

echo "=== Per-Kernel bt2 Isolation Test ==="
echo "Sessions: $SESSIONS x $PREFILL_RUNS = $((SESSIONS * PREFILL_RUNS)) samples per arm"
echo ""

python3 -c "
import json
data = {
    'config': {
        'model_md5': '$MODEL_MD5', 'bench_md5': '$BENCH_MD5', 'git_commit': '$GIT_COMMIT',
        'sessions': $SESSIONS, 'prefill_runs': $PREFILL_RUNS, 'warmup': $WARMUP,
    },
    'arms': {'baseline': [], 'gate_up_only': [], 'gate_up_qkvza': [], 'all_bt2': []},
}
with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)
"

run_arm() {
    local arm_name="$1"
    local env_vars="$2"

    echo "  [$arm_name] running..."

    local output
    output=$(env $env_vars \
        HIPFIRE_KV_MODE=q8 HIPFIRE_VERIFY_GRAPH=0 HIPFIRE_DPM_WARMUP_SECS=$DPM_WARMUP \
        $BENCH "$MODEL" --prefill 32 --prefill-runs $PREFILL_RUNS --gen 128 --warmup $WARMUP 2>&1)

    local prefill_summary
    prefill_summary=$(echo "$output" | grep "PREFILL_SUMMARY" | grep -oP 'prefill_tok_s=\K\d+\.\d+' || echo "0")
    local run_samples
    run_samples=$(echo "$output" | grep "run.*tok/s" | grep -oP '\d+\.\d+(?= tok/s)' || true)

    python3 -c "
import json
with open('$OUTPUT', 'r') as f:
    data = json.load(f)
run_samples = [float(x) for x in '''$run_samples'''.strip().split('\n') if x.strip()]
data['arms']['$arm_name'].extend(run_samples)
with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)
print(f'    $arm_name: prefill=$prefill_summary runs={len(run_samples)}')
"
}

# Alternating: A, B, A, C, A, D per session cycle
for session in $(seq 1 $SESSIONS); do
    run_arm "baseline"       "HIPFIRE_BT2_DISABLE=1"
    run_arm "gate_up_only"   "HIPFIRE_BT2_DISABLE=1 HIPFIRE_GATE_UP_VARIANT=bt2"
    run_arm "baseline"       "HIPFIRE_BT2_DISABLE=1"
    run_arm "gate_up_qkvza"  "HIPFIRE_BT2_DISABLE=1 HIPFIRE_GATE_UP_VARIANT=bt2 HIPFIRE_QKVZA_BT2_FORCE=1"
    run_arm "baseline"       "HIPFIRE_BT2_DISABLE=1"
    run_arm "all_bt2"        "HIPFIRE_BT2_DISABLE=0"
done

echo ""
echo "=== Isolation Analysis ==="
python3 << 'PYEOF'
import json, statistics, math

with open('.codeinsight+research/kernel-tune/deep_ab_isolation.json', 'r') as f:
    data = json.load(f)

def stats(vals, name):
    if not vals:
        print(f'  {name}: NO DATA')
        return None
    med = statistics.median(vals)
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0
    cv = (stdev / mean * 100) if mean > 0 else 0
    print(f'  {name:20s}: n={len(vals):3d} median={med:7.1f} mean={mean:7.1f} stdev={stdev:5.1f} cv={cv:.2f}%')
    return {'median': med, 'mean': mean, 'stdev': stdev}

print()
arms = data['arms']
base = stats(arms['baseline'], 'baseline (plain)')
gu   = stats(arms['gate_up_only'], 'gate_up bt2 only')
guq  = stats(arms['gate_up_qkvza'], 'gate_up+qkvza bt2')
allb = stats(arms['all_bt2'], 'all bt2')

print()
if base and gu and guq and allb:
    print('  Incremental contributions (median):')
    print(f'    baseline -> gate_up:       {gu["median"]-base["median"]:+7.1f} tok/s  ({(gu["median"]-base["median"])/base["median"]*100:+5.2f}%)')
    print(f'    gate_up -> +qkvza:         {guq["median"]-gu["median"]:+7.1f} tok/s  ({(guq["median"]-gu["median"])/gu["median"]*100:+5.2f}%)')
    print(f'    +qkvza -> +ksplit_det:     {allb["median"]-guq["median"]:+7.1f} tok/s  ({(allb["median"]-guq["median"])/guq["median"]*100:+5.2f}%)')
    print(f'    baseline -> all bt2:       {allb["median"]-base["median"]:+7.1f} tok/s  ({(allb["median"]-base["median"])/base["median"]*100:+5.2f}%)')

    # Welch t-test for each arm vs baseline
    print()
    print('  Significance vs baseline (Welch t-test):')
    for name, arm_stats, vals in [('gate_up only', gu, arms['gate_up_only']),
                                   ('gate_up+qkvza', guq, arms['gate_up_qkvza']),
                                   ('all bt2', allb, arms['all_bt2'])]:
        n1, n2 = len(arms['baseline']), len(vals)
        se = math.sqrt(base['stdev']**2/n1 + arm_stats['stdev']**2/n2)
        t = (arm_stats['mean'] - base['mean']) / se if se > 0 else 0
        sig = 'p<0.01 ***' if abs(t) > 2.576 else 'p<0.05 **' if abs(t) > 1.96 else 'p<0.10 *' if abs(t) > 1.645 else 'n.s.'
        print(f'    {name:20s}: t={t:7.3f}  {sig}')

data['summary'] = {
    'baseline': base, 'gate_up_only': gu, 'gate_up_qkvza': guq, 'all_bt2': allb,
}
with open('.codeinsight+research/kernel-tune/deep_ab_isolation.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f'\nResults saved to .codeinsight+research/kernel-tune/deep_ab_isolation.json')
PYEOF

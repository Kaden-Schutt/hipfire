#!/bin/bash
# Deep A/B testing harness for bt2 kernel variants.
#
# Methodology:
#   - Alternating A/B/A/B runs to control for thermal/DPM drift
#   - 20 warmup + 20 measurement prefill runs per session (in-process)
#   - 5 fresh-process sessions per arm (5 × 20 = 100 samples per arm)
#   - HIPFIRE_VERIFY_GRAPH=0 for tighter stdev
#   - HIPFIRE_DPM_WARMUP_SECS=20 for full thermal settlement
#   - Records all samples + identity hashes to JSON
#
# Usage: scripts/deep_ab_bt2.sh [model_path] [output_json]
set -euo pipefail

MODEL="${1:-$HOME/.hipfire/models/qwen3.5-4b.mq4}"
OUTPUT="${2:-.codeinsight+research/kernel-tune/deep_ab_results.json}"
BENCH="./target/release/examples/bench_qwen35_mq4"
SESSIONS=5
PREFILL_RUNS=20
WARMUP=20
DPM_WARMUP=20

mkdir -p "$(dirname "$OUTPUT")"

# Record identity hashes
MODEL_MD5=$(md5sum "$MODEL" | awk '{print $1}')
BENCH_MD5=$(md5sum "$BENCH" | awk '{print $1}')
GIT_COMMIT=$(git rev-parse HEAD)
GPU_ARCH=$(HIPFIRE_KV_MODE=q8 $BENCH "$MODEL" --prefill 1 --gen 1 --warmup 0 2>&1 | grep "GPU:" | awk '{print $2}' || echo "gfx1100")

echo "=== Deep A/B Testing: bt2 vs baseline ==="
echo "Model: $MODEL (md5: $MODEL_MD5)"
echo "Bench: $BENCH (md5: $BENCH_MD5)"
echo "Git: $GIT_COMMIT"
echo "GPU: $GPU_ARCH"
echo "Sessions: $SESSIONS × $PREFILL_RUNS runs = $((SESSIONS * PREFILL_RUNS)) samples per arm"
echo "Output: $OUTPUT"
echo ""

# Initialize JSON output
python3 -c "
import json
data = {
    'config': {
        'model': '$MODEL',
        'model_md5': '$MODEL_MD5',
        'bench_md5': '$BENCH_MD5',
        'git_commit': '$GIT_COMMIT',
        'gpu_arch': '$GPU_ARCH',
        'sessions': $SESSIONS,
        'prefill_runs': $PREFILL_RUNS,
        'warmup': $WARMUP,
        'dpm_warmup_secs': $DPM_WARMUP,
        'prefill_tokens': 32,
        'gen_tokens': 128,
    },
    'baseline_samples': [],
    'bt2_samples': [],
    'baseline_decode_samples': [],
    'bt2_decode_samples': [],
    'sessions': [],
}
with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)
print('Initialized output file')
"

run_session() {
    local arm="$1"      # "baseline" or "bt2"
    local session="$2"
    local env_prefix=""

    if [ "$arm" = "baseline" ]; then
        env_prefix="HIPFIRE_BT2_DISABLE=1"
    else
        env_prefix="HIPFIRE_BT2_DISABLE=0"
    fi

    echo "  [$arm] session $session/$SESSIONS..."

    # Run bench: 20 warmup, 20 prefill runs, pp=32, gen=128
    # Extract all prefill samples and decode samples
    local output
    output=$(env $env_prefix \
        HIPFIRE_KV_MODE=q8 \
        HIPFIRE_VERIFY_GRAPH=0 \
        HIPFIRE_DPM_WARMUP_SECS=$DPM_WARMUP \
        $BENCH "$MODEL" \
        --prefill 32 --prefill-runs $PREFILL_RUNS \
        --gen 128 --warmup $WARMUP 2>&1)

    # Extract prefill samples (lines like "  run  1: 17.8ms  1794.2 tok/s")
    local prefill_vals
    prefill_vals=$(echo "$output" | grep -oP '\d+\.\d+ tok/s' | grep -oP '^\d+\.\d+' || true)

    # Extract decode summary
    local decode_val
    decode_val=$(echo "$output" | grep "SUMMARY" | grep -oP 'gen_tok_s=\K\d+\.\d+' || echo "0")

    # Extract prefill summary
    local prefill_summary
    prefill_summary=$(echo "$output" | grep "PREFILL_SUMMARY" | grep -oP 'prefill_tok_s=\K\d+\.\d+' || echo "0")

    # Extract individual prefill run samples
    local run_samples
    run_samples=$(echo "$output" | grep "run.*tok/s" | grep -oP '\d+\.\d+(?= tok/s)' || true)

    # Feed to Python for JSON accumulation
    python3 -c "
import json, sys

with open('$OUTPUT', 'r') as f:
    data = json.load(f)

arm = '$arm'
session = $session
prefill_summary = $prefill_summary
decode_val = $decode_val

# Individual run samples
run_samples_str = '''$run_samples'''
run_samples = [float(x) for x in run_samples_str.strip().split('\n') if x.strip()]

# Append to global lists
if arm == 'baseline':
    data['baseline_samples'].extend(run_samples)
    data['baseline_decode_samples'].append(decode_val)
else:
    data['bt2_samples'].extend(run_samples)
    data['bt2_decode_samples'].append(decode_val)

# Session record
data['sessions'].append({
    'arm': arm,
    'session': session,
    'prefill_tok_s': prefill_summary,
    'decode_tok_s': decode_val,
    'run_count': len(run_samples),
    'run_samples': run_samples,
})

with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)

print(f'    {arm}: prefill={prefill_summary:.1f} decode={decode_val:.1f} runs={len(run_samples)}')
" 
}

# Alternating A/B sessions
for session in $(seq 1 $SESSIONS); do
    # Alternate: even sessions start with baseline, odd with bt2
    if [ $((session % 2)) -eq 1 ]; then
        run_session "bt2" "$session"
        run_session "baseline" "$session"
    else
        run_session "baseline" "$session"
        run_session "bt2" "$session"
    fi
done

echo ""
echo "=== Statistical Analysis ==="
python3 -c "
import json, statistics, math

with open('$OUTPUT', 'r') as f:
    data = json.load(f)

baseline = data['baseline_samples']
bt2 = data['bt2_samples']

def stats(vals, name):
    if not vals:
        print(f'  {name}: NO DATA')
        return {}
    med = statistics.median(vals)
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0
    mn = min(vals)
    mx = max(vals)
    cv = (stdev / mean * 100) if mean > 0 else 0
    print(f'  {name}: n={len(vals)} median={med:.1f} mean={mean:.1f} stdev={stdev:.1f} cv={cv:.2f}% min={mn:.1f} max={mx:.1f}')
    return {'median': med, 'mean': mean, 'stdev': stdev, 'min': mn, 'max': mx, 'cv': cv}

print(f'Baseline samples: {len(baseline)}')
print(f'BT2 samples: {len(bt2)}')
print()

bs = stats(baseline, 'Baseline (plain WMMA)')
bt = stats(bt2, 'BT2 (batch-tiled B=2)')
print()

if bs and bt:
    delta_median = bt['median'] - bs['median']
    delta_pct = delta_median / bs['median'] * 100
    print(f'  Delta (median): {delta_median:+.1f} tok/s ({delta_pct:+.2f}%)')
    
    # Welch's t-test (manual computation)
    n1, n2 = len(baseline), len(bt2)
    m1, m2 = bs['mean'], bt['mean']
    s1, s2 = bs['stdev'], bt['stdev']
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    t_stat = (m2 - m1) / se if se > 0 else 0
    # Welch-Satterthwaite degrees of freedom
    if s1 > 0 and s2 > 0:
        df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    else:
        df = n1 + n2 - 2
    print(f'  Welch t-statistic: {t_stat:.4f} (df={df:.1f})')
    print(f'  Effect size (Cohen d): {(m2-m1)/((s1+s2)/2):.3f}')
    
    # 95% CI for the difference
    ci_low = delta_median - 1.96 * se
    ci_high = delta_median + 1.96 * se
    print(f'  95% CI for delta: [{ci_low:+.1f}, {ci_high:+.1f}] tok/s')
    
    # Significance
    if abs(t_stat) > 2.576:
        sig = 'p < 0.01 (highly significant)'
    elif abs(t_stat) > 1.96:
        sig = 'p < 0.05 (significant)'
    elif abs(t_stat) > 1.645:
        sig = 'p < 0.10 (marginally significant)'
    else:
        sig = 'not significant'
    print(f'  Significance: {sig}')

# Decode comparison
print()
bd = [float(x) for x in data['baseline_decode_samples']]
bd2 = [float(x) for x in data['bt2_decode_samples']]
if bd and bd2:
    print(f'  Decode baseline: median={statistics.median(bd):.1f} mean={statistics.mean(bd):.1f}')
    print(f'  Decode bt2:      median={statistics.median(bd2):.1f} mean={statistics.mean(bd2):.1f}')
    dd = statistics.median(bd2) - statistics.median(bd)
    print(f'  Decode delta: {dd:+.1f} tok/s ({dd/statistics.median(bd)*100:+.2f}%)')

# Save summary to JSON
data['summary'] = {
    'baseline_stats': bs,
    'bt2_stats': bt,
    'delta_median': delta_median if bs and bt else 0,
    'delta_pct': delta_pct if bs and bt else 0,
    'welch_t': t_stat if bs and bt else 0,
    'welch_df': df if bs and bt else 0,
    'ci_95': [ci_low, ci_high] if bs and bt else [0, 0],
    'significance': sig if bs and bt else 'N/A',
}
with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)

print(f'\nResults saved to $OUTPUT')
"

#!/usr/bin/env bash
# Fresh-process A/B for WMMA-FA prefill on gfx1100, interleaved.
# Adapted from benchmarks/results/wmma-fa-probe.sh (gfx1151 version).
# Differences: ROCm path is /opt/rocm (Arch), MODEL is the user-validated
# 4B MQ4 by default.
set -u

cd /home/kread/git/hipfire
source ./scripts/gpu-lock.sh
gpu_acquire "wmma-fa-gfx1100-ab" || exit 1

MODEL="${MODEL:-/data/hipfire/qwen3.5-4b.mq4-cuda.hfq}"
N="${N:-5}"
NCTX="${NCTX:-2048}"

OUT="${OUT:-/home/kread/git/hipfire/.tmp/wmma-fa-ab/gfx1100-results.csv}"
mkdir -p "$(dirname "$OUT")"
echo "round,config,prefill_tok_s,model,nctx" > "$OUT"

run_one() {
    local config="$1"
    local wmma_env="$2"
    local round="$3"
    local out
    out=$(timeout 200 env $wmma_env \
        target/release/examples/prefill_microbench \
            --model "$MODEL" --kv-mode asym4 \
            --n-ctx "$NCTX" --warmup-iters 0 --measure-iters 1 \
        2>&1 | grep -oE "prefill [0-9.]+s \([0-9.]+ tok/s\)" \
             | grep -oE "[0-9.]+ tok/s" | head -1 | awk '{print $1}')
    if [ -z "$out" ]; then
        echo "  round=$round config=$config FAIL"
        echo "$round,$config,FAIL,$(basename "$MODEL"),$NCTX" >> "$OUT"
    else
        echo "  round=$round config=$config prefill_tok_s=$out"
        echo "$round,$config,$out,$(basename "$MODEL"),$NCTX" >> "$OUT"
    fi
}

for r in $(seq 1 $N); do
    echo "=== round $r ==="
    # Interleave scalar-first vs wmma-first per round to defeat trend bias.
    if [ $((r % 2)) -eq 0 ]; then
        run_one "wmma"   "HIPFIRE_WMMA_FA=1" $r
        run_one "scalar" ""                  $r
    else
        run_one "scalar" ""                  $r
        run_one "wmma"   "HIPFIRE_WMMA_FA=1" $r
    fi
done

gpu_release

echo
echo "=== summary ==="
python3 - "$OUT" <<'PYEOF'
import csv, statistics, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
scalar = [float(r['prefill_tok_s']) for r in rows if r['config']=='scalar' and r['prefill_tok_s']!='FAIL']
wmma   = [float(r['prefill_tok_s']) for r in rows if r['config']=='wmma'   and r['prefill_tok_s']!='FAIL']
def stats(name, xs):
    if not xs: return f"{name}: no data"
    return f"{name}: n={len(xs)}  median={statistics.median(xs):.2f}  min={min(xs):.2f}  max={max(xs):.2f}  stdev={statistics.pstdev(xs):.2f}"
print(stats("scalar", scalar))
print(stats("wmma  ", wmma))
if scalar and wmma:
    delta = (statistics.median(wmma) - statistics.median(scalar)) / statistics.median(scalar) * 100
    print(f"delta (wmma/scalar median): {delta:+.2f}%")
    by_round = {}
    for r in rows:
        if r['prefill_tok_s']=='FAIL': continue
        by_round.setdefault(r['round'], {})[r['config']] = float(r['prefill_tok_s'])
    pairs = [(d['wmma']-d['scalar']) for d in by_round.values() if 'wmma' in d and 'scalar' in d]
    if pairs:
        m = statistics.mean(pairs)
        s = statistics.pstdev(pairs)
        print(f"paired delta (wmma-scalar): mean={m:+.2f} tok/s  stdev={s:.2f}  n={len(pairs)}")
        if s > 0:
            t = m / (s / max(1, len(pairs)-1)**0.5)
            print(f"paired t-stat: {t:+.2f} (|t|>2 ~= significant at p<0.05)")
PYEOF

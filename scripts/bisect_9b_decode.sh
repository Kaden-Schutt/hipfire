#!/usr/bin/env bash
# Bisect driver for the 9B decode regression between c825dfa (good, 130.8
# tok/s) and 0532579 (bad, 112 tok/s) on gfx1100.
#
# Criterion: 9B MQ4 decode at pp=16 / warmup=3 / gen=30 must hit ≥ 125 tok/s
# (within 5% of the 130.8 baseline). Below that = BAD.
#
# Skips commits whose build fails with exit code 125 (git-bisect convention).

set -u
MODEL="${HIPFIRE_9B_MODEL:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"
THRESHOLD="${BISECT_TOK_S:-125.0}"

cd "$(git rev-parse --show-toplevel)"

commit=$(git rev-parse --short HEAD)
echo "=== bisect at $commit ==="

# Incremental build. Also remove the old binary so a silent build-skip
# can't hand us a stale target/release/examples/bench_qwen35_mq4.
rm -f target/release/examples/bench_qwen35_mq4
build_log=$(cargo build --release --features deltanet -p hipfire-runtime --example bench_qwen35_mq4 2>&1)
build_rc=$?
echo "$build_log" | tail -3
if [ $build_rc -ne 0 ] || [ ! -x target/release/examples/bench_qwen35_mq4 ]; then
    echo "BUILD FAILED — skipping"
    exit 125
fi

# Run the bench. asym3 KV matches speed-gate config.
out=$(HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
    target/release/examples/bench_qwen35_mq4 "$MODEL" \
    --prefill 16 --warmup 3 --gen 30 2>&1)

tok_s=$(echo "$out" | grep -oE 'gen_tok_s=[0-9.]+' | sed 's/gen_tok_s=//')
if [ -z "$tok_s" ]; then
    echo "BENCH FAILED — output:"
    echo "$out" | tail -10
    exit 125
fi

# Compare with bc since floats.
result=$(echo "$tok_s >= $THRESHOLD" | bc -l)
if [ "$result" -eq 1 ]; then
    echo "GOOD ($commit: $tok_s tok/s ≥ $THRESHOLD)"
    exit 0
else
    echo "BAD  ($commit: $tok_s tok/s < $THRESHOLD)"
    exit 1
fi

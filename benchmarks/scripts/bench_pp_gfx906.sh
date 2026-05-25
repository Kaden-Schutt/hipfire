#!/usr/bin/env bash
# Canonical pp{32,64,128,256,512} prefill bench for the gfx906 MMQ kernel.
# Pinned invocation + environment so the 5× pp512 claim in PR #158 is
# reproducible byte-for-byte on any MI50.
#
# The bench uses *synthetic* token input via `bench_qwen35_mq4`'s built-in
# deterministic generator (`prompt_tokens = [0, 1, 2, ..., prefill_len-1]`
# in crates/hipfire-runtime/examples/bench_qwen35_mq4.rs). There is no
# external prompt file — the input is fully determined by `--prefill N`,
# the model file, and the binary commit. The CLAUDE.md prompt-md5 rule
# applies to AR-decode / DFlash where prompt-structure swings τ; for
# prefill throughput on synthetic input, the reproducibility artifact is
# the harness invocation + commit hash, recorded below.
#
# Usage:
#   ./benchmarks/scripts/bench_pp_gfx906.sh /path/to/qwen3.5-9b.mq4
#
# Prints a row per pp{N}, plus the `git rev-parse HEAD` and binary md5.

set -u
MODEL="${1:-$HOME/.hipfire/models/qwen3.5-9b.mq4}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    echo "usage: $0 <path-to-qwen3.5-9b.mq4>" >&2
    exit 2
fi

EXE=./target/release/examples/bench_qwen35_mq4
if [ ! -x "$EXE" ]; then
    echo "building bench_qwen35_mq4..." >&2
    cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 \
        --features deltanet 2>&1 | tail -3
fi

echo "## Reproducibility metadata"
echo "- commit:        $(git rev-parse HEAD)"
echo "- binary md5:    $(md5sum "$EXE" | awk '{print $1}')"
echo "- model:         $MODEL"
echo "- model md5:     $(md5sum "$MODEL" | awk '{print $1}')"
echo "- arch:          $(rocminfo 2>/dev/null | grep -E "^\s+Name:\s+gfx" | head -1 | awk '{print $2}')"
echo

echo "## Prefill throughput (3-run median per N, --warmup 0 --gen 1)"
printf "%-6s %-12s\n" "N" "tok/s"
for N in 32 64 128 256 512; do
    BEST=$(
        for run in 1 2 3; do
            HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
            HIPFIRE_KV_MODE=asym3 HIPFIRE_GRAPH=1 \
                "$EXE" "$MODEL" \
                    --prefill "$N" --prefill-runs 2 --warmup 0 --gen 1 \
                    2>&1 | grep -oE 'prefill_tok_s=[0-9.]+' | tail -1 | cut -d= -f2
        done | sort -n | awk 'NR==2 {print}'  # median of 3
    )
    printf "%-6d %-12s\n" "$N" "$BEST"
done

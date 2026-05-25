#!/usr/bin/env bash
# rocprof-wrap.sh -- wrap a command under rocprofv3 kernel tracing.
#
# Usage:
#   scripts/rocprof-wrap.sh <output-dir> -- <command...>
#
# Sets HIPFIRE_ROCPROF_CSV=<output-dir>/trace_kernel_stats.csv in the child
# environment so that a bench binary built with the rdna-compute rocprof
# integration can pick up the stats file automatically.
#
# The output CSV files are:
#   <output-dir>/trace_kernel_trace.csv   -- per-dispatch kernel trace
#   <output-dir>/trace_kernel_stats.csv   -- per-kernel aggregate stats (what we use)
#   <output-dir>/trace_domain_stats.csv   -- domain-level HIP API stats
#
# Exit code: propagates the wrapped command's exit code.
#
# Example:
#   mkdir -p /tmp/cov-run
#   scripts/rocprof-wrap.sh /tmp/cov-run -- \
#     ./target/release/examples/bench_qwen35_mq4 model.hfq \
#       --prefill 32 --emit-atlas /tmp/cov-run/atlas.jsonl
#   scripts/coverage-audit.py \
#     --internal /tmp/cov-run/atlas.jsonl \
#     --rocprof  /tmp/cov-run/trace_kernel_stats.csv

set -euo pipefail

if [[ $# -lt 3 ]] || [[ "$2" != "--" ]]; then
    echo "Usage: $0 <output-dir> -- <command...>" >&2
    exit 1
fi

OUTPUT_DIR="$1"
shift 2  # consume <output-dir> and --

mkdir -p "$OUTPUT_DIR"

PREFIX="trace"
STATS_CSV="${OUTPUT_DIR}/${PREFIX}_kernel_stats.csv"

# Export so the child bench binary sees it.
export HIPFIRE_ROCPROF_CSV="${STATS_CSV}"

echo "[rocprof-wrap] output dir: ${OUTPUT_DIR}" >&2
echo "[rocprof-wrap] stats CSV:  ${STATS_CSV}" >&2
echo "[rocprof-wrap] command:    $*" >&2

# Run under rocprofv3.
# --kernel-trace   : record per-dispatch kernel execution times
# --stats          : aggregate per-kernel totals into _kernel_stats.csv
# -S               : print a human-readable summary to stderr after the run
# --output-format  : CSV output format (newer rocprofv3 dropped the `-f csv` short form)
# -d <dir>         : output directory
# -o <prefix>      : output filename prefix
exec rocprofv3 \
    --kernel-trace \
    --stats \
    -S \
    --output-format csv \
    -d "${OUTPUT_DIR}" \
    -o "${PREFIX}" \
    -- "$@"

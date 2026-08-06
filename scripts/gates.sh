#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Unified wrapper for the maintained runtime harnesses plus an optional
# fresh-process performance comparison. The retired coherence-gate scripts are
# intentionally not called here.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)" || {
    echo "gates.sh: not a git repo" >&2
    exit 2
}
cd "$ROOT"

MODEL="${HIPFIRE_GATE_MODEL:-${BENCH_MODEL:-}}"
WORK_DIR="${HIPFIRE_GATE_WORK_DIR:-$ROOT/.redline-work/gates}"
RUN_REDLINE=1
RUN_SERVE=1
RUN_PERF=1
PM4=1
PERF_BASE="HEAD~1"

usage() {
    cat <<'EOF'
Usage: scripts/gates.sh --model /path/to/model [options]

Options:
  --model PATH       exact model under test (or HIPFIRE_GATE_MODEL)
  --redline-only     run retained-dispatch capture/parity only
  --serve-only       run user-facing serve battery only
  --no-perf          skip probe_commits.sh
  --perf REF         compare performance against REF (default HEAD~1)
  --aql              shadow the retained AQL path instead of one PM4 IB
  --work-dir PATH    artifact directory
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:?--model requires a path}"; shift ;;
        --redline-only) RUN_REDLINE=1; RUN_SERVE=0 ;;
        --serve-only) RUN_REDLINE=0; RUN_SERVE=1 ;;
        --no-perf) RUN_PERF=0 ;;
        --perf) PERF_BASE="${2:?--perf requires a git ref}"; shift ;;
        --aql) PM4=0 ;;
        --work-dir) WORK_DIR="${2:?--work-dir requires a path}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "gates.sh: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$MODEL" ]; then
    echo "gates.sh: --model (or HIPFIRE_GATE_MODEL) is required" >&2
    exit 2
fi
if [ ! -r "$MODEL" ]; then
    echo "gates.sh: model is not readable: $MODEL" >&2
    exit 2
fi

mkdir -p "$WORK_DIR"

if [ "$RUN_REDLINE" -eq 1 ]; then
    REDLINE_ARGS=(
        --model "$MODEL"
        --decode-context 128
        --decode-iterations 32
        --capture-repeats 2
        --measure-repeats 3
        --shadow-iterations 15
        --out "$WORK_DIR/redline.json"
        --log "$WORK_DIR/redline.log"
    )
    if [ "$PM4" -eq 1 ]; then
        REDLINE_ARGS+=(--pm4)
    fi
    python3 scripts/redline_daemon_harness.py "${REDLINE_ARGS[@]}"
fi

if [ "$RUN_SERVE" -eq 1 ]; then
    python3 scripts/serve_harness.py \
        --model "$MODEL" \
        --mode battery \
        --sampling greedy \
        --max-tokens 128 \
        --out "$WORK_DIR/serve.json"
fi

if [ "$RUN_PERF" -eq 1 ]; then
    BASE_SHA="$(git rev-parse --verify "${PERF_BASE}^{commit}")"
    HEAD_SHA="$(git rev-parse HEAD)"
    BENCH_MODEL="$MODEL" scripts/probe_commits.sh "$BASE_SHA" "$HEAD_SHA"
fi

echo "runtime validation artifacts: $WORK_DIR"

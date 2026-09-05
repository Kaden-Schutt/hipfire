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
# Escha-W2 G1-G6. OFF by default: these are checkpoint-specific (they need the
# escha .hfq and, for G1, the source safetensors tree), so running them against
# whatever `--model` happens to be would either fail or, worse, pass vacuously.
RUN_ESCHA=0
ESCHA_SRC="${ESCHA_SRC:-/data/hipfire-models/escha-35b}"
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
  --escha            ALSO run the Escha-W2 correctness battery G1-G6
  --escha-only       run ONLY G1-G6 (--model must be the escha .hfq)
  --escha-src PATH   source safetensors tree for G1 (or ESCHA_SRC)
                     default /data/hipfire-models/escha-35b

Escha-W2 gates (see docs/plans/escha-w2-port-design.md §10.6):
  G1  verbatim repack: every escha_code tensor byte-identical to source
  G2  GPU tile decode == escha_ref::reconstruct, bit-exact
  G3  the H128 pair == escha_ref, bit-exact, every launch form
  G4  the whole MoE block against escha's moeblk_out.f16 golden
  G4b arch-6 router selects the same experts as escha
  G5  KLD on a fixed teacher-forced corpus, with a negative control
  G6  batched prefill vs the per-token route, whole model
G5 and G6 load the model (37.6 GB resident) and take minutes; G1-G4b do not
need a GPU-resident model beyond the checkpoint and the committed fixtures.
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
        --escha) RUN_ESCHA=1 ;;
        --escha-only) RUN_ESCHA=1; RUN_REDLINE=0; RUN_SERVE=0; RUN_PERF=0 ;;
        --escha-src) ESCHA_SRC="${2:?--escha-src requires a path}"; shift ;;
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

if [ "$RUN_ESCHA" -eq 1 ]; then
    echo "== Escha-W2 correctness battery (G1-G6) =="
    echo "   model:  $MODEL"
    echo "   source: $ESCHA_SRC"
    echo
    # Built once; every gate below is a release example.
    cargo build --release --workspace --all-targets --locked

    echo "-- G1: verbatim repack (expect 80/80 byte-identical) --"
    python3 scripts/escha-verify-roundtrip.py "$ESCHA_SRC" "$MODEL"

    echo "-- G2: tile decode vs escha_ref (expect 0 mismatched) --"
    cargo run --release -p rdna-compute --example test_escha_decode_gpu_vs_cpu

    echo "-- G3: H128 pair vs escha_ref (expect 0 mismatched) --"
    cargo run --release -p rdna-compute --example test_escha_h128_gpu_vs_cpu

    echo "-- G4b: router contract (expect 0/8 differing sets) --"
    cargo run --release -p hipfire-arch-qwen35 \
        --example escha_router_contract -- "$MODEL"

    echo "-- G4: MoE block vs golden (expect F32 1.828e-4/9.673e-6," \
         "Q8_0 2.633e-4/3.027e-5, 0 differing floats on both routes) --"
    cargo run --release -p hipfire-arch-qwen35 \
        --example escha_moe_block_gate -- "$MODEL"

    echo "-- G6: batched prefill vs per-token (expect a stable argmax) --"
    cargo run --release -p hipfire-arch-qwen35 \
        --example escha_prefill_batch_gate -- "$MODEL"

    echo "-- G5: KLD (expect 0.0027576 nats, PPL 7.6585, control 0.000000) --"
    scripts/escha-kld.sh "$MODEL"

    echo "Escha-W2 G1-G6: all green."
fi

echo "runtime validation artifacts: $WORK_DIR"

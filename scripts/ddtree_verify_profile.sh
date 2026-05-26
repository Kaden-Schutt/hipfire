#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Profile batched-verify at b12-k2 (our current winner) vs b22-k4 (Lucebox
# config we match on τ but lose on tok/s). Uses HIPFIRE_HOST_TIMING=1 to
# dump per-cycle host breakdown (launch/h2d/d2h/d2d/memset/ssync/esync
# plus API call counts). Run 3× each for median.
#
# Output: per-config "host timing" lines, easy to diff by hand.
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_PROFILE_MAX:-192}"
RUNS="${HIPFIRE_PROFILE_RUNS:-3}"

if [ ! -x "$EXE" ]; then
    echo "build with: cargo build --release --example dflash_spec_demo --features deltanet" >&2
    exit 2
fi

if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "ddtree-verify-profile" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

CODE_PROMPT='from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'

run_one() {
    local tag="$1"; shift
    local budget="$1"; shift
    local topk="$1"; shift
    echo "### $tag (b=$budget k=$topk) ###"
    for i in $(seq 1 "$RUNS"); do
        echo "-- run $i --"
        HIPFIRE_HOST_TIMING=1 "$EXE" \
            --target "$TARGET_27B" --draft "$DRAFT_27B" \
            --prompt "$CODE_PROMPT" --max "$MAX_TOKENS" --ctx 2048 \
            --kv-mode q8 --no-chatml \
            --ddtree-batched --ddtree-budget "$budget" --ddtree-topk "$topk" 2>&1 | \
            grep -aE '^(emitted:|cycles:|host timing|  launch=|  ssync=|τ=|accept_rate)'
    done
    echo
}

run_one "b12-k2" 12 2
run_one "b22-k4" 22 4

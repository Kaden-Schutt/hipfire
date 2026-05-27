#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Task #93 Phase B seed-prediction oracle data collection.
#
# Runs DFlash spec-decode across three prompt genres (code, prose, instruct)
# with HIPFIRE_DFLASH_SEED_ORACLE=1 set. Parses the per-run summary line:
#   seed-oracle: cycles=N match=M full_accept=F predictable=P overall=X among_predictable=Y
#
# The "among_predictable" rate is what matters — it answers "when the draft
# has a seed guess, how often is it right?" which is the real question for
# inter-cycle pipelining. <70% kills Phase D-E; >=85% is the PRD assumption.
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_ORACLE_MAX:-128}"

if [ ! -x "$EXE" ]; then
    echo "build with: cargo build --release --example dflash_spec_demo --features deltanet" >&2
    exit 2
fi
if [ ! -f "$TARGET_27B" ] || [ ! -f "$DRAFT_27B" ]; then
    echo "27B target or draft model missing" >&2
    exit 2
fi

if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "seed-oracle" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

PROSE_PROMPT="The Roman Empire, at its height, stretched from the windswept moors of northern Britain to the sands of the Arabian peninsula. Its decline was not a single event but a long slow unraveling that took centuries. Several factors contributed to this gradual collapse. The first and perhaps most important was"

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

INSTRUCT_PROMPT="Explain, in three or four sentences, why the sky appears blue during the day. Your answer should be accessible to a curious middle-school student."

printf '%-10s %6s %8s %11s %10s %10s %10s %7s\n' \
    "genre" "cycles" "full_acc" "mean_accept" "rej_rate" "tail_rate" "anypos" "tau"
echo "----------------------------------------------------------------------------------"

PARSE_PY=$(cat <<'PY'
import sys, re
label = sys.argv[1]
out = sys.stdin.read()
m = re.search(
    r"seed-oracle: cycles=(\d+) full_accept=(\d+) mean_accept_len=([\d.]+) \| "
    r"rej_match=([\d.]+) tail_match=([\d.]+) anypos_match=([\d.]+)", out)
tau_m = re.search(r"τ=([\d.]+)|\xcf\x84=([\d.]+)", out)
if not m:
    print(f"{label:<10} NO_ORACLE_LINE")
    sys.exit(0)
cyc, full, meana, rej, tail, anypos = m.groups()
tau = (tau_m.group(1) or tau_m.group(2)) if tau_m else "-"
print(f"{label:<10} {cyc:>6} {full:>8} {meana:>11} {rej:>10} {tail:>10} {anypos:>10} {tau:>7}")
PY
)

run_one() {
    local label="$1"
    local prompt="$2"
    local max="$3"
    local out
    out=$("$EXE" \
        --target "$TARGET_27B" --draft "$DRAFT_27B" \
        --prompt "$prompt" --max "$max" --ctx 2048 \
        --kv-mode q8 2>&1)
    printf '%s\n' "$out" | python3 -c "$PARSE_PY" "$label"
}

run_one "code"     "$CODE_PROMPT"     "$MAX_TOKENS"
run_one "prose"    "$PROSE_PROMPT"    "$MAX_TOKENS"
run_one "instruct" "$INSTRUCT_PROMPT" "$MAX_TOKENS"

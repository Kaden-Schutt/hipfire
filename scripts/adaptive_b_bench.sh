#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Adaptive-B scheduler bench — compare fixed-B=16, default adaptive (8..=16),
# widened adaptive (8..=20). 3 runs each on code + prose + instruct.
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_BENCH_MAX:-192}"
RUNS="${HIPFIRE_BENCH_RUNS:-3}"

if [ ! -x "$EXE" ]; then
    echo "build with: cargo build --release --example dflash_spec_demo --features deltanet" >&2
    exit 2
fi

if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "adaptive-b-bench" || { echo "could not acquire GPU lock" >&2; exit 2; }
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

PARSE_PY=$(cat <<'PY'
import sys, re, statistics
label = sys.argv[1]
genre = sys.argv[2]
runs = []
mean_bs, changes = [], []
for block in sys.stdin.read().split("\x1e"):
    m_toks = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", block)
    m_tau  = re.search(r"τ=([\d.]+)|\xcf\x84=([\d.]+)", block)
    m_ab   = re.search(r"adaptive-b: range=[\d]+\.\.=[\d]+ mean_B=([\d.]+) changes=(\d+)", block)
    if not m_toks or not m_tau:
        continue
    runs.append((float(m_toks.group(3)), float(m_tau.group(1) or m_tau.group(2))))
    if m_ab:
        mean_bs.append(float(m_ab.group(1)))
        changes.append(int(m_ab.group(2)))
if not runs:
    print(f"{label:<14} {genre:<8} PARSE_FAIL")
    sys.exit(0)
toks = sorted([r[0] for r in runs])
taus = sorted([r[1] for r in runs])
def med(a): return a[len(a)//2]
extra = ""
if mean_bs:
    extra = f" meanB={sum(mean_bs)/len(mean_bs):.2f} chg/run={sum(changes)/len(changes):.1f}"
print(f"{label:<14} {genre:<8} n={len(runs)} toks med={med(toks):6.1f} "
      f"[{min(toks):6.1f},{max(toks):6.1f}] tau med={med(taus):5.2f}{extra}")
PY
)

printf '%-14s %-8s %s\n' "config" "genre" "result"
echo "------------------------------------------------------------------------"

run_one() {
    local label="$1"
    local genre="$2"
    local prompt="$3"
    shift 3
    local extra=("$@")
    local blob=""
    for i in $(seq 1 "$RUNS"); do
        out=$("$EXE" \
            --target "$TARGET_27B" --draft "$DRAFT_27B" \
            --prompt "$prompt" --max "$MAX_TOKENS" --ctx 2048 \
            --kv-mode q8 --no-chatml "${extra[@]}" 2>&1)
        blob+="$out"$'\x1e'
    done
    printf '%s' "$blob" | python3 -c "$PARSE_PY" "$label" "$genre"
}

# Each config tested on all 3 genres.
for cfg in "fixed-B16|--no-adaptive-b" "adaptive-8-16|" "adaptive-8-20|--adaptive-b-range 8:20"; do
    label=${cfg%|*}
    args=${cfg#*|}
    # Split args on spaces but preserve empty.
    read -ra argv <<<"$args"
    run_one "$label" "code"     "$CODE_PROMPT"     "${argv[@]}"
    run_one "$label" "prose"    "$PROSE_PROMPT"    "${argv[@]}"
    run_one "$label" "instruct" "$INSTRUCT_PROMPT" "${argv[@]}"
done

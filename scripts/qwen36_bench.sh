#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Qwen3.6-27B full-genre bench with its new z-lab DFlash draft.
# 3 runs each × (code, prose, instruct) × (DFlash linear, DDTree b12-k2).
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET="$MODELS_DIR/qwen3.6-27b.mq4"
DRAFT="$MODELS_DIR/qwen36-27b-dflash-mq4-new.hfq"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_BENCH_MAX:-192}"
RUNS="${HIPFIRE_BENCH_RUNS:-3}"

if [ -r "$LOCK_SCRIPT" ]; then . "$LOCK_SCRIPT"
    gpu_acquire "qwen36-bench" || { echo "could not acquire GPU lock" >&2; exit 2; }
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
import sys, re
label = sys.argv[1]; genre = sys.argv[2]
runs=[]
for block in sys.stdin.read().split("\x1e"):
    m_toks = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", block)
    m_tau  = re.search(r"τ=([\d.]+)|\xcf\x84=([\d.]+)", block)
    if not m_toks or not m_tau: continue
    runs.append((float(m_toks.group(3)), float(m_tau.group(1) or m_tau.group(2))))
if not runs: print(f"{label:<14} {genre:<8} PARSE_FAIL"); sys.exit(0)
toks=sorted([r[0] for r in runs]); taus=sorted([r[1] for r in runs])
def med(a): return a[len(a)//2]
print(f"{label:<14} {genre:<8} n={len(runs)} toks med={med(toks):6.1f} [{min(toks):6.1f},{max(toks):6.1f}] tau med={med(taus):5.2f}")
PY
)

printf '%-14s %-8s %s\n' "config" "genre" "result"
echo "------------------------------------------------------------------------"

run_one() {
    local label="$1" genre="$2" prompt="$3"; shift 3; local extra=("$@")
    local blob=""
    for i in $(seq 1 "$RUNS"); do
        out=$("$EXE" \
            --target "$TARGET" --draft "$DRAFT" \
            --prompt "$prompt" --max "$MAX_TOKENS" --ctx 2048 \
            --kv-mode q8 --no-chatml "${extra[@]}" 2>&1)
        blob+="$out"$'\x1e'
    done
    printf '%s' "$blob" | python3 -c "$PARSE_PY" "$label" "$genre"
}

for cfg in "3.6-linear|" "3.6-ddtree-b12|--ddtree-batched --ddtree-budget 12 --ddtree-topk 2"; do
    label=${cfg%|*}; args=${cfg#*|}; read -ra argv <<<"$args"
    run_one "$label" "code"     "$CODE_PROMPT"     "${argv[@]}"
    run_one "$label" "prose"    "$PROSE_PROMPT"    "${argv[@]}"
    run_one "$label" "instruct" "$INSTRUCT_PROMPT" "${argv[@]}"
done

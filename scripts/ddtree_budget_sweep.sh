#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# DDTree budget/topk sweep — post-kernel-fuse era.
#
# Task #71 picked b12-k2 on 2026-04-14, before: #90 gemm_mw16 fix,
# #73 wo_residual, #74 gate_up, #75 qkvza, #81 wave_reduce. Per-cycle
# verify cost at higher budgets should be ~half what it was at that
# measurement. Sweep re-measures against current tree.
#
# Compares to Lucebox (blog/dflash27b, 2026-04): 129.5 tok/s mean on
# HumanEval @ b22, τ=8.31 (Qwen3.5-27B Q4_K_M target + bf16 draft on 3090).
#
# 3-run median per config per genre. Uses canonical code + prose prompts
# from the coherence gate (same as our baseline reporting).
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_SWEEP_MAX:-192}"
RUNS="${HIPFIRE_SWEEP_RUNS:-3}"

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
    gpu_acquire "ddtree-sweep" || { echo "could not acquire GPU lock" >&2; exit 2; }
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

# Format: "label|budget|topk"
CONFIGS=(
    "b12-k2|12|2"
    "b16-k2|16|2"
    "b12-k3|12|3"
    "b16-k3|16|3"
    "b22-k2|22|2"
    "b22-k4|22|4"
)

PARSE_PY=$(cat <<'PY'
import sys, re, statistics
label = sys.argv[1]
genre = sys.argv[2]
runs = []
for block in sys.stdin.read().split("\x1e"):
    m_toks = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", block)
    m_tau  = re.search(r"τ=([\d.]+)|\xcf\x84=([\d.]+)", block)
    m_acc  = re.search(r"accept_rate.*?:\s*([\d.]+)", block)
    if not m_toks or not m_tau:
        continue
    toks = float(m_toks.group(3))
    tau  = float(m_tau.group(1) or m_tau.group(2))
    acc  = float(m_acc.group(1)) if m_acc else float("nan")
    runs.append((toks, tau, acc))
if not runs:
    print(f"{label:<8} {genre:<6} PARSE_FAIL")
    sys.exit(0)
toks = sorted([r[0] for r in runs])
taus = sorted([r[1] for r in runs])
accs = sorted([r[2] for r in runs])
def med(a): return a[len(a)//2]
def mn(a):  return min(a)
def mx(a):  return max(a)
print(f"{label:<8} {genre:<6} n={len(runs)} "
      f"toks med={med(toks):6.1f} [{mn(toks):6.1f},{mx(toks):6.1f}] "
      f"tau med={med(taus):5.2f} [{mn(taus):5.2f},{mx(taus):5.2f}] "
      f"acc med={med(accs):4.2f}")
PY
)

printf '%-8s %-6s %s\n' "config" "genre" "result"
echo "-----------------------------------------------------------------------"

run_config() {
    local label="$1"
    local budget="$2"
    local topk="$3"
    local genre="$4"
    local prompt="$5"
    local max="$6"
    local blob=""
    for i in $(seq 1 "$RUNS"); do
        out=$("$EXE" \
            --target "$TARGET_27B" --draft "$DRAFT_27B" \
            --prompt "$prompt" --max "$max" --ctx 2048 \
            --kv-mode q8 --no-chatml \
            --ddtree-batched --ddtree-budget "$budget" --ddtree-topk "$topk" 2>&1)
        blob+="$out"$'\x1e'
    done
    printf '%s' "$blob" | python3 -c "$PARSE_PY" "$label" "$genre"
}

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label budget topk <<< "$entry"
    run_config "$label" "$budget" "$topk" "code"  "$CODE_PROMPT"  "$MAX_TOKENS"
    run_config "$label" "$budget" "$topk" "prose" "$PROSE_PROMPT" "$MAX_TOKENS"
done

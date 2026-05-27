#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# DDTree meta-verifier pruner sweep — tests HIPFIRE_DDTREE_LOGW_CUTOFF
# against the b12-k2 baseline on prose/instruct (where DDTree wins
# currently). Cutoff stops heap expansion when next candidate's cumulative
# log-prob falls below -cutoff. Budget cap of 12 still applies as an upper
# bound; effective tree size = min(budget, cutoff-truncation).
#
# Cutoffs tested: off (baseline), 6, 4, 3, 2.
#   cutoff=3 → drop candidates with absolute prob < e^-3 ≈ 5%
#   cutoff=4 → drop absolute prob < e^-4 ≈ 1.8%
#   cutoff=6 → drop absolute prob < e^-6 ≈ 0.25%  (minimal pruning)
set -u
cd "$(dirname "$0")/.."

EXE="./target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET_27B="$MODELS_DIR/qwen3.5-27b.mq4"
DRAFT_27B="$MODELS_DIR/qwen35-27b-dflash.mq4"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MAX_TOKENS="${HIPFIRE_META_MAX:-192}"
RUNS="${HIPFIRE_META_RUNS:-3}"

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
    gpu_acquire "ddtree-meta-sweep" || { echo "could not acquire GPU lock" >&2; exit 2; }
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
label = sys.argv[1]
genre = sys.argv[2]
runs = []
meta = []
for block in sys.stdin.read().split("\x1e"):
    m_toks = re.search(r"emitted: (\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", block)
    m_tau  = re.search(r"τ=([\d.]+)|\xcf\x84=([\d.]+)", block)
    m_meta = re.search(r"ddtree-meta: cycles=(\d+) mean_nodes=([\d.]+) min=(\d+) max=(\d+)", block)
    if not m_toks or not m_tau:
        continue
    runs.append((float(m_toks.group(3)), float(m_tau.group(1) or m_tau.group(2))))
    if m_meta:
        meta.append(float(m_meta.group(2)))
if not runs:
    print(f"{label:<10} {genre:<8} PARSE_FAIL")
    sys.exit(0)
toks = sorted([r[0] for r in runs])
taus = sorted([r[1] for r in runs])
def med(a): return a[len(a)//2]
meta_s = f"  meanNodes={sum(meta)/len(meta):5.2f}" if meta else ""
print(f"{label:<10} {genre:<8} n={len(runs)} toks med={med(toks):6.1f} "
      f"[{min(toks):6.1f},{max(toks):6.1f}] tau med={med(taus):5.2f}{meta_s}")
PY
)

printf '%-10s %-8s %s\n' "cutoff" "genre" "result"
echo "------------------------------------------------------------------------"

run_one() {
    local label="$1"
    local cutoff="$2"
    local genre="$3"
    local prompt="$4"
    local blob=""
    for i in $(seq 1 "$RUNS"); do
        if [ -z "$cutoff" ]; then
            env_prefix=()
        else
            env_prefix=("HIPFIRE_DDTREE_LOGW_CUTOFF=$cutoff")
        fi
        out=$(env "${env_prefix[@]}" "$EXE" \
            --target "$TARGET_27B" --draft "$DRAFT_27B" \
            --prompt "$prompt" --max "$MAX_TOKENS" --ctx 2048 \
            --kv-mode q8 --no-chatml \
            --ddtree-batched --ddtree-budget 12 --ddtree-topk 2 2>&1)
        blob+="$out"$'\x1e'
    done
    printf '%s' "$blob" | python3 -c "$PARSE_PY" "$label" "$genre"
}

for cfg in "off|" "cutoff-6|6" "cutoff-4|4" "cutoff-3|3" "cutoff-2|2"; do
    label=${cfg%|*}
    cutoff=${cfg#*|}
    run_one "$label" "$cutoff" "code"     "$CODE_PROMPT"
    run_one "$label" "$cutoff" "prose"    "$PROSE_PROMPT"
    run_one "$label" "$cutoff" "instruct" "$INSTRUCT_PROMPT"
done

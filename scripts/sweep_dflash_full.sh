#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# DFlash + DDTree sweep across 9B/27B-3.5/27B-3.6 post-EOT-fix (commit 6cff40e).
# Matrix: 3 models × 2 modes (DFlash linear, DDTree b12-k2) × 3 genres
#         (code/prose/instruct) × 3 runs each, median reported.
set -u
cd "$(dirname "$0")/.."

EXE=./target/release/examples/dflash_spec_demo
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
LOCK_SCRIPT=./scripts/gpu-lock.sh
MAX_TOKENS="${HIPFIRE_BENCH_MAX:-192}"
RUNS="${HIPFIRE_BENCH_RUNS:-3}"

if [ -r "$LOCK_SCRIPT" ]; then . "$LOCK_SCRIPT"
    gpu_acquire "dflash-sweep" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# LRU-cache prompt — keeps the model in pure code gen for 100+ tokens
# without hitting EOT (findings 2026-04-24 §3.4).
CODE_PROMPT='from typing import Optional


class ListNode:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev: Optional["ListNode"] = None
        self.next: Optional["ListNode"] = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = ListNode(0, 0)
        self.tail = ListNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: ListNode) -> None:
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: ListNode) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
'

PROSE_PROMPT="The Roman Empire, at its height, stretched from the windswept moors of northern Britain to the sands of the Arabian peninsula. Its decline was not a single event but a long slow unraveling that took centuries. Several factors contributed to this gradual collapse. The first and perhaps most important was"

INSTRUCT_PROMPT="Explain, in three or four sentences, why the sky appears blue during the day. Your answer should be accessible to a curious middle-school student."

PARSE_PY=$(cat <<'PY'
import sys, re
label, genre = sys.argv[1], sys.argv[2]
runs = []
for block in sys.stdin.read().split("\x1e"):
    m_toks = re.search(r"emitted:\s+(\d+) tokens in ([\d.]+)s\s+\(([\d.]+) tok/s\)", block)
    m_tau  = re.search(r"τ=([\d.]+)", block)  # τ=
    if not m_toks or not m_tau: continue
    runs.append((int(m_toks.group(1)), float(m_toks.group(3)), float(m_tau.group(1))))
if not runs:
    print(f"{label:<14} {genre:<8} PARSE_FAIL")
    sys.exit(0)
n_tok = sorted(r[0] for r in runs)
tok_s = sorted(r[1] for r in runs)
tau   = sorted(r[2] for r in runs)
med = lambda a: a[len(a)//2]
print(f"{label:<14} {genre:<8} n={len(runs)} "
      f"toks={med(n_tok):3d} tok/s med={med(tok_s):6.1f} [{min(tok_s):6.1f},{max(tok_s):6.1f}] "
      f"τ med={med(tau):5.2f}")
PY
)

# (label, target, draft, extra_args_per_run)
#   3.5 drafts trained on raw text → --no-chatml
#   3.6 draft trained with ChatML → default
MODELS=(
  "9b-3.5|$MODELS_DIR/qwen3.5-9b.mq4|models/qwen35-9b-dflash-mq4.hfq|--no-chatml"
  "27b-3.5|$MODELS_DIR/qwen3.5-27b.mq4|$MODELS_DIR/qwen35-27b-dflash.mq4|--no-chatml"
  "27b-3.6|$MODELS_DIR/qwen3.6-27b.mq4|$MODELS_DIR/qwen36-27b-dflash-mq4-new.hfq|"
)

MODES=(
  "dflash|"
  "ddtree-b12-k2|--ddtree-batched --ddtree-budget 12 --ddtree-topk 2"
)

GENRES=(
  "code|$CODE_PROMPT"
  "prose|$PROSE_PROMPT"
  "instruct|$INSTRUCT_PROMPT"
)

printf '%-14s %-8s %s\n' "config" "genre" "result"
printf -- '------------------------------------------------------------------------\n'

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r mlabel target draft chatml_flag <<<"$model_entry"
  [ -r "$target" ] || { echo "$mlabel skip: missing $target"; continue; }
  [ -r "$draft"  ] || { echo "$mlabel skip: missing $draft";  continue; }

  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode_label mode_args <<<"$mode_entry"
    label="${mlabel}-${mode_label}"

    for genre_entry in "${GENRES[@]}"; do
      IFS='|' read -r genre prompt <<<"$genre_entry"

      blob=""
      for i in $(seq 1 "$RUNS"); do
        # shellcheck disable=SC2086
        out=$("$EXE" \
          --target "$target" --draft "$draft" \
          --prompt "$prompt" --max "$MAX_TOKENS" --ctx 2048 \
          --kv-mode q8 --no-adaptive-b $chatml_flag $mode_args 2>&1)
        blob+="$out"$'\x1e'
      done
      printf '%s' "$blob" | python3 -c "$PARSE_PY" "$label" "$genre"
    done
  done
done

#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

# Compare each resident DFlash prompt row with the same row in a fresh
# standalone process. The complete deterministic eight-token output is
# compared after the explicit per-row target reset; equality catches row-to-row
# state leakage that a first-token-only check would miss.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXE="$ROOT/target/release/examples/dflash_spec_demo"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
TARGET="${HIPFIRE_DFLASH_TARGET:-$MODELS_DIR/qwen3.6-27b.mq4}"
DRAFT="${HIPFIRE_DFLASH_DRAFT:-$MODELS_DIR/qwen36-27b-dflash-mq4.hfq}"

if [[ ! -x "$EXE" || ! -f "$TARGET" || ! -f "$DRAFT" ]]; then
    echo "dflash-two-row-reset-gate: missing binary or fixtures" >&2
    exit 2
fi

evidence_root="${HIPFIRE_EVIDENCE_DIR:-${TMPDIR:-/tmp}/hipfire-cor-002}"
mkdir -p "$evidence_root"
tmp=$(mktemp -d "$evidence_root/dflash-XXXXXX")
manifest="$tmp/prompts.jsonl"
python3 - "$manifest" <<'PY'
import json
import sys

rows = [
    {"label": "row-a", "prompt": "Write one short sentence about apples.", "max": 8},
    {"label": "row-b", "prompt": "Write one short sentence about pears.", "max": 8},
]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
PY

run_one() {
    local prompt=$1 out=$2 err=$3
    "$EXE" --target "$TARGET" --draft "$DRAFT" --prompt "$prompt" \
        --max 8 --ctx 512 --no-chatml --no-adaptive-b >"$out" 2>"$err"
}

run_one "Write one short sentence about apples." "$tmp/one-a.out" "$tmp/one-a.err"
run_one "Write one short sentence about pears." "$tmp/one-b.out" "$tmp/one-b.err"
"$EXE" --target "$TARGET" --draft "$DRAFT" --prompts-file "$manifest" \
    --ctx 512 --no-chatml --no-adaptive-b >"$tmp/two.out" 2>"$tmp/two.err"

python3 - "$tmp/one-a.err" "$tmp/one-b.err" "$tmp/two.err" "$tmp" <<'PY'
import re
import sys

pattern = re.compile(r"\[dflash-reset-check\] row=(\d+) tokens=\[(.*?)\] target_pos=0")

def rows(path):
    result = []
    for row, tokens in pattern.findall(open(path, encoding="utf-8").read()):
        values = [int(token.strip()) for token in tokens.split(",") if token.strip()]
        if len(values) < 8:
            raise SystemExit(f"row {row} did not produce 8 generated tokens: {values}")
        result.append((int(row), values[:8]))
    return result

single_a, single_b, resident = map(rows, sys.argv[1:4])
if len(single_a) != 1 or single_a[0][0] != 0:
    raise SystemExit(f"row A reset evidence missing: {single_a}")
if len(single_b) != 1 or single_b[0][0] != 0:
    raise SystemExit(f"row B reset evidence missing: {single_b}")
if len(resident) != 2 or [row for row, _ in resident] != [0, 1]:
    raise SystemExit(f"resident reset evidence missing: {resident}")
expected = [single_a[0][1], single_b[0][1]]
actual = [tokens for _, tokens in resident]
if actual != expected:
    raise SystemExit(f"resident rows diverged from fresh rows: resident={actual} fresh={expected}")
print(f"DFlash two-row reset parity: PASS (full outputs {actual})")
print(f"evidence={sys.argv[4]}")
PY

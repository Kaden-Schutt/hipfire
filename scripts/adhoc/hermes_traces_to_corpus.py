#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
hermes_traces_to_corpus.py — convert the lambda/hermes-agent-reasoning-traces
Parquet files to a plain-text corpus that our sidecar calibration and
draft training scripts accept.

Each row is a multi-turn ShareGPT conversation with real tool-calling
trajectories (<think>/<tool_call>/<tool_response> blocks). We linearize
each conversation into one "doc" separated by blank lines.

For DFlash draft training specifically, this corpus is gold: the draft
learns to predict the actual post-tool-call reasoning + subsequent tool
calls that Carnice-9b / Hermes targets produce.

Usage:
    python3 scripts/hermes_traces_to_corpus.py \\
        <traces_dir_with_parquet> \\
        <out_corpus.txt> \\
        [--max-rows N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: install pyarrow first (comes with `pip install datasets`)", file=sys.stderr)
    sys.exit(2)


# Mapping from ShareGPT `from` field to a prefix header in the linearized text.
ROLE_PREFIX = {
    "system": "SYSTEM",
    "human": "USER",
    "gpt": "ASSISTANT",
    "tool": "TOOL",
}


def linearize(convo: list[dict]) -> str:
    """Turn a ShareGPT conversation list into a single text doc.

    The assistant turns already contain <think>/<tool_call> inline; we
    preserve them verbatim so the draft can learn to predict the full
    thinking + tool-call sequence.
    """
    parts = []
    for turn in convo:
        role = turn.get("from", "unknown")
        value = (turn.get("value") or "").strip()
        if not value:
            continue
        header = ROLE_PREFIX.get(role, role.upper())
        parts.append(f"{header}: {value}")
    return "\n\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Dir containing parquet files (recurses)")
    p.add_argument("out", help="Output plain-text corpus")
    p.add_argument("--max-rows", type=int, default=0, help="Cap on rows (0 = all)")
    args = p.parse_args()

    src = Path(args.src)
    if src.is_dir():
        files = sorted(src.rglob("*.parquet"))
    elif src.is_file():
        files = [src]
    else:
        print(f"source not found: {src}", file=sys.stderr)
        return 2

    if not files:
        print(f"no parquet files under {src}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out, total_chars = 0, 0, 0
    with out.open("w") as w:
        for f in files:
            tbl = pq.read_table(f)
            # Extract only the columns we need
            convos = tbl.column("conversations").to_pylist()
            for convo in convos:
                n_in += 1
                if args.max_rows and n_out >= args.max_rows:
                    break
                try:
                    doc = linearize(convo)
                except Exception as e:
                    print(f"skip row {n_in}: {e}", file=sys.stderr)
                    continue
                if not doc:
                    continue
                w.write(doc)
                w.write("\n\n")
                n_out += 1
                total_chars += len(doc) + 2
            if args.max_rows and n_out >= args.max_rows:
                break

    print(f"[hermes] read {n_in:,} rows, wrote {n_out:,} docs ({total_chars / 1e6:.1f} MB)")
    # Rough token estimate: 4 chars/token for English + code
    print(f"[hermes] approx tokens: {total_chars // 4:,}")
    print(f"[hermes] output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

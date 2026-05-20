#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""
aureth_to_corpus.py — convert the OusiaResearch Aureth-Corpus JSONL to a
plain-text corpus that our sidecar calibration and draft training scripts
accept.

Each JSONL row is a DPO pair with `prompt` and `chosen` fields. We emit
one concatenated prompt+chosen "doc" per row, separated by blank lines
so the corpus chunker in triattn_validate splits them correctly.

Filters out `rejected` and the DPO negatives — for both sidecar cal and
draft training we only want the high-quality responses (the draft is
meant to mimic the target's preferred behavior, not its failure modes).

Usage:
    python3 scripts/aureth_to_corpus.py \\
        <aureth_dir_with_jsonl_files> \\
        <out_corpus.txt> \\
        [--max-rows N] [--min-quality 0.80]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Directory containing Aureth *.jsonl files (or single .jsonl)")
    p.add_argument("out", help="Output plain-text corpus")
    p.add_argument("--max-rows", type=int, default=0, help="Cap on rows (0 = all)")
    p.add_argument("--min-quality", type=float, default=0.0, help="Drop rows below this quality_score")
    p.add_argument("--format", default="prompt_chosen", choices=["prompt_chosen", "chosen_only"],
                   help="prompt_chosen: 'USER: <prompt>\\nASSISTANT: <chosen>'; chosen_only: just the chosen text")
    args = p.parse_args()

    src = Path(args.src)
    if src.is_dir():
        files = sorted(src.rglob("*.jsonl"))
    elif src.is_file():
        files = [src]
    else:
        print(f"source not found: {src}", file=sys.stderr)
        return 2

    if not files:
        print(f"no .jsonl files under {src}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in, n_out, n_dropped_quality = 0, 0, 0
    total_chars = 0
    with out.open("w") as w:
        for f in files:
            with f.open() as r:
                for line in r:
                    n_in += 1
                    if args.max_rows and n_out >= args.max_rows:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = float(row.get("quality_score", 1.0) or 0.0)
                    if q < args.min_quality:
                        n_dropped_quality += 1
                        continue
                    prompt = (row.get("prompt") or "").strip()
                    chosen = (row.get("chosen") or "").strip()
                    if not chosen:
                        continue
                    if args.format == "prompt_chosen":
                        doc = f"USER: {prompt}\n\nASSISTANT: {chosen}"
                    else:
                        doc = chosen
                    w.write(doc)
                    w.write("\n\n")  # blank line separates docs
                    n_out += 1
                    total_chars += len(doc)
            if args.max_rows and n_out >= args.max_rows:
                break

    print(f"[aureth] read {n_in:,} rows, wrote {n_out:,} docs ({total_chars / 1e6:.1f} MB)", flush=True)
    if n_dropped_quality:
        print(f"[aureth] dropped {n_dropped_quality:,} rows below quality {args.min_quality}")
    print(f"[aureth] output: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

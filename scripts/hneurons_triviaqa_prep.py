#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit `{question, aliases}` JSONL from TriviaQA rc.nocontext parquet.

Tooling for the on-paper H-Neurons protocol: the resident model generates its
own answer per question, which is labeled correct/hallucinated by matching the
generated text against the TriviaQA `normalized_aliases`. This step just extracts
questions + gold aliases; generation + labeling + CETT capture happen in
`hipfire-hneurons-probe --self-generate`.

Each output line: {"qid", "question", "aliases": [normalized_alias, ...]}.
The question gets the paper's short-answer instruction appended.
"""
import argparse
import json
import os

import pyarrow.parquet as pq

INSTRUCTION = " Respond with the answer only, without any explanation."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triviaqa-dir", required=True, help="rc.nocontext dir")
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = os.path.join(args.triviaqa_dir, f"{args.split}-00000-of-00001.parquet")
    cols = pq.read_table(path, columns=["question", "question_id", "answer"])
    n = min(args.limit, cols.num_rows)
    rows = cols.slice(0, n).to_pylist()

    written = 0
    with open(args.out, "w") as f:
        for r in rows:
            ans = r.get("answer") or {}
            aliases = ans.get("normalized_aliases") or []
            if not aliases:
                nv = ans.get("normalized_value")
                aliases = [nv] if nv else []
            aliases = [a for a in aliases if a and a.strip()]
            if not aliases:
                continue
            q = (r.get("question") or "").strip()
            if not q:
                continue
            f.write(
                json.dumps(
                    {
                        "qid": r.get("question_id"),
                        "question": q + INSTRUCTION,
                        "aliases": aliases,
                    }
                )
                + "\n"
            )
            written += 1

    print(f"wrote {args.out}: {written} questions from {args.split} (scanned {n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

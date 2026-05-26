#!/usr/bin/env python3
"""Count total tokens in a dump_corpus_openai*.py JSONL output.

Uses the Qwen3.5 tokenizer (transformers) to tokenize the 'output' field
of each JSONL record and sum the token counts.

Usage:
  .venv/bin/python3 scripts/count_corpus_tokens.py corpus/qwen36_dump.jsonl
  .venv/bin/python3 scripts/count_corpus_tokens.py corpus/*.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", help="JSONL dump file(s) to count")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B",
                   help="tokenizer model (default: Qwen/Qwen3.5-27B)")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    total_records = 0
    total_tokens = 0
    total_reported = 0  # sum of completion_tokens from API
    per_file = {}

    for fpath in args.files:
        fp = Path(fpath)
        if not fp.exists():
            print(f"  (skip) {fpath}: not found", file=sys.stderr)
            continue

        rec_count = 0
        tok_count = 0
        reported_count = 0

        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_records += 1
            rec_count += 1

            output = rec.get("output")
            if isinstance(output, str) and output.strip():
                tokens = len(tokenizer.encode(output, add_special_tokens=False))
                total_tokens += tokens
                tok_count += tokens

            ct = rec.get("completion_tokens")
            if ct is not None:
                total_reported += ct
                reported_count += ct

        per_file[str(fp)] = {
            "records": rec_count,
            "tokenized_tokens": tok_count,
            "reported_tokens": reported_count,
        }

    print(f"{'File':<50} {'Records':>7} {'Tokenized':>10} {'Reported':>10}")
    print(f"{'-'*50}")
    for fp, stats in per_file.items():
        name = fp.split("/")[-1] if "/" in fp else fp
        print(f"{name:<50} {stats['records']:>7} {stats['tokenized_tokens']:>10,} {stats['reported_tokens']:>10,}")

    print(f"{'='*50}")
    print(f"{'TOTAL':<50} {total_records:>7} {total_tokens:>10,} {total_reported:>10,}")
    if total_reported > 0:
        ratio = total_tokens / total_reported * 100
        print(f"Tokenized vs reported: {ratio:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

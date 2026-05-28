#!/usr/bin/env python3
"""Merge two FastMTP-style vocab sidecars into a combined top-K sidecar.

Use case: v1 input-corpus sidecar covers code/structured tokens well (because
the corpus is canonical-bench-aligned), v2 trunk-argmax distill sidecar
covers broader distribution. Combining gives a sidecar that's strong on
the deployment workload AND on out-of-distribution prompts.

Combination is rank-weighted: for each token id, score = sum(1 / rank_i)
across input sidecars (where rank is 1-indexed). Tokens absent from a
sidecar get a 0 contribution from that source. Final ordering is by
combined score descending.

Usage:
  python3 scripts/distill/merge_sidecars.py \
    --in v1_sidecar.json v2_sidecar.json \
    --out merged_sidecar.json \
    --top-k 32768
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True,
                   help="Two or more sidecar JSONs to merge")
    p.add_argument("--out", required=True, help="Output sidecar JSON path")
    p.add_argument("--top-k", type=int, default=32768)
    p.add_argument("--full-vocab-size", type=int, default=248320)
    args = p.parse_args()

    sources: list[dict] = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text())
        sources.append(data)
        n = len(data.get("draft_to_full", []))
        print(f"  source: {path} ({n} tokens, "
              f"unique_observed={data.get('stats', {}).get('unique_tokens', 'n/a')})",
              file=sys.stderr)

    score: dict[int, float] = {}
    for src in sources:
        for rank, tid in enumerate(src["draft_to_full"]):
            score[tid] = score.get(tid, 0.0) + 1.0 / (rank + 1)

    must_include: list[int] = []
    for src in sources:
        ms = src.get("stats", {}).get("must_include_specials", [])
        must_include.extend(ms)
    must_include = sorted(set(must_include))
    for tid in must_include:
        score[tid] = max(score.get(tid, 0.0), 1e6)

    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = [tid for tid, _ in ranked[: args.top_k]]
    selected_set = set(selected)

    if len(selected) < args.top_k:
        for t in range(args.full_vocab_size):
            if t not in selected_set:
                selected.append(t)
                selected_set.add(t)
                if len(selected) == args.top_k:
                    break

    out = {
        "draft_to_full": selected,
        "compressed_vocab_size": args.top_k,
        "full_vocab_size": args.full_vocab_size,
        "stats": {
            "source": "merged sidecar (rank-weighted union of inputs)",
            "input_sidecars": [str(p) for p in args.inputs],
            "must_include_specials": must_include,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

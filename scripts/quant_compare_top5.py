#!/usr/bin/env python3
"""Compare greedy_dump_top5 CSV/token outputs across quantization modes."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


WIDE_MARGIN_DEFAULT = 1.0


def artifact_path(prefix: Path, suffix: str) -> Path:
    """Return <prefix><suffix>, even when prefix itself contains dots."""
    return Path(f"{prefix}{suffix}")


def load_tokens(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().splitlines() if x.strip()]


def load_top5(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: quant_compare_top5.py <baseline-prefix> <mode=prefix>... [--wide-margin N]",
            file=sys.stderr,
        )
        return 2

    baseline_prefix = Path(sys.argv[1])
    args = sys.argv[2:]
    wide_margin = WIDE_MARGIN_DEFAULT
    if "--wide-margin" in args:
        i = args.index("--wide-margin")
        try:
            wide_margin = float(args[i + 1])
        except (IndexError, ValueError):
            print("--wide-margin requires a numeric value", file=sys.stderr)
            return 2
        del args[i : i + 2]

    base_tokens = load_tokens(artifact_path(baseline_prefix, ".tokens"))
    base_top5 = load_top5(artifact_path(baseline_prefix, ".top5.csv"))

    rows = []
    for spec in args:
        if "=" not in spec:
            print(f"bad mode spec: {spec}", file=sys.stderr)
            return 2
        mode, prefix_s = spec.split("=", 1)
        prefix = Path(prefix_s)
        toks = load_tokens(artifact_path(prefix, ".tokens"))
        top5 = load_top5(artifact_path(prefix, ".top5.csv"))

        n = min(len(base_tokens), len(toks), len(base_top5), len(top5))
        first_div = None
        top1_match = 0
        top5_overlap_sum = 0
        max_top1_delta = 0.0
        margin_sum = 0.0
        mode_margin_sum = 0.0
        margin_at_div = None
        wide_argmax_flips = 0
        for i in range(n):
            base_margin = float(base_top5[i]["margin_top12"])
            mode_margin = float(top5[i]["margin_top12"])
            margin_sum += base_margin
            mode_margin_sum += mode_margin
            if base_tokens[i] == toks[i]:
                top1_match += 1
            else:
                if first_div is None:
                    first_div = i
                    margin_at_div = base_margin
                if base_margin >= wide_margin:
                    wide_argmax_flips += 1

            base_ids = {int(base_top5[i][f"r{r}_id"]) for r in range(1, 6)}
            ids = {int(top5[i][f"r{r}_id"]) for r in range(1, 6)}
            top5_overlap_sum += len(base_ids & ids)
            max_top1_delta = max(
                max_top1_delta,
                abs(float(base_top5[i]["r1_logit"]) - float(top5[i]["r1_logit"])),
            )

        row = {
            "mode": mode,
            "steps_compared": n,
            "top1_agreement": top1_match / n if n else math.nan,
            "mean_top5_overlap": top5_overlap_sum / n if n else math.nan,
            "mean_baseline_margin_top12": margin_sum / n if n else math.nan,
            "mean_mode_margin_top12": mode_margin_sum / n if n else math.nan,
            "first_token_agreement": (base_tokens[0] == toks[0]) if n else None,
            "first_divergence": first_div,
            "baseline_margin_at_divergence": margin_at_div,
            "wide_margin_threshold": wide_margin,
            "wide_argmax_flips": wide_argmax_flips,
            "max_top1_logit_delta": max_top1_delta,
            "baseline_tokens": len(base_tokens),
            "mode_tokens": len(toks),
        }
        rows.append(row)

    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

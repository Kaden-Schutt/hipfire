#!/usr/bin/env python3
"""Explain the first greedy-token divergence between two top-5 dumps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_tokens(prefix: Path) -> list[int]:
    return [int(x) for x in prefix.with_suffix(".tokens").read_text().splitlines() if x.strip()]


def load_top5(prefix: Path) -> list[dict[str, str]]:
    with prefix.with_suffix(".top5.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def top5(row: dict[str, str]) -> str:
    parts = []
    for rank in range(1, 6):
        parts.append(f"{row[f'r{rank}_id']}:{float(row[f'r{rank}_logit']):.4f}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_prefix", type=Path)
    parser.add_argument("mode_prefix", type=Path)
    parser.add_argument("--window", type=int, default=6)
    args = parser.parse_args()

    base_tokens = load_tokens(args.baseline_prefix)
    mode_tokens = load_tokens(args.mode_prefix)
    base_top5 = load_top5(args.baseline_prefix)
    mode_top5 = load_top5(args.mode_prefix)
    n = min(len(base_tokens), len(mode_tokens), len(base_top5), len(mode_top5))

    first = next((i for i in range(n) if base_tokens[i] != mode_tokens[i]), None)
    if first is None:
        print(f"no divergence across {n} compared steps")
        return 0

    start = max(0, first - args.window)
    end = min(n, first + args.window + 1)
    print(f"first_divergence={first}")
    print(f"steps_compared={n}")
    print(
        f"baseline_token={base_tokens[first]} mode_token={mode_tokens[first]} "
        f"baseline_margin_top12={float(base_top5[first]['margin_top12']):.4f}"
    )
    print()
    print("step,match,baseline_token,mode_token,baseline_margin,baseline_top5,mode_top5")
    for i in range(start, end):
        print(
            f"{i},{base_tokens[i] == mode_tokens[i]},{base_tokens[i]},{mode_tokens[i]},"
            f"{float(base_top5[i]['margin_top12']):.4f},{top5(base_top5[i])},{top5(mode_top5[i])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

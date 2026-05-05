#!/usr/bin/env python3
"""Analyze a MoE expert hit-count heatmap dumped by daemon (Phase 0 MoE eGPU offload).

Reads the CSV produced when HIPFIRE_MOE_EXPERT_HEATMAP=1 is set, computes per-layer
hit-rate at LRU cache sizes 8/16/32/64/128, and reports the overall decision gate
(>=80% at 32 → proceed, <=70% → scrap) per docs/plans/moe-egpu-offload.prd.

Usage: scripts/analyze_moe_heatmap.py <heatmap-*.csv> [<more.csv> ...]
"""
import csv
import math
import sys
from pathlib import Path


def parse(path: Path):
    n_layers = n_experts = tokens = decisions = 0
    model = ""
    counts: dict[tuple[int, int], int] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line.lstrip("#").strip().split():
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    if k == "n_layers":
                        n_layers = int(v)
                    elif k == "n_experts":
                        n_experts = int(v)
                    elif k == "tokens_seen":
                        tokens = int(v)
                    elif k == "routed_decisions":
                        decisions = int(v)
                    elif k == "model":
                        model = v
                continue
            if line.startswith("layer,"):
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            layer, expert, c = int(parts[0]), int(parts[1]), int(parts[2])
            counts[(layer, expert)] = c
    return n_layers, n_experts, tokens, decisions, model, counts


def hit_rate_at_size(layer_counts: list[int], cache_size: int) -> float:
    """Fraction of decisions covered by the top-`cache_size` experts in this layer."""
    total = sum(layer_counts)
    if total == 0:
        return 0.0
    sorted_desc = sorted(layer_counts, reverse=True)
    covered = sum(sorted_desc[:cache_size])
    return covered / total


def entropy(layer_counts: list[int]) -> float:
    total = sum(layer_counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in layer_counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def analyze(path: Path):
    n_layers, n_experts, tokens, decisions, model, counts = parse(path)
    if n_layers == 0 or n_experts == 0:
        print(f"  {path.name}: empty heatmap", file=sys.stderr)
        return

    print(f"=== {path.name} ===")
    print(f"  model: {model}")
    print(f"  n_layers={n_layers} n_experts={n_experts}")
    print(f"  tokens_seen={tokens}  routed_decisions={decisions}")
    if tokens == 0:
        print("  (no tokens recorded)\n")
        return

    cache_sizes = [8, 16, 32, 64, 128]
    if n_experts < cache_sizes[-1]:
        cache_sizes = [s for s in cache_sizes if s <= n_experts]

    layer_rates = {cs: [] for cs in cache_sizes}
    layer_entropies = []
    for layer in range(n_layers):
        layer_counts = [counts.get((layer, e), 0) for e in range(n_experts)]
        layer_entropies.append(entropy(layer_counts))
        for cs in cache_sizes:
            layer_rates[cs].append(hit_rate_at_size(layer_counts, cs))

    print(f"  per-layer entropy:  min={min(layer_entropies):.2f}  "
          f"mean={sum(layer_entropies)/len(layer_entropies):.2f}  "
          f"max={max(layer_entropies):.2f}  "
          f"max_possible={math.log2(n_experts):.2f}")
    print()
    header = "  cache_size " + "  ".join(f"  L{layer:>2}" for layer in range(min(n_layers, 8)))
    if n_layers > 8:
        header += "  ...  mean   min"
    else:
        header += "  mean   min"
    print(header)
    for cs in cache_sizes:
        rates = layer_rates[cs]
        sample = rates[: min(n_layers, 8)]
        rest = f"  {sum(rates)/len(rates)*100:5.1f}  {min(rates)*100:5.1f}"
        if n_layers > 8:
            rest = "  ..." + rest
        cells = "  ".join(f"{r*100:5.1f}" for r in sample)
        print(f"  {cs:>5}      {cells}{rest}")
    print()

    decision_size = 32 if 32 in cache_sizes else cache_sizes[-1]
    rates_at_decision = layer_rates[decision_size]
    mean_rate = sum(rates_at_decision) / len(rates_at_decision)
    min_rate = min(rates_at_decision)
    pct = mean_rate * 100
    if mean_rate >= 0.80:
        verdict = "PROCEED"
    elif mean_rate <= 0.70:
        verdict = "SCRAP — try per-token pull or skip cache"
    else:
        verdict = "MARGINAL — close call"
    print(f"  Phase 0 gate at cache={decision_size}: mean={pct:.1f}%, min(layer)={min_rate*100:.1f}%")
    print(f"  decision: {verdict}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    for arg in sys.argv[1:]:
        analyze(Path(arg))


if __name__ == "__main__":
    main()

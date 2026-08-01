#!/usr/bin/env python3
"""Paired ABBA bootstrap stopping-rule helper for DS4 gfx942/MI300X gate-A levers.

Consumes ``hipfire bench --json`` result files named::

    pair-<i>-<arm>-<UTC timestamp>.json

where ``<arm>`` is ``A`` (baseline) or ``B`` (candidate). Each complete pair has
exactly four files in ABBA order (two A, two B). Incomplete pairs are excluded
from the statistics and reported by name.

Pair relative delta (percent)::

    100 * (median(B_values) / median(A_values) - 1)

For metrics whose names end in ``_ms`` (e.g. ``ttft_ms``), lower is better.
Those deltas are negated so that positive always means "B is better than A".
The output JSON records this via ``"lower_is_better": true``.

Bootstrap: resample complete pair deltas with replacement ``--resamples`` times
using ``random.Random(seed)`` (deterministic). Each resample mean contributes to
the empirical distribution; CI is the 2.5th and 97.5th percentiles of those means.

Decision order (exactly)::

    n_pairs < min_pairs      -> continue   (insufficient pairs)
    ci_low  >= threshold     -> promote
    ci_high <  threshold     -> kill
    n_pairs >= max_pairs     -> cap
    otherwise                -> continue

Default threshold is +0.5 (campaign promotion checkpoint). Exit 0 on successful
analysis (decision is data); nonzero only on usage errors or zero valid files.

stdlib only: json, glob, argparse, statistics, random, math, pathlib, re, sys.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# pair-<i>-<arm>-<rest>.json  — arm is a single A or B token between hyphens.
PAIR_NAME_RE = re.compile(
    r"^pair-(?P<idx>\d+)-(?P<arm>[ABab])-(?P<rest>.+)\.json$",
    re.ASCII,
)


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def extract_metric(payload: Any, metric: str) -> Tuple[Optional[float], Optional[str]]:
    """Pull a scalar metric from a bench JSON object.

    Accepts a top-level scalar or a list of per-run samples (median). Nested
    dict lookup is not required by the contract; only the named top-level key.
    """
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    if metric not in payload:
        return None, f"missing metric key {metric!r}"
    raw = payload[metric]
    if isinstance(raw, list):
        if not raw:
            return None, f"metric {metric!r} is an empty list"
        values: List[float] = []
        for i, item in enumerate(raw):
            if not _is_finite_number(item):
                return None, f"metric {metric!r} list item[{i}] is non-finite or non-numeric"
            values.append(float(item))
        return float(statistics.median(values)), None
    if _is_finite_number(raw):
        return float(raw), None
    return None, f"metric {metric!r} is neither a finite scalar nor a list of finite numbers"


def parse_filename(path: Path) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Return (pair_index, arm_upper, error_reason)."""
    m = PAIR_NAME_RE.match(path.name)
    if not m:
        return None, None, f"filename does not match pair-<i>-<A|B>-<timestamp>.json: {path.name}"
    return int(m.group("idx")), m.group("arm").upper(), None


def load_file(path: Path, metric: str) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[str]]:
    """Load one bench JSON. Returns (pair_idx, arm, value, error)."""
    pair_idx, arm, name_err = parse_filename(path)
    if name_err is not None:
        return None, None, None, name_err
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return pair_idx, arm, None, f"unreadable: {exc}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return pair_idx, arm, None, f"unparseable JSON: {exc}"
    value, metric_err = extract_metric(payload, metric)
    if metric_err is not None:
        return pair_idx, arm, None, metric_err
    return pair_idx, arm, value, None


def pair_delta_percent(a_values: Sequence[float], b_values: Sequence[float], lower_is_better: bool) -> Tuple[Optional[float], Optional[str], float, float]:
    """Compute signed pair delta. Returns (delta, error, a_median, b_median)."""
    a_med = float(statistics.median(a_values))
    b_med = float(statistics.median(b_values))
    if a_med == 0.0:
        return None, "divide-by-zero: median(A_values) is 0", a_med, b_med
    if not math.isfinite(a_med) or not math.isfinite(b_med):
        return None, "non-finite median", a_med, b_med
    delta = 100.0 * (b_med / a_med - 1.0)
    if lower_is_better:
        delta = -delta
    if not math.isfinite(delta):
        return None, "non-finite delta", a_med, b_med
    return delta, None, a_med, b_med


def percentile_nearest_rank(sorted_vals: Sequence[float], p: float) -> float:
    """Inclusive nearest-rank percentile on a pre-sorted non-empty sequence.

    rank = ceil(p/100 * n), clamped to [1, n], then 0-indexed.
    """
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty sample for percentile")
    if n == 1:
        return float(sorted_vals[0])
    # Nearest-rank: smallest k such that k/n >= p/100.
    rank = int(math.ceil((p / 100.0) * n))
    if rank < 1:
        rank = 1
    if rank > n:
        rank = n
    return float(sorted_vals[rank - 1])


def bootstrap_ci(
    deltas: Sequence[float],
    resamples: int,
    seed: int,
) -> Tuple[float, float, List[float]]:
    """Resample pair deltas with replacement; return (ci_low, ci_high, means)."""
    rng = random.Random(seed)
    n = len(deltas)
    means: List[float] = []
    if n == 0:
        return float("nan"), float("nan"), means
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(float(statistics.fmean(sample)))
    ordered = sorted(means)
    lo = percentile_nearest_rank(ordered, 2.5)
    hi = percentile_nearest_rank(ordered, 97.5)
    return lo, hi, means


def decide(
    n_pairs: int,
    ci_low: float,
    ci_high: float,
    threshold: float,
    min_pairs: int,
    max_pairs: int,
) -> Tuple[str, str]:
    """Stopping rule evaluated in the contract's exact order."""
    if n_pairs < min_pairs:
        return "continue", f"insufficient pairs: n_complete_pairs={n_pairs} < min_pairs={min_pairs}"
    if math.isfinite(ci_low) and ci_low >= threshold:
        return "promote", f"ci_low={ci_low:.6g} >= threshold={threshold}"
    if math.isfinite(ci_high) and ci_high < threshold:
        return "kill", f"ci_high={ci_high:.6g} < threshold={threshold}"
    if n_pairs >= max_pairs:
        return "cap", f"n_complete_pairs={n_pairs} >= max_pairs={max_pairs} and CI straddles threshold"
    return (
        "continue",
        f"CI straddles threshold with n_complete_pairs={n_pairs} < max_pairs={max_pairs}",
    )


def analyze(
    paths: Sequence[Path],
    metric: str,
    threshold: float,
    min_pairs: int,
    max_pairs: int,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    lower_is_better = metric.endswith("_ms")

    invalid_files: List[Dict[str, str]] = []
    # pair_idx -> {"A": [values...], "B": [values...], "A_paths": [...], "B_paths": [...],
    #              "invalid_paths": [str, ...]}
    buckets: Dict[int, Dict[str, Any]] = {}
    input_files = sorted(str(p) for p in paths)

    for path in sorted(paths, key=lambda p: str(p)):
        pair_idx, arm, value, err = load_file(path, metric)
        if err is not None:
            invalid_files.append({"path": str(path), "reason": err})
            # Named pair files still reserve a bucket so the pair is reported incomplete.
            if pair_idx is not None:
                bucket = buckets.setdefault(
                    pair_idx,
                    {"A": [], "B": [], "A_paths": [], "B_paths": [], "invalid_paths": []},
                )
                bucket["invalid_paths"].append(str(path))
            continue
        assert pair_idx is not None and arm is not None and value is not None
        bucket = buckets.setdefault(
            pair_idx,
            {"A": [], "B": [], "A_paths": [], "B_paths": [], "invalid_paths": []},
        )
        bucket[arm].append(value)
        bucket[f"{arm}_paths"].append(str(path))

    pairs_out: List[Dict[str, Any]] = []
    incomplete_pairs: List[Dict[str, Any]] = []
    deltas: List[float] = []

    for pair_idx in sorted(buckets):
        bucket = buckets[pair_idx]
        a_vals: List[float] = list(bucket["A"])
        b_vals: List[float] = list(bucket["B"])
        n_a, n_b = len(a_vals), len(b_vals)
        if n_a != 2 or n_b != 2:
            incomplete_pairs.append(
                {
                    "pair": pair_idx,
                    "n_a": n_a,
                    "n_b": n_b,
                    "a_paths": list(bucket["A_paths"]),
                    "b_paths": list(bucket["B_paths"]),
                    "invalid_paths": list(bucket["invalid_paths"]),
                    "reason": f"incomplete: need exactly 2 A and 2 B, got n_a={n_a} n_b={n_b}",
                }
            )
            continue
        delta, derr, a_med, b_med = pair_delta_percent(a_vals, b_vals, lower_is_better)
        if derr is not None:
            # Treat as invalid pair members rather than a complete pair.
            for pth in bucket["A_paths"] + bucket["B_paths"]:
                invalid_files.append({"path": pth, "reason": f"pair {pair_idx}: {derr}"})
            incomplete_pairs.append(
                {
                    "pair": pair_idx,
                    "n_a": n_a,
                    "n_b": n_b,
                    "a_paths": list(bucket["A_paths"]),
                    "b_paths": list(bucket["B_paths"]),
                    "invalid_paths": list(bucket["invalid_paths"]),
                    "reason": derr,
                }
            )
            continue
        assert delta is not None
        deltas.append(delta)
        pairs_out.append(
            {
                "pair": pair_idx,
                "a_values": a_vals,
                "b_values": b_vals,
                "a_median": a_med,
                "b_median": b_med,
                "delta_percent": delta,
            }
        )

    n_complete = len(pairs_out)
    if n_complete == 0:
        overall_median = None
        dispersion = {"min": None, "max": None, "stdev": None}
        ci_low: Any = None
        ci_high: Any = None
        decision = "continue"
        decision_reason = "no complete pairs"
    else:
        overall_median = float(statistics.median(deltas))
        if n_complete >= 2:
            stdev = float(statistics.stdev(deltas))
        else:
            stdev = 0.0
        dispersion = {
            "min": float(min(deltas)),
            "max": float(max(deltas)),
            "stdev": stdev,
        }
        ci_low, ci_high, _means = bootstrap_ci(deltas, resamples, seed)
        decision, decision_reason = decide(
            n_complete, ci_low, ci_high, threshold, min_pairs, max_pairs
        )

    # Stable sort invalid_files by path for determinism.
    invalid_files_sorted = sorted(invalid_files, key=lambda x: x["path"])

    return {
        "metric": metric,
        "lower_is_better": lower_is_better,
        "threshold": threshold,
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "resamples": resamples,
        "seed": seed,
        "n_complete_pairs": n_complete,
        "pairs": pairs_out,
        "overall_median_delta_percent": overall_median,
        "dispersion": dispersion,
        "ci95_percent": [ci_low, ci_high] if n_complete else [None, None],
        "decision": decision,
        "decision_reason": decision_reason,
        "invalid_files": invalid_files_sorted,
        "incomplete_pairs": incomplete_pairs,
        "input_files": input_files,
    }


def format_summary(result: Dict[str, Any]) -> str:
    ci = result["ci95_percent"]
    ci_s = (
        f"[{ci[0]:.6g}, {ci[1]:.6g}]"
        if ci[0] is not None and ci[1] is not None
        else "[null, null]"
    )
    med = result["overall_median_delta_percent"]
    med_s = f"{med:.6g}" if med is not None else "null"
    lines = [
        f"metric={result['metric']} lower_is_better={result['lower_is_better']}",
        f"n_complete_pairs={result['n_complete_pairs']} "
        f"incomplete={len(result['incomplete_pairs'])} "
        f"invalid={len(result['invalid_files'])}",
        f"overall_median_delta_percent={med_s}",
        f"ci95_percent={ci_s}",
        f"decision={result['decision']}  ({result['decision_reason']})",
        f"threshold={result['threshold']} min_pairs={result['min_pairs']} "
        f"max_pairs={result['max_pairs']} resamples={result['resamples']} seed={result['seed']}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ds4_abba_bootstrap.py",
        description="Paired ABBA bootstrap stopping rule for DS4 gate-A levers.",
    )
    p.add_argument(
        "--glob",
        required=True,
        dest="glob_pat",
        help="Glob of pair-*-*.json bench result files",
    )
    p.add_argument(
        "--metric",
        required=True,
        help="Metric key in each bench JSON (e.g. decode_tok_s, ttft_ms)",
    )
    p.add_argument(
        "--resamples",
        type=int,
        default=10000,
        help="Bootstrap resamples (default 10000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic bootstrap (default 0)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Path to write full result JSON",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Promotion checkpoint in percent (default 0.5)",
    )
    p.add_argument(
        "--min-pairs",
        type=int,
        default=5,
        dest="min_pairs",
        help="Minimum complete pairs before promote/kill (default 5)",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=15,
        dest="max_pairs",
        help="Pair cap forcing 'cap' when CI still straddles (default 15)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resamples < 1:
        print("error: --resamples must be >= 1", file=sys.stderr)
        return 2
    if args.min_pairs < 1:
        print("error: --min-pairs must be >= 1", file=sys.stderr)
        return 2
    if args.max_pairs < args.min_pairs:
        print("error: --max-pairs must be >= --min-pairs", file=sys.stderr)
        return 2

    matched = sorted(Path(p) for p in globmod.glob(args.glob_pat))
    # Also accept directories that only match via pathlib if shell already expanded;
    # stick to glob module for the contract's --glob flag.
    if not matched:
        print(f"error: no files matched glob {args.glob_pat!r}", file=sys.stderr)
        return 1

    result = analyze(
        matched,
        metric=args.metric,
        threshold=args.threshold,
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        resamples=args.resamples,
        seed=args.seed,
    )

    # Zero valid files (every match invalid / unparseable name with no complete pair
    # material) is a hard failure when nothing usable was consumed as a value.
    any_valid_value = result["n_complete_pairs"] > 0 or any(
        # A file that parsed into a bucket with at least one value counts as valid.
        True
        for pair in result["pairs"]
    )
    # Count files that contributed a numeric value even if pairs incomplete.
    n_invalid = len(result["invalid_files"])
    n_input = len(result["input_files"])
    if n_input > 0 and n_invalid == n_input and result["n_complete_pairs"] == 0:
        # All matched files were invalid — exit nonzero per contract.
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(format_summary(result))
        print("error: zero valid files matched", file=sys.stderr)
        return 1

    _ = any_valid_value  # kept for readability; gate above covers the exit case

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

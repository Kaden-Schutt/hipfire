#!/usr/bin/env python3
"""kld_diff.py — V1/V2 four-metric A/B comparison of two HFKSEQ files.

Reads two per-sequence KLD files (typically one prefill-mode and one
per-token-mode) and reports the four metrics specified in
`docs/plans/eval_hipfire_speedup.md` §"Validation V1":

  1. mean delta: mean(A) - mean(B), abs + relative
  2. Pearson correlation on per-seq KLD vectors
  3. p99 |delta_per_seq|
  4. 95% bootstrap CI overlap on mean

Tolerance schedule per the plan (defaults; override via flags):

  abs mean delta      <= --epsilon-abs (default 5e-5; calibrate empirically)
  relative mean delta <= --epsilon-rel (default 0.01 = 1%)
  Pearson per-seq     >= --rho-min     (default 0.9999)
  p99 |delta_per_seq| <= --p99-rel     (default 0.05 = 5% of variant mean)
  95% bootstrap CI    overlapping

Decision rule:
  - All four pass            → PASS (prefill ≡ per-token within ε for this variant)
  - Any fail, ranking-preserved → SOFT FAIL (see plan §M5 escalation tree)
  - Ranking changes          → HARD FAIL (per-variant fallback)

Output exits with 0 on PASS, 1 on SOFT FAIL, 2 on HARD FAIL.

Usage:
  kld_diff.py <a.kldseq> <b.kldseq> \
              [--label-a NAME] [--label-b NAME] \
              [--epsilon-abs FLOAT] [--epsilon-rel FLOAT] \
              [--rho-min FLOAT] [--p99-rel FLOAT] \
              [--n-boot INT] [--seed INT]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kldref_format import read_per_seq_kld  # noqa: E402


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) at 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--epsilon-abs", type=float, default=5e-5)
    ap.add_argument("--epsilon-rel", type=float, default=0.01)
    ap.add_argument("--rho-min", type=float, default=0.9999)
    ap.add_argument("--p99-rel", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_means, a_p99s, _ = read_per_seq_kld(args.a)
    b_means, b_p99s, _ = read_per_seq_kld(args.b)

    if len(a_means) != len(b_means):
        sys.exit(
            f"ERROR: per-seq length mismatch: {args.a.name}={len(a_means)} vs "
            f"{args.b.name}={len(b_means)}"
        )

    label_a = args.label_a or args.a.stem
    label_b = args.label_b or args.b.stem
    a = np.array(a_means, dtype=np.float64)
    b = np.array(b_means, dtype=np.float64)
    n = len(a)

    # --- statistics ---
    mean_a, ci_lo_a, ci_hi_a = bootstrap_mean_ci(a, args.n_boot, args.seed)
    mean_b, ci_lo_b, ci_hi_b = bootstrap_mean_ci(b, args.n_boot, args.seed + 1)
    delta = a - b
    mean_delta_abs = float(mean_a - mean_b)
    base = max(abs(mean_b), 1e-12)  # use B as denominator (B = per-token = reference)
    mean_delta_rel = mean_delta_abs / base
    rho = float(np.corrcoef(a, b)[0, 1])
    p99_abs = float(np.percentile(np.abs(delta), 99))
    p99_rel_b = p99_abs / base
    ci_overlaps = not (ci_hi_a < ci_lo_b or ci_hi_b < ci_lo_a)

    # --- gates ---
    gate_abs = abs(mean_delta_abs) <= args.epsilon_abs
    gate_rel = abs(mean_delta_rel) <= args.epsilon_rel
    gate_rho = rho >= args.rho_min
    gate_p99 = p99_rel_b <= args.p99_rel
    gate_ci = ci_overlaps
    gates = [gate_abs, gate_rel, gate_rho, gate_p99, gate_ci]
    all_pass = all(gates)
    # Sign-test for systematic bias: under H0 of no bias, per-seq sign of
    # (A - B) is Bernoulli(0.5). Reject H0 (call it "one-sided / systematic
    # bias") when a binomial two-tailed p-value < 0.001 — i.e. the observed
    # one-sided count is wildly inconsistent with chance. Prior rev used a
    # naive 90%-threshold heuristic which missed the canonical V1 gfx1100
    # MQ4 result (n=1175, ~80–88% one-signed, definitively systematic but
    # below the 90% bar).
    if n > 0:
        pos = int((delta > 0).sum())
        neg = int((delta < 0).sum())
        pos_frac = pos / n
        neg_frac = neg / n
        ones = max(pos, neg)
        # Binomial CDF: P(X >= ones | n, 0.5). Two-tailed via *2.
        from math import comb
        # tail_p = 2 * Σ_{k=ones..n} C(n,k) * 0.5^n
        # For large n this underflows; switch to log-domain for n>200.
        if n <= 200:
            tail = sum(comb(n, k) for k in range(ones, n + 1)) * (0.5 ** n)
            p_value = min(1.0, 2.0 * tail)
        else:
            # Normal approx: under H0, count ~ N(n/2, sqrt(n)/2). Standardise.
            from math import erfc, sqrt
            z = (ones - n / 2) / (sqrt(n) / 2)
            p_value = float(erfc(z / sqrt(2.0)))
        one_sided = p_value < 0.001
    else:
        pos_frac = neg_frac = 0.0
        p_value = 1.0
        one_sided = False

    # --- report ---
    print(f"# kld_diff: {label_a}  vs  {label_b}")
    print(f"  n_chunks                : {n}")
    print(f"  mean KLD ({label_a:<14}): {mean_a:.6f}  (CI {ci_lo_a:.6f}–{ci_hi_a:.6f})")
    print(f"  mean KLD ({label_b:<14}): {mean_b:.6f}  (CI {ci_lo_b:.6f}–{ci_hi_b:.6f})")
    print(f"  delta                    : abs {mean_delta_abs:+.6e}  rel {mean_delta_rel:+.4%}")
    print(f"  Pearson per-seq          : {rho:.6f}")
    print(f"  p99 |delta_per_seq|      : abs {p99_abs:.6f}  rel-to-B {p99_rel_b:.4%}")
    print(f"  CI overlap               : {ci_overlaps}")
    if one_sided:
        print(
            f"  WARN: {max(pos_frac, neg_frac):.0%} of per-seq deltas are "
            f"one-signed (sign-test p={p_value:.2e}) — systematic bias"
        )
    print()
    print("# Gates")
    fmt = lambda ok, name, lhs, op, rhs: (
        f"  {'PASS' if ok else 'FAIL'}  {name:<30} {lhs}  {op}  {rhs}"
    )
    print(fmt(gate_abs, "abs mean delta",
              f"{abs(mean_delta_abs):.6e}", "<=", f"{args.epsilon_abs:.6e}"))
    print(fmt(gate_rel, "rel mean delta",
              f"{abs(mean_delta_rel):.4%}", "<=", f"{args.epsilon_rel:.4%}"))
    print(fmt(gate_rho, "Pearson rho",
              f"{rho:.6f}", ">=", f"{args.rho_min:.6f}"))
    print(fmt(gate_p99, "p99 rel-to-B",
              f"{p99_rel_b:.4%}", "<=", f"{args.p99_rel:.4%}"))
    print(fmt(gate_ci, "95% CI overlap",
              str(ci_overlaps), "==", "True"))

    print()
    if all_pass:
        print("VERDICT: PASS — prefill ≡ per-token within tolerance for this variant.")
        sys.exit(0)
    elif one_sided:
        print(
            "VERDICT: HARD FAIL — systematic bias detected (sign-test "
            f"p={p_value:.2e}). For prefill-vs-per-token comparisons this "
            "is the expected kernel-path effect documented in PRD §5.3. "
            "Investigate via V0 microtests if you see this on a same-mode A/B."
        )
        sys.exit(2)
    else:
        print("VERDICT: SOFT FAIL — some gates failed but no systematic bias detected. See plan §M5 escalation tree.")
        sys.exit(1)


if __name__ == "__main__":
    main()

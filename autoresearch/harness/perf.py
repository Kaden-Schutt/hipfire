#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""perf — the perf arm's resolution logic for the v2 certify.

The v2 correction (gap #3/#4): perf is measured on the PRODUCTION Q8 serve, and the PRIMARY
discriminator is the pinned-clock per-kernel DURATION delta (low variance, directly attributes the
change) — NOT end-to-end decode tok/s (below its own ±1-3% noise floor for a single-kernel effect).
This module is metric-agnostic: it takes interleaved base/variant sample lists and resolves a
verdict. The caller supplies DURATION samples for the bank gate; end-to-end tok/s is only a
magnitude confirmation at rollover.

Because duration is lower-is-better but the loop's convention is "variant should improve," the caller
passes samples already oriented so that HIGHER = BETTER (e.g. pass 1/duration, or throughput). Helper
`orient_lower_is_better` flips a lower-is-better metric for you.

Pure stdlib, no GPU — the stats + VOID logic are unit-testable.
"""
import statistics

WIN_F = 0.90     # Mann-Whitney dominance >= this (and gain > FLOOR) => WIN
DEAD_F = 0.65    # dominance <= this => DEAD
FLOOR = 0.15     # min |delta%| above clock noise; f carries the reliability
CAP = 16         # max interleaved A/B rounds before parking INCONCLUSIVE
CLOCK_TOL = 4.0  # % median clock skew between arms => VOID


def orient_lower_is_better(samples):
    """Flip a lower-is-better metric (e.g. kernel duration ms) so higher=better throughput."""
    return [1.0 / s for s in samples if s]


def dominance_f(base, var):
    """Mann-Whitney dominance = P(variant > baseline) over all pairs. 1.0 = variant always better."""
    n = len(base) * len(var)
    if not n:
        return 0.5
    return sum(1.0 if v > b else 0.5 if v == b else 0.0 for v in var for b in base) / n


def median_delta_pct(base, var):
    if not base or not var:
        return 0.0
    bm, vm = statistics.median(base), statistics.median(var)
    return 100 * (vm - bm) / bm if bm else 0.0


def clock_void(base_clks, var_clks, tol=CLOCK_TOL):
    """True (VOID) if the two arms ran at materially different clocks -> the delta is DPM, not kernel.
    Missing/zero clock data is NOT a void (can't prove skew) — returns False."""
    b = [c for c in base_clks if c]
    v = [c for c in var_clks if c]
    if not b or not v:
        return False
    bm, vm = statistics.median(b), statistics.median(v)
    if bm <= 0:
        return False
    return abs(vm - bm) / bm * 100.0 > tol


def should_continue(f, n, cap=CAP, dead_f=DEAD_F, win_f=WIN_F):
    """Adaptive sampling: keep sampling while the dominance is in the undecided band and under CAP."""
    return dead_f < f < win_f and n < cap


def resolve(base, var, base_clks=None, var_clks=None,
            win_f=WIN_F, dead_f=DEAD_F, floor=FLOOR):
    """Classify the accumulated (higher=better) samples.

    Returns (verdict, f, delta_pct) where verdict in {WIN, DEAD, INCONCLUSIVE, VOID}.
    VOID takes precedence (a clock-skewed measurement can't be trusted either way).
    """
    if base_clks is not None and var_clks is not None and clock_void(base_clks, var_clks):
        return ("VOID", dominance_f(base, var), median_delta_pct(base, var))
    f = dominance_f(base, var)
    d = median_delta_pct(base, var)
    if f >= win_f and d > floor:
        return ("WIN", f, d)
    if f <= dead_f or (d < -floor and f <= 0.35):
        return ("DEAD", f, d)
    return ("INCONCLUSIVE", f, d)

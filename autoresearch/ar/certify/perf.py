# Copyright (c) Kaden Schutt
"""perf — low-level perf statistics for the certify perf arm.

Ports the metric-agnostic helpers from the legacy ``autoresearch/harness/perf.py``
(dominance / median-delta / clock-VOID / adaptive sampling) and ADDS the
one-sided Mann-Whitney U p-value (`mwu`) the CONJUNCTIVE perf gate needs.

The v2 gate is conjunctive: a WIN requires kernel_decode_tok_s UP *and* rocprof
pinned-clock kernel-duration DOWN, each by an INDEPENDENT one-sided Mann-Whitney
U (`verdict.perf_result` calls `mwu` twice). This closes the failure mode where a
thermal artifact inflates tok/s while duration is flat (or a duration win hides a
tok/s regression).

Pure stdlib, no scipy — the U p-value uses the normal approximation with tie +
continuity correction (guarding sd==0 → p=1.0), which is exact-enough for the
extreme-separation regime the gate lives in and needs no external dependency.
"""
import math
import statistics
from collections import Counter

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


def _sf(z):
    """Standard-normal survival function P(Z > z) via erfc (no scipy)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mwu(a, b):
    """One-sided Mann-Whitney U p-value for H1: samples `b` are stochastically GREATER than `a`.

    Returns the p-value in [0, 1]. Uses U_b = #{y>x} + 0.5·#{y==x} over all (x∈a, y∈b) pairs,
    the tie-corrected variance, and a continuity-corrected normal approximation. Degenerate cases
    (empty arm, zero variance from all-tied values) return 1.0 — no evidence of an effect.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    u_b = sum(1.0 if y > x else 0.5 if y == x else 0.0 for y in b for x in a)
    n = n1 + n2
    mean_u = n1 * n2 / 2.0
    # tie correction from the combined-sample tie-group sizes
    counts = Counter(a) + Counter(b)
    tie_term = sum(t ** 3 - t for t in counts.values())
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0:
        return 1.0
    z = (u_b - mean_u - 0.5) / math.sqrt(var_u)   # continuity correction (upper tail)
    return _sf(z)


def resolve(base, var, base_clks=None, var_clks=None,
            win_f=WIN_F, dead_f=DEAD_F, floor=FLOOR):
    """Classify accumulated (higher=better) samples via dominance-f (legacy single-metric path).

    Returns (verdict, f, delta_pct) where verdict in {WIN, DEAD, INCONCLUSIVE, VOID}. VOID takes
    precedence (a clock-skewed measurement can't be trusted either way). Retained for the legacy
    duration-only helpers; the v2 gate uses `verdict.perf_result` (conjunctive, `mwu`-based).
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

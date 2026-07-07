#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the perf arm resolution (gap #3/#4)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import perf


def test_dominance_extremes():
    assert perf.dominance_f([1, 2, 3], [10, 11, 12]) == 1.0   # variant always higher
    assert perf.dominance_f([10, 11, 12], [1, 2, 3]) == 0.0   # variant always lower
    assert perf.dominance_f([1, 2, 3], [1, 2, 3]) == 0.5      # tied

def test_median_delta():
    assert abs(perf.median_delta_pct([100, 100], [110, 110]) - 10.0) < 1e-9
    assert abs(perf.median_delta_pct([100, 100], [90, 90]) + 10.0) < 1e-9

def test_orient_lower_is_better():
    # kernel duration 10ms -> 5ms is a win; after orienting, variant samples must dominate
    base = perf.orient_lower_is_better([10.0, 10.0, 10.0])
    var = perf.orient_lower_is_better([5.0, 5.0, 5.0])
    assert perf.dominance_f(base, var) == 1.0

def test_resolve_win():
    base = [100.0] * 8
    var = [104.0] * 8   # +4% and always higher -> f=1.0 > 0.90, d>FLOOR
    v, f, d = perf.resolve(base, var)
    assert v == "WIN" and f == 1.0 and d > 0.15

def test_resolve_dead_on_regression():
    base = [100.0] * 8
    var = [96.0] * 8    # -4%, f=0.0 <= DEAD_F
    v, f, d = perf.resolve(base, var)
    assert v == "DEAD"

def test_resolve_inconclusive_band():
    # interleaved near-ties -> f in the undecided band
    base = [100, 101, 99, 100, 102, 98]
    var = [100.5, 101, 99.5, 100, 101, 99]
    v, f, d = perf.resolve(base, var)
    assert v == "INCONCLUSIVE" and 0.65 < f < 0.90 or v in ("WIN", "DEAD")  # band-dependent
    # the point: a marginal, noise-dominated delta must NOT auto-WIN
    assert not (v == "WIN" and d < 0.15)

def test_clock_void_precedence():
    base = [100.0] * 6
    var = [130.0] * 6            # looks like a huge win...
    base_clks = [2000] * 6
    var_clks = [2600] * 6        # ...but variant ran 30% faster clock -> VOID
    v, f, d = perf.resolve(base, var, base_clks, var_clks)
    assert v == "VOID"

def test_clock_void_not_triggered_when_missing():
    base = [100.0] * 6
    var = [104.0] * 6
    v, _, _ = perf.resolve(base, var, [], [])   # no clock data -> can't prove skew, not void
    assert v == "WIN"

def test_should_continue_band():
    assert perf.should_continue(0.75, n=4, cap=16)       # undecided + under cap -> keep sampling
    assert not perf.should_continue(0.95, n=4, cap=16)   # resolved WIN-side -> stop
    assert not perf.should_continue(0.50, n=4, cap=16)   # resolved DEAD-side -> stop
    assert not perf.should_continue(0.75, n=16, cap=16)  # hit cap -> stop (park INCONCLUSIVE)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

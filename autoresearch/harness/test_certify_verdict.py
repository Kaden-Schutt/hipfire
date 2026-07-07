#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the v2 verdict combiner."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import certify_verdict as cv


def test_parity_fail_short_circuits():
    # even a coherent perf-win is rejected if the change isn't value-preserving
    assert cv.decide(parity_ok=False, coherence_ok=True, perf_verdict="WIN") == "PARITY_FAIL"

def test_coherence_hard_gate_beats_perf_win():
    # fast BUT incoherent -> COHERENCE_FAIL (not WIN) — the whole point of the redesign
    assert cv.decide(parity_ok=True, coherence_ok=False, perf_verdict="WIN") == "COHERENCE_FAIL"

def test_full_win_requires_all_three():
    assert cv.decide(parity_ok=True, coherence_ok=True, perf_verdict="WIN") == "WIN"

def test_perf_dead_passes_through():
    assert cv.decide(parity_ok=True, coherence_ok=True, perf_verdict="DEAD") == "DEAD"

def test_perf_inconclusive_passes_through():
    assert cv.decide(parity_ok=True, coherence_ok=True, perf_verdict="INCONCLUSIVE") == "INCONCLUSIVE"

def test_perf_void_passes_through():
    assert cv.decide(parity_ok=True, coherence_ok=True, perf_verdict="VOID") == "VOID"

def test_bad_perf_verdict_raises():
    try:
        cv.decide(parity_ok=True, coherence_ok=True, perf_verdict="WHATEVER")
        assert False, "should have raised"
    except ValueError:
        pass

def test_only_win_is_bankable():
    assert cv.is_bankable("WIN")
    for v in ("DEAD", "INCONCLUSIVE", "COHERENCE_FAIL", "PARITY_FAIL", "VOID"):
        assert not cv.is_bankable(v)

def test_make_row_carries_v2_fields():
    row = cv.make_row("gfx1151", "attn", "drop_barrier", "WIN",
                      parity={"fp32_exact": True, "q8_tol": True},
                      perf_delta=1.3, perf_f=1.0,
                      coherence={"pass": True, "b": 0, "c": 0, "p": 1.0, "seeds": 12},
                      base_ref="5f101504", seeds=12)
    assert row["verdict"] == "WIN" and row["WIN"] is True
    assert row["parity"]["q8_tol"] is True
    assert row["base_ref"] == "5f101504"
    assert row["coherence"]["seeds"] == 12
    assert row["label"] == "drop_barrier"   # digest reads label=lever


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

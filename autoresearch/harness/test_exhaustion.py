#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the unified exhaustion definition (gap #5/#14)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import exhaustion as ex


def test_coherence_fail_counts_toward_exhaustion():
    # the #5 bug: COHERENCE_FAIL used to never increment -> looped forever
    state = {}
    for _ in range(5):
        ex.apply_round(state, {"k": ["COHERENCE_FAIL"]})
    assert ex.is_exhausted(state, "k", K=5)

def test_dead_counts():
    state = {}
    for _ in range(5):
        ex.apply_round(state, {"k": ["DEAD"]})
    assert ex.is_exhausted(state, "k", K=5)

def test_win_resets_streak():
    state = {}
    for _ in range(4):
        ex.apply_round(state, {"k": ["DEAD"]})
    ex.apply_round(state, {"k": ["WIN"]})
    assert ex.dead_streak(state, "k") == 0
    assert not ex.is_exhausted(state, "k", K=5)

def test_inconclusive_does_not_increment_but_queues():
    # the #14 bug: INCONCLUSIVE used to count as DEAD -> abandoned sub-2% wins
    state = {}
    needs = set()
    for _ in range(10):
        _, n = ex.apply_round(state, {"k": ["INCONCLUSIVE"]})
        needs |= n
    assert ex.dead_streak(state, "k") == 0
    assert not ex.is_exhausted(state, "k", K=5)
    assert "k" in needs

def test_per_round_cap_is_one():
    # three DEAD in ONE round advances the streak by 1, not 3 (K = consecutive ROUNDS)
    state = {}
    ex.apply_round(state, {"k": ["DEAD", "DEAD", "DEAD"]})
    assert ex.dead_streak(state, "k") == 1

def test_parity_fail_counts_only_when_repeated():
    state = {}
    # 2 parity fails (below repeat=3) -> no dead progress
    for _ in range(2):
        ex.apply_round(state, {"k": ["PARITY_FAIL"]})
    assert ex.dead_streak(state, "k") == 0
    # 3rd parity fail crosses the repeat threshold -> starts counting dead
    ex.apply_round(state, {"k": ["PARITY_FAIL"]})
    assert ex.dead_streak(state, "k") == 1

def test_infra_uses_separate_cap_not_dead_streak():
    state = {}
    for _ in range(8):
        ex.apply_round(state, {"k": ["VARIANT_BUILD_FAIL"]})
    assert ex.dead_streak(state, "k") == 0          # infra never touches the dead streak
    assert ex.is_exhausted(state, "k", K=5, infra_cap=8)  # but the infra cap stops the spin

def test_all_exhausted_global_stop():
    state = {}
    for _ in range(5):
        ex.apply_round(state, {"a": ["DEAD"], "b": ["DEAD"]})
    assert ex.all_exhausted(state, ["a", "b"], K=5)
    assert not ex.all_exhausted(state, ["a", "b", "c"], K=5)  # c untouched -> not global
    assert not ex.all_exhausted(state, [], K=5)               # no candidates -> not "exhausted"

def test_win_after_dead_streak_reopens():
    state = {}
    for _ in range(5):
        ex.apply_round(state, {"k": ["COHERENCE_FAIL"]})
    assert ex.is_exhausted(state, "k", K=5)
    ex.apply_round(state, {"k": ["WIN"]})
    assert not ex.is_exhausted(state, "k", K=5)


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

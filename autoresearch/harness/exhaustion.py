#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""exhaustion — the ONE per-kernel exhaustion definition for the autoresearch loop.

Fixes gap #5/#14: the old rule counted only DEAD/INCONCLUSIVE, so a kernel that keeps failing on
COHERENCE_FAIL or PARITY_FAIL never accumulated exhaustion and the loop burned rounds on it forever;
and it counted INCONCLUSIVE (a real sub-2% win noise didn't resolve) identically to DEAD, abandoning
exactly the wins that are all that's left. It also let the driver and the supervisor keep divergent
counters. This module is the single source of truth both must import.

Verdict taxonomy:
  WIN           -> the lever helped; reset the kernel's dead streak (and parity counter).
  DEAD          -> resolved no-gain; counts toward exhaustion.
  COHERENCE_FAIL-> fast-but-incoherent; counts (a kernel only made fast by breaking coherence IS
                   exhausted for the loop).
  PARITY_FAIL   -> value-changing edit; counts only once REPEATED (parity_repeat) — a one-off may be
                   a codex slip, but a kernel where every edit changes values is out of value-
                   preserving room.
  INCONCLUSIVE  -> MWU didn't resolve (real-but-small) -> does NOT increment; routed to a
                   needs-confirmation re-measure queue instead of being treated as dead.
  BUILD_FAIL/VOID -> retryable infra -> a SEPARATE infra-fail cap so an un-buildable/void variant
                   can't spin forever, but never the dead streak.

The dead streak increments by AT MOST 1 per round, so K = consecutive ROUNDS (not attempts) — a
round that tries a kernel 3× all-dead advances the streak by 1, not 3.
"""

WIN = "WIN"
DEAD_VERDICTS = frozenset({"DEAD", "COHERENCE_FAIL", "LOSS", "NOISE"})
INFRA_VERDICTS = frozenset({"BUILD_FAIL", "VARIANT_BUILD_FAIL", "BASELINE_BUILD_FAIL", "VOID"})
INCONCLUSIVE = "INCONCLUSIVE"
PARITY_FAIL = "PARITY_FAIL"


def _st(state, kernel):
    return state.setdefault(kernel, {"dead": 0, "infra": 0, "parity": 0})


def apply_round(state, round_verdicts, parity_repeat=3):
    """Advance exhaustion state by one round.

    state: {kernel: {"dead": int, "infra": int, "parity": int}} (mutated + returned).
    round_verdicts: {kernel: [verdict, ...]} for THIS round.
    Returns (state, needs_confirmation: set[kernel]) — kernels with an INCONCLUSIVE this round and
    no dead-progress, to route to a higher-sample re-measure.
    """
    needs = set()
    for kernel, verdicts in round_verdicts.items():
        if not kernel:
            continue
        st = _st(state, kernel)
        if WIN in verdicts:
            st["dead"] = 0
            st["parity"] = 0
            continue
        st["infra"] += sum(1 for v in verdicts if v in INFRA_VERDICTS)
        st["parity"] += sum(1 for v in verdicts if v == PARITY_FAIL)
        hard_dead = any(v in DEAD_VERDICTS for v in verdicts)
        parity_dead = (PARITY_FAIL in verdicts) and st["parity"] >= parity_repeat
        if hard_dead or parity_dead:
            st["dead"] += 1                       # round-capped at +1
        elif INCONCLUSIVE in verdicts:
            needs.add(kernel)                     # real-but-small -> re-measure, do NOT count dead
    return state, needs


def is_exhausted(state, kernel, K, infra_cap=8):
    st = state.get(kernel, {})
    return st.get("dead", 0) >= K or st.get("infra", 0) >= infra_cap


def all_exhausted(state, candidates, K, infra_cap=8):
    """True iff there ARE candidates and every one is exhausted (global stop)."""
    return bool(candidates) and all(is_exhausted(state, k, K, infra_cap) for k in candidates)


def dead_streak(state, kernel):
    return state.get(kernel, {}).get("dead", 0)

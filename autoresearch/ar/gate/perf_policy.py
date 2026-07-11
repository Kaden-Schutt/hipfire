# Copyright (c) Kaden Schutt
"""ar.gate.perf_policy — the gate's perf decision, mirror of the loop's WIN gate.

The loop declares a WIN iff tok/s UP and duration DOWN, both dominant (>= WIN_F)
and beyond FLOOR. The gate declares a REGRESSION iff tok/s DOWN and duration UP,
both dominant against the variant (<= DEAD_F) and beyond FLOOR. Everything in
between — including any one-sided or sub-FLOOR move — is NEUTRAL and PASSes.
"""
from __future__ import annotations

import statistics as _st

from ..certify import perf


def _delta_pct(base, var) -> float:
    """Median %% change of var vs base (+ = var larger). Empty either side → 0.0
    (no samples ⇒ no measurable delta; never crash on median([]))."""
    if not base or not var:
        return 0.0
    mb = _st.median(base)
    return (_st.median(var) - mb) / mb * 100.0 if mb else 0.0


def classify_perf(base_tok, var_tok, base_dur, var_dur,
                  floor=perf.FLOOR, win_f=perf.WIN_F, dead_f=perf.DEAD_F) -> str:
    """Return "IMPROVEMENT" | "REGRESSION" | "NEUTRAL" (conjunctive, both axes)."""
    tok_d = _delta_pct(base_tok, var_tok)     # + = variant faster
    dur_d = _delta_pct(base_dur, var_dur)     # + = variant slower (worse)
    f_tok = perf.dominance_f(base_tok, var_tok)   # P(var_tok > base_tok): high => faster
    f_dur = perf.dominance_f(base_dur, var_dur)   # P(var_dur > base_dur): high => slower

    improvement = (f_tok >= win_f and tok_d > floor and f_dur <= dead_f and dur_d < -floor)
    if improvement:
        return "IMPROVEMENT"
    regression = (f_tok <= dead_f and tok_d < -floor and f_dur >= win_f and dur_d > floor)
    if regression:
        return "REGRESSION"
    return "NEUTRAL"

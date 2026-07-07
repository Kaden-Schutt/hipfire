#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""certify_verdict — combine the three v2 arms into one verdict + ledger row.

Precedence (v2 spec):
  1. PARITY (FP32-det text-exact AND Q8 tolerance) — value-preservation. Fails first + short-circuits
     (a value-changing variant is wrong; don't waste the Q8 serve on perf/coherence).
  2. COHERENCE — HARD gate, independent of timing. A fast-but-incoherent OR neutral-but-incoherent
     variant is COHERENCE_FAIL (which the unified exhaustion counts as a dead-progress verdict).
  3. PERF (duration-delta on Q8) — WIN only reaches here with parity+coherence already passed.

The orchestrator measures in this order and short-circuits; `decide` formalizes the precedence and
is safe to call with the results it has (skipped arms pass their known short-circuit result).
"""

VERDICTS = ("WIN", "DEAD", "INCONCLUSIVE", "VOID", "PARITY_FAIL", "COHERENCE_FAIL",
            "BUILD_FAIL", "VARIANT_BUILD_FAIL", "BASELINE_BUILD_FAIL")


def decide(parity_ok, coherence_ok, perf_verdict):
    """parity_ok/coherence_ok: bool. perf_verdict: one of WIN/DEAD/INCONCLUSIVE/VOID from perf.resolve.
    Returns the final certify verdict."""
    if not parity_ok:
        return "PARITY_FAIL"
    if not coherence_ok:
        return "COHERENCE_FAIL"
    if perf_verdict not in ("WIN", "DEAD", "INCONCLUSIVE", "VOID"):
        raise ValueError(f"bad perf_verdict {perf_verdict!r}")
    return perf_verdict          # WIN here == parity ∧ coherence ∧ perf-gain


def is_bankable(verdict):
    """Only a WIN advances the agent's baseline B_a and commits to loop/cardN."""
    return verdict == "WIN"


def make_row(arch, kernel, lever, verdict, *, parity=None, perf_delta=None, perf_f=None,
             coherence=None, base_ref=None, seeds=None, mcnemar=None, extra=None):
    """Assemble the ledger row. Carries the NEW v2 fields (parity/coherence/base_ref) alongside the
    existing shape so `ar ingest` + the digest keep working."""
    row = {
        "arch": arch, "kernel": kernel, "lever": lever, "label": lever, "verdict": verdict,
        "WIN": verdict == "WIN",
        "parity": parity,             # {"fp32_exact": bool, "q8_tol": bool|None}
        "perf_delta": perf_delta,     # kernel-duration delta % (primary), variant vs B_a
        "mwu_dominance": perf_f,
        "coherence": coherence,       # {"pass": bool, "b": int, "c": int, "p": float, "seeds": N}
        "base_ref": base_ref,         # B_a@sha the variant was measured against (advancing baseline)
        "seeds": seeds,
        "mcnemar": mcnemar,
    }
    if extra:
        row.update(extra)
    return row

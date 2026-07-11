# Copyright (c) Kaden Schutt
"""ar.gate.sweep — backlog sweep for the staging merge-train (spec §11.1).

Sweep the open eligible PRs onto a collection branch (kernel-oracle for the current
batch); land the whole stack on master once it is proven non-clobbering. Rules:

  * A PR the gate REJECTS is PUNTED — in particular a **perf regression is never
    resolved, it is skipped** ("just punted, move onto the next"). Merge *conflicts*
    (textual) are still resolved during the fold (Gate-4 merge-fix); only functional
    gate failures punt.
  * The special case ("perf superseded, but lost in merge"): if folding a PR LOSES a
    perf gain another PR won, the **perf-PRESERVING branch WINS** and supersedes the
    loser — gracefully (the fold's merge-fix tries to keep both) if possible, else the
    higher-perf branch stays on the train and the loser is dropped.

The decision logic here is injected-seam based and unit-tested; the derived-staging
rebuild after a supersession is prod wiring (staging_prod).
"""
from __future__ import annotations

__all__ = ["sweep_verdict", "punt_reason", "perf_conflict_winner", "sweep"]


def sweep_verdict(gate_result) -> str:
    """'approve' iff the gate PASSED; anything it REJECTS (perf regression / parity /
    coherence) is 'punt' — a perf regression is skipped, never resolved."""
    return "approve" if gate_result.get("verdict") == "PASS" else "punt"


def punt_reason(gate_result) -> str:
    """A human-readable punt reason from the gate's REJECT reasons."""
    reasons = gate_result.get("reasons") or [str(gate_result.get("verdict", "reject")).lower()]
    for tag, name in (("perf", "perf-regression"), ("parity", "parity"), ("coher", "coherence")):
        if any(tag in r for r in reasons):
            return name
    return reasons[0]


def perf_conflict_winner(a, b, *, perf_delta) -> dict:
    """'perf superseded, but lost in merge' -> the perf-PRESERVING branch wins. Returns
    {winner, loser}: the higher perf delta wins; a tie favors ``a`` (the incumbent)."""
    da = perf_delta.get(a, 0.0)
    db = perf_delta.get(b, 0.0)
    return {"winner": a, "loser": b} if da >= db else {"winner": b, "loser": a}


def sweep(*, open_prs, base_ref, repo, gate_fn, fold_fn, perf_delta=None) -> dict:
    """Sweep the backlog onto ``base_ref`` (the collection branch, e.g. kernel-oracle).

    gate_fn(pr) -> {"verdict", "reasons", "perf_delta"}; fold_fn(pr, staging_ref) -> a
    fold_pr-style result that may carry ``clobbered`` = the prior PR whose recorded perf
    the fold lost. Returns {train, punted, superseded, staging_ref}."""
    perf_delta = dict(perf_delta or {})
    train, punted, superseded = [], [], []
    staging_ref = base_ref
    for pr in open_prs:
        gr = gate_fn(pr)
        if sweep_verdict(gr) == "punt":
            punted.append({"pr": pr, "reason": punt_reason(gr)})   # perf regression / fail -> skip
            continue
        perf_delta.setdefault(pr, gr.get("perf_delta", 0.0))
        fold = fold_fn(pr, staging_ref)
        if fold["verdict"] == "FOLDED":
            staging_ref = fold["staging_ref"]
            train.append(pr)
            continue

        clob = fold.get("clobbered")
        if fold.get("reason") == "clobber" and clob:
            # perf lost in the merge -> the perf-preserving branch wins over the clobbered one.
            res = perf_conflict_winner(pr, clob, perf_delta=perf_delta)
            if res["winner"] == pr:                    # candidate preserves more perf
                if clob in train:
                    train.remove(clob)                 # loser superseded, dropped from the train
                superseded.append({"winner": pr, "loser": clob})
                train.append(pr)
                staging_ref = fold.get("staging_ref", staging_ref)
            else:                                      # incumbent preserves more perf -> keep it
                superseded.append({"winner": clob, "loser": pr})
                punted.append({"pr": pr, "reason": "superseded-perf"})
        else:
            punted.append({"pr": pr, "reason": fold.get("reason", "clobber")})
    return {"train": train, "punted": punted, "superseded": superseded, "staging_ref": staging_ref}

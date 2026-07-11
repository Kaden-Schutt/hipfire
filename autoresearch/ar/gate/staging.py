# Copyright (c) Kaden Schutt
"""ar.gate.staging — the staging merge-train (spec §11 + §10 recall-reproduce).

Stack gate-approved PRs onto a derived `staging`, resolving conflicts via the
Gate-4 merge-fix instead of punting, validating each fold by RECALLING the PR's
already-recorded behaviors (not re-running), and landing the whole train to master
in one non-clobber merge. Every side-effect (git, codex merge-fix, recall-reproduce)
is an injected seam, so this is unit-testable with no GPU and no real git.
"""
from __future__ import annotations

from .merge import trial_merge as _trial_merge

__all__ = ["classify_conflict", "recall_reproduce", "fold_pr", "stack_train", "land_train"]


def classify_conflict(pr_ref, master_ref, repo, *, trial_merge_fn=None) -> str:
    """Split a fold conflict: 'stale' (conflicts with master itself -> rebase) vs
    'stack' (clean vs master, so it conflicts with an already-folded PR)."""
    tm = trial_merge_fn or (lambda b, h, r: _trial_merge(b, h, r))
    return "stack" if tm(master_ref, pr_ref, repo)["clean"] else "stale"


def recall_reproduce(pr_ref, merged_ref, recorded, repo, *, reproduce_fn) -> dict:
    """Confirm the PR's ALREADY-RECORDED behaviors REPRODUCE on the merged tree
    (spec §10). Does not re-run the full PR gate or re-measure master — delegates to
    reproduce_fn, which re-runs only ``recorded`` on ``merged_ref``. Empty recorded
    -> trivially reproduced (no call)."""
    if not recorded:
        return {"reproduced": True, "failures": []}
    r = reproduce_fn(pr_ref, merged_ref, recorded, repo)
    return {"reproduced": bool(r.get("reproduced")), "failures": list(r.get("failures", []))}


def fold_pr(*, pr_ref, staging_ref, master_ref, repo, recorded,
            trial_merge_fn, merge_fix_fn=None, reproduce_fn) -> dict:
    """Fold one PR onto staging: trial-merge -> (resolve) -> recall-reproduce -> FOLDED/BOD."""
    def _validate(stg):
        tm = trial_merge_fn(stg, pr_ref, repo)
        if not tm["clean"]:
            return None
        rr = recall_reproduce(pr_ref, tm["merged_tree"], recorded, repo, reproduce_fn=reproduce_fn)
        if not rr["reproduced"]:
            return {"pr": pr_ref, "verdict": "BOD", "staging_ref": stg,
                    "reason": "clobber", "detail": ", ".join(rr["failures"])}
        return {"pr": pr_ref, "verdict": "FOLDED", "staging_ref": tm["merged_tree"],
                "reason": "folded", "detail": ""}

    res = _validate(staging_ref)
    if res is not None:
        return res

    # trial conflicted: try the codex merge-fix (resolve on staging), then re-validate.
    if merge_fix_fn is not None:
        fix = merge_fix_fn(pr_ref, staging_ref, repo)
        if fix.get("resolved"):
            res = _validate(fix["staging_ref"])
            if res is not None:
                return res

    # unresolved -> BOD, split reason for an actionable message.
    reason = classify_conflict(pr_ref, master_ref, repo, trial_merge_fn=trial_merge_fn)
    return {"pr": pr_ref, "verdict": "BOD", "staging_ref": staging_ref,
            "reason": reason, "detail": "rebase on master" if reason == "stale"
            else "conflicts with an already-approved PR on the stack"}

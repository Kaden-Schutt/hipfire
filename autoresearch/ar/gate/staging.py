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
    """Fold one PR onto staging: trial-merge -> (merge-fix resolve) -> recall-reproduce
    -> FOLDED/BOD. Per spec §10 BOTH a textual conflict AND a semantic clobber (clean
    merge but a recorded behavior fails to reproduce) get a merge-fix attempt before a
    BOD — the fixer is never skipped for a clobber."""
    def _try(stg):
        # -> ("FOLDED", merged_tree) | ("clobber", detail) | ("conflict", None)
        tm = trial_merge_fn(stg, pr_ref, repo)
        if not tm["clean"]:
            return ("conflict", None)
        rr = recall_reproduce(pr_ref, tm["merged_tree"], recorded, repo, reproduce_fn=reproduce_fn)
        if not rr["reproduced"]:
            return ("clobber", ", ".join(rr["failures"]))
        return ("FOLDED", tm["merged_tree"])

    outcome, data = _try(staging_ref)
    if outcome == "FOLDED":
        return {"pr": pr_ref, "verdict": "FOLDED", "staging_ref": data, "reason": "folded", "detail": ""}

    # Conflict OR clobber -> dispatch the codex merge-fix (resolve on staging), re-validate.
    if merge_fix_fn is not None:
        fix = merge_fix_fn(pr_ref, staging_ref, repo)
        if fix.get("resolved"):
            o2, d2 = _try(fix["staging_ref"])
            if o2 == "FOLDED":
                return {"pr": pr_ref, "verdict": "FOLDED", "staging_ref": d2, "reason": "folded", "detail": ""}

    # Unresolved -> BOD with an actionable reason.
    if outcome == "clobber":
        return {"pr": pr_ref, "verdict": "BOD", "staging_ref": staging_ref,
                "reason": "clobber", "detail": data}
    reason = classify_conflict(pr_ref, master_ref, repo, trial_merge_fn=trial_merge_fn)
    return {"pr": pr_ref, "verdict": "BOD", "staging_ref": staging_ref,
            "reason": reason, "detail": "rebase on master" if reason == "stale"
            else "conflicts with an already-approved PR on the stack"}


def stack_train(*, approved_prs, master_ref, repo, fold_fn) -> dict:
    """Greedily stack the approved PRs onto a derived staging (starting at master).
    fold_fn(pr, staging_ref) -> a fold_pr-style result. FOLDED advances the tip and
    joins the train; BOD is collected as debt (tip unchanged) and stacking continues."""
    staging_ref = master_ref
    train, debt = [], []
    for pr in approved_prs:
        res = fold_fn(pr, staging_ref)
        if res["verdict"] == "FOLDED":
            staging_ref = res["staging_ref"]
            train.append(pr)
        else:
            debt.append({"pr": pr, "reason": res["reason"], "detail": res.get("detail", "")})
    return {"train": train, "debt": debt, "staging_ref": staging_ref}


def land_train(*, train, staging_ref, master_ref, repo, git, land_reproduce_fn) -> dict:
    """Flush the whole approved train to master in one non-clobber merge (spec §11).
    Re-validates the LANDED result (recall-reproduce the stack's behaviors) BEFORE
    finalizing, then merges non-squash so folded commits become master ancestors; the
    folded PRs are returned in ``closed`` for close-behind bookkeeping."""
    if not train:
        return {"landed": False, "master_sha": None, "closed": [], "reason": "empty-train"}

    rr = land_reproduce_fn(staging_ref, master_ref, repo)
    if not rr.get("reproduced"):
        return {"landed": False, "master_sha": None, "closed": [], "reason": "landing-clobber"}

    rc, _ = git(repo, "merge", "--no-ff", staging_ref)
    if rc != 0:
        return {"landed": False, "master_sha": None, "closed": [], "reason": "merge-failed"}
    _, head = git(repo, "rev-parse", "HEAD")
    return {"landed": True, "master_sha": head.strip(), "closed": list(train), "reason": "landed"}

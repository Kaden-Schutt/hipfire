# Copyright (c) Kaden Schutt
"""ar.gate.staging_prod — the production seam bindings for the staging merge-train.

The tested core (staging.py) is seam-injected; this module binds those seams to the
real world — git, codex-on-staging merge-fix, the recall-reproduce re-run, and the
GitHub close-behind. These bindings touch git / codex / the GPU / the GitHub API, so
they are LIVE-VALIDATED (exercised on the first real landing), not unit-tested; the
train LOGIC they drive is fully covered in test_gate_staging.py.
"""
from __future__ import annotations

import json
import subprocess

from . import staging
from .merge import default_run_git, trial_merge


def codex_merge_fix(pr_ref, staging_ref, repo, *, model="gpt-5.6-sol", effort="xhigh"):
    """Resolve a fold conflict ON the agent-owned staging branch via codex (spec §11).
    codex rebases the PR onto the staging tip and resolves conflicts; on success we
    return the new staging commit. Never pushes to the PR's own branch — the fix lives
    on staging, so it works for fork PRs too. Returns {resolved: bool, staging_ref: str}."""
    from ..agent_exec import run_round_resilient

    prompt = (
        f"On the git repo at {repo}, the branch/ref `{staging_ref}` is the staging tip and "
        f"`{pr_ref}` is a PR to fold onto it. `git merge {pr_ref}` (or a rebase of {pr_ref} "
        f"onto {staging_ref}) conflicts. RESOLVE the conflict on {staging_ref} — keep BOTH the "
        f"stacked changes and this PR's intent, do not drop either side's behavior — and commit "
        f"the resolution on {staging_ref}. Do NOT touch the PR's own branch. When the working "
        f"tree is clean and committed, print the resulting commit sha on a line by itself as "
        f"`RESOLVED <sha>`; if you cannot resolve it, print `UNRESOLVED`."
    )
    # gpt-5.6-sol => sol-class: on a codex usage-limit this WAITS for reset (no grok
    # downgrade) — a merge-conflict resolution needs the top tier.
    run_round_resilient(harness="codex", model=model, effort=effort, prompt=prompt, cwd=repo)
    # The resolution's new staging tip = staging's HEAD after codex committed.
    code, out = default_run_git(repo, "rev-parse", staging_ref)
    new_ref = out.strip() if code == 0 else staging_ref
    # Trust the RESULT, not codex's exit code: re-confirm the PR now merges cleanly onto
    # the new staging tip. If it does, the fix resolved it.
    tm = trial_merge(new_ref, pr_ref, repo)
    return {"resolved": bool(tm["clean"]), "staging_ref": new_ref}


def reproduce_recorded(pr_ref, merged_ref, recorded, repo, *, arch_gate_fn):
    """Recall-reproduce (spec §10): re-run ONLY the PR's already-recorded behaviors on
    the merged tree — never the full PR gate, never a fresh master measurement. Delegates
    to arch_gate_fn (bound to run_gate + the §8 behavior tests over `merged_ref`) and
    reports which recorded behaviors failed to reproduce. Returns {reproduced, failures}."""
    if not recorded:
        return {"reproduced": True, "failures": []}
    res = arch_gate_fn(merged_ref, recorded)          # GPU: re-run recorded behaviors on merged_ref
    failures = list(res.get("failures", []))
    return {"reproduced": not failures, "failures": failures}


def close_behind(prs, master_sha, repo, *, gh=None):
    """Close each folded PR behind the landed stack (spec §11 GitHub close semantics):
    a real (non-squash) landing makes the commits master ancestors; each PR is closed
    with a `landed via staging stack -> <sha>` comment. gh(args) -> (rc, out) injected;
    default shells `gh`. Returns the list of closed PR numbers."""
    run = gh or (lambda *a: (subprocess.run(["gh", *a], capture_output=True, text=True).returncode, ""))
    closed = []
    for pr in prs:
        body = f"Landed via staging stack → {master_sha}. Closing behind the train."
        run("pr", "comment", str(pr), "--body", body, "--repo", repo)
        run("pr", "close", str(pr), "--repo", repo)
        closed.append(pr)
    return closed


def stage_train(*, approved_prs, master_ref, repo, arch_gate_fn, recorded_by_pr,
                run_git=None, fix_model="gpt-5.6-sol", fix_effort="xhigh"):
    """Derive staging + stack the approved PRs with REAL seams (resolve-not-punt).
    ``recorded_by_pr[pr]`` = that PR's recorded behaviors (from its gate run) to recall.
    Returns the stack_train result {train, debt, staging_ref}."""
    def fold_fn(pr, staging_ref):
        return staging.fold_pr(
            pr_ref=pr, staging_ref=staging_ref, master_ref=master_ref, repo=repo,
            recorded=recorded_by_pr.get(pr, []),
            trial_merge_fn=lambda b, h, r: trial_merge(b, h, r),
            merge_fix_fn=lambda p, s, r: codex_merge_fix(p, s, r, model=fix_model, effort=fix_effort),
            reproduce_fn=lambda p, merged, rec, r: reproduce_recorded(p, merged, rec, r, arch_gate_fn=arch_gate_fn),
        )

    return staging.stack_train(approved_prs=approved_prs, master_ref=master_ref, repo=repo, fold_fn=fold_fn)


def sweep_backlog(*, open_prs, collection_ref, repo, arch, dev=0, card=0,
                  recorded_by_pr=None, gate_config=None):
    """LIVE: sweep the open PRs onto the collection branch (spec §11.1). Binds the
    sweep's gate_fn (run the live gate per PR vs the collection tip → verdict +
    perf_delta) and fold_fn (fold_pr with real trial-merge + codex merge-fix +
    recall-reproduce). Perf regressions punt; conflicts resolve; a fold that loses a
    perf gain triggers the perf-preserving supersede. Returns sweep()'s result. This is
    LIVE-VALIDATED glue over the tested sweep/staging cores."""
    import os

    from . import run, staging
    from .config import load_gate_config
    from .sweep import sweep as _sweep_run  # the fn; `gate.sweep` re-exports it, not the module

    cfg = load_gate_config(gate_config or os.path.join(repo, "autoresearch", "config", "pr_gate.toml"))
    recorded_by_pr = recorded_by_pr or {}

    # Fetch each PR's head into a stable local ref (works for forks via refs/pull/N/head).
    pr_ref = {}
    for pr in open_prs:
        ref = f"refs/sweep/pr-{pr}"
        default_run_git(repo, "fetch", "-q", "origin", f"pull/{pr}/head:{ref}")
        pr_ref[pr] = ref

    def _arch_gate(ref, recorded=None):
        r = run.live_arch_gate(arch, [], collection_ref, ref, repo, cfg, dev=dev, card=card)
        return {"failures": [] if r["verdict"] == "PASS" else list(r.get("reasons", []))}

    def gate_fn(pr):
        r = run.live_arch_gate(arch, [], collection_ref, pr_ref[pr], repo, cfg, dev=dev, card=card)
        return {"verdict": "PASS" if r["verdict"] == "PASS" else "REJECT",
                "reasons": list(r.get("reasons", [])),
                "perf_delta": float(r.get("tok_delta_pct") or 0.0)}

    def fold_fn(pr, staging_ref):
        return staging.fold_pr(
            pr_ref=pr_ref[pr], staging_ref=staging_ref, master_ref=collection_ref, repo=repo,
            recorded=recorded_by_pr.get(pr, []),
            trial_merge_fn=lambda b, h, r: trial_merge(b, h, r),
            merge_fix_fn=lambda p, s, r: codex_merge_fix(p, s, r),
            reproduce_fn=lambda p, merged, rec, r: reproduce_recorded(p, merged, rec, r, arch_gate_fn=_arch_gate),
        )

    return _sweep_run(open_prs=open_prs, base_ref=collection_ref, repo=repo,
                      gate_fn=gate_fn, fold_fn=fold_fn)


def land(*, train, staging_ref, master_ref, repo, arch_gate_fn, recorded_all, run_git=None):
    """Land the whole train to master with REAL seams, then close-behind. ``recorded_all``
    = the union of every folded PR's recorded behaviors, re-validated on the landed master
    (spec §11 landing re-validation). Returns land_train's result + ``closed``."""
    git = run_git or default_run_git

    def land_reproduce_fn(stg, master, r):
        return reproduce_recorded("stack", stg, recorded_all, r, arch_gate_fn=arch_gate_fn)

    res = staging.land_train(train=train, staging_ref=staging_ref, master_ref=master_ref,
                             repo=repo, git=git, land_reproduce_fn=land_reproduce_fn)
    if res["landed"]:
        res["closed"] = close_behind(res["closed"], res["master_sha"], repo)
    return res

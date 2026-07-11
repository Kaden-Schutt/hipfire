# Copyright (c) Kaden Schutt
"""ar.gate.merge — Gate 4: non-clobber merge test (spec §10).

trial_merge does a `git merge-tree --write-tree` merge of the PR into the staging
tip WITHOUT touching the working tree, reporting conflicts. gate4 orchestrates
merge -> (codex merge-fix) -> re-gate the merged tree -> (merge-fix) -> PASS/BOD.
Every side-effect (git, the merged-tree gate, the codex fixer) is an injected
seam, so the flow is unit-testable with no GPU and no real git.
"""
from __future__ import annotations

import subprocess


def default_run_git(repo: str, *args: str) -> tuple[int, str]:
    """Run `git -C repo <args>`; return (returncode, stdout). stderr is folded in
    so a caller can surface it, but the parse only needs stdout."""
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return p.returncode, p.stdout


def trial_merge(base_ref: str, head_ref: str, repo: str, run_git=None) -> dict:
    """Trial-merge head_ref into base_ref via `git merge-tree --write-tree`
    (no working-tree touch). Returns {clean, merged_tree, conflicts}."""
    run = run_git or default_run_git
    rc, out = run(repo, "merge-tree", "--write-tree", "--name-only", base_ref, head_ref)
    lines = out.splitlines()
    merged_tree = lines[0].strip() if lines else None
    if rc == 0:
        return {"clean": True, "merged_tree": merged_tree, "conflicts": []}
    # rc != 0: conflicts. Real `git merge-tree --write-tree --name-only` layout is
    #   <tree OID>\n<conflicted path>*\n\n<freeform informational messages>
    # i.e. the conflicted paths are the lines AFTER the OID and BEFORE the first
    # blank separator; everything after the blank ("Auto-merging …", "CONFLICT …")
    # is non-machine-parseable prose and must NOT be treated as a path.
    conflicts: list[str] = []
    for ln in lines[1:]:
        if not ln.strip():
            break                       # blank line ends the conflicted-path section
        conflicts.append(ln.strip())
    return {"clean": False, "merged_tree": merged_tree, "conflicts": conflicts}


# run_gate REJECT reasons (autoresearch/ar/gate/engine.py) -> BOD blocker kinds.
# A reason with no explicit mapping is itemized under its own name (never dropped).
_REASON_KIND = {
    "perf_regression": "perf_regression",
    "coherence": "coherence",
    "parity": "parity",
    "cross_arch": "cross_arch",
}


def assemble_bod(*, conflicts=None, perf_regressions=None, coherence_fails=None,
                 reasons=None) -> dict:
    """Assemble the Bill of Debt — the itemized blockers a PR must clear (spec §10).

    ``reasons`` itemizes raw run_gate REJECT reasons generically (parity /
    cross_arch / perf_regression / coherence / …), so no real blocker is ever
    silently dropped."""
    blockers: list[dict] = []
    for c in conflicts or []:
        blockers.append({"kind": "merge_conflict", "detail": c})
    for r in perf_regressions or []:
        blockers.append({"kind": "perf_regression", "detail": r})
    for c in coherence_fails or []:
        blockers.append({"kind": "coherence", "detail": c})
    for r in reasons or []:
        blockers.append({"kind": _REASON_KIND.get(r, r), "detail": r})
    summary = f"{len(blockers)} blocker(s)" if blockers else "no blockers"
    return {"blockers": blockers, "summary": summary}


def gate4(*, base_ref, head_ref, staging_ref, repo, run_merged_gate,
          merge_fix=None, trial_merge_fn=None) -> dict:
    """Gate 4 (spec §10): non-clobber merge test with a codex merge-fix retry.

    merge -> (fix conflict, re-trial) -> re-gate merged tree -> (fix clobber,
    re-gate) -> PASS or BOD. All side-effects are injected seams."""
    tmf = trial_merge_fn or (lambda b, h, r: trial_merge(b, h, r))

    # 1. Trial-merge the PR into the staging tip.
    tm = tmf(staging_ref, head_ref, repo)
    if not tm["clean"]:
        if merge_fix is None or not merge_fix("merge_conflict", tm["conflicts"]).get("fixed"):
            return {"verdict": "BOD", "gate": None, "merged": tm,
                    "bod": assemble_bod(conflicts=tm["conflicts"])}
        tm = tmf(staging_ref, head_ref, repo)                       # re-trial after fix
        if not tm["clean"]:
            return {"verdict": "BOD", "gate": None, "merged": tm,
                    "bod": assemble_bod(conflicts=tm["conflicts"])}

    # 2. Re-gate the MERGED tree (does the merge clobber?).
    g = run_merged_gate()
    if g["verdict"] == "PASS":
        return {"verdict": "PASS", "bod": None, "merged": tm, "gate": g}

    # 3. Post-merge clobber: try one codex merge-fix, then re-gate.
    if merge_fix is not None and merge_fix("clobber", g.get("reasons", [])).get("fixed"):
        g = run_merged_gate()
        if g["verdict"] == "PASS":
            return {"verdict": "PASS", "bod": None, "merged": tm, "gate": g}

    # Itemize EVERY REJECT reason (parity / cross_arch / perf_regression /
    # coherence) so a parity- or cross-arch-only clobber still yields a populated,
    # non-self-contradictory BOD.
    bod = assemble_bod(reasons=g.get("reasons", []))
    return {"verdict": "BOD", "bod": bod, "merged": tm, "gate": g}

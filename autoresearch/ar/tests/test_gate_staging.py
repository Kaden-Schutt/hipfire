# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.staging import classify_conflict


def _tm(clean):
    return lambda base, head, repo: {"clean": clean, "merged_tree": "t", "conflicts": [] if clean else ["f"]}


def test_conflict_vs_master_is_stale():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(False)) == "stale"


def test_clean_vs_master_is_stack_conflict():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(True)) == "stack"


from autoresearch.ar.gate.staging import recall_reproduce


def test_recall_reproduce_delegates_and_passes():
    rf = lambda pr, merged, rec, repo: {"reproduced": True, "failures": []}
    out = recall_reproduce("pr", "merged", ["parity", "coh"], "/r", reproduce_fn=rf)
    assert out["reproduced"] is True


def test_recall_reproduce_reports_failures():
    rf = lambda pr, merged, rec, repo: {"reproduced": False, "failures": ["behavior:cli"]}
    out = recall_reproduce("pr", "merged", ["cli"], "/r", reproduce_fn=rf)
    assert out["reproduced"] is False and out["failures"] == ["behavior:cli"]


def test_recall_reproduce_empty_recorded_is_trivially_reproduced():
    called = {"n": 0}

    def rf(pr, merged, rec, repo):
        called["n"] += 1
        return {"reproduced": True, "failures": []}

    out = recall_reproduce("pr", "merged", [], "/r", reproduce_fn=rf)
    assert out["reproduced"] is True and called["n"] == 0   # nothing to reproduce -> no call


from autoresearch.ar.gate.staging import fold_pr


def _clean_tm(base, head, repo):
    return {"clean": True, "merged_tree": "merged-" + head, "conflicts": []}


def _conflict_tm(base, head, repo):
    return {"clean": False, "merged_tree": "t", "conflicts": ["f.rs"]}


def _repro(ok):
    return lambda pr, merged, rec, repo: {"reproduced": ok, "failures": [] if ok else ["behavior:x"]}


_K = dict(master_ref="master", repo="/r", recorded=["parity"])


def test_clean_fold_reproduces_is_folded():
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=_clean_tm, merge_fix_fn=None, reproduce_fn=_repro(True))
    assert r["verdict"] == "FOLDED" and r["staging_ref"] == "merged-pr1"


def test_clean_fold_but_behavior_broken_is_bod_clobber():
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=_clean_tm, merge_fix_fn=None, reproduce_fn=_repro(False))
    assert r["verdict"] == "BOD" and r["reason"] == "clobber"


def test_conflict_no_fixer_is_bod_with_split_reason():
    # conflicts on staging; clean vs master -> 'stack'
    def tm(base, head, repo):
        return {"clean": base == "master", "merged_tree": "t", "conflicts": ["f"]}
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=tm, merge_fix_fn=None, reproduce_fn=_repro(True))
    assert r["verdict"] == "BOD" and r["reason"] == "stack"


def test_conflict_fixed_then_reproduces_is_folded():
    calls = {"n": 0}

    def tm(base, head, repo):
        calls["n"] += 1
        # first trial (on 'stg') conflicts; after fix, trial on 'fixed' is clean
        return {"clean": base == "fixed", "merged_tree": "merged", "conflicts": ["f"]}

    fix = lambda pr, stg, repo: {"resolved": True, "staging_ref": "fixed"}
    r = fold_pr(pr_ref="pr1", staging_ref="stg", **_K,
                trial_merge_fn=tm, merge_fix_fn=fix, reproduce_fn=_repro(True))
    assert r["verdict"] == "FOLDED" and r["staging_ref"] == "merged"


from autoresearch.ar.gate.staging import stack_train


def test_stack_train_folds_clean_and_collects_debt():
    # fold_fn: pr2 BODs (stale); others FOLD, advancing the tip.
    def fold_fn(pr, staging_ref):
        if pr == "pr2":
            return {"pr": pr, "verdict": "BOD", "staging_ref": staging_ref,
                    "reason": "stale", "detail": "rebase on master"}
        return {"pr": pr, "verdict": "FOLDED", "staging_ref": staging_ref + "+" + pr,
                "reason": "folded", "detail": ""}

    out = stack_train(approved_prs=["pr1", "pr2", "pr3"], master_ref="M", repo="/r", fold_fn=fold_fn)
    assert out["train"] == ["pr1", "pr3"]
    assert [d["pr"] for d in out["debt"]] == ["pr2"] and out["debt"][0]["reason"] == "stale"
    assert out["staging_ref"] == "M+pr1+pr3"     # tip advanced only by folded PRs


def test_stack_train_empty_is_master():
    out = stack_train(approved_prs=[], master_ref="M", repo="/r", fold_fn=lambda p, s: None)
    assert out["train"] == [] and out["staging_ref"] == "M" and out["debt"] == []

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

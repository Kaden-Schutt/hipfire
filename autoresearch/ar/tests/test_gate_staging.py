# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.staging import classify_conflict


def _tm(clean):
    return lambda base, head, repo: {"clean": clean, "merged_tree": "t", "conflicts": [] if clean else ["f"]}


def test_conflict_vs_master_is_stale():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(False)) == "stale"


def test_clean_vs_master_is_stack_conflict():
    assert classify_conflict("pr", "master", "/r", trial_merge_fn=_tm(True)) == "stack"

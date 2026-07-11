# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.sweep import (
    perf_conflict_winner,
    punt_reason,
    sweep,
    sweep_verdict,
)


# ---- sweep_verdict / punt_reason ----

def test_pass_is_approve_reject_is_punt():
    assert sweep_verdict({"verdict": "PASS"}) == "approve"
    assert sweep_verdict({"verdict": "REJECT", "reasons": ["perf_regression"]}) == "punt"
    assert sweep_verdict({"verdict": "BOD"}) == "punt"


def test_punt_reason_names_the_failure():
    assert punt_reason({"verdict": "REJECT", "reasons": ["perf_regression"]}) == "perf-regression"
    assert punt_reason({"verdict": "REJECT", "reasons": ["parity"]}) == "parity"
    assert punt_reason({"verdict": "REJECT", "reasons": ["coherence"]}) == "coherence"


# ---- perf_conflict_winner: perf-preserving wins ----

def test_perf_preserving_branch_wins():
    # a improves +3%, b clobbers it (-0.1%) -> a (perf-preserving) wins.
    assert perf_conflict_winner("a", "b", perf_delta={"a": 3.0, "b": -0.1}) == {"winner": "a", "loser": "b"}
    # b has the bigger perf gain -> b wins.
    assert perf_conflict_winner("a", "b", perf_delta={"a": 1.0, "b": 4.0}) == {"winner": "b", "loser": "a"}
    # tie -> incumbent (a) wins.
    assert perf_conflict_winner("a", "b", perf_delta={"a": 2.0, "b": 2.0})["winner"] == "a"


# ---- sweep orchestration ----

def _gate(verdicts):
    """gate_fn from a dict pr -> gate_result."""
    return lambda pr: verdicts[pr]


def test_sweep_folds_approved_and_punts_perf_regression():
    gates = {
        "pr1": {"verdict": "PASS", "perf_delta": 0.0},
        "pr2": {"verdict": "REJECT", "reasons": ["perf_regression"]},   # PUNT, do not resolve
        "pr3": {"verdict": "PASS", "perf_delta": 2.0},
    }

    def fold_fn(pr, staging_ref):
        return {"pr": pr, "verdict": "FOLDED", "staging_ref": staging_ref + "+" + pr}

    out = sweep(open_prs=["pr1", "pr2", "pr3"], base_ref="ko", repo="/r",
                gate_fn=_gate(gates), fold_fn=fold_fn)
    assert out["train"] == ["pr1", "pr3"]                          # pr2 never folded
    assert out["punted"] == [{"pr": "pr2", "reason": "perf-regression"}]
    assert out["staging_ref"] == "ko+pr1+pr3"


def test_sweep_perf_clobber_candidate_wins_supersedes_incumbent():
    # pr1 folds (perf +1). pr2 folds but CLOBBERS pr1's perf; pr2 has the bigger gain (+4)
    # -> pr2 (perf-preserving) supersedes pr1: pr1 dropped from train, pr2 joins.
    gates = {"pr1": {"verdict": "PASS", "perf_delta": 1.0},
             "pr2": {"verdict": "PASS", "perf_delta": 4.0}}

    def fold_fn(pr, staging_ref):
        if pr == "pr1":
            return {"pr": "pr1", "verdict": "FOLDED", "staging_ref": "ko+pr1"}
        return {"pr": "pr2", "verdict": "BOD", "reason": "clobber", "clobbered": "pr1",
                "staging_ref": "ko+pr2"}

    out = sweep(open_prs=["pr1", "pr2"], base_ref="ko", repo="/r",
                gate_fn=_gate(gates), fold_fn=fold_fn)
    assert out["train"] == ["pr2"]                                 # pr1 superseded + dropped
    assert {"winner": "pr2", "loser": "pr1"} in out["superseded"]
    assert out["punted"] == []


def test_sweep_perf_clobber_incumbent_wins_punts_candidate():
    # pr1 folds (perf +5). pr2 clobbers pr1 but pr2's gain is smaller (+1)
    # -> pr1 (perf-preserving) wins; pr2 is punted as superseded-perf.
    gates = {"pr1": {"verdict": "PASS", "perf_delta": 5.0},
             "pr2": {"verdict": "PASS", "perf_delta": 1.0}}

    def fold_fn(pr, staging_ref):
        if pr == "pr1":
            return {"pr": "pr1", "verdict": "FOLDED", "staging_ref": "ko+pr1"}
        return {"pr": "pr2", "verdict": "BOD", "reason": "clobber", "clobbered": "pr1"}

    out = sweep(open_prs=["pr1", "pr2"], base_ref="ko", repo="/r",
                gate_fn=_gate(gates), fold_fn=fold_fn)
    assert out["train"] == ["pr1"]                                 # incumbent kept
    assert {"winner": "pr1", "loser": "pr2"} in out["superseded"]
    assert {"pr": "pr2", "reason": "superseded-perf"} in out["punted"]


def test_sweep_unresolvable_conflict_is_punted():
    gates = {"pr1": {"verdict": "PASS", "perf_delta": 0.0}}

    def fold_fn(pr, staging_ref):
        return {"pr": pr, "verdict": "BOD", "reason": "stale", "detail": "rebase on master"}

    out = sweep(open_prs=["pr1"], base_ref="ko", repo="/r", gate_fn=_gate(gates), fold_fn=fold_fn)
    assert out["train"] == [] and out["punted"] == [{"pr": "pr1", "reason": "stale"}]

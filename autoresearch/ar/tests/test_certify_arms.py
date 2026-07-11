# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the certify decision arms (Task 2.1).

Ports test_certify_verdict.py + test_coherence_arm.py and adds the NEW
conjunctive-perf contract: a WIN requires kernel_decode_tok_s UP *and* rocprof
kernel-duration DOWN, each by an independent one-sided Mann-Whitney U. A gain in
only one statistic ⇒ DEAD (closes the thermal-inflation / hidden-regression hole).
"""
from autoresearch.ar.certify.verdict import (
    perf_result,
    parity_result,
    coherence_result,
    decide,
    is_bankable,
    make_row,
)
from autoresearch.ar.certify import coherence as ca


# ---------------------------------------------------------------------------
# conjunctive perf gate — the NEW contract (Task 2.1 canonical tests)
# ---------------------------------------------------------------------------

def test_perf_win_requires_both():
    r = perf_result(base_tok=[150, 151, 149, 150, 150], var_tok=[160, 161, 159, 160, 160],
                    base_dur=[10.0, 10.1, 9.9, 10.0, 10.0], var_dur=[9.0, 9.1, 8.9, 9.0, 9.0])
    assert r["verdict"] == "WIN" and r["tok_delta_pct"] > 0 and r["dur_delta_pct"] < 0


def test_perf_tok_up_dur_flat_is_dead():
    r = perf_result([150] * 5, [160] * 5, base_dur=[10.0] * 5, var_dur=[10.0] * 5)
    assert r["verdict"] == "DEAD"


def test_perf_dur_down_tok_flat_is_dead():
    r = perf_result([150] * 5, [150] * 5, base_dur=[10.0] * 5, var_dur=[9.0] * 5)
    assert r["verdict"] == "DEAD"


def test_perf_regression_is_dead():
    # variant SLOWER on both axes: tok down, dur up -> DEAD (never a WIN)
    r = perf_result([160] * 5, [150] * 5, base_dur=[9.0] * 5, var_dur=[10.0] * 5)
    assert r["verdict"] == "DEAD"


def test_perf_reports_both_stats():
    r = perf_result([150, 151, 149, 150, 150], [160, 161, 159, 160, 160],
                    [10.0, 10.1, 9.9, 10.0, 10.0], [9.0, 9.1, 8.9, 9.0, 9.0])
    for k in ("verdict", "tok_delta_pct", "dur_delta_pct", "tok_p", "dur_p"):
        assert k in r
    assert 0.0 <= r["tok_p"] <= 1.0 and 0.0 <= r["dur_p"] <= 1.0


def test_perf_no_samples_inconclusive():
    r = perf_result([], [], [], [])
    assert r["verdict"] == "INCONCLUSIVE"


def test_perf_clock_skew_inconclusive():
    # a "win" produced entirely by a 30% clock skew must NOT be a WIN
    r = perf_result([100] * 6, [130] * 6, [10.0] * 6, [7.0] * 6,
                    base_clk=[2000] * 6, var_clk=[2600] * 6)
    assert r["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# verdict combiner (ported from test_certify_verdict.py)
# ---------------------------------------------------------------------------

def test_parity_fail_short_circuits():
    assert decide(parity_ok=False, coherence_ok=True, perf_verdict="WIN") == "PARITY_FAIL"


def test_coherence_hard_gate_beats_perf_win():
    assert decide(parity_ok=True, coherence_ok=False, perf_verdict="WIN") == "COHERENCE_FAIL"


def test_full_win_requires_all_three():
    assert decide(parity_ok=True, coherence_ok=True, perf_verdict="WIN") == "WIN"


def test_perf_dead_passes_through():
    assert decide(parity_ok=True, coherence_ok=True, perf_verdict="DEAD") == "DEAD"


def test_perf_inconclusive_passes_through():
    assert decide(parity_ok=True, coherence_ok=True, perf_verdict="INCONCLUSIVE") == "INCONCLUSIVE"


def test_bad_perf_verdict_raises():
    import pytest
    with pytest.raises(ValueError):
        decide(parity_ok=True, coherence_ok=True, perf_verdict="WHATEVER")


def test_only_win_is_bankable():
    assert is_bankable("WIN")
    for v in ("DEAD", "INCONCLUSIVE", "COHERENCE_FAIL", "PARITY_FAIL"):
        assert not is_bankable(v)


def test_make_row_carries_v2_fields():
    row = make_row("gfx1151", "attn", "drop_barrier", "WIN",
                   parity={"fp32_exact": True, "q8_tol": True},
                   perf_delta=1.3, perf_f=1.0,
                   coherence={"pass": True, "b": 0, "c": 0, "p": 1.0, "seeds": 12},
                   base_ref="5f101504", seeds=12)
    assert row["verdict"] == "WIN" and row["WIN"] is True
    assert row["parity"]["q8_tol"] is True
    assert row["base_ref"] == "5f101504"
    assert row["label"] == "drop_barrier"


# ---------------------------------------------------------------------------
# parity arm
# ---------------------------------------------------------------------------

def _pgen(pid, ids):
    return {"prompt_id": pid, "token_ids": ids, "text": ""}


def test_parity_token_id_exact():
    base = [_pgen("p1", [1, 2, 3]), _pgen("p2", [4, 5])]
    same = [_pgen("p1", [1, 2, 3]), _pgen("p2", [4, 5])]
    diff = [_pgen("p1", [1, 2, 3]), _pgen("p2", [4, 9])]
    assert parity_result(base, same)[0]
    assert not parity_result(base, diff)[0]


def test_parity_empty_baseline_not_a_pass():
    base = [_pgen("p1", [])]
    var = [_pgen("p1", [1, 2])]
    assert not parity_result(base, var)[0]


def test_parity_positional_when_no_prompt_id():
    # a runner may emit gens without prompt_id; fall back to positional pairing
    base = [{"token_ids": [1, 2, 3]}]
    same = [{"token_ids": [1, 2, 3]}]
    diff = [{"token_ids": [1, 9, 3]}]
    assert parity_result(base, same)[0]
    assert not parity_result(base, diff)[0]


# ---------------------------------------------------------------------------
# coherence arm (ported from test_coherence_arm.py)
# ---------------------------------------------------------------------------

def _gen(pid, seed=0, genre="code", text="fine", toks=None, finish="stop", empty=False):
    return {"prompt_id": pid, "seed": seed, "genre": genre, "text": text,
            "token_ids": toks if toks is not None else list(range(1000, 1000 + 60)),
            "finish": finish, "empty": empty, "tool_calls": []}


_ATTR = [7] * 60


def test_empty_is_attractor():
    assert ca.detect_attractor([]) == (True, "empty")


def test_clean_stream_passes():
    toks = list(range(200))
    is_a, reason = ca.detect_attractor(toks)
    assert not is_a and reason == "ok"


def test_short_answer_not_attractor():
    assert ca.detect_attractor(["42"]) == (False, "short-ok")
    assert ca.detect_attractor([5] * 10) == (False, "short-ok")
    assert ca.detect_attractor([5] * 40)[0]


def test_first128_single_token_loop():
    is_a, reason = ca.detect_attractor([7] * 130)
    assert is_a and reason == "first128"


def test_number_match():
    assert ca.validate_number_match("...therefore the total is 210 km.", 210)[0]
    assert not ca.validate_number_match("...the total is 205 km.", 210)[0]


def test_run_validators_routing():
    ok, fails = ca.run_validators("reason", "answer is 42", expect={"number": 42})
    assert ok and not fails
    ok, fails = ca.run_validators("factual", "One. Two.", expect={"sentences": 3})
    assert not ok and fails


def test_mcnemar_variant_strictly_worse_flags():
    pairs = [(False, True)] * 8 + [(False, False)] * 8
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert worse and b == 8 and c == 0 and p < 0.05


def test_mcnemar_variant_better_not_flagged():
    pairs = [(True, False)] * 8 + [(False, False)] * 8
    worse, p, b, c = ca.mcnemar_worse(pairs)
    assert not worse and c == 8 and b == 0


def test_coherence_result_variant_introduces_attractors():
    base = [_gen("prose", seed=s) for s in range(10)]
    var = [_gen("prose", seed=s, toks=_ATTR) for s in range(10)]
    ok, d = coherence_result(base, var)
    assert not ok and d["b"] == 10 and d["c"] == 0


def test_coherence_result_shared_attractor_not_worse():
    base = [_gen("prose", seed=s, toks=_ATTR) for s in range(8)]
    var = [_gen("prose", seed=s, toks=_ATTR) for s in range(8)]
    ok, d = coherence_result(base, var)
    assert ok and d["b"] == 0

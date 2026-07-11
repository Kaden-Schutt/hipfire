# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the certify orchestrator (Task 2.2) — mock ServeRunner.

Ports test_ab_certify_serve.py to the v2 package: the perf arm is now CONJUNCTIVE
(perf_measure returns (tok_list, dur_list)) and every row is SELF-DESCRIBING
(gpu_arch/model/base_sha/variant_sha/prompt_md5/kv/maxtok/measurement_hash +
tok_delta_pct/dur_delta_pct).
"""
import hashlib

from autoresearch.ar.certify.orchestrator import certify, ServeRunner


# ---- the plan's canonical mock + self-describing-row test ----

class MockRunner(ServeRunner):
    def __init__(self, **k):
        self.k = k

    def parity_gens(self, d):
        return [{"token_ids": [1, 2, 3], "text": "x"}]            # identical base==var

    def perf_measure(self, d):
        return (self.k["tok"][d], self.k["dur"][d])

    def coherence_gens(self, d, seeds):
        return [{"text": "ok fine", "token_ids": [5, 6, 7, 8]}]

    def clocks(self, d):
        return [3200]


_KW = dict(arch="gfx1201", kernel="k1", lever="L", base_daemon="base", var_daemon="var",
           base_ref="loop/gfx1201", model="qwen3.6-35b-a3b.mq4r", kv="q8", maxtok=128,
           prompt_md5="d97ec9d3")

_SELF_DESC = {"gpu_arch", "model", "base_sha", "variant_sha", "prompt_md5", "kv", "maxtok",
              "measurement_hash", "tok_delta_pct", "dur_delta_pct"}


def test_win_row_is_self_describing():
    r = MockRunner(tok={"base": [150] * 5, "var": [160] * 5},
                   dur={"base": [10.0] * 5, "var": [9.0] * 5})
    row = certify(r, **_KW)
    assert row["verdict"] == "WIN"
    assert _SELF_DESC <= set(row)


def test_measurement_hash_recipe():
    r = MockRunner(tok={"base": [150] * 5, "var": [160] * 5},
                   dur={"base": [10.0] * 5, "var": [9.0] * 5})
    row = certify(r, **_KW)
    expect = hashlib.sha256("|".join([
        row["gpu_arch"], row["model"], row["base_sha"], row["variant_sha"],
        row["prompt_md5"], row["kv"], str(row["maxtok"])]).encode()).hexdigest()[:16]
    assert row["measurement_hash"] == expect
    assert len(row["measurement_hash"]) == 16


def test_win_records_both_perf_stats():
    r = MockRunner(tok={"base": [150] * 5, "var": [160] * 5},
                   dur={"base": [10.0] * 5, "var": [9.0] * 5})
    row = certify(r, **_KW)
    assert row["tok_delta_pct"] > 0 and row["dur_delta_pct"] < 0


# ---- conjunctive gate through the orchestrator ----

def test_tok_up_dur_flat_is_dead_not_win():
    r = MockRunner(tok={"base": [150] * 5, "var": [160] * 5},
                   dur={"base": [10.0] * 5, "var": [10.0] * 5})   # duration flat
    assert certify(r, **_KW)["verdict"] == "DEAD"


def test_dur_down_tok_flat_is_dead_not_win():
    r = MockRunner(tok={"base": [150] * 5, "var": [150] * 5},     # tok flat
                   dur={"base": [10.0] * 5, "var": [9.0] * 5})
    assert certify(r, **_KW)["verdict"] == "DEAD"


# ---- parity short-circuit ----

class ParityMock(ServeRunner):
    def __init__(self, base_ids, var_ids):
        self.base_ids, self.var_ids = base_ids, var_ids

    def parity_gens(self, d):
        return [{"prompt_id": "p1", "token_ids": self.base_ids if d == "base" else self.var_ids}]

    def perf_measure(self, d):
        raise AssertionError("perf must not run after a parity fail")

    def coherence_gens(self, d, seeds):
        raise AssertionError("coherence must not run after a parity fail")

    def clocks(self, d):
        return []


def test_parity_short_circuits_value_change():
    r = ParityMock([1, 2, 3], [1, 9, 3])                          # token flipped
    row = certify(r, **{**_KW, "base_daemon": "base", "var_daemon": "var"})
    assert row["verdict"] == "PARITY_FAIL"
    assert _SELF_DESC <= set(row)                                 # still self-describing


# ---- coherence hard-gate beats a perf win ----

_ATTR = [7] * 60


class CoherenceMock(ServeRunner):
    def __init__(self, base_coh, var_coh):
        self.base_coh, self.var_coh = base_coh, var_coh

    def parity_gens(self, d):
        return [{"prompt_id": "p1", "token_ids": [1, 2, 3]}]      # base==var -> parity pass

    def perf_measure(self, d):
        return ([150] * 5, [10.0] * 5) if d == "base" else ([160] * 5, [8.0] * 5)  # var faster

    def coherence_gens(self, d, seeds):
        gens = self.base_coh if d == "base" else self.var_coh
        return [dict(g, seed=s) for s in seeds for g in gens]

    def clocks(self, d):
        return []


def _cg(genre="prose", text="fine", toks=None):
    return {"prompt_id": genre, "genre": genre, "text": text,
            "token_ids": toks if toks is not None else list(range(1000, 1060)), "tool_calls": []}


def test_coherence_fail_beats_perf_win():
    # variant is FASTER on both axes but attractors on every seed -> COHERENCE_FAIL, not WIN
    r = CoherenceMock([_cg()], [_cg(toks=_ATTR)])
    row = certify(r, **_KW)
    assert row["verdict"] == "COHERENCE_FAIL"


def test_full_win_all_arms_pass():
    r = CoherenceMock([_cg()], [_cg()])                           # both coherent
    row = certify(r, **_KW)
    assert row["verdict"] == "WIN"


def test_certify_threads_expects_catches_fluent_but_wrong():
    # parity-clean, attractor-free, FASTER — but answers WRONG. Rejected only when expects threaded.
    r = CoherenceMock([_cg(genre="reason", text="the answer is 42")],
                      [_cg(genre="reason", text="the answer is 41")])
    kw = dict(_KW)
    assert certify(r, expects={"reason": {"number": 42}}, **kw)["verdict"] == "COHERENCE_FAIL"
    assert certify(r, expects=None, **kw)["verdict"] == "WIN"

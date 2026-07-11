# Copyright (c) Kaden Schutt
from autoresearch.ar.certify.orchestrator import ServeRunner
from autoresearch.ar.gate.engine import gate_cell

_CELL = dict(arch="gfx1201", model="qwen3.6-a3b", base_ref="master",
             kv="q8", maxtok=128, prompt_md5="abc123")


def _cg(genre="prose", text="fine", toks=None):
    return {"prompt_id": genre, "genre": genre, "text": text,
            "token_ids": toks if toks is not None else list(range(1000, 1060)),
            "tool_calls": []}


class Runner(ServeRunner):
    """Configurable mock: parity ids, perf sample maps, coherence gens per daemon."""

    def __init__(self, *, parity=None, tok=None, dur=None, coh=None):
        self._parity = parity or {"base": [1, 2, 3], "var": [1, 2, 3]}
        self._tok = tok or {"base": [150] * 8, "var": [150] * 8}
        self._dur = dur or {"base": [10.0] * 8, "var": [10.0] * 8}
        self._coh = coh or {"base": [_cg()], "var": [_cg()]}

    def parity_gens(self, d):
        return [{"prompt_id": "p1", "token_ids": self._parity[d]}]

    def perf_measure(self, d):
        return (self._tok[d], self._dur[d])

    def coherence_gens(self, d, seeds):
        return [dict(g, seed=s) for s in seeds for g in self._coh[d]]

    def clocks(self, d):
        return []


def test_neutral_passes_and_is_self_describing():
    r = Runner()  # identical everything
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "PASS"
    assert row["perf_class"] == "NEUTRAL"
    assert row["measurement_hash"] and len(row["measurement_hash"]) == 16
    assert row["gpu_arch"] == "gfx1201" and row["model"] == "qwen3.6-a3b"


def test_parity_fail_rejects_and_short_circuits_perf():
    class NoPerf(Runner):
        def perf_measure(self, d):
            raise AssertionError("perf must not run after parity fail")

    r = NoPerf(parity={"base": [1, 2, 3], "var": [1, 9, 3]})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "parity"


def test_significant_regression_rejects():
    r = Runner(tok={"base": [150] * 8, "var": [140] * 8},
               dur={"base": [10.0] * 8, "var": [10.8] * 8})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "perf_regression"


def test_coherence_runs_even_when_perf_neutral():
    # perf identical (neutral) but variant attractors -> must still REJECT on coherence
    r = Runner(coh={"base": [_cg()], "var": [_cg(toks=[7] * 60)]})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "REJECT" and row["reason"] == "coherence"


def test_improvement_passes():
    r = Runner(tok={"base": [150] * 8, "var": [162] * 8},
               dur={"base": [10.0] * 8, "var": [9.2] * 8})
    row = gate_cell(r, base_daemon="base", var_daemon="var", **_CELL)
    assert row["gate_verdict"] == "PASS" and row["reason"] == "improvement"

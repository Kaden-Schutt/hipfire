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


from autoresearch.ar.gate.config import GateConfig
from autoresearch.ar.gate.engine import run_gate

_CFG = GateConfig(
    archs=["gfx1100", "gfx1151", "gfx1201"],
    canonical_models=["qwen3.6-27b", "qwen3.6-a3b"],
    fit={"qwen3.6-27b": ["gfx1100", "gfx1151", "gfx1201"],
         "qwen3.6-a3b": ["gfx1100", "gfx1151", "gfx1201"],
         "deepseek4": ["gfx1151"]},
    maintainers=["Kaden-Schutt"],
)


def _no_leak(*a, **k):
    return []


def _factory(**per_model):
    """runner_factory returning a fresh Runner per model (per_model overrides)."""
    def make(model):
        return per_model.get(model, Runner())
    return make


def test_run_gate_all_neutral_passes():
    res = run_gate(arch="gfx1201", changed_kernel_files=[],
                   models=["qwen3.6-27b", "qwen3.6-a3b"], base_ref="master",
                   head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=_factory(), cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS"
    assert len(res["cells"]) == 2


def test_run_gate_cross_arch_leak_rejects():
    def leak(kernel_file, arch, other_archs, repo, base_sha=None, preprocess=None):
        return ["gfx1100"]                       # this file perturbs gfx1100 codegen
    res = run_gate(arch="gfx1201", changed_kernel_files=["kernels/src/x.hip"],
                   models=["qwen3.6-a3b"], base_ref="master", head_ref="pr",
                   repo="/repo", cfg=_CFG, runner_factory=_factory(), cross_arch_fn=leak)
    assert res["verdict"] == "REJECT"
    assert res["cross_arch_leaks"] == [{"file": "kernels/src/x.hip", "leaks": ["gfx1100"]}]


def test_run_gate_no_fitting_model_is_pass_na():
    res = run_gate(arch="gfx1100", changed_kernel_files=[], models=[],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=_factory(), cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS" and "no-fitting-model" in res["reasons"]


def test_run_gate_confirm_rerun_flips_noise_to_pass():
    # first runner regresses, the rerun (fresh from factory) is neutral -> PASS
    calls = {"n": 0}
    regress = Runner(tok={"base": [150] * 8, "var": [140] * 8},
                     dur={"base": [10.0] * 8, "var": [10.8] * 8})

    def make(model):
        calls["n"] += 1
        return regress if calls["n"] == 1 else Runner()   # 2nd call = neutral
    res = run_gate(arch="gfx1201", changed_kernel_files=[], models=["qwen3.6-a3b"],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=make, cross_arch_fn=_no_leak)
    assert res["verdict"] == "PASS" and calls["n"] == 2      # reran once


def test_run_gate_replicated_regression_rejects():
    def make(model):
        return Runner(tok={"base": [150] * 8, "var": [140] * 8},
                      dur={"base": [10.0] * 8, "var": [10.8] * 8})
    res = run_gate(arch="gfx1201", changed_kernel_files=[], models=["qwen3.6-a3b"],
                   base_ref="master", head_ref="pr", repo="/repo", cfg=_CFG,
                   runner_factory=make, cross_arch_fn=_no_leak)
    assert res["verdict"] == "REJECT" and "perf_regression" in res["reasons"]

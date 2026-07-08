#!/usr/bin/env python3
# Copyright (c) Kaden Schutt
"""No-GPU unit tests for the v2 certify orchestrator (mock ServeRunner)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import ab_certify_serve as ab


def _gen(pid, seed=0, genre="code", text="fine", toks=None, finish="stop", empty=False):
    return {"prompt_id": pid, "seed": seed, "genre": genre, "text": text,
            "token_ids": toks if toks is not None else list(range(1000, 1000 + 60)),
            "finish": finish, "empty": empty, "tool_calls": []}

_ATTR = [7] * 60  # single-token loop -> detect_attractor -> first128


class MockRunner(ab.ServeRunner):
    def __init__(self, parity, coherence, durations, clocks=None):
        self.p, self.c, self.d, self.k = parity, coherence, durations, clocks or {}
    def parity_gens(self, daemon):        return self.p[daemon]
    def coherence_gens(self, daemon, seeds): return self.c[daemon]
    def perf_durations(self, daemon):     return self.d[daemon]
    def clocks(self, daemon):             return self.k.get(daemon, [])


# ---- arm units ----

def test_parity_stable_prefix():
    # baseline self-reproducible (base_a == base_b) -> stable prefix is the whole text
    base = [_gen("code", text="X"), _gen("reason", text="Y")]
    same = [_gen("code", text="X"), _gen("reason", text="Y")]
    diff = [_gen("code", text="X"), _gen("reason", text="Y2")]
    assert ab.parity_result(base, base, same)[0]         # variant matches -> PASS
    assert not ab.parity_result(base, base, diff)[0]     # variant diverges in stable region -> FAIL

def test_parity_tolerates_late_q8_noise():
    # baseline diverges from ITSELF after "answer: 42" (Q8 noise in the long tail) -> stable prefix
    # is only the reproducible head. A variant that agrees on that head PASSES despite a different tail.
    base_a = [_gen("reason", text="answer: 42 because six sevens")]
    base_b = [_gen("reason", text="answer: 42 since 6 times 7")]      # diverges after "answer: 42 "
    var_ok = [_gen("reason", text="answer: 42 blah blah different tail")]
    var_bad = [_gen("reason", text="answer: 41 because six sevens")]  # diverges INSIDE the stable head
    assert ab.parity_result(base_a, base_b, var_ok)[0]
    assert not ab.parity_result(base_a, base_b, var_bad)[0]

def test_parity_unstable_baseline_not_a_pass():
    # baseline can't reproduce even the first char -> no signal -> not a silent PASS
    base_a = [_gen("reason", text="AXY")]
    base_b = [_gen("reason", text="BXY")]
    var = [_gen("reason", text="AXY")]
    assert not ab.parity_result(base_a, base_b, var)[0]

def test_gen_fails_detects():
    assert ab._gen_fails(_gen("p", toks=_ATTR), {})[0]          # attractor
    assert ab._gen_fails(_gen("p", finish="length"), {})[0]     # runaway
    assert ab._gen_fails(_gen("p", empty=True), {})[0]          # empty
    assert not ab._gen_fails(_gen("p"), {})[0]                  # clean

def test_coherence_baseline_shared_attractor_not_worse():
    # baseline ALSO attractors on this prompt -> variant sharing it is NOT a regression
    base = [_gen("prose", seed=s, toks=_ATTR) for s in range(8)]
    var = [_gen("prose", seed=s, toks=_ATTR) for s in range(8)]
    ok, d = ab.coherence_result(base, var)
    assert ok and d["b"] == 0

def test_coherence_variant_introduces_attractors():
    base = [_gen("prose", seed=s) for s in range(10)]                 # baseline clean
    var = [_gen("prose", seed=s, toks=_ATTR) for s in range(10)]      # variant attractors every seed
    ok, d = ab.coherence_result(base, var)
    assert not ok and d["b"] == 10 and d["c"] == 0

def test_coherence_variant_validator_fail():
    base = [_gen("reason", seed=s, genre="reason", text="answer 42") for s in range(8)]
    var = [_gen("reason", seed=s, genre="reason", text="answer 41") for s in range(8)]
    expects = {"reason": {"number": 42}}
    ok, d = ab.coherence_result(base, var, expects=expects)
    assert not ok  # variant gives wrong number on every seed (b=8, c=0)

def test_perf_faster_wins():
    v, f, delta = ab.perf_result([10.0] * 8, [9.0] * 8)   # 10ms -> 9ms = faster
    assert v == "WIN" and delta < 0                        # duration went down
    v2, _, d2 = ab.perf_result([10.0] * 8, [11.0] * 8)     # slower
    assert v2 == "DEAD" and d2 > 0


# ---- full orchestration ----

def _seeds(n=8): return list(range(n))

def test_certify_full_win():
    r = MockRunner(
        parity={"base": [_gen("c", text="X")], "var": [_gen("c", text="X")]},          # byte-exact
        coherence={"base": [_gen("c", seed=s) for s in range(8)],
                   "var": [_gen("c", seed=s) for s in range(8)]},                       # both clean
        durations={"base": [10.0] * 8, "var": [9.0] * 8})                               # variant faster
    row = ab.certify(r, arch="gfx1151", kernel="attn", lever="drop_barrier",
                     base_daemon="base", var_daemon="var", base_ref="5f101504", seeds=_seeds())
    assert row["verdict"] == "WIN" and row["perf_delta"] < 0 and row["base_ref"] == "5f101504"

def test_certify_parity_short_circuits():
    r = MockRunner(
        parity={"base": [_gen("c", text="X")], "var": [_gen("c", text="DIFFERENT")]},   # value-changed
        coherence={"base": [], "var": []}, durations={"base": [], "var": []})
    row = ab.certify(r, arch="gfx1151", kernel="attn", lever="bad", base_daemon="base",
                     var_daemon="var", base_ref="s", seeds=_seeds())
    assert row["verdict"] == "PARITY_FAIL"   # never touched coherence/perf (empty lists would crash if it had)

def test_certify_coherence_beats_perf_win():
    # variant is FASTER but breaks coherence -> COHERENCE_FAIL, not WIN
    r = MockRunner(
        parity={"base": [_gen("c", text="X")], "var": [_gen("c", text="X")]},
        coherence={"base": [_gen("prose", seed=s) for s in range(10)],
                   "var": [_gen("prose", seed=s, toks=_ATTR) for s in range(10)]},      # variant attractors
        durations={"base": [10.0] * 8, "var": [8.0] * 8})                               # and is faster
    row = ab.certify(r, arch="gfx1151", kernel="attn", lever="fast_but_broken",
                     base_daemon="base", var_daemon="var", base_ref="s", seeds=_seeds(10))
    assert row["verdict"] == "COHERENCE_FAIL"


def test_load_expects_from_prompts_file():
    import json, tempfile
    rows = [{"genre": "reason", "prompt": "6*7?", "expect": {"number": 42}},
            {"genre": "code", "prompt": "fib"},                       # no expect -> not in map
            {"genre": "factual", "prompt": "cap?", "expect": {"sentences": 1}}]
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(rows, f); f.close()
    assert ab.load_expects(f.name) == {"reason": {"number": 42}, "factual": {"sentences": 1}}
    assert ab.load_expects(None) == {}


def test_certify_threads_expects_catches_fluent_but_wrong():
    # A variant that is parity-clean, attractor-free, AND faster — but ANSWERS WRONG. It must be
    # rejected COHERENCE_FAIL once expects are threaded (#2), and (contrast) would WIN without them.
    base_par = [_gen("reason", genre="reason", text="42")]
    var_par = [_gen("reason", genre="reason", text="42")]                       # byte-exact -> parity PASS
    base_coh = [_gen("reason", seed=s, genre="reason", text="the answer is 42") for s in range(8)]
    var_coh = [_gen("reason", seed=s, genre="reason", text="the answer is 41") for s in range(8)]  # WRONG
    r = MockRunner(parity={"B": base_par, "V": var_par},
                   coherence={"B": base_coh, "V": var_coh},
                   durations={"B": [10.0] * 8, "V": [5.0] * 8})                 # variant is FASTER
    kw = dict(arch="gfx1151", kernel="k", lever="wrong_answer",
              base_daemon="B", var_daemon="V", base_ref="r", seeds=_seeds(8))
    assert ab.certify(r, expects={"reason": {"number": 42}}, **kw)["verdict"] == "COHERENCE_FAIL"
    assert ab.certify(r, expects=None, **kw)["verdict"] == "WIN"                # the hole #2 closes


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

# Copyright (c) Kaden Schutt
"""No-GPU unit tests for ar.driver — the self-exhausting per-worker loop.

Ports ``harness/v2/driver_v3.sh``. The loop's decision logic is exercised with
NO agent / GPU / git: every side-effecting seam (``is_exhausted``,
``gen_digest``, ``run_round``, ``update_exhaustion``, ``advance``) is injected
through the ``hooks`` dict, so this suite verifies termination + per-round
sequencing purely in-process.
"""
from autoresearch.ar.config import Bounds, LoopConfig, WorkerCfg
from autoresearch.ar.driver import run_loop


def _cfg(workers=None):
    return LoopConfig(
        "gfx1201",
        "loop/gfx1201",
        "m",
        "q8",
        128,
        "md5",
        3.0,
        5,
        "codex",
        workers or [WorkerCfg(1, 1, "gpt-5.6-luna", "max")],
        Bounds(400, 43200),
    )


# ── plan Task 4.2 Step 1 (the mandated contract) ──────────────────────────────
def test_loop_terminates_on_global_exhaustion():
    cfg = _cfg()
    calls = {"n": 0}
    hooks = {
        "is_exhausted": lambda *a, **k: calls["n"] >= 2,  # exhausted after 2 rounds
        "run_round": lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or 0,
        "update_exhaustion": lambda *a, **k: None,
        "advance": lambda *a, **k: True,
    }
    rounds = run_loop(cfg, cfg.workers[0], safety_cap=100, hooks=hooks)
    assert rounds == 2  # stops when is_exhausted flips true


# ── SAFETY_CAP backstops an eternally-open candidate set ───────────────────────
def test_safety_cap_backstops_when_never_exhausted():
    cfg = _cfg()
    n = {"rounds": 0}
    hooks = {
        "is_exhausted": lambda *a, **k: False,  # never exhausts
        "run_round": lambda *a, **k: n.__setitem__("rounds", n["rounds"] + 1) or 0,
        "update_exhaustion": lambda *a, **k: None,
        "advance": lambda *a, **k: True,
    }
    rounds = run_loop(cfg, cfg.workers[0], safety_cap=3, hooks=hooks)
    assert rounds == 3 and n["rounds"] == 3


# ── already-exhausted at entry runs zero rounds (never dispatches an agent) ─────
def test_already_exhausted_runs_zero_rounds():
    cfg = _cfg()

    def _must_not_run(*a, **k):
        raise AssertionError("run_round must not fire when already exhausted")

    hooks = {
        "is_exhausted": lambda *a, **k: True,
        "run_round": _must_not_run,
        "update_exhaustion": lambda *a, **k: None,
        "advance": lambda *a, **k: True,
    }
    assert run_loop(cfg, cfg.workers[0], safety_cap=100, hooks=hooks) == 0


# ── per-round sequencing: digest built, injected into the prompt, then
#    update_exhaustion + advance fire once each ────────────────────────────────
def test_digest_injected_and_update_advance_called_each_round():
    cfg = _cfg()
    log = {"digest": 0, "update": 0, "advance": 0, "prompts": []}
    state = {"n": 0}

    def _run_round(*a, **k):
        state["n"] += 1
        log["prompts"].append(k.get("prompt", ""))
        return 0

    hooks = {
        "is_exhausted": lambda *a, **k: state["n"] >= 2,
        "gen_digest": lambda *a, **k: log.__setitem__("digest", log["digest"] + 1) or "DIGEST-XYZ",
        "run_round": _run_round,
        "update_exhaustion": lambda *a, **k: log.__setitem__("update", log["update"] + 1),
        "advance": lambda *a, **k: log.__setitem__("advance", log["advance"] + 1) or True,
    }
    rounds = run_loop(cfg, cfg.workers[0], safety_cap=100, hooks=hooks)
    assert rounds == 2
    assert log["digest"] == 2 and log["update"] == 2 and log["advance"] == 2
    # the digest the driver built is carried into every round prompt
    assert all("DIGEST-XYZ" in p for p in log["prompts"])
    # and the round prompt is self-describing (round number + loop identity)
    assert "ROUND 1" in log["prompts"][0] and "ROUND 2" in log["prompts"][1]


# ── run_round receives this worker's model + effort (per-worker heterogeneity) ─
def test_run_round_gets_worker_model_and_effort():
    cfg = _cfg([WorkerCfg(2, 2, "gpt-5.6-terra", "xhigh")])
    seen = {}
    state = {"n": 0}

    def _run_round(*a, **k):
        state["n"] += 1
        seen.update(k)
        return 0

    hooks = {
        "is_exhausted": lambda *a, **k: state["n"] >= 1,
        "gen_digest": lambda *a, **k: "",
        "run_round": _run_round,
        "update_exhaustion": lambda *a, **k: None,
        "advance": lambda *a, **k: True,
    }
    run_loop(cfg, cfg.workers[0], safety_cap=100, hooks=hooks)
    assert seen.get("model") == "gpt-5.6-terra"
    assert seen.get("effort") == "xhigh"
    assert seen.get("harness") == "codex"

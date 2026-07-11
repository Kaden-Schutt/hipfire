# Copyright (c) Kaden Schutt
import json
import os

from autoresearch.ar.gate.dispatch import (
    aggregate,
    floor_risk,
    parse_plan,
    run_behavior_test,
    run_behavior_tests,
)


# ---- floor_risk: escalate-only ----

def test_floor_escalates_but_never_deescalates():
    # Claude says trivial but the paths floor to high-risk -> high-risk wins.
    assert floor_risk("trivial", "high-risk") == "high-risk"
    # Claude escalates above the floor -> Claude wins.
    assert floor_risk("high-risk", "trivial") == "high-risk"
    # equal
    assert floor_risk("moderate", "moderate") == "moderate"
    # unknown floor -> treated as max (safe)
    assert floor_risk("low", "bogus") == "high-risk"


# ---- parse_plan: apply the classify_pr floor ----

def test_parse_plan_floors_a_kernel_pr_even_if_claude_says_trivial():
    plan = {"risk": "trivial", "serve_floor": {"perf_ab": True},
            "behavior_tests": [{"what": "x", "prompt": "p"}], "reason": "r"}
    out = parse_plan(plan, ["kernels/src/gemv.hip"], lines_changed=10)
    assert out["floor_risk"] == "high-risk"
    assert out["risk"] == "high-risk"          # floored up
    assert out["behavior_tests"][0]["what"] == "x"
    assert out["serve_floor"]["perf_ab"] is True


def test_parse_plan_accepts_json_string_and_keeps_claude_escalation():
    plan = json.dumps({"risk": "moderate", "behavior_tests": []})
    out = parse_plan(plan, ["docs/x.md"])   # floor = trivial
    assert out["risk"] == "moderate"          # Claude's higher read kept


# ---- run_behavior_test: codex verdict via injected seam ----

def _writer(passed, detail="ok"):
    """A mock agent_exec_fn that writes a codex-style verdict file, returns rc 0.
    The verdict_path is embedded in the prompt; parse it out and honor it."""
    def fn(*, harness, model, effort, prompt, cwd):
        # extract the verdict path the runner asked codex to write to
        path = prompt.split("write your verdict as JSON to ")[1].split(":")[0].strip()
        with open(path, "w") as fh:
            json.dump({"passed": passed, "detail": detail}, fh)
        return 0
    return fn


def test_behavior_test_pass(tmp_path):
    bt = {"what": "cli --foo works", "prompt": "run hipfire --foo", "harness": "codex",
          "model": "gpt-5.6-sol", "effort": "xhigh"}
    r = run_behavior_test(bt, agent_exec_fn=_writer(True, "printed X"), cwd="/repo",
                          verdict_path=str(tmp_path / "v.json"))
    assert r["passed"] is True and r["what"] == "cli --foo works" and r["detail"] == "printed X"
    assert r["model"] == "gpt-5.6-sol"


def test_behavior_test_fail_reports_detail(tmp_path):
    bt = {"what": "cli --foo works", "prompt": "run it"}
    r = run_behavior_test(bt, agent_exec_fn=_writer(False, "crashed"), cwd="/r",
                          verdict_path=str(tmp_path / "v.json"))
    assert r["passed"] is False and r["detail"] == "crashed"


def test_behavior_test_missing_verdict_is_fail_not_silent_pass(tmp_path):
    # codex ran (rc 0) but wrote NO verdict -> must FAIL, never silently pass.
    def no_verdict(*, harness, model, effort, prompt, cwd):
        return 0
    r = run_behavior_test({"what": "x", "prompt": "p"}, agent_exec_fn=no_verdict, cwd="/r",
                          verdict_path=str(tmp_path / "missing.json"))
    assert r["passed"] is False and "verdict unreadable" in r["detail"]


def test_run_behavior_tests_all(tmp_path):
    tests = [{"what": "a", "prompt": "pa"}, {"what": "b", "prompt": "pb"}]
    res = run_behavior_tests(tests, agent_exec_fn=_writer(True), cwd="/r",
                             verdict_dir=str(tmp_path / "vd"))
    assert [r["what"] for r in res] == ["a", "b"] and all(r["passed"] for r in res)


# ---- aggregate: floor AND all behavior tests ----

def test_aggregate_pass_requires_floor_and_all_behaviors():
    floor = {"verdict": "PASS", "reasons": []}
    behs = [{"what": "a", "passed": True}, {"what": "b", "passed": True}]
    assert aggregate(floor, behs)["verdict"] == "PASS"


def test_aggregate_floor_fail_rejects_even_if_behaviors_pass():
    floor = {"verdict": "REJECT", "reasons": ["perf_regression"]}
    behs = [{"what": "a", "passed": True}]
    out = aggregate(floor, behs)
    assert out["verdict"] == "REJECT" and "perf_regression" in out["reasons"]


def test_aggregate_behavior_fail_rejects_even_if_floor_green():
    floor = {"verdict": "PASS", "reasons": []}
    behs = [{"what": "cli --foo", "passed": False}, {"what": "b", "passed": True}]
    out = aggregate(floor, behs)
    assert out["verdict"] == "REJECT" and "behavior:cli --foo" in out["reasons"]

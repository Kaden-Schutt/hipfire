# Copyright (c) Kaden Schutt
"""ar.gate.dispatch — the codex/dispatch layer (spec §8).

Claude reads the diff and emits a plan: a risk class, a serve_harness FLOOR spec,
and a list of bespoke behavior tests (each a prompt + which codex tier runs it).
This module is the deterministic glue around that plan:

  * parse_plan   — parse Claude's plan and apply the classify_pr FLOOR to its risk
                   (escalate-only: Claude can go higher than the path floor, never
                   lower — a kernel PR can't be de-classified to trivial).
  * run_behavior_test(s) — execute each bespoke test via the injected codex seam
                   (agent_exec), reading a structured verdict codex writes. This is
                   the "codex tests beyond serve_harness" piece — codex runs the
                   test GENERALLY on-box (build / run the new path / verify), not
                   bound to serve_harness.
  * aggregate    — combine the deterministic serve_harness floor verdict with the
                   agentic behavior-test results: PASS iff floor PASS AND every
                   behavior test passed.

Everything here is pure/no-GPU with an injected agent_exec_fn; the only real GPU/
codex touch is agent_exec itself in production.
"""
from __future__ import annotations

import json
import os

from .routing import classify_pr

# Ascending risk; the executor tier grows with it. Unknown -> treated as the
# extreme (claude unknown -> trivial index; floor unknown -> max, the safe side).
RISK_ORDER = ("trivial", "low", "moderate", "high-risk")

__all__ = ["floor_risk", "parse_plan", "run_behavior_test", "run_behavior_tests", "aggregate"]


def floor_risk(claude_risk: str, floor: str) -> str:
    """Escalate-only: the HIGHER of Claude's semantic read and the classify_pr floor."""
    ci = RISK_ORDER.index(claude_risk) if claude_risk in RISK_ORDER else 0
    fi = RISK_ORDER.index(floor) if floor in RISK_ORDER else len(RISK_ORDER) - 1
    return RISK_ORDER[max(ci, fi)]


def parse_plan(plan, changed_files, lines_changed=None) -> dict:
    """Parse Claude's dispatch plan (dict or JSON string), floor its risk with
    classify_pr, and normalize. Returns {risk, floor_risk, serve_floor, behavior_tests, reason}."""
    if isinstance(plan, str):
        plan = json.loads(plan)
    floor = classify_pr(changed_files, lines_changed=lines_changed)
    return {
        "risk": floor_risk(plan.get("risk", "trivial"), floor),
        "floor_risk": floor,
        "serve_floor": dict(plan.get("serve_floor", {}) or {}),
        "behavior_tests": list(plan.get("behavior_tests", []) or []),
        "reason": str(plan.get("reason", "")),
    }


def run_behavior_test(bt, *, agent_exec_fn, cwd, verdict_path) -> dict:
    """Run ONE bespoke behavior test via codex. ``agent_exec_fn(harness, model, effort,
    prompt, cwd) -> int`` is the injected seam (agent_exec.run_round in prod). codex is
    told to write a structured verdict to ``verdict_path``; a missing/unreadable verdict
    is a FAIL (never a silent pass). Returns {what, passed, harness, model, detail}."""
    prompt = (
        (bt.get("prompt") or "")
        + f"\n\nExpected: {bt.get('expect', 'the behavior works correctly')}."
        + f"\nBuild/run whatever is needed to verify this on-box (you are NOT limited to"
        + f" serve_harness). When done, write your verdict as JSON to {verdict_path}:"
        + ' {"passed": <true|false>, "detail": "<one-line reason>"}.'
    )
    rc = agent_exec_fn(
        harness=bt.get("harness", "codex"), model=bt.get("model"),
        effort=bt.get("effort", "high"), prompt=prompt, cwd=cwd,
    )
    passed, detail = False, f"no verdict written (exec rc={rc})"
    try:
        with open(verdict_path) as fh:
            v = json.load(fh)
        passed = bool(v.get("passed"))
        detail = str(v.get("detail", ""))
    except Exception as e:  # missing / malformed verdict -> FAIL, do not silently pass
        detail = f"verdict unreadable ({e}); exec rc={rc}"
    return {"what": bt.get("what", "behavior"), "passed": passed,
            "harness": bt.get("harness", "codex"), "model": bt.get("model"), "detail": detail}


def run_behavior_tests(behavior_tests, *, agent_exec_fn, cwd, verdict_dir) -> list[dict]:
    """Run every bespoke behavior test, each with its own verdict file."""
    os.makedirs(verdict_dir, exist_ok=True)
    out = []
    for i, bt in enumerate(behavior_tests):
        vp = os.path.join(verdict_dir, f"behavior_{i}.json")
        out.append(run_behavior_test(bt, agent_exec_fn=agent_exec_fn, cwd=cwd, verdict_path=vp))
    return out


def aggregate(floor_verdict, behavior_results) -> dict:
    """Combine the deterministic serve_harness FLOOR (a run_gate-style {verdict, reasons})
    with the agentic behavior-test results. PASS iff floor PASS AND every behavior test
    passed; otherwise REJECT, itemizing the failed floor reasons + failed behaviors."""
    floor_ok = floor_verdict.get("verdict") == "PASS"
    failed = [b for b in behavior_results if not b["passed"]]
    reasons = list(floor_verdict.get("reasons", []))
    if not floor_ok and not reasons:
        reasons.append(str(floor_verdict.get("verdict", "floor_fail")).lower())
    reasons += [f"behavior:{b['what']}" for b in failed]
    return {
        "verdict": "PASS" if (floor_ok and not failed) else "REJECT",
        "floor": floor_verdict,
        "behavior_results": behavior_results,
        "reasons": reasons,
    }

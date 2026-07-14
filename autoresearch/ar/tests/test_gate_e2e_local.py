# Copyright (c) Kaden Schutt
"""Local end-to-end simulations of Claude dispatch -> matrix -> outcome -> action."""
import json

from autoresearch.ar.gate.config import load_gate_config
from autoresearch.ar.gate.dispatch import validate_dispatch_plan, validate_interpret_action
from autoresearch.ar.gate.run import interpret_results


def _git(repo, *args):
    if "--name-only" in args:
        return 0, "kernels/src/shared.hip\n"
    if "--numstat" in args:
        return 0, "12\t3\tkernels/src/shared.hip\n"
    return 0, ""


def _shared_plan():
    base = {"behavior_tests": [], "models": ["qwen3.6-27b"]}
    return {
        "schema": 1,
        "decision": "run",
        "risk": "high-risk",
        "reason": "shared kernel affects all production archs",
        "boxes": {
            "hipx": {**base, "run": True, "archs": ["gfx1100", "gfx1151"]},
            "hiptrx": {**base, "run": True, "archs": ["gfx1201"]},
        },
    }


def _write_result(root, arch, verdict="PASS", reason=None):
    (root / f"{arch}.json").write_text(json.dumps({
        "arch": arch,
        "verdict": verdict,
        "reasons": [reason] if reason else [],
        "bod": None,
        "rows": [],
        "tok_delta_pct": 0.0,
    }))


def test_local_e2e_clear_pass(pr_gate_toml, tmp_path):
    cfg = load_gate_config(pr_gate_toml)
    dispatch = validate_dispatch_plan(
        _shared_plan(), ["kernels/src/shared.hip"], cfg, lines_changed=15)
    assert dispatch["valid"] is True
    assert [r["box"] for r in dispatch["matrix"]["include"]] == ["hipx", "hiptrx"]

    results = tmp_path / "pass-results"
    results.mkdir()
    for arch in cfg.archs:
        _write_result(results, arch)
    interp = interpret_results(
        results_dir=str(results), base="base", head="head", repo="/repo",
        author="Kaden-Schutt", is_draft=False, helpful=True, cfg=cfg,
        run_git=_git, behavior_results=[],
    )
    assert interp["outcome"]["status"] == "success"
    assert interp["outcome"]["action"] == "auto_merge"
    action = validate_interpret_action(
        {"outcome": interp["outcome"]},
        {"action": "auto_merge", "comment_markdown": "### GPU PR Gate\nPASS"},
    )
    assert action["valid"] is True


def test_local_e2e_clear_reject_and_no_green_reinterpretation(pr_gate_toml, tmp_path):
    cfg = load_gate_config(pr_gate_toml)
    dispatch = validate_dispatch_plan(
        _shared_plan(), ["kernels/src/shared.hip"], cfg, lines_changed=15)
    assert dispatch["valid"] is True

    results = tmp_path / "reject-results"
    results.mkdir()
    _write_result(results, "gfx1100")
    _write_result(results, "gfx1151", "REJECT", "parity_fail")
    _write_result(results, "gfx1201")
    interp = interpret_results(
        results_dir=str(results), base="base", head="head", repo="/repo",
        author="Kaden-Schutt", is_draft=False, helpful=True, cfg=cfg,
        run_git=_git, behavior_results=[],
    )
    assert interp["outcome"]["status"] == "failure"
    assert interp["outcome"]["action"] == "bod"
    green = validate_interpret_action(
        {"outcome": interp["outcome"]},
        {"action": "auto_merge", "comment_markdown": "green"},
    )
    assert green["valid"] is False
    bod = validate_interpret_action(
        {"outcome": interp["outcome"]},
        {"action": "bod", "comment_markdown": "### GPU PR Gate\nREJECT parity_fail"},
    )
    assert bod["valid"] is True

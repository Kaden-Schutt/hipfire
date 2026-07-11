# Copyright (c) Kaden Schutt
import io
import json
from contextlib import redirect_stdout

from autoresearch.ar.cli import main


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_gate_plan_lists_fitting_models_and_other_archs(pr_gate_toml):
    rc, out = _run(["gate", "--arch", "gfx1201", "--plan", "--gate-config", pr_gate_toml])
    assert rc == 0
    d = json.loads(out)
    assert d["arch"] == "gfx1201"
    assert d["models"] == ["qwen3.6-27b", "qwen3.6-a3b"]
    assert d["other_archs"] == ["gfx1100", "gfx1151"]
    assert d["floor"] == 0.15 and d["alpha"] == 0.05


def test_gate_plan_extra_model_included_only_where_it_fits(pr_gate_toml):
    rc, out = _run(["gate", "--arch", "gfx1151", "--plan", "--models", "deepseek4",
                    "--gate-config", pr_gate_toml])
    assert json.loads(out)["models"] == ["qwen3.6-27b", "qwen3.6-a3b", "deepseek4"]
    rc, out = _run(["gate", "--arch", "gfx1100", "--plan", "--models", "deepseek4",
                    "--gate-config", pr_gate_toml])
    assert json.loads(out)["models"] == ["qwen3.6-27b", "qwen3.6-a3b"]


def test_gate_is_operator_only():
    rc, out = _run(["--role", "agent", "gate", "--arch", "gfx1201", "--plan"])
    assert rc == 3
    assert json.loads(out)["reason"] == "ROLE_FORBIDDEN"


def test_gate_validate_plan_emits_empty_matrix_for_trivial_skip(tmp_path, pr_gate_toml):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": 1,
        "decision": "skip",
        "risk": "trivial",
        "reason": "no runtime change",
        "boxes": {
            "hipx": {"run": False, "archs": [], "models": [], "behavior_tests": []},
            "hiptrx": {"run": False, "archs": [], "models": [], "behavior_tests": []},
        },
    }))
    rc, out = _run(["gate", "--validate-plan", str(plan), "--base", "HEAD", "--head", "HEAD",
                    "--gate-config", pr_gate_toml])
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["valid"] is True and parsed["matrix"] == {"include": []}


def test_gate_validate_action_refuses_green_for_bod(tmp_path):
    interp = tmp_path / "interp.json"
    action = tmp_path / "action.json"
    interp.write_text(json.dumps({"outcome": {"status": "failure", "action": "bod"}}))
    action.write_text(json.dumps({"action": "auto_merge", "comment_markdown": "green"}))
    rc, out = _run(["gate", "--validate-action", str(action), "--interp-json", str(interp)])
    assert rc == 2
    assert json.loads(out)["valid"] is False

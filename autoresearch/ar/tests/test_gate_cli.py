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

# Copyright (c) Kaden Schutt
import os
import tempfile

from autoresearch.ar.gate.config import GateConfig, load_gate_config

_TOML = """
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
floor = 0.15
drift_pct = 3.0
alpha = 0.05

[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]
"qwen3.6-a3b" = ["gfx1100", "gfx1151", "gfx1201"]
"deepseek4" = ["gfx1151"]
"""


def _write(tmp):
    p = os.path.join(tmp, "pr_gate.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    return p


def test_load_and_fields():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert isinstance(cfg, GateConfig)
    assert cfg.canonical_models == ["qwen3.6-27b", "qwen3.6-a3b"]
    assert cfg.maintainers == ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
    assert cfg.floor == 0.15 and cfg.drift_pct == 3.0 and cfg.alpha == 0.05


def test_fits_respects_map():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.fits("qwen3.6-27b", "gfx1100") is True
    assert cfg.fits("deepseek4", "gfx1151") is True
    assert cfg.fits("deepseek4", "gfx1100") is False        # DS4 does not fit 24 GB
    assert cfg.fits("unknown-sku", "gfx1201") is False       # unknown model -> not fit


def test_other_archs_excludes_self():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    assert cfg.other_archs("gfx1201") == ["gfx1100", "gfx1151"]


def test_models_for_filters_by_fit_and_adds_extra():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_gate_config(_write(tmp))
    # canonical only, gfx1100: DS4 excluded automatically
    assert cfg.models_for("gfx1100") == ["qwen3.6-27b", "qwen3.6-a3b"]
    # extra DS4 requested on gfx1151 -> included (fits); on gfx1100 -> dropped (no fit)
    assert cfg.models_for("gfx1151", extra=("deepseek4",)) == [
        "qwen3.6-27b", "qwen3.6-a3b", "deepseek4"]
    assert cfg.models_for("gfx1100", extra=("deepseek4",)) == [
        "qwen3.6-27b", "qwen3.6-a3b"]

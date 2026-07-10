# Copyright (c) Kaden Schutt
from autoresearch.ar.config import load_config


def test_loads_workers_and_bounds(tmp_path):
    cfg = load_config("autoresearch/config/loop_gfx1201.toml")
    assert cfg.arch == "gfx1201"
    assert cfg.model.endswith("mq4r")
    assert cfg.k_exhaust == 5
    assert [w.model for w in cfg.workers] == ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
    assert [w.effort for w in cfg.workers] == ["max", "xhigh", "medium"]
    assert cfg.bounds.call_budget == 400

# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Bjoern Agent

import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "oracle_gemma4_26b.py"
_spec = importlib.util.spec_from_file_location("oracle_gemma4_26b", SCRIPT)
_oracle = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_oracle)


def test_parser_accepts_absolute_capture_position_and_boundaries():
    args = _oracle.build_parser().parse_args(
        ["--ids", "2,9259,106", "--position", "1", "--boundaries"]
    )
    assert args.position == 1
    assert args.boundaries is True


def test_capture_position_defaults_to_last_and_rejects_out_of_range():
    assert _oracle.resolve_capture_position(None, 5) == 4
    assert _oracle.resolve_capture_position(0, 5) == 0
    with pytest.raises(ValueError):
        _oracle.resolve_capture_position(5, 5)
    with pytest.raises(ValueError):
        _oracle.resolve_capture_position(-1, 5)


def test_nonfinite_stats_become_json_null_under_strict_encoding():
    for value in (math.nan, math.inf, -math.inf):
        assert _oracle.finite_round(value, 4) is None
    payload = {"nan": _oracle.finite_round(math.nan, 4)}
    assert json.dumps(payload, allow_nan=False) == '{"nan": null}'
    assert _oracle.finite_round(1.23456, 4) == 1.2346

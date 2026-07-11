# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.perf_policy import classify_perf


def test_clear_regression_var_slower_both_axes():
    # variant tok/s down (150 -> 140) AND duration up (10.0 -> 10.8), tight -> significant
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[140] * 8,
        base_dur=[10.0] * 8, var_dur=[10.8] * 8,
    ) == "REGRESSION"


def test_clear_improvement_var_faster_both_axes():
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[162] * 8,
        base_dur=[10.0] * 8, var_dur=[9.2] * 8,
    ) == "IMPROVEMENT"


def test_neutral_within_floor():
    # 0.05% moves, well under FLOOR=0.15 -> NEUTRAL even though tight
    assert classify_perf(
        base_tok=[1000.0] * 8, var_tok=[999.5] * 8,
        base_dur=[10.0] * 8, var_dur=[10.005] * 8,
    ) == "NEUTRAL"


def test_one_sided_move_is_neutral_not_regression():
    # tok/s down but duration FLAT -> not a conjunctive regression -> NEUTRAL
    assert classify_perf(
        base_tok=[150] * 8, var_tok=[140] * 8,
        base_dur=[10.0] * 8, var_dur=[10.0] * 8,
    ) == "NEUTRAL"


def test_noisy_overlap_is_neutral():
    # large overlap between arms -> dominance in the mushy middle -> NEUTRAL
    assert classify_perf(
        base_tok=[150, 149, 151, 148, 152, 150, 149, 151],
        var_tok=[149, 151, 148, 152, 150, 149, 151, 150],
        base_dur=[10.0, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1],
        var_dur=[10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1, 9.9],
    ) == "NEUTRAL"

# Copyright (c) Kaden Schutt
"""serve_harness-driven cell grading: parity (content) -> coherence (attractor) -> perf
(decode_tok_s WIN-gate mirror). Pure grading, no GPU/serve."""
from autoresearch.ar.gate.serve_probe import grade_cell


def _row(content, tok_s=130.0, wall=1.0, attractor=False):
    return {"assistant_content": content, "decode_tok_s": tok_s, "wall_s": wall, "attractor": attractor}


def test_identical_content_neutral_perf_passes():
    base = [_row("hello world", 130.0), _row("foo bar", 131.0)]
    head = [_row("hello world", 129.5), _row("foo bar", 130.5)]   # ~neutral
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "PASS"


def test_content_mismatch_is_parity_reject():
    base = [_row("hello world"), _row("foo bar")]
    head = [_row("hello world"), _row("foo BAZ")]                 # second prompt differs
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "REJECT" and r["reason"] == "parity"


def test_new_attractor_on_head_is_coherence_reject():
    base = [_row("hello", attractor=False)]
    head = [_row("hello", attractor=True)]                        # same content, new attractor
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "REJECT" and r["reason"] == "coherence"


def test_preexisting_attractor_on_both_is_not_a_new_regression():
    # if base ALSO attractors (same content), the head didn't introduce it -> not a reject.
    base = [_row("loop loop", attractor=True)]
    head = [_row("loop loop", attractor=True)]
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "PASS"


def test_empty_generation_either_side_rejects():
    assert grade_cell([], [_row("x")], arch="a", model="m", floor=0.15)["reason"] == "empty_generation"
    assert grade_cell([_row("")], [_row("")], arch="a", model="m", floor=0.15)["reason"] == "empty_generation"


def test_perf_regression_rejects_with_negative_delta():
    # head much slower on tok/s AND slower wall, replicated across prompts -> REGRESSION.
    base = [_row("x", tok_s=200.0, wall=1.0) for _ in range(4)]
    head = [_row("x", tok_s=150.0, wall=1.4) for _ in range(4)]
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "REJECT" and r["reason"] == "perf_regression"
    assert r["tok_delta_pct"] < 0


def test_no_perf_samples_does_not_crash_or_fail_perf():
    base = [{"assistant_content": "x"}]      # no decode_tok_s / wall_s
    head = [{"assistant_content": "x"}]
    r = grade_cell(base, head, arch="gfx1201", model="m", floor=0.15)
    assert r["gate_verdict"] == "PASS" and r["tok_delta_pct"] == 0.0

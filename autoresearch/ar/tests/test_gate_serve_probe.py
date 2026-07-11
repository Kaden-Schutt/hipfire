# Copyright (c) Kaden Schutt
"""serve_harness-driven cell grading -> a ledger ROW (verdict.make_row schema, gate
verdict vocabulary) + per-(model,arch) BOD blockers. Pure grading, no GPU/serve."""
import subprocess

import pytest

from autoresearch.ar.gate.serve_probe import cell_blocker, grade_cell, run_serve_harness


def _row(content, tok_s=130.0, wall=1.0, attractor=False):
    return {"assistant_content": content, "decode_tok_s": tok_s, "wall_s": wall, "attractor": attractor}


def _grade(base, head, model="qwen3.6-27b.mq4"):
    return grade_cell(base, head, arch="gfx1201", model=model, floor=0.15)


# ---- row shape (loop ledger schema) ----

def test_cell_row_is_ledger_shaped():
    r = _grade([_row("hi", 130.0)], [_row("hi", 130.0)])
    for k in ("arch", "verdict", "parity", "perf_delta", "coherence", "model", "base_decode", "var_decode"):
        assert k in r, k
    assert r["verdict"] == "PASS" and r["arch"] == "gfx1201" and r["model"] == "qwen3.6-27b.mq4"


# ---- verdict vocabulary (aligned with the loop) ----

def test_identical_content_neutral_perf_passes():
    r = _grade([_row("hello world", 130.0), _row("foo bar", 131.0)],
               [_row("hello world", 129.5), _row("foo bar", 130.5)])
    assert r["verdict"] == "PASS" and r["parity"]["content_exact"] is True


def test_content_mismatch_is_parity_fail():
    r = _grade([_row("hello"), _row("foo bar")], [_row("hello"), _row("foo BAZ")])
    assert r["verdict"] == "PARITY_FAIL" and r["parity"]["content_exact"] is False


def test_new_attractor_on_head_is_coherence_fail():
    r = _grade([_row("hi", attractor=False)], [_row("hi", attractor=True)])
    assert r["verdict"] == "COHERENCE_FAIL" and r["coherence"]["pass"] is False


def test_preexisting_attractor_on_both_is_not_new():
    r = _grade([_row("loop", attractor=True)], [_row("loop", attractor=True)])
    assert r["verdict"] == "PASS"


def test_empty_generation_either_side():
    assert _grade([], [_row("x")])["verdict"] == "EMPTY"
    assert _grade([_row("")], [_row("")])["verdict"] == "EMPTY"


def test_perf_regression_negative_delta():
    r = _grade([_row("x", tok_s=200.0, wall=1.0) for _ in range(4)],
               [_row("x", tok_s=150.0, wall=1.4) for _ in range(4)])
    assert r["verdict"] == "REGRESSION" and r["tok_delta_pct"] < 0


def test_no_perf_samples_passes_without_crash():
    r = _grade([{"assistant_content": "x"}], [{"assistant_content": "x"}])
    assert r["verdict"] == "PASS" and (r["tok_delta_pct"] or 0.0) == 0.0


# ---- 27b-breaks-not-a3b: per-model isolation + itemized BOD ----

def test_change_breaks_27b_but_not_a3b():
    # 27b path changed (content differs) -> PARITY_FAIL; a3b unchanged (identical) -> PASS.
    r_27b = _grade([_row("dense out A")], [_row("dense out B")], model="qwen3.6-27b.mq4")
    r_a3b = _grade([_row("moe out")], [_row("moe out")], model="qwen3.6-35b-a3b.mq4r")
    assert r_27b["verdict"] == "PARITY_FAIL" and r_a3b["verdict"] == "PASS"
    # the BOD blocker names the exact (model, arch) that broke.
    b = cell_blocker(r_27b)
    assert b["kind"] == "parity" and b["model"] == "qwen3.6-27b.mq4" and b["arch"] == "gfx1201"
    assert "qwen3.6-27b.mq4 @ gfx1201: parity" in b["detail"]


def test_perf_blocker_detail_includes_delta():
    r = _grade([_row("x", tok_s=200.0, wall=1.0) for _ in range(4)],
               [_row("x", tok_s=150.0, wall=1.4) for _ in range(4)])
    b = cell_blocker(r)
    assert b["kind"] == "perf_regression" and "% tok/s" in b["detail"]


def test_serve_probe_timeout_becomes_structured_runtime_error():
    def timeout_runner(argv, env, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    with pytest.raises(RuntimeError, match="serve_harness timeout after 7s"):
        run_serve_harness(
            "/tmp/daemon", "/tmp/model", 0, repo="/repo", timeout=7,
            run=timeout_runner,
        )

# Copyright (c) Kaden Schutt
"""Certify verdict parity vs ab_certify_v2p.sh on captured real ledger cases (Task 2.5).

This is the Codex-verify seam: before the bash certify is deleted (Phase 8), the ported ``certify()``
MUST reproduce the SAME verdict + delta sign that ``ab_certify_v2p.sh`` recorded on real rows.

``fixtures/certify_cases.json`` captures one real WIN and one real DEAD row (arch/kernel/label +
per-round ``kernel_decode_tok_s`` samples). The v2 gate is CONJUNCTIVE (tok/s UP *and* duration DOWN),
while the old ledger recorded only tok/s; we replay by deriving duration = C / tok_s (physically: a
real kernel win lowers per-token duration exactly as it raises tok/s, and a DEAD moves neither axis).
A mock runner replays those measurements through the full parity→perf→coherence orchestrator; parity
and coherence are held clean (the captured rows have base_coh/var_coh == "OK"), isolating the perf
reproduction.
"""
import json
import math
import os

from autoresearch.ar.certify.orchestrator import certify, ServeRunner

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "certify_cases.json")
_CASES = json.load(open(_FIX))
_C = 1000.0  # duration scale constant (duration_ms = C / tok_s)


class ReplayRunner(ServeRunner):
    """Replays captured tok/s (and derived duration) samples; parity + coherence held clean."""

    def __init__(self, base_tok, var_tok):
        self._tok = {"base": base_tok, "var": var_tok}
        self._dur = {"base": [_C / t for t in base_tok], "var": [_C / t for t in var_tok]}

    def parity_gens(self, daemon):
        return [{"prompt_id": "p1", "token_ids": [1, 2, 3]}]        # base==var -> parity PASS

    def perf_measure(self, daemon):
        return (self._tok[daemon], self._dur[daemon])

    def coherence_gens(self, daemon, seeds):
        return [{"prompt_id": "prose", "genre": "prose", "seed": s,
                 "text": "a coherent sentence here", "token_ids": list(range(1000, 1060)),
                 "tool_calls": []} for s in seeds]

    def clocks(self, daemon):
        return []


def _certify_case(case):
    r = ReplayRunner(case["base_tok_s"], case["var_tok_s"])
    return certify(r, arch=case["arch"], kernel=case["kernel"], lever=case["label"],
                   base_daemon="base", var_daemon="var", base_ref="loop/gfx1201",
                   model="qwen3.6-35b-a3b.mq4r", kv="q8", maxtok=128, prompt_md5="d97ec9d3")


def test_win_case_reproduces_win():
    case = _CASES["win"]
    row = _certify_case(case)
    assert row["verdict"] == case["ledger_verdict"] == "WIN"
    # tok/s went UP (positive) — same direction the bash recorded
    assert row["tok_delta_pct"] > 0 and case["ledger_delta_pct"] > 0
    # duration went DOWN (the conjunctive second axis)
    assert row["dur_delta_pct"] < 0


def test_win_delta_matches_ledger_magnitude():
    # the ported tok/s delta must match the bash-recorded delta_pct (same median math, same samples)
    case = _CASES["win"]
    row = _certify_case(case)
    assert math.isclose(row["tok_delta_pct"], case["ledger_delta_pct"], abs_tol=0.02)


def test_dead_case_reproduces_dead():
    case = _CASES["dead"]
    row = _certify_case(case)
    assert row["verdict"] == case["ledger_verdict"] == "DEAD"
    # flat tok/s (delta ~0, not an improvement) — the bash recorded delta_pct == 0.0
    assert row["tok_delta_pct"] <= 0.0 or abs(row["tok_delta_pct"]) < 0.05
    assert math.isclose(row["tok_delta_pct"], case["ledger_delta_pct"], abs_tol=0.05)


def test_both_cases_self_describing():
    for name in ("win", "dead"):
        row = _certify_case(_CASES[name])
        assert {"gpu_arch", "measurement_hash", "tok_delta_pct", "dur_delta_pct"} <= set(row)

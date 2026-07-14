# Copyright (c) Kaden Schutt
"""codex usage-limit resilience (operator directive: codex out of usage -> grok fallback,
OR wait for codex to reset when sol-class intelligence is required)."""
from autoresearch.ar.agent_exec import (
    _is_codex_usage_limit,
    is_sol_class,
    run_round_resilient,
)

USAGE = "stream error: You've reached your usage limit. resets at 3pm"


class _Recorder:
    """A run_fn stub: returns queued (rc, output) per call and records the harness/model."""

    def __init__(self, script):
        self.script = list(script)   # list of (rc, output)
        self.calls = []              # list of (harness, model)

    def __call__(self, harness, model, effort, prompt, cwd, max_turns):
        self.calls.append((harness, model))
        return self.script[len(self.calls) - 1]


# ---- marker detection ----

def test_usage_limit_only_trips_on_nonzero_exit():
    assert _is_codex_usage_limit(1, USAGE) is True
    assert _is_codex_usage_limit(0, USAGE) is False          # succeeded -> not a refusal
    assert _is_codex_usage_limit(1, "AssertionError in test") is False  # real failure, not usage
    assert is_sol_class("gpt-5.6-sol") and not is_sol_class("gpt-5.6-terra")


# ---- codex succeeds: no fallback, no wait ----

def test_codex_success_runs_once():
    run = _Recorder([(0, "ok")])
    rc = run_round_resilient("codex", "gpt-5.6-terra", "high", "p", "/r", run_fn=run)
    assert rc == 0 and run.calls == [("codex", "gpt-5.6-terra")]


def test_codex_real_failure_is_not_masked_by_grok():
    # nonzero WITHOUT a usage marker => genuine failure, returned as-is (no model swap).
    run = _Recorder([(2, "panic: boom")])
    rc = run_round_resilient("codex", "gpt-5.6-terra", "high", "p", "/r", run_fn=run)
    assert rc == 2 and run.calls == [("codex", "gpt-5.6-terra")]


# ---- non-sol: usage-limit -> grok fallback ----

def test_non_sol_usage_limit_falls_back_to_grok():
    run = _Recorder([(1, USAGE), (0, "grok ok")])
    rc = run_round_resilient("codex", "gpt-5.6-terra", "high", "p", "/r",
                             grok_model="grok-4.5", run_fn=run)
    assert rc == 0
    assert run.calls == [("codex", "gpt-5.6-terra"), ("grok", "grok-4.5")]


# ---- sol-class: usage-limit -> wait for reset, then retry codex (NEVER grok) ----

def test_sol_usage_limit_waits_then_recovers_on_codex():
    run = _Recorder([(1, USAGE), (1, USAGE), (0, "codex back")])
    slept = []
    rc = run_round_resilient("codex", "gpt-5.6-sol", "xhigh", "p", "/r",
                             run_fn=run, sleep_fn=slept.append,
                             reset_poll_secs=5, max_codex_waits=8)
    assert rc == 0
    # all retries are codex — no grok downgrade for sol-class
    assert [h for h, _ in run.calls] == ["codex", "codex", "codex"]
    assert slept == [5, 5]                    # waited twice before the recovery


def test_sol_usage_limit_exhausts_waits_and_punts_without_grok():
    run = _Recorder([(1, USAGE), (1, USAGE), (1, USAGE)])
    slept = []
    rc = run_round_resilient("codex", "gpt-5.6-sol", "xhigh", "p", "/r",
                             run_fn=run, sleep_fn=slept.append,
                             reset_poll_secs=5, max_codex_waits=2)
    assert rc == 1                            # codex's failing rc -> caller punts
    assert [h for h, _ in run.calls] == ["codex", "codex", "codex"]   # 1 + 2 waits
    assert "grok" not in [h for h, _ in run.calls]
    assert slept == [5, 5]


def test_explicit_require_sol_overrides_model_name():
    # a terra round the operator marks sol-critical waits instead of degrading.
    run = _Recorder([(1, USAGE), (0, "codex back")])
    slept = []
    rc = run_round_resilient("codex", "gpt-5.6-terra", "high", "p", "/r",
                             require_sol=True, run_fn=run, sleep_fn=slept.append,
                             reset_poll_secs=3, max_codex_waits=4)
    assert rc == 0 and [h for h, _ in run.calls] == ["codex", "codex"] and slept == [3]


# ---- non-codex harness: single run, policy untouched ----

def test_grok_harness_runs_once():
    run = _Recorder([(0, "ok")])
    rc = run_round_resilient("grok", "grok-4.5", "high", "p", "/r", run_fn=run)
    assert rc == 0 and run.calls == [("grok", "grok-4.5")]

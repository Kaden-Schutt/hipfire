# Copyright (c) Kaden Schutt
"""No-GPU unit tests for ar.agent_exec — per-worker codex/grok round dispatch.

Ports ``harness/agent_exec.sh``. The contract that later phases depend on:
``build_argv`` emits a ``codex exec`` invocation carrying ``-m <model>`` and
``-c model_reasoning_effort=<effort>``, with the prompt delivered via **stdin**
(the trailing ``-`` sentinel), and ``run_round`` pipes the prompt on stdin and
returns the child's return code.
"""
import subprocess

import pytest

from autoresearch.ar.agent_exec import build_argv, run_round


# ── plan Task 4.1 Step 1 (the mandated contract) ──────────────────────────────
def test_codex_argv_has_model_and_effort():
    argv = build_argv("codex", "gpt-5.6-luna", "max", "/tmp/p.md", "/repo", 100)
    s = " ".join(argv)
    assert "codex" in argv[0] and "-m" in argv and "gpt-5.6-luna" in argv
    assert 'model_reasoning_effort="max"' in s or "model_reasoning_effort=max" in s


# ── codex shape: exec subcommand, -C cwd, prompt via stdin sentinel ───────────
def test_codex_argv_is_exec_with_cwd_and_stdin_sentinel():
    argv = build_argv("codex", "m", "medium", "-", "/work", 100)
    assert argv[0] == "codex" and argv[1] == "exec"
    assert "-C" in argv and argv[argv.index("-C") + 1] == "/work"
    assert argv[-1] == "-"  # prompt delivered via stdin, not embedded in argv


def test_codex_effort_threads_through_each_worker():
    for eff in ("max", "xhigh", "medium"):
        s = " ".join(build_argv("codex", "m", eff, "-", "/w", 1))
        assert f'model_reasoning_effort="{eff}"' in s


def test_codex_model_flag_pairs_with_model():
    argv = build_argv("codex", "gpt-5.6-terra", "xhigh", "-", "/w", 1)
    assert argv[argv.index("-m") + 1] == "gpt-5.6-terra"


# ── grok shape (strictly additive; not exercised live in no-GPU CI) ───────────
def test_grok_argv_has_max_turns_model_and_bypass():
    argv = build_argv("grok", "grok-code", "xhigh", "-", "/repo", 120)
    assert "grok" in argv[0]
    assert "--max-turns" in argv and argv[argv.index("--max-turns") + 1] == "120"
    assert argv[argv.index("-m") + 1] == "grok-code"
    assert "bypassPermissions" in argv
    assert argv[argv.index("--cwd") + 1] == "/repo"


def test_unknown_harness_raises():
    with pytest.raises(ValueError):
        build_argv("frobnitz", "m", "max", "-", "/w", 1)


def test_harness_case_insensitive_and_defaults_codex():
    assert build_argv("CODEX", "m", "max", "-", "/w", 1)[0] == "codex"
    assert build_argv(None, "m", "max", "-", "/w", 1)[0] == "codex"


# ── run_round pipes the prompt on stdin and returns the child's rc ────────────
def test_run_round_pipes_prompt_via_stdin(monkeypatch):
    seen = {}

    def fake_run(argv, input=None, text=None, **kw):
        seen["argv"] = argv
        seen["input"] = input

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = run_round("codex", "gpt-5.6-luna", "max", "PROMPT-BODY", "/repo", 100)
    assert rc == 0
    assert seen["input"] == "PROMPT-BODY"  # prompt via stdin, not argv
    assert "PROMPT-BODY" not in " ".join(seen["argv"])
    assert "gpt-5.6-luna" in seen["argv"]  # model still on argv
    assert seen["argv"][-1] == "-"  # stdin sentinel


def test_run_round_forwards_returncode(monkeypatch):
    def fake_run(argv, input=None, text=None, **kw):
        class R:
            returncode = 42

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert run_round("codex", "m", "max", "p", "/repo", 1) == 42

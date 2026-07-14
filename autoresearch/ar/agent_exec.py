# Copyright (c) Kaden Schutt
"""ar.agent_exec — one harness-agnostic autonomous coding round for the loop.

Ports ``harness/agent_exec.sh``: one user prompt → the agent works autonomously
(reads ``kernels/src``, edits, builds, runs the certify wrapper, commits branch
wins) → exits. Dispatches on the harness so a Grok worker and a Codex worker can
run on DIFFERENT cards over the SAME certify/rollover substrate (heterogeneous
model fleet, orthogonal kernel generation).

Two differences from the bash predecessor, both mandated by the migration plan:

* **Per-worker ``model`` + reasoning ``effort``.** Each worker carries its own
  ``{model, effort}`` (the ``swarm_explore.sh`` ``sed``-munge replacement), so
  the Codex round is emitted as ``codex exec ... -m <model>
  -c model_reasoning_effort="<effort>" ...`` — the effort knob the bash never
  plumbed through.
* **Prompt via stdin.** The round prompt (digest + arch prompt, built by the
  driver) can be large and whitespace-sensitive; delivering it on **stdin**
  (the trailing ``-`` sentinel) avoids argv-length limits and shell-quoting
  fragility. :func:`run_round` pipes the prompt string to the child's stdin.

``build_argv`` is the pure, unit-testable argv constructor; :func:`run_round`
is the thin executor (pipe prompt on stdin, return the child's rc).
"""
from __future__ import annotations

import os
import subprocess
import time

# grok agent-turn cap default (codex is bounded by its per-round wall timeout,
# not a turn count — it ignores this).
DEFAULT_MAX_TURNS = 120

# Substrings that mark a codex USAGE-LIMIT / rate-limit refusal (not a task
# failure). Matched case-insensitively against the codex round's combined
# stdout+stderr, and ONLY when the round also exited non-zero — so a codex
# answer that merely *mentions* "usage limit" while succeeding never trips it.
# Broad on purpose (codex's exact wording drifts between releases); the
# nonzero-exit gate is what keeps it from false-positiving on task content.
CODEX_USAGE_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "rate-limit",
    "quota",
    "429",
    "too many requests",
    "you've reached your",
    "you have reached your",
    "resets at",
    "try again later",
    "resource_exhausted",
)

# stdin sentinel: codex/grok read the prompt from stdin when the positional
# prompt slot is ``-`` (the driver always pipes the prompt via :func:`run_round`).
STDIN = "-"


def build_argv(
    harness: str,
    model: str,
    effort: str,
    prompt_file: str,
    cwd: str,
    max_turns: int,
) -> list[str]:
    """Build the argv for one autonomous round.

    ``prompt_file`` occupies the harness's positional prompt slot; pass ``"-"``
    (the :data:`STDIN` sentinel) to have the agent read the prompt from stdin —
    which is what :func:`run_round` does. The prompt is NEVER embedded as a
    shell string.

    * ``codex`` → ``codex exec --dangerously-bypass-approvals-and-sandbox
      --skip-git-repo-check -m <model> -c model_reasoning_effort="<effort>"
      -C <cwd> <prompt_file>``. The bypass flag keeps the round unattended (the
      exact prior loop semantics); ``-c`` threads the per-worker reasoning
      effort through as a TOML string override.
    * ``grok`` → the equivalent unattended one-turn invocation
      (``--permission-mode bypassPermissions``, ``--output-format plain``,
      ``--max-turns``). Grok has no reasoning-effort knob, so ``effort`` is
      accepted for signature uniformity but not emitted. ``GROK_BIN`` overrides
      the binary (the loop runs non-interactively, so PATH usually lacks it).

    Raises ``ValueError`` on an unknown harness.
    """
    h = (harness or "codex").lower()
    if h == "codex":
        # exact prior autonomy (bypass-approvals) + the new per-worker model/effort;
        # prompt via stdin sentinel instead of an argv positional string.
        return [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
            "-C",
            cwd,
            prompt_file,
        ]
    if h == "grok":
        grok = os.environ.get("GROK_BIN", "grok")
        argv = [
            grok,
            "-p",
            prompt_file,
            "--cwd",
            cwd,
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "plain",
            "--max-turns",
            str(max_turns),
        ]
        if model:
            argv += ["-m", model]
        return argv
    raise ValueError(f"unknown harness {harness!r} (want: codex | grok)")


def run_round(
    harness: str,
    model: str,
    effort: str,
    prompt: str,
    cwd: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> int:
    """Run one autonomous round; return the agent process's exit code.

    Codex reads the prompt from **stdin** (the argv carries the ``-`` sentinel).
    Grok's ``-p/--single`` takes the prompt as an ARG *value* (it does NOT read
    stdin), so for grok the prompt is embedded in the argv and nothing is piped.
    """
    h = (harness or "codex").lower()
    if h == "grok":
        argv = build_argv(harness, model, effort, prompt, cwd, max_turns)
        proc = subprocess.run(argv, text=True)
    else:
        argv = build_argv(harness, model, effort, STDIN, cwd, max_turns)
        proc = subprocess.run(argv, input=prompt, text=True)
    return proc.returncode


def _run_capture(harness, model, effort, prompt, cwd, max_turns):
    """One round like :func:`run_round`, but capturing combined stdout+stderr so the
    caller can scan it for a usage-limit marker. Echoes the output back so CI logs
    still show the round. Returns ``(returncode, combined_output)``."""
    h = (harness or "codex").lower()
    if h == "grok":
        argv = build_argv(harness, model, effort, prompt, cwd, max_turns)
        proc = subprocess.run(argv, text=True, capture_output=True)
    else:
        argv = build_argv(harness, model, effort, STDIN, cwd, max_turns)
        proc = subprocess.run(argv, input=prompt, text=True, capture_output=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    return proc.returncode, out


def is_sol_class(model: str) -> bool:
    """A 'sol-class' round needs the top intelligence tier (gpt-5.6-sol). We DON'T
    silently degrade these to grok — when codex is usage-limited we wait for it to
    reset instead. luna/terra rounds are grok-substitutable."""
    return "sol" in (model or "").lower()


def _is_codex_usage_limit(rc: int, output: str) -> bool:
    if rc == 0:
        return False
    low = (output or "").lower()
    return any(m in low for m in CODEX_USAGE_LIMIT_MARKERS)


def run_round_resilient(
    harness: str,
    model: str,
    effort: str,
    prompt: str,
    cwd: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    *,
    require_sol: bool | None = None,
    grok_model: str | None = None,
    grok_effort: str | None = None,
    run_fn=None,
    sleep_fn=None,
    reset_poll_secs: int | None = None,
    max_codex_waits: int | None = None,
) -> int:
    """Run one round, resilient to a codex USAGE-LIMIT refusal (a drop-in for
    :func:`run_round`, same positional signature, returns the exit code).

    Policy (per operator directive): a codex round that fails **because codex is out
    of usage** either falls back to grok or waits for codex to reset —

    * **sol-class** round (``require_sol``; defaults to ``is_sol_class(model)``): the
      top intelligence tier is *required*, so we DON'T degrade to grok. We wait
      ``reset_poll_secs`` and retry codex, up to ``max_codex_waits`` times, until it
      recovers; if the wait budget is exhausted we return codex's failing rc (the
      caller punts — never a silent sol→grok downgrade).
    * **non-sol** round (luna/terra): fall back to grok immediately with the SAME
      prompt (``grok_model``, default ``$GATE_GROK_MODEL`` or ``grok-4.5``).

    Only a usage-limit refusal triggers this — a genuine codex error (nonzero exit
    without a usage marker) is returned as-is (never masked by a model swap), and a
    non-codex harness runs exactly once. ``run_fn(harness,model,effort,prompt,cwd,
    max_turns)->(rc,output)`` and ``sleep_fn(secs)`` are injected for testing."""
    run_fn = run_fn or _run_capture
    h = (harness or "codex").lower()
    if h != "codex":
        rc, _ = run_fn(harness, model, effort, prompt, cwd, max_turns)
        return rc

    if require_sol is None:
        require_sol = is_sol_class(model)

    rc, out = run_fn("codex", model, effort, prompt, cwd, max_turns)
    if not _is_codex_usage_limit(rc, out):
        return rc  # success, or a real (non-usage) failure — return as-is

    if not require_sol:
        gm = grok_model or os.environ.get("GATE_GROK_MODEL", "grok-4.5")
        ge = grok_effort if grok_effort is not None else effort
        print(f"[agent_exec] codex usage-limited on non-sol '{model}' -> grok fallback ({gm})")
        rc2, _ = run_fn("grok", gm, ge, prompt, cwd, max_turns)
        return rc2

    # sol-class: wait for codex to reset, then retry (no grok downgrade).
    sleep_fn = sleep_fn or time.sleep
    poll = reset_poll_secs if reset_poll_secs is not None else int(os.environ.get("CODEX_RESET_POLL_SECS", "900"))
    waits = max_codex_waits if max_codex_waits is not None else int(os.environ.get("CODEX_MAX_WAITS", "8"))
    for i in range(waits):
        print(f"[agent_exec] codex usage-limited on sol '{model}'; waiting {poll}s for reset ({i + 1}/{waits})")
        sleep_fn(poll)
        rc, out = run_fn("codex", model, effort, prompt, cwd, max_turns)
        if not _is_codex_usage_limit(rc, out):
            return rc
    print(f"[agent_exec] codex still usage-limited on sol '{model}' after {waits} waits — punting (no grok downgrade)")
    return rc

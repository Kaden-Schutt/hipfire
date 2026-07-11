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

# grok agent-turn cap default (codex is bounded by its per-round wall timeout,
# not a turn count — it ignores this).
DEFAULT_MAX_TURNS = 120

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

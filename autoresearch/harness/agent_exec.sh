#!/usr/bin/env bash
# agent_exec.sh — harness-agnostic autonomous coding round for the autoresearch loop.
#
# One user prompt -> the agent works autonomously (reads kernels/src, edits, builds,
# runs the certify wrapper, commits branch wins) -> exits. Dispatches on AGENT_HARNESS
# so a Grok worker and a Codex worker can run on DIFFERENT cards over the SAME
# certify/rollover substrate (heterogeneous model fleet, orthogonal kernel generation).
#
# Positional:
#   $1  timeout seconds        (per-round wall cap)
#   $2  working directory      (this worker's advancing worktree; the agent's cwd)
#   $3  full round prompt       (single arg — build it in the caller, pass verbatim)
#
# Env:
#   AGENT_HARNESS   codex (default) | grok   -- which model family runs the round
#   AGENT_MODEL     optional model id override (-m for both harnesses)
#   AGENT_MAX_TURNS grok agent-turn cap (default 120); codex is bounded by the timeout
#   GROK_BIN        grok binary path (default ~/.local/bin/grok). The loop runs
#                   NON-interactively, so PATH usually will NOT contain grok — this
#                   default + the command -v fallback handle that.
#
# The DEFAULT path (AGENT_HARNESS unset) is byte-identical to the prior inlined
# `codex exec --dangerously-bypass-approvals-and-sandbox -C <cwd> <prompt>` call, so
# every existing campaign behaves exactly as before. Grok is strictly additive.
set -u
T="${1:?agent_exec: timeout seconds required}"
CWD="${2:?agent_exec: working directory required}"
PROMPT="${3:?agent_exec: prompt required}"
H="${AGENT_HARNESS:-codex}"

case "$H" in
  codex)
    # exact prior semantics — do not change without re-baselining the loop
    exec timeout "$T" codex exec --dangerously-bypass-approvals-and-sandbox \
      ${AGENT_MODEL:+-m "$AGENT_MODEL"} -C "$CWD" "$PROMPT"
    ;;
  grok)
    GROK="${GROK_BIN:-$HOME/.local/bin/grok}"
    [ -x "$GROK" ] || GROK="$(command -v grok 2>/dev/null || echo grok)"
    # -p = one user turn, agent runs autonomously to completion then exits (== codex exec);
    # bypassPermissions = unattended edits + shell + git; grok's OAuth creds live in ~/.grok
    # so this MUST run under the real HOME (the loop does — no HOME override before the round).
    exec timeout "$T" "$GROK" -p "$PROMPT" --cwd "$CWD" \
      --permission-mode bypassPermissions --output-format plain \
      --max-turns "${AGENT_MAX_TURNS:-120}" ${AGENT_MODEL:+-m "$AGENT_MODEL"}
    ;;
  *)
    echo "agent_exec: unknown AGENT_HARNESS='$H' (want: codex | grok)" >&2
    exit 2
    ;;
esac

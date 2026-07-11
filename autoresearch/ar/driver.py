# Copyright (c) Kaden Schutt
"""ar.driver — the SELF-EXHAUSTING, arch-configurable per-worker loop.

Ports ``harness/v2/driver_v3.sh``. No fixed round/time limit: the loop stops
when every live candidate kernel (BOD wall% >= ``cand_wall``, unfolded) has hit
``k_exhaust`` consecutive DEAD/INCONCLUSIVE rounds — ``safety_cap`` is a pure
backstop. Each round injects the per-kernel tried-levers digest so the agent
avoids re-derivation and drives coverage toward exhaustion.

**Dependency-injected seams (the whole point of the port).** Every side effect
— exhaustion check, digest render, agent round, exhaustion update, baseline
advance — is a hook. :func:`run_loop` merges the caller's ``hooks`` over a set
of real defaults bound to :mod:`ar.candidates`, :mod:`ar.agent_exec`, and
:mod:`ar.gitpilot`, so the loop's termination + sequencing logic is unit-tested
with NO agent / GPU / git. The defaults are all best-effort (never crash the
loop) because the live wiring is validated in the Phase-5 acceptance run.

Per-round order mirrors ``driver_v3``:
  1. global stop? (``is_exhausted``) → break
  2. build the round prompt = ``ROUND n … {digest} {arch prompt}``
  3. run one autonomous agent round (this worker's ``model`` + ``effort``)
  4. fold the round's ledger verdicts into the exhaustion counters
  5. advance the shared baseline on a banked WIN (``update-ref`` CAS)
"""
from __future__ import annotations

import json
import os

from . import agent_exec, candidates, gitpilot
from .config import LoopConfig, WorkerCfg


def _worker_cwd(repo: str, worker: WorkerCfg) -> str:
    """This worker's advancing worktree (``.aw/sw_card<card>``); repo if absent.

    The agent reads ``kernels/src`` from ITS OWN worktree (this worker's
    advancing baseline), never a foreign checkout (``driver_v3`` line 83-84).
    """
    wt = os.path.join(repo, ".aw", f"sw_card{worker.card}")
    return wt if os.path.isdir(os.path.join(wt, "kernels", "src")) else repo


def _load_prompt(repo: str, arch: str) -> str:
    """The arch round prompt (new ``config/prompt_<arch>.md``; harness fallback)."""
    for rel in (
        os.path.join("autoresearch", "config", f"prompt_{arch}.md"),
        os.path.join("autoresearch", "harness", f"loop_round_prompt_{arch}.txt"),
        os.path.join("autoresearch", "harness", "loop_round_prompt.txt"),
    ):
        try:
            with open(os.path.join(repo, rel), encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            continue
    return ""


def _retarget(prompt: str, repo: str, worker: WorkerCfg) -> str:
    """Fill the per-worker placeholders ``swarm_explore.sh``'s ``sed`` used to inject.

    ``{{repo}}`` → the host checkout, ``{{card}}`` / ``{{dev}}`` → this worker's
    DRM card / HIP device. Without this, every worker's prompt would target
    ``card0`` / ``dev 0`` and clobber the same ``/tmp/exp_…c0.hip`` (the migration
    dropped the sed-munge; the driver injects the targeting instead).
    """
    return (
        prompt.replace("{{repo}}", repo)
        .replace("{{card}}", str(worker.card))
        .replace("{{dev}}", str(worker.dev))
    )


def _default_hooks(cfg: LoopConfig, worker: WorkerCfg, repo: str) -> dict:
    """Real seams bound to the state store, agent harness, and git CAS.

    Each is defensive (returns a neutral value on any error) so a missing state
    file can never wedge the loop; the live paths are exercised in Phase 5.
    """
    state_dir = os.path.join(repo, "autoresearch", "state")
    bod_path = os.path.join(state_dir, f"bod_{cfg.arch}.json")
    exh_path = os.path.join(state_dir, f"exhaustion_{cfg.arch}.json")
    folded_path = os.path.join(state_dir, f"baseline_folded_{cfg.arch}.txt")

    def _load_json(path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _folded() -> list[str]:
        try:
            with open(folded_path, encoding="utf-8") as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            return []

    def is_exhausted(*_a, **_k) -> bool:
        try:
            return candidates.is_exhausted(
                _load_json(exh_path), _load_json(bod_path), cfg.cand_wall, cfg.k_exhaust, _folded()
            )
        except Exception:
            return False  # never prematurely self-terminate on a state read error

    def gen_digest(*_a, **_k) -> str:
        try:
            return candidates.gen_digest(
                _load_json(exh_path), _load_json(bod_path), cfg.cand_wall, cfg.k_exhaust, _folded()
            )
        except Exception:
            return ""

    def run_round(*, harness, model, effort, prompt, cwd, max_turns, **_k) -> int:
        return agent_exec.run_round(harness, model, effort, prompt, cwd, max_turns)

    def update_exhaustion(*, round: int, **_k) -> None:  # noqa: A002 (round mirrors driver_v3)
        try:
            candidates.update_exhaustion(exh_path, round, repo, cfg.arch)
        except Exception:
            pass

    def advance(*_a, **_k) -> bool:
        """Advance the shared baseline to this worker's HEAD on a banked WIN.

        ``git update-ref <baseline> <head> <expected>`` is a compare-and-swap:
        it only moves the ref when the worktree's HEAD has advanced past the
        baseline (a win landed) AND no parallel worker moved the ref meanwhile.
        """
        try:
            head = gitpilot.current_sha(_worker_cwd(repo, worker), "HEAD")
            base = gitpilot.current_sha(repo, cfg.baseline_ref)
            if head and base and head != base:
                return gitpilot.update_ref_cas(cfg.baseline_ref, head, base, repo)
        except Exception:
            pass
        return False

    return {
        "is_exhausted": is_exhausted,
        "gen_digest": gen_digest,
        "run_round": run_round,
        "update_exhaustion": update_exhaustion,
        "advance": advance,
    }


def run_loop(cfg: LoopConfig, worker: WorkerCfg, safety_cap: int, *, hooks: dict | None = None) -> int:
    """Run this worker's self-exhausting loop; return the number of rounds run.

    Loops until every live candidate is exhausted (``is_exhausted``) or
    ``safety_cap`` rounds elapse. ``hooks`` overrides any of the five seams
    (``is_exhausted`` / ``gen_digest`` / ``run_round`` / ``update_exhaustion`` /
    ``advance``) for testing; unspecified seams use the real defaults.
    """
    repo = os.getcwd()
    h = _default_hooks(cfg, worker, repo)
    if hooks:
        h.update(hooks)

    cwd = _worker_cwd(repo, worker)
    base_prompt = _retarget(_load_prompt(repo, cfg.arch), repo, worker)

    rounds = 0
    while rounds < safety_cap and not h["is_exhausted"]():
        rounds += 1
        digest = h["gen_digest"]() or ""
        round_prompt = (
            f"ROUND {rounds} of a SELF-EXHAUSTING autoresearch loop "
            f"(adaptive certify + branch wins). {digest} {base_prompt}"
        )
        h["run_round"](
            harness=cfg.agent_harness,
            model=worker.model,
            effort=worker.effort,
            prompt=round_prompt,
            cwd=cwd,
            max_turns=agent_exec.DEFAULT_MAX_TURNS,
        )
        h["update_exhaustion"](round=rounds)
        h["advance"](round=rounds)
    return rounds

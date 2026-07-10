# Copyright (c) Kaden Schutt
"""ar.swarm — config-driven parallel worker launcher (kills the ``sed``-munge).

Ports ``harness/swarm_explore.sh``. The bash configured each worker by
``sed``-rewriting a shared prompt file — retargeting HIP dev, DRM card, worktree
slot, wins-branch and lock path with a stack of substitutions. That is exactly
the fragile mechanism the migration deletes: here each worker's
``{card, dev, model, effort}`` comes STRAIGHT from the TOML ``[[workers]]`` list
(:class:`~autoresearch.ar.config.WorkerCfg`) into a per-worker launch plan, with
NO prompt rewrite.

:func:`plan_workers` is the pure, unit-testable mapping — one
``WorkerCfg`` → one ``{card, dev, model, effort, worktree, anchor, lockfile}``
dict:

* ``worktree`` — this worker's advancing checkout ``<repo>/.aw/sw_card<card>``
  (mirrors :func:`ar.driver._worker_cwd`).
* ``anchor`` — the per-worker checkout branch ``loop/<arch>_w<i>`` (0-based
  index). The anchor is ONLY the worktree checkout head; banked WINs advance the
  SHARED ``cfg.baseline_ref`` via ``update-ref`` CAS (the driver's ``advance``
  seam), so a win from ANY worker compounds into the baseline all workers
  inherit next round.
* ``lockfile`` — the per-**dev** GPU lock ``/tmp/hipfire-gpu-<arch>-dev<dev>.lock``.
  Two workers sharing a dev share this lock, so their CERTIFIES serialize (no
  measurement contention) while their BUILDS + agent reasoning overlap; workers
  on different devs run fully parallel.

:func:`launch` drives :func:`plan_workers`, ensures each worker's anchor branch,
and spawns a detached :func:`ar.driver.run_loop` per worker. The ``spawn`` and
``prepare`` seams are dependency-injected so the launcher's iteration logic is
no-GPU/no-git/no-fork unit-testable; the real defaults double-fork + ``setsid``
to fully detach each worker and pin its GPU via ``HIP_VISIBLE_DEVICES`` +
``HIPFIRE_GPU_LOCKFILE`` (the config-driven replacement for the bash ``sed``).
"""
from __future__ import annotations

import os
import subprocess

from . import driver
from .config import LoopConfig, WorkerCfg

# Per-worker backstop rounds — the loop self-exhausts long before this
# (``k_exhaust`` consecutive DEADs per candidate); this only bounds a pathologically
# never-exhausting candidate set. See ``ar.driver.run_loop`` ``safety_cap``.
DEFAULT_SAFETY_CAP = 1000


def _worktree(repo: str, card: int) -> str:
    """This worker's advancing checkout ``<repo>/.aw/sw_card<card>``."""
    return os.path.join(repo, ".aw", f"sw_card{card}")


def _anchor(arch: str, index: int) -> str:
    """The per-worker checkout branch ``loop/<arch>_w<index>`` (0-based)."""
    return f"loop/{arch}_w{index}"


def _lockfile(arch: str, dev: int) -> str:
    """The per-dev GPU lock ``/tmp/hipfire-gpu-<arch>-dev<dev>.lock``.

    In ``/tmp`` on purpose: it is an ``flock`` lockfile (kernel releases on holder
    death), never a bench artifact — this is where the bash harness kept it too.
    """
    return f"/tmp/hipfire-gpu-{arch}-dev{dev}.lock"


def plan_workers(cfg: LoopConfig, repo: str) -> list[dict]:
    """Map ``cfg.workers`` → per-worker launch plans (the ``sed``-munge replacement).

    Each plan is exactly ``{card, dev, model, effort, worktree, anchor, lockfile}``
    derived from the ``WorkerCfg`` and the arch — no prompt rewrite, no hardcoded
    card/dev/model. The list is 1:1 and order-preserving with ``cfg.workers`` (the
    0-based position is the ``_w<i>`` anchor index).
    """
    plans: list[dict] = []
    for i, w in enumerate(cfg.workers):
        plans.append(
            {
                "card": w.card,
                "dev": w.dev,
                "model": w.model,
                "effort": w.effort,
                "worktree": _worktree(repo, w.card),
                "anchor": _anchor(cfg.arch, i),
                "lockfile": _lockfile(cfg.arch, w.dev),
            }
        )
    return plans


def _default_prepare(cfg: LoopConfig, worker: WorkerCfg, plan: dict, repo: str) -> None:
    """Ensure this worker's anchor branch exists in its worktree (best-effort).

    Mirrors ``swarm_explore.sh``: ``git -C <worktree> branch <anchor> <baseline>``
    — the anchor points at the shared baseline at spawn time. Failure (branch
    already exists, or the worktree/ref is absent) is non-fatal: the driver
    advances the SHARED ``baseline_ref`` via CAS regardless of the anchor.
    """
    subprocess.run(
        ["git", "-C", plan["worktree"], "branch", plan["anchor"], cfg.baseline_ref],
        capture_output=True,
        text=True,
    )


def _redirect_stdio(log_path: str) -> None:
    """Point stdin at ``/dev/null`` and stdout/stderr at ``log_path`` (detach)."""
    devnull = os.open(os.devnull, os.O_RDONLY)
    logfd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(devnull, 0)
    os.dup2(logfd, 1)
    os.dup2(logfd, 2)
    for fd in (devnull, logfd):
        if fd > 2:
            os.close(fd)


def _run_worker_detached(cfg: LoopConfig, worker: WorkerCfg, plan: dict, safety_cap: int, repo: str) -> None:
    """Grandchild body: pin the GPU, redirect stdio, run the worker loop.

    The config-driven ``sed`` replacement lives here: ``HIP_VISIBLE_DEVICES`` +
    ``HIPFIRE_GPU_LOCKFILE`` are set from this worker's ``dev`` / ``lockfile`` so
    every daemon/certify subprocess the loop spawns targets THIS card and takes
    THIS dev's lock. ``run_loop`` re-derives ``repo`` from the CWD, so chdir first.
    """
    os.environ["HIP_VISIBLE_DEVICES"] = str(worker.dev)
    os.environ["HIPFIRE_GPU_LOCKFILE"] = plan["lockfile"]
    # The agent harness (codex/grok) + cargo + rocprof live in per-user bin dirs
    # that a NON-login spawn context lacks (swarm_explore.sh set this via PATH=…).
    # Without it run_round's subprocess.run(["codex",…]) FileNotFound-crashes.
    _home = os.path.expanduser("~")
    _bins = [os.path.join(_home, ".local/bin"), os.path.join(_home, ".cargo/bin"),
             os.path.join(_home, ".bun/bin"), "/opt/rocm/bin"]
    os.environ["PATH"] = os.pathsep.join(_bins + [os.environ.get("PATH", "")])
    try:
        os.chdir(repo)
    except OSError:
        pass
    log_dir = os.path.join(repo, "autoresearch", "state")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        pass
    wname = plan["anchor"].rsplit("/", 1)[-1]  # loop/gfx1201_w0 -> gfx1201_w0
    _redirect_stdio(os.path.join(log_dir, f"loop_driver_{wname}.log"))
    # Log the traceback to the (redirected) stderr BEFORE the caller's os._exit(0)
    # swallows it — otherwise a crashed worker leaves only a 0-byte log.
    try:
        driver.run_loop(cfg, worker, safety_cap)
    except BaseException:
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise


def _default_spawn(cfg: LoopConfig, worker: WorkerCfg, plan: dict, safety_cap: int, repo: str) -> int:
    """Double-fork + ``setsid`` a detached worker; return the grandchild pid.

    The grandchild is reparented to init (fully detached, session leader) so this
    launcher never accrues a zombie and the worker outlives it — the Python
    equivalent of the bash ``setsid nohup ... &``. The middle child reports the
    grandchild pid back over a pipe; the original process reaps the middle child.
    """
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid > 0:  # original process: read grandchild pid, reap middle child
        os.close(write_fd)
        data = os.read(read_fd, 32)
        os.close(read_fd)
        os.waitpid(pid, 0)
        try:
            return int(data.decode().strip() or "0")
        except ValueError:
            return 0

    # middle child
    os.close(read_fd)
    try:
        os.setsid()
    except OSError:
        pass
    gpid = os.fork()
    if gpid > 0:  # middle child: publish grandchild pid upward, exit
        try:
            os.write(write_fd, str(gpid).encode())
        finally:
            os.close(write_fd)
        os._exit(0)

    # grandchild: the detached worker (never returns to the caller)
    os.close(write_fd)
    try:
        _run_worker_detached(cfg, worker, plan, safety_cap, repo)
    finally:
        os._exit(0)


def launch(
    cfg: LoopConfig,
    repo: str,
    *,
    spawn=None,
    prepare=None,
    safety_cap: int = DEFAULT_SAFETY_CAP,
    require_worktree: bool = True,
) -> list[int]:
    """Launch one detached :func:`ar.driver.run_loop` per configured worker.

    Returns the list of spawned pids (one per launched worker). ``spawn`` and
    ``prepare`` are injectable seams (default: real double-fork detach + anchor
    branch ensure) so the iteration logic is unit-testable no-GPU. With
    ``require_worktree`` (default), a worker whose ``.aw/sw_card<card>`` checkout
    is absent is SKIPPED (bash ``swarm_explore`` parity: "w$i SKIP: no worktree")
    rather than spawned blind. ``spawn`` is called
    ``spawn(cfg, worker, plan, safety_cap, repo) -> pid``.
    """
    plans = plan_workers(cfg, repo)
    do_spawn = spawn or _default_spawn
    do_prepare = prepare if prepare is not None else _default_prepare

    pids: list[int] = []
    for worker, plan in zip(cfg.workers, plans):
        if require_worktree and not os.path.isdir(os.path.join(plan["worktree"], "kernels", "src")):
            continue  # no advancing worktree for this card — skip (bash parity)
        do_prepare(cfg, worker, plan, repo)
        pids.append(do_spawn(cfg, worker, plan, safety_cap, repo))
    return pids

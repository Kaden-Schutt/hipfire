# Copyright (c) Kaden Schutt
"""ar.gitpilot — Python drives git via ``subprocess`` (not GitPython/pygit2).

Shelling out to ``git`` preserves the two death-safe correctness guarantees the
bash harness relied on:

* **GPU lock** — ``flock`` held on an *open* file descriptor, so the kernel
  releases the lock when the holder dies for any reason (kill -9, crash, OOM,
  terminal close). We NEVER unlink the lockfile (unlinking an flock'd file lets
  a second acquirer lock a fresh inode → two holders). See ``scripts/gpu-lock.sh``.
* **Baseline advance** — ``git update-ref <ref> <new> <expected>`` is an atomic
  compare-and-swap: it fails (no change) if the ref moved since we read it,
  which is exactly the optimistic-concurrency advance ``ab_certify_v2p.sh`` used
  to compound wins across parallel workers without double-counting.

Ports the git bits of ``ab_certify_v2p.sh`` (CAS advance, ``git show SHA:path``)
and the flock-on-fd semantics of ``scripts/gpu-lock.sh``.
"""
from __future__ import annotations

import contextlib
import datetime
import fcntl
import os
import socket
import subprocess


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    """Run ``git -C <repo> <args...>`` capturing text stdout/stderr."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
    )


def current_sha(repo: str, ref: str) -> str:
    """Resolve ``ref`` to its full SHA in ``repo`` (``""`` if it doesn't resolve)."""
    r = _git(repo, "rev-parse", ref)
    return r.stdout.strip() if r.returncode == 0 else ""


def update_ref_cas(ref: str, new_sha: str, expected: str, repo: str) -> bool:
    """Compare-and-swap ``ref`` from ``expected`` → ``new_sha`` atomically.

    ``git update-ref <ref> <new> <old>`` only moves the ref if it currently
    points at ``expected``; a concurrent advance since we measured makes it fail.
    Returns ``True`` on success, ``False`` if the CAS was rejected (stale) or the
    command errored.
    """
    r = _git(repo, "update-ref", ref, new_sha, expected)
    return r.returncode == 0


def show_file(repo: str, sha: str, path: str) -> bytes:
    """Return the bytes of ``path`` as of commit ``sha`` (``git show SHA:path``).

    Raises ``FileNotFoundError`` if the blob doesn't exist at that revision.
    """
    r = subprocess.run(
        ["git", "-C", repo, "show", f"{sha}:{path}"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise FileNotFoundError(f"{sha}:{path} not in {repo}: {r.stderr.decode(errors='ignore').strip()}")
    return r.stdout


# ── GPU lock (flock on a held-open fd; kernel releases on holder death) ───────


@contextlib.contextmanager
def gpu_lock(lockfile: str):
    """Hold an exclusive ``flock`` on ``lockfile`` for the ``with`` body.

    The lock is held on an open fd, so if this process dies the kernel drops it.
    The file is created if missing and NEVER unlinked. Diagnostics
    (``agent pid=… host=… acquired=…``) are stamped into it for observers.
    """
    fd = os.open(lockfile, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        info = (
            f"agent pid={os.getpid()} host={socket.gethostname()} "
            f"acquired={datetime.datetime.now().isoformat()}\n"
        )
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, info.encode())
        except OSError:
            pass  # diagnostics are best-effort; the lock itself is what matters
        yield lockfile
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)  # NB: never os.unlink(lockfile) — see module docstring


# ── worktree lifecycle (per-worker checkout anchors) ─────────────────────────


def worktree_add(repo: str, path: str, ref: str) -> bool:
    """``git worktree add <path> <ref>`` — a per-worker checkout anchor."""
    return _git(repo, "worktree", "add", path, ref).returncode == 0


def worktree_list(repo: str) -> list[dict]:
    """Parsed ``git worktree list --porcelain`` → ``[{worktree, HEAD, branch}, …]``."""
    r = _git(repo, "worktree", "list", "--porcelain")
    if r.returncode != 0:
        return []
    trees: list[dict] = []
    cur: dict = {}
    for line in r.stdout.splitlines():
        if not line.strip():
            if cur:
                trees.append(cur)
                cur = {}
            continue
        key, _, val = line.partition(" ")
        cur[key] = val
    if cur:
        trees.append(cur)
    return trees


def worktree_remove(repo: str, path: str) -> bool:
    """``git worktree remove <path>`` (best-effort cleanup of a worker anchor)."""
    return _git(repo, "worktree", "remove", path).returncode == 0

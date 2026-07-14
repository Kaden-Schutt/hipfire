# Copyright (c) Kaden Schutt
"""ar.gate.build — build a ref's daemon binary on-box, cached by sha.

The gate's parity/perf/coherence A/B needs the ACTUAL base-ref and head-ref daemon
binaries, not opaque handles — that was the scaffold bug (``subprocess.run("base")``).
This builds one daemon per ref via the SAME cargo invocation the loop's
``ab_certify_v2p.sh`` uses, caches the binary keyed by the ref's resolved sha, and
returns its path. The binary is **arch-agnostic** (kernels JIT per arch at runtime),
so one build per sha serves every arch on the box — the base (master tip) is built
once and reused across every PR/arch cell.

Pure orchestration over injectable ``run_git`` / ``run_cmd`` seams so the caching and
control flow are unit-testable without cargo or a real repo.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from .merge import default_run_git

# The exact build the loop uses (ab_certify_v2p.sh:71 / rollover_v2.sh:71).
CARGO_BUILD = ["cargo", "build", "--release", "--features", "deltanet",
               "-p", "hipfire-runtime", "--example", "daemon"]
DAEMON_REL = "target/release/examples/daemon"


def _default_run_cmd(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def build_daemon(ref, repo, *, cache_dir="/tmp", run_git=None, run_cmd=None) -> str:
    """Build the daemon at ``ref`` and return its cached binary path.

    Caches at ``<cache_dir>/gate_daemon_<sha>`` — a re-run at the same sha skips the
    build entirely. Builds in a throwaway git worktree at the sha (so the live
    checkout is untouched), copies the binary out, then removes the worktree. Raises
    ``RuntimeError`` if the ref can't be resolved or the build fails.
    """
    run_git = run_git or default_run_git
    run_cmd = run_cmd or _default_run_cmd

    code, out = run_git(repo, "rev-parse", ref)
    sha = (out or "").strip()
    if code != 0 or not sha:
        raise RuntimeError(f"gate build: cannot resolve ref {ref!r} ({(out or '').strip()})")

    cached = os.path.join(cache_dir, f"gate_daemon_{sha}")
    if os.path.exists(cached):
        return cached

    wt = os.path.join(cache_dir, f"gate_wt_{sha}")
    # A worktree may linger from a killed run; force-replace it.
    run_git(repo, "worktree", "remove", "--force", wt)
    code, out = run_git(repo, "worktree", "add", "--force", "--detach", wt, sha)
    if code != 0:
        raise RuntimeError(f"gate build: worktree add failed for {sha}: {(out or '').strip()}")
    try:
        r = run_cmd(CARGO_BUILD, wt)
        src = os.path.join(wt, DAEMON_REL)
        if getattr(r, "returncode", 1) != 0 or not os.path.exists(src):
            tail = (getattr(r, "stderr", "") or "")[-2000:]
            raise RuntimeError(f"gate build: daemon build failed at {sha}: {tail}")
        # Copy to the sha-cache; the worktree is disposable.
        shutil.copy2(src, cached)
    finally:
        run_git(repo, "worktree", "remove", "--force", wt)
    return cached

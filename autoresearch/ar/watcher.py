# Copyright (c) Kaden Schutt
"""ar.watcher — the guardrailed watcher daemon (run tracking + auto-enforce).

One persistent process per box owns run lifecycle and **auto-enforces**
fold/rollover under leashes. It is the supervisor seed of
``autoresearch/ar/hipfire_ar.py`` (``cmd_start/stop/status`` run table +
budget/TTL bounds) and the enforcement half of ``v2/rollover_v2.sh`` (fold a
certified WIN into the shared baseline, then re-census + re-ingest), converged
onto ``ar.db``'s ``runs`` table and the death-safe ``update-ref`` CAS in
:mod:`autoresearch.ar.gitpilot`.

The whole point is the guardrails:

1. **Dry-run first.** Every enforced mutation is previewable (``dry_run=True``
   returns the planned action, mutates nothing).
2. **Git-reversible.** Each fold/rollover records the *prior* SHA so the ref can
   be rolled straight back (the CAS advance is ``git update-ref <ref> <new>
   <prior>`` — the prior is exactly the rollback target).
3. **Leashed.** A run past its ``call_budget`` or ``wall_ttl_s`` is auto-stopped;
   a dead pid is reaped. Leashes are mechanical, not prompt-discipline.

**Human gates preserved:** there is deliberately NO ``push_master`` and NO
default-flip method on :class:`Watcher`. Master-push and default flips are staged
+ notified by a human — the absence is asserted by the test suite, so do not add
one here.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Callable, Optional

from . import gitpilot

# Run-table columns (mirrors autoresearch/db/schema.sql `runs`).
_RUN_COLS = ("id", "arch", "model", "card", "status", "budget", "calls", "ttl", "pid", "ts")


def _local_pid_alive(pid: Optional[int]) -> bool:
    """Local liveness probe: ``kill -0``. Unknown (no pid) => not reapable."""
    if not pid:
        return True  # no pid yet => can't call it dead; leave it to the budget/TTL leash
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours => alive
    except (OSError, ValueError):
        return False


class RunStore:
    """The ``runs`` table of ``ar.db`` as a small, injectable interface.

    This is the ``db.runs`` seam the :class:`Watcher` consumes. Backed by a real
    (possibly ``:memory:``) sqlite connection so tests exercise the actual store
    rather than a diverging mock. Rows come back as plain ``dict``.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_run(
        self,
        id: str,
        arch: Optional[str] = None,
        model: Optional[str] = None,
        card: Optional[int] = None,
        status: str = "running",
        budget: int = 0,
        calls: int = 0,
        ttl: int = 0,
        pid: Optional[int] = None,
        ts: Optional[int] = None,
    ) -> None:
        ts = int(time.time()) if ts is None else int(ts)
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(id,arch,model,card,status,budget,calls,ttl,pid,ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (id, arch, model, card, status, budget, calls, ttl, pid, ts),
        )
        self.conn.commit()

    def get_run(self, id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM runs WHERE id=?", (id,)).fetchone()
        return dict(row) if row is not None else None

    def running_runs(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM runs WHERE status='running'").fetchall()
        return [dict(r) for r in rows]

    def set_status(self, id: str, status: str) -> None:
        self.conn.execute("UPDATE runs SET status=? WHERE id=?", (status, id))
        self.conn.commit()

    def bump_calls(self, id: str, n: int = 1) -> None:
        self.conn.execute("UPDATE runs SET calls=calls+? WHERE id=?", (n, id))
        self.conn.commit()


class Watcher:
    """Tracks runs, reaps dead pids, auto-stops over-budget/TTL runs, and folds
    certified WINs into the shared baseline — every mutation dry-run-loggable and
    git-reversible.

    No ``push_master`` / default-flip method exists (human gate).
    """

    def __init__(
        self,
        db: RunStore,
        repo: str,
        *,
        recensus: Optional[Callable[[str], None]] = None,
        pid_alive: Optional[Callable[[Optional[int]], bool]] = None,
        clock: Callable[[], float] = time.time,
        log: Optional[list] = None,
    ):
        self.db = db
        self.repo = repo
        self._recensus = recensus
        self._pid_alive = pid_alive or _local_pid_alive
        self._clock = clock
        self.log: list[dict] = log if log is not None else []

    # ── run tracking ─────────────────────────────────────────────────────────

    def track(self, run: dict) -> None:
        """Register a run row (defaults status=running)."""
        fields = {k: run.get(k) for k in _RUN_COLS if k in run}
        fields.setdefault("status", "running")
        self.db.add_run(**fields)

    def reap(self) -> list[str]:
        """Auto-stop runs whose recorded pid is no longer alive. Returns reaped ids."""
        reaped: list[str] = []
        for r in self.db.running_runs():
            pid = r.get("pid")
            if pid and not self._pid_alive(pid):
                self.db.set_status(r["id"], "stopped")
                self._record("reap", run=r["id"], reason="dead_pid", pid=pid, applied=True)
                reaped.append(r["id"])
        return reaped

    def enforce(self) -> dict:
        """One enforcement sweep: reap dead pids + auto-stop over-budget/TTL runs.

        Returns a summary ``{"reaped": [...], "stopped": [...]}`` — ``stopped``
        includes leash-stops (budget/TTL). Every stop is recorded in ``self.log``.
        """
        reaped = self.reap()
        stopped: list[str] = list(reaped)
        now = self._clock()
        for r in self.db.running_runs():
            reason = self._leash_reason(r, now)
            if reason:
                self.db.set_status(r["id"], "stopped")
                self._record("leash_stop", run=r["id"], reason=reason, applied=True)
                stopped.append(r["id"])
        return {"reaped": reaped, "stopped": stopped}

    @staticmethod
    def _leash_reason(run: dict, now: float) -> Optional[str]:
        budget = run.get("budget") or 0
        calls = run.get("calls") or 0
        if budget and calls >= budget:
            return "over_budget"
        ttl = run.get("ttl") or 0
        ts = run.get("ts") or 0
        if ttl and now > ts + ttl:
            return "over_ttl"
        return None

    # ── guardrailed enforcement: fold + rollover ─────────────────────────────

    def enforce_fold(self, ref: str, win_sha: str, *, dry_run: bool = False) -> dict:
        """Fold a certified WIN into the shared baseline via ``update-ref`` CAS.

        Reversibility: the *prior* SHA of ``ref`` is read and recorded before the
        advance; it is both the CAS ``expected`` guard AND the rollback target.
        ``dry_run=True`` records the planned action and mutates nothing.

        Returns the audit entry ``{action, ref, prior_sha, new_sha, dry_run,
        applied}``. ``applied`` is ``False`` on dry-run, on an unresolved ref, or
        on a lost CAS (concurrent advance since we read ``prior_sha``).
        """
        full_ref = self._full_ref(ref)
        prior_sha = gitpilot.current_sha(self.repo, ref)
        entry = {
            "action": "fold",
            "ref": ref,
            "prior_sha": prior_sha,
            "new_sha": win_sha,
            "dry_run": dry_run,
            "applied": False,
            "ts": int(self._clock()),
        }
        if dry_run or not prior_sha or not full_ref:
            self.log.append(entry)
            return entry
        entry["applied"] = gitpilot.update_ref_cas(full_ref, win_sha, prior_sha, self.repo)
        self.log.append(entry)
        return entry

    def enforce_rollover(self, *, reason: str = "advance", dry_run: bool = False) -> dict:
        """Re-census the BOD + re-ingest after a baseline advance or exhaustion.

        The GPU-touching census/ingest is injected (``recensus`` callback) so this
        stays no-GPU-unit-testable. ``dry_run=True`` records the planned rollover
        and does NOT invoke the callback.
        """
        entry = {
            "action": "rollover",
            "reason": reason,
            "dry_run": dry_run,
            "applied": False,
            "ts": int(self._clock()),
        }
        if dry_run:
            self.log.append(entry)
            return entry
        if self._recensus is not None:
            self._recensus(reason)
        entry["applied"] = True
        self.log.append(entry)
        return entry

    def _full_ref(self, ref: str) -> str:
        """Resolve a possibly-short ref to its fully-qualified name.

        ``update-ref`` needs ``refs/heads/loop/gfx1201``, not the short
        ``loop/gfx1201`` (a partial name would write a stray top-level ref rather
        than move the branch). Already-qualified ``refs/...`` names pass through.
        """
        if ref.startswith("refs/"):
            return ref
        r = gitpilot._git(self.repo, "rev-parse", "--symbolic-full-name", ref)
        return r.stdout.strip() if r.returncode == 0 else ""

    # ── audit log ────────────────────────────────────────────────────────────

    def _record(self, action: str, **fields) -> dict:
        entry = {"action": action, "ts": int(self._clock()), **fields}
        self.log.append(entry)
        return entry

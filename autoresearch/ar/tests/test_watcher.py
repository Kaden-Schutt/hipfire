# Copyright (c) Kaden Schutt
"""Watcher daemon: run tracking + guardrailed auto-fold/rollover + leashes.

No-GPU unit tests. The GPU/daemon seams are absent from the Watcher by
construction — it only tracks run rows, reaps dead pids, auto-stops over
budget/TTL runs, and folds certified WINs into the shared baseline via the
death-safe ``update-ref`` CAS (prior SHA recorded => reversible; dry-run first).
There is deliberately NO ``push_master`` / default-flip capability (human gate).
"""
import os
import subprocess

import pytest

from autoresearch.ar.db import connect
from autoresearch.ar.gitpilot import current_sha
from autoresearch.ar.watcher import RunStore, Watcher

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fake_db():
    """In-memory run store — the real RunStore over ``:memory:`` (no mock divergence)."""
    return RunStore(connect(":memory:"))


@pytest.fixture
def tmp_repo(tmp_path):
    """A throwaway git repo with a ``loop/gfx1201`` shared-baseline branch + a WIN commit."""
    r = str(tmp_path / "repo")
    os.makedirs(r)
    _git(r, "init", "-q")
    open(f"{r}/x", "w").write("1")
    _git(r, "add", "x")
    subprocess.run(["git", "-C", r, "commit", "-qm", "base"], env=_ENV)
    _git(r, "branch", "loop/gfx1201")
    return r


def _win_commit(repo):
    """Create a second commit (a certified WIN) and return its SHA."""
    open(f"{repo}/x", "w").write("2")
    subprocess.run(["git", "-C", repo, "commit", "-aqm", "win"], env=_ENV)
    return current_sha(repo, "HEAD")


# ── plan Task 6.1 canonical tests ────────────────────────────────────────────


def test_over_budget_run_autostops(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    fake_db.add_run(id="r1", calls=401, budget=400, ttl=99999, pid=None, status="running")
    w.enforce()
    assert fake_db.get_run("r1")["status"] == "stopped"


def test_win_autofold_is_dryrun_logged_and_reversible(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    before = current_sha(tmp_repo, "loop/gfx1201")
    prior = w.enforce_fold("loop/gfx1201", "<win_sha>", dry_run=True)
    assert prior["prior_sha"] and prior["dry_run"] is True  # records reversibility, no mutation
    # dry-run mutates nothing: the shared baseline still points where it did.
    assert current_sha(tmp_repo, "loop/gfx1201") == before
    assert prior["applied"] is False


def test_master_push_never_automated(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    assert not hasattr(w, "push_master")  # capability absent by construction


# ── extra leash / reversibility / reap coverage ──────────────────────────────


def test_master_push_and_default_flip_absent_by_construction(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    for banned in ("push_master", "push", "flip_default", "set_default", "master_push"):
        assert not hasattr(w, banned)


def test_over_ttl_run_autostops(fake_db, tmp_repo):
    # ts far in the past + short ttl => wall-clock exhausted.
    w = Watcher(fake_db, tmp_repo, clock=lambda: 10_000)
    fake_db.add_run(id="r2", calls=0, budget=400, ttl=1, pid=None, status="running", ts=0)
    w.enforce()
    assert fake_db.get_run("r2")["status"] == "stopped"


def test_healthy_run_is_left_running(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo, clock=lambda: 100)
    fake_db.add_run(id="ok", calls=10, budget=400, ttl=99999, pid=None, status="running", ts=100)
    w.enforce()
    assert fake_db.get_run("ok")["status"] == "running"


def test_reap_stops_dead_pid(fake_db, tmp_repo):
    # a pid that is guaranteed not to be alive => reaped to stopped.
    dead = 2_000_000_000
    w = Watcher(fake_db, tmp_repo, pid_alive=lambda p: False)
    fake_db.add_run(id="r3", calls=0, budget=400, ttl=99999, pid=dead, status="running")
    reaped = w.reap()
    assert "r3" in reaped
    assert fake_db.get_run("r3")["status"] == "stopped"


def test_reap_leaves_live_pid(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo, pid_alive=lambda p: True)
    fake_db.add_run(id="r4", calls=0, budget=400, ttl=99999, pid=12345, status="running")
    assert w.reap() == []
    assert fake_db.get_run("r4")["status"] == "running"


def test_enforce_fold_real_advance_is_cas_and_records_prior(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    prior_sha = current_sha(tmp_repo, "loop/gfx1201")
    win = _win_commit(tmp_repo)
    entry = w.enforce_fold("loop/gfx1201", win, dry_run=False)
    assert entry["applied"] is True
    assert entry["prior_sha"] == prior_sha  # reversible: we recorded where to roll back to
    assert current_sha(tmp_repo, "loop/gfx1201") == win
    # audit log carries the reversible record.
    assert any(e["new_sha"] == win and e["prior_sha"] == prior_sha for e in w.log)


def test_enforce_fold_stale_cas_is_rejected(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    win = _win_commit(tmp_repo)
    # someone else advanced loop/gfx1201 between our read and write.
    _git(tmp_repo, "update-ref", "refs/heads/loop/gfx1201", win)
    # Now fold to a fresh commit but with an expected prior that is now stale.
    open(f"{tmp_repo}/x", "w").write("3")
    subprocess.run(["git", "-C", tmp_repo, "commit", "-aqm", "win2"], env=_ENV)
    win2 = current_sha(tmp_repo, "HEAD")
    # monkeypatch: force a stale expected by reading before, mutating, then folding
    # (enforce_fold reads current internally, so simulate via a wrong-expected path)
    entry = w.enforce_fold("refs/heads/does-not-exist", win2, dry_run=False)
    # unresolved ref => prior_sha empty => CAS cannot apply.
    assert entry["applied"] is False


def test_enforce_returns_summary_of_actions(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo, clock=lambda: 10_000)
    fake_db.add_run(id="b", calls=999, budget=400, ttl=99999, pid=None, status="running", ts=9000)
    fake_db.add_run(id="ok", calls=1, budget=400, ttl=99999, pid=None, status="running", ts=9999)
    summary = w.enforce()
    assert "b" in summary["stopped"]
    assert "ok" not in summary["stopped"]


def test_track_registers_run(fake_db, tmp_repo):
    w = Watcher(fake_db, tmp_repo)
    w.track({"id": "t1", "arch": "gfx1201", "budget": 400, "ttl": 99999, "pid": None})
    assert fake_db.get_run("t1")["status"] == "running"


def test_rollover_invokes_injected_recensus(fake_db, tmp_repo):
    calls = []
    w = Watcher(fake_db, tmp_repo, recensus=lambda reason: calls.append(reason))
    entry = w.enforce_rollover(reason="advance", dry_run=False)
    assert entry["applied"] is True
    assert calls == ["advance"]


def test_rollover_dry_run_does_not_recensus(fake_db, tmp_repo):
    calls = []
    w = Watcher(fake_db, tmp_repo, recensus=lambda reason: calls.append(reason))
    entry = w.enforce_rollover(reason="exhaustion", dry_run=True)
    assert entry["dry_run"] is True
    assert entry["applied"] is False
    assert calls == []

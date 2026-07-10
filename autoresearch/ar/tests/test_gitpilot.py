# Copyright (c) Kaden Schutt
import os
import subprocess

from autoresearch.ar.gitpilot import current_sha, gpu_lock, show_file, update_ref_cas

_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


def _mkrepo(tmp):
    r = str(tmp)
    _git(r, "init", "-q")
    open(f"{r}/x", "w").write("1")
    _git(r, "add", "x")
    subprocess.run(["git", "-C", r, "commit", "-qm", "a"], env=_ENV)
    return r


def test_cas_succeeds_when_expected_matches(tmp_path):
    r = _mkrepo(tmp_path)
    a = current_sha(r, "HEAD")
    open(f"{r}/x", "w").write("2")
    subprocess.run(["git", "-C", r, "commit", "-aqm", "b"], env=_ENV)
    b = current_sha(r, "HEAD")
    _git(r, "branch", "base", a)
    assert update_ref_cas("refs/heads/base", b, a, r) is True
    assert current_sha(r, "refs/heads/base") == b


def test_cas_fails_when_stale(tmp_path):
    r = _mkrepo(tmp_path)
    a = current_sha(r, "HEAD")
    _git(r, "branch", "base", a)
    assert update_ref_cas("refs/heads/base", a, "0" * 40, r) is False


def test_show_file_reads_blob_at_sha(tmp_path):
    r = _mkrepo(tmp_path)
    a = current_sha(r, "HEAD")
    assert show_file(r, a, "x") == b"1"


def test_gpu_lock_acquires_releases_and_never_unlinks(tmp_path):
    lf = str(tmp_path / "gpu.lock")
    with gpu_lock(lf) as held:
        assert held == lf
        assert os.path.exists(lf)  # created
        assert "pid=" in open(lf).read()  # diagnostics stamped
    assert os.path.exists(lf)  # NEVER unlinked on release
    # re-acquirable after release (the flock was dropped, fd closed)
    with gpu_lock(lf):
        pass

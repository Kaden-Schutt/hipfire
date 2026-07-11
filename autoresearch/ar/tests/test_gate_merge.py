# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.merge import trial_merge


def _git_clean(repo, *args):
    # `git merge-tree --write-tree ...` clean: rc 0, tree OID on line 1
    return (0, "a1b2c3d4e5f6\n")


def _git_conflict(repo, *args):
    # rc 1, tree OID line, blank line, then conflicted paths (--name-only)
    return (1, "deadbeef\n\ncrates/hipfire-runtime/examples/daemon.rs\n")


def test_clean_merge():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_clean)
    assert r["clean"] is True
    assert r["merged_tree"] == "a1b2c3d4e5f6"
    assert r["conflicts"] == []


def test_conflicted_merge_lists_paths():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_conflict)
    assert r["clean"] is False
    assert r["conflicts"] == ["crates/hipfire-runtime/examples/daemon.rs"]


def test_passes_refs_to_git():
    seen = {}

    def spy(repo, *args):
        seen["repo"] = repo
        seen["args"] = args
        return (0, "abc\n")

    trial_merge("staging", "pr", "/repo", run_git=spy)
    assert seen["repo"] == "/repo"
    assert "merge-tree" in seen["args"]
    assert seen["args"][-2:] == ("staging", "pr")


from autoresearch.ar.gate.merge import assemble_bod


def test_bod_collects_all_kinds():
    bod = assemble_bod(
        conflicts=["daemon.rs"],
        perf_regressions=["perf_regression"],
        coherence_fails=["coherence"],
    )
    kinds = [b["kind"] for b in bod["blockers"]]
    assert kinds == ["merge_conflict", "perf_regression", "coherence"]
    assert bod["blockers"][0]["detail"] == "daemon.rs"
    assert "3" in bod["summary"]


def test_bod_empty_is_clean():
    bod = assemble_bod()
    assert bod["blockers"] == []
    assert bod["summary"] == "no blockers"

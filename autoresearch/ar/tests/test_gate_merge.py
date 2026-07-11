# Copyright (c) Kaden Schutt
from autoresearch.ar.gate.merge import trial_merge


def _git_clean(repo, *args):
    # `git merge-tree --write-tree ...` clean: rc 0, tree OID on line 1
    return (0, "a1b2c3d4e5f6\n")


def _git_conflict(repo, *args):
    # REAL `git merge-tree --write-tree --name-only` conflict layout (verified
    # against live git): <OID>\n<conflicted path>*\n\n<freeform informational prose>.
    # The conflicted paths come BEFORE the blank; Auto-merging/CONFLICT lines after.
    return (1, "deadbeef\ncrates/hipfire-runtime/examples/daemon.rs\n\n"
               "Auto-merging crates/hipfire-runtime/examples/daemon.rs\n"
               "CONFLICT (content): Merge conflict in crates/hipfire-runtime/examples/daemon.rs\n")


def test_clean_merge():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_clean)
    assert r["clean"] is True
    assert r["merged_tree"] == "a1b2c3d4e5f6"
    assert r["conflicts"] == []


def test_conflicted_merge_lists_paths():
    r = trial_merge("staging", "pr", "/repo", run_git=_git_conflict)
    assert r["clean"] is False
    assert r["conflicts"] == ["crates/hipfire-runtime/examples/daemon.rs"]


def test_conflict_excludes_informational_prose_and_handles_multiple():
    # Two conflicted paths before the blank; Auto-merging/CONFLICT prose after it
    # must NOT be captured as paths (the defect the shared-assumption mock hid).
    def two(repo, *args):
        return (1, "oid\ncrates/a.rs\ncrates/b.rs\n\n"
                   "Auto-merging crates/a.rs\nCONFLICT (content): Merge conflict in crates/a.rs\n")

    r = trial_merge("staging", "pr", "/repo", run_git=two)
    assert r["conflicts"] == ["crates/a.rs", "crates/b.rs"]


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


from autoresearch.ar.gate.merge import gate4


def _clean_tm(*a, **k):
    return {"clean": True, "merged_tree": "t", "conflicts": []}


def _conflict_tm(*a, **k):
    return {"clean": False, "merged_tree": "t", "conflicts": ["daemon.rs"]}


def _gate(verdict, reasons=()):
    return lambda: {"verdict": verdict, "reasons": list(reasons)}


def test_clean_merge_clean_gate_passes():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"), trial_merge_fn=_clean_tm)
    assert r["verdict"] == "PASS" and r["bod"] is None


def test_conflict_no_fixer_is_bod():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"), merge_fix=None, trial_merge_fn=_conflict_tm)
    assert r["verdict"] == "BOD"
    assert r["bod"]["blockers"][0]["kind"] == "merge_conflict"


def test_conflict_fixed_then_passes():
    # trial-merge conflicts first, then (after fix) is clean
    seq = [_conflict_tm(), _clean_tm()]
    tm = lambda *a, **k: seq.pop(0)
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("PASS"),
              merge_fix=lambda kind, detail: {"fixed": True}, trial_merge_fn=tm)
    assert r["verdict"] == "PASS"


def test_post_merge_clobber_no_fixer_is_bod_partitioned():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("REJECT", ["perf_regression", "coherence"]),
              merge_fix=None, trial_merge_fn=_clean_tm)
    assert r["verdict"] == "BOD"
    kinds = sorted(b["kind"] for b in r["bod"]["blockers"])
    assert kinds == ["coherence", "perf_regression"]


def test_post_merge_clobber_fixed_then_passes():
    gates = [{"verdict": "REJECT", "reasons": ["perf_regression"]},
             {"verdict": "PASS", "reasons": []}]
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=lambda: gates.pop(0),
              merge_fix=lambda kind, detail: {"fixed": True}, trial_merge_fn=_clean_tm)
    assert r["verdict"] == "PASS"


def test_post_merge_clobber_parity_is_itemized_not_empty():
    # A parity-only clobber must produce a POPULATED BOD (the defect: parity/
    # cross_arch reasons were dropped, yielding verdict=BOD + "no blockers").
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("REJECT", ["parity"]),
              merge_fix=None, trial_merge_fn=_clean_tm)
    assert r["verdict"] == "BOD"
    assert r["bod"]["blockers"] == [{"kind": "parity", "detail": "parity"}]
    assert r["bod"]["summary"] != "no blockers"


def test_post_merge_clobber_cross_arch_is_itemized():
    r = gate4(base_ref="m", head_ref="pr", staging_ref="staging", repo="/repo",
              run_merged_gate=_gate("REJECT", ["cross_arch"]),
              merge_fix=None, trial_merge_fn=_clean_tm)
    assert [b["kind"] for b in r["bod"]["blockers"]] == ["cross_arch"]

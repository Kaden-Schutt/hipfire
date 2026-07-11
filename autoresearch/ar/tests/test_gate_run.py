# Copyright (c) Kaden Schutt
import os
import tempfile

from autoresearch.ar.gate.config import load_gate_config
from autoresearch.ar.gate.run import changed_files, diff_lines, run_pr_gate, stub_arch_gate

_TOML = """
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "Kaden-Schutt"]
auto_merge_authors = ["Kaden-Schutt"]
floor = 0.15
drift_pct = 3.0
alpha = 0.05

[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]
"qwen3.6-a3b" = ["gfx1100", "gfx1151", "gfx1201"]

[routing]
trivial = { harness = "none", model = "", effort = "" }
low = { harness = "codex", model = "gpt-5.6-luna", effort = "high" }
moderate = { harness = "codex", model = "gpt-5.6-terra", effort = "high" }
"high-risk" = { harness = "codex", model = "gpt-5.6-sol", effort = "xhigh" }
"""


def _cfg():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pr_gate.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    return load_gate_config(p)


def _git(name_only="", numstat=""):
    def g(repo, *args):
        if "--name-only" in args:
            return (0, name_only)
        if "--numstat" in args:
            return (0, numstat)
        return (0, "")
    return g


def test_changed_files_parses_name_only():
    g = _git(name_only="kernels/src/a.hip\ncrates/x.rs\n")
    assert changed_files("m", "pr", "/r", run_git=g) == ["kernels/src/a.hip", "crates/x.rs"]


def test_diff_lines_sums_numstat_and_ignores_binary():
    g = _git(numstat="10\t2\tkernels/src/a.hip\n5\t0\tcrates/x.rs\n-\t-\tbin.png\n")
    assert diff_lines("m", "pr", "/r", run_git=g) == 17


def test_run_pr_gate_kernel_pr_kaden_auto_merges():
    g = _git(name_only="kernels/src/a.hip\n", numstat="10\t2\tkernels/src/a.hip\n")
    r = run_pr_gate(base="m", head="pr", repo="/r", author="Kaden-Schutt", is_draft=False,
                    helpful=True, cfg=_cfg(), arch_gate_fn=stub_arch_gate, run_git=g)
    assert r["pr_class"] == "high-risk"
    assert r["route"]["model"] == "gpt-5.6-sol" and r["route"]["effort"] == "xhigh"
    assert r["outcome"]["action"] == "auto_merge"
    assert [a["arch"] for a in r["arch_results"]] == ["gfx1100", "gfx1151", "gfx1201"]
    assert "gfx1201" in r["comment"]


def test_run_pr_gate_regressing_arch_is_bod():
    g = _git(name_only="kernels/src/a.hip\n", numstat="10\t2\tkernels/src/a.hip\n")

    def gate(arch, files, base, head, repo, cfg):
        v = "REJECT" if arch == "gfx1151" else "PASS"
        return stub_arch_gate(arch, files, base, head, repo, cfg,
                              verdict=v, reasons=["perf_regression"] if v == "REJECT" else None)

    r = run_pr_gate(base="m", head="pr", repo="/r", author="Kaden-Schutt", is_draft=False,
                    helpful=True, cfg=_cfg(), arch_gate_fn=gate, run_git=g)
    assert r["outcome"]["action"] == "bod" and r["outcome"]["status"] == "failure"
    assert any(b["arch"] == "gfx1151" for b in r["outcome"]["bod"]["blockers"])


def test_interpret_folds_failed_behavior_as_bod(tmp_path):
    import json as _j

    from autoresearch.ar.gate.run import interpret_results

    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "gfx1201.json").write_text(
        _j.dumps({"arch": "gfx1201", "verdict": "PASS", "reasons": [], "bod": None}))
    g = _git(name_only="cli/index.ts\n", numstat="5\t0\tcli/index.ts\n")
    # serve_harness floor is green, but a bespoke behavior test failed -> BOD (spec §8).
    res = interpret_results(results_dir=str(rd), base="m", head="pr", repo="/r",
                            author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg(),
                            run_git=g, behavior_results=[{"what": "cli --foo", "passed": False}])
    assert res["outcome"]["action"] == "bod"
    assert any("behavior:cli --foo" in str(b) for b in res["outcome"]["bod"]["blockers"])


def test_run_pr_gate_docs_only_is_trivial_and_tags_non_kaden():
    g = _git(name_only="docs/x.md\n", numstat="3\t0\tdocs/x.md\n")
    r = run_pr_gate(base="m", head="pr", repo="/r", author="fivetide", is_draft=False,
                    helpful=True, cfg=_cfg(), arch_gate_fn=stub_arch_gate, run_git=g)
    assert r["pr_class"] == "trivial"
    assert r["route"]["harness"] == "none"
    assert r["outcome"]["action"] == "tag_maintainer"

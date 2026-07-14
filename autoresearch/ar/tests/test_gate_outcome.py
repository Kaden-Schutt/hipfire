# Copyright (c) Kaden Schutt
import os
import tempfile

from autoresearch.ar.gate.config import load_gate_config
from autoresearch.ar.gate.outcome import decide_pr, format_pr_comment

_TOML = """
archs = ["gfx1100", "gfx1151", "gfx1201"]
canonical_models = ["qwen3.6-27b", "qwen3.6-a3b"]
maintainers = ["fivetide", "unverbraucht", "nwoolmer", "Kaden-Schutt"]
auto_merge_authors = ["Kaden-Schutt"]
floor = 0.15
drift_pct = 3.0
alpha = 0.05

[fit]
"qwen3.6-27b" = ["gfx1100", "gfx1151", "gfx1201"]

[routing]
"high-risk" = { harness = "codex", model = "gpt-5.6-sol", effort = "xhigh" }
"""


def _cfg():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "pr_gate.toml")
    with open(p, "w") as fh:
        fh.write(_TOML)
    return load_gate_config(p)


_PASS = [{"arch": "gfx1201", "verdict": "PASS", "reasons": [], "bod": None},
         {"arch": "gfx1100", "verdict": "PASS", "reasons": [], "bod": None}]


def test_kaden_clean_helpful_auto_merges():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "auto_merge" and o["status"] == "success"


def test_other_maintainer_clean_helpful_is_tagged():
    o = decide_pr(arch_results=_PASS, author="fivetide", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "tag_maintainer" and o["status"] == "success"


def test_non_maintainer_clean_is_tagged_for_maintainer_merge():
    o = decide_pr(arch_results=_PASS, author="randocontrib", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "tag_maintainer"


def test_draft_never_merges():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=True, helpful=True, cfg=_cfg())
    assert o["action"] == "draft_report" and o["status"] == "success"


def test_not_helpful_is_neutral():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=False, cfg=_cfg())
    assert o["action"] == "neutral"


def test_any_arch_reject_is_bod_failure():
    mixed = _PASS + [{"arch": "gfx1151", "verdict": "REJECT", "reasons": ["perf_regression"], "bod": None}]
    o = decide_pr(arch_results=mixed, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "bod" and o["status"] == "failure"
    assert any(b["detail"] == "perf_regression" for b in o["bod"]["blockers"])


def test_empty_or_unknown_evidence_is_failure_not_green():
    empty = decide_pr(arch_results=[], author="Kaden-Schutt", is_draft=False,
                      helpful=True, cfg=_cfg())
    assert empty["action"] == "bod" and empty["status"] == "failure"
    unknown = decide_pr(
        arch_results=[{"arch": "gfx1201", "verdict": "MAYBE", "reasons": []}],
        author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg(),
    )
    assert unknown["action"] == "bod" and unknown["status"] == "failure"
    assert unknown["bod"]["blockers"][0]["kind"] == "invalid_verdict"


def test_arch_bod_blockers_are_aggregated():
    b = {"blockers": [{"kind": "merge_conflict", "detail": "daemon.rs"}], "summary": "1 blocker(s)"}
    mixed = _PASS + [{"arch": "gfx1201", "verdict": "BOD", "reasons": [], "bod": b}]
    o = decide_pr(arch_results=mixed, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    assert o["action"] == "bod"
    assert {"kind": "merge_conflict", "detail": "daemon.rs", "arch": "gfx1201"} in o["bod"]["blockers"]


def test_comment_renders_verdict_and_table():
    o = decide_pr(arch_results=_PASS, author="Kaden-Schutt", is_draft=False, helpful=True, cfg=_cfg())
    md = format_pr_comment(o, _PASS)
    assert "auto_merge" in md.lower() or "merge" in md.lower()
    assert "gfx1201" in md and "gfx1100" in md

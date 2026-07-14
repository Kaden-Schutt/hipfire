# Copyright (c) Kaden Schutt
"""ar.gate.outcome — the interpreter's deterministic PR decision (spec §12).

decide_pr reduces the per-arch gate verdicts + author + draft-state + Claude's
helpfulness judgment into a single PR action. This is the mechanical skeleton;
Claude (Tier 2) wraps it with the human-facing prose + the actual merge/tag call.
"""
from __future__ import annotations


def _aggregate_bod(arch_results) -> dict:
    """Merge every failing arch's blockers into one BOD, tagging each by arch."""
    blockers: list[dict] = []
    for r in arch_results:
        if r["verdict"] == "BOD" and r.get("bod"):
            for b in r["bod"].get("blockers", []):
                blockers.append({**b, "arch": r["arch"]})
        elif r["verdict"] == "REJECT":
            for reason in r.get("reasons", []):
                blockers.append({"kind": reason, "detail": reason, "arch": r["arch"]})
        elif r["verdict"] != "PASS":
            blockers.append({"kind": "invalid_verdict",
                             "detail": f"unexpected verdict {r['verdict']!r}",
                             "arch": r["arch"]})
    summary = f"{len(blockers)} blocker(s) across "
    summary += ", ".join(sorted({r["arch"] for r in arch_results
                                 if r.get("verdict") != "PASS"}))
    return {"blockers": blockers, "summary": summary}


def decide_pr(*, arch_results, author, is_draft, helpful, cfg) -> dict:
    """Decide the PR action from the per-arch verdicts + authority + helpfulness."""
    arch_results = list(arch_results)
    if not arch_results:
        arch_results.append({"arch": "evidence", "verdict": "REJECT",
                             "reasons": ["missing_results"], "bod": None})
    # PASS is the only green value. New, malformed, or agent-invented verdicts
    # must not become green merely because this reducer does not recognize them.
    failed = [r for r in arch_results if r.get("verdict") != "PASS"]
    if failed:
        return {"action": "bod", "status": "failure",
                "reasons": sorted({r["arch"] for r in failed}),
                "bod": _aggregate_bod(arch_results)}

    # All arches PASS from here.
    if is_draft:
        return {"action": "draft_report", "status": "success", "reasons": [], "bod": None}
    if not helpful:
        return {"action": "neutral", "status": "success", "reasons": ["not-helpful"], "bod": None}
    if cfg.is_auto_merge_author(author):
        return {"action": "auto_merge", "status": "success", "reasons": [], "bod": None}
    # Any other author (maintainer or non-maintainer) is tagged: a maintainer must
    # comment `@claude /merge` — the agent never merges another author's PR unasked.
    return {"action": "tag_maintainer", "status": "success", "reasons": [], "bod": None}


_ACTION_LINE = {
    "auto_merge": "✅ All gates green — **auto-merge** (flushes the staging train).",
    "tag_maintainer": "✅ All gates green — comment `@claude /merge` to land it.",
    "neutral": "🟡 All gates green, but no measurable improvement — clarify intent.",
    "draft_report": "📋 Draft — verdict + BOD only, no merge.",
    "bod": "❌ Blocked — see the Bill of Debt below.",
}


def format_pr_comment(outcome: dict, arch_results) -> str:
    """Render the PR comment markdown: verdict line + per-arch table + BOD/notes."""
    lines = [_ACTION_LINE.get(outcome["action"], outcome["action"]), "",
             "| arch | verdict |", "|---|---|"]
    for r in arch_results:
        lines.append(f"| {r['arch']} | {r['verdict']} |")
    if outcome.get("bod") and outcome["bod"]["blockers"]:
        lines += ["", "**Bill of Debt:**"]
        for b in outcome["bod"]["blockers"]:
            arch = b.get("arch", "")
            lines.append(f"- `{arch}` **{b['kind']}**: {b['detail']}")
    return "\n".join(lines)

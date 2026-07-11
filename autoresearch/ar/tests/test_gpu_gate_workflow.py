# Copyright (c) Kaden Schutt
"""Static contract tests for the GitHub gate DAG.

These assertions intentionally pin the safety properties that the live failures
violated: self-hosted jobs require a validated Claude plan, the matrix is dynamic,
and interpretation cannot proceed without complete runner artifacts.
"""
from pathlib import Path


WORKFLOW = Path(__file__).parents[3] / ".github" / "workflows" / "gpu-gates.yml"


def _text() -> str:
    return WORKFLOW.read_text()


def test_gpu_runners_are_released_only_by_validated_claude_dispatch():
    text = _text()
    gate = text.split("\n  gate:\n", 1)[1].split("\n  interpret:\n", 1)[0]
    assert "needs: [decide, dispatch]" in gate
    assert "needs.dispatch.outputs.valid == 'true'" in gate
    assert "needs.dispatch.outputs.decision == 'run'" in gate
    assert "matrix: ${{ fromJSON(needs.dispatch.outputs.matrix) }}" in gate
    assert "always()" not in gate.split("strategy:", 1)[0]


def test_missing_claude_plan_has_no_generic_gpu_fallback():
    text = _text()
    dispatch = text.split("\n  dispatch:\n", 1)[1].split("\n  no_runner:\n", 1)[0]
    assert "--validate-plan dispatch_plan.json" in dispatch
    assert "if-no-files-found: error" in dispatch
    assert "fall back to a generic" not in text.lower()
    assert "codex_gate_prompt.txt" not in text


def test_selected_runner_uses_mechanical_collect_and_grade_not_agent_verdict():
    text = _text()
    gate = text.split("\n  gate:\n", 1)[1].split("\n  interpret:\n", 1)[0]
    collect = gate.split("- name: Collect and grade assigned GPU cells", 1)[1]
    collect = collect.split("- name: Run assigned bespoke behavior tests", 1)[0]
    assert "gate --collect" in collect
    assert "gate --grade" in collect
    assert "codex exec" not in collect


def test_interpret_requires_dispatch_and_downloaded_runner_evidence():
    text = _text()
    interpret = text.split("\n  interpret:\n", 1)[1]
    assert "needs.dispatch.result == 'success'" in interpret
    assert "needs.dispatch.outputs.valid == 'true'" in interpret
    assert "pattern: gate-results-*" in interpret
    assert "pattern: gate-behavior-*" in interpret
    assert "EVIDENCE_BLOCKED" in interpret
    assert "--verify-evidence validated_dispatch.json" in interpret
    assert "joined_evidence.json" in interpret
    assert "--validate-action interpret_action.json" in interpret
    assert 'current_head=$(gh pr view "$PR" --json headRefOid --jq .headRefOid)' in interpret
    assert 'if [ "$current_head" != "$HEAD" ]' in interpret
    assert "continue-on-error: true" not in interpret


def test_gate_comment_cannot_hide_deterministic_reject():
    text = _text()
    interpret = text.split("\n  interpret:\n", 1)[1]
    assert "Preserve deterministic REJECT as a red check" in interpret
    assert 'if [ "$RC" != "0" ]' in interpret


def test_gate_comment_does_not_cancel_an_active_same_pr_run():
    assert "cancel-in-progress: false" in _text()

import pytest
from pathlib import Path

SKILLS_DIR = Path(".agents/skills")
DISCOVERY_SKILL = SKILLS_DIR / "agentic-pr-discovery" / "SKILL.md"
INSPECTOR_SH = SKILLS_DIR / "agentic-pr-static-review" / "run-inspector.sh"
DISCOVER_SH = SKILLS_DIR / "agentic-pr-discovery" / "discover.sh"
PREFLIGHT_DISCOVERY_SH = SKILLS_DIR / "agentic-pr-discovery" / "preflight.sh"
PREFLIGHT_REVIEW_SH = SKILLS_DIR / "agentic-pr-static-review" / "preflight.sh"
STATIC_SKILL = SKILLS_DIR / "agentic-pr-static-review" / "SKILL.md"


def test_static_review_skill_forbids_checkout_and_test_execution():
    body = STATIC_SKILL.read_text()
    assert "git checkout" in body and "must not" in body
    assert "test execution" in body and "out of scope" in body


def test_inspector_wrapper_invokes_only_toolless_cli():
    body = INSPECTOR_SH.read_text()
    assert "autoresearch.ar.review.cli inspect" in body
    assert "codex exec" not in body
    assert "opencode run" not in body


def test_discovery_wrapper_invokes_only_cli():
    body = DISCOVER_SH.read_text()
    assert "autoresearch.ar.review.cli discover" in body
    assert "codex exec" not in body
    assert "opencode run" not in body


def test_discovery_preflight_invokes_only_cli():
    body = PREFLIGHT_DISCOVERY_SH.read_text()
    assert "autoresearch.ar.review.cli preflight" in body
    assert "discover" not in body  # preflight, not discovery


def test_review_preflight_invokes_only_cli():
    body = PREFLIGHT_REVIEW_SH.read_text()
    assert "autoresearch.ar.review.cli preflight" in body
    assert "controller" in body  # mode hint in the SKILL.md or shell


def test_every_skill_has_metadata():
    for skill_dir in [SKILLS_DIR / "agentic-pr-discovery", SKILLS_DIR / "agentic-pr-static-review"]:
        assert (skill_dir / "skill.json").exists()
        assert (skill_dir / "SKILL.md").exists()

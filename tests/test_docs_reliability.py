#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused unit tests for scripts/check-docs-reliability.py."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import re
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "check-docs-reliability.py"


def load_checker():
    name = "check_docs_reliability"
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass / typing resolve reliably.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CHECKER = load_checker()

# Pre-commit runtime-path filter (mirrors .githooks/pre-commit DOCS_ONLY_PATH).
# Keep the pattern string identical to the hook so docs-only short-circuit
# behavior can be unit-tested without invoking the hook or any GPU gate.
DOCS_ONLY_PATH = (
    r"^(docs/|\.agents/|\.agent-memory/|[^/]+\.md$|"
    r"\.github/workflows/no-gpu-ci\.yml$|"
    r"\.githooks/pre-commit$|"
    r"scripts/check-docs-reliability\.py$|"
    r"scripts/no-gpu-ci\.sh$|"
    r"scripts/mem\.sh$|"
    r"tests/test_docs_reliability\.py$)"
)


def filter_runtime_changed(paths):
    """Return staged paths that still count as runtime for hotspot gates.

    Documentation, agent guidance, and the no-GPU docs-enforcement surface are
    excluded after the reliability checker has already run. Empty result means
    the pre-commit hook must exit success before HOTSPOT/serve/PP/GPU gates.
    """
    pat = re.compile(DOCS_ONLY_PATH)
    return [p for p in paths if p and not pat.search(p)]


def run(cmd, cwd, check=True, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged.setdefault("GIT_AUTHOR_NAME", "tester")
    merged.setdefault("GIT_AUTHOR_EMAIL", "tester@example.com")
    merged.setdefault("GIT_COMMITTER_NAME", "tester")
    merged.setdefault("GIT_COMMITTER_EMAIL", "tester@example.com")
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=merged,
    )


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


SHA_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SHA_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

VALID_INDEX = f"""
# Documentation index

| Field | Value |
|---|---|
| Working branch | `feature/docs` |
| Audited source ref | `{SHA_A}` |
| Comparison base | `origin/main` @ `{SHA_B}` |

## Ownership

| Concern | Canonical owner | State | Notes |
|---|---|---|---|
| Docs navigation | [`docs/INDEX.md`](INDEX.md) | shipped / ref-pinned | This file. |
| Validation routes | [`docs/VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Sole selector. |
| Admissions | [`docs/admissions.yml`](admissions.yml) | shipped / ref-pinned | Empty until earned. |
| Env vars | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | |
| Branch-only demo | [`docs/REDLINE.md`](REDLINE.md) | branch-implemented | Not shipped vs comparison base. |
| Universal gate | — | **BLOCKED** | No universal gate. |

## Top-level page classification

| Page | State | Owner role |
|---|---|---|
| [`INDEX.md`](INDEX.md) | shipped / ref-pinned | Navigation. |
| [`VALIDATION.md`](VALIDATION.md) | shipped / ref-pinned | Routes. |
| [`admissions.yml`](admissions.yml) | shipped / ref-pinned | Registry. |
| [`env-vars.md`](env-vars.md) | shipped / ref-pinned | Env. |
| [`REDLINE.md`](REDLINE.md) | branch-implemented | Branch policy. |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Onboarding. |

## Collection classification

| Collection | State | Policy |
|---|---|---|
| [`methodology/`](methodology/) | shipped / ref-pinned | Active methodology. |
| [`perf-checkpoints/`](perf-checkpoints/) | measured | Immutable. |
| [`design/`](design/) | planned | Designs. |
| [`plans/`](plans/) | planned | Plans. |
| [`specs/`](specs/) | planned | Specs. |
| [`investigations/`](investigations/) | historical | Investigations. |
| [`reviews/`](reviews/) | historical | Reviews. |
| [`lessons_learned/`](lessons_learned/) | historical | Lessons. |
| [`superpowers/`](superpowers/) | planned | Workflow records. |
"""

VALID_INDEX_ALIAS_HEADERS = VALID_INDEX.replace(
    "| Concern | Canonical owner | State | Notes |",
    "| Concern | Canonical owner | Truth state | Scope/limits |",
)

VALID_VALIDATION = """
# Validation routes

## Retired coherence-gate scripts

The fixed `scripts/coherence-gate-*.sh` batteries are **retired** as current
acceptance evidence. They must not be required for merge.

| Pattern | Status |
|---|---|
| `scripts/coherence-gate-*.sh` | **Historical reproduction only.** Never promotion or acceptance. |
"""

VALID_ENV = """
# Environment variables

| Variable | Meaning |
|---|---|
| `HIPFIRE_GRAPH` | Graph capture. |
| `HIPFIRE_KV_MODE` | KV mode. |
"""


class RepoFixture:
    """Minimal git repo with base + target commits."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        run(["git", "init"], self.root)
        run(["git", "checkout", "-b", "main"], self.root)
        self._seed_base()

    def close(self) -> None:
        self.tmp.cleanup()

    def _seed_base(self) -> None:
        write(self.root / "README.md", "# Product\n\nSee docs.\n")
        write(self.root / "AGENTS.md", "# Agents\n\nUse `HIPFIRE_GRAPH`.\n")
        write(self.root / "CONTRIBUTING.md", "# Contributing\n\nNo extra env.\n")
        write(self.root / "CLAUDE.md", "# Claude\n\nRouting only.\n")
        write(self.root / "docs" / "INDEX.md", VALID_INDEX)
        write(self.root / "docs" / "VALIDATION.md", VALID_VALIDATION)
        write(self.root / "docs" / "env-vars.md", VALID_ENV)
        write(
            self.root / "docs" / "admissions.yml",
            '{\n  "schema_version": 1,\n  "records": []\n}\n',
        )
        write(self.root / "docs" / "REDLINE.md", "# Redline\n\nBranch policy.\n")
        write(self.root / "docs" / "GETTING_STARTED.md", "# Start\n\nHello.\n")
        write(
            self.root / "docs" / "perf-checkpoints" / "2026-01-01-example.md",
            "# Checkpoint\n\nimmutable body v1\n",
        )
        for coll in (
            "methodology",
            "design",
            "plans",
            "specs",
            "investigations",
            "reviews",
            "lessons_learned",
            "superpowers",
        ):
            write(self.root / "docs" / coll / "README.md", f"# {coll}\n\nBanner.\n")
        write(
            self.root / ".agents" / "skills" / "demo" / "SKILL.md",
            "---\nname: demo\n---\n# Demo skill\n\nWorkflow only.\n",
        )
        run(["git", "add", "-A"], self.root)
        run(["git", "commit", "-m", "base"], self.root)
        self.base_ref = run(["git", "rev-parse", "HEAD"], self.root).stdout.strip()

    def commit_all(self, message: str = "change") -> str:
        run(["git", "add", "-A"], self.root)
        run(["git", "commit", "-m", message], self.root)
        return run(["git", "rev-parse", "HEAD"], self.root).stdout.strip()

    def run_checker(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(CHECKER_PATH), "--root", str(self.root), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)


class DocsReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = RepoFixture()

    def tearDown(self) -> None:
        self.fx.close()

    # --- CLI ------------------------------------------------------------

    def test_cli_requires_mode(self) -> None:
        proc = self.fx.run_checker("--base-ref", "HEAD")
        self.assertNotEqual(proc.returncode, 0)

    def test_cli_target_ref_mode_valid(self) -> None:
        proc = self.fx.run_checker(
            "--target-ref", self.fx.base_ref, "--base-ref", self.fx.base_ref
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("docs-reliability: ok", proc.stdout)

    def test_cli_staged_mode_valid(self) -> None:
        proc = self.fx.run_checker("--staged", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_cli_staged_sees_index_not_workdir_only(self) -> None:
        write(self.fx.root / "docs" / "admissions.yml", "{bad")
        proc = self.fx.run_checker("--staged", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        run(["git", "add", "docs/admissions.yml"], self.fx.root)
        proc2 = self.fx.run_checker("--staged", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc2.returncode, 0)
        self.assertIn("admissions", proc2.stdout)

    # --- valid baseline -------------------------------------------------

    def test_valid_tree_passes(self) -> None:
        proc = self.fx.run_checker(
            "--target-ref", "HEAD", "--base-ref", self.fx.base_ref
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_alias_ownership_headers_pass(self) -> None:
        write(self.fx.root / "docs" / "INDEX.md", VALID_INDEX_ALIAS_HEADERS)
        self.fx.commit_all("alias headers")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    # --- links / anchors / images / escape ------------------------------

    def test_broken_local_link_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [missing](nope.md).\n",
        )
        self.fx.commit_all("broken link")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("link:", proc.stdout)
        self.assertIn("nope.md", proc.stdout)

    def test_broken_image_link_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n![diagram](images/missing.png)\n",
        )
        self.fx.commit_all("broken image")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("link:", proc.stdout)
        self.assertIn("missing.png", proc.stdout)

    def test_repo_escaping_link_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [out](../../etc/passwd).\n",
        )
        self.fx.commit_all("escape link")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("link:", proc.stdout)
        self.assertIn("escapes", proc.stdout)

    def test_repo_root_absolute_markdown_link_resolves_from_root(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [idx](/docs/INDEX.md).\n",
        )
        self.fx.commit_all("root abs link")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_repo_root_absolute_markdown_link_missing_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [x](/docs/nope-missing.md).\n",
        )
        self.fx.commit_all("root abs missing")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("link:", proc.stdout)

    def test_external_link_exempt(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [ext](https://example.com/nope) and <https://example.org/x>.\n",
        )
        self.fx.commit_all("external links")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_missing_github_anchor_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n## Real Section\n\nBody.\n\nSee [x](#missing-section).\n",
        )
        self.fx.commit_all("bad anchor")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("anchor:", proc.stdout)

    def test_valid_github_anchor_passes(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n## Real Section\n\nBody.\n\nSee [x](#real-section).\n",
        )
        self.fx.commit_all("good anchor")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_missing_path_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nRun `scripts/not-a-real-tool.py` today.\n",
        )
        self.fx.commit_all("bad path")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("path:", proc.stdout)
        self.assertIn("scripts/not-a-real-tool.py", proc.stdout)

    def test_backticked_command_span_missing_script_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nRun `python scripts/missing_tool.py --arg 1` before merge.\n",
        )
        self.fx.commit_all("command span missing")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("path:", proc.stdout)
        self.assertIn("scripts/missing_tool.py", proc.stdout)

    def test_backticked_existing_path_passes(self) -> None:
        write(self.fx.root / "scripts" / "gates.sh", "#!/bin/sh\n")
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nRun `scripts/gates.sh`.\n",
        )
        self.fx.commit_all("good path")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_path_template_angle_braces_ignored(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "Kernel path template: `kernels/src/<name>.<chip>.hip`.\n",
        )
        self.fx.commit_all("path template angles")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_path_template_braces_and_ellipsis_ignored(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "See `crates/hipfire-runtime/src/{hfq,llama}.rs` and "
            "`tests/speed-baselines/…` notes.\n",
        )
        self.fx.commit_all("path template braces")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_glob_template_ignored(self) -> None:
        write(
            self.fx.root / "docs" / "VALIDATION.md",
            "# Validation\n\n"
            "Pattern `scripts/coherence-gate-*.sh` is not a concrete path check.\n",
        )
        self.fx.commit_all("path template glob")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_source_line_suffix_stripped(self) -> None:
        write(
            self.fx.root / "crates" / "hipfire-runtime" / "src" / "spec.rs",
            "// stub\n",
        )
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "See `crates/hipfire-runtime/src/spec.rs:91-108` for the seam.\n",
        )
        self.fx.commit_all("source line suffix")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_source_line_suffix_missing_path_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "See `crates/hipfire-runtime/src/missing.rs:12` for details.\n",
        )
        self.fx.commit_all("missing with line suffix")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("path:", proc.stdout)
        self.assertIn("crates/hipfire-runtime/src/missing.rs", proc.stdout)
        self.assertNotIn("missing.rs:12", proc.stdout)

    def test_backticked_missing_path_with_same_line_denial_passes(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "`scripts/old-pre-modular-tool.py` is retired/historical "
            "(pre-modular; does not exist on this tree).\n",
        )
        self.fx.commit_all("denied missing path")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_backticked_missing_path_without_denial_still_fails(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nRun `scripts/old-pre-modular-tool.py` in CI.\n",
        )
        self.fx.commit_all("undenied missing path")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("path:", proc.stdout)
        self.assertIn("scripts/old-pre-modular-tool.py", proc.stdout)

    def test_backticked_missing_path_denial_must_be_same_line(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "The following path is retired and historical.\n\n"
            "See `scripts/old-pre-modular-tool.py` for context.\n",
        )
        self.fx.commit_all("denial not same line")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("path:", proc.stdout)
        self.assertIn("scripts/old-pre-modular-tool.py", proc.stdout)

    def test_backticked_agents_skills_path_not_corrupted(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee `.agents/skills/demo/SKILL.md` for the workflow.\n",
        )
        self.fx.commit_all("agents skills path")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_strip_leading_dot_slash_helper(self) -> None:
        self.assertEqual(CHECKER.strip_leading_dot_slash("./scripts/x.sh"), "scripts/x.sh")
        self.assertEqual(
            CHECKER.strip_leading_dot_slash(".agents/skills/demo/SKILL.md"),
            ".agents/skills/demo/SKILL.md",
        )

    # --- skill roots ----------------------------------------------------

    def test_dot_skills_tree_fails(self) -> None:
        write(self.fx.root / ".skills" / "foo" / "SKILL.md", "# no\n")
        self.fx.commit_all("dot skills")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("skills-root:", proc.stdout)

    def test_active_docs_skills_executable_reference_fails(self) -> None:
        # Classified non-exec collection exists, but active prose must not link
        # an executable docs/skills entrypoint.
        write(
            self.fx.root / "docs" / "skills" / "legacy" / "SKILL.md",
            "---\nname: legacy\n---\n# Legacy\n",
        )
        idx = VALID_INDEX.replace(
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n",
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n"
            "| [`skills/`](skills/) | historical | Non-exec only. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", idx)
        # Reference from an explicit ACTIVE_MARKDOWN page.
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\nSee [legacy skill](skills/legacy/SKILL.md).\n",
        )
        self.fx.commit_all("active docs skills ref")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("skills-root:", proc.stdout)
        self.assertIn("docs/skills/", proc.stdout)

    def test_non_active_docs_body_not_scanned_for_skill_refs(self) -> None:
        # docs/methodology/extra-note.md is outside ACTIVE_MARKDOWN even though
        # the methodology/ collection is lifecycle-classified as shipped.
        write(
            self.fx.root / "docs" / "methodology" / "extra-note.md",
            "# Extra\n\nDo not use `.skills/foo` anymore.\n"
            "Also see `scripts/totally-missing.py`.\n",
        )
        self.fx.commit_all("non-active methodology body")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_historical_plan_body_not_scanned(self) -> None:
        write(
            self.fx.root / "docs" / "plans" / "old-plan.md",
            "# Old plan\n\n"
            "See [broken](nope-missing.md) and `scripts/never-existed.py`.\n"
            "We used coherence-gate as acceptance in 2025.\n",
        )
        self.fx.commit_all("historical plan noise")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_historical_investigation_body_not_scanned(self) -> None:
        write(
            self.fx.root / "docs" / "investigations" / "trail.md",
            "# Trail\n\n"
            "Broken link [x](../nope.md) and missing `crates/missing/mod.rs:12-40`.\n",
        )
        self.fx.commit_all("historical investigation noise")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_docs_skills_executable_fails(self) -> None:
        write(self.fx.root / "docs" / "skills" / "foo" / "SKILL.md", "# exec\n")
        # skills/ is a new top-level docs collection — must classify or lifecycle fails.
        # Prefer isolating skills-root: classify it in INDEX.
        idx = VALID_INDEX.replace(
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n",
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n"
            "| [`skills/`](skills/) | historical | Non-exec only. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", idx)
        self.fx.commit_all("docs skills exec")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("skills-root:", proc.stdout)
        self.assertIn("docs/skills/", proc.stdout)

    def test_docs_skills_non_executable_markdown_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "skills" / "note.md",
            "# Note\n\nNon-executable prose.\n",
        )
        idx = VALID_INDEX.replace(
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n",
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n"
            "| [`skills/`](skills/) | historical | Non-exec only. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", idx)
        self.fx.commit_all("docs skills prose")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_active_reference_to_dot_skills_fails(self) -> None:
        write(
            self.fx.root / "AGENTS.md",
            "# Agents\n\nSee `.skills/hipfire-diag` for diagnostics.\n",
        )
        self.fx.commit_all("ref dot skills")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("skills-root:", proc.stdout)

    def test_skill_subtree_without_skill_md_fails(self) -> None:
        write(
            self.fx.root / ".agents" / "skills" / "orphan" / "notes.md",
            "# Not a skill entrypoint\n",
        )
        self.fx.commit_all("orphan skill")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("skills-root:", proc.stdout)
        self.assertIn("SKILL.md", proc.stdout)

    # --- coherence-gate -------------------------------------------------

    def test_active_coherence_acceptance_fails(self) -> None:
        write(
            self.fx.root / "README.md",
            "# Product\n\nRun scripts/coherence-gate.sh as acceptance before merge.\n",
        )
        self.fx.commit_all("bad coherence")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("coherence-gate:", proc.stdout)

    def test_historical_coherence_mention_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "investigations" / "old.md",
            "# Old\n\nWe used coherence-gate as acceptance in 2025.\n",
        )
        self.fx.commit_all("historical coherence")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_retired_coherence_language_allowed(self) -> None:
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_coherence_denial_do_not_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "VALIDATION.md",
            "# Validation\n\n"
            "Do not treat coherence-gate as acceptance evidence.\n",
        )
        self.fx.commit_all("coherence do not")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_coherence_denial_must_not_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n"
            "coherence-gate must not be required for merge or promotion.\n",
        )
        self.fx.commit_all("coherence must not")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_coherence_denial_never_allowed(self) -> None:
        write(
            self.fx.root / "README.md",
            "# Product\n\n"
            "Never use coherence-gate as acceptance or the merge bar.\n",
        )
        self.fx.commit_all("coherence never")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_coherence_denial_historical_only_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "VALIDATION.md",
            "# Validation\n\n"
            "coherence-gate remains historical-only, never promotion acceptance.\n",
        )
        self.fx.commit_all("coherence historical-only")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_positive_coherence_acceptance_still_fails_with_nearby_noise(self) -> None:
        write(
            self.fx.root / "docs" / "REDLINE.md",
            "# Redline\n\n"
            "Background note elsewhere: suites can be retired.\n\n"
            "Current policy requires coherence-gate as mandatory acceptance "
            "before merge on this branch.\n",
        )
        self.fx.commit_all("positive coherence still fails")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("coherence-gate:", proc.stdout)

    def test_unknown_lifecycle_label_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| [`GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Onboarding. |\n",
            "| [`GETTING_STARTED.md`](GETTING_STARTED.md) | unknown | Onboarding. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("unknown lifecycle")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("lifecycle:", proc.stdout)

    def test_composite_lifecycle_label_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| [`design/`](design/) | planned | Designs. |\n",
            "| [`design/`](design/) | planned / historical | Designs. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("composite lifecycle")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("lifecycle:", proc.stdout)

    def test_blank_lifecycle_label_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| [`GETTING_STARTED.md`](GETTING_STARTED.md) | shipped / ref-pinned | Onboarding. |\n",
            "| [`GETTING_STARTED.md`](GETTING_STARTED.md) |  | Onboarding. |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("blank lifecycle")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("lifecycle:", proc.stdout)

    def test_external_canonical_owner_link_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| Env vars | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | |\n",
            "| Env vars | [env](https://example.com/env) | shipped / ref-pinned | |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("external owner")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("index-owner:", proc.stdout)
        self.assertIn("local", proc.stdout)

    # --- INDEX schema / owners / lifecycle ------------------------------

    def test_index_wrong_headers_fail(self) -> None:
        bad = VALID_INDEX.replace(
            "| Concern | Canonical owner | State | Notes |",
            "| Concern | Owner | Status | Comment |",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("bad headers")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("index-schema:", proc.stdout)

    def test_duplicate_concern_owner_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| Env vars | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | |\n",
            "| Env vars | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | |\n"
            "| Env vars | [`docs/CONFIG.md`](CONFIG.md) | shipped / ref-pinned | duplicate |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("dup concern")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("index-owner:", proc.stdout)
        self.assertIn("Env vars", proc.stdout)

    def test_owner_without_markdown_link_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| Env vars | [`docs/env-vars.md`](env-vars.md) | shipped / ref-pinned | |\n",
            "| Env vars | docs/env-vars.md | shipped / ref-pinned | bare path |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("bare owner")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("index-owner:", proc.stdout)

    def test_missing_lifecycle_collection_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| [`superpowers/`](superpowers/) | planned | Workflow records. |\n",
            "",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("missing lifecycle")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("lifecycle:", proc.stdout)
        self.assertIn("superpowers/", proc.stdout)

    def test_unclassified_top_level_page_fails(self) -> None:
        write(self.fx.root / "docs" / "ORPHAN.md", "# Orphan\n")
        self.fx.commit_all("orphan page")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("lifecycle:", proc.stdout)
        self.assertIn("ORPHAN.md", proc.stdout)

    def test_branch_row_without_full_sha_fails(self) -> None:
        bad = VALID_INDEX.replace(SHA_A, "abc1234").replace(SHA_B, "def5678")
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("short sha")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        joined = proc.stdout + proc.stderr
        self.assertTrue(
            "index-meta:" in joined or "branch-meta:" in joined,
            joined,
        )

    def test_branch_row_empty_branch_name_fails(self) -> None:
        bad = VALID_INDEX.replace(
            "| Working branch | `feature/docs` |\n",
            "| Working branch |  |\n",
        )
        write(self.fx.root / "docs" / "INDEX.md", bad)
        self.fx.commit_all("empty branch")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("index-meta:", proc.stdout)

    # --- admissions -----------------------------------------------------

    def test_admissions_non_empty_fails(self) -> None:
        write(
            self.fx.root / "docs" / "admissions.yml",
            json.dumps({"schema_version": 1, "records": [{"id": "x"}]}, indent=2) + "\n",
        )
        self.fx.commit_all("nonempty admissions")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("admissions:", proc.stdout)

    def test_admissions_wrong_schema_fails(self) -> None:
        write(
            self.fx.root / "docs" / "admissions.yml",
            json.dumps({"schema_version": 2, "records": []}) + "\n",
        )
        self.fx.commit_all("bad schema")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("admissions:", proc.stdout)

    def test_admissions_not_json_fails(self) -> None:
        write(self.fx.root / "docs" / "admissions.yml", "schema_version: 1\nrecords: []\n")
        self.fx.commit_all("yaml not json")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("admissions:", proc.stdout)

    # --- env coverage ---------------------------------------------------

    def test_env_coverage_missing_var_fails(self) -> None:
        write(
            self.fx.root / "README.md",
            "# Product\n\nSet `HIPFIRE_NEW_FLAG` before run.\n",
        )
        self.fx.commit_all("env drift")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("env:", proc.stdout)
        self.assertIn("HIPFIRE_NEW_FLAG", proc.stdout)

    def test_env_coverage_documented_passes(self) -> None:
        write(
            self.fx.root / "docs" / "env-vars.md",
            VALID_ENV + "\n| `HIPFIRE_NEW_FLAG` | New flag. |\n",
        )
        write(
            self.fx.root / "README.md",
            "# Product\n\nSet `HIPFIRE_NEW_FLAG` before run.\n",
        )
        self.fx.commit_all("env ok")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_env_source_extraction_missing_vars_fails(self) -> None:
        write(
            self.fx.root / "src" / "env_probe.rs",
            'fn probe() { let _ = std::env::var_os("HIPFIRE_KERNEL_CACHE"); }\n',
        )
        write(
            self.fx.root / "cli" / "env_probe.ts",
            "const x = process.env.HIPFIRE_REGISTRY_URL || process.env.NODE_ENV;\n",
        )
        self.fx.commit_all("source env drift")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("HIPFIRE_KERNEL_CACHE", proc.stdout)
        self.assertIn("HIPFIRE_REGISTRY_URL", proc.stdout)
        self.assertNotIn("NODE_ENV", proc.stdout)

    def test_env_source_extraction_documented_passes(self) -> None:
        write(
            self.fx.root / "src" / "env_probe.rs",
            'fn probe() { let _ = env::var("HIPFIRE_KERNEL_CACHE"); }\n',
        )
        write(
            self.fx.root / "cli" / "env_probe.ts",
            "const x = process.env.HIPFIRE_REGISTRY_URL;\n",
        )
        write(
            self.fx.root / "docs" / "env-vars.md",
            VALID_ENV
            + "\n| `HIPFIRE_KERNEL_CACHE` | Rust source flag. |\n"
            + "| `HIPFIRE_REGISTRY_URL` | TypeScript source flag. |\n",
        )
        self.fx.commit_all("source env documented")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    # --- checkpoints ----------------------------------------------------

    def test_checkpoint_byte_change_fails(self) -> None:
        write(
            self.fx.root / "docs" / "perf-checkpoints" / "2026-01-01-example.md",
            "# Checkpoint\n\nimmutable body v2 CHANGED\n",
        )
        self.fx.commit_all("mutate checkpoint")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("checkpoint:", proc.stdout)
        self.assertIn("bytes changed", proc.stdout)

    def test_checkpoint_deletion_fails(self) -> None:
        path = self.fx.root / "docs" / "perf-checkpoints" / "2026-01-01-example.md"
        path.unlink()
        self.fx.commit_all("delete checkpoint")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("checkpoint:", proc.stdout)
        self.assertIn("deleted", proc.stdout)

    def test_new_checkpoint_allowed(self) -> None:
        write(
            self.fx.root / "docs" / "perf-checkpoints" / "2026-07-19-new.md",
            "# New checkpoint\n\nfresh evidence\n",
        )
        self.fx.commit_all("add checkpoint")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    # --- diagnostics ordering / exit ------------------------------------

    def test_diagnostics_sorted_and_nonzero(self) -> None:
        write(self.fx.root / "docs" / "admissions.yml", "{")
        write(
            self.fx.root / "docs" / "GETTING_STARTED.md",
            "# Start\n\n[x](missing.md)\n",
        )
        self.fx.commit_all("multi fail")
        proc = self.fx.run_checker("--target-ref", "HEAD", "--base-ref", self.fx.base_ref)
        self.assertNotEqual(proc.returncode, 0)
        lines = [
            ln
            for ln in proc.stdout.splitlines()
            if ln and not ln.startswith("docs-reliability")
        ]
        self.assertEqual(lines, sorted(lines))
        self.assertIn("docs-reliability:", proc.stderr)

    def test_from_ref_requires_commit_object(self) -> None:
        # A tree object must not masquerade as an explicit commit ref.
        tree = run(["git", "rev-parse", "HEAD^{tree}"], self.fx.root).stdout.strip()
        with self.assertRaises(SystemExit):
            CHECKER.GitSnapshot.from_ref(self.fx.root, tree)

    # --- unit helpers ---------------------------------------------------

    def test_active_markdown_includes_hfp4_and_speculation_inventory(self) -> None:
        self.assertIn("docs/quant-formats/hfp4.md", CHECKER.ACTIVE_MARKDOWN)
        self.assertIn(
            "docs/speculation-support-inventory.md", CHECKER.ACTIVE_MARKDOWN
        )

    def test_is_active_markdown_excludes_non_listed_docs(self) -> None:
        class Dummy:
            pass

        checker = CHECKER.Checker(Dummy(), Dummy())  # type: ignore[arg-type]
        self.assertTrue(checker.is_active_markdown("docs/GETTING_STARTED.md"))
        self.assertTrue(
            checker.is_active_markdown(".agents/skills/demo/SKILL.md")
        )
        self.assertFalse(checker.is_active_markdown("docs/methodology/extra-note.md"))
        self.assertFalse(checker.is_active_markdown("docs/plans/old-plan.md"))
        self.assertFalse(
            checker.is_active_markdown("docs/investigations/trail.md")
        )

    def test_path_template_and_suffix_helpers(self) -> None:
        self.assertTrue(CHECKER.is_path_template_token("kernels/src/<name>.hip"))
        self.assertTrue(
            CHECKER.is_path_template_token("crates/hipfire-runtime/src/{a,b}.rs")
        )
        self.assertTrue(CHECKER.is_path_template_token("scripts/coherence-gate-*.sh"))
        self.assertTrue(CHECKER.is_path_template_token("tests/speed-baselines/…"))
        self.assertFalse(CHECKER.is_path_template_token("scripts/gates.sh"))
        self.assertEqual(
            CHECKER.strip_source_location_suffix(
                "crates/hipfire-runtime/src/spec.rs:91-108"
            ),
            "crates/hipfire-runtime/src/spec.rs",
        )
        self.assertEqual(
            CHECKER.strip_source_location_suffix("scripts/gates.sh"),
            "scripts/gates.sh",
        )
        tokens = CHECKER.extract_path_tokens(
            "see crates/hipfire-runtime/src/spec.rs:12-14 please"
        )
        self.assertEqual(tokens, ["crates/hipfire-runtime/src/spec.rs"])
        self.assertEqual(
            CHECKER.extract_path_tokens("kernels/src/<name>.<chip>.hip"),
            [],
        )

    def test_line_has_path_denial_helper(self) -> None:
        self.assertTrue(
            CHECKER.line_has_path_denial(
                "`scripts/x.py` is retired/historical (pre-modular)."
            )
        )
        self.assertTrue(
            CHECKER.line_has_path_denial("missing path disclosure: `docs/old.md`")
        )
        self.assertFalse(
            CHECKER.line_has_path_denial("Run `scripts/x.py` before merge.")
        )

    def test_github_anchor_helper(self) -> None:
        self.assertEqual(CHECKER.github_anchor("Real Section"), "real-section")
        self.assertEqual(CHECKER.github_anchor("Truth states"), "truth-states")

    def test_extract_path_tokens_from_command_span(self) -> None:
        tokens = CHECKER.extract_path_tokens("python scripts/foo.py --arg")
        self.assertEqual(tokens, ["scripts/foo.py"])
        tokens2 = CHECKER.extract_path_tokens(".agents/skills/demo/SKILL.md")
        self.assertEqual(tokens2, [".agents/skills/demo/SKILL.md"])


class PrecommitRuntimeFilterTests(unittest.TestCase):
    """Pure path-filter tests for the docs-only pre-commit short-circuit.

    These do not invoke the hook, git, GPU gates, or the reliability checker.
    """

    def test_docs_only_paths_yield_empty_runtime_set(self) -> None:
        docs_only = [
            "README.md",
            "AGENTS.md",
            "CONTRIBUTING.md",
            "docs/INDEX.md",
            "docs/VALIDATION.md",
            "docs/methodology/foo.md",
            ".agents/skills/demo/SKILL.md",
            ".agent-memory/notes.md",
            ".github/workflows/no-gpu-ci.yml",
            ".githooks/pre-commit",
            "scripts/check-docs-reliability.py",
            "scripts/no-gpu-ci.sh",
            "scripts/mem.sh",
            "tests/test_docs_reliability.py",
        ]
        self.assertEqual(filter_runtime_changed(docs_only), [])

    def test_runtime_only_paths_are_preserved(self) -> None:
        runtime = [
            "crates/hipfire-arch-lfm2moe/src/forward.rs",
            "crates/rdna-compute/src/dispatch.rs",
            "examples/daemon.rs",
            "scripts/speed-gate.sh",
            "kernels/src/gemm.hip",
            ".github/workflows/ci.yml",
        ]
        self.assertEqual(filter_runtime_changed(runtime), runtime)

    def test_mixed_commit_preserves_runtime_paths(self) -> None:
        mixed = [
            "README.md",
            "docs/INDEX.md",
            ".agents/skills/demo/SKILL.md",
            "scripts/check-docs-reliability.py",
            "crates/hipfire-arch-lfm2moe/src/forward.rs",
            "examples/daemon.rs",
            "tests/test_docs_reliability.py",
        ]
        self.assertEqual(
            filter_runtime_changed(mixed),
            [
                "crates/hipfire-arch-lfm2moe/src/forward.rs",
                "examples/daemon.rs",
            ],
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
INSTALLER = REPO / "scripts" / "install.sh"
INSTALLER_TEXT = INSTALLER.read_text(encoding="utf-8")


def installer(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        env=run_env,
    )


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _write_commit(repo: Path, path: str, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _init_remote(tmp: Path) -> Path:
    remote = tmp / "remote.git"
    work = tmp / "remote-work"
    work.mkdir()
    _git(work, "init", "-b", "master")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "test")
    _write_commit(work, "README", "base\n", "base")
    # Bare remote for file:// clone/fetch.
    subprocess.run(
        ["git", "clone", "--bare", str(work), str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    return remote


def _seed_managed(home: Path, remote: Path, *, extra_local: bool = False) -> Path:
    src = home / "src"
    subprocess.run(
        ["git", "clone", str(remote), str(src)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(src, "config", "user.email", "test@example.com")
    _git(src, "config", "user.name", "test")
    if extra_local:
        _write_commit(src, "local-only.txt", "local\n", "local-only commit")
    return src


def _curl_installer(
    tmp: Path,
    *args: str,
    home: Path,
    remote: Path,
    path_prefix: list[Path] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run install.sh from a path without a parent Cargo.toml (curl|bash mode)."""
    boot = tmp / "boot-scripts"
    boot.mkdir(parents=True, exist_ok=True)
    script = boot / "install.sh"
    shutil.copy2(INSTALLER, script)

    path_parts = [str(p) for p in (path_prefix or [])]
    path_parts.append(os.environ.get("PATH", ""))
    env = os.environ.copy()
    env.update(
        {
            "HIPFIRE_HOME": str(home),
            "HIPFIRE_GITHUB_URL": str(remote),
            "PATH": os.pathsep.join(path_parts),
            # Force non-interactive channel default if needed.
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", str(script), *args],
        cwd=str(tmp),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _make_failing_cargo(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    cargo = bin_dir / "cargo"
    cargo.write_text(
        "#!/bin/bash\n"
        "# Synthetic cargo: mutate worktree then fail before any real build.\n"
        "if [ -n \"${CARGO_TARGET_DIR:-}\" ]; then\n"
        "  root=\"$(dirname \"$CARGO_TARGET_DIR\")\"\n"
        "  echo dirtied-by-cargo > \"$root/cargo-mutated.txt\" 2>/dev/null || true\n"
        "fi\n"
        "echo 'synthetic cargo failure' >&2\n"
        "exit 42\n",
        encoding="utf-8",
    )
    cargo.chmod(0o755)


class InstallRevisionTests(unittest.TestCase):
    def test_help_documents_branch_install_without_touching_hardware(self):
        result = installer("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--branch NAME", result.stdout)
        self.assertIn("bash -s -- --branch beta", result.stdout)

    def test_multiple_revision_selectors_fail_before_hardware_probe(self):
        result = installer("--branch", "beta", "--tag", "v0.3.0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("choose only one", result.stderr)
        self.assertNotIn("Checking for AMD GPU", result.stdout)
        self.assertNotIn("cargo build", result.stdout + result.stderr)

    def test_unsafe_ref_fails_before_hardware_probe(self):
        result = installer("--ref", "../beta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe or invalid", result.stderr)
        self.assertNotIn("Checking for AMD GPU", result.stdout)
        self.assertNotIn("cargo build", result.stdout + result.stderr)

    def test_option_like_branch_fails_before_build(self):
        result = installer("--branch", "--yes")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe or invalid", result.stderr)

    def test_commit_selector_requires_hex(self):
        result = installer("--commit", "beta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal git commit", result.stderr)

    def test_commit_selector_rejects_short_or_long_hex(self):
        short = installer("--commit", "abc123")
        self.assertEqual(short.returncode, 2)
        self.assertIn("hexadecimal git commit", short.stderr)
        long = installer("--commit", "a" * 41)
        self.assertEqual(long.returncode, 2)
        self.assertIn("hexadecimal git commit", long.stderr)

    def test_install_ref_env_is_generic_ref_when_cli_absent(self):
        # Static contract: env fills --ref semantics only when no CLI selector,
        # and only after CLI arg parsing has finished counting selectors.
        # Match the executable gate (not usage/help text mentioning the env var).
        env_gate = (
            'if [ "$SELECTOR_COUNT" -eq 0 ] && [ -n "${HIPFIRE_INSTALL_REF:-}" ]; then'
        )
        env_at = INSTALLER_TEXT.find(env_gate)
        self.assertNotEqual(env_at, -1, "missing SELECTOR_COUNT==0 HIPFIRE_INSTALL_REF gate")
        gate_block = INSTALLER_TEXT[env_at : env_at + 200]
        self.assertIn('SELECTOR="$HIPFIRE_INSTALL_REF"', gate_block)
        self.assertIn('SELECTOR_KIND="ref"', gate_block)

        # Argument-parsing while-loop must complete before env consumption.
        while_at = INSTALLER_TEXT.find('while [ "$#" -gt 0 ]; do')
        self.assertNotEqual(while_at, -1)
        self.assertLess(while_at, env_at)
        done_at = INSTALLER_TEXT.find("\ndone\n", while_at)
        self.assertNotEqual(done_at, -1)
        self.assertLess(done_at, env_at)

        # SELECTOR_COUNT is incremented during CLI parsing, before the env gate.
        inc_at = INSTALLER_TEXT.find(
            "SELECTOR_COUNT=$((SELECTOR_COUNT + 1))", while_at, done_at
        )
        self.assertNotEqual(inc_at, -1)

    def test_cli_selector_wins_over_install_ref_env(self):
        # Invalid CLI selector must still fail even when env would be valid.
        result = installer(
            "--ref",
            "../beta",
            env={"HIPFIRE_INSTALL_REF": "master"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe or invalid", result.stderr)

    def test_cargo_target_dir_pinned_to_repo_target(self):
        # Bootstrap must force artifacts into <source>/target regardless of ambient config.
        self.assertIn('export CARGO_TARGET_DIR="$REPO_DIR/target"', INSTALLER_TEXT)
        self.assertIn('HIPFIRE_BIN="$REPO_DIR/target/release/hipfire"', INSTALLER_TEXT)
        # Pin happens before the cargo build step.
        pin_at = INSTALLER_TEXT.index('export CARGO_TARGET_DIR="$REPO_DIR/target"')
        build_at = INSTALLER_TEXT.index("cargo build --release -p hipfire-cli")
        self.assertLess(pin_at, build_at)

    def test_invalid_env_ref_fails_without_home_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "hipfire-home"
            result = installer(
                env={
                    "HIPFIRE_HOME": str(home),
                    "HIPFIRE_INSTALL_REF": "../beta",
                },
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsafe or invalid", result.stderr)
            self.assertFalse(home.exists())

    def test_github_url_override_default(self):
        self.assertIn(
            'GITHUB_URL="${HIPFIRE_GITHUB_URL:-https://github.com/warpfront/hipfire.git}"',
            INSTALLER_TEXT,
        )

    def test_managed_dirty_tree_refused_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            remote = _init_remote(tmp)
            home = tmp / "home"
            src = _seed_managed(home, remote)
            prior_head = _git(src, "rev-parse", "HEAD").stdout.strip()
            prior_branch = _git(src, "symbolic-ref", "--short", "HEAD").stdout.strip()
            # Dirty tracked + untracked work.
            (src / "README").write_text("dirty tracked\n", encoding="utf-8")
            (src / "untracked.dat").write_text("u\n", encoding="utf-8")

            bins = tmp / "bins"
            _make_failing_cargo(bins)

            result = _curl_installer(
                tmp,
                "--branch",
                "master",
                "--yes",
                home=home,
                remote=remote,
                path_prefix=[bins],
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("dirty", result.stderr.lower())
            self.assertEqual(_git(src, "rev-parse", "HEAD").stdout.strip(), prior_head)
            self.assertEqual(
                _git(src, "symbolic-ref", "--short", "HEAD").stdout.strip(),
                prior_branch,
            )
            # Cargo must not have been invoked (failing cargo would leave marker).
            self.assertFalse((src / "cargo-mutated.txt").exists())

    def test_managed_ahead_target_refused_before_cargo(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            remote = _init_remote(tmp)
            home = tmp / "home"
            src = _seed_managed(home, remote, extra_local=True)
            prior_head = _git(src, "rev-parse", "HEAD").stdout.strip()
            prior_master = _git(src, "rev-parse", "refs/heads/master").stdout.strip()
            self.assertEqual(prior_head, prior_master)

            bins = tmp / "bins"
            _make_failing_cargo(bins)

            result = _curl_installer(
                tmp,
                "--branch",
                "master",
                "--yes",
                home=home,
                remote=remote,
                path_prefix=[bins],
            )
            combined = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, combined)
            self.assertIn("not contained", result.stderr)
            self.assertEqual(_git(src, "rev-parse", "HEAD").stdout.strip(), prior_head)
            self.assertEqual(
                _git(src, "rev-parse", "refs/heads/master").stdout.strip(),
                prior_master,
            )
            self.assertFalse((src / "cargo-mutated.txt").exists())
            self.assertNotIn("synthetic cargo failure", combined)

    def test_managed_restores_after_cargo_failure(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            remote = _init_remote(tmp)
            home = tmp / "home"
            # Seed managed src at the original remote tip first so checkout must move.
            src = _seed_managed(home, remote)
            prior_head = _git(src, "rev-parse", "HEAD").stdout.strip()
            prior_master = _git(src, "rev-parse", "refs/heads/master").stdout.strip()
            prior_origin = _git(src, "remote", "get-url", "origin").stdout.strip()

            # Advance remote master so local is behind (ancestor-ok, must checkout then fail).
            remote_work = tmp / "remote-advance"
            subprocess.run(
                ["git", "clone", str(remote), str(remote_work)],
                check=True,
                capture_output=True,
                text=True,
            )
            _git(remote_work, "config", "user.email", "test@example.com")
            _git(remote_work, "config", "user.name", "test")
            remote_tip = _write_commit(
                remote_work, "upstream.txt", "new\n", "upstream advance"
            )
            subprocess.run(
                ["git", "push", "origin", "master"],
                cwd=str(remote_work),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(prior_head, remote_tip)

            bins = tmp / "bins"
            _make_failing_cargo(bins)

            result = _curl_installer(
                tmp,
                "--branch",
                "master",
                "--yes",
                home=home,
                remote=remote,
                path_prefix=[bins],
            )
            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 42, combined)
            self.assertIn("synthetic cargo failure", combined)

            # Branch/HEAD/target-ref restored to pre-install checkpoint.
            self.assertEqual(_git(src, "rev-parse", "HEAD").stdout.strip(), prior_head)
            self.assertEqual(
                _git(src, "symbolic-ref", "--short", "HEAD").stdout.strip(),
                "master",
            )
            self.assertEqual(
                _git(src, "rev-parse", "refs/heads/master").stdout.strip(),
                prior_master,
            )
            self.assertEqual(
                _git(src, "remote", "get-url", "origin").stdout.strip(),
                prior_origin,
            )
            # Post-checkout cargo mutation discarded.
            self.assertFalse((src / "cargo-mutated.txt").exists())
            self.assertFalse((src / "upstream.txt").exists())


if __name__ == "__main__":
    unittest.main()

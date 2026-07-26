#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
UNINSTALLER = REPO / "scripts" / "uninstall.sh"


def uninstaller(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["SHELL"] = "/bin/bash"
    env.pop("SUDO_USER", None)
    return subprocess.run(
        ["bash", str(UNINSTALLER), *args],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def make_managed_source(
    source: Path,
    dirty: bool = False,
    remote: str = "https://github.com/warpfront/hipfire.git",
) -> None:
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "remote",
            "add",
            "origin",
            remote,
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    tracked = source / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "update-ref", "refs/remotes/origin/master", "HEAD"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "branch", "--set-upstream-to=origin/master"],
        check=True,
        capture_output=True,
    )
    if dirty:
        tracked.write_text("local work\n")


class UninstallTests(unittest.TestCase):
    def test_help_has_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install = home / ".hipfire"
            (install / "bin").mkdir(parents=True)

            result = uninstaller(home, "--help")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--purge", result.stdout)
            self.assertTrue((install / "bin").is_dir())

    def test_default_removes_program_and_preserves_user_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install = home / ".hipfire"
            (install / "bin").mkdir(parents=True)
            (install / "bin" / "hipfire").write_text("binary fixture\n")
            (install / "models").mkdir()
            (install / "models" / "model.mq4").write_text("model fixture\n")
            (install / "config.toml").write_text("schema_version = 1\n")
            make_managed_source(install / "src")
            bashrc = home / ".bashrc"
            bashrc.write_text(
                'export EDITOR=vi\n\n# hipfire\nexport PATH="$HOME/.hipfire/bin:$PATH"\n'
            )

            result = uninstaller(home)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((install / "bin").exists())
            self.assertFalse((install / "src").exists())
            self.assertTrue((install / "models" / "model.mq4").is_file())
            self.assertTrue((install / "config.toml").is_file())
            self.assertEqual(bashrc.read_text(), "export EDITOR=vi\n\n")

    def test_default_removes_source_cloned_before_the_org_transfer(self) -> None:
        # The canonical repo moved Kaden-Schutt/hipfire -> warpfront/hipfire.
        # Installs predating the transfer still have the old origin; uninstall
        # must still recognize them as managed or it silently leaves them behind.
        for legacy_remote in (
            "https://github.com/Kaden-Schutt/hipfire.git",
            "git@github.com:Kaden-Schutt/hipfire",
        ):
            with self.subTest(remote=legacy_remote):
                with tempfile.TemporaryDirectory() as tmp:
                    home = Path(tmp)
                    install = home / ".hipfire"
                    (install / "bin").mkdir(parents=True)
                    (install / "bin" / "hipfire").write_text("binary fixture\n")
                    make_managed_source(install / "src", remote=legacy_remote)

                    result = uninstaller(home)

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse((install / "src").exists())

    def test_default_preserves_source_with_local_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            source = home / ".hipfire" / "src"
            make_managed_source(source, dirty=True)

            result = uninstaller(home)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(source.is_dir())
            self.assertIn("contains local work", result.stdout)

    def test_dry_run_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install = home / ".hipfire"
            (install / "bin").mkdir(parents=True)
            (install / "models").mkdir()
            bashrc = home / ".bashrc"
            bashrc.write_text('# hipfire\nexport PATH="$HOME/.hipfire/bin:$PATH"\n')

            result = uninstaller(home, "--dry-run")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((install / "bin").is_dir())
            self.assertTrue((install / "models").is_dir())
            self.assertIn(".hipfire/bin", bashrc.read_text())
            self.assertIn("nothing was changed", result.stdout)

    def test_purge_yes_removes_only_fake_home_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            install = home / ".hipfire"
            (install / "models").mkdir(parents=True)
            (install / "models" / "model.mq4").write_text("model fixture\n")
            keep = home / "keep.txt"
            keep.write_text("keep\n")

            result = uninstaller(home, "--purge", "--yes")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(install.exists())
            self.assertTrue(keep.is_file())


if __name__ == "__main__":
    unittest.main()

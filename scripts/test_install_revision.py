#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
INSTALLER = REPO / "scripts" / "install.sh"


def installer(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )


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

    def test_unsafe_ref_fails_before_hardware_probe(self):
        result = installer("--ref", "../beta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe or invalid", result.stderr)
        self.assertNotIn("Checking for AMD GPU", result.stdout)

    def test_commit_selector_requires_hex(self):
        result = installer("--commit", "beta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("hexadecimal git commit", result.stderr)


if __name__ == "__main__":
    unittest.main()

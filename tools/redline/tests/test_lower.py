# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[3]


class LowerImportAndMainShapeTests(unittest.TestCase):
    def test_lower_module_exports_main(self):
        from tools.redline import lower

        self.assertTrue(callable(lower.main))


class LowerDispatcherTests(unittest.TestCase):
    def run_module(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_package_help_lists_lower(self):
        result = self.run_module("-m", "tools.redline", "-h")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lower", result.stderr)

    def test_unknown_package_subcommand_still_exits_2(self):
        result = self.run_module("-m", "tools.redline", "not-a-real-command")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown subcommand", result.stderr)

    def test_dispatches_lower_into_module_main(self):
        result = self.run_module("-m", "tools.redline", "lower")
        self.assertEqual(result.returncode, 2)
        # Wrapper usage (not package-unknown); after impl, prefix is tools.redline.lower
        self.assertNotIn("unknown subcommand", result.stderr)
        self.assertTrue(
            "lower" in result.stderr.lower() or "kernel" in result.stderr.lower(),
            result.stderr,
        )


class LowerKernelArgvTests(unittest.TestCase):
    def setUp(self):
        from tools.redline import lower as lower_mod

        self.lower = lower_mod

    def test_kernel_forwards_args_with_release_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "target" / "release" / "radiowave"
            release.parent.mkdir(parents=True)
            release.write_text("#!/bin/sh\n", encoding="utf-8")
            release.chmod(0o755)
            completed = MagicMock(returncode=7)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                code = self.lower.main(
                    ["kernel", "compile", "--source", "x.hip", "--arch", "gfx1201"]
                )
            self.assertEqual(code, 7)
            run.assert_called_once()
            args, kwargs = run.call_args
            argv = args[0]
            self.assertEqual(argv[0], str(release))
            self.assertEqual(
                argv[1:],
                ["compile", "--source", "x.hip", "--arch", "gfx1201"],
            )
            self.assertEqual(kwargs.get("cwd"), root)
            self.assertFalse(kwargs.get("check", False))
            # Inherited stdio: do not capture or redirect child streams.
            for forbidden in (
                "capture_output",
                "input",
                "stdin",
                "stdout",
                "stderr",
            ):
                self.assertNotIn(forbidden, kwargs)


    def test_kernel_radiowave_override_stripped_and_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Competing default release binary must lose to explicit --radiowave.
            release = root / "target" / "release" / "radiowave"
            release.parent.mkdir(parents=True)
            release.write_text("#!/bin/sh\n# release\n", encoding="utf-8")
            release.chmod(0o755)
            rw = root / "custom-radiowave"
            rw.write_text("#!/bin/sh\n# override\n", encoding="utf-8")
            rw.chmod(0o755)
            completed = MagicMock(returncode=0)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                code = self.lower.main(
                    ["kernel", "--radiowave", str(rw), "inspect", "--input", "a.hsaco"]
                )
            self.assertEqual(code, 0)
            argv = run.call_args.args[0]
            self.assertEqual(argv[0], str(rw))
            self.assertNotEqual(argv[0], str(release))
            self.assertEqual(argv[1:], ["inspect", "--input", "a.hsaco"])
            self.assertNotIn("--radiowave", argv)


    def test_kernel_prefers_release_over_debug_over_cargo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "target" / "release" / "radiowave"
            debug = root / "target" / "debug" / "radiowave"
            release.parent.mkdir(parents=True)
            debug.parent.mkdir(parents=True)
            release.write_text("r", encoding="utf-8")
            debug.write_text("d", encoding="utf-8")
            completed = MagicMock(returncode=0)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                self.lower.main(["kernel", "inspect", "--input", "x"])
            self.assertEqual(run.call_args.args[0][0], str(release))

    def test_kernel_debug_when_no_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            debug = root / "target" / "debug" / "radiowave"
            debug.parent.mkdir(parents=True)
            debug.write_text("d", encoding="utf-8")
            completed = MagicMock(returncode=0)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                self.lower.main(["kernel", "inspect", "--input", "x"])
            self.assertEqual(run.call_args.args[0][0], str(debug))

    def test_kernel_cargo_fallback_when_no_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = MagicMock(returncode=0)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                self.lower.main(["kernel", "compile", "--source", "x.hip"])
            argv = run.call_args.args[0]
            self.assertEqual(
                argv[:5],
                ["cargo", "run", "-q", "-p", "radiowave"],
            )
            self.assertEqual(argv[5], "--")
            self.assertEqual(argv[6:], ["compile", "--source", "x.hip"])

    def test_kernel_missing_radiowave_path_exits_2(self):
        missing = REPO / "definitely-missing-radiowave-binary-for-test"
        err = io.StringIO()
        with patch.object(sys, "stderr", err):
            code = self.lower.main(
                ["kernel", "--radiowave", str(missing), "compile"]
            )
        self.assertEqual(code, 2)
        msg = err.getvalue()
        self.assertIn("tools.redline.lower:", msg)
        self.assertIn(str(missing), msg)

    def test_kernel_spawn_oserror_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # No binaries → cargo prefix; force general spawn OSError
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(
                    self.lower.subprocess,
                    "run",
                    side_effect=OSError(2, "No such file or directory", "cargo"),
                ),
                patch.object(sys, "stderr", io.StringIO()) as err,
            ):
                code = self.lower.main(["kernel", "compile"])
            self.assertEqual(code, 2)
            msg = err.getvalue()
            self.assertIn("tools.redline.lower:", msg)
            self.assertIn("cargo", msg)



class LowerPm4ArgvTests(unittest.TestCase):
    def setUp(self):
        from tools.redline import lower as lower_mod

        self.lower = lower_mod

    def test_pm4_inserts_flag_and_forwards_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = root / "scripts" / "redline_daemon_harness.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("# harness\n", encoding="utf-8")
            completed = MagicMock(returncode=3)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                code = self.lower.main(
                    ["pm4", "--model", "M", "--daemon", "D", "--prefix", "32"]
                )
            self.assertEqual(code, 3)
            args, kwargs = run.call_args
            argv = args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1], str(harness))
            self.assertEqual(argv[2], "--pm4")
            self.assertEqual(
                argv[3:],
                ["--model", "M", "--daemon", "D", "--prefix", "32"],
            )
            self.assertEqual(argv.count("--pm4"), 1)
            self.assertEqual(kwargs.get("cwd"), root)
            # Inherited stdio: do not capture or redirect child streams.
            for forbidden in (
                "capture_output",
                "input",
                "stdin",
                "stdout",
                "stderr",
            ):
                self.assertNotIn(forbidden, kwargs)

    def test_pm4_caller_provided_flag_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = root / "scripts" / "redline_daemon_harness.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("# harness\n", encoding="utf-8")
            completed = MagicMock(returncode=0)
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(self.lower.subprocess, "run", return_value=completed) as run,
            ):
                # Duplicate caller --pm4 flags must collapse to exactly one.
                self.lower.main(
                    [
                        "pm4",
                        "--pm4",
                        "--model",
                        "M",
                        "--pm4",
                        "--daemon",
                        "D",
                        "--prefix",
                        "32",
                    ]
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv[0], sys.executable)
            self.assertEqual(argv[1], str(harness))
            self.assertEqual(argv.count("--pm4"), 1)
            self.assertEqual(argv[2], "--pm4")
            # Non-flag args preserved in caller order after the single --pm4.
            self.assertEqual(
                argv[3:],
                ["--model", "M", "--daemon", "D", "--prefix", "32"],
            )


    def test_pm4_missing_harness_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            err = io.StringIO()
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(sys, "stderr", err),
            ):
                code = self.lower.main(["pm4", "--model", "M"])
            self.assertEqual(code, 2)
            msg = err.getvalue()
            self.assertIn("tools.redline.lower:", msg)
            self.assertIn("redline_daemon_harness.py", msg)

    def test_pm4_spawn_oserror_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = root / "scripts" / "redline_daemon_harness.py"
            harness.parent.mkdir(parents=True)
            harness.write_text("# harness\n", encoding="utf-8")
            err = io.StringIO()
            with (
                patch.object(self.lower, "REPO", root),
                patch.object(
                    self.lower.subprocess,
                    "run",
                    side_effect=OSError(13, "Permission denied", sys.executable),
                ),
                patch.object(sys, "stderr", err),
            ):
                code = self.lower.main(["pm4", "--model", "M"])
            self.assertEqual(code, 2)
            msg = err.getvalue()
            self.assertIn("tools.redline.lower:", msg)
            self.assertTrue(
                sys.executable in msg
                or "Permission denied" in msg
                or "redline_daemon_harness" in msg,
                msg,
            )



class LowerModeTests(unittest.TestCase):
    def setUp(self):
        from tools.redline import lower as lower_mod

        self.lower = lower_mod

    def test_bare_lower_usage_exit_2(self):
        err = io.StringIO()
        with patch.object(sys, "stderr", err):
            code = self.lower.main([])
        self.assertEqual(code, 2)
        self.assertTrue(err.getvalue().strip())

    def test_unknown_mode_exit_2(self):
        err = io.StringIO()
        with patch.object(sys, "stderr", err):
            code = self.lower.main(["foo"])
        self.assertEqual(code, 2)
        self.assertIn("unknown mode", err.getvalue())
        self.assertIn("foo", err.getvalue())


class LowerHelpSmokeTests(unittest.TestCase):
    """Delegation wiring only; no GPU. Child may be cargo or binary."""

    def run_lower(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "tools.redline", "lower", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    def test_kernel_help_delegates_without_gpu(self):
        result = self.run_lower("kernel", "-h")
        # Must reach real child help: not package unknown-subcommand, not empty.
        self.assertNotIn("unknown subcommand", result.stderr)
        self.assertEqual(
            result.returncode,
            0,
            f"rc={result.returncode} out={result.stdout!r} err={result.stderr!r}",
        )
        combined = ((result.stdout or "") + (result.stderr or "")).lower()
        self.assertTrue(
            "usage" in combined
            or "help" in combined
            or "radiowave" in combined
            or "compile" in combined
            or "inspect" in combined,
            f"rc={result.returncode} out={result.stdout!r} err={result.stderr!r}",
        )

    def test_pm4_help_delegates_without_gpu(self):
        result = self.run_lower("pm4", "-h")
        self.assertNotIn("unknown subcommand", result.stderr)
        self.assertEqual(
            result.returncode,
            0,
            f"rc={result.returncode} out={result.stdout!r} err={result.stderr!r}",
        )
        combined = (result.stdout or "") + (result.stderr or "")
        combined_l = combined.lower()
        # Harness argparse help must surface; reject generic errors / empty.
        self.assertTrue(
            "--pm4" in combined
            or "usage" in combined_l
            or "help" in combined_l
            or "model" in combined_l
            or "daemon" in combined_l,
            f"rc={result.returncode} out={result.stdout!r} err={result.stderr!r}",
        )


if __name__ == "__main__":
    unittest.main()

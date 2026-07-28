# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from tools.redline import serve_diff


REPO = Path(__file__).resolve().parents[3]


class ServeToolsEntrypointTests(unittest.TestCase):
    def run_module(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args, "--help"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_tools_serve_harness_runs_existing_cli(self):
        result = self.run_module("-m", "tools.serve_harness")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--session", result.stdout)

    def test_redline_dispatches_serve_diff(self):
        result = self.run_module("-m", "tools.redline", "serve-diff")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--session", result.stdout)
        self.assertIn("--thinking", result.stdout)
        self.assertIn("--max-tokens", result.stdout)
        self.assertIn("--max-seq", result.stdout)


class ServeDiffValidationTests(unittest.TestCase):
    @staticmethod
    def turns() -> list[dict]:
        turns = [{"content": f"prompt {index}"} for index in range(1, 9)]
        turns[6]["expect"] = ["dedupe", "hash", "Result"]
        turns[7]["expect"] = ["dedupe", "rayon", "chunk"]
        return turns

    @staticmethod
    def rows() -> list[dict]:
        rows = []
        for index in range(1, 9):
            content = f"coherent answer {index}"
            if index == 7:
                content = "The dedupe hash function returns a Result."
            elif index == 8:
                content = "The dedupe design uses rayon and content-defined chunk boundaries."
            rows.append(
                {
                    "finish": "stop",
                    "assistant_content": content,
                    "empty": False,
                    "runaway": False,
                    "attractor": False,
                    "ctx": index * 100,
                    "gen": 64,
                    "decode_tok_s": 100.0 - index,
                }
            )
        return rows

    def test_accepts_exact_coherent_hip_pm4_match(self):
        validator = getattr(serve_diff, "validate_comparison", None)
        self.assertIsNotNone(validator, "serve diff comparison validator is missing")
        report = validator(
            self.turns(),
            self.rows(),
            self.rows(),
            {"observed": False, "transport": None, "position": None, "lines": []},
            {"observed": True, "transport": "pm4", "position": 1, "lines": ["proof"]},
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["turns"], 8)
        self.assertEqual(report["matched_turns"], 8)

    def test_rejects_sampled_output_divergence(self):
        validator = getattr(serve_diff, "validate_comparison", None)
        self.assertIsNotNone(validator, "serve diff comparison validator is missing")
        hip_rows = self.rows()
        pm4_rows = self.rows()
        pm4_rows[7]["assistant_content"] = "different sampled answer"
        report = validator(
            self.turns(),
            hip_rows,
            pm4_rows,
            {"observed": False, "transport": None, "position": None, "lines": []},
            {"observed": True, "transport": "pm4", "position": 1, "lines": ["proof"]},
        )
        self.assertFalse(report["valid"])
        self.assertIn("turn 8: sampled output differs between HIP and PM4", report["errors"])


if __name__ == "__main__":
    unittest.main()

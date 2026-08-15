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
                    "request_id": f"chatcmpl-turn-{index}",
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

    @staticmethod
    def pm4_route(request_ids: list[str] | None = None) -> dict:
        ids = request_ids or [f"chatcmpl-turn-{index}" for index in range(1, 9)]
        hits = [
            {
                "transport": "pm4",
                "position": index,
                "request_id": request_id,
                "replays": 64,
                "line": (
                    "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 "
                    f"position={index} request_id={request_id} replays=64"
                ),
            }
            for index, request_id in enumerate(ids, 1)
        ]
        return {
            "observed": bool(hits),
            "transport": "pm4" if hits else None,
            "position": hits[0]["position"] if hits else None,
            "lines": [hit["line"] for hit in hits],
            "count": len(hits),
            "hits": hits,
        }

    def test_accepts_exact_coherent_hip_pm4_match(self):
        validator = getattr(serve_diff, "validate_comparison", None)
        self.assertIsNotNone(validator, "serve diff comparison validator is missing")
        report = validator(
            self.turns(),
            self.rows(),
            self.rows(),
            {"observed": False, "transport": None, "position": None, "lines": [], "count": 0, "hits": []},
            self.pm4_route(),
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["turns"], 8)
        self.assertEqual(report["matched_turns"], 8)

    def test_accepts_multi_turn_valid_bound_markers(self):
        report = serve_diff.validate_comparison(
            self.turns(),
            self.rows(),
            self.rows(),
            {
                "observed": False,
                "transport": None,
                "position": None,
                "lines": [],
                "count": 0,
                "hits": [],
            },
            self.pm4_route(),
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["matched_turns"], 8)

    def test_rejects_duplicate_marker_masking_a_missing_turn(self):
        request_ids = [f"chatcmpl-turn-{index}" for index in range(1, 8)]
        request_ids.append("chatcmpl-turn-7")
        report = serve_diff.validate_comparison(
            self.turns(),
            self.rows(),
            self.rows(),
            {
                "observed": False,
                "transport": None,
                "position": None,
                "lines": [],
                "count": 0,
                "hits": [],
            },
            self.pm4_route(request_ids),
        )
        self.assertFalse(report["valid"])
        self.assertTrue(
            any(
                "expected exactly one PM4 route proof marker for request_id "
                "'chatcmpl-turn-7'"
                in error
                for error in report["errors"]
            ),
            report["errors"],
        )
        self.assertTrue(
            any(
                "expected exactly one PM4 route proof marker for request_id "
                "'chatcmpl-turn-8'"
                in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_rejects_multi_turn_extra_other_transport(self):
        route = self.pm4_route()
        route["hits"].append(
            {
                "transport": "aql",
                "position": 99,
                "request_id": "chatcmpl-turn-1",
                "replays": 1,
                "line": (
                    "HIPFIRE_REPLAY_ROUTE_PROOF transport=aql position=99 "
                    "request_id=chatcmpl-turn-1 replays=1"
                ),
            }
        )
        route["count"] = len(route["hits"])
        route["lines"] = [hit["line"] for hit in route["hits"]]
        report = serve_diff.validate_comparison(
            self.turns(),
            self.rows(),
            self.rows(),
            {
                "observed": False,
                "transport": None,
                "position": None,
                "lines": [],
                "count": 0,
                "hits": [],
            },
            route,
        )
        self.assertFalse(report["valid"])
        self.assertTrue(
            any("non-PM4" in error for error in report["errors"]),
            report["errors"],
        )
        self.assertTrue(
            any("exactly 8 route proof marker(s) total, got 9" in error for error in report["errors"]),
            report["errors"],
        )

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
            {"observed": False, "transport": None, "position": None, "lines": [], "count": 0, "hits": []},
            self.pm4_route(),
        )
        self.assertFalse(report["valid"])
        self.assertIn("turn 8: sampled output differs between HIP and PM4", report["errors"])


if __name__ == "__main__":
    unittest.main()

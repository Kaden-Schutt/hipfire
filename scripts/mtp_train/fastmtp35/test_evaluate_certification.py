#!/usr/bin/env python3
"""Unit tests for the fail-closed FastMTP promotion evaluator."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_certification import evaluate


def rows(rate: float, tau: float | None, *, candidate: bool = False):
    result = []
    for turn in range(8):
        row = {
            "ctx": 1000 + turn * 100,
            "cached": 0 if turn == 0 else 900 + turn * 100,
            "decode_tok_s": rate,
            "decode_estimated": False,
            "tau": tau,
            "runaway": False,
            "empty": False,
            "attractor": False,
            "recall_hits": 0,
            "recall_total": 0,
        }
        if candidate and turn >= 6:
            row["recall_hits"] = 3
            row["recall_total"] = 3
        result.append(row)
    return result


class CertificationTest(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        artifact = {
            "path": "/fixture",
            "bytes": 1,
            "sha256": "a" * 64,
        }
        manifest = {
            "schema_version": 1,
            "producer_git_commit": "f" * 40,
            "trunk": artifact,
            "stock_mtp": {**artifact, "sha256": "b" * 64},
            "candidate_mtp": {**artifact, "sha256": "c" * 64},
            "session": {**artifact, "sha256": "d" * 64},
            "contract": {
                "turns": 8,
                "sampling": "registry",
                "thinking": "med",
                "max_tokens": 4096,
                "kv_mode": "q8",
                "mtp_k": 3,
                "redline_shadow_iterations": 15,
            },
        }
        (root / "certification-manifest.json").write_text(json.dumps(manifest))
        (root / "ar.json").write_text(json.dumps(rows(170.0, None)))
        (root / "stock-mtp.json").write_text(json.dumps(rows(150.0, 2.2)))
        (root / "candidate-mtp.json").write_text(
            json.dumps(rows(180.0, 2.8, candidate=True))
        )
        (root / "redline-shadow.json").write_text(json.dumps({"pass": True}))
        return temp, root

    def test_complete_promotion_contract_passes(self):
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        summary = evaluate(root)
        self.assertTrue(summary["promotion_pass"], summary["failures"])

    def test_missing_prefix_reuse_fails(self):
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        candidate = json.loads((root / "candidate-mtp.json").read_text())
        candidate[4]["cached"] = 0
        (root / "candidate-mtp.json").write_text(json.dumps(candidate))
        summary = evaluate(root)
        self.assertFalse(summary["promotion_pass"])
        self.assertTrue(any("prefix reuse" in item for item in summary["failures"]))

    def test_failed_recall_fails(self):
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        candidate = json.loads((root / "candidate-mtp.json").read_text())
        candidate[-1]["recall_hits"] = 2
        (root / "candidate-mtp.json").write_text(json.dumps(candidate))
        summary = evaluate(root)
        self.assertFalse(summary["promotion_pass"])
        self.assertTrue(any("recall" in item for item in summary["failures"]))


if __name__ == "__main__":
    unittest.main()

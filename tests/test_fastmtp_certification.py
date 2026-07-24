import json
import sys
import tempfile
import unittest
from pathlib import Path


FAST_MTP = Path(__file__).parents[1] / "scripts" / "mtp_train" / "fastmtp35"
sys.path.insert(0, str(FAST_MTP))

from evaluate_certification import evaluate  # noqa: E402


def rows(rate: float, tau: float | None) -> list[dict]:
    return [
        {
            "decode_tok_s": rate,
            "decode_estimated": False,
            "tau": tau,
            "runaway": False,
            "empty": False,
            "attractor": False,
        }
        for _ in range(8)
    ]


class FastMtpCertificationTest(unittest.TestCase):
    def write_fixture(
        self,
        root: Path,
        *,
        ar_rate: float = 200.0,
        stock_rate: float = 180.0,
        candidate_rate: float = 220.0,
        stock_tau: float = 3.0,
        candidate_tau: float = 5.0,
        redline_pass: bool = True,
    ) -> None:
        fixtures = {
            "ar": rows(ar_rate, None),
            "stock-mtp": rows(stock_rate, stock_tau),
            "candidate-mtp": rows(candidate_rate, candidate_tau),
        }
        for label, body in fixtures.items():
            (root / f"{label}.json").write_text(json.dumps(body))
        (root / "redline-shadow.json").write_text(
            json.dumps({"pass": redline_pass})
        )

    def test_accepts_candidate_that_clears_every_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_fixture(root)
            result = evaluate(root)
            self.assertTrue(result["promotion_pass"])
            self.assertEqual(result["failures"], [])

    def test_fails_closed_on_quality_perf_tau_and_redline_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_fixture(
                root,
                candidate_rate=170.0,
                candidate_tau=2.0,
                redline_pass=False,
            )
            candidate = json.loads((root / "candidate-mtp.json").read_text())
            candidate[0]["attractor"] = True
            candidate[1]["decode_estimated"] = True
            (root / "candidate-mtp.json").write_text(json.dumps(candidate))

            result = evaluate(root)
            self.assertFalse(result["promotion_pass"])
            failures = "\n".join(result["failures"])
            self.assertIn("native decode measurements", failures)
            self.assertIn("attractor", failures)
            self.assertIn("does not beat ar", failures)
            self.assertIn("does not beat stock", failures)
            self.assertIn("Redline shadow/parity", failures)


if __name__ == "__main__":
    unittest.main()

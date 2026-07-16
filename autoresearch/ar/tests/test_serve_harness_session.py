import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "serve_harness", REPO / "scripts" / "serve_harness.py"
)
serve_harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve_harness)


class ServeHarnessSessionTest(unittest.TestCase):
    def test_committed_eight_turn_session_schema(self):
        rows = serve_harness.load_session(
            REPO / "benchmarks" / "prompts" / "session_coding_8turn.json"
        )
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["content"] for row in rows))

    def test_session_rejects_battery_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"prompt":"wrong schema"}]')
            with self.assertRaisesRegex(ValueError, "content"):
                serve_harness.load_session(path)


if __name__ == "__main__":
    unittest.main()

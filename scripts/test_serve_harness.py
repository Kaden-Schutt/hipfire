import unittest
from unittest.mock import patch

from scripts import serve_harness


class _SseResponse:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)


def _config():
    return {
        "model": "gemma4-test.mq4",
        "max_tokens": 8,
        "port": 12345,
        "sampling": {},
    }


def _urlopen_for(lines):
    return patch.object(
        serve_harness.urllib.request,
        "urlopen",
        return_value=_SseResponse(lines),
    )


class ServeHarnessSseTests(unittest.TestCase):
    def test_complete_sse_requires_terminal_and_done_marker(self):
        lines = [
            'data: {"id":"req-1","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n',
            'data: {"id":"req-1","choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            'data: {"id":"req-1","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":0}}}\n',
            "data: [DONE]\n",
        ]
        with _urlopen_for(lines):
            row = serve_harness.send(_config(), [{"role": "user", "content": "hello"}])

        self.assertNotIn("error", row)
        self.assertEqual(row["finish"], "stop")
        self.assertEqual(row["ctx"], 11)
        self.assertEqual(row["gen"], 7)
        self.assertEqual(row["cached"], 0)

    def test_premature_eof_returns_failed_row(self):
        lines = [
            'data: {"id":"req-2","choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n',
        ]
        with _urlopen_for(lines):
            row = serve_harness.send(_config(), [{"role": "user", "content": "hello"}])

        self.assertEqual(row["error"], "premature EOF before terminal event")

    def test_missing_done_marker_returns_failed_row(self):
        lines = [
            'data: {"id":"req-3","choices":[{"delta":{},"finish_reason":"stop"}]}\n',
        ]
        with _urlopen_for(lines):
            row = serve_harness.send(_config(), [{"role": "user", "content": "hello"}])

        self.assertEqual(row["error"], "premature EOF before terminal event")


if __name__ == "__main__":
    unittest.main()

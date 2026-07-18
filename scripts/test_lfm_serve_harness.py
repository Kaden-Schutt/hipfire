#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt

import json
import tempfile
import unittest
from pathlib import Path

import lfm_serve_harness as harness


class LfmServeHarnessTests(unittest.TestCase):
    def test_build_sampling_keeps_card_defaults_and_forces_thinking_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({
                "models": {
                    "lfm2.5:350m": {
                        "recommended_settings": {
                            "temperature": 0.1,
                            "top_p": 1.0,
                            "top_k": 50,
                            "repeat_penalty": 1.05,
                        }
                    }
                }
            }))
            self.assertEqual(
                harness.build_sampling(registry, "lfm2.5:350m", "medium"),
                {
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "top_k": 50,
                    "repeat_penalty": 1.05,
                    "reasoning_effort": "medium",
                },
            )

    def test_warmup_failures_are_fatal(self):
        self.assertIsNotNone(harness.FATAL_WARMUP.search("pre-warm failed: bad model"))
        self.assertIsNotNone(harness.FATAL_WARMUP.search("pre-warm load failed: bad model"))

    def test_runtime_defaults_are_per_run(self):
        first = harness.allocate_runtime_defaults(None, None, 0)
        second = harness.allocate_runtime_defaults(None, None, 0)
        self.addCleanup(lambda: __import__("shutil").rmtree(Path(first[0]).parent, ignore_errors=True))
        self.addCleanup(lambda: __import__("shutil").rmtree(Path(second[0]).parent, ignore_errors=True))
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])
        self.assertGreater(first[2], 0)
        self.assertGreater(second[2], 0)

    def test_reasoning_effort_choices_match_the_serve_api(self):
        self.assertNotIn("max", harness.EFFORTS)

    def test_validate_rejects_reasoning_only_completion(self):
        # The framing bug: whole answer in reasoning_content, content empty.
        # The content-channel guard (unconditional) must reject it — this shape
        # used to be (wrongly) accepted.
        row = {
            "finish": "stop",
            "reasoning_content": "The capital of France is Paris.",
            "content": "",
            "combined_content": "The capital of France is Paris.",
            "attractor": False,
        }
        self.assertEqual(
            harness.validate_turn(row, "Paris"),
            ["missing expected text: Paris"],
        )

    def test_validate_accepts_content_only_completion(self):
        row = {
            "finish": "stop",
            "reasoning_content": "",
            "content": "Paris is the capital of France.",
            "combined_content": "Paris is the capital of France.",
            "attractor": False,
        }
        self.assertEqual(harness.validate_turn(row, "Paris"), [])

    def test_short_answer_is_not_an_attractor(self):
        self.assertFalse(harness.is_attractor("Paris"))

    def test_validate_rejects_empty_or_length_completion(self):
        row = {
            "finish": "length",
            "reasoning_content": "",
            "content": "",
            "combined_content": "",
            "attractor": False,
        }
        self.assertEqual(
            harness.validate_turn(row, "Paris"),
            ["finish_reason=length", "empty reasoning_content + content", "missing expected text: Paris"],
        )

    def test_validate_nothink_rejects_reasoning(self):
        # --nothink (forbid_reasoning): any reasoning row is a failure.
        row = {
            "finish": "stop",
            "reasoning_content": "thinking...",
            "content": "The capital of France is Paris.",
            "combined_content": "thinking...\nThe capital of France is Paris.",
            "attractor": False,
        }
        self.assertEqual(
            harness.validate_turn(row, "Paris", forbid_reasoning=True),
            ["unexpected reasoning_content (--nothink)"],
        )

    def test_validate_nothink_accepts_content_only(self):
        row = {
            "finish": "stop",
            "reasoning_content": "",
            "content": "The capital of France is Paris.",
            "combined_content": "The capital of France is Paris.",
            "attractor": False,
        }
        self.assertEqual(harness.validate_turn(row, "Paris", forbid_reasoning=True), [])

    def test_validate_think_allows_reasoning_with_content(self):
        # Thinking on: reasoning present AND the answer in content is valid.
        row = {
            "finish": "stop",
            "reasoning_content": "Hmm, the capital is Paris.",
            "content": "The capital of France is Paris.",
            "combined_content": "Hmm, the capital is Paris.\nThe capital of France is Paris.",
            "attractor": False,
        }
        self.assertEqual(harness.validate_turn(row, "Paris"), [])


if __name__ == "__main__":
    unittest.main()

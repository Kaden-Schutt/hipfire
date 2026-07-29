#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "redline_dispatch_profile.py"
SPEC = importlib.util.spec_from_file_location("redline_dispatch_profile", SCRIPT)
profile_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_tool)


def synthetic_profile():
    kernels = ["prepare", "gemv", "gemv", "finish"]
    dispatches = [
        {
            "index": index,
            "kernel": kernel,
            "previous_kernel": kernels[index - 1] if index else None,
            "grid": [64 if kernel == "gemv" else 1, 1, 1],
            "block": [32, 1, 1],
            "boundary": {
                "entry_acquire": index == 0,
                "wait_compute_idle": index == 3,
                "acquire_inter_node": index == 3,
                "acquire_vmem": False,
            },
        }
        for index, kernel in enumerate(kernels)
    ]
    spans = [
        [100, 1_000, 2_000, 400],
        [110, 1_100, 2_100, 410],
        [90, 900, 1_900, 390],
    ]
    return {
        "schema_version": 1,
        "type": "redline_dispatch_profile",
        "context_tokens": 128,
        "warmup_replays": 10,
        "sample_replays": len(spans),
        "steady_state": True,
        "exactly_once_per_sample": True,
        "timestamp_semantics": "baseline before stream plus post-dispatch stamps; span i is PM4 after timestamp i through dispatch i",
        "route": {
            "launches": len(dispatches),
            "unique_kernels": 3,
            "sequence_hash": "0123456789abcdef",
            "command_dwords": 80,
            "timestamp_slots": len(dispatches) + 1,
            "queue_id": 7,
        },
        "dispatches": dispatches,
        "samples": [
            {
                "sample": index,
                "host_ns": sum(row) + 500,
                "total_gpu_ns": sum(row),
                "spans_ns": row,
            }
            for index, row in enumerate(spans)
        ],
        "correctness": {"performed": True, "bit_exact": True},
    }


class ProfileTests(unittest.TestCase):
    def test_accepts_steady_exactly_once_profile(self):
        profile_tool.validate_profile(synthetic_profile())

    def test_rejects_non_steady_or_double_execution_contract(self):
        for field in ("steady_state", "exactly_once_per_sample"):
            profile = synthetic_profile()
            profile[field] = False
            with self.assertRaisesRegex(ValueError, "steady-state exactly-once"):
                profile_tool.validate_profile(profile)

    def test_rejects_span_shape_and_total_mismatch(self):
        profile = synthetic_profile()
        profile["samples"][0]["spans_ns"].pop()
        with self.assertRaisesRegex(ValueError, "span length"):
            profile_tool.validate_profile(profile)

        profile = synthetic_profile()
        profile["samples"][0]["total_gpu_ns"] += 100
        with self.assertRaisesRegex(ValueError, "total"):
            profile_tool.validate_profile(profile)

    def test_capture_must_match_profile_route(self):
        profile = synthetic_profile()
        capture = {
            "launches": 4,
            "unique_kernels": 3,
            "sequence_hash": "0123456789abcdef",
        }
        profile_tool.validate_capture(capture, profile)
        capture["sequence_hash"] = "fedcba9876543210"
        with self.assertRaisesRegex(ValueError, "route mismatch"):
            profile_tool.validate_capture(capture, profile)

    def test_analysis_preserves_dispatch_and_kernel_attribution(self):
        analysis = profile_tool.analyze(synthetic_profile())
        dispatches = analysis["dispatches"]
        self.assertEqual(
            [row["median_ns"] for row in dispatches], [100, 1_000, 2_000, 400]
        )
        self.assertEqual(dispatches[2]["rank"], 1)
        self.assertTrue(dispatches[0]["boundary"]["entry_acquire"])
        self.assertFalse(dispatches[0]["boundary"]["acquire_inter_node"])
        self.assertFalse(dispatches[3]["boundary"]["entry_acquire"])
        self.assertTrue(dispatches[3]["boundary"]["wait_compute_idle"])
        self.assertTrue(dispatches[3]["boundary"]["acquire_inter_node"])
        gemv = [row for row in analysis["kernels"] if row["kernel"] == "gemv"]
        self.assertEqual(gemv[0]["dispatch_indices"], [1, 2])
        self.assertEqual(gemv[0]["total_median_ns"], 3_000)


if __name__ == "__main__":
    unittest.main()

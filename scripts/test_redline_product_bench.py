#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import unittest

from redline_product_bench import analyze_stationarity


DEFAULTS = {
    "window": 10,
    "min_runs": 10,
    "confirmation_runs": 10,
    "max_slope_pct": 0.05,
    "max_spread_pct": 1.0,
    "max_median_drift_pct": 0.5,
}


class StationarityTests(unittest.TestCase):
    def test_stable_signal_requires_confirmation(self):
        values = [100.0 + (0.02 if i % 2 else -0.02) for i in range(20)]
        before = analyze_stationarity(values[:19], **DEFAULTS)
        after = analyze_stationarity(values, **DEFAULTS)
        self.assertFalse(before["stationary"])
        self.assertTrue(after["stationary"])
        self.assertEqual(after["candidate"]["at_row"], 10)
        self.assertEqual(after["confirmed_at_row"], 20)

    def test_false_plateau_in_tg128_trace_is_rejected(self):
        # Real gfx1100 retained-PM4 settling trace: an apparent plateau near
        # 210.8 tok/s resumes climbing before settling near 213.4 tok/s.
        values = [
            200.913, 202.040, 202.714, 202.570, 203.417,
            204.519, 204.621, 208.824, 210.386, 210.516,
            210.603, 210.746, 210.597, 210.820, 211.038,
            210.993, 210.876, 211.524, 212.203, 212.174,
            212.563, 212.876, 213.165, 213.506, 213.551,
            213.550, 213.527, 213.431, 213.536, 213.399,
        ]
        values.extend(213.45 + (0.03 if i % 2 else -0.03) for i in range(20))
        result = analyze_stationarity(values, **DEFAULTS)
        self.assertTrue(result["stationary"])
        self.assertGreaterEqual(result["candidate"]["at_row"], 30)
        self.assertGreaterEqual(result["confirmed_at_row"], 40)
        self.assertGreater(len(result["rejections"]), 0)
        self.assertGreater(result["confirmed_window"]["median"], 213.0)

    def test_continuous_ramp_never_passes(self):
        values = [100.0 + i * 0.2 for i in range(60)]
        result = analyze_stationarity(values, **DEFAULTS)
        self.assertFalse(result["stationary"])


if __name__ == "__main__":
    unittest.main()

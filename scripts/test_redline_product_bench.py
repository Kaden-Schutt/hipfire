#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from redline_product_bench import (
    bench_decode_input_identity,
    analyze_stationarity,
    effective_pm4_policy,
    evaluate_fresh_samples,
    fresh_abba_order,
    run_fresh_sample,
    run_fresh_abba,
    validate_route_proof,
)


DEFAULTS = {
    "window": 10,
    "min_runs": 10,
    "confirmation_runs": 10,
    "max_slope_pct": 0.05,
    "max_spread_pct": 1.0,
    "max_median_drift_pct": 0.5,
}

FRESH_IDENTITY = {
    "model": "/models/fixture.mq4",
    "model_bytes": 123,
    "model_md5": "model-md5",
    "daemon": "/bin/daemon",
    "daemon_md5": "daemon-md5",
    "daemon_sha256": "abc123",
    "context": 129,
    "iterations": 8,
    "warmups": 1,
    "warmup_iterations": 8,
    "max_seq": 2048,
    "kv_mode": "q8",
    "transport": "aql",
    "dpm_warmup_secs": 10.0,
    "measurement_input_identity": {
        "prime_md5": "measurement-prime",
        "decode_md5": "measurement-decode",
        "combined_md5": "measurement-combined",
    },
    "warmup_input_identity": {
        "prime_md5": "warmup-prime",
        "decode_md5": "warmup-decode",
        "combined_md5": "warmup-combined",
    },
}


class FreshAbbaTests(unittest.TestCase):
    @staticmethod
    def route_row(backend, tok_s, iterations=8):
        if backend == "hip":
            route = {
                "requested_backend": "hip",
                "transport": "aql",
                "state": "hip",
                "fallback_reason": None,
                "prepared": None,
                "observed": {
                    "count_delta": 0,
                    "first_position": None,
                    "last_position": None,
                },
                "retained_replay_observed": False,
            }
        else:
            route = {
                "requested_backend": "auto",
                "transport": "aql",
                "state": "ready",
                "fallback_reason": None,
                "execution_mode": "plain_ar",
                "prepared": {
                    "dispatches": 286,
                    "packets": 287,
                    "queue_id": 7,
                    "command_dwords": None,
                },
                "sequence": {
                    "launches": 286,
                    "unique_kernels": 11,
                    "hash": "440bb2b5df220117",
                },
                "observed": {
                    "count_delta": iterations,
                    "first_position": 129,
                    "last_position": 129 + iterations - 1,
                },
                "retained_replay_observed": True,
            }
        return {
            "context_tokens": 129,
            "iterations": iterations,
            "tok_s": tok_s,
            "redline_route": route,
        }

    def test_bench_decode_input_identity_matches_canonical_fixture(self):
        measurement = bench_decode_input_identity(127, 1536)
        warmup = bench_decode_input_identity(127, 32)
        self.assertEqual(
            measurement,
            {
                "prime_md5": "0463527adf7db905ef50cf8de73ccde8",
                "decode_md5": "5739859f1585cfd21736e32b86277d06",
                "combined_md5": "d1c2dcade6fe0e60c7d6438b6dad4eb5",
            },
        )
        self.assertEqual(
            warmup,
            {
                "prime_md5": "0463527adf7db905ef50cf8de73ccde8",
                "decode_md5": "e06c556d549dd97778b760886fc8691d",
                "combined_md5": "2c3430ba84b17c63f9118d102b8ddb13",
            },
        )

    @classmethod
    def samples(cls, hip_values, auto_values, cycles=10):
        values = {"hip": iter(hip_values), "auto": iter(auto_values)}
        samples = []
        for ordinal, backend in enumerate(
            fresh_abba_order(cycles), start=1
        ):
            tok_s = next(values[backend])
            measured = cls.route_row(backend, tok_s)
            samples.append(
                {
                    "process_ordinal": ordinal,
                    "cycle": (ordinal - 1) // 4 + 1,
                    "cycle_slot": (ordinal - 1) % 4 + 1,
                    "pid": 1000 + ordinal,
                    "backend": backend,
                    "daemon_process_started": True,
                    "daemon_process_closed": True,
                    "lifecycle": "completed",
                    "loaded": {"type": "loaded"},
                    "warmups": [cls.route_row(backend, tok_s)],
                    "measured_row": measured,
                    "tok_s": tok_s,
                    "input_identity": {
                        "measurement_combined_md5": FRESH_IDENTITY[
                            "measurement_input_identity"
                        ]["combined_md5"],
                        "warmup_combined_md5": FRESH_IDENTITY[
                            "warmup_input_identity"
                        ]["combined_md5"],
                    },
                    "identity": dict(FRESH_IDENTITY),
                    "log_path": f"/logs/sample-{ordinal}.log",
                    "errors": [],
                }
            )
        return samples

    def test_order_is_exact_hip_auto_auto_hip_cycles(self):
        self.assertEqual(
            fresh_abba_order(2),
            ["hip", "auto", "auto", "hip", "hip", "auto", "auto", "hip"],
        )

    def test_complete_stationary_route_valid_report_is_valid(self):
        hip = [100.0 + i * 0.01 for i in range(20)]
        auto = [120.0 + i * 0.02 for i in range(20)]
        decision = evaluate_fresh_samples(
            self.samples(hip, auto),
            10,
            FRESH_IDENTITY,
            DEFAULTS,
            minimum_uplift_pct=5.0,
        )
        self.assertTrue(decision["complete_cycles_only"])
        self.assertTrue(decision["stationarity_decisions"]["hip"]["stationary"])
        self.assertTrue(decision["stationarity_decisions"]["auto"]["stationary"])
        self.assertTrue(decision["valid"], decision["errors"])
        self.assertTrue(decision["descriptive_statistics"]["promotable"])
        comparison = decision["comparison"]
        hip_confirmed = decision["stationarity_decisions"]["hip"][
            "confirmed_window"
        ]["median"]
        auto_confirmed = decision["stationarity_decisions"]["auto"][
            "confirmed_window"
        ]["median"]
        self.assertEqual(
            comparison["estimator"], "confirmed_final_window_median_ratio"
        )
        self.assertEqual(comparison["hip_confirmed_median"], hip_confirmed)
        self.assertEqual(comparison["auto_confirmed_median"], auto_confirmed)
        self.assertAlmostEqual(
            decision["descriptive_statistics"]["speedup"],
            auto_confirmed / hip_confirmed,
        )
    def test_stationary_route_valid_but_subthreshold_uplift_is_rejected(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [104.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        decision = evaluate_fresh_samples(
            self.samples(hip, auto),
            10,
            FRESH_IDENTITY,
            DEFAULTS,
            minimum_uplift_pct=5.0,
        )
        self.assertTrue(decision["stationarity_decisions"]["hip"]["stationary"])
        self.assertTrue(decision["stationarity_decisions"]["auto"]["stationary"])
        self.assertFalse(decision["comparison"]["minimum_uplift_met"])
        self.assertLess(decision["comparison"]["uplift_pct"], 5.0)
        self.assertFalse(decision["valid"])
        self.assertFalse(decision["descriptive_statistics"]["promotable"])


    def test_stationarity_is_required_independently_for_each_arm(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [120.0 + i * 0.3 for i in range(20)]
        decision = evaluate_fresh_samples(
            self.samples(hip, auto), 10, FRESH_IDENTITY, DEFAULTS
        )
        self.assertTrue(decision["stationarity_decisions"]["hip"]["stationary"])
        self.assertFalse(decision["stationarity_decisions"]["auto"]["stationary"])
        self.assertFalse(decision["valid"])

    def test_any_invalid_sample_route_proof_invalidates_report(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [120.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        samples = self.samples(hip, auto)
        samples[1]["measured_row"]["redline_route"]["fallback_reason"] = "hip_fallback"
        decision = evaluate_fresh_samples(
            samples, 10, FRESH_IDENTITY, DEFAULTS
        )
        self.assertFalse(decision["sample_decisions"][1]["valid"])
        self.assertFalse(decision["valid"])
    def test_route_identity_must_match_across_fresh_processes(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [120.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        samples = self.samples(hip, auto)
        for row in (samples[1]["warmups"][0], samples[1]["measured_row"]):
            row["redline_route"]["sequence"]["hash"] = "1111111111111111"
        decision = evaluate_fresh_samples(
            samples, 10, FRESH_IDENTITY, DEFAULTS
        )
        self.assertFalse(decision["cross_process_route_identity"]["valid"])
        self.assertFalse(decision["valid"])

    def test_latest_boundary_blocks_stale_stationarity(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        hip.extend(130.0 + (0.01 if i % 2 else -0.01) for i in range(10))
        auto = [105.0 + i for i in range(10)]
        auto.extend(120.0 + (0.01 if i % 2 else -0.01) for i in range(20))
        decision = evaluate_fresh_samples(
            self.samples(hip, auto, cycles=15),
            15,
            FRESH_IDENTITY,
            DEFAULTS,
        )
        self.assertFalse(decision["stationarity_decisions"]["hip"]["stationary"])
        self.assertTrue(decision["stationarity_decisions"]["auto"]["stationary"])
        self.assertFalse(decision["valid"])


    def test_missing_or_partial_cycle_samples_remain_invalid(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [120.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        decision = evaluate_fresh_samples(
            self.samples(hip, auto)[:-1], 10, FRESH_IDENTITY, DEFAULTS
        )
        self.assertFalse(decision["complete_cycles_only"])
        self.assertFalse(decision["valid"])

    def test_per_sample_input_identity_mismatch_invalidates_report(self):
        hip = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        auto = [120.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        samples = self.samples(hip, auto)
        samples[0]["input_identity"]["measurement_combined_md5"] = "wrong"
        decision = evaluate_fresh_samples(
            samples, 10, FRESH_IDENTITY, DEFAULTS
        )
        self.assertFalse(decision["sample_decisions"][0]["valid"])
        self.assertFalse(decision["valid"])

    def test_each_sample_constructs_and_closes_a_new_daemon(self):
        class FakeDaemon:
            instances = []

            def __init__(
                self,
                binary,
                backend,
                transport,
                log_path,
                timeout,
                kv_mode,
                dpm_warmup_secs,
            ):
                self.backend = backend
                self.proc = SimpleNamespace(pid=2000 + len(self.instances))
                self.closed = False
                self.instances.append(self)

            def request(self, request):
                if request["type"] == "load":
                    return {"type": "loaded"}
                return FreshAbbaTests.route_row(
                    self.backend, 100.0, request["iterations"]
                )

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as work_dir:
            args = SimpleNamespace(
                daemon="/bin/daemon",
                transport="aql",
                work_dir=work_dir,
                timeout=1.0,
                kv_mode="q8",
                dpm_warmup_secs=10.0,
                model="/models/fixture.mq4",
                max_seq=2048,
                context=129,
                warmup_iterations=8,
                warmups=1,
                iterations=8,
            )
            samples = [
                run_fresh_sample(
                    args,
                    backend,
                    ordinal,
                    1,
                    slot,
                    FRESH_IDENTITY,
                    daemon_factory=FakeDaemon,
                )
                for ordinal, (slot, backend) in enumerate(
                    enumerate(fresh_abba_order(1), start=1), start=1
                )
            ]

        self.assertEqual(
            [sample["backend"] for sample in samples],
            ["hip", "auto", "auto", "hip"],
        )
        self.assertEqual(len(FakeDaemon.instances), 4)
        self.assertEqual(len({id(instance) for instance in FakeDaemon.instances}), 4)
        self.assertTrue(all(instance.closed for instance in FakeDaemon.instances))
        self.assertTrue(all(sample["valid"] for sample in samples))
    def test_fresh_campaign_stops_after_first_invalid_sample(self):
        sample = self.samples([100.0] * 20, [120.0] * 20)[0]
        sample["errors"] = ["synthetic route failure"]
        sample["valid"] = False
        sample["validation_errors"] = ["runtime: synthetic route failure"]
        with tempfile.TemporaryDirectory() as work_dir:
            output = Path(work_dir) / "fresh-report.json"
            args = SimpleNamespace(
                out=str(output),
                model="/models/fixture.mq4",
                daemon="/bin/daemon",
                fresh_abba_cycles=2,
                settle_window=10,
                settle_min_runs=10,
                settle_confirmation_runs=10,
                settle_max_slope_pct=0.05,
                settle_max_spread_pct=1.0,
                settle_max_median_drift_pct=0.5,
                minimum_uplift_pct=0.0,
            )
            with (
                patch(
                    "redline_product_bench._fresh_identity",
                    return_value=dict(FRESH_IDENTITY),
                ),
                patch("redline_product_bench.git_head", return_value="deadbeef"),
                patch(
                    "redline_product_bench.run_fresh_sample",
                    return_value=sample,
                ) as run_sample,
                self.assertRaises(SystemExit),
            ):
                run_fresh_abba(args)
            report = json.loads(output.read_text())

        self.assertEqual(run_sample.call_count, 1)
        self.assertEqual(len(report["samples"]), 1)
        self.assertEqual(report["disposition"], "rejected_null")
        self.assertFalse(report["valid"])



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
    def test_late_drift_invalidates_an_earlier_confirmation(self):
        values = [100.0 + (0.01 if i % 2 else -0.01) for i in range(20)]
        values.extend(130.0 + (0.01 if i % 2 else -0.01) for i in range(10))
        result = analyze_stationarity(values, **DEFAULTS)
        self.assertFalse(result["stationary"])
        self.assertGreater(len(result["rejections"]), 0)



class RouteProofTests(unittest.TestCase):
    @staticmethod
    def route_row(iterations, delta, retained):
        return {
            "context_tokens": 129,
            "iterations": iterations,
            "redline_route": {
                "requested_backend": "auto",
                "transport": "aql",
                "state": "ready",
                "fallback_reason": None,
                "execution_mode": "plain_ar",
                "prepared": {
                    "dispatches": 286,
                    "packets": 287,
                    "queue_id": 7,
                    "command_dwords": None,
                },
                "sequence": {
                    "launches": 286,
                    "unique_kernels": 11,
                    "hash": "440bb2b5df220117",
                },
                "observed": {
                    "count_delta": delta,
                    "first_position": 129 if delta else None,
                    "last_position": 129 + delta - 1 if delta else None,
                },
                "retained_replay_observed": retained,
            },
        }

    def test_lifecycle_allows_partial_replay_but_timed_rows_reject_it(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=0, retained=False),
        ]
        lifecycle = validate_route_proof(rows, "auto", "aql")
        timed = validate_route_proof(
            rows, "auto", "aql", require_complete_replay=True
        )
        self.assertTrue(lifecycle["valid"], lifecycle["errors"])
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("timed row observed no retained replay" in error for error in timed["errors"])
        )
        self.assertTrue(
            any("0 retained replays for 8 timed iterations" in error for error in timed["errors"])
        )

    def test_timed_rows_require_one_replay_per_iteration(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        timed = validate_route_proof(
            rows, "auto", "aql", require_complete_replay=True
        )
        self.assertTrue(timed["valid"], timed["errors"])

    def test_timed_rows_reject_cumulative_position_evidence(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["observed"]["first_position"] = 128
        timed = validate_route_proof(
            [row], "auto", "aql", require_complete_replay=True
        )
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("first replay position 128 != 129" in error for error in timed["errors"])
        )



class Pm4PolicyTests(unittest.TestCase):
    def test_defaults_are_conservative_and_reproducible(self):
        self.assertEqual(
            effective_pm4_policy({}),
            {
                "HIPFIRE_REPLAY_PM4_QUEUES": "1",
                "HIPFIRE_REPLAY_PM4_WAIT_POLICY": "conservative",
                "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY": "conservative",
                "HIPFIRE_REPLAY_PM4_STATEFUL": "legacy",
                "HIPFIRE_REPLAY_PM4_GCR_TRIM": "1",
            },
        )

    def test_explicit_policy_overrides_are_preserved(self):
        self.assertEqual(
            effective_pm4_policy(
                {
                    "HIPFIRE_REPLAY_PM4_WAIT_POLICY": "resource",
                    "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY": "required-only",
                    "HIPFIRE_REPLAY_PM4_STATEFUL": "stateful",
                }
            ),
            {
                "HIPFIRE_REPLAY_PM4_QUEUES": "1",
                "HIPFIRE_REPLAY_PM4_WAIT_POLICY": "resource",
                "HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY": "required-only",
                "HIPFIRE_REPLAY_PM4_STATEFUL": "stateful",
                "HIPFIRE_REPLAY_PM4_GCR_TRIM": "1",
            },
        )


if __name__ == "__main__":
    unittest.main()

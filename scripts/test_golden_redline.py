#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("golden-redline.py")
SPEC = importlib.util.spec_from_file_location("golden_redline", SCRIPT)
golden_redline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(golden_redline)


class GoldenRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = golden_redline.load_registry(
            golden_redline.DEFAULT_REGISTRY
        )

    def test_registry_card_and_sampling_are_pinned(self):
        golden_redline.validate_model_registry_card(self.registry)

    def test_registry_has_one_fixture_per_supported_arch(self):
        fixtures = self.registry["fixtures"]
        self.assertEqual(
            {fixture["architecture"] for fixture in fixtures},
            {"gfx1100", "gfx1151", "gfx1201"},
        )

    def test_device_visibility_uses_physical_rocr_and_logical_hip(self):
        env = golden_redline.visible_environment(3)
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")

    def test_product_command_uses_the_sealed_tg128_contract(self):
        fixture = self.registry["fixtures"][0]
        command = golden_redline.product_command(
            fixture,
            self.registry,
            model=Path("/models/model.mq4r"),
            daemon=Path("/bin/daemon"),
            work_dir=Path("/tmp/work"),
            output=Path("/tmp/report.json"),
            timeout=1200,
        )
        self.assertIn("--expected-model-sha256", command)
        self.assertEqual(command[command.index("--iterations") + 1], "128")
        self.assertEqual(command[command.index("--settle-max-runs") + 1], "120")
        self.assertEqual(command[command.index("--transport") + 1], "pm4")
        self.assertEqual(command[command.index("--kv-mode") + 1], "q8")

    def test_default_handoff_pins_registry_sampling_over_global_overrides(self):
        with (
            mock.patch.object(
                golden_redline, "find_hipfire", return_value=Path("/bin/hipfire")
            ),
            mock.patch.object(golden_redline.subprocess, "run") as run,
        ):
            golden_redline.configure_default(self.registry, hipfire_path=None)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            [
                "/bin/hipfire",
                "config",
                "qwen3.6:35b-a3b-mq4r",
                "set",
                "temperature",
                "1.0",
            ],
            commands,
        )
        self.assertIn(
            [
                "/bin/hipfire",
                "config",
                "qwen3.6:35b-a3b-mq4r",
                "set",
                "kv_cache",
                "q8",
            ],
            commands,
        )
        self.assertEqual(
            commands[-1],
            [
                "/bin/hipfire",
                "config",
                "set",
                "serve.default_model",
                "qwen3.6:35b-a3b-mq4r",
            ],
        )


class GoldenReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = golden_redline.load_registry(
            golden_redline.DEFAULT_REGISTRY
        )
        cls.fixture = cls.registry["fixtures"][0]

    def report(self):
        fixture = self.fixture
        bench = fixture["benchmark"]
        route = fixture["route"]
        reference = fixture["reference"]
        return {
            "git_commit": reference["source_commit"],
            "daemon_sha256": reference["daemon_sha256"],
            "model_bytes": self.registry["model"]["size_bytes"],
            "model_sha256": self.registry["model"]["sha256"],
            "context": bench["context"],
            "iterations": bench["iterations"],
            "warmups": bench["warmups"],
            "warmup_iterations": bench["warmup_iterations"],
            "runs": bench["runs"],
            "transport": bench["transport"],
            "kv_mode": bench["kv_mode"],
            "stationarity": {
                "window": bench["settle_window"],
                "min_runs": bench["settle_min_runs"],
                "confirmation_runs": bench["settle_confirmation_runs"],
                "max_runs": bench["settle_max_runs"],
                "max_slope_pct": bench["settle_max_slope_pct"],
                "max_spread_pct": bench["settle_max_spread_pct"],
                "max_median_drift_pct": bench[
                    "settle_max_median_drift_pct"
                ],
            },
            "pm4_policy": self.registry["pm4_policy"],
            "valid": True,
            "speedup": reference["speedup"],
            "hip": {
                "tok_s": {"median": reference["hip_median_tok_s"]},
            },
            "auto": {
                "tok_s": {"median": reference["pm4_median_tok_s"]},
                "route_proof": {
                    "valid": True,
                    "errors": [],
                    "retained_rows": bench["runs"],
                    "observed_positions": route["observed_positions"],
                    "prepared_identities": [
                        [
                            route["dispatches"],
                            route["packets"],
                            route["phases"],
                            route["command_dwords"],
                        ]
                    ],
                    "sequences": [
                        [
                            route["dispatches"],
                            route["unique_kernels"],
                            route["sequence_hash"],
                        ]
                    ],
                },
            },
        }

    def test_exact_reference_report_passes(self):
        result = golden_redline.validate_report(
            self.report(),
            self.fixture,
            self.registry,
            strict_binary=True,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["classification"], "exact-reference-binary")

    def test_route_identity_mismatch_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["sequences"][0][2] = "wrong"
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("tape identity" in error for error in result["errors"])
        )

    def test_performance_below_floor_fails_closed(self):
        report = self.report()
        report["auto"]["tok_s"]["median"] = (
            self.fixture["acceptance"]["minimum_pm4_tok_s"] - 0.001
        )
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("PM4 median" in error for error in result["errors"])
        )

    def test_new_binary_is_labeled_compatible_not_exact(self):
        report = self.report()
        report["git_commit"] = "f" * 40
        report["daemon_sha256"] = "e" * 64
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(
            result["classification"], "route-compatible-reproduction"
        )
        self.assertEqual(len(result["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()

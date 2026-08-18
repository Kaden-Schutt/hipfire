#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.redline import golden as golden_redline


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
        arches = [fixture["architecture"] for fixture in fixtures]
        self.assertEqual(
            arches,
            ["gfx1100", "gfx1151", "gfx1201"],
        )

    def test_device_visibility_uses_physical_rocr_and_logical_hip(self):
        env = golden_redline.visible_environment(3)
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")

    def test_product_command_forwards_resolved_cli(self):
        fixture = self.registry["fixtures"][0]
        command = golden_redline.product_command(
            fixture,
            self.registry,
            model=Path("/models/model.mq4r"),
            daemon=Path("/bin/daemon"),
            cli=Path("/bin/hipfire"),
            work_dir=Path("/tmp/work"),
            output=Path("/tmp/report.json"),
            timeout=1200,
        )
        self.assertEqual(command[:4], [
            command[0],
            "-m",
            "tools.redline",
            "bench",
        ])
        self.assertEqual(command[command.index("--cli") + 1], "/bin/hipfire")
        self.assertEqual(command[command.index("--daemon") + 1], "/bin/daemon")
        self.assertIn("--expected-model-sha256", command)
        self.assertEqual(command[command.index("--iterations") + 1], "128")
        self.assertEqual(command[command.index("--settle-max-runs") + 1], "120")
        self.assertEqual(command[command.index("--transport") + 1], "pm4")
        self.assertEqual(command[command.index("--kv-mode") + 1], "q8")

    def test_ensure_binaries_builds_daemon_and_cli_explicitly(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            daemon = root / "daemon"
            cli = root / "hipfire"
            calls: list[list[str]] = []

            def fake_run(argv, cwd=None, check=None):
                calls.append(list(argv))
                if argv[3:6] == ["--example", "daemon", "-p"]:
                    daemon.write_text("daemon")
                elif argv[3:6] == ["--bin", "hipfire", "-p"]:
                    cli.write_text("cli")
                else:
                    self.fail(f"unexpected cargo argv: {argv}")
                return mock.Mock(returncode=0)

            with mock.patch.object(
                golden_redline.subprocess, "run", side_effect=fake_run
            ) as run:
                golden_redline.ensure_binaries(daemon, cli, build=True)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                calls,
                [
                    [
                        "cargo",
                        "build",
                        "--release",
                        "--example",
                        "daemon",
                        "-p",
                        "hipfire-runtime",
                    ],
                    [
                        "cargo",
                        "build",
                        "--release",
                        "--bin",
                        "hipfire",
                        "-p",
                        "hipfire-cli",
                    ],
                ],
            )
            self.assertTrue(daemon.is_file())
            self.assertTrue(cli.is_file())

    def test_ensure_binaries_skips_present_targets(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            daemon = root / "daemon"
            cli = root / "hipfire"
            daemon.write_text("daemon")
            calls: list[list[str]] = []

            def fake_run(argv, cwd=None, check=None):
                calls.append(list(argv))
                cli.write_text("cli")
                return mock.Mock(returncode=0)

            with mock.patch.object(
                golden_redline.subprocess, "run", side_effect=fake_run
            ):
                golden_redline.ensure_binaries(daemon, cli, build=True)
            self.assertEqual(
                calls,
                [
                    [
                        "cargo",
                        "build",
                        "--release",
                        "--bin",
                        "hipfire",
                        "-p",
                        "hipfire-cli",
                    ]
                ],
            )

    def test_ensure_binaries_no_build_fails_if_either_missing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            daemon = root / "daemon"
            cli = root / "hipfire"
            daemon.write_text("daemon")
            with self.assertRaisesRegex(
                golden_redline.GoldenError,
                r"required binary missing: .*hipfire",
            ):
                golden_redline.ensure_binaries(daemon, cli, build=False)

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


class GoldenArchitectureTests(unittest.TestCase):
    def test_kfd_fallback_maps_physical_gpu_index(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for node, properties in (
                (0, "cpu_cores_count 24\ngfx_target_version 0\n"),
                (1, "cpu_cores_count 0\ngfx_target_version 110000\n"),
                (2, "cpu_cores_count 0\ngfx_target_version 110501\n"),
            ):
                path = root / str(node)
                path.mkdir()
                (path / "properties").write_text(properties)
            self.assertEqual(
                golden_redline.detect_architecture_from_kfd(1, root),
                "gfx1151",
            )

    def test_architecture_detection_uses_kfd_without_rocminfo(self):
        with (
            mock.patch.object(golden_redline.shutil, "which", return_value=None),
            mock.patch.object(
                golden_redline,
                "detect_architecture_from_kfd",
                return_value="gfx1151",
            ) as detect_kfd,
        ):
            self.assertEqual(golden_redline.detect_architecture(1), "gfx1151")
        detect_kfd.assert_called_once_with(1)


class GoldenReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = golden_redline.load_registry(
            golden_redline.DEFAULT_REGISTRY
        )
        cls.fixture = cls.registry["fixtures"][0]

    def _route_proof(
        self,
        *,
        backend: str,
        transport: str,
        retained_rows: int,
        require_complete_replay: bool,
        include_identity: bool,
    ):
        route = self.fixture["route"]
        proof = {
            "valid": True,
            "backend": backend,
            "transport": transport,
            "rows": retained_rows if backend == "auto" else self.fixture["benchmark"]["runs"],
            "require_complete_replay": require_complete_replay,
            "retained_rows": retained_rows,
            "observed_positions": list(route["observed_positions"]),
            "prepared_identities": [],
            "sequences": [],
            "errors": [],
        }
        if include_identity:
            # queue_id is independent of fixture route.phases; identity shape is
            # [dispatches, packets, queue_id, command_dwords, queues, phases].
            proof["prepared_identities"] = [
                [
                    route["dispatches"],
                    route["packets"],
                    route.get("queue_id", 7),
                    route["command_dwords"],
                    route["queues"],
                    route["phases"],
                ]
            ]
            proof["sequences"] = [
                [
                    route["dispatches"],
                    route["unique_kernels"],
                    route["sequence_hash"],
                ]
            ]
        return proof

    def report(self):
        fixture = self.fixture
        bench = fixture["benchmark"]
        reference = fixture["reference"]
        transport = bench["transport"]
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
            "transport": transport,
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
            "pm4_preflight": {
                "seconds": 1.0,
                "redline_route": {
                    "state": "ready",
                    "retained_replay_observed": True,
                    "prepared": {
                        "dispatches": fixture["route"]["dispatches"],
                        "packets": fixture["route"]["packets"],
                        "queue_id": fixture["route"].get("queue_id", 7),
                        "command_dwords": fixture["route"]["command_dwords"],
                        "queues": fixture["route"]["queues"],
                        "phases": fixture["route"]["phases"],
                    },
                },
                "route_proof": self._route_proof(
                    backend="auto",
                    transport=transport,
                    retained_rows=1,
                    require_complete_replay=False,
                    include_identity=True,
                ),
            },
            "coherence": {
                "hip": {
                    "backend": "hip",
                    "valid": True,
                    "errors": [],
                    "route_evidence": {
                        "required": False,
                        "observed": False,
                        "transport": None,
                        "position": None,
                        "marker": None,
                        "lines": [],
                    },
                },
                "auto": {
                    "backend": "auto",
                    "valid": True,
                    "errors": [],
                    "route_evidence": {
                        "required": True,
                        "observed": True,
                        "transport": "pm4",
                        "position": 128,
                        "marker": "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=128",
                        "lines": [
                            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=128"
                        ],
                    },
                },
                "multiturn": None,
            },
            "hip": {
                "tok_s": {"median": reference["hip_median_tok_s"]},
                "measurement_validation": {"valid": True},
                "lifecycle_route_proof": self._route_proof(
                    backend="hip",
                    transport=transport,
                    retained_rows=0,
                    require_complete_replay=False,
                    include_identity=False,
                ),
                "route_proof": self._route_proof(
                    backend="hip",
                    transport=transport,
                    retained_rows=0,
                    require_complete_replay=False,
                    include_identity=False,
                ),
            },
            "auto": {
                "tok_s": {"median": reference["pm4_median_tok_s"]},
                "measurement_validation": {"valid": True},
                "lifecycle_route_proof": self._route_proof(
                    backend="auto",
                    transport=transport,
                    retained_rows=bench["runs"] + bench["warmups"],
                    require_complete_replay=False,
                    include_identity=True,
                ),
                "route_proof": self._route_proof(
                    backend="auto",
                    transport=transport,
                    retained_rows=bench["runs"],
                    require_complete_replay=True,
                    include_identity=True,
                ),
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

    def test_queue_id_is_not_compared_to_fixture_phases(self):
        report = self.report()
        # Fixture records independent queue_id; never alias it to phases.
        self.assertNotEqual(
            report["auto"]["route_proof"]["prepared_identities"][0][2],
            self.fixture["route"]["phases"],
        )
        self.assertEqual(
            report["auto"]["route_proof"]["prepared_identities"][0][2],
            self.fixture["route"]["queue_id"],
        )
        self.assertEqual(
            report["auto"]["route_proof"]["prepared_identities"][0][4:],
            [
                self.fixture["route"]["queues"],
                self.fixture["route"]["phases"],
            ],
        )
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_missing_pm4_preflight_fails_closed(self):
        report = self.report()
        del report["pm4_preflight"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("pm4_preflight is missing" in error for error in result["errors"])
        )

    def test_missing_pm4_preflight_route_proof_fails_closed(self):
        report = self.report()
        del report["pm4_preflight"]["route_proof"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "pm4_preflight route_proof is missing" in error
                for error in result["errors"]
            )
        )

    def test_invalid_pm4_preflight_route_proof_fails_closed(self):
        report = self.report()
        report["pm4_preflight"]["route_proof"]["valid"] = False
        report["pm4_preflight"]["route_proof"]["errors"] = ["bad preflight"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "pm4_preflight route_proof is invalid" in error
                for error in result["errors"]
            )
        )

    def test_pm4_preflight_identity_mismatch_fails_closed(self):
        report = self.report()
        report["pm4_preflight"]["route_proof"]["prepared_identities"][0][0] = 1
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "pm4_preflight route_proof prepared identity" in error
                for error in result["errors"]
            )
        )

    def test_false_coherence_arm_fails_closed(self):
        report = self.report()
        report["coherence"]["auto"]["valid"] = False
        report["coherence"]["auto"]["errors"] = ["bad answer"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("coherence.auto is invalid" in error for error in result["errors"])
        )

    def test_auto_coherence_without_retained_route_fails(self):
        report = self.report()
        report["coherence"]["auto"]["route_evidence"]["observed"] = False
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "did not observe retained replay route evidence" in error
                for error in result["errors"]
            )
        )

    def test_missing_measurement_validation_fails(self):
        report = self.report()
        del report["hip"]["measurement_validation"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "hip measurement_validation is missing" in error
                for error in result["errors"]
            )
        )

    def test_missing_lifecycle_route_proof_fails(self):
        report = self.report()
        del report["auto"]["lifecycle_route_proof"]
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "auto lifecycle_route_proof is missing" in error
                for error in result["errors"]
            )
        )

    def test_hip_timed_proof_with_retained_rows_fails(self):
        report = self.report()
        report["hip"]["route_proof"]["retained_rows"] = 3
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "hip route_proof retained_rows" in error
                for error in result["errors"]
            )
        )

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

    def test_non_positive_queue_id_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][2] = 0
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("queue_id=0" in error for error in result["errors"])
        )

    def test_string_false_top_level_valid_fails_closed(self):
        report = self.report()
        report["valid"] = "false"
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "product benchmark or route proof is invalid" in error
                for error in result["errors"]
            )
        )

    def test_numeric_truthy_route_proof_valid_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["valid"] = 1
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("auto route_proof is invalid" in error for error in result["errors"])
        )

    def test_string_true_measurement_valid_fails_closed(self):
        report = self.report()
        report["hip"]["measurement_validation"]["valid"] = "true"
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "hip measurement_validation is invalid" in error
                for error in result["errors"]
            )
        )

    def test_numeric_truthy_coherence_valid_fails_closed(self):
        report = self.report()
        report["coherence"]["auto"]["valid"] = 1
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("coherence.auto is invalid" in error for error in result["errors"])
        )

    def test_bool_queue_id_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][2] = True
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "queue_id=True" in error or "must be a positive int" in error
                for error in result["errors"]
            )
        )

    def test_bool_queues_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][4] = True
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "queues=True" in error or "must be a positive int" in error
                for error in result["errors"]
            )
        )

    def test_bool_phases_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][5] = True
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "phases=True" in error or "must be a positive int" in error
                for error in result["errors"]
            )
        )

    def test_bool_true_does_not_match_fixture_queue_id_one(self):
        report = self.report()
        # Even if fixture queue_id were 1, True must not match via True == 1.
        route = dict(self.fixture["route"])
        route["queue_id"] = 1
        fixture = dict(self.fixture)
        fixture["route"] = route
        report["auto"]["route_proof"]["prepared_identities"][0][2] = True
        # Also poison other identity slots that must reject bool before equality.
        result = golden_redline.validate_report(
            report,
            fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("positive int" in error for error in result["errors"])
        )



    def test_queues_mismatch_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][4] = (
            self.fixture["route"]["queues"] + 1
        )
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "auto route_proof prepared identity" in error
                and "queues=" in error
                for error in result["errors"]
            )
        )

    def test_phases_mismatch_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][5] = (
            self.fixture["route"]["phases"] + 1
        )
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "auto route_proof prepared identity" in error
                and "phases=" in error
                for error in result["errors"]
            )
        )

    def test_zero_queues_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][4] = 0
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("queues=0" in error for error in result["errors"])
        )

    def test_zero_phases_fails_closed(self):
        report = self.report()
        report["auto"]["route_proof"]["prepared_identities"][0][5] = 0
        result = golden_redline.validate_report(
            report,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("phases=0" in error for error in result["errors"])
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

    def test_legacy_sparse_report_cannot_classify_exact(self):
        # Historical product reports without the current schema fail closed and
        # never receive exact/current certification classifications.
        legacy = {
            "git_commit": self.fixture["reference"]["source_commit"],
            "daemon_sha256": self.fixture["reference"]["daemon_sha256"],
            "model_bytes": self.registry["model"]["size_bytes"],
            "model_sha256": self.registry["model"]["sha256"],
            "context": self.fixture["benchmark"]["context"],
            "iterations": self.fixture["benchmark"]["iterations"],
            "warmups": self.fixture["benchmark"]["warmups"],
            "warmup_iterations": self.fixture["benchmark"]["warmup_iterations"],
            "runs": self.fixture["benchmark"]["runs"],
            "transport": self.fixture["benchmark"]["transport"],
            "kv_mode": self.fixture["benchmark"]["kv_mode"],
            "stationarity": {
                "window": self.fixture["benchmark"]["settle_window"],
                "min_runs": self.fixture["benchmark"]["settle_min_runs"],
                "confirmation_runs": self.fixture["benchmark"][
                    "settle_confirmation_runs"
                ],
                "max_runs": self.fixture["benchmark"]["settle_max_runs"],
                "max_slope_pct": self.fixture["benchmark"]["settle_max_slope_pct"],
                "max_spread_pct": self.fixture["benchmark"][
                    "settle_max_spread_pct"
                ],
                "max_median_drift_pct": self.fixture["benchmark"][
                    "settle_max_median_drift_pct"
                ],
            },
            "pm4_policy": self.registry["pm4_policy"],
            "valid": True,
            "speedup": self.fixture["reference"]["speedup"],
            "hip": {
                "tok_s": {
                    "median": self.fixture["reference"]["hip_median_tok_s"]
                },
            },
            "auto": {
                "tok_s": {
                    "median": self.fixture["reference"]["pm4_median_tok_s"]
                },
                "route_proof": {
                    "valid": True,
                    "errors": [],
                    "retained_rows": self.fixture["benchmark"]["runs"],
                    "observed_positions": self.fixture["route"][
                        "observed_positions"
                    ],
                    "prepared_identities": [
                        [
                            self.fixture["route"]["dispatches"],
                            self.fixture["route"]["packets"],
                            self.fixture["route"]["queue_id"],
                            self.fixture["route"]["command_dwords"],
                            self.fixture["route"]["queues"],
                            self.fixture["route"]["phases"],
                        ]
                    ],
                    "sequences": [
                        [
                            self.fixture["route"]["dispatches"],
                            self.fixture["route"]["unique_kernels"],
                            self.fixture["route"]["sequence_hash"],
                        ]
                    ],
                },
            },
        }
        result = golden_redline.validate_report(
            legacy,
            self.fixture,
            self.registry,
            strict_binary=False,
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["classification"], "failed")
        self.assertNotEqual(result["classification"], "exact-reference-binary")
        self.assertTrue(
            any("pm4_preflight is missing" in error for error in result["errors"])
        )
        self.assertTrue(
            any("coherence is missing" in error for error in result["errors"])
        )

class GoldenSampledParityTests(unittest.TestCase):
    """Golden must reject archived reports with divergent sampled outputs; byte-identical passes."""

    @classmethod
    def setUpClass(cls):
        cls.registry = golden_redline.load_registry(golden_redline.DEFAULT_REGISTRY)
        cls.fixture = cls.registry["fixtures"][0]

    @staticmethod
    def _paris_row(content: str = "Paris is the capital of France.", ctx: int = 128, gen: int = 8) -> dict:
        return {"assistant_content": content, "ctx": ctx, "gen": gen}

    def _base_report(self):
        # Use the canonical valid report helper from GoldenReportTests
        helper = GoldenReportTests()
        helper.registry = self.registry  # type: ignore
        helper.fixture = self.fixture  # type: ignore
        return helper.report()

    def test_exact_paris_rows_pass(self):
        report = self._base_report()
        report["coherence"]["hip"]["rows"] = [self._paris_row()]
        report["coherence"]["auto"]["rows"] = [self._paris_row()]
        result = golden_redline.validate_report(report, self.fixture, self.registry, strict_binary=False)
        self.assertTrue(result["valid"], result["errors"])

    def test_paris_county_long_wrong_fails_despite_shared_substring(self):
        report = self._base_report()
        report["coherence"]["hip"]["rows"] = [self._paris_row("Paris is the capital of France.")]
        report["coherence"]["auto"]["rows"] = [
            self._paris_row("Paris is the capital of France. Paris County is a fictional county with many parishes and a long deterministically wrong suffix that contains Paris as substring.")
        ]
        result = golden_redline.validate_report(report, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result["valid"])
        self.assertTrue(any("assistant_content" in e or "sampled output" in e for e in result["errors"]))
        # Substring health hint would have passed, but exact parity must fail
        hip_content = report["coherence"]["hip"]["rows"][0]["assistant_content"]
        auto_content = report["coherence"]["auto"]["rows"][0]["assistant_content"]
        self.assertIn("Paris", hip_content)
        self.assertIn("Paris", auto_content)
        self.assertNotEqual(hip_content, auto_content)

    def test_count_type_field_mismatches_fail(self):
        # count mismatch
        report = self._base_report()
        report["coherence"]["hip"]["rows"] = [self._paris_row(), self._paris_row()]
        report["coherence"]["auto"]["rows"] = [self._paris_row()]
        result = golden_redline.validate_report(report, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result["valid"])
        self.assertTrue(any("row count" in e for e in result["errors"]))
        # type mismatch: hip_rows not a list
        report2 = self._base_report()
        report2["coherence"]["hip"]["rows"] = "not-a-list"  # type: ignore
        report2["coherence"]["auto"]["rows"] = [self._paris_row()]
        result2 = golden_redline.validate_report(report2, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result2["valid"])
        self.assertTrue(any("must be a list" in e for e in result2["errors"]))
        # row not a dict
        report3 = self._base_report()
        report3["coherence"]["hip"]["rows"] = [self._paris_row()]
        report3["coherence"]["auto"]["rows"] = ["not-a-dict"]  # type: ignore
        result3 = golden_redline.validate_report(report3, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result3["valid"])
        self.assertTrue(any("must be an object" in e for e in result3["errors"]))
        # missing field
        report4 = self._base_report()
        report4["coherence"]["hip"]["rows"] = [self._paris_row()]
        report4["coherence"]["auto"]["rows"] = [{"assistant_content": "Paris is the capital of France.", "ctx": 128}]  # missing gen
        result4 = golden_redline.validate_report(report4, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result4["valid"])
        self.assertTrue(any("gen" in e for e in result4["errors"]))
        # ctx mismatch
        report5 = self._base_report()
        report5["coherence"]["hip"]["rows"] = [self._paris_row(ctx=128, gen=8)]
        report5["coherence"]["auto"]["rows"] = [self._paris_row(ctx=129, gen=8)]
        result5 = golden_redline.validate_report(report5, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result5["valid"])
        self.assertTrue(any("ctx" in e for e in result5["errors"]))

    def test_no_normalization_case_sensitivity(self):
        report = self._base_report()
        report["coherence"]["hip"]["rows"] = [self._paris_row("Paris is the capital of France.")]
        report["coherence"]["auto"]["rows"] = [self._paris_row("paris is the capital of france.")]
        result = golden_redline.validate_report(report, self.fixture, self.registry, strict_binary=False)
        self.assertFalse(result["valid"])




class GoldenCliLifecycleTests(unittest.TestCase):
    """main() CLI discovery must tolerate a fresh checkout without a prebuilt hipfire."""

    @classmethod
    def setUpClass(cls):
        cls.registry = golden_redline.load_registry(
            golden_redline.DEFAULT_REGISTRY
        )
        cls.fixture = cls.registry["fixtures"][0]

    def _ok_validation(self):
        return {
            "valid": True,
            "classification": "route-compatible-reproduction",
            "errors": [],
            "warnings": [],
            "hip_median_tok_s": 1.0,
            "pm4_median_tok_s": 2.0,
            "speedup": 2.0,
        }

    def test_report_only_works_without_hipfire_cli(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = root / "model.mq4r"
            model.write_bytes(b"model-bytes")
            report_path = root / "product-report.json"
            report_path.write_text(
                json.dumps(
                    {"model_sha256": self.registry["model"]["sha256"], "valid": True}
                )
                + "\n"
            )
            find_required_flags: list[bool] = []
            real_find = golden_redline.find_hipfire

            def tracking_find(explicit, *, required=True):
                find_required_flags.append(required)
                if required:
                    raise AssertionError(
                        "report-only validation must not require a hipfire CLI"
                    )
                return real_find(explicit, required=False)

            expected_sha = self.registry["model"]["sha256"]

            def fake_sha256(path):
                path = Path(path)
                if path.resolve() == model.resolve():
                    return expected_sha
                digest = hashlib.sha256()
                digest.update(path.read_bytes())
                return digest.hexdigest()

            with (
                mock.patch.object(
                    golden_redline, "find_hipfire", side_effect=tracking_find
                ),
                mock.patch.object(golden_redline, "ensure_binaries") as ensure_bins,
                mock.patch.object(
                    golden_redline,
                    "detect_architecture",
                    return_value=self.fixture["architecture"],
                ),
                mock.patch.object(golden_redline, "sha256_file", side_effect=fake_sha256),
                mock.patch.object(
                    golden_redline,
                    "validate_report",
                    return_value=self._ok_validation(),
                ),
            ):
                rc = golden_redline.main(
                    [
                        "--fixture",
                        self.fixture["id"],
                        "--report",
                        str(report_path),
                        "--model",
                        str(model),
                        "--no-prompt",
                    ]
                )

            self.assertEqual(rc, 0)
            ensure_bins.assert_not_called()
            self.assertTrue(find_required_flags)
            self.assertTrue(all(flag is False for flag in find_required_flags))
            self.assertTrue(report_path.with_suffix(".golden.json").is_file())

    def test_live_build_path_tolerates_missing_release_cli(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = root / "model.mq4r"
            model.write_bytes(b"model-bytes")
            daemon = root / "daemon"
            cli = root / "hipfire"
            output = root / "live-report.json"
            find_required_flags: list[bool] = []
            real_find = golden_redline.find_hipfire

            def tracking_find(explicit, *, required=True):
                find_required_flags.append(required)
                return real_find(explicit, required=required)

            def fake_ensure(daemon_path, cli_path, *, build):
                self.assertTrue(build)
                # Fresh checkout: intended release CLI path is still absent here.
                self.assertFalse(cli_path.is_file())
                daemon_path.write_text("daemon\n")
                cli_path.write_text("cli\n")

            def fake_run_product_bench(
                fixture,
                golden,
                *,
                model,
                daemon,
                cli,
                work_dir,
                output,
                timeout,
                env,
            ):
                self.assertTrue(cli.is_file())
                output.write_text(
                    json.dumps(
                        {
                            "model_sha256": self.registry["model"]["sha256"],
                            "valid": True,
                        }
                    )
                    + "\n"
                )

            expected_sha = self.registry["model"]["sha256"]

            def fake_sha256(path):
                path = Path(path)
                if path.resolve() == model.resolve():
                    return expected_sha
                digest = hashlib.sha256()
                digest.update(path.read_bytes())
                return digest.hexdigest()

            with (
                mock.patch.object(
                    golden_redline, "find_hipfire", side_effect=tracking_find
                ),
                mock.patch.object(
                    golden_redline, "ensure_binaries", side_effect=fake_ensure
                ) as ensure_bins,
                mock.patch.object(
                    golden_redline,
                    "detect_architecture",
                    return_value=self.fixture["architecture"],
                ),
                mock.patch.object(golden_redline, "ensure_model"),
                mock.patch.object(
                    golden_redline,
                    "run_product_bench",
                    side_effect=fake_run_product_bench,
                ),
                mock.patch.object(golden_redline, "sha256_file", side_effect=fake_sha256),
                mock.patch.object(
                    golden_redline,
                    "validate_report",
                    return_value=self._ok_validation(),
                ),
            ):
                rc = golden_redline.main(
                    [
                        "--fixture",
                        self.fixture["id"],
                        "--hipfire",
                        str(cli),
                        "--daemon",
                        str(daemon),
                        "--model",
                        str(model),
                        "--out",
                        str(output),
                        "--no-prompt",
                    ]
                )

            self.assertEqual(rc, 0)
            ensure_bins.assert_called_once()
            self.assertEqual(find_required_flags, [False, True])
            self.assertTrue(cli.is_file())
            self.assertTrue(output.with_suffix(".golden.json").is_file())

    def test_live_no_build_missing_cli_fails_at_ensure_not_prelookup(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = root / "model.mq4r"
            model.write_bytes(b"model-bytes")
            daemon = root / "daemon"
            daemon.write_text("daemon\n")
            cli = root / "hipfire"
            output = root / "live-report.json"
            find_required_flags: list[bool] = []
            real_find = golden_redline.find_hipfire

            def tracking_find(explicit, *, required=True):
                find_required_flags.append(required)
                return real_find(explicit, required=required)

            with (
                mock.patch.object(
                    golden_redline, "find_hipfire", side_effect=tracking_find
                ),
                mock.patch.object(
                    golden_redline,
                    "detect_architecture",
                    return_value=self.fixture["architecture"],
                ),
                mock.patch.object(golden_redline, "ensure_model") as ensure_model,
                mock.patch.object(
                    golden_redline, "run_product_bench"
                ) as run_bench,
            ):
                with self.assertRaisesRegex(
                    golden_redline.GoldenError,
                    r"required binary missing: .*hipfire",
                ):
                    golden_redline.main(
                        [
                            "--fixture",
                            self.fixture["id"],
                            "--hipfire",
                            str(cli),
                            "--daemon",
                            str(daemon),
                            "--model",
                            str(model),
                            "--out",
                            str(output),
                            "--no-build",
                            "--no-prompt",
                        ]
                    )

            # Initial discovery stays non-required; failure is ensure_binaries.
            self.assertEqual(find_required_flags, [False])
            ensure_model.assert_not_called()
            run_bench.assert_not_called()


if __name__ == "__main__":
    unittest.main()

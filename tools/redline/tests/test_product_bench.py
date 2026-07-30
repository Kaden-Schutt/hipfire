#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import errno
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.redline import product_bench
from tools.redline.product_bench import (
    CERTIFIED_PM4_POLICY,
    COHERENCE_MAX_TOKENS,
    COHERENCE_MODE,
    COHERENCE_MTP,
    COHERENCE_PROMPT,
    COHERENCE_SAMPLING,
    COHERENCE_SEED,
    COHERENCE_THINKING,
    COHERENCE_THINKING_CAP_TOKENS,
    analyze_stationarity,
    backend_config_value,
    load_pm4_multiturn_session,
    require_retained_pm4,
    run_coherence_smoke,
    run_pm4_multiturn_session,
    run_pm4_preflight,
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


class RouteProofLogTests(unittest.TestCase):
    def test_parser_extracts_request_scoped_marker(self):
        parsed = product_bench.parse_route_proof_log(
            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=128 "
            "request_id=chatcmpl-turn-7 replays=64"
        )
        self.assertEqual(parsed["literal_count"], 1)
        self.assertEqual(parsed["malformed"], [])
        self.assertEqual(
            parsed["hits"],
            [
                {
                    "line": (
                        "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=128 "
                        "request_id=chatcmpl-turn-7 replays=64"
                    ),
                    "transport": "pm4",
                    "position": 128,
                    "request_id": "chatcmpl-turn-7",
                    "replays": 64,
                }
            ],
        )

    def test_parser_records_malformed_and_same_line_duplicates(self):
        malformed_line = "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 bogus"
        dual = (
            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=1 "
            "request_id=a replays=1 "
            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=2 "
            "request_id=b replays=2"
        )
        parsed = product_bench.parse_route_proof_log(
            f"noise\n{malformed_line}\n{dual}\n"
        )
        self.assertEqual(parsed["literal_count"], 3)
        self.assertEqual(parsed["hits"], [])
        self.assertEqual(len(parsed["malformed"]), 2)
        reasons = {item["reason"] for item in parsed["malformed"]}
        self.assertIn("malformed_marker", reasons)
        self.assertIn("multiple_markers_on_line", reasons)


class CoherenceRouteEvidenceTests(unittest.TestCase):
    @staticmethod
    def _hit(
        *,
        transport="pm4",
        position=1,
        request_id="chatcmpl-coherence-1",
        replays=64,
        legacy=False,
    ):
        if legacy:
            line = (
                f"HIPFIRE_REPLAY_ROUTE_PROOF transport={transport} "
                f"position={position}"
            )
            return {
                "line": line,
                "transport": transport,
                "position": position,
                "request_id": None,
                "replays": None,
            }
        line = (
            f"HIPFIRE_REPLAY_ROUTE_PROOF transport={transport} "
            f"position={position} request_id={request_id} replays={replays}"
        )
        return {
            "line": line,
            "transport": transport,
            "position": position,
            "request_id": request_id,
            "replays": replays,
        }

    @staticmethod
    def _evidence(hits, *, occurrences=None, malformed=None, literal_count=None):
        first = hits[0] if hits else None
        if occurrences is None:
            occurrences = [{"line": hit["line"], "offset": 0} for hit in hits]
        if malformed is None:
            malformed = []
        if literal_count is None:
            literal_count = len(occurrences)
        return {
            "observed": bool(hits) or literal_count > 0 or bool(malformed),
            "transport": None if first is None else first["transport"],
            "position": None if first is None else first["position"],
            "marker": None if first is None else first["line"],
            "lines": [hit["line"] for hit in hits]
            + [item.get("line") for item in malformed if item.get("line")],
            "count": len(hits),
            "hits": hits,
            "occurrences": occurrences,
            "malformed": malformed,
            "literal_count": literal_count,
        }

    def test_accepts_one_valid_bound_single_marker(self):
        hit = self._hit()
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            self._evidence([hit]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["required"])
        self.assertTrue(result["observed"])
        self.assertEqual(result["transport"], "pm4")
        self.assertEqual(result["hits"], [hit])
        self.assertEqual(result["literal_count"], 1)
        self.assertEqual(result["errors"], [])

    def test_rejects_legacy_marker(self):
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            self._evidence([self._hit(legacy=True)]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("legacy/unscoped" in error for error in result["errors"]),
            result["errors"],
        )

    def test_rejects_wrong_request_id(self):
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            self._evidence([self._hit(request_id="chatcmpl-other")]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "expected exactly one PM4 route proof marker for request_id "
                "'chatcmpl-coherence-1'"
                in error
                for error in result["errors"]
            ),
            result["errors"],
        )

    def test_rejects_extra_aql_marker(self):
        hits = [
            self._hit(request_id="chatcmpl-coherence-1", position=1),
            self._hit(
                transport="aql",
                request_id="chatcmpl-coherence-1",
                position=2,
            ),
        ]
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            self._evidence(hits),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("non-PM4" in error for error in result["errors"]),
            result["errors"],
        )
        self.assertTrue(
            any("exactly 1 route proof marker(s) total, got 2" in error for error in result["errors"]),
            result["errors"],
        )

    def test_rejects_zero_replays(self):
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            self._evidence([self._hit(replays=0)]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("positive replays" in error for error in result["errors"]),
            result["errors"],
        )

    def test_hip_requires_zero_markers(self):
        result = product_bench.validate_coherence_route_evidence(
            "hip",
            "aql",
            self._evidence([]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertTrue(result["valid"], result["errors"])
        bad = product_bench.validate_coherence_route_evidence(
            "hip",
            "aql",
            self._evidence([self._hit()]),
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(bad["valid"])
        self.assertTrue(
            any("HIP coherence must not emit" in error for error in bad["errors"]),
            bad["errors"],
        )

    def test_hip_rejects_malformed_marker_occurrence(self):
        evidence = self._evidence(
            [],
            occurrences=[{"line": "HIPFIRE_REPLAY_ROUTE_PROOF broken", "offset": 0}],
            malformed=[
                {
                    "line": "HIPFIRE_REPLAY_ROUTE_PROOF broken",
                    "reason": "malformed_marker",
                }
            ],
            literal_count=1,
        )
        result = product_bench.validate_coherence_route_evidence(
            "hip",
            "aql",
            evidence,
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("HIP coherence must not emit" in error for error in result["errors"]),
            result["errors"],
        )

    def test_auto_rejects_malformed_extra_alongside_valid(self):
        hit = self._hit()
        evidence = self._evidence(
            [hit],
            occurrences=[
                {"line": hit["line"], "offset": 0},
                {"line": "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 broken", "offset": 0},
            ],
            malformed=[
                {
                    "line": "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 broken",
                    "reason": "malformed_marker",
                }
            ],
            literal_count=2,
        )
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            evidence,
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("unparsed/malformed" in error for error in result["errors"]),
            result["errors"],
        )

    def test_auto_rejects_two_same_line_markers(self):
        dual = (
            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=1 "
            "request_id=chatcmpl-coherence-1 replays=64 "
            "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 position=2 "
            "request_id=chatcmpl-coherence-1 replays=8"
        )
        parsed = product_bench.parse_route_proof_log(dual)
        evidence = product_bench.collect_route_proof_evidence(
            None, stdout=dual, stderr=""
        )
        self.assertEqual(parsed["literal_count"], 2)
        self.assertEqual(parsed["hits"], [])
        self.assertTrue(parsed["malformed"])
        result = product_bench.validate_coherence_route_evidence(
            "auto",
            "pm4",
            evidence,
            rows=[{"request_id": "chatcmpl-coherence-1"}],
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("unparsed/malformed" in error for error in result["errors"]),
            result["errors"],
        )


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


class BackendConfigTests(unittest.TestCase):
    def test_auto_report_arm_explicitly_opts_into_redline(self):
        self.assertEqual(backend_config_value("auto"), "redline")
        self.assertEqual(backend_config_value("hip"), "hip")

    def test_product_policy_reemits_dynamic_gfx12_registers(self):
        self.assertEqual(CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_STATEFUL"], "static")
        self.assertEqual(CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_QUEUES"], "1")
        self.assertEqual(
            CERTIFIED_PM4_POLICY["HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY"],
            "required-only",
        )


class RouteProofTests(unittest.TestCase):
    @staticmethod
    def route_row(iterations, delta, retained, transport="pm4"):
        return {
            "context_tokens": 129,
            "iterations": iterations,
            "redline_route": {
                "requested_backend": "auto",
                "transport": transport,
                "state": "ready",
                "fallback_reason": None,
                "execution_mode": "plain_ar",
                "prepared": {
                    # prepared identity:
                    # [dispatches, packets, queue_id, command_dwords, queues, phases]
                    # index 2 is queue_id (queue identifier), never a phase count
                    "dispatches": 604,
                    "packets": 1,
                    "queue_id": 2,
                    "command_dwords": 16831 if transport == "pm4" else None,
                    "queues": 1,
                    "phases": 1,
                },
                "sequence": {
                    # tape identity: [launches, unique_kernels, hash]
                    "launches": 604,
                    "unique_kernels": 23,
                    "hash": "42f566b752920679",
                },
                "observed": {
                    "count_delta": delta,
                    "first_position": 129 if delta else None,
                    "last_position": 129 + delta - 1 if delta else None,
                },
                "retained_replay_observed": retained,
            },
        }

    def test_timed_rows_require_one_replay_per_iteration(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        timed = validate_route_proof(
            rows, "auto", "pm4", require_complete_replay=True
        )
        self.assertTrue(timed["valid"], timed["errors"])

    def test_timed_rows_reject_partial_replay(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=0, retained=False),
        ]
        timed = validate_route_proof(
            rows, "auto", "pm4", require_complete_replay=True
        )
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("timed row observed no retained replay" in error for error in timed["errors"])
        )

    def test_timed_rows_reject_cumulative_position_evidence(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["observed"]["first_position"] = 128
        timed = validate_route_proof(
            [row], "auto", "pm4", require_complete_replay=True
        )
        self.assertFalse(timed["valid"])
        self.assertTrue(
            any("first replay position 128 != 129" in error for error in timed["errors"])
        )

    def test_first_auto_warmup_fast_fails_on_pm4_fallback(self):
        row = self.route_row(iterations=32, delta=0, retained=False)
        row["redline_route"]["state"] = "fallback"
        row["redline_route"]["fallback_reason"] = "kernel requires scratch"
        with self.assertRaisesRegex(RuntimeError, "kernel requires scratch"):
            require_retained_pm4(row)

    def test_first_auto_warmup_accepts_retained_pm4(self):
        row = self.route_row(iterations=32, delta=32, retained=True)
        require_retained_pm4(row)


    def test_prepared_identity_is_dispatches_packets_queue_id_command_dwords_queues_phases(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        # Distinct observed positions so multi-position auto check passes.
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertTrue(proof["valid"], proof["errors"])
        self.assertEqual(proof["prepared_identities"], [[604, 1, 2, 16831, 1, 1]])
        self.assertEqual(
            proof["sequences"], [[604, 23, "42f566b752920679"]]
        )

    def test_rejects_null_queue_id(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["queue_id"] = None
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queue_id" in error for error in proof["errors"])
        )

    def test_rejects_zero_queue_id(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["queue_id"] = 0
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queue_id" in error for error in proof["errors"])
        )

    def test_rejects_missing_queues(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        del row["redline_route"]["prepared"]["queues"]
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queues" in error for error in proof["errors"])
        )

    def test_rejects_zero_queues(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["queues"] = 0
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queues" in error for error in proof["errors"])
        )

    def test_rejects_missing_phases(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        del row["redline_route"]["prepared"]["phases"]
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid phases" in error for error in proof["errors"])
        )

    def test_rejects_zero_phases(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["phases"] = 0
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid phases" in error for error in proof["errors"])
        )

    def test_rejects_bool_queue_id(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["queue_id"] = True
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queue_id" in error for error in proof["errors"])
        )

    def test_rejects_bool_queues(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["queues"] = True
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queues" in error for error in proof["errors"])
        )

    def test_rejects_bool_phases(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["phases"] = True
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid phases" in error for error in proof["errors"])
        )

    def test_rejects_bool_dispatches(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["prepared"]["dispatches"] = True
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid dispatch count" in error for error in proof["errors"])
        )


    def test_rejects_inconsistent_queues_across_rows(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        rows[1]["redline_route"]["prepared"]["queues"] = 2
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any(
                "prepared identity changed across rows" in error
                for error in proof["errors"]
            )
        )

    def test_rejects_inconsistent_phases_across_rows(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        rows[1]["redline_route"]["prepared"]["phases"] = 2
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any(
                "prepared identity changed across rows" in error
                for error in proof["errors"]
            )
        )

    def test_rejects_zero_unique_kernels(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["sequence"]["unique_kernels"] = 0
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid unique_kernels" in error for error in proof["errors"])
        )

    def test_rejects_missing_unique_kernels(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        del row["redline_route"]["sequence"]["unique_kernels"]
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid unique_kernels" in error for error in proof["errors"])
        )

    def test_rejects_empty_sequence_hash(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["sequence"]["hash"] = ""
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid sequence hash" in error for error in proof["errors"])
        )

    def test_rejects_nonhex_sequence_hash(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["sequence"]["hash"] = "gggggggggggggggg"
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid sequence hash" in error for error in proof["errors"])
        )

    def test_rejects_zero_sequence_hash(self):
        row = self.route_row(iterations=8, delta=8, retained=True)
        row["redline_route"]["sequence"]["hash"] = "0000000000000000"
        proof = validate_route_proof([row], "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid sequence hash" in error for error in proof["errors"])
        )

    def test_rejects_inconsistent_prepared_identity_across_rows(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        rows[1]["redline_route"]["prepared"]["queue_id"] = 9
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any(
                "prepared identity changed across rows" in error
                for error in proof["errors"]
            )
        )

    def test_rejects_inconsistent_sequence_identity_across_rows(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        rows[1]["redline_route"]["sequence"]["unique_kernels"] = 24
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any(
                "sequence identity changed across rows" in error
                for error in proof["errors"]
            )
        )

    def test_mixed_valid_and_malformed_rows_fail_closed_without_throwing(self):
        rows = [
            self.route_row(iterations=8, delta=8, retained=True),
            self.route_row(iterations=8, delta=8, retained=True),
        ]
        rows[1]["redline_route"]["prepared"]["queue_id"] = None
        rows[1]["redline_route"]["sequence"]["unique_kernels"] = None
        rows[1]["redline_route"]["sequence"]["hash"] = None
        rows[1]["redline_route"]["observed"]["first_position"] = 137
        rows[1]["redline_route"]["observed"]["last_position"] = 144
        rows[1]["context_tokens"] = 137
        proof = validate_route_proof(rows, "auto", "pm4")
        self.assertFalse(proof["valid"])
        self.assertTrue(
            any("invalid queue_id" in error for error in proof["errors"])
        )
        self.assertTrue(
            any("invalid unique_kernels" in error for error in proof["errors"])
        )
        self.assertTrue(
            any("invalid sequence hash" in error for error in proof["errors"])
        )
        # Valid row still contributes its identity; malformed row is omitted.
        self.assertEqual(proof["prepared_identities"], [[604, 1, 2, 16831, 1, 1]])
        self.assertEqual(
            proof["sequences"], [[604, 23, "42f566b752920679"]]
        )




class Pm4PreflightTests(unittest.TestCase):
    @staticmethod
    def args(work_dir):
        return SimpleNamespace(
            daemon="/unused/daemon",
            model="/unused/model",
            work_dir=work_dir,
            timeout=30.0,
            kv_mode="q8",
            max_seq=2048,
            context=128,
        )

    def test_preflight_smoke_tests_pm4_before_any_warmup(self):
        row = RouteProofTests.route_row(iterations=3, delta=2, retained=True)
        fake = unittest.mock.Mock()
        fake.request.side_effect = [{"type": "loaded"}, row]
        with tempfile.TemporaryDirectory() as work_dir:
            with patch("tools.redline.product_bench.Daemon", return_value=fake) as daemon:
                result = run_pm4_preflight(self.args(work_dir))

        daemon.assert_called_once_with(
            Path("/unused/daemon"),
            "auto",
            "pm4",
            Path(work_dir) / "product-pm4-preflight.log",
            30.0,
            "q8",
            0.0,
        )
        self.assertEqual(fake.request.call_args_list[0].args[0]["type"], "load")
        smoke = fake.request.call_args_list[1].args[0]
        self.assertEqual(smoke["type"], "bench_decode")
        self.assertEqual(smoke["iterations"], 3)
        self.assertTrue(smoke["redline_product_route"])
        self.assertEqual(result["redline_route"]["state"], "ready")
        self.assertIn("route_proof", result)
        self.assertTrue(result["route_proof"]["valid"], result["route_proof"])
        self.assertGreater(result["route_proof"]["retained_rows"], 0)
        fake.close.assert_called_once_with()

    def test_preflight_fast_fails_with_fallback_reason(self):
        row = RouteProofTests.route_row(iterations=2, delta=0, retained=False)
        row["redline_route"]["state"] = "fallback"
        row["redline_route"]["fallback_reason"] = "kernel requires scratch"
        fake = unittest.mock.Mock()
        fake.request.side_effect = [{"type": "loaded"}, row]
        with tempfile.TemporaryDirectory() as work_dir:
            with patch("tools.redline.product_bench.Daemon", return_value=fake):
                with self.assertRaisesRegex(RuntimeError, "kernel requires scratch"):
                    run_pm4_preflight(self.args(work_dir))
        fake.close.assert_called_once_with()

class CoherenceSmokeTests(unittest.TestCase):
    @staticmethod
    def args(work_dir, transport="pm4"):
        return SimpleNamespace(
            daemon=str(Path(work_dir) / "daemon"),
            cli=str(Path(work_dir) / "hipfire"),
            model=str(Path(work_dir) / "model.hfq"),
            work_dir=work_dir,
            timeout=30.0,
            kv_mode="q8",
            max_seq=2048,
            transport=transport,
        )

    @staticmethod
    def coherent_row(**overrides):
        row = {
            "request_id": "chatcmpl-coherence-1",
            "finish": "stop",
            "empty": False,
            "runaway": False,
            "attractor": False,
            "ans_words": 12,
            "decode_tok_s": 999.0,
            "ans_preview": (
                "Flagstaff was named after settlers raised a flag on a pine."
            ),
            "assistant_content": (
                "Flagstaff was named after settlers raised a flag on a pine."
            ),
        }
        row.update(overrides)
        return row

    def _prepare_binaries(self, work_dir):
        model = Path(work_dir) / "model.hfq"
        daemon = Path(work_dir) / "daemon"
        cli = Path(work_dir) / "hipfire"
        model.write_bytes(b"model")
        daemon.write_bytes(b"daemon")
        cli.write_bytes(b"cli")
        daemon.chmod(0o755)
        cli.chmod(0o755)
        return model, daemon, cli

    @staticmethod
    def _write_route_proof_marker(
        argv,
        transport="pm4",
        position=1,
        request_id="chatcmpl-coherence-1",
        replays=64,
        legacy=False,
    ):
        """Emit the retained-replay proof line into the harness --serve-log path."""
        if "--serve-log" not in argv:
            return
        serve_log = Path(argv[argv.index("--serve-log") + 1])
        serve_log.parent.mkdir(parents=True, exist_ok=True)
        if legacy:
            marker = (
                f"HIPFIRE_REPLAY_ROUTE_PROOF transport={transport} "
                f"position={position}\n"
            )
        else:
            marker = (
                f"HIPFIRE_REPLAY_ROUTE_PROOF transport={transport} "
                f"position={position} request_id={request_id} replays={replays}\n"
            )
        with serve_log.open("a", encoding="utf-8") as handle:
            handle.write(marker)

    def test_budget_allows_visible_answer(self):
        self.assertEqual(COHERENCE_THINKING, "low")
        self.assertEqual(COHERENCE_THINKING_CAP_TOKENS, 512)
        self.assertEqual(COHERENCE_MAX_TOKENS, 1024)
        self.assertGreater(COHERENCE_MAX_TOKENS, COHERENCE_THINKING_CAP_TOKENS)
        self.assertEqual(
            COHERENCE_PROMPT, "What is the origin of Flagstaff, Arizona's name?"
        )
        self.assertEqual(COHERENCE_SEED, 1)
        self.assertEqual(COHERENCE_SAMPLING, "registry")
        self.assertEqual(COHERENCE_MODE, "battery")
        self.assertEqual(COHERENCE_MTP, "off")

    def test_smoke_invokes_serve_harness_with_certified_env(self):
        captured = {}

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            captured["argv"] = list(argv)
            captured["cwd"] = cwd
            captured["env"] = dict(env)
            captured["timeout"] = timeout
            out_idx = argv.index("--out") + 1
            out_path = Path(argv[out_idx])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            self._write_route_proof_marker(argv, transport="pm4")
            daemon_bin = Path(env["HIPFIRE_DAEMON_BIN"])
            captured["daemon_bin"] = str(daemon_bin)
            captured["daemon_exists_during_run"] = daemon_bin.is_file()
            captured["daemon_basename"] = daemon_bin.name
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            args = self.args(work_dir, transport="pm4")
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                result = run_coherence_smoke(args, "auto")

            smoke_dir = Path(result["smoke_dir"])
            self.assertTrue(smoke_dir.is_dir())
            self.assertTrue(str(smoke_dir).startswith(str(Path(work_dir))))
            self.assertIn(f"product-auto-coherence-{os.getpid()}-", smoke_dir.name)
            prompts = json.loads((smoke_dir / "prompts.json").read_text())
            self.assertEqual(
                prompts, [{"genre": "factual", "prompt": COHERENCE_PROMPT}]
            )
            argv = captured["argv"]
            self.assertEqual(argv[1], str(product_bench.REPO / "scripts" / "serve_harness.py"))
            self.assertIn("--thinking", argv)
            self.assertEqual(argv[argv.index("--thinking") + 1], "low")
            self.assertEqual(argv[argv.index("--max-tokens") + 1], "1024")
            self.assertEqual(argv[argv.index("--seed") + 1], "1")
            self.assertEqual(argv[argv.index("--sampling") + 1], "registry")
            self.assertEqual(argv[argv.index("--mode") + 1], "battery")
            self.assertEqual(argv[argv.index("--mtp") + 1], "off")
            self.assertEqual(argv[argv.index("--kv") + 1], "q8")
            self.assertEqual(argv[argv.index("--max-seq") + 1], "2048")
            self.assertEqual(
                argv[argv.index("--model") + 1],
                str((Path(work_dir) / "model.hfq").resolve()),
            )
            self.assertEqual(
                argv[argv.index("--cli") + 1] if "--cli" in argv else None,
                None,
            )
            port = int(argv[argv.index("--port") + 1])
            self.assertNotIn(port, (11621, 11622, 11623))
            self.assertEqual(result["port"], port)
            self.assertEqual(result["config"]["port"], port)
            self.assertEqual(result["config"]["smoke_dir"], str(smoke_dir))
            env = captured["env"]
            self.assertEqual(
                env["HIPFIRE_CLI_BIN"], str((Path(work_dir) / "hipfire").resolve())
            )
            self.assertEqual(env["HIPFIRE_REPLAY_BACKEND"], "redline")
            self.assertEqual(env["HIPFIRE_REPLAY_TRANSPORT"], "pm4")
            self.assertEqual(env["HIPFIRE_KV_MODE"], "q8")
            for key, value in CERTIFIED_PM4_POLICY.items():
                self.assertEqual(env[key], value)
            self.assertNotIn("HIPFIRE_REPLAY_MANUAL_CAPTURE", env)
            self.assertNotIn("HIPFIRE_HOME", env)
            # Route proof is requested through temporary TOML, not ambient env.
            self.assertNotIn("HIPFIRE_REPLAY_ROUTE_PROOF_LOG", env)
            self.assertIn("--replay-route-proof-log", argv)
            self.assertTrue(result["config"]["replay_route_proof_log"])
            self.assertEqual(
                result["config"]["diagnostic_replay_route_proof_log"],
                "diagnostic.replay.route_proof_log",
            )
            self.assertTrue(result["route_evidence"]["required"])
            self.assertTrue(result["route_evidence"]["observed"])
            self.assertEqual(result["route_evidence"]["transport"], "pm4")
            expected_pid = str(smoke_dir / "serve-auto.pid")
            self.assertEqual(env["HIPFIRE_SERVE_HARNESS_PID_FILE"], expected_pid)
            self.assertEqual(result["config"]["serve_pid_file"], expected_pid)
            self.assertTrue(captured["daemon_exists_during_run"])
            self.assertTrue(
                captured["daemon_basename"].startswith("daemon-product-auto-coherence-")
            )
            self.assertFalse(Path(captured["daemon_bin"]).exists())
            self.assertTrue(result["valid"])
            self.assertFalse(result["speed_checked"])
            self.assertEqual(result["row"]["decode_tok_s"], 999.0)
            self.assertEqual(result["prompt"], COHERENCE_PROMPT)
            self.assertIn("stdout", result)
            self.assertIn("config", result)

    def test_hip_backend_maps_replay_env(self):
        captured = {}

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            captured["env"] = dict(env)
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                run_coherence_smoke(self.args(work_dir, transport="aql"), "hip")

        self.assertEqual(captured["env"]["HIPFIRE_REPLAY_BACKEND"], "hip")
        self.assertEqual(captured["env"]["HIPFIRE_REPLAY_TRANSPORT"], "aql")

    def test_strips_inherited_hipfire_home(self):
        captured = {}

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            captured["env"] = dict(env)
            captured["argv"] = list(argv)
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            inherited = dict(os.environ)
            inherited["HIPFIRE_HOME"] = "/tmp/should-not-leak"
            with (
                patch.dict(os.environ, inherited, clear=True),
                patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run),
            ):
                result = run_coherence_smoke(self.args(work_dir), "hip")

        self.assertNotIn("HIPFIRE_HOME", captured["env"])
        smoke_dir = Path(result["smoke_dir"])
        self.assertEqual(
            captured["env"]["HIPFIRE_SERVE_HARNESS_PID_FILE"],
            str(smoke_dir / "serve-hip.pid"),
        )
        self.assertEqual(
            Path(captured["argv"][captured["argv"].index("--home") + 1]),
            smoke_dir / "home",
        )

    def test_timeout_kills_process_group_and_reports_bad_pid(self):
        killed = {}

        def timeout_with_leader(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            pid_path = Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"])
            pid_path.write_text("4242\n", encoding="utf-8")
            killed["pid_path"] = pid_path
            daemon_bin = Path(env["HIPFIRE_DAEMON_BIN"])
            killed["daemon_bin"] = daemon_bin
            raise subprocess.TimeoutExpired(argv, timeout)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with (
                patch("tools.redline.product_bench.subprocess.run", side_effect=timeout_with_leader),
                patch("tools.redline.product_bench.os.getpgid") as getpgid,
                patch("tools.redline.product_bench.os.killpg") as killpg,
            ):
                with self.assertRaisesRegex(RuntimeError, r"timed out after 30\.0s$"):
                    run_coherence_smoke(self.args(work_dir), "hip")
            getpgid.assert_not_called()
            killpg.assert_called_once_with(4242, signal.SIGKILL)
            self.assertFalse(killed["daemon_bin"].exists())

        def timeout_missing_pid(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            raise subprocess.TimeoutExpired(argv, timeout)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch(
                "tools.redline.product_bench.subprocess.run",
                side_effect=timeout_missing_pid,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"timed out.*cleanup failed:.*unreadable"
                ):
                    run_coherence_smoke(self.args(work_dir), "auto")

        def timeout_invalid_pid(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"]).write_text("nope\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, timeout)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch(
                "tools.redline.product_bench.subprocess.run",
                side_effect=timeout_invalid_pid,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"timed out.*invalid contents"
                ):
                    run_coherence_smoke(self.args(work_dir), "auto")

        def timeout_nonpositive_pid(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"]).write_text("0\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, timeout)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with (
                patch(
                    "tools.redline.product_bench.subprocess.run",
                    side_effect=timeout_nonpositive_pid,
                ),
                patch("tools.redline.product_bench.os.killpg") as killpg,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"timed out.*not positive"
                ):
                    run_coherence_smoke(self.args(work_dir), "hip")
            killpg.assert_not_called()

        def timeout_esrch_is_benign(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"]).write_text("777\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, timeout)

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with (
                patch(
                    "tools.redline.product_bench.subprocess.run",
                    side_effect=timeout_esrch_is_benign,
                ),
                patch("tools.redline.product_bench.os.getpgid") as getpgid,
                patch(
                    "tools.redline.product_bench.os.killpg",
                    side_effect=OSError(errno.ESRCH, "gone"),
                ) as killpg,
            ):
                with self.assertRaisesRegex(RuntimeError, r"timed out after 30\.0s$"):
                    run_coherence_smoke(self.args(work_dir), "hip")
            getpgid.assert_not_called()
            killpg.assert_called_once_with(777, signal.SIGKILL)

    def test_timeout_killpg_when_leader_gone_group_remains(self):
        """Dead CLI leader must not block killpg of surviving group members."""

        def timeout_dead_leader(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"]).write_text("5150\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, timeout)

        def getpgid_leader_gone(pid):
            # Simulates ESRCH after leader exit; cleanup must not consult this.
            raise OSError(errno.ESRCH, "No such process")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with (
                patch(
                    "tools.redline.product_bench.subprocess.run",
                    side_effect=timeout_dead_leader,
                ),
                patch(
                    "tools.redline.product_bench.os.getpgid",
                    side_effect=getpgid_leader_gone,
                ) as getpgid,
                patch("tools.redline.product_bench.os.killpg") as killpg,
            ):
                with self.assertRaisesRegex(RuntimeError, r"timed out after 30\.0s$"):
                    run_coherence_smoke(self.args(work_dir), "hip")
            getpgid.assert_not_called()
            killpg.assert_called_once_with(5150, signal.SIGKILL)

    def test_rejects_malformed_row_unhealthy_finish_and_unrelated_answer(self):
        cases = [
            (["not-a-dict"], r"harness row must be a dict"),
            (
                [self.coherent_row(finish="length", runaway=False)],
                r"finish must be 'stop'",
            ),
            (
                [self.coherent_row(assistant_content="")],
                r"assistant_content must be a nonempty string",
            ),
            (
                [
                    self.coherent_row(
                        assistant_content="The capital of France is Paris.",
                        ans_preview="The capital of France is Paris.",
                    )
                ],
                r"answer missing",
            ),
            (
                [
                    self.coherent_row(
                        assistant_content="They raised a flag on the mountain.",
                        ans_preview="They raised a flag on the mountain.",
                    )
                ],
                r"naming/history cue",
            ),
            (
                [
                    self.coherent_row(
                        assistant_content="Flagstaff is a city in Arizona.",
                        ans_preview="Flagstaff is a city in Arizona.",
                    )
                ],
                r"real flag object",
            ),
            (
                [
                    self.coherent_row(
                        assistant_content="Flagstaff was named for its pine trees.",
                        ans_preview="Flagstaff was named for its pine trees.",
                    )
                ],
                r"real flag object",
            ),
            (
                [
                    self.coherent_row(
                        assistant_content=(
                            "Flagstaff is named for its staff of settlers in Arizona."
                        ),
                        ans_preview=(
                            "Flagstaff is named for its staff of settlers in Arizona."
                        ),
                    )
                ],
                r"real flag object",
            ),
        ]
        for rows, pattern in cases:
            with self.subTest(pattern=pattern, rows=rows):

                def fake_run(
                    argv, cwd=None, env=None, capture_output=None, text=None, timeout=None,
                    _rows=rows,
                ):
                    out_path = Path(argv[argv.index("--out") + 1])
                    out_path.write_text(json.dumps(_rows) + "\n")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                with tempfile.TemporaryDirectory() as work_dir:
                    self._prepare_binaries(work_dir)
                    with patch(
                        "tools.redline.product_bench.subprocess.run",
                        side_effect=fake_run,
                    ):
                        with self.assertRaisesRegex(RuntimeError, pattern):
                            run_coherence_smoke(self.args(work_dir), "hip")

    def test_rejects_city_fact_and_pine_trees_named_flagstaff(self):
        """Focused regressions for both wrong Flagstaff statements."""
        wrong = [
            "Flagstaff is a city in Arizona.",
            "Flagstaff was named for its pine trees.",
        ]
        for text in wrong:
            with self.subTest(text=text):
                errors = product_bench._flagstaff_answer_errors(text)
                self.assertTrue(
                    any("real flag object" in err for err in errors),
                    msg=errors,
                )

                def fake_run(
                    argv, cwd=None, env=None, capture_output=None, text=None, timeout=None,
                    _text=text,
                ):
                    out_path = Path(argv[argv.index("--out") + 1])
                    out_path.write_text(
                        json.dumps(
                            [
                                self.coherent_row(
                                    assistant_content=_text,
                                    ans_preview=_text,
                                )
                            ]
                        )
                        + "\n"
                    )
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                with tempfile.TemporaryDirectory() as work_dir:
                    self._prepare_binaries(work_dir)
                    with patch(
                        "tools.redline.product_bench.subprocess.run",
                        side_effect=fake_run,
                    ):
                        with self.assertRaisesRegex(RuntimeError, r"real flag object"):
                            run_coherence_smoke(self.args(work_dir), "auto")

    def test_accepts_coherent_flagstaff_pole_history_answer(self):
        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    [
                        self.coherent_row(
                            assistant_content=(
                                "Flagstaff takes its name from a flagpole "
                                "raised on a pine tree by settlers."
                            ),
                        )
                    ]
                )
                + "\n"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                result = run_coherence_smoke(self.args(work_dir), "hip")
        self.assertTrue(result["valid"])
        self.assertEqual(
            product_bench._flagstaff_answer_errors(
                "Flagstaff takes its name from a tall flag staff / pole "
                "raised for the 1876 centennial."
            ),
            [],
        )

    def test_unique_smoke_paths_and_ephemeral_ports(self):
        ports = []
        dirs = []

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            ports.append(int(argv[argv.index("--port") + 1]))
            out_path = Path(argv[argv.index("--out") + 1])
            dirs.append(out_path.parent)
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            # auto+pm4 coherence requires retained route proof in serve.log.
            if "auto" in Path(out_path).parent.name:
                self._write_route_proof_marker(argv, transport="pm4")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            args = self.args(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                r1 = run_coherence_smoke(args, "hip")
                r2 = run_coherence_smoke(args, "auto")

            self.assertEqual(len(ports), 2)
            self.assertEqual(len(set(ports)), 2)
            self.assertNotIn(11621, ports)
            self.assertNotIn(11622, ports)
            self.assertNotIn(11623, ports)
            self.assertEqual(r1["port"], ports[0])
            self.assertEqual(r2["port"], ports[1])
            self.assertNotEqual(r1["smoke_dir"], r2["smoke_dir"])
            self.assertEqual(Path(r1["smoke_dir"]), dirs[0])
            self.assertEqual(Path(r2["smoke_dir"]), dirs[1])
            for smoke in (r1["smoke_dir"], r2["smoke_dir"]):
                self.assertTrue(str(smoke).startswith(work_dir))
                self.assertIn(str(os.getpid()), Path(smoke).name)

    def test_auto_pm4_requires_route_proof_marker(self):
        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            # Intentionally omit serve.log marker.
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(
                    RuntimeError, r"HIPFIRE_REPLAY_ROUTE_PROOF marker"
                ):
                    run_coherence_smoke(self.args(work_dir, transport="pm4"), "auto")

    def test_hip_rejects_unexpected_route_proof_marker(self):
        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            self._write_route_proof_marker(argv, transport="pm4")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(
                    RuntimeError, r"HIP coherence must not emit retained route proof"
                ):
                    run_coherence_smoke(self.args(work_dir, transport="aql"), "hip")

    def test_auto_success_reports_route_evidence(self):
        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(json.dumps([self.coherent_row()]) + "\n")
            self._write_route_proof_marker(argv, transport="pm4", position=3)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                result = run_coherence_smoke(self.args(work_dir, transport="pm4"), "auto")

        self.assertTrue(result["valid"])
        self.assertTrue(result["route_evidence"]["required"])
        self.assertTrue(result["route_evidence"]["observed"])
        self.assertEqual(result["route_evidence"]["transport"], "pm4")
        self.assertEqual(result["route_evidence"]["position"], 3)
        self.assertIn("HIPFIRE_REPLAY_ROUTE_PROOF", result["route_evidence"]["marker"])
        self.assertTrue(result["route_evidence"]["valid"])
        self.assertEqual(result["route_evidence"]["errors"], [])
        hits = result["route_evidence"]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["transport"], "pm4")
        self.assertEqual(hits[0]["request_id"], "chatcmpl-coherence-1")
        self.assertEqual(hits[0]["replays"], 64)
        self.assertEqual(hits[0]["position"], 3)

    def test_quality_only_verdict_ignores_speed(self):
        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(
                json.dumps(
                    [
                        self.coherent_row(
                            decode_tok_s=0.01,
                            assistant_content=(
                                "Flagstaff takes its name from a flagpole "
                                "raised on a pine tree by settlers."
                            ),
                        )
                    ]
                )
                + "\n"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=fake_run):
                result = run_coherence_smoke(self.args(work_dir), "hip")

        self.assertTrue(result["valid"])
        self.assertFalse(result["speed_checked"])
        self.assertEqual(result["row"]["decode_tok_s"], 0.01)
        content = result["row"]["assistant_content"].lower()
        self.assertTrue("flagpole" in content or "flag" in content)
        self.assertTrue(any(token in content for token in ("name", "named", "origin")))

    def test_rejects_empty_runaway_attractor_and_subprocess_failure(self):
        cases = [
            {"empty": True},
            {"runaway": True, "finish": "length"},
            {"attractor": True},
        ]
        for bad in cases:
            with self.subTest(bad=bad):

                def fake_run(
                    argv, cwd=None, env=None, capture_output=None, text=None, timeout=None
                ):
                    out_path = Path(argv[argv.index("--out") + 1])
                    out_path.write_text(json.dumps([self.coherent_row(**bad)]) + "\n")
                    return SimpleNamespace(returncode=0, stdout="", stderr="")

                with tempfile.TemporaryDirectory() as work_dir:
                    self._prepare_binaries(work_dir)
                    with patch(
                        "tools.redline.product_bench.subprocess.run",
                        side_effect=fake_run,
                    ):
                        with self.assertRaises(RuntimeError):
                            run_coherence_smoke(self.args(work_dir), "auto")

        def failing_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            return SimpleNamespace(returncode=7, stdout="", stderr="boom")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=failing_run
            ):
                with self.assertRaisesRegex(RuntimeError, "exited 7"):
                    run_coherence_smoke(self.args(work_dir), "hip")

    def test_rejects_missing_or_multiple_rows(self):
        def zero_rows(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text("[]\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def two_rows(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            out_path.write_text(
                json.dumps([self.coherent_row(), self.coherent_row()]) + "\n"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=zero_rows
            ):
                with self.assertRaisesRegex(RuntimeError, "exactly one harness row"):
                    run_coherence_smoke(self.args(work_dir), "hip")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=two_rows
            ):
                with self.assertRaisesRegex(RuntimeError, "exactly one harness row"):
                    run_coherence_smoke(self.args(work_dir), "auto")

    def test_unique_daemon_cleaned_even_on_failure(self):
        seen = {}

        def boom(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            daemon_bin = Path(env["HIPFIRE_DAEMON_BIN"])
            seen["path"] = daemon_bin
            seen["existed"] = daemon_bin.is_file()
            raise RuntimeError("spawn failed")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            with patch("tools.redline.product_bench.subprocess.run", side_effect=boom):
                with self.assertRaisesRegex(RuntimeError, "spawn failed"):
                    run_coherence_smoke(self.args(work_dir), "auto")
            self.assertTrue(seen["existed"])
            self.assertFalse(seen["path"].exists())

    def test_main_orders_coherence_before_benchmark_warmups(self):
        order = []
        arm = {
            "tok_s": {"median": 100.0},
            "measurement_validation": {"valid": True},
            "route_proof": {"valid": True},
            "lifecycle_route_proof": {"valid": True},
        }
        preflight = {
            "redline_route": {"prepared": {"dispatches": 604}},
            "seconds": 1.0,
        }
        coherence = {"valid": True, "errors": [], "seconds": 0.5}

        def record_preflight(_args):
            order.append("preflight")
            return preflight

        def record_coherence(_args, backend):
            order.append(f"coherence-{backend}")
            return coherence

        def record_arm(_args, backend):
            order.append(f"arm-{backend}")
            return arm

        with tempfile.TemporaryDirectory() as work_dir:
            model = Path(work_dir) / "model.hfq"
            daemon = Path(work_dir) / "daemon"
            cli = Path(work_dir) / "hipfire"
            output = Path(work_dir) / "report.json"
            model.write_bytes(b"model")
            daemon.write_bytes(b"daemon")
            cli.write_bytes(b"cli")
            with (
                patch(
                    "tools.redline.product_bench.run_pm4_preflight",
                    side_effect=record_preflight,
                ),
                patch(
                    "tools.redline.product_bench.run_coherence_smoke",
                    side_effect=record_coherence,
                ),
                patch(
                    "tools.redline.product_bench.run_arm",
                    side_effect=record_arm,
                ),
                patch("tools.redline.product_bench.git_head", return_value="deadbeef"),
            ):
                product_bench.main(
                    [
                        "--model",
                        str(model),
                        "--daemon",
                        str(daemon),
                        "--cli",
                        str(cli),
                        "--transport",
                        "pm4",
                        "--work-dir",
                        work_dir,
                        "--out",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text())

        self.assertEqual(
            order,
            [
                "preflight",
                "coherence-hip",
                "coherence-auto",
                "arm-hip",
                "arm-auto",
            ],
        )
        self.assertIn("coherence", report)
        self.assertTrue(report["coherence"]["hip"]["valid"])
        self.assertTrue(report["coherence"]["auto"]["valid"])
        self.assertIsNone(report["coherence"]["multiturn"])

    def test_main_runs_coherence_for_aql_without_preflight(self):
        order = []
        arm = {
            "tok_s": {"median": 100.0},
            "measurement_validation": {"valid": True},
            "route_proof": {"valid": True},
            "lifecycle_route_proof": {"valid": True},
        }
        coherence = {"valid": True, "errors": [], "seconds": 0.5}

        def record_coherence(_args, backend):
            order.append(f"coherence-{backend}")
            return coherence

        def record_arm(_args, backend):
            order.append(f"arm-{backend}")
            return arm

        with tempfile.TemporaryDirectory() as work_dir:
            model = Path(work_dir) / "model.hfq"
            daemon = Path(work_dir) / "daemon"
            cli = Path(work_dir) / "hipfire"
            output = Path(work_dir) / "report.json"
            model.write_bytes(b"model")
            daemon.write_bytes(b"daemon")
            cli.write_bytes(b"cli")
            with (
                patch(
                    "tools.redline.product_bench.run_pm4_preflight",
                    side_effect=AssertionError("preflight must not run for aql"),
                ),
                patch(
                    "tools.redline.product_bench.run_coherence_smoke",
                    side_effect=record_coherence,
                ),
                patch(
                    "tools.redline.product_bench.run_arm",
                    side_effect=record_arm,
                ),
                patch("tools.redline.product_bench.git_head", return_value="deadbeef"),
            ):
                product_bench.main(
                    [
                        "--model",
                        str(model),
                        "--daemon",
                        str(daemon),
                        "--cli",
                        str(cli),
                        "--transport",
                        "aql",
                        "--work-dir",
                        work_dir,
                        "--out",
                        str(output),
                    ]
                )

        self.assertEqual(
            order,
            [
                "coherence-hip",
                "coherence-auto",
                "arm-hip",
                "arm-auto",
            ],
        )



class Pm4MultiturnSessionTests(unittest.TestCase):
    @staticmethod
    def args(work_dir, transport="pm4", session=None):
        return SimpleNamespace(
            daemon=str(Path(work_dir) / "daemon"),
            cli=str(Path(work_dir) / "hipfire"),
            model=str(Path(work_dir) / "model.hfq"),
            work_dir=work_dir,
            timeout=30.0,
            kv_mode="q8",
            max_seq=2048,
            transport=transport,
            pm4_multiturn_session=session,
        )

    @staticmethod
    def coherent_row(**overrides):
        row = {
            "request_id": "chatcmpl-turn-1",
            "finish": "stop",
            "empty": False,
            "runaway": False,
            "attractor": False,
            "ans_words": 8,
            "decode_tok_s": 12.5,
            "ans_preview": "Remember the codeword ALPHA.",
            "assistant_content": "Remember the codeword ALPHA.",
        }
        row.update(overrides)
        return row

    def _prepare_binaries(self, work_dir):
        model = Path(work_dir) / "model.hfq"
        daemon = Path(work_dir) / "daemon"
        cli = Path(work_dir) / "hipfire"
        model.write_bytes(b"model")
        daemon.write_bytes(b"daemon")
        cli.write_bytes(b"cli")
        daemon.chmod(0o755)
        cli.chmod(0o755)
        return model, daemon, cli

    def _write_session(self, work_dir, turns):
        path = Path(work_dir) / "session.json"
        path.write_text(json.dumps(turns) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _write_route_proof_markers(argv, request_ids, *, transport="pm4", replays=64):
        """Emit one well-formed PM4 marker per request_id into --serve-log."""
        if "--serve-log" not in argv:
            return
        serve_log = Path(argv[argv.index("--serve-log") + 1])
        serve_log.parent.mkdir(parents=True, exist_ok=True)
        with serve_log.open("a", encoding="utf-8") as handle:
            for index, request_id in enumerate(request_ids, 1):
                handle.write(
                    f"HIPFIRE_REPLAY_ROUTE_PROOF transport={transport} "
                    f"position={index} request_id={request_id} replays={replays}\n"
                )

    def test_cli_rejects_multiturn_without_pm4_transport(self):
        with tempfile.TemporaryDirectory() as work_dir:
            model, daemon, cli = self._prepare_binaries(work_dir)
            session = self._write_session(
                work_dir, [{"content": "hi", "expect": ["x"]}]
            )
            with self.assertRaises(SystemExit) as raised:
                product_bench.main(
                    [
                        "--model",
                        str(model),
                        "--daemon",
                        str(daemon),
                        "--cli",
                        str(cli),
                        "--transport",
                        "aql",
                        "--pm4-multiturn-session",
                        str(session),
                        "--work-dir",
                        work_dir,
                        "--out",
                        str(Path(work_dir) / "out.json"),
                    ]
                )
            self.assertNotEqual(raised.exception.code, 0)

    def test_rejects_empty_and_malformed_session_before_gpu(self):
        with tempfile.TemporaryDirectory() as work_dir:
            empty = Path(work_dir) / "empty.json"
            empty.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "nonempty list"):
                load_pm4_multiturn_session(empty)

            bad = Path(work_dir) / "bad.json"
            bad.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "nonempty list"):
                load_pm4_multiturn_session(bad)

            missing = Path(work_dir) / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "not found"):
                load_pm4_multiturn_session(missing)

            model, daemon, cli = self._prepare_binaries(work_dir)
            with (
                patch(
                    "tools.redline.product_bench.run_pm4_preflight",
                    side_effect=AssertionError("must not start GPU"),
                ),
                patch(
                    "tools.redline.product_bench.run_coherence_smoke",
                    side_effect=AssertionError("must not start GPU"),
                ),
                patch(
                    "tools.redline.product_bench.run_arm",
                    side_effect=AssertionError("must not start GPU"),
                ),
            ):
                with self.assertRaisesRegex(SystemExit, "multiturn session invalid"):
                    product_bench.main(
                        [
                            "--model",
                            str(model),
                            "--daemon",
                            str(daemon),
                            "--cli",
                            str(cli),
                            "--transport",
                            "pm4",
                            "--pm4-multiturn-session",
                            str(empty),
                            "--work-dir",
                            work_dir,
                            "--out",
                            str(Path(work_dir) / "out.json"),
                        ]
                    )

    def test_rejects_malformed_and_missing_all_expectations(self):
        with tempfile.TemporaryDirectory() as work_dir:
            no_expect = self._write_session(
                work_dir,
                [
                    {"content": "setup only"},
                    {"content": "still no expect"},
                ],
            )
            with self.assertRaisesRegex(RuntimeError, r"at least one nonempty expect"):
                load_pm4_multiturn_session(no_expect)

            empty_expect = Path(work_dir) / "empty_expect.json"
            empty_expect.write_text(
                json.dumps([{"content": "hi", "expect": []}]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, r"nonempty list of nonempty strings"):
                load_pm4_multiturn_session(empty_expect)

            coerced = Path(work_dir) / "coerced.json"
            coerced.write_text(
                json.dumps([{"content": "hi", "expect": [1, "ok"]}]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, r"nonempty string \(no coercion\)"):
                load_pm4_multiturn_session(coerced)

            not_list = Path(work_dir) / "not_list.json"
            not_list.write_text(
                json.dumps([{"content": "hi", "expect": "ALPHA"}]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, r"nonempty list of nonempty strings"):
                load_pm4_multiturn_session(not_list)

            empty_string = Path(work_dir) / "empty_string.json"
            empty_string.write_text(
                json.dumps([{"content": "hi", "expect": [""]}]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, r"nonempty string \(no coercion\)"):
                load_pm4_multiturn_session(empty_string)

            # Setup turn may omit expect when another turn declares one.
            ok = self._write_session(
                work_dir,
                [
                    {"content": "remember this"},
                    {"content": "recall it", "expect": ["ALPHA"]},
                ],
            )
            path, turns = load_pm4_multiturn_session(ok)
            self.assertEqual(path, ok.resolve())
            self.assertEqual(len(turns), 2)

    def test_rejects_missing_expect_substring(self):
        turns = [
            {"content": "Say the codeword.", "expect": ["ALPHA"]},
            {"content": "Repeat it.", "expect": ["ALPHA", "bravo"]},
        ]

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            rows = [
                self.coherent_row(
                    request_id="chatcmpl-turn-1",
                    assistant_content="The codeword is ALPHA.",
                ),
                self.coherent_row(
                    request_id="chatcmpl-turn-2",
                    assistant_content="Still ALPHA only.",
                ),
            ]
            out_path.write_text(json.dumps(rows) + "\n")
            self._write_route_proof_markers(
                argv, [row["request_id"] for row in rows]
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            session = self._write_session(work_dir, turns)
            args = self.args(work_dir, session=str(session))
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=fake_run
            ):
                with self.assertRaisesRegex(RuntimeError, r"missing expected substring 'bravo'"):
                    run_pm4_multiturn_session(args)

    def test_successful_session_validity_and_harness_contract(self):
        turns = [
            {"content": "Remember ALPHA.", "expect": ["alpha"]},
            {"content": "What was it?", "expect": ["ALPHA"]},
            {"content": "Thanks."},  # setup-style turn: expect omitted
        ]
        captured = {}

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            captured["argv"] = list(argv)
            captured["env"] = dict(env)
            out_path = Path(argv[argv.index("--out") + 1])
            rows = [
                self.coherent_row(
                    request_id="chatcmpl-turn-1",
                    assistant_content="Stored ALPHA.",
                ),
                self.coherent_row(
                    request_id="chatcmpl-turn-2",
                    assistant_content="It was ALPHA.",
                ),
                self.coherent_row(
                    request_id="chatcmpl-turn-3",
                    assistant_content="You're welcome.",
                    decode_tok_s=0.01,
                ),
            ]
            out_path.write_text(json.dumps(rows) + "\n")
            self._write_route_proof_markers(
                argv, [row["request_id"] for row in rows]
            )
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            session = self._write_session(work_dir, turns)
            args = self.args(work_dir, session=str(session))
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=fake_run
            ):
                result = run_pm4_multiturn_session(args)

            argv = captured["argv"]
            self.assertEqual(
                argv[1], str(product_bench.REPO / "scripts" / "serve_harness.py")
            )
            self.assertEqual(argv[argv.index("--mode") + 1], "session")
            self.assertEqual(
                Path(argv[argv.index("--session") + 1]).resolve(), session.resolve()
            )
            self.assertEqual(argv[argv.index("--thinking") + 1], "low")
            self.assertEqual(argv[argv.index("--max-tokens") + 1], "1024")
            self.assertEqual(argv[argv.index("--seed") + 1], "1")
            self.assertEqual(argv[argv.index("--sampling") + 1], "registry")
            self.assertIn("--replay-route-proof-log", argv)
            port = int(argv[argv.index("--port") + 1])
            self.assertNotEqual(port, 11623)
            self.assertEqual(result["port"], port)
            env = captured["env"]
            self.assertEqual(env["HIPFIRE_REPLAY_BACKEND"], "redline")
            self.assertEqual(env["HIPFIRE_REPLAY_TRANSPORT"], "pm4")
            for key, value in CERTIFIED_PM4_POLICY.items():
                self.assertEqual(env[key], value)
            self.assertNotIn("HIPFIRE_HOME", env)
            self.assertNotIn("HIPFIRE_REPLAY_ROUTE_PROOF_LOG", env)
            smoke_dir = Path(result["smoke_dir"])
            self.assertTrue(smoke_dir.is_dir())
            self.assertIn(f"product-auto-multiturn-{os.getpid()}-", smoke_dir.name)
            self.assertEqual(result["config"]["smoke_dir"], str(smoke_dir))
            self.assertTrue(result["config"]["replay_route_proof_log"])
            self.assertEqual(
                result["config"]["diagnostic_replay_route_proof_log"],
                "diagnostic.replay.route_proof_log",
            )
            self.assertEqual(
                env["HIPFIRE_SERVE_HARNESS_PID_FILE"],
                str(smoke_dir / "serve-auto-multiturn.pid"),
            )
            self.assertTrue(result["valid"])
            self.assertFalse(result["speed_checked"])
            self.assertEqual(result["turns"], 3)
            self.assertEqual(len(result["rows"]), 3)
            self.assertTrue(result["route_evidence"]["required"])
            self.assertTrue(result["route_evidence"]["observed"])
            self.assertTrue(result["route_evidence"]["valid"])
            self.assertEqual(result["route_evidence"]["literal_count"], 3)
            self.assertEqual(len(result["route_evidence"]["hits"]), 3)

    def test_quality_only_ignores_speed(self):
        turns = [{"content": "hi", "expect": ["hello"]}]

        def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            out_path = Path(argv[argv.index("--out") + 1])
            row = self.coherent_row(
                request_id="chatcmpl-turn-1",
                assistant_content="Hello there.",
                decode_tok_s=0.001,
            )
            out_path.write_text(json.dumps([row]) + "\n")
            self._write_route_proof_markers(argv, [row["request_id"]])
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            session = self._write_session(work_dir, turns)
            args = self.args(work_dir, session=str(session))
            with patch(
                "tools.redline.product_bench.subprocess.run", side_effect=fake_run
            ):
                result = run_pm4_multiturn_session(args)

        self.assertTrue(result["valid"])
        self.assertFalse(result["speed_checked"])
        self.assertEqual(result["rows"][0]["decode_tok_s"], 0.001)

    def test_multiturn_fails_without_bound_markers_and_passes_with_one_each(self):
        turns = [
            {"content": "Remember ALPHA.", "expect": ["ALPHA"]},
            {"content": "Recall it.", "expect": ["ALPHA"]},
        ]

        def fake_run_no_markers(
            argv, cwd=None, env=None, capture_output=None, text=None, timeout=None
        ):
            out_path = Path(argv[argv.index("--out") + 1])
            rows = [
                self.coherent_row(
                    request_id="chatcmpl-turn-1",
                    assistant_content="Stored ALPHA.",
                ),
                self.coherent_row(
                    request_id="chatcmpl-turn-2",
                    assistant_content="It was ALPHA.",
                ),
            ]
            out_path.write_text(json.dumps(rows) + "\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def fake_run_with_markers(
            argv, cwd=None, env=None, capture_output=None, text=None, timeout=None
        ):
            out_path = Path(argv[argv.index("--out") + 1])
            rows = [
                self.coherent_row(
                    request_id="chatcmpl-turn-1",
                    assistant_content="Stored ALPHA.",
                ),
                self.coherent_row(
                    request_id="chatcmpl-turn-2",
                    assistant_content="It was ALPHA.",
                ),
            ]
            out_path.write_text(json.dumps(rows) + "\n")
            self._write_route_proof_markers(
                argv, [row["request_id"] for row in rows]
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            session = self._write_session(work_dir, turns)
            args = self.args(work_dir, session=str(session))
            with patch(
                "tools.redline.product_bench.subprocess.run",
                side_effect=fake_run_no_markers,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"HIPFIRE_REPLAY_ROUTE_PROOF marker"
                ):
                    run_pm4_multiturn_session(args)

            with patch(
                "tools.redline.product_bench.subprocess.run",
                side_effect=fake_run_with_markers,
            ):
                result = run_pm4_multiturn_session(args)

        self.assertTrue(result["valid"])
        self.assertEqual(result["route_evidence"]["literal_count"], 2)
        self.assertEqual(len(result["route_evidence"]["hits"]), 2)
        self.assertEqual(result["route_evidence"]["malformed"], [])
        self.assertTrue(result["route_evidence"]["valid"])

    def test_timeout_killpg_when_leader_gone_group_remains(self):
        """Shared cleanup: killpg known PGID even if getpgid(leader) would ESRCH."""

        def timeout_dead_leader(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
            Path(env["HIPFIRE_SERVE_HARNESS_PID_FILE"]).write_text("6161\n", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, timeout)

        def getpgid_leader_gone(pid):
            raise OSError(errno.ESRCH, "No such process")

        with tempfile.TemporaryDirectory() as work_dir:
            self._prepare_binaries(work_dir)
            session = self._write_session(
                work_dir, [{"content": "hi", "expect": ["hello"]}]
            )
            args = self.args(work_dir, session=str(session))
            with (
                patch(
                    "tools.redline.product_bench.subprocess.run",
                    side_effect=timeout_dead_leader,
                ),
                patch(
                    "tools.redline.product_bench.os.getpgid",
                    side_effect=getpgid_leader_gone,
                ) as getpgid,
                patch("tools.redline.product_bench.os.killpg") as killpg,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"multiturn session timed out after 30\.0s$"
                ):
                    run_pm4_multiturn_session(args)
            getpgid.assert_not_called()
            killpg.assert_called_once_with(6161, signal.SIGKILL)

    def test_main_runs_multiturn_after_both_arms_only(self):
        order = []
        arm = {
            "tok_s": {"median": 100.0},
            "measurement_validation": {"valid": True},
            "route_proof": {"valid": True},
            "lifecycle_route_proof": {"valid": True},
        }
        preflight = {
            "redline_route": {"prepared": {"dispatches": 604}},
            "seconds": 1.0,
        }
        coherence = {"valid": True, "errors": [], "seconds": 0.5}
        multiturn = {
            "valid": True,
            "errors": [],
            "seconds": 0.75,
            "turns": 2,
            "speed_checked": False,
        }

        def record_preflight(_args):
            order.append("preflight")
            return preflight

        def record_coherence(_args, backend):
            order.append(f"coherence-{backend}")
            return coherence

        def record_arm(_args, backend):
            order.append(f"arm-{backend}")
            return arm

        def record_multiturn(_args):
            order.append("multiturn")
            return multiturn

        with tempfile.TemporaryDirectory() as work_dir:
            model = Path(work_dir) / "model.hfq"
            daemon = Path(work_dir) / "daemon"
            cli = Path(work_dir) / "hipfire"
            output = Path(work_dir) / "report.json"
            session = self._write_session(
                work_dir,
                [
                    {"content": "one", "expect": ["a"]},
                    {"content": "two", "expect": ["b"]},
                ],
            )
            model.write_bytes(b"model")
            daemon.write_bytes(b"daemon")
            cli.write_bytes(b"cli")
            with (
                patch(
                    "tools.redline.product_bench.run_pm4_preflight",
                    side_effect=record_preflight,
                ),
                patch(
                    "tools.redline.product_bench.run_coherence_smoke",
                    side_effect=record_coherence,
                ),
                patch(
                    "tools.redline.product_bench.run_arm",
                    side_effect=record_arm,
                ),
                patch(
                    "tools.redline.product_bench.run_pm4_multiturn_session",
                    side_effect=record_multiturn,
                ),
                patch("tools.redline.product_bench.git_head", return_value="deadbeef"),
            ):
                product_bench.main(
                    [
                        "--model",
                        str(model),
                        "--daemon",
                        str(daemon),
                        "--cli",
                        str(cli),
                        "--transport",
                        "pm4",
                        "--pm4-multiturn-session",
                        str(session),
                        "--work-dir",
                        work_dir,
                        "--out",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text())

        self.assertEqual(
            order,
            [
                "preflight",
                "coherence-hip",
                "coherence-auto",
                "arm-hip",
                "arm-auto",
                "multiturn",
            ],
        )
        self.assertTrue(report["coherence"]["multiturn"]["valid"])
        self.assertTrue(report["valid"])


class ServeHarnessWarmTests(unittest.TestCase):
    """Regressions for scripts/serve_harness spawn warm acceptance."""

    @staticmethod
    def _load_serve_harness():
        import importlib.util

        path = product_bench.REPO / "scripts" / "serve_harness.py"
        spec = importlib.util.spec_from_file_location("serve_harness_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _health_response(health):
        import io

        class _RespIO:
            def __enter__(self):
                return io.BytesIO(json.dumps(health).encode())

            def __exit__(self, *args):
                return False

        return _RespIO()

    def test_rejects_dead_spawn_even_when_foreign_service_warm(self):
        sh = self._load_serve_harness()
        dead = SimpleNamespace(poll=lambda: 1)  # exited
        health = {"model": "/models/other.hfq", "loading_model": False}

        with patch.object(
            sh.urllib.request,
            "urlopen",
            side_effect=lambda *a, **k: self._health_response(health),
        ):
            self.assertFalse(
                sh._native_service_warm(
                    9,
                    expected_model="/models/wanted.hfq",
                    proc=dead,
                )
            )

    def test_rejects_warm_health_with_wrong_model(self):
        sh = self._load_serve_harness()
        alive = SimpleNamespace(poll=lambda: None)
        health = {"model": "/models/foreign.hfq", "loading_model": False}

        with patch.object(
            sh.urllib.request,
            "urlopen",
            side_effect=lambda *a, **k: self._health_response(health),
        ):
            self.assertFalse(
                sh._native_service_warm(
                    9,
                    expected_model="/models/wanted.hfq",
                    proc=alive,
                )
            )

    def test_accepts_live_proc_with_matching_model(self):
        sh = self._load_serve_harness()
        alive = SimpleNamespace(poll=lambda: None)
        model = "/tmp/wanted-model.hfq"
        health = {"model": model, "loading_model": False}

        with patch.object(
            sh.urllib.request,
            "urlopen",
            side_effect=lambda *a, **k: self._health_response(health),
        ):
            self.assertTrue(
                sh._native_service_warm(9, expected_model=model, proc=alive)
            )

    def test_send_records_sse_completion_id(self):
        sh = self._load_serve_harness()
        chunks = [
            (
                'data: {"id":"chatcmpl-turn-1","choices":[{"delta":{"content":"ok"},'
                '"finish_reason":"stop"}],"usage":{"prompt_tokens":3,'
                '"completion_tokens":1}}\n'
            ).encode(),
            b"data: [DONE]\n",
        ]
        cfg = {
            "model": "test-model",
            "port": 11435,
            "max_tokens": 8,
            "sampling": {},
            "seed": None,
            "expect_visible": True,
        }
        with patch.object(sh.urllib.request, "urlopen", return_value=chunks):
            row = sh.send(cfg, [{"role": "user", "content": "hello"}])
        self.assertEqual(row["request_id"], "chatcmpl-turn-1")

    def test_write_native_config_emits_route_proof_log_when_requested(self):
        sh = self._load_serve_harness()
        with tempfile.TemporaryDirectory() as home:
            cfg = {
                "model": "/tmp/model.hfq",
                "port": 11520,
                "max_seq": 2048,
                "kv": "q8",
                "mtp": "off",
                "thinking_budget": "low",
                "replay_route_proof_log": True,
            }
            os.makedirs(os.path.join(home, ".hipfire"), exist_ok=True)
            sh._write_native_config(cfg, home)
            text = Path(home, ".hipfire", "config.toml").read_text()
            self.assertIn("[diagnostic.replay]", text)
            self.assertIn("route_proof_log = true", text)

    def test_write_native_config_omits_route_proof_log_by_default(self):
        sh = self._load_serve_harness()
        with tempfile.TemporaryDirectory() as home:
            cfg = {
                "model": "/tmp/model.hfq",
                "port": 11520,
                "max_seq": 2048,
                "kv": "q8",
                "mtp": "off",
                "thinking_budget": "low",
            }
            os.makedirs(os.path.join(home, ".hipfire"), exist_ok=True)
            sh._write_native_config(cfg, home)
            text = Path(home, ".hipfire", "config.toml").read_text()
            self.assertNotIn("route_proof_log", text)
            self.assertNotIn("[diagnostic.replay]", text)

    def test_spawn_serve_strips_harness_pid_file_from_child_env(self):
        """HIPFIRE_SERVE_HARNESS_PID_FILE is parent IPC only — never hand to serve."""
        sh = self._load_serve_harness()
        captured = {}

        class _FakeProc:
            def __init__(self, pid=4242):
                self.pid = pid

            def poll(self):
                return None

        def fake_popen(argv, cwd=None, env=None, stdout=None, stderr=None, start_new_session=None):
            captured["env"] = dict(env)
            captured["argv"] = list(argv)
            return _FakeProc()

        with tempfile.TemporaryDirectory() as home:
            log = Path(home) / "serve.log"
            pid_path = Path(home) / "parent.pid"
            cfg = {
                "model": "/tmp/model.hfq",
                "port": 11520,
                "max_seq": 2048,
                "kv": "q8",
                "mtp": "off",
                "thinking_budget": "low",
            }
            os.makedirs(os.path.join(home, ".hipfire"), exist_ok=True)
            inherited = dict(os.environ)
            inherited["HIPFIRE_SERVE_HARNESS_PID_FILE"] = str(pid_path)
            inherited["HIPFIRE_CLI_BIN"] = str(Path(home) / "fake-cli")
            Path(inherited["HIPFIRE_CLI_BIN"]).write_bytes(b"x")
            Path(inherited["HIPFIRE_CLI_BIN"]).chmod(0o755)

            with (
                patch.dict(os.environ, inherited, clear=True),
                patch.object(sh.subprocess, "Popen", side_effect=fake_popen),
                patch.object(sh, "_native_service_warm", return_value=True),
                patch.object(sh.time, "sleep", return_value=None),
                patch.object(sh, "_native_cli", return_value=inherited["HIPFIRE_CLI_BIN"]),
                patch.object(sh, "atexit", SimpleNamespace(register=lambda *_a, **_k: None)),
            ):
                ok = sh.spawn_serve(cfg, home, str(log))

            self.assertIsNotNone(ok)
            self.assertIn("env", captured)
            self.assertNotIn("HIPFIRE_SERVE_HARNESS_PID_FILE", captured["env"])
            # Parent still wrote the PID file via its own os.environ.
            self.assertTrue(pid_path.is_file())
            self.assertEqual(pid_path.read_text(encoding="utf-8").strip(), "4242")


if __name__ == "__main__":
    unittest.main()

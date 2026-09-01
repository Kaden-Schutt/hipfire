import copy
import unittest

from scripts.redline_daemon_harness import DFLASH_EXACT_FIELDS, dflash_shadow_failures


def valid_shadow():
    arm = {field: True for field in DFLASH_EXACT_FIELDS}
    arm["state"] = {"max_abs": 0.0, "max_rel": 0.0}
    return {
        "route": {
            "phase": "ready",
            "counters": {
                "replays": 1,
                "replay_failures": 0,
                "poison_count": 0,
                "contract_failures": 0,
                "prepare_failures": 0,
            },
        },
        "capture": {"aql_equals_unique_kernels": True},
        "prepared_identity": {
            "dispatch_equals_launches": True,
            "queue_count": 1,
            "phase_count": 1,
        },
        "parity": {
            "q8_error_feedback_enabled": True,
            "q8_byte_parity_invalid": False,
            "windows": [
                {
                    "position": 128,
                    "recorded_hip": copy.deepcopy(arm),
                    "pm4": copy.deepcopy(arm),
                }
            ],
        },
    }


class DflashShadowVerdictTests(unittest.TestCase):
    def test_production_q8_ef_report_passes(self):
        self.assertEqual(dflash_shadow_failures(valid_shadow()), [])

    def test_missing_or_divergent_ef_snapshot_fails(self):
        for field in ("dn_ef_after_forward_equal", "dn_ef_after_rollback_equal"):
            with self.subTest(field=field, case="missing"):
                shadow = valid_shadow()
                del shadow["parity"]["windows"][0]["pm4"][field]
                self.assertTrue(any(field in failure for failure in dflash_shadow_failures(shadow)))
            with self.subTest(field=field, case="divergent"):
                shadow = valid_shadow()
                shadow["parity"]["windows"][0]["pm4"][field] = False
                self.assertTrue(any(field in failure for failure in dflash_shadow_failures(shadow)))

    def test_legacy_or_unreported_q8_parity_is_rejected(self):
        shadow = valid_shadow()
        del shadow["parity"]["q8_byte_parity_invalid"]
        self.assertIn(
            "parity.q8_byte_parity_invalid missing",
            dflash_shadow_failures(shadow),
        )

        shadow = valid_shadow()
        shadow["parity"]["q8_byte_parity_invalid"] = True
        self.assertTrue(
            any("production Q8 EF evidence required" in failure for failure in dflash_shadow_failures(shadow))
        )


if __name__ == "__main__":
    unittest.main()

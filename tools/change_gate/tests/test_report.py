# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Report builder / markdown renderer tests — pure, no GPU."""

from __future__ import annotations

import json
import unittest

from tools.change_gate.model import SCHEMA_ID, RouteResult, Selection
from tools.change_gate.report import (
    build_report,
    compute_verdict,
    render_markdown,
    to_json,
)


def _sel(
    rid: str,
    status: str = "selected",
    *,
    detail: str = "",
    paths: tuple[str, ...] = ("x.rs",),
    reason: str = "because",
) -> Selection:
    return Selection(
        route_id=rid,
        matched_paths=paths,
        rule_reason=reason,
        status=status,
        detail=detail or status,
    )


def _res(rid: str, status: str, duration_s: float = 1.0) -> RouteResult:
    return RouteResult(
        route_id=rid,
        status=status,
        duration_s=duration_s,
        verdict={},
        artifacts=(),
    )


class VerdictPrecedence(unittest.TestCase):
    def test_pass_when_all_clear(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a")],
                not_run=[],
                results=[_res("a", "pass")],
            ),
            "pass",
        )

    def test_fail_beats_pass(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a"), _sel("b")],
                not_run=[],
                results=[_res("a", "pass"), _res("b", "fail")],
            ),
            "fail",
        )

    def test_incomplete_beats_fail(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a")],
                not_run=[_sel("b", "blocked_model", detail="missing m")],
                results=[_res("a", "fail")],
            ),
            "incomplete",
        )

    def test_incomplete_from_blocked_arch(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[],
                not_run=[_sel("x", "blocked_arch")],
                results=[],
            ),
            "incomplete",
        )

    def test_incomplete_from_result_blocked(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a")],
                not_run=[],
                results=[_res("a", "blocked")],
            ),
            "incomplete",
        )

    def test_incomplete_beats_pass_on_trimmed(self) -> None:
        # trimmed_budget is a form of incomplete coverage
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a")],
                not_run=[_sel("b", "trimmed_budget")],
                results=[_res("a", "pass")],
            ),
            "incomplete",
        )

    def test_excluded_heavy_is_incomplete(self) -> None:
        self.assertEqual(
            compute_verdict(
                selected=[_sel("a")],
                not_run=[_sel("h", "excluded_heavy")],
                results=[_res("a", "pass")],
            ),
            "incomplete",
        )


class ReportJsonContract(unittest.TestCase):
    REQUIRED_TOP = (
        "schema",
        "base",
        "head",
        "dirty",
        "host",
        "changed_files",
        "selected",
        "not_run",
        "results",
        "totals",
        "verdict",
    )

    def _sample(self, **kwargs):
        defaults = dict(
            base="abc",
            head="def",
            dirty=False,
            host={"gfx": "gfx1201", "rocm": "7.14", "models_dir": "/m"},
            changed_files=["cli/x.ts"],
            selected=[_sel("unit.control")],
            not_run=[],
            results=[_res("unit.control", "pass", 0.4)],
            est_minutes=0.5,
        )
        defaults.update(kwargs)
        return build_report(**defaults)

    def test_schema_id_and_required_keys(self) -> None:
        report = self._sample()
        self.assertEqual(report["schema"], SCHEMA_ID)
        self.assertEqual(report["schema"], "hipfire.change_gate/1")
        for key in self.REQUIRED_TOP:
            self.assertIn(key, report, msg=f"missing top-level key {key}")
        for key in ("gfx", "rocm", "models_dir"):
            self.assertIn(key, report["host"])
        for key in ("est_minutes", "actual_s", "routes_selected", "routes_blocked"):
            self.assertIn(key, report["totals"])

    def test_to_json_roundtrip(self) -> None:
        report = self._sample()
        blob = to_json(report)
        self.assertTrue(blob.endswith("\n"))
        parsed = json.loads(blob)
        self.assertEqual(parsed["schema"], SCHEMA_ID)
        self.assertEqual(parsed["verdict"], "pass")

    def test_verdict_incomplete_when_blocked_in_not_run(self) -> None:
        report = self._sample(
            selected=[_sel("a")],
            not_run=[_sel("b", "blocked_model", detail="no model")],
            results=[_res("a", "pass")],
        )
        self.assertEqual(report["verdict"], "incomplete")
        self.assertGreaterEqual(report["totals"]["routes_blocked"], 1)


class RenderMarkdown(unittest.TestCase):
    def test_always_emits_not_run_table_even_when_empty(self) -> None:
        report = build_report(
            base="a",
            head="b",
            dirty=False,
            host={"gfx": "gfx1201", "rocm": "7", "models_dir": "/m"},
            changed_files=[],
            selected=[_sel("unit.control")],
            not_run=[],
            results=[_res("unit.control", "pass")],
        )
        md = render_markdown(report)
        self.assertIn("### Routes NOT RUN", md)
        # table header present
        self.assertIn("| route", md.lower())
        self.assertIn("reason", md.lower())

    def test_never_claims_pass_while_route_blocked(self) -> None:
        report = build_report(
            base="a",
            head="b",
            dirty=False,
            host={"gfx": "gfx1201", "rocm": "7", "models_dir": "/m"},
            changed_files=["crates/x/src/lib.rs"],
            selected=[_sel("unit.control")],
            not_run=[_sel("serve.x", "blocked_model", detail="missing qwen")],
            results=[_res("unit.control", "pass")],
        )
        self.assertEqual(report["verdict"], "incomplete")
        md = render_markdown(report)
        self.assertIn("INCOMPLETE", md)
        # Must not present a bare PASS badge as the verdict.
        self.assertNotIn("**change_gate: PASS**", md)
        self.assertIn("blocked_model", md)
        # Honesty line always present
        self.assertIn("incomplete", md.lower())

    def test_not_run_rows_listed(self) -> None:
        report = build_report(
            base="a",
            head="b",
            dirty=True,
            host={"gfx": "", "rocm": "", "models_dir": ""},
            changed_files=["y"],
            selected=[],
            not_run=[
                _sel("serve.a", "blocked_arch", detail="no gpu"),
                _sel("serve.b", "excluded_heavy", detail="heavy"),
            ],
            results=[],
        )
        md = render_markdown(report)
        self.assertIn("serve.a", md)
        self.assertIn("serve.b", md)
        self.assertIn("INCOMPLETE", md)


if __name__ == "__main__":
    unittest.main()

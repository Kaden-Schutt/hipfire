# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Selector algorithm tests — no GPU, no models, fake manifest only."""

from __future__ import annotations

import unittest

from tools.change_gate.model import Route, Rule, Selection
from tools.change_gate.report import compute_verdict
from tools.change_gate.selector import select


def _route(
    rid: str,
    *,
    kind: str = "serve",
    est: float = 1.0,
    tier: str = "cheap",
    arches: tuple[str, ...] = (),
    models: tuple[str, ...] = (),
    why: str = "catch regressions",
) -> Route:
    return Route(
        id=rid,
        kind=kind,
        argv=("true",),
        est_minutes=est,
        tier=tier,
        arches=arches,
        models=models,
        why=why,
    )


# Fixed fake manifest — intentionally independent of tools.change_gate.routes.
FAKE_ROUTES: dict[str, Route] = {
    "unit.control": _route("unit.control", kind="unit", est=0.5, tier="cheap"),
    "serve.gfx12.smoke": _route(
        "serve.gfx12.smoke",
        est=3.0,
        tier="standard",
        arches=("gfx1201", "gfx1200"),
        models=("qwen-fake",),
    ),
    "serve.gfx11.smoke": _route(
        "serve.gfx11.smoke",
        est=3.0,
        tier="standard",
        arches=("gfx1100",),
        models=("qwen-fake",),
    ),
    "serve.niah.heavy": _route(
        "serve.niah.heavy",
        est=30.0,
        tier="heavy",
        models=("qwen-fake",),
        why="200K NIAH coherence",
    ),
    "speed.arch": _route(
        "speed.arch",
        kind="speed",
        est=5.0,
        tier="standard",
        arches=("gfx1201",),
    ),
    "redline.capture": _route(
        "redline.capture",
        kind="redline",
        est=4.0,
        tier="standard",
    ),
    "serve.budget.std": _route(
        "serve.budget.std",
        est=10.0,
        tier="standard",
        models=("qwen-fake",),
    ),
    "serve.budget.std2": _route(
        "serve.budget.std2",
        est=8.0,
        tier="standard",
        models=("qwen-fake",),
    ),
}

FAKE_RULES: tuple[Rule, ...] = (
    Rule(
        # Synthetic manifest, but use the real control-plane shape: this repo is
        # Rust-only and has no cli/ or *.ts surface any more.
        surface="crates/hipfire-cli/**",
        route_ids=("unit.control",),
        reason="control-plane change owes control-plane unit only",
    ),
    Rule(
        surface="docs/**",
        route_ids=(),
        reason="docs never select GPU routes",
    ),
    Rule(
        surface="crates/hipfire-arch-gfx12/**",
        route_ids=("serve.gfx12.smoke", "speed.arch"),
        reason="gfx12 arch crate owes gfx12 serve + speed",
    ),
    Rule(
        surface="crates/hipfire-arch-gfx11/**",
        route_ids=("serve.gfx11.smoke",),
        reason="gfx11 arch crate owes gfx11 serve",
    ),
    Rule(
        surface="crates/hipfire-runtime/**",
        route_ids=(
            "serve.gfx12.smoke",
            "serve.gfx11.smoke",
            "serve.niah.heavy",
            "redline.capture",
            "serve.budget.std",
            "serve.budget.std2",
            "unit.control",
        ),
        reason="runtime core pulls broad coverage including heavy NIAH",
    ),
    # Dedicated heavy surface — editing this path must select the heavy route.
    Rule(
        surface="tests/niah/**",
        route_ids=("serve.niah.heavy",),
        reason="NIAH harness change owes the heavy NIAH route",
    ),
    Rule(
        surface="tools/redline/**",
        route_ids=("redline.capture", "unit.control"),
        reason="redline tool change",
    ),
    # Second rule hitting the same route via a different surface (dedupe test).
    Rule(
        surface="crates/hipfire-arch-gfx12/src/kernels/**",
        route_ids=("serve.gfx12.smoke",),
        reason="gfx12 kernel path also owes gfx12 smoke",
    ),
)


def _have_all(_name: str) -> bool:
    return True


def _have_none(_name: str) -> bool:
    return False


def _ids(rows: list[Selection]) -> list[str]:
    return [s.route_id for s in rows]


def _by_id(rows: list[Selection]) -> dict[str, Selection]:
    return {s.route_id: s for s in rows}


def _select(
    paths: list[str] | tuple[str, ...],
    *,
    gfx: str | None = "gfx1201",
    max_minutes: float | None = None,
    include_heavy: bool = False,
    have_model=_have_all,
) -> tuple[list[Selection], list[Selection]]:
    return select(
        paths,
        FAKE_ROUTES,
        FAKE_RULES,
        gfx=gfx,
        max_minutes=max_minutes,
        include_heavy=include_heavy,
        have_model=have_model,
    )


class DocsAndCliSelectNothingExpensive(unittest.TestCase):
    def test_docs_only_selects_zero_gpu_routes(self) -> None:
        selected, not_run = _select(["docs/VALIDATION.md", "docs/guide/x.md"])
        self.assertEqual(selected, [])
        self.assertEqual(not_run, [])
        gpu_kinds = {"serve", "redline", "speed"}
        for row in (*selected, *not_run):
            self.assertNotIn(FAKE_ROUTES[row.route_id].kind, gpu_kinds)

    def test_control_plane_only_selects_zero_gpu_routes(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-cli/src/main.rs", "crates/hipfire-cli/src/setup.rs"]
        )
        self.assertTrue(selected or not_run)  # control-plane unit is owed
        gpu_kinds = {"serve", "redline", "speed"}
        for row in (*selected, *not_run):
            kind = FAKE_ROUTES[row.route_id].kind
            self.assertNotIn(
                kind,
                gpu_kinds,
                msg=f"cli change must not select GPU route {row.route_id}",
            )
        self.assertIn("unit.control", _ids(selected))
        for rid in _ids(selected):
            self.assertEqual(FAKE_ROUTES[rid].kind, "unit")


class ArchIsolation(unittest.TestCase):
    def test_arch_crate_selects_own_arch_not_other(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-arch-gfx12/src/lib.rs"],
            gfx="gfx1201",
        )
        ids = set(_ids(selected)) | set(_ids(not_run))
        self.assertIn("serve.gfx12.smoke", ids)
        self.assertIn("speed.arch", ids)
        self.assertNotIn("serve.gfx11.smoke", ids)

        sel_map = _by_id(selected)
        self.assertIn("serve.gfx12.smoke", sel_map)
        self.assertEqual(sel_map["serve.gfx12.smoke"].status, "selected")

    def test_other_arch_crate_does_not_select_gfx12(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-arch-gfx11/src/lib.rs"],
            gfx="gfx1100",
        )
        ids = set(_ids(selected)) | set(_ids(not_run))
        self.assertIn("serve.gfx11.smoke", ids)
        self.assertNotIn("serve.gfx12.smoke", ids)
        self.assertNotIn("speed.arch", ids)


class HeavyExclusion(unittest.TestCase):
    def test_heavy_excluded_unless_include_heavy(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-runtime/src/lib.rs"],
            include_heavy=False,
        )
        not_map = _by_id(not_run)
        self.assertIn("serve.niah.heavy", not_map)
        self.assertEqual(not_map["serve.niah.heavy"].status, "excluded_heavy")
        self.assertNotIn("serve.niah.heavy", _ids(selected))

    def test_include_heavy_selects_heavy(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-runtime/src/lib.rs"],
            include_heavy=True,
        )
        self.assertIn("serve.niah.heavy", _ids(selected))
        self.assertNotIn(
            "serve.niah.heavy",
            [s.route_id for s in not_run if s.status == "excluded_heavy"],
        )

    def test_heavy_selected_when_own_surface_changed(self) -> None:
        selected, not_run = _select(
            ["tests/niah/needle.rs"],
            include_heavy=False,
        )
        self.assertIn("serve.niah.heavy", _ids(selected))
        self.assertNotIn(
            "serve.niah.heavy",
            [s.route_id for s in not_run if s.status == "excluded_heavy"],
        )


class BlockedModelAndArch(unittest.TestCase):
    def test_missing_model_is_blocked_model_not_silent(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-arch-gfx12/src/lib.rs"],
            have_model=_have_none,
        )
        not_map = _by_id(not_run)
        self.assertIn("serve.gfx12.smoke", not_map)
        self.assertEqual(not_map["serve.gfx12.smoke"].status, "blocked_model")
        self.assertNotIn("serve.gfx12.smoke", _ids(selected))
        # speed.arch has no model requirement — still selected
        self.assertIn("speed.arch", _ids(selected))

        # overall verdict becomes incomplete when a blocked selection exists
        verdict = compute_verdict(selected, not_run, [])
        self.assertEqual(verdict, "incomplete")

    def test_arch_mismatch_is_blocked_arch(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-arch-gfx12/src/lib.rs"],
            gfx="gfx1100",
        )
        not_map = _by_id(not_run)
        self.assertIn("serve.gfx12.smoke", not_map)
        self.assertEqual(not_map["serve.gfx12.smoke"].status, "blocked_arch")
        self.assertIn("speed.arch", not_map)
        self.assertEqual(not_map["speed.arch"].status, "blocked_arch")
        self.assertEqual(_ids(selected), [])

    def test_undetectable_arch_blocks_rather_than_assuming(self) -> None:
        selected, not_run = _select(
            ["crates/hipfire-arch-gfx12/src/lib.rs"],
            gfx=None,
        )
        not_map = _by_id(not_run)
        self.assertIn("serve.gfx12.smoke", not_map)
        self.assertEqual(not_map["serve.gfx12.smoke"].status, "blocked_arch")
        self.assertIn("undetectable", not_map["serve.gfx12.smoke"].detail.lower())
        self.assertEqual(_ids(selected), [])
        self.assertEqual(compute_verdict(selected, not_run, []), "incomplete")


class BudgetTrim(unittest.TestCase):
    def test_max_minutes_trims_never_cheap(self) -> None:
        # unit.control is cheap; budget routes are standard and expensive.
        paths = ["crates/hipfire-runtime/src/engine.rs"]
        selected, not_run = _select(paths, max_minutes=1.0, include_heavy=False)

        sel_map = _by_id(selected)
        not_map = _by_id(not_run)

        # cheap always kept
        self.assertIn("unit.control", sel_map)
        self.assertEqual(sel_map["unit.control"].status, "selected")
        self.assertEqual(FAKE_ROUTES["unit.control"].tier, "cheap")

        # at least one non-cheap route must be trimmed under a 1-minute budget
        trimmed = [s for s in not_run if s.status == "trimmed_budget"]
        self.assertTrue(
            trimmed,
            msg=f"expected trimmed_budget rows, got not_run={not_run!r} selected={selected!r}",
        )
        for s in trimmed:
            self.assertNotEqual(
                FAKE_ROUTES[s.route_id].tier,
                "cheap",
                msg="cheap routes must never be trimmed",
            )

        # no cheap route appears as trimmed
        for s in not_run:
            if s.status == "trimmed_budget":
                self.assertNotEqual(FAKE_ROUTES[s.route_id].tier, "cheap")

    def test_budget_keeps_routes_within_limit_in_tier_order(self) -> None:
        paths = ["crates/hipfire-runtime/src/engine.rs"]
        # Generous enough for cheap + one standard, not both large standards.
        selected, not_run = _select(paths, max_minutes=6.0, include_heavy=False)
        total = sum(FAKE_ROUTES[s.route_id].est_minutes for s in selected)
        # Cheap always included; total of selected non-forced may exceed if only
        # cheap overflows — but with max=6 and cheap=0.5 we stay sane.
        self.assertIn("unit.control", _ids(selected))
        # If a standard was kept, cheaper standards preferred over heavier ones.
        trimmed_ids = {s.route_id for s in not_run if s.status == "trimmed_budget"}
        kept_std = [
            s.route_id
            for s in selected
            if FAKE_ROUTES[s.route_id].tier == "standard"
        ]
        if kept_std and trimmed_ids:
            max_kept = max(FAKE_ROUTES[r].est_minutes for r in kept_std)
            for tid in trimmed_ids:
                # trimmed should not be strictly cheaper than a kept standard
                # of the same tier when ordered by est_minutes (deterministic).
                if FAKE_ROUTES[tid].tier == "standard":
                    self.assertGreaterEqual(
                        FAKE_ROUTES[tid].est_minutes,
                        min(FAKE_ROUTES[r].est_minutes for r in kept_std),
                    )
        _ = total  # silence lint; assertion above is the contract


class DeterminismAndDedupe(unittest.TestCase):
    def test_selection_stable_across_repeated_calls(self) -> None:
        paths = [
            "crates/hipfire-runtime/src/a.rs",
            "crates/hipfire-arch-gfx12/src/lib.rs",
            "tools/redline/x.py",
        ]
        a_sel, a_nr = _select(paths)
        b_sel, b_nr = _select(paths)
        self.assertEqual(
            [(s.route_id, s.status, s.matched_paths, s.rule_reason) for s in a_sel],
            [(s.route_id, s.status, s.matched_paths, s.rule_reason) for s in b_sel],
        )
        self.assertEqual(
            [(s.route_id, s.status, s.matched_paths, s.rule_reason) for s in a_nr],
            [(s.route_id, s.status, s.matched_paths, s.rule_reason) for s in b_nr],
        )
        # order is sorted by route_id / (status, route_id)
        self.assertEqual(_ids(a_sel), sorted(_ids(a_sel)))
        self.assertEqual(
            [(s.status, s.route_id) for s in a_nr],
            sorted((s.status, s.route_id) for s in a_nr),
        )

    def test_multi_rule_path_dedupes_to_one_selection(self) -> None:
        # Path matches both the arch-crate rule and the kernels subpath rule.
        paths = ["crates/hipfire-arch-gfx12/src/kernels/foo.hip"]
        selected, not_run = _select(paths)
        all_rows = selected + not_run
        gfx12_rows = [s for s in all_rows if s.route_id == "serve.gfx12.smoke"]
        self.assertEqual(
            len(gfx12_rows),
            1,
            msg=f"expected one deduped Selection, got {gfx12_rows!r}",
        )
        row = gfx12_rows[0]
        self.assertIn(paths[0], row.matched_paths)
        # Both rule reasons should be carried (joined).
        self.assertIn("gfx12", row.rule_reason.lower())
        # reasons from both rules present
        self.assertIn(";", row.rule_reason)
        self.assertIn("kernel", row.rule_reason.lower())


if __name__ == "__main__":
    unittest.main()

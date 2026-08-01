# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Manifest invariants + cheap real-manifest smoke (no GPU, no models)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from tools.change_gate.model import Route, Rule
from tools.change_gate.routes import ROUTES, RULES, routes_by_id, rules
from tools.change_gate.selector import select


_DOTTED_ID = re.compile(r"^[a-z][a-z0-9_-]*(\.[a-z0-9][a-z0-9._-]*)+$")
_TIERS = frozenset({"cheap", "standard", "heavy"})
_GPU_PREFIXES = ("serve.", "redline.", "speed.")


def _tier_for_minutes(est: float) -> str:
    if est < 2.0:
        return "cheap"
    if est <= 15.0:
        return "standard"
    return "heavy"


class ManifestInvariants(unittest.TestCase):
    def test_routes_export_is_dict_of_route(self) -> None:
        self.assertIsInstance(ROUTES, dict)
        self.assertGreater(len(ROUTES), 0)
        for key, route in ROUTES.items():
            self.assertIsInstance(route, Route)
            self.assertEqual(key, route.id)

    def test_rules_export(self) -> None:
        self.assertTrue(RULES)
        for rule in RULES:
            self.assertIsInstance(rule, Rule)

    def test_routes_by_id_and_rules_helpers(self) -> None:
        by_id = routes_by_id()
        self.assertEqual(set(by_id), set(ROUTES))
        self.assertEqual(tuple(rules()), tuple(RULES))

    def test_every_rule_route_id_resolves(self) -> None:
        missing: list[str] = []
        for rule in RULES:
            for rid in rule.route_ids:
                if rid not in ROUTES:
                    missing.append(f"{rule.surface!r} -> {rid}")
        self.assertEqual(missing, [], msg=f"unresolved route ids: {missing}")

    def test_every_why_nonempty(self) -> None:
        empty = [r.id for r in ROUTES.values() if not (r.why and r.why.strip())]
        self.assertEqual(empty, [], msg=f"routes with empty why: {empty}")

    def test_tier_literals_and_est_consistency(self) -> None:
        bad_tier: list[str] = []
        inconsistent: list[str] = []
        for r in ROUTES.values():
            if r.tier not in _TIERS:
                bad_tier.append(f"{r.id}={r.tier!r}")
                continue
            expected = _tier_for_minutes(r.est_minutes)
            # Boundary: cheap <2, standard 2-15 inclusive, heavy >15.
            if r.tier == "cheap" and not (r.est_minutes < 2.0):
                inconsistent.append(f"{r.id} tier=cheap est={r.est_minutes}")
            elif r.tier == "standard" and not (2.0 <= r.est_minutes <= 15.0):
                inconsistent.append(f"{r.id} tier=standard est={r.est_minutes}")
            elif r.tier == "heavy" and not (r.est_minutes > 15.0):
                inconsistent.append(f"{r.id} tier=heavy est={r.est_minutes}")
            # Cross-check helper agrees with declared tier.
            if expected != r.tier:
                # Allow standard exactly at 2.0 / 15.0 already covered above;
                # flag only if helper disagrees with the ranges we enforce.
                if not (
                    (r.tier == "standard" and 2.0 <= r.est_minutes <= 15.0)
                    or (r.tier == "cheap" and r.est_minutes < 2.0)
                    or (r.tier == "heavy" and r.est_minutes > 15.0)
                ):
                    inconsistent.append(
                        f"{r.id} tier={r.tier} est={r.est_minutes} expected~{expected}"
                    )
        self.assertEqual(bad_tier, [], msg=f"bad tiers: {bad_tier}")
        self.assertEqual(inconsistent, [], msg=f"tier/est mismatch: {inconsistent}")

    def test_no_non_heavy_over_15_minutes(self) -> None:
        offenders = [
            f"{r.id} tier={r.tier} est={r.est_minutes}"
            for r in ROUTES.values()
            if r.tier != "heavy" and r.est_minutes > 15.0
        ]
        self.assertEqual(offenders, [], msg=f"non-heavy >15min: {offenders}")

    def test_route_ids_unique_and_dotted(self) -> None:
        ids = [r.id for r in ROUTES.values()]
        self.assertEqual(len(ids), len(set(ids)), msg="duplicate route ids")
        bad = [rid for rid in ids if not _DOTTED_ID.match(rid)]
        self.assertEqual(bad, [], msg=f"non-dotted route ids: {bad}")

    def test_rule_reasons_nonempty(self) -> None:
        empty = [r.surface for r in RULES if not (r.reason and r.reason.strip())]
        self.assertEqual(empty, [], msg=f"rules with empty reason: {empty}")


class RealManifestSmoke(unittest.TestCase):
    """Cheap smoke against the real ROUTES/RULES — no host models required."""

    def _select_paths(self, paths: list[str]):
        # have_model always True so we observe selection, not model blocking.
        return select(
            paths,
            routes_by_id(),
            rules(),
            gfx="gfx1201",
            include_heavy=False,
            have_model=lambda _m: True,
        )

    def test_docs_selects_no_gpu_routes(self) -> None:
        selected, not_run = self._select_paths(["docs/x.md"])
        for row in (*selected, *not_run):
            for prefix in _GPU_PREFIXES:
                self.assertFalse(
                    row.route_id.startswith(prefix),
                    msg=f"docs change selected GPU route {row.route_id}",
                )

    def test_cli_selects_no_gpu_routes(self) -> None:
        selected, not_run = self._select_paths(["crates/hipfire-cli/src/main.rs"])
        for row in (*selected, *not_run):
            for prefix in _GPU_PREFIXES:
                self.assertFalse(
                    row.route_id.startswith(prefix),
                    msg=f"cli change selected GPU route {row.route_id}",
                )


class SurfaceHygiene(unittest.TestCase):
    """Invariants that catch surfaces which can never match.

    A rule whose surface cannot match anything is a silent coverage hole: the
    gate looks like it guards something and guards nothing. Both failure modes
    below shipped once and were caught by hand, so they are pinned here.
    """

    def test_no_brace_globs_in_fnmatch_surfaces(self) -> None:
        """`fnmatch` has no brace expansion, so "{a,b}" silently never matches."""
        offenders = [
            r.surface
            for r in RULES
            if "{" in r.surface and not r.surface.startswith("re:")
        ]
        self.assertEqual(
            offenders,
            [],
            msg="brace globs never match under fnmatch; use a 're:' surface instead",
        )

    def test_no_dead_typescript_or_cli_surfaces(self) -> None:
        """The control plane is Rust-only: no `cli/` tree and no `.ts` files."""
        offenders = [
            r.surface
            for r in RULES
            if ".ts" in r.surface
            or r.surface.startswith("cli/")
            or "/cli/" in r.surface
        ]
        self.assertEqual(
            offenders,
            [],
            msg="this repo has no cli/ or *.ts surface; such a rule can never fire",
        )

    def test_every_surface_matches_at_least_one_tracked_path(self) -> None:
        """A surface matching nothing in the tree is dead weight.

        Allowlist surfaces that intentionally guard files which may not exist
        yet (e.g. a not-yet-added arch or kernel); everything else must match
        something that is actually checked in.
        """
        import subprocess

        from tools.change_gate.selector import _surface_matches

        repo = Path(__file__).resolve().parents[3]

        def _git(*args: str) -> list[str]:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return []
            return [p for p in proc.stdout.splitlines() if p]

        # Untracked-but-not-ignored counts: a rule guarding a freshly added
        # package is live coverage, not dead weight, before the first commit.
        tracked = _git("ls-files") + _git("ls-files", "--others", "--exclude-standard")
        if not tracked:
            self.skipTest("git ls-files unavailable")

        allow: set[str] = set()  # add a surface here only with a reason in review
        dead = [
            r.surface
            for r in RULES
            if r.surface not in allow
            and not any(_surface_matches(r.surface, p) for p in tracked)
        ]
        self.assertEqual(
            dead,
            [],
            msg="rule surfaces match no tracked file (dead coverage): " + repr(dead),
        )


if __name__ == "__main__":
    unittest.main()

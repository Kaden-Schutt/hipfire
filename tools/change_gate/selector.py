# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Change → route selection for the targeted validation gate.

``select`` is pure given paths + manifest + gfx + a model-check callable:
no subprocess and no filesystem access beyond the injected ``have_model``.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from tools.change_gate import hostinfo
from tools.change_gate.model import Route, Rule, Selection

_TIER_ORDER = {"cheap": 0, "standard": 1, "heavy": 2}


def _run_git(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_rev_parse(ref: str) -> str:
    proc = _run_git(["rev-parse", ref])
    if proc.returncode != 0:
        raise RuntimeError(f"git rev-parse {ref!r} failed: {(proc.stderr or proc.stdout).strip()}")
    return (proc.stdout or "").strip()


def _git_name_only(*diff_args: str) -> list[str]:
    proc = _run_git(["diff", "--name-only", *diff_args])
    if proc.returncode != 0:
        # Empty tree / missing paths still ok; real errors surface empty + message.
        err = (proc.stderr or "").strip()
        if err and "unknown revision" in err.lower():
            raise RuntimeError(f"git diff failed: {err}")
        if proc.returncode not in (0, 1):
            # git diff returns 0 always for name-only unless bad rev
            if err:
                raise RuntimeError(f"git diff failed: {err}")
    paths = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return paths


def _git_untracked() -> list[str]:
    proc = _run_git(["ls-files", "--others", "--exclude-standard"])
    if proc.returncode != 0:
        return []
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def _is_ancestor(maybe_ancestor: str, descendant: str) -> bool:
    proc = _run_git(["merge-base", "--is-ancestor", maybe_ancestor, descendant])
    return proc.returncode == 0


def _merge_base(a: str, b: str) -> str | None:
    proc = _run_git(["merge-base", a, b])
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def changed_files(
    base: str | None,
    head: str = "HEAD",
) -> tuple[list[str], str, str, bool]:
    """Return ``(paths, base_sha, head_sha, dirty)`` for the change under test.

    * When ``base`` is set: ``git diff --name-only <base>...<head>`` (three-dot),
      **unioned with any uncommitted work** (unstaged, staged, and untracked).
      If ``base`` is not an ancestor of ``head``, fall back to
      ``git merge-base(base, head)...head``.
    * When ``base`` is ``None``: working tree + index vs ``HEAD``
      (``git diff --name-only HEAD``, ``--cached``, plus untracked).

    ``dirty`` is True when the worktree or index differs from HEAD or has
    untracked files.

    The union matters: a gate is normally run *before* committing or opening a
    PR, so a range diff alone would silently ignore exactly the code under
    test. Reporting ``dirty`` while excluding the dirty paths would let an
    unreviewed change collect a ``pass`` on an empty selection.
    """
    head_sha = _git_rev_parse(head)

    if base is None:
        base_sha = head_sha
        unstaged = _git_name_only("HEAD")
        staged = _git_name_only("--cached")
        untracked = _git_untracked()
        paths = sorted(set(unstaged) | set(staged) | set(untracked))
        porcelain = _run_git(["status", "--porcelain"])
        dirty = bool((porcelain.stdout or "").strip())
        return paths, base_sha, head_sha, dirty

    base_sha = _git_rev_parse(base)
    left = base
    if not _is_ancestor(base_sha, head_sha):
        mb = _merge_base(base_sha, head_sha)
        if mb is None:
            raise RuntimeError(
                f"cannot find merge-base between {base!r} ({base_sha[:12]}) "
                f"and {head!r} ({head_sha[:12]})"
            )
        left = mb
        base_sha = mb

    range_paths = set(_git_name_only(f"{left}...{head}"))
    porcelain = _run_git(["status", "--porcelain"])
    dirty = bool((porcelain.stdout or "").strip())
    # Union in uncommitted work; see the docstring for why this is not optional.
    if dirty:
        range_paths |= set(_git_name_only("HEAD"))
        range_paths |= set(_git_name_only("--cached"))
        range_paths |= set(_git_untracked())
    paths = sorted(range_paths)
    return paths, base_sha, head_sha, dirty


def _surface_matches(surface: str, path: str) -> bool:
    if surface.startswith("re:"):
        pattern = surface[3:]
        try:
            return re.search(pattern, path) is not None
        except re.error:
            return False
    # fnmatch globs are matched against the full repo-relative path.
    # Also allow matching the basename for simple patterns like "*.rs".
    if fnmatch.fnmatch(path, surface):
        return True
    base = os.path.basename(path)
    if base != path and fnmatch.fnmatch(base, surface):
        return True
    return False


def heavy_directly_matched(
    route_id: str,
    matched_rule_surfaces: Iterable[str],
    routes_by_id: Mapping[str, Route],
) -> bool:
    """True when a heavy route is owed because the change hit *its* surface.

    A ``tier == "heavy"`` route is normally ``excluded_heavy`` unless
    ``include_heavy`` is set **or** a rule whose ``route_ids`` contain this
    route matched paths under a surface that is "about" this route — i.e. the
    change is specifically in the surface the heavy route guards.

    Heuristic (documented, deterministic): the matched rule's ``surface``
    string contains the route id, or the route id's final dotted segment, as a
    substring (case-sensitive), **or** the rule lists only this single route
    id (a dedicated guard). This keeps "CLI-only change must not pull 200K
    NIAH" while still selecting a heavy route when you edit the path it
    protects.
    """
    route = routes_by_id.get(route_id)
    if route is None:
        return False
    segment = route_id.rsplit(".", 1)[-1]
    surfaces = list(matched_rule_surfaces)
    # Caller may pass rule objects via a side channel — also accept Rule.reason
    # is not used here; surfaces only.
    for surface in surfaces:
        if route_id in surface or (segment and segment in surface):
            return True
    return False


def _rule_is_dedicated(rule: Rule, route_id: str) -> bool:
    return list(rule.route_ids) == [route_id]


def select(
    paths: Sequence[str],
    routes_by_id: Mapping[str, Route],
    rules: Sequence[Rule],
    *,
    gfx: str | None,
    models_dir: Path | str | None = None,
    max_minutes: float | None = None,
    include_heavy: bool = False,
    have_model: Callable[[str], bool] | None = None,
) -> tuple[list[Selection], list[Selection]]:
    """Map changed paths to routes; return ``(to_run, not_run)``.

    Every candidate route becomes a :class:`Selection` — nothing is dropped
    silently. ``to_run`` holds ``status == "selected"``; ``not_run`` holds
    blocked / excluded / trimmed rows.

    ``have_model`` defaults to :func:`tools.change_gate.hostinfo.have_model`
    (optionally bound to ``models_dir``). Inject a pure callable in tests.
    """
    if have_model is None:
        root = Path(models_dir) if models_dir is not None else hostinfo.models_dir()

        def have_model(basename: str, _root: Path = root) -> bool:
            return hostinfo.have_model(basename, models_dir=_root)

    # route_id → (paths set, reason parts, matched rules, matched surfaces)
    hit: dict[str, dict] = {}

    for rule in rules:
        matched = sorted({p for p in paths if _surface_matches(rule.surface, p)})
        if not matched:
            continue
        for rid in rule.route_ids:
            if rid not in routes_by_id:
                continue
            bucket = hit.setdefault(
                rid,
                {
                    "paths": set(),
                    "reasons": [],
                    "rules": [],
                    "surfaces": [],
                },
            )
            bucket["paths"].update(matched)
            if rule.reason not in bucket["reasons"]:
                bucket["reasons"].append(rule.reason)
            bucket["rules"].append(rule)
            if rule.surface not in bucket["surfaces"]:
                bucket["surfaces"].append(rule.surface)

    if not hit:
        return [], []

    # Build preliminary selections with filter statuses.
    prelim: list[Selection] = []
    for rid in sorted(hit.keys()):
        route = routes_by_id[rid]
        info = hit[rid]
        matched_paths = tuple(sorted(info["paths"]))
        rule_reason = "; ".join(info["reasons"])

        # Arch gate
        if route.arches:
            if gfx is None:
                prelim.append(
                    Selection(
                        route_id=rid,
                        matched_paths=matched_paths,
                        rule_reason=rule_reason,
                        status="blocked_arch",
                        detail="host GPU arch undetectable; route requires "
                        + ",".join(route.arches),
                    )
                )
                continue
            if gfx not in route.arches:
                prelim.append(
                    Selection(
                        route_id=rid,
                        matched_paths=matched_paths,
                        rule_reason=rule_reason,
                        status="blocked_arch",
                        detail=f"host arch {gfx} not in route arches {list(route.arches)}",
                    )
                )
                continue

        # Model gate
        missing = [m for m in route.models if not have_model(m)]
        if missing:
            prelim.append(
                Selection(
                    route_id=rid,
                    matched_paths=matched_paths,
                    rule_reason=rule_reason,
                    status="blocked_model",
                    detail="missing model(s): " + ", ".join(missing),
                )
            )
            continue

        # Heavy exclusion
        if route.tier == "heavy" and not include_heavy:
            dedicated = any(_rule_is_dedicated(r, rid) for r in info["rules"])
            direct = dedicated or heavy_directly_matched(
                rid, info["surfaces"], routes_by_id
            )
            if not direct:
                prelim.append(
                    Selection(
                        route_id=rid,
                        matched_paths=matched_paths,
                        rule_reason=rule_reason,
                        status="excluded_heavy",
                        detail=(
                            "tier=heavy excluded (pass include_heavy=True or "
                            "change a surface dedicated to this route)"
                        ),
                    )
                )
                continue

        prelim.append(
            Selection(
                route_id=rid,
                matched_paths=matched_paths,
                rule_reason=rule_reason,
                status="selected",
                detail=f"selected ({route.tier}, ~{route.est_minutes:g} min): {route.why}",
            )
        )

    # Split selected vs blocked-so-far
    selected = [s for s in prelim if s.status == "selected"]
    not_run = [s for s in prelim if s.status != "selected"]

    # Budget trim — never trim cheap; order cheap→standard→heavy, then est_minutes, then id
    if max_minutes is not None and selected:

        def sort_key(s: Selection) -> tuple:
            r = routes_by_id[s.route_id]
            return (
                _TIER_ORDER.get(r.tier, 99),
                r.est_minutes,
                s.route_id,
            )

        ordered = sorted(selected, key=sort_key)
        keep: list[Selection] = []
        trim: list[Selection] = []
        total = 0.0
        for s in ordered:
            r = routes_by_id[s.route_id]
            # Never trim cheap routes — they always run and still consume budget.
            if r.tier == "cheap":
                keep.append(s)
                total += r.est_minutes
                continue
            if total + r.est_minutes <= max_minutes:
                keep.append(s)
                total += r.est_minutes
            else:
                trim.append(
                    Selection(
                        route_id=s.route_id,
                        matched_paths=s.matched_paths,
                        rule_reason=s.rule_reason,
                        status="trimmed_budget",
                        detail=(
                            f"est {r.est_minutes:g} min exceeds remaining "
                            f"budget ({max_minutes:g} max, {total:g} used)"
                        ),
                    )
                )
        selected = sorted(keep, key=lambda s: s.route_id)
        not_run.extend(trim)

    selected = sorted(selected, key=lambda s: s.route_id)
    not_run = sorted(not_run, key=lambda s: (s.status, s.route_id))
    return selected, not_run

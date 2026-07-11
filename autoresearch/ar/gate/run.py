# Copyright (c) Kaden Schutt
"""ar.gate.run — the Tier-3 PR-gate execution path (spec §4/§8.1/§12).

run_pr_gate ties the no-GPU dispatch core together: read the PR's changed files,
classify the risk, run every arch's gate, and reduce to a PR outcome + comment.
The per-arch gate is an injected seam:

  * --dry-run  -> ``stub_arch_gate`` (no GPU, no daemon build): exercises the whole
                  classify -> route -> decide -> comment pipeline on a REAL diff.
  * real (on a self-hosted runner) -> ``live_arch_gate`` builds base/head daemons
                  and runs the Phase-1 ``run_gate`` + Phase-2 ``gate4`` over the
                  ``LiveServeRunner`` GPU adapter.

Only ``live_arch_gate`` touches the GPU/ROCm/cargo; everything else is pure and
unit-testable with an injected ``run_git`` + ``arch_gate_fn``.
"""
from __future__ import annotations

import glob
import json
import os
import re

from .config import GateConfig
from .merge import default_run_git
from .outcome import decide_pr, format_pr_comment
from .routing import classify_pr

__all__ = ["changed_files", "diff_lines", "stub_arch_gate", "run_pr_gate",
           "live_arch_gate", "collect_cell_data", "grade_collected", "interpret_results", "run_behavior_plan"]


def changed_files(base, head, repo, run_git=None) -> list[str]:
    """Repo-relative paths changed between base..head (git diff --name-only)."""
    rc, out = (run_git or default_run_git)(repo, "diff", "--name-only", base, head)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def diff_lines(base, head, repo, run_git=None) -> int:
    """Total lines changed (added + deleted) between base..head, via numstat.
    Binary files (numstat '-') contribute 0. Used only to split low vs moderate."""
    rc, out = (run_git or default_run_git)(repo, "diff", "--numstat", base, head)
    total = 0
    for ln in out.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            for n in parts[:2]:
                if n.isdigit():
                    total += int(n)
    return total


def stub_arch_gate(arch, files, base, head, repo, cfg, verdict="PASS", reasons=None) -> dict:
    """A no-GPU stand-in for the per-arch gate — used by --dry-run to demonstrate
    the pipeline without building daemons or touching a GPU."""
    return {"arch": arch, "verdict": verdict,
            "reasons": list(reasons or ([] if verdict == "PASS" else [verdict])),
            "bod": None}


def run_pr_gate(*, base, head, repo, author, is_draft, helpful, cfg: GateConfig,
                arch_gate_fn, archs=None, run_git=None) -> dict:
    """Execute the PR gate: classify the diff, gate every arch, decide the outcome.

    Returns {pr_class, route, arch_results, outcome, comment}. ``arch_gate_fn`` is
    the injected per-arch gate: (arch, files, base, head, repo, cfg) -> arch result
    dict {arch, verdict, reasons, bod}."""
    files = changed_files(base, head, repo, run_git=run_git)
    lines = diff_lines(base, head, repo, run_git=run_git)
    pr_class = classify_pr(files, lines_changed=lines)
    route = cfg.route(pr_class)
    arch_results = [arch_gate_fn(a, files, base, head, repo, cfg) for a in (archs or cfg.archs)]
    outcome = decide_pr(arch_results=arch_results, author=author, is_draft=is_draft,
                        helpful=helpful, cfg=cfg)
    return {"pr_class": pr_class, "route": route, "arch_results": arch_results,
            "outcome": outcome, "comment": format_pr_comment(outcome, arch_results)}


def run_behavior_plan(plan_path, *, repo, verdict_dir, base, head, run_git=None) -> dict:
    """Load Claude's dispatch plan.json, floor its risk (classify_pr), and run every
    bespoke behavior test via codex (agent_exec) GENERALLY on-box — the piece that
    tests behaviors serve_harness cannot reach. Returns {plan, behavior_results}."""
    from ..agent_exec import run_round_resilient  # lazy: codex shell-out (+grok fallback), prod only
    from . import dispatch

    with open(plan_path) as fh:
        raw = json.load(fh)
    plan = dispatch.parse_plan(raw, changed_files(base, head, repo, run_git=run_git))
    # run_round_resilient: on a codex usage-limit, luna/terra tests fall back to grok;
    # sol tests wait for codex to reset (no silent downgrade). Drop-in for run_round.
    results = dispatch.run_behavior_tests(
        plan["behavior_tests"], agent_exec_fn=run_round_resilient, cwd=repo, verdict_dir=verdict_dir)
    return {"plan": plan, "behavior_results": results}


def interpret_results(*, results_dir, base, head, repo, author, is_draft, helpful,
                      cfg: GateConfig, run_git=None, behavior_results=None) -> dict:
    """Aggregate per-arch result JSONs (each an arch-result dict emitted by a matrix
    job) plus any bespoke behavior-test results, and decide the PR outcome. A failed
    behavior test folds in as a synthetic REJECT arch (so decide_pr routes to BOD):
    the verdict is the serve_harness floor AND every behavior test (spec §8)."""
    arch_results = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p) as fh:
            arch_results.append(json.load(fh))
    failed_behaviors = [b for b in (behavior_results or []) if not b.get("passed")]
    if failed_behaviors:
        arch_results.append({
            "arch": "behavior", "verdict": "REJECT",
            "reasons": [f"behavior:{b.get('what', '?')}" for b in failed_behaviors],
            "bod": None,
        })
    files = changed_files(base, head, repo, run_git=run_git)
    pr_class = classify_pr(files, lines_changed=diff_lines(base, head, repo, run_git=run_git))
    outcome = decide_pr(arch_results=arch_results, author=author, is_draft=is_draft,
                        helpful=helpful, cfg=cfg)
    return {"pr_class": pr_class, "route": cfg.route(pr_class), "arch_results": arch_results,
            "outcome": outcome, "comment": format_pr_comment(outcome, arch_results)}


_ARCH_SUFFIX = re.compile(r"\.(gfx[0-9a-z_]+)\.hip$")


def daemon_touched(files) -> bool:
    """True iff the diff changes code COMPILED INTO the daemon (so the daemon binary
    can differ base-vs-head). Changes under autoresearch/, docs/, .github/, cli/,
    scripts/ etc. do NOT touch the daemon — base≡head there, so every arch defers."""
    return any(f.startswith("crates/") or f.startswith("kernels/") or f.startswith("Cargo.")
               for f in files)


def _file_archs(f, cfg) -> list:
    """Which GATED archs a single changed file affects.
    - crates/ or Cargo.*  → ALL archs (shared Rust in the daemon).
    - kernels/**/*.gfxNNNN.hip → just gfxNNNN (an arch-suffixed kernel is that arch's
      variant only); a suffix for an arch we don't gate → none.
    - kernels/** shared (no arch suffix) → ALL archs (conservative — a shared kernel
      or an internal #if-gated block could touch any arch).
    - anything else (docs/ar/CI) → none."""
    if f.startswith("crates/") or f.startswith("Cargo."):
        return list(cfg.archs)
    if f.startswith("kernels/"):
        m = _ARCH_SUFFIX.search(f)
        if m:
            a = m.group(1)
            return [a] if a in cfg.archs else []
        return list(cfg.archs)
    return []


def affected_archs(files, cfg) -> list:
    """The archs whose GPU battery must actually run (§4.1 arch→box deferral), ordered
    by ``cfg.archs``. An arch-SPECIFIC change (``foo.gfx1201.hip``) affects only that
    arch, so the other box defers it (hipx runs nothing, hiptrx runs gfx1201); a shared
    daemon change affects ALL archs; a non-daemon change (docs/ar/CI) affects NONE.
    Combined with the box→arch matrix ownership, this makes the deferral faithful: a
    box only ever runs an arch it OWNS *and* the diff affects."""
    got = set()
    for f in files:
        got.update(_file_archs(f, cfg))
    return [a for a in cfg.archs if a in got]


def collect_cell_data(arch, files, base, head, repo, cfg, *, dev=None, models=None) -> dict:
    """MECHANICAL eval (a TOOL the codex agent drives) — build base+head daemons and run
    ``serve_harness.py`` base-vs-head per model. Returns RAW rows + any errors AS DATA
    (never crashes on empty/build-fail), so the AGENT can judge and ADAPT (re-run an
    empty cell, diagnose a build error) rather than a program hard-failing. Deferral
    short-circuits with no build (§4.1). Cross-arch leak (deferred archs only) is a
    file-based flag carried alongside, not a hard reject here — the agent weighs it.

    ``models`` = the SKUs to serve_harness — **Claude selects these from the diff** (a
    dense-only change → the 27B; an MoE/router change → a3b; a shared change → all) and
    codex passes them in; they are NOT hardcoded. Any SKU that doesn't fit ``arch`` is
    dropped. When ``models`` is None (e.g. Claude's dispatch was skipped on a workflow-
    touching PR), fall back to the canonical set for the arch (``cfg.models_for``)."""
    from .build import build_daemon
    from .device import resolve_device
    from . import serve_probe
    from ..certify import cross_arch

    if arch not in affected_archs(files, cfg):
        return {"arch": arch, "deferred": True, "cells": []}

    kernel_files = [f for f in files if f.startswith("kernels/") and f.endswith(".hip")]
    deferred_archs = [a for a in cfg.archs if a not in affected_archs(files, cfg)]
    bleed = [f for f in kernel_files if deferred_archs
             and cross_arch.check_cross_arch(f, arch, deferred_archs, repo, base_sha=base)]

    dev = resolve_device(arch, default=dev if dev is not None else 0)
    kv = getattr(cfg, "kv_mode", None) or "q8"
    models_dir = os.path.expanduser(os.environ.get("HIPFIRE_MODELS_DIR", "~/.hipfire/models"))
    out = {"arch": arch, "deferred": False, "dev": dev, "cross_arch_bleed": bleed}

    try:
        out["base_bin"] = build_daemon(base, repo)
        out["head_bin"] = build_daemon(head, repo)
    except RuntimeError as e:
        out["build_error"] = str(e)
        out["cells"] = []
        return out

    # Claude-selected models (filtered to those that fit the arch); canonical fallback.
    sel = [m for m in (models or []) if cfg.fits(m, arch)] or cfg.models_for(arch)
    out["models"] = sel
    cells, base_port = [], 11540 + dev * 40
    for i, m in enumerate(sel):
        mp = os.path.join(models_dir, m)
        cell = {"model": m}
        for role, daemon, port in (("base_rows", out["base_bin"], base_port + i * 2),
                                   ("head_rows", out["head_bin"], base_port + i * 2 + 1)):
            try:
                cell[role] = serve_probe.run_serve_harness(daemon, mp, dev, repo=repo, kv=kv, port=port)
            except RuntimeError as e:
                cell[role] = None
                cell.setdefault("errors", {})[role] = str(e)
        cells.append(cell)
    out["cells"] = cells
    return out


def grade_collected(collected, *, cfg) -> dict:
    """Deterministic baseline grade of ``collect_cell_data`` output → arch result (ledger
    rows + itemized BOD). This is the TOOL the codex agent runs, then reviews/adapts (e.g.
    re-collect an EMPTY cell) before emitting the final result. Never crashes on empty."""
    from . import serve_probe
    arch = collected["arch"]
    if collected.get("deferred"):
        return {"arch": arch, "verdict": "PASS", "reasons": ["deferred"], "bod": None,
                "rows": [], "tok_delta_pct": 0.0}
    if collected.get("build_error"):
        return {"arch": arch, "verdict": "REJECT", "reasons": ["build_fail"], "bod": None,
                "rows": [], "tok_delta_pct": 0.0, "detail": collected["build_error"]}

    rows, deltas = [], []
    for c in collected.get("cells", []):
        br, hr = c.get("base_rows"), c.get("head_rows")
        if not br or not hr:            # a serve error/empty -> EMPTY row (agent re-runs)
            rows.append(serve_probe._cell_row("EMPTY", arch=arch, model=c["model"],
                                              base_rows=br or [], head_rows=hr or [],
                                              parity={"content_exact": None, "empty": True}))
            continue
        rows.append(serve_probe.grade_cell(br, hr, arch=arch, model=c["model"], floor=cfg.floor))
        deltas.append(rows[-1].get("tok_delta_pct") or 0.0)

    tok_delta = min(deltas) if deltas else 0.0
    fails = [r for r in rows if r["verdict"] != "PASS"]
    if fails:
        blockers = [serve_probe.cell_blocker(r) for r in fails]
        return {"arch": arch, "verdict": "REJECT", "reasons": [b["kind"] for b in blockers],
                "bod": {"blockers": blockers,
                        "summary": f"{len(blockers)} cell(s) failed: " + ", ".join(b["detail"] for b in blockers)},
                "rows": rows, "tok_delta_pct": tok_delta}
    return {"arch": arch, "verdict": "PASS", "reasons": [], "bod": None, "rows": rows,
            "tok_delta_pct": tok_delta}


def live_arch_gate(arch, files, base, head, repo, cfg, *, dev=None, card=None, model=None) -> dict:
    """Deterministic per-arch gate = collect (build + serve_harness A/B) → grade. Kept for
    ``ar gate --run`` (local/fallback, no agent). The WORKFLOW instead has codex DRIVE
    collect+grade and adapt (re-run empties, judge cross-arch) — the agentic path."""
    return grade_collected(collect_cell_data(arch, files, base, head, repo, cfg, dev=dev), cfg=cfg)

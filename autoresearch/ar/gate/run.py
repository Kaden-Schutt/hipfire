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
           "live_arch_gate", "interpret_results", "run_behavior_plan"]


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


def live_arch_gate(arch, files, base, head, repo, cfg, *, dev=None, card=None, model=None) -> dict:
    """REAL per-arch gate (GPU) — self-hosted runner only. Deferral → cross-arch leak →
    BUILD base+head daemons (``gate.build``, sha-cached) → per (model,arch) A/B via
    ``scripts/serve_harness.py`` (``gate.serve_probe``): greedy content parity, decode
    tok/s perf (WIN-gate mirror), attractor coherence. serve_harness drives the daemon
    (spawn/warm/battery) so the gate reuses it rather than the untested LiveServeRunner
    raw-daemon+rocprof arms. GPU/cargo imported lazily; needs ROCm + the models under
    ``$HIPFIRE_MODELS_DIR`` (default ~/.hipfire/models). Not runnable on the dev box."""
    from .build import build_daemon                 # lazy: cargo/git, prod-only
    from .device import resolve_device
    from . import serve_probe
    from ..certify import cross_arch

    # Arch→box deferral (§4.1): a box only runs archs the diff actually affects; an
    # unaffected arch is a no-op PASS (no build, no GPU) so it can never spuriously
    # parity-fail. A docs/ar-only PR (base≡head daemon) defers on every arch.
    if arch not in affected_archs(files, cfg):
        return {"arch": arch, "verdict": "PASS", "reasons": ["deferred"], "bod": None,
                "tok_delta_pct": 0.0}

    # Gate 1 — cross-arch leak (file-based preprocessor invariance, no GPU): a changed
    # kernel must not alter ANOTHER arch's device codegen.
    kernel_files = [f for f in files if f.startswith("kernels/") and f.endswith(".hip")]
    leaks = [f for f in kernel_files
             if cross_arch.check_cross_arch(f, arch, cfg.other_archs(arch), repo, base_sha=base)]
    if leaks:
        return {"arch": arch, "verdict": "REJECT", "reasons": ["cross_arch"], "bod": None,
                "tok_delta_pct": 0.0, "detail": f"arch-bleed: {leaks}"}

    dev = resolve_device(arch, default=dev if dev is not None else 0)
    kv = getattr(cfg, "kv_mode", None) or "q8"
    models_dir = os.path.expanduser(os.environ.get("HIPFIRE_MODELS_DIR", "~/.hipfire/models"))

    # Build the two daemons — a head build failure = the PR doesn't compile at this ref
    # => REJECT (not a crash); a base build failure is an infra ERROR.
    try:
        base_bin = build_daemon(base, repo)
    except RuntimeError as e:
        return {"arch": arch, "verdict": "ERROR", "reasons": [f"base_build:{e}"], "bod": None}
    try:
        head_bin = build_daemon(head, repo)
    except RuntimeError as e:
        return {"arch": arch, "verdict": "REJECT", "reasons": ["build_fail"],
                "bod": None, "detail": str(e)}

    # Per fitting model: serve_harness greedy A/B, graded parity → coherence → perf. Each
    # cell is a self-describing ledger row, so a change that breaks 27b but not a3b keeps
    # a PASS row for a3b and a PARITY_FAIL row for 27b — both land in the itemized BOD.
    rows, deltas = [], []
    base_port = 11540 + dev * 40
    for i, m in enumerate(cfg.models_for(arch)):
        mp = os.path.join(models_dir, m)
        try:
            base_rows = serve_probe.run_serve_harness(base_bin, mp, dev, repo=repo, kv=kv,
                                                      port=base_port + i * 2)
            head_rows = serve_probe.run_serve_harness(head_bin, mp, dev, repo=repo, kv=kv,
                                                      port=base_port + i * 2 + 1)
        except RuntimeError as e:
            return {"arch": arch, "verdict": "ERROR", "reasons": [f"serve:{m}"],
                    "bod": None, "tok_delta_pct": 0.0, "detail": str(e)}
        cell = serve_probe.grade_cell(base_rows, head_rows, arch=arch, model=m, floor=cfg.floor)
        rows.append(cell)
        deltas.append(cell.get("tok_delta_pct") or 0.0)

    tok_delta = min(deltas) if deltas else 0.0
    fails = [r for r in rows if r["verdict"] != "PASS"]
    if fails:
        blockers = [serve_probe.cell_blocker(r) for r in fails]
        bod = {"blockers": blockers,
               "summary": f"{len(blockers)} cell(s) failed: " + ", ".join(b["detail"] for b in blockers)}
        return {"arch": arch, "verdict": "REJECT", "reasons": [b["kind"] for b in blockers],
                "bod": bod, "rows": rows, "tok_delta_pct": tok_delta}
    return {"arch": arch, "verdict": "PASS", "reasons": [], "bod": None, "rows": rows,
            "tok_delta_pct": tok_delta}

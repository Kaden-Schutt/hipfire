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

from .config import GateConfig
from .merge import default_run_git
from .outcome import decide_pr, format_pr_comment
from .routing import classify_pr

__all__ = ["changed_files", "diff_lines", "stub_arch_gate", "run_pr_gate",
           "live_arch_gate", "interpret_results"]


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


def interpret_results(*, results_dir, base, head, repo, author, is_draft, helpful,
                      cfg: GateConfig, run_git=None) -> dict:
    """Aggregate per-arch result JSONs (each an arch-result dict emitted by a matrix
    job) and decide the PR outcome. This is the ``interpret`` job's core: the matrix
    runs one arch each, this reduces them all to a single verdict + comment."""
    arch_results = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p) as fh:
            arch_results.append(json.load(fh))
    files = changed_files(base, head, repo, run_git=run_git)
    pr_class = classify_pr(files, lines_changed=diff_lines(base, head, repo, run_git=run_git))
    outcome = decide_pr(arch_results=arch_results, author=author, is_draft=is_draft,
                        helpful=helpful, cfg=cfg)
    return {"pr_class": pr_class, "route": cfg.route(pr_class), "arch_results": arch_results,
            "outcome": outcome, "comment": format_pr_comment(outcome, arch_results)}


def live_arch_gate(arch, files, base, head, repo, cfg, *, dev=0, card=0, model=None) -> dict:
    """REAL per-arch gate (GPU) — runs on a self-hosted runner only. Builds the
    base + head daemons, runs the Phase-1 run_gate over a LiveServeRunner, then the
    Phase-2 gate4 non-clobber merge. Imports the GPU adapter lazily so this module
    stays GPU-free at import time.

    NOTE: requires the runner environment (ROCm, cargo, a built daemon, the GPU) and
    the Phase-0 runners; it cannot execute on the zero-validation dev box. The daemon
    build + LiveServeRunner construction are wired here; the workflow (gpu-gates.yml)
    supplies dev/card and the built daemon paths per arch."""
    from .engine import run_gate                    # lazy: keep this module import-light
    from .merge import gate4, trial_merge
    from ..certify.serve_runner import LiveServeRunner

    kernel_files = [f for f in files if f.startswith("kernels/") and f.endswith(".hip")]
    sku = model or (cfg.canonical_models[0] if cfg.canonical_models else "qwen3.6-a3b")

    def factory(m):
        return LiveServeRunner(model=m, arch=arch, dev=dev, card=card, kv=cfg.kv_mode
                               if hasattr(cfg, "kv_mode") else "q8")

    models = cfg.models_for(arch)
    gate = run_gate(arch=arch, changed_kernel_files=kernel_files, models=models,
                    base_ref=base, head_ref=head, repo=repo, cfg=cfg,
                    runner_factory=factory)
    if gate["verdict"] == "REJECT":
        return {"arch": arch, "verdict": "REJECT", "reasons": gate["reasons"], "bod": None}

    # Non-clobber merge: gate the merged tree vs the staging tip (here: base).
    g4 = gate4(base_ref=base, head_ref=head, staging_ref=base, repo=repo,
               run_merged_gate=lambda: run_gate(
                   arch=arch, changed_kernel_files=kernel_files, models=models,
                   base_ref=base, head_ref=head, repo=repo, cfg=cfg, runner_factory=factory),
               merge_fix=None)
    if g4["verdict"] == "BOD":
        return {"arch": arch, "verdict": "BOD", "reasons": [], "bod": g4["bod"]}
    return {"arch": arch, "verdict": "PASS", "reasons": [], "bod": None}

# Copyright (c) Kaden Schutt
"""ar.cli — the role-scoped ``ar`` entrypoint over the autoresearch loop.

The ONE surface an agent (or operator) touches — ssh-in → ``ar``, never a raw
script. Ports the CLI surface of ``autoresearch/ar/hipfire_ar.py`` (``main`` +
``cmd_*``) onto the converged ``ar/`` package (``db`` / ``candidates`` /
``config`` / ``swarm`` / ``watcher`` / ``gitpilot``), and hardens the two
contracts into the tool itself:

**Roles** (from the supervisor design):

* ``operator`` (Claude) — ``start · stop · status · why · bod · ingest · fold ·
  rollover · config``.
* ``agent`` (codex/grok) — ``why · status · bod · certify`` — read state + submit
  candidates only.

An ``agent`` calling an operator-only verb is refused **mechanically** (exit 3),
not by prompt-discipline.

**Certify bounds** (the leash). ``ar certify`` is the bounds gate — it refuses
(exit 3) a submission on a kernel that is:

* **EXHAUSTED** — ``k`` consecutive DEAD/… with no WIN (see :func:`ar.db.kernel_stats`);
* **off-target** — not a BOD census candidate (``wall% < cand_wall``);
* **over-budget / TTL-expired** — the arch's running loop has spent its call
  budget or wall-TTL.

A refusal is not merely discouraged; the agent literally cannot re-burn a dead
kernel. When bounds pass, ``certify`` accepts (exit 0) — the actual A/B grade is
run by the driver loop's :class:`~autoresearch.ar.certify.orchestrator.certify`
(over the ``LiveServeRunner`` GPU adapter); the row lands in the ledger, then
``ar ingest`` re-indexes it.

Config resolves the store: ``--db`` / ``$AR_DB`` (default
``<repo>/autoresearch/db/ar.db``), ``$AR_LEDGER`` (default
``<repo>/autoresearch/ledger``), ``$AR_BOD_GLOB`` (default
``<repo>/autoresearch/state/bod_*.json``). Nothing hardcodes arch/model/card.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time

from .config import load_config
from .db import connect, ingest, kernel_stats

# Defaults (match hipfire_ar / driver_v3 / config §6): a WIN resets the streak,
# k consecutive deads ⇒ EXHAUSTED; wall% >= cand_wall to be a candidate.
K_DEFAULT = 5
CAND_WALL = 3.0

# Role scoping. Agent may ONLY read state + submit candidates.
AGENT_VERBS = frozenset({"why", "status", "bod", "certify"})
OPERATOR_VERBS = frozenset(
    {"start", "stop", "status", "why", "bod", "ingest", "fold", "rollover", "config", "certify", "gate"}
)


# ── path resolution (all overridable via env / flags; nothing hardcoded) ──────


def _repo() -> str:
    """Repo root: ``$AR_REPO`` or three levels up from this file (…/autoresearch/ar/cli.py)."""
    return os.environ.get("AR_REPO") or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


def _db_path(a) -> str:
    """Store path: ``--db`` › ``$AR_DB`` › ``<repo>/autoresearch/db/ar.db``."""
    return getattr(a, "db", None) or os.environ.get("AR_DB") or os.path.join(
        _repo(), "autoresearch", "db", "ar.db"
    )


def _ledger_dir() -> str:
    return os.environ.get("AR_LEDGER") or os.path.join(_repo(), "autoresearch", "ledger")


def _bod_glob() -> str:
    return os.environ.get("AR_BOD_GLOB") or os.path.join(_repo(), "autoresearch", "state", "bod_*.json")


def _connect(a):
    return connect(_db_path(a))


def _arches(conn) -> str:
    rows = conn.execute("SELECT DISTINCT arch FROM bod").fetchall()
    return ",".join(r[0] for r in rows if r[0]) or "-"


def _bound_class(l2, mem) -> str:
    """Roofline lens for the operator's eyeball (traffic vs occ lever)."""
    if l2 is None:
        return "?"
    if l2 < 40:
        return "DRAM-thrash"  # traffic-reduction lever
    if l2 > 60:
        return "cache-resident"  # ALU/occ lever
    return "mixed"


# ── read verbs (agent + operator) ─────────────────────────────────────────────


def cmd_bod(a) -> int:
    """Ranked candidate kernels + coverage + where levers remain (codex/Claude review)."""
    conn = _connect(a)
    rows = conn.execute("SELECT * FROM bod WHERE arch=? ORDER BY wall_pct DESC", (a.arch,)).fetchall()
    if not rows:
        print(json.dumps({"arch": a.arch, "kernels": [], "note": "no BOD — run `ar ingest` after a census"}))
        return 2
    out = {"arch": a.arch, "kernels": []}
    for b in rows:
        st = kernel_stats(conn, a.arch, b["kernel"], a.k)
        cand = (b["wall_pct"] or 0) >= CAND_WALL
        out["kernels"].append(
            {
                "kernel": b["kernel"],
                "wall_pct": round(b["wall_pct"] or 0, 1),
                "l2_hit": round(b["l2_hit"] or 0, 1),
                "mem_busy": round(b["mem_busy"] or 0, 1),
                "bound": _bound_class(b["l2_hit"], b["mem_busy"]),
                "tried": st["tried"],
                "wins": st["wins"],
                "dead_streak": st["consecutive_dead"],
                "best_win_pct": st["best_win_pct"],
                "status": ("EXHAUSTED" if st["exhausted"] else "OPEN" if cand else "below-threshold"),
            }
        )
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"# BOD review — {a.arch}  (candidate wall%>={CAND_WALL}, exhausted at {a.k} consecutive deads)")
    print(f"{'kernel':44} {'wall%':>6} {'L2%':>5} {'bound':>14} {'tried':>5} {'win':>3} {'best%':>6} status")
    for k in out["kernels"]:
        print(
            f"{k['kernel'][:44]:44} {k['wall_pct']:6.1f} {k['l2_hit']:5.1f} {k['bound']:>14} "
            f"{k['tried']:5d} {k['wins']:3d} {(k['best_win_pct'] or 0):6.2f} {k['status']}"
        )
    openk = sum(1 for k in out["kernels"] if k["status"] == "OPEN")
    exhk = sum(1 for k in out["kernels"] if k["status"] == "EXHAUSTED")
    print(f"\n{openk} OPEN candidate kernels with lever room; {exhk} exhausted.")
    return 0


def cmd_why(a) -> int:
    """What was tried on a kernel + verdicts + why-dead (the tried/failed surface)."""
    conn = _connect(a)
    st = kernel_stats(conn, a.arch, a.kernel, a.k)
    rows = conn.execute(
        "SELECT lever, verdict, tok_delta, dur_delta, profile FROM attempts "
        "WHERE arch=? AND kernel=? ORDER BY ts, id",
        (a.arch, a.kernel),
    ).fetchall()
    if a.json:
        print(
            json.dumps(
                {
                    "arch": a.arch,
                    "kernel": a.kernel,
                    "exhausted": st["exhausted"],
                    "dead_streak": st["consecutive_dead"],
                    "tried": st["tried"],
                    "wins": st["wins"],
                    "attempts": [dict(r) for r in rows],
                },
                indent=2,
            )
        )
        return 0
    print(
        f"# {a.kernel} on {a.arch}  ({st['tried']} attempts, {st['wins']} wins, "
        f"dead-streak {st['consecutive_dead']}/{a.k}{' — EXHAUSTED' if st['exhausted'] else ''})"
    )
    for r in rows:
        mark = "WIN " if r["verdict"] == "WIN" else "    "
        print(f"  {mark}{r['verdict']:14} tok{(r['tok_delta'] or 0):+6.2f}% dur{(r['dur_delta'] or 0):+6.2f}%  {r['lever']}")
    if st["exhausted"]:
        print("  ==> EXHAUSTED: certify will REJECT new levers on this kernel (bounds enforced).")
    return 0


def cmd_status(a) -> int:
    """Run health: banked attempts + wins + running loops (budget/TTL left)."""
    conn = _connect(a)
    runs = conn.execute("SELECT * FROM runs WHERE status='running'").fetchall()
    tot = conn.execute("SELECT COUNT(*) n, SUM(verdict='WIN') w FROM attempts").fetchone()
    now = int(time.time())
    summary = {
        "attempts": tot["n"] or 0,
        "wins": tot["w"] or 0,
        "arches": _arches(conn),
        "running": [
            {
                "run": r["id"],
                "arch": r["arch"],
                "card": r["card"],
                "pid": r["pid"],
                "calls": r["calls"],
                "budget": r["budget"],
                "ttl_left_s": ((r["ts"] + r["ttl"] - now) if r["ttl"] else None),
            }
            for r in runs
        ],
    }
    if a.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"# ar status  ({summary['attempts']} attempts banked, {summary['wins']} wins, archs: {summary['arches']})")
    if not summary["running"]:
        print("  no running loops.")
        return 0
    for r in summary["running"]:
        print(
            f"  run {r['run']} arch={r['arch']} card={r['card']} pid={r['pid']} "
            f"calls={r['calls']}/{r['budget']} ttl_left={r['ttl_left_s']}s"
        )
    return 0


# ── certify (the mechanical leash — agent + operator) ─────────────────────────


def cmd_certify(a) -> int:
    """Bounds gate. Refuse (exit 3) EXHAUSTED / off-target / over-budget; else accept (0).

    The leash lives HERE so an agent cannot re-burn a dead kernel. On accept the
    actual A/B grade is run by the driver loop's certify orchestrator (over the
    ``LiveServeRunner`` GPU adapter); the row lands in the ledger, then
    ``ar ingest``.
    """
    conn = _connect(a)
    st = kernel_stats(conn, a.arch, a.kernel, a.k)
    if st["exhausted"]:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "KERNEL_EXHAUSTED",
                    "detail": f"{a.kernel} hit {a.k} consecutive deads — try an OPEN kernel (see: ar bod)",
                }
            )
        )
        return 3
    bod = conn.execute(
        "SELECT wall_pct FROM bod WHERE arch=? AND kernel=?", (a.arch, a.kernel)
    ).fetchone()
    if not bod or (bod["wall_pct"] or 0) < a.cand_wall:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "OFF_TARGET",
                    "detail": f"{a.kernel} is not a census candidate (wall%<{a.cand_wall})",
                }
            )
        )
        return 3
    run = conn.execute(
        "SELECT * FROM runs WHERE arch=? AND status='running' ORDER BY ts DESC", (a.arch,)
    ).fetchone()
    if run:
        if run["budget"] and (run["calls"] or 0) >= run["budget"]:
            print(
                json.dumps(
                    {
                        "accepted": False,
                        "reason": "BUDGET_SPENT",
                        "detail": f"run {run['id']} used {run['calls']}/{run['budget']} agent-calls",
                    }
                )
            )
            return 3
        if run["ttl"] and time.time() > (run["ts"] or 0) + run["ttl"]:
            print(json.dumps({"accepted": False, "reason": "TTL_EXPIRED", "detail": f"run {run['id']} past wall-TTL"}))
            return 3
        conn.execute("UPDATE runs SET calls=calls+1 WHERE id=?", (run["id"],))
        conn.commit()
    print(
        json.dumps(
            {
                "accepted": True,
                "run": run["id"] if run else None,
                "kernel": a.kernel,
                "lever": a.lever,
                "variant": a.variant,
                "note": "bounds OK — grade via the driver certify orchestrator; row lands in ledger, then `ar ingest`.",
            }
        )
    )
    return 0


# ── operator verbs ────────────────────────────────────────────────────────────


def cmd_ingest(a) -> int:
    """Re-index the ledger + refresh the BOD snapshots into ``ar.db`` (idempotent)."""
    conn = _connect(a)
    ledger = a.ledger or _ledger_dir()
    bod = a.bod or _bod_glob()
    n = ingest(conn, ledger, bod)
    print(json.dumps({"ingested": n, "ledger": ledger, "bod_glob": bod, "arches": _arches(conn)}))
    return 0


def cmd_config(a) -> int:
    """Print the resolved per-arch loop config (the TOML → :class:`LoopConfig`)."""
    path = a.config_path
    if not path and a.arch:
        path = os.path.join(_repo(), "autoresearch", "config", f"loop_{a.arch}.toml")
    if not path:
        print(json.dumps({"error": "pass --config <path> or --arch <arch>"}))
        return 2
    cfg = load_config(path)
    d = dataclasses.asdict(cfg)
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        print(json.dumps(d))
    return 0


def cmd_gate(a) -> int:
    """Resolve (--plan) or execute (--run) the Tier-3 PR gate.

    --plan: print an arch's resolved gate plan (fitting models / cross-arch targets
    / thresholds), no GPU. --run: execute the PR gate (classify the diff -> route ->
    per-arch gate -> decide). --run --dry-run uses a STUB per-arch gate (no GPU / no
    daemon build) to demonstrate the classify->route->decide->comment pipeline on a
    real diff; --run without --dry-run uses the live GPU adapter (self-hosted runner
    only). Reads autoresearch/config/pr_gate.toml."""
    from .gate.config import load_gate_config

    path = a.gate_config or os.path.join(_repo(), "autoresearch", "config", "pr_gate.toml")
    cfg = load_gate_config(path)
    repo = _repo()

    if getattr(a, "validate_plan", None):
        # Fail-closed release gate for the self-hosted matrix. Claude writes the
        # plan; deterministic path/arch/model floors decide whether any runner
        # may start and emit the exact dynamic matrix when valid.
        from .gate import dispatch
        from .gate.run import changed_files, diff_lines

        try:
            with open(a.validate_plan) as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(json.dumps({"valid": False, "errors": [f"plan unreadable: {e}"],
                              "matrix": {"include": []}}, indent=2))
            return 2
        out = dispatch.validate_dispatch_plan(
            raw, changed_files(a.base, a.head, repo), cfg,
            lines_changed=diff_lines(a.base, a.head, repo),
        )
        print(json.dumps(out, indent=2))
        return 0 if out["valid"] else 2

    if getattr(a, "validate_action", None):
        # Claude interprets the deterministic result into a proposed action; this
        # check prevents a failed gate from being narrated into a green action.
        from .gate import dispatch

        if not a.interp_json:
            print(json.dumps({"valid": False, "errors": ["--interp-json is required"]},
                             indent=2))
            return 2
        try:
            with open(a.interp_json) as fh:
                interp = json.load(fh)
            with open(a.validate_action) as fh:
                action = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(json.dumps({"valid": False, "errors": [f"action input unreadable: {e}"]},
                             indent=2))
            return 2
        out = dispatch.validate_interpret_action(interp, action)
        print(json.dumps(out, indent=2))
        return 0 if out["valid"] else 2

    if getattr(a, "verify_evidence", None):
        from .gate import dispatch

        try:
            with open(a.verify_evidence) as fh:
                plan = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(json.dumps({"valid": False, "errors": [f"dispatch unreadable: {e}"],
                              "behavior_results": []}, indent=2))
            return 2
        out = dispatch.verify_evidence(
            plan, results_dir=a.results, behavior_dir=a.behavior_dir,
            pr=a.pr, base_sha=a.base, head_sha=a.head,
        )
        print(json.dumps(out, indent=2))
        return 0 if out["valid"] else 2

    if getattr(a, "sweep", False):
        # Backlog sweep (spec §11.1): gate all open eligible PRs, punt regressions,
        # stack the approved onto the collection branch. LIVE (GPU / codex / git).
        import subprocess as _sp

        from .gate.staging_prod import sweep_backlog

        prs = [p for p in (a.prs.split(",") if a.prs else []) if p]
        if not prs:
            r = _sp.run(
                ["gh", "pr", "list", "--state", "open", "--limit", "80",
                 "--json", "number,isDraft,baseRefName",
                 "--jq", '.[] | select(.isDraft==false and .baseRefName=="master") | .number'],
                capture_output=True, text=True, cwd=repo,
            )
            prs = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        res = sweep_backlog(open_prs=prs, collection_ref=(a.collection or "feat/rdna-kernel-oracle"),
                            repo=repo, arch=(a.arch or "gfx1201"), dev=a.dev, card=a.card)
        print(json.dumps(res, indent=2))
        return 0

    if getattr(a, "behavior_tests", None):
        # Run the bespoke codex behavior tests from Claude's plan.json (spec §8) —
        # tests behaviors serve_harness can't reach. Emits the behavior_results JSON.
        from .gate.run import run_behavior_plan

        out = run_behavior_plan(
            a.behavior_tests, repo=repo, verdict_dir=(a.verdict_dir or "/tmp/gate-behavior"),
            base=a.base, head=a.head,
        )
        print(json.dumps(out["behavior_results"], indent=2))
        return 1 if any(not b["passed"] for b in out["behavior_results"]) else 0

    if getattr(a, "interpret", False):
        # The interpret job: aggregate the matrix's per-arch result JSONs + any
        # behavior-test results -> decide.
        from .gate.run import interpret_results

        beh = None
        if getattr(a, "behavior_results", None):
            with open(a.behavior_results) as fh:
                beh = json.load(fh)
        res = interpret_results(
            results_dir=a.results, base=a.base, head=a.head, repo=repo,
            author=(a.author or ""), is_draft=a.draft, helpful=(not a.not_helpful), cfg=cfg,
            behavior_results=beh,
        )
        payload = {"pr": a.pr, "pr_class": res["pr_class"], "executor": res["route"],
                   "outcome": res["outcome"], "comment": res["comment"]}
        print(json.dumps(payload, indent=2))
        if not a.json:
            print("\n--- PR comment ---\n" + res["comment"])
        return 1 if res["outcome"]["status"] == "failure" else 0

    if getattr(a, "collect", False):
        # MECHANICAL eval tool for the agentic gate: build base+head daemons + serve_harness
        # base-vs-head per model -> RAW rows (errors as data). codex drives this, then judges.
        from .gate.run import changed_files, collect_cell_data

        files = changed_files(a.base, a.head, repo)
        # --models = Claude's SKU selection for this change (codex passes it); None => canonical.
        models = [m for m in (a.models.split(",") if a.models else []) if m] or None
        print(json.dumps(collect_cell_data(a.arch, files, a.base, a.head, repo, cfg,
                                            dev=a.dev, models=models), indent=2))
        return 0

    if getattr(a, "grade", None):
        # Deterministic baseline grade of a collect JSON -> arch result (ledger rows + BOD).
        # codex runs this, reviews it, re-collects any EMPTY cell, then emits the final result.
        from .gate.run import grade_collected

        with open(a.grade) as fh:
            res = grade_collected(json.load(fh), cfg=cfg)
        print(json.dumps(res, indent=2))
        return 1 if res["verdict"] in ("REJECT", "BOD") else 0

    if getattr(a, "run", False):
        from .gate.run import changed_files, live_arch_gate, run_pr_gate, stub_arch_gate

        gate_fn = stub_arch_gate if a.dry_run else (
            lambda arch, files, base, head, r, c:
            live_arch_gate(arch, files, base, head, r, c, dev=a.dev, card=a.card))

        # Matrix per-arch mode (live, one arch): emit just this arch's result; the
        # interpret job aggregates all archs and decides.
        if a.arch and not a.dry_run:
            files = changed_files(a.base, a.head, repo)
            ar = gate_fn(a.arch, files, a.base, a.head, repo, cfg)
            print(json.dumps(ar, indent=2))
            return 1 if ar["verdict"] in ("REJECT", "BOD") else 0

        # Full pipeline (dry-run demo, or a single-box all-arch run).
        res = run_pr_gate(
            base=a.base, head=a.head, repo=repo, author=(a.author or ""),
            is_draft=a.draft, helpful=(not a.not_helpful), cfg=cfg,
            arch_gate_fn=gate_fn, archs=([a.arch] if a.arch else None),
        )
        out = {
            "mode": "dry-run" if a.dry_run else "live", "pr": a.pr,
            "base": a.base, "head": a.head, "author": a.author,
            "pr_class": res["pr_class"], "executor": res["route"],
            "arch_results": res["arch_results"], "outcome": res["outcome"],
        }
        print(json.dumps(out, indent=2))
        print("\n--- PR comment ---\n" + res["comment"])
        # CI exit: a failure verdict (any arch REJECT/BOD) reds the check.
        return 1 if res["outcome"]["status"] == "failure" else 0

    # --plan (default): per-arch resolved plan.
    if not a.arch:
        print(json.dumps({"error": "pass --arch for --plan, or --run for a PR-level gate"}))
        return 2
    extra = tuple(m for m in (a.models.split(",") if a.models else []) if m)
    out = {
        "arch": a.arch,
        "models": cfg.models_for(a.arch, extra=extra),
        "other_archs": cfg.other_archs(a.arch),
        "floor": cfg.floor,
        "alpha": cfg.alpha,
        "drift_pct": cfg.drift_pct,
    }
    print(json.dumps(out, indent=2) if a.json else json.dumps(out))
    return 0


def cmd_start(a) -> int:
    """operator-only. Launch the config's workers (swarm) as detached loops.

    Reuses the Phase-5 :func:`ar.swarm.launch`; a worker without an advancing
    ``.aw/sw_card<card>`` worktree is SKIPPED (bash parity), so a bare checkout
    launches nothing rather than blindly forking. Records each launched worker as
    a bounded run in ``ar.db``."""
    from . import swarm  # lazy: keeps the CLI import light + fork machinery unloaded until needed
    from .watcher import RunStore, Watcher

    cfg = load_config(a.config)
    repo = _repo()
    pids = swarm.launch(cfg, repo)
    conn = _connect(a)
    w = Watcher(RunStore(conn), repo)
    now = int(time.time())
    for i, pid in enumerate(pids):
        wk = cfg.workers[i] if i < len(cfg.workers) else None
        w.track(
            {
                "id": f"{cfg.arch}-w{i}-{now}",
                "arch": cfg.arch,
                "model": cfg.model,
                "card": wk.card if wk else None,
                "status": "running",
                "budget": cfg.bounds.call_budget,
                "calls": 0,
                "ttl": cfg.bounds.wall_ttl_s,
                "pid": pid,
                "ts": now,
            }
        )
    print(json.dumps({"arch": cfg.arch, "launched": len(pids), "pids": pids}))
    return 0


def cmd_stop(a) -> int:
    """operator-only. Flip running run rows to ``stopped`` (optionally one ``--run``).

    Signalling the remote loop tree is a fleet-side concern; here we own the ``ar.db``
    run-table flag so ``ar status`` reflects the stop."""
    conn = _connect(a)
    if a.run:
        conn.execute("UPDATE runs SET status='stopped' WHERE id=? AND status='running'", (a.run,))
    else:
        conn.execute("UPDATE runs SET status='stopped' WHERE status='running'")
    n = conn.total_changes
    conn.commit()
    print(json.dumps({"stopped": n, "run": a.run}))
    return 0


def cmd_fold(a) -> int:
    """operator-only. Fold a certified WIN into the shared baseline via ``update-ref`` CAS.

    Guardrailed by :meth:`Watcher.enforce_fold`: the prior SHA is recorded
    (reversible) and the advance is a compare-and-swap. ``--dry-run`` previews
    without mutating."""
    from .watcher import RunStore, Watcher

    conn = _connect(a)
    w = Watcher(RunStore(conn), _repo())
    entry = w.enforce_fold(a.ref, a.sha, dry_run=a.dry_run)
    print(json.dumps(entry))
    return 0


def cmd_rollover(a) -> int:
    """operator-only. Re-census + re-ingest after a baseline advance / exhaustion.

    Guardrailed by :meth:`Watcher.enforce_rollover`. The GPU census is the
    injected ``recensus`` seam (absent here — a bare CLI rollover records the
    intent); ``--dry-run`` previews only."""
    from .watcher import RunStore, Watcher

    conn = _connect(a)
    w = Watcher(RunStore(conn), _repo())
    entry = w.enforce_rollover(reason=a.reason, dry_run=a.dry_run)
    print(json.dumps(entry))
    return 0


# ── argparse + dispatch ───────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ar", description="autoresearch loop — role-scoped control")
    p.add_argument("--role", choices=["operator", "agent"], default="operator")
    p.add_argument("--db", default=None, help="ar.db path (overrides $AR_DB)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="re-index the ledger + BOD into ar.db (idempotent)")
    s.add_argument("--ledger", default=None)
    s.add_argument("--bod", default=None)

    s = sub.add_parser("bod", help="ranked candidate kernels + coverage")
    s.add_argument("--arch", required=True)
    s.add_argument("--json", action="store_true")
    s.add_argument("--k", type=int, default=K_DEFAULT)

    s = sub.add_parser("why", help="what was tried on a kernel + verdicts")
    s.add_argument("kernel")
    s.add_argument("--arch", required=True)
    s.add_argument("--json", action="store_true")
    s.add_argument("--k", type=int, default=K_DEFAULT)

    s = sub.add_parser("status", help="run health + banked attempts/wins")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("certify", help="submit a candidate (bounds mechanically enforced)")
    s.add_argument("--arch", required=True)
    s.add_argument("--kernel", required=True)
    s.add_argument("--lever", default="?")
    s.add_argument("--variant", default=None)
    s.add_argument("--k", type=int, default=K_DEFAULT)
    s.add_argument("--cand-wall", type=float, default=CAND_WALL, dest="cand_wall")

    s = sub.add_parser("start", help="operator: launch the config's workers")
    s.add_argument("--config", required=True)

    s = sub.add_parser("stop", help="operator: stop running loops")
    s.add_argument("--run", default=None)

    s = sub.add_parser("fold", help="operator: fold a WIN into the shared baseline (CAS)")
    s.add_argument("--ref", required=True)
    s.add_argument("--sha", required=True)
    s.add_argument("--dry-run", action="store_true", dest="dry_run")

    s = sub.add_parser("rollover", help="operator: re-census + re-ingest")
    s.add_argument("--reason", default="advance")
    s.add_argument("--dry-run", action="store_true", dest="dry_run")

    s = sub.add_parser("config", help="operator: print the resolved loop config")
    s.add_argument("--config", dest="config_path", default=None)
    s.add_argument("--arch", default=None)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("gate", help="operator: resolve (--plan) or execute (--run) the Tier-3 PR gate")
    s.add_argument("--arch", default=None, help="single arch (for --plan); omit for a --run over all archs")
    s.add_argument("--plan", action="store_true", help="print the resolved gate plan (no GPU)")
    s.add_argument("--run", action="store_true", help="execute the PR gate (classify -> per-arch -> decide)")
    s.add_argument("--collect", action="store_true",
                   help="MECHANICAL tool: build base+head daemons + serve_harness A/B -> raw rows "
                        "(errors as data). codex drives this in the agentic gate, then judges/adapts")
    s.add_argument("--grade", default=None,
                   help="deterministic baseline grade of a --collect JSON file -> arch result (rows + BOD)")
    s.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="run mode with a STUB per-arch gate (no GPU / no daemon build)")
    s.add_argument("--interpret", action="store_true",
                   help="aggregate per-arch result JSONs in --results and decide the PR outcome")
    s.add_argument("--results", default=None, help="dir of per-arch result JSONs (for --interpret)")
    s.add_argument("--behavior-tests", dest="behavior_tests", default=None,
                   help="Claude plan.json — run its bespoke codex behavior tests (spec §8)")
    s.add_argument("--behavior-results", dest="behavior_results", default=None,
                   help="behavior_results JSON to fold into --interpret")
    s.add_argument("--validate-plan", dest="validate_plan", default=None,
                   help="validate Claude dispatch JSON and emit the dynamic runner matrix")
    s.add_argument("--validate-action", dest="validate_action", default=None,
                   help="validate Claude action JSON against --interp-json")
    s.add_argument("--interp-json", dest="interp_json", default=None,
                   help="pure deterministic interpretation JSON for --validate-action")
    s.add_argument("--verify-evidence", dest="verify_evidence", default=None,
                   help="verify every selected runner/behavior artifact in a validated dispatch")
    s.add_argument("--behavior-dir", dest="behavior_dir", default=None,
                   help="downloaded behavior artifact directory for --verify-evidence")
    s.add_argument("--verdict-dir", dest="verdict_dir", default=None,
                   help="dir for codex behavior-test verdict files (--behavior-tests)")
    s.add_argument("--sweep", action="store_true",
                   help="backlog sweep: gate all open PRs, stack the approved onto --collection (spec §11.1)")
    s.add_argument("--collection", default=None,
                   help="collection branch for --sweep (default feat/rdna-kernel-oracle)")
    s.add_argument("--prs", default=None,
                   help="comma PR numbers for --sweep (default: all open non-draft base=master)")
    s.add_argument("--base", default="origin/master", help="base ref the PR merges into")
    s.add_argument("--head", default="HEAD", help="head ref (the PR)")
    s.add_argument("--pr", default=None, help="PR number (label only)")
    s.add_argument("--author", default=None, help="PR author login (authority)")
    s.add_argument("--draft", action="store_true", help="PR is a draft (report only, no merge)")
    s.add_argument("--not-helpful", action="store_true", dest="not_helpful",
                   help="stub Claude's helpfulness judgment as false")
    s.add_argument("--dev", type=int, default=0, help="HIP device (live mode)")
    s.add_argument("--card", type=int, default=0, help="DRM card (live mode)")
    s.add_argument("--models", default=None, help="comma-separated change-specific SKUs to add")
    s.add_argument("--gate-config", dest="gate_config", default=None)
    s.add_argument("--json", action="store_true")

    return p


_DISPATCH = {
    "ingest": cmd_ingest,
    "bod": cmd_bod,
    "why": cmd_why,
    "status": cmd_status,
    "certify": cmd_certify,
    "start": cmd_start,
    "stop": cmd_stop,
    "fold": cmd_fold,
    "rollover": cmd_rollover,
    "config": cmd_config,
    "gate": cmd_gate,
}


def main(argv=None) -> int:
    """The ``ar`` entrypoint. Returns an exit code (3 = mechanical refusal)."""
    a = build_parser().parse_args(argv)
    # Role scoping: an agent may ONLY read state + submit candidates. Refuse
    # operator-only verbs mechanically (exit 3), before any dispatch/side-effect.
    if a.role == "agent" and a.cmd not in AGENT_VERBS:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "reason": "ROLE_FORBIDDEN",
                    "role": a.role,
                    "verb": a.cmd,
                    "allowed": sorted(AGENT_VERBS),
                    "detail": f"'{a.cmd}' is operator-only; agents may call {sorted(AGENT_VERBS)}",
                }
            )
        )
        return 3
    return _DISPATCH[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())

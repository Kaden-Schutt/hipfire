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
    """Resolve + print the Tier-3 gate plan for an arch (the fitting models, the
    cross-arch targets, the thresholds). No GPU here — the GPU run is the Phase-3
    workflow. Reads autoresearch/config/pr_gate.toml."""
    from .gate.config import load_gate_config

    path = a.gate_config or os.path.join(_repo(), "autoresearch", "config", "pr_gate.toml")
    cfg = load_gate_config(path)
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

    s = sub.add_parser("gate", help="operator: resolve/run the Tier-3 PR gate for an arch")
    s.add_argument("--arch", required=True)
    s.add_argument("--plan", action="store_true", help="print the resolved gate plan (no GPU)")
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

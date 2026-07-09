#!/usr/bin/env python3
"""hipfire-ar — autoresearch supervisor: a thin, reliable, role-scoped control/persistence
layer over the proven driver_v3 pipeline (oracle_profile census + check/update/gen_digest
exhaustion trio + rollover fold). Replaces the fragile fire_*.sh + ka.sh keep-alive.

Design: autoresearch/AR_SUPERVISOR_DESIGN.md. Non-ephemeral SQLite store (not /tmp).

Roles:
  operator (Claude): start / stop / status / why / ingest / bod
  agent (codex):     why / status / bod / certify   (bounds MECHANICALLY enforced)

Runaway prevention (defense in depth):
  1. driver_v3 self-terminates on global per-kernel exhaustion (K dead/kernel). No keep-alive.
  2. every run has a codex-call budget + wall-TTL; supervisor auto-stops past either.
  3. every `certify` is bounds-checked HERE: exhausted-kernel / over-budget / off-target
     submissions are REFUSED (exit 3), not merely discouraged. Codex can't go crazy.
"""
import argparse, json, os, sqlite3, sys, time, glob, subprocess

REPO = os.environ.get("AR_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(REPO, "autoresearch", "state")
LEDGER = os.path.join(REPO, "autoresearch", "ledger")
DB = os.path.join(STATE, "ar.db")
K_DEFAULT = 5           # consecutive DEAD/INCONCLUSIVE per kernel => EXHAUSTED (matches driver_v3)
CAND_WALL = 3.0         # BOD wall_pct threshold to be a candidate kernel (matches driver_v3)
DEAD_VERDICTS = {"DEAD", "INCONCLUSIVE", "LOSS", "NOISE"}

# --- Remote process control (the loop runs on a GPU box; this tool drives it over ssh) ------------
# Host is config, NOT hardcoded (contributor-generic): pass --host or set AR_SSH_HOST. Remote paths
# are the driver's deployed location + the repo checkout on the box.
SSH_HOST = os.environ.get("AR_SSH_HOST")            # None => remote ops require --host
REMOTE_REPO = os.environ.get("AR_REMOTE_REPO", "~/hipfire")
# PERSISTENT box-side harness (NOT /tmp — survives reboot). Deployed copies of autoresearch/harness/.
REMOTE_DRIVER = os.environ.get("AR_REMOTE_DRIVER", "~/hipfire/autoresearch/harness/v2/driver_v3.sh")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=5"]

def _ssh(cmd, host):
    """Run a shell command on the GPU box; returns (rc, stdout, stderr). Liveness/launch/kill only."""
    try:
        r = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return 255, "", str(e)

def _alive(pid, host):
    if not pid: return "no-pid"
    _, out, _ = _ssh(f"kill -0 {pid} 2>/dev/null && echo LIVE || echo DEAD", host)
    return out or "unreachable"

def db():
    os.makedirs(STATE, exist_ok=True)
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS attempts(
      id INTEGER PRIMARY KEY, run_id TEXT, arch TEXT, kernel TEXT, lever TEXT,
      verdict TEXT, delta_pct REAL, f REAL, profile TEXT, ts INTEGER);
    CREATE TABLE IF NOT EXISTS bod(
      arch TEXT, kernel TEXT, wall_pct REAL, l2_hit REAL, mem_busy REAL,
      occ REAL, vgpr INTEGER, snap_ts INTEGER);
    CREATE TABLE IF NOT EXISTS runs(
      id TEXT PRIMARY KEY, arch TEXT, model TEXT, card INTEGER, status TEXT,
      started INTEGER, stopped INTEGER, budget INTEGER, calls INTEGER, ttl INTEGER, pid INTEGER);
    CREATE INDEX IF NOT EXISTS ix_att ON attempts(arch, kernel, lever);
    """)
    return c

def ingest(a):
    """Pull the jsonl ledger + BOD snapshots into SQLite (idempotent). With --host, first rsync the
    REMOTE box ledger + bod snapshots down (the loop runs there; the DB is local) so the DB isn't blind."""
    host = getattr(a, "host", None) or SSH_HOST
    if host:
        os.makedirs(LEDGER, exist_ok=True)
        # pull ONLY the ledger (the durable verdict record written on the box); bod snapshots are
        # already local+committed, so we don't rsync state/ (would clobber committed bod/queues).
        try:
            subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS),
                            f"{host}:{REMOTE_REPO}/autoresearch/ledger/", LEDGER + "/"],
                           capture_output=True, text=True, timeout=120)
        except Exception as e:
            print(json.dumps({"warn": f"rsync ledger from {host} failed: {e}"}))
    c = db(); n = 0
    for f in glob.glob(os.path.join(LEDGER, "swarm_*.jsonl")):
        base = os.path.basename(f)[len("swarm_"):-len(".jsonl")]
        arch = base.split("_", 1)[0]
        kern = base.split("_", 1)[1] if "_" in base else base
        for line in open(f, errors="ignore"):
            try: d = json.loads(line)
            except Exception: continue
            lever = d.get("label") or d.get("lever") or "?"
            ts = int(d.get("ts") or 0)
            if c.execute("SELECT 1 FROM attempts WHERE arch=? AND kernel=? AND lever=? AND verdict=?",
                         (arch, d.get("kernel", kern), lever, d.get("verdict"))).fetchone(): continue
            c.execute("INSERT INTO attempts(run_id,arch,kernel,lever,verdict,delta_pct,f,profile,ts) VALUES(?,?,?,?,?,?,?,?,?)",
                      (d.get("run_id"), arch, d.get("kernel", kern), lever, d.get("verdict"),
                       _num(d.get("delta", d.get("delta_pct"))), _num(d.get("f")),
                       d.get("profile_feedback", ""), ts)); n += 1
    for f in glob.glob(os.path.join(STATE, "bod_*.json")):
        arch = os.path.basename(f)[len("bod_"):-len(".json")]
        try: rows = json.load(open(f)).get("rows", [])
        except Exception: continue
        c.execute("DELETE FROM bod WHERE arch=?", (arch,))
        for r in rows:
            c.execute("INSERT INTO bod VALUES(?,?,?,?,?,?,?,?)", (arch, r.get("kernel"),
                      _num(r.get("wall_pct")), _num(r.get("l2_hit_pct", r.get("l2_hit"))),
                      _num(r.get("mem_busy")), _num(r.get("occ")), r.get("vgpr"), int(time.time())))
    c.commit(); print(f"ingested {n} new attempts + BOD snapshots for {_arches(c)}")

def _num(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def _arches(c): return ",".join(r[0] for r in c.execute("SELECT DISTINCT arch FROM bod").fetchall()) or "-"

def _kernel_stats(c, arch, kernel, K):
    rows = c.execute("SELECT lever,verdict,delta_pct FROM attempts WHERE arch=? AND kernel=? ORDER BY id",
                     (arch, kernel)).fetchall()
    wins = [r for r in rows if r["verdict"] == "WIN"]
    # consecutive-dead streak from the tail (matches update_exhaustion: WIN resets)
    streak = 0
    for r in reversed(rows):
        if r["verdict"] == "WIN": break
        if r["verdict"] in DEAD_VERDICTS: streak += 1
    return {"tried": len(rows), "wins": len(wins), "dead_streak": streak,
            "exhausted": streak >= K and not wins, "rows": rows,
            "best": max((r["delta_pct"] or -9 for r in wins), default=None)}

def cmd_bod(a):
    """Claude/codex-shaped profile review: ranked candidate kernels + coverage + where levers remain."""
    c = db(); arch = a.arch
    rows = c.execute("SELECT * FROM bod WHERE arch=? ORDER BY wall_pct DESC", (arch,)).fetchall()
    if not rows: print(f"no BOD for {arch}. run: hipfire_ar.py ingest (after a census)"); return 2
    out = {"arch": arch, "kernels": []}
    for b in rows:
        st = _kernel_stats(c, arch, b["kernel"], a.K)
        cand = (b["wall_pct"] or 0) >= CAND_WALL
        out["kernels"].append({
            "kernel": b["kernel"], "wall_pct": round(b["wall_pct"] or 0, 1),
            "l2_hit": round(b["l2_hit"] or 0, 1), "mem_busy": round(b["mem_busy"] or 0, 1),
            "bound": _bound_class(b["l2_hit"], b["mem_busy"]),
            "candidate": cand, "tried": st["tried"], "wins": st["wins"],
            "dead_streak": st["dead_streak"], "exhausted": st["exhausted"],
            "best_win_pct": st["best"],
            "status": ("EXHAUSTED" if st["exhausted"] else "OPEN" if cand else "below-threshold")})
    if a.json: print(json.dumps(out, indent=2)); return 0
    print(f"# BOD review — {arch}  (candidate wall%>={CAND_WALL}, exhausted at {a.K} consecutive deads)")
    print(f"{'kernel':44} {'wall%':>6} {'L2%':>5} {'bound':>12} {'tried':>5} {'win':>3} {'best%':>6} status")
    for k in out["kernels"]:
        print(f"{k['kernel'][:44]:44} {k['wall_pct']:6.1f} {k['l2_hit']:5.1f} {k['bound']:>12} "
              f"{k['tried']:5d} {k['wins']:3d} {(k['best_win_pct'] or 0):6.2f} {k['status']}")
    openk = [k for k in out["kernels"] if k["status"] == "OPEN"]
    print(f"\n{len(openk)} OPEN candidate kernels with lever room; "
          f"{sum(1 for k in out['kernels'] if k['status']=='EXHAUSTED')} exhausted.")
    return 0

def _bound_class(l2, mem):
    if l2 is None: return "?"
    if l2 < 40: return "DRAM-thrash"     # traffic-reduction lever
    if l2 > 60: return "cache-resident"  # ALU/occ lever
    return "mixed"

def cmd_why(a):
    """What was tried on a kernel + verdicts + why-dead. The tried/failed feedback surface."""
    c = db()
    st = _kernel_stats(c, a.arch, a.kernel, a.K)
    if a.json:
        print(json.dumps({"arch": a.arch, "kernel": a.kernel, "exhausted": st["exhausted"],
              "dead_streak": st["dead_streak"], "attempts": [dict(r) for r in st["rows"]]}, indent=2)); return 0
    print(f"# {a.kernel} on {a.arch}  ({st['tried']} attempts, {st['wins']} wins, "
          f"dead-streak {st['dead_streak']}/{a.K}{' — EXHAUSTED' if st['exhausted'] else ''})")
    for r in st["rows"]:
        mark = "WIN " if r["verdict"] == "WIN" else "    "
        print(f"  {mark}{r['verdict']:14} {(r['delta_pct'] or 0):+6.2f}%  {r['lever']}")
    if st["exhausted"]:
        print("  ==> EXHAUSTED: certify will REJECT new levers on this kernel (bounds enforced).")
    return 0

def cmd_certify(a):
    """codex-callable submit. Bounds enforced HERE — refuse exhausted/over-budget/off-target."""
    c = db()
    st = _kernel_stats(c, a.arch, a.kernel, a.K)
    if st["exhausted"]:
        print(json.dumps({"accepted": False, "reason": "KERNEL_EXHAUSTED",
              "detail": f"{a.kernel} hit {a.K} consecutive deads — try an OPEN kernel (see: ar bod)"})); return 3
    bod = c.execute("SELECT wall_pct FROM bod WHERE arch=? AND kernel=?", (a.arch, a.kernel)).fetchone()
    if not bod or (bod["wall_pct"] or 0) < CAND_WALL:
        print(json.dumps({"accepted": False, "reason": "OFF_TARGET",
              "detail": f"{a.kernel} is not a census candidate (wall%<{CAND_WALL})"})); return 3
    run = c.execute("SELECT * FROM runs WHERE arch=? AND status='running' ORDER BY started DESC", (a.arch,)).fetchone()
    if run:
        if run["calls"] >= run["budget"]:
            print(json.dumps({"accepted": False, "reason": "BUDGET_SPENT",
                  "detail": f"run {run['id']} used {run['calls']}/{run['budget']} codex-calls"})); return 3
        if run["ttl"] and time.time() > run["started"] + run["ttl"]:
            print(json.dumps({"accepted": False, "reason": "TTL_EXPIRED"})); return 3
        c.execute("UPDATE runs SET calls=calls+1 WHERE id=?", (run["id"],)); c.commit()
    # NOTE: actual certify delegated to ab_certify_v2p (the proven grader); this gate is the leash.
    print(json.dumps({"accepted": True, "run": run["id"] if run else None,
          "note": "bounds OK — invoke ab_certify_v2p; result lands in ledger, then `ar ingest`."})); return 0

def cmd_status(a):
    c = db()
    runs = c.execute("SELECT * FROM runs WHERE status='running'").fetchall()
    tot = c.execute("SELECT COUNT(*) n,SUM(verdict='WIN') w FROM attempts").fetchone()
    print(f"# ar status  ({tot['n']} attempts banked, {tot['w'] or 0} wins, archs: {_arches(c)})")
    if not runs: print("  no running loops."); return 0
    host = a.host or SSH_HOST
    for r in runs:
        left = (r["started"] + r["ttl"] - int(time.time())) if r["ttl"] else None
        live = _alive(r["pid"], host) if host else "no-host"   # LIVE via real ssh kill -0, not a stale field
        print(f"  run {r['id']} arch={r['arch']} card={r['card']} pid={r['pid']} [{live}] "
              f"calls={r['calls']}/{r['budget']} ttl_left={left}s")
    return 0

def cmd_start(a):
    """operator-only. LAUNCHES driver_v3 on the GPU box over ssh, captures the REAL pid, records a
    bounded run. --adopt-pid N adopts an already-running loop (records it) instead of launching."""
    host = a.host or SSH_HOST
    if not host: print(json.dumps({"error": "no host — pass --host or set AR_SSH_HOST"})); return 2
    c = db(); rid = f"{a.arch}-c{a.card}-{int(time.time())}"
    baseline = a.baseline or f"loop_baseline_{a.arch}"
    prompt = a.prompt or f"~/hipfire/autoresearch/harness/loop_round_prompt_{a.arch}.txt"
    # SWARM: launch N parallel per-card workers via swarm_explore.sh (each an independent driver_v3
    # over agent_exec.sh -> the chosen harness). swarm_explore setsid-detaches every worker and echoes
    # its pid, then exits — so this ssh returns after launch. Each worker is recorded as a bounded run.
    if getattr(a, "swarm", False):
        cards = (a.cards or "0 1 2 3").split()
        workers = a.workers or " ".join(f"{x}:{x}:{x}" for x in cards)  # slot:dev:drmcard
        tip = a.baseline or f"loop/{a.arch}"
        sw = f"{REMOTE_REPO}/autoresearch/harness/swarm_explore.sh"
        out_f = f"/tmp/swarm_{a.arch}.out"
        env = (f"ARCH={a.arch} MODEL={a.model} K={a.K} CAP={a.cap} BASE_TIP={tip} "
               f"WORKERS='{workers}' AGENT_HARNESS={a.harness}"
               + (f" AGENT_MODEL={a.agent_model}" if a.agent_model else ""))
        launch = (f"env {env} bash {sw} > {out_f} 2>&1; "
                  f"grep -oE 'pid=[0-9]+' {out_f} | grep -oE '[0-9]+' | tr '\\n' ' '")
        rc, out, err = _ssh(launch, host)
        pids = [int(p) for p in out.strip().split() if p.strip().isdigit()]
        if not pids:
            _, dump, _ = _ssh(f"tail -25 {out_f} 2>/dev/null", host)
            print(json.dumps({"error": "swarm launch produced no worker pids", "swarm_out": dump[-1000:]})); return 1
        runs = []
        for idx, p in enumerate(pids):
            wid = f"{a.arch}-swarm-w{idx}-{int(time.time())}"
            c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (wid, a.arch, a.model, idx, "running", int(time.time()), None, a.budget, 0, a.ttl, p))
            runs.append({"run": wid, "worker": idx, "pid": p, "live": _alive(p, host)})
        c.commit()
        nlive = sum(1 for r in runs if r["live"] == "LIVE")
        print(json.dumps({"swarm": a.arch, "workers": len(pids), "live": nlive, "tip": tip,
                          "harness": a.harness, "K": a.K, "cap": a.cap, "ttl": a.ttl, "runs": runs,
                          "note": f"swarm launched on {host}: {nlive}/{len(pids)} workers LIVE"}))
        return 0 if nlive else 1
    if a.adopt_pid:
        if _alive(a.adopt_pid, host) != "LIVE":
            print(json.dumps({"error": f"adopt pid {a.adopt_pid} not LIVE on {host}"})); return 1
        pid, note = a.adopt_pid, f"adopted running loop pid={a.adopt_pid}"
    else:
        pidf = f"/tmp/loop_driver_{a.arch}.pid"; out_f = f"/tmp/loop_driver_{a.arch}.out"
        env = (f"ARCH={a.arch} BASELINE_REF={baseline} CARDS={a.card} PROMPT={prompt} "
               f"ROLLOVER={a.rollover} K={a.K} AGENT_HARNESS={a.harness}"
               + (f" AGENT_MODEL={a.agent_model}" if a.agent_model else ""))
        # Pattern A: `env .. setsid nohup .. & echo $!`. setsid detaches into a new session so ssh
        # RETURNS IMMEDIATELY (a channel-hang here previously timed out the client while the driver
        # kept running => orphan runaway). setsid exec-replaces (not pgroup leader) so $! IS the real
        # driver pid; also stamped to a pidfile so a launch is recoverable even if pid capture fails.
        launch = (f"env {env} setsid nohup bash {a.driver} {a.cap} >{out_f} 2>&1 </dev/null "
                  f"& p=$!; echo $p > {pidf}; echo $p")
        rc, out, err = _ssh(launch, host)
        try: pid = int(out.strip().split()[-1])
        except (ValueError, IndexError): pid = None
        if not pid or _alive(pid, host) != "LIVE":
            # tear down any orphan we may have spawned before failing (no silent runaway)
            if pid: _ssh(f"kill {pid} 2>/dev/null; true", host)
            print(json.dumps({"error": "launch failed / driver not LIVE", "stdout": out, "stderr": err})); return 1
        note = f"launched driver_v3 pid={pid} on {host} (cap={a.cap}, harness={a.harness}, self-terminates on exhaustion)"
    c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
              (rid, a.arch, a.model, a.card, "running", int(time.time()), None, a.budget, 0, a.ttl, pid))
    c.commit()
    print(json.dumps({"run": rid, "pid": pid, "host": host, "budget": a.budget, "ttl": a.ttl, "note": note}))
    return 0

def cmd_stop(a):
    """Actually KILLS the remote driver pid + its codex round over ssh, then flips the DB flag."""
    c = db(); host = a.host or SSH_HOST
    runs = c.execute("SELECT * FROM runs WHERE status='running'" + (" AND id=?" if a.run else ""),
                     (a.run,) if a.run else ()).fetchall()
    killed = []
    for r in runs:
        if host and r["pid"]:
            # kill the recorded driver pid, then reap the whole loop tree: agent round (codex/grok),
            # the driver, and any live daemon so the GPU is freed. Broad by design for a clean `ar stop`.
            _ssh(f"kill {r['pid']} 2>/dev/null; "
                 f"pkill -9 -f '[c]odex exec --dangerously' 2>/dev/null; "
                 f"pkill -9 -f '[g]rok -p' 2>/dev/null; "
                 f"pkill -9 -f '[d]river_v3.sh' 2>/dev/null; "
                 f"pkill -9 -f '[a]gent_exec.sh' 2>/dev/null; true", host)
        killed.append({"run": r["id"], "pid": r["pid"], "after": _alive(r["pid"], host) if host and r["pid"] else "?"})
    c.execute("UPDATE runs SET status='stopped',stopped=? WHERE status='running'" + (" AND id=?" if a.run else ""),
              (int(time.time()), a.run) if a.run else (int(time.time()),)); c.commit()
    print(json.dumps({"stopped": killed or "none running", "note": "remote driver+codex signalled; DB flag flipped"}))
    return 0

def cmd_purge(a):
    """Drop stale run rows (e.g. pid=None phantoms from before remote-pid capture)."""
    c = db()
    if a.pidless:
        n = c.execute("DELETE FROM runs WHERE pid IS NULL").rowcount
    elif a.run:
        n = c.execute("DELETE FROM runs WHERE id=?", (a.run,)).rowcount
    else:
        n = c.execute("DELETE FROM runs WHERE status='stopped'").rowcount
    c.commit(); print(json.dumps({"purged_runs": n}))
    return 0

def main():
    p = argparse.ArgumentParser(prog="hipfire_ar", description="autoresearch supervisor over driver_v3")
    p.add_argument("--K", type=int, default=K_DEFAULT); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest"); s.add_argument("--host", default=None)   # --host: rsync remote ledger first
    for name in ("bod",):
        s = sub.add_parser(name); s.add_argument("--arch", required=True); s.add_argument("--json", action="store_true")
    s = sub.add_parser("why"); s.add_argument("kernel"); s.add_argument("--arch", required=True); s.add_argument("--json", action="store_true")
    s = sub.add_parser("certify"); s.add_argument("--arch", required=True); s.add_argument("--kernel", required=True); s.add_argument("--label", default="?")
    s = sub.add_parser("status"); s.add_argument("--host", default=None)
    s = sub.add_parser("start"); s.add_argument("--arch", required=True); s.add_argument("--model", required=True)
    s.add_argument("--card", type=int, default=0); s.add_argument("--budget", type=int, default=40); s.add_argument("--ttl", type=int, default=7200)
    s.add_argument("--cap", type=int, default=30); s.add_argument("--host", default=None)
    s.add_argument("--baseline", default=None); s.add_argument("--prompt", default=None)
    s.add_argument("--rollover", default="~/hipfire/autoresearch/harness/noop_rollover.sh")
    s.add_argument("--driver", default=REMOTE_DRIVER); s.add_argument("--adopt-pid", type=int, default=None, dest="adopt_pid")
    s.add_argument("--harness", default="codex", choices=["codex", "grok"]); s.add_argument("--agent-model", default=None, dest="agent_model")
    s.add_argument("--swarm", action="store_true"); s.add_argument("--cards", default=None); s.add_argument("--workers", default=None)
    s = sub.add_parser("stop"); s.add_argument("--run", default=None); s.add_argument("--host", default=None)
    s = sub.add_parser("purge"); s.add_argument("--run", default=None); s.add_argument("--pidless", action="store_true")
    a = p.parse_args()
    fn = {"ingest": ingest, "bod": cmd_bod, "why": cmd_why, "certify": cmd_certify,
          "status": cmd_status, "start": cmd_start, "stop": cmd_stop, "purge": cmd_purge}[a.cmd]
    sys.exit(fn(a) or 0)

if __name__ == "__main__":
    main()

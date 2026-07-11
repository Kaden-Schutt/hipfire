#!/usr/bin/env python3
"""oracle-db — query the autoresearch ledger corpus (autoresearch/ledger/*.jsonl).

The ledgers are git-tracked, so `git pull` gives every contributor the full
research history; this CLI indexes it. The ledger IS the research record: every
A/B (win, loss, or noise) is one append-only JSON line with decode, clock,
coherence, roofline (L2-hit/occ/mem), register metadata (VGPR/LDS/scratch), and
knock-on. Winners' sources live in autoresearch/variants/ (reproducible).

Usage:
  oracle_db.py wins                    # every certified WIN, best-first
  oracle_db.py best <arch> <kernel>    # the best certified variant for a kernel
  oracle_db.py history <arch> <kernel> # full A/B history for a kernel
  oracle_db.py kernel <arch> <kernel>  # roofline + register history (the WHY)
  oracle_db.py summary                 # per-arch tally (wins/losses/noise/void)
  oracle_db.py banked                  # compounded perf banked from committed durable wins
"""
import json, sys, glob, os, collections

LEDGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger")

def load(arch=None, kernel=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(LEDGER_DIR, "*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if arch and r.get("arch") != arch:
                continue
            if kernel and r.get("kernel") != kernel:
                continue
            rows.append(r)
    return rows

def _rf(r, side):
    rf = (r.get("roofline") or {}).get("target_" + side) or {}
    return rf

def wins():
    ws = [r for r in load() if r.get("WIN")]
    ws.sort(key=lambda r: -(r.get("delta_pct") or 0))
    if not ws:
        print("no certified WINs yet")
        return
    for r in ws:
        rf = _rf(r, "var")
        extra = f"  occ {rf.get('occ')} L2 {rf.get('l2_hit_pct')} VGPR {rf.get('vgpr')}" if rf else ""
        print(f"  +{r['delta_pct']:>5}%  {r['arch']:<8} {r['kernel']:<40} {r.get('label','')}  ({r.get('var_decode')} vs {r.get('base_decode')} tok/s){extra}")

def best(arch, kernel):
    ws = [r for r in load(arch, kernel) if r.get("WIN")]
    if not ws:
        print(f"no certified WIN for {arch}/{kernel}")
        return
    b = max(ws, key=lambda r: r.get("var_decode") or 0)
    print(json.dumps(b, indent=1))

def history(arch, kernel):
    rows = load(arch, kernel)
    for r in rows:
        print(f"  {r.get('verdict','?'):<6} Δ{r.get('delta_pct'):>6}%  clk {r.get('base_sclk')}/{r.get('var_sclk')}  "
              f"coh {r.get('base_coh')}/{r.get('var_coh')}  {r.get('label','')}  bswap={r.get('bswap','')} benv={r.get('benv','')}")

def kernel(arch, kernel):
    rows = load(arch, kernel)
    print(f"# {arch}/{kernel} — roofline + register history (the WHY)")
    print(f"  {'verdict':<7}{'Δ%':>7}  {'wall%':>6}{'L2hit':>7}{'occ':>6}{'mem':>6}{'VGPR':>6}{'LDS':>6}{'scr':>5}  roofline")
    for r in rows:
        rf = _rf(r, "var") or _rf(r, "base")
        print(f"  {r.get('verdict','?'):<7}{str(r.get('delta_pct')):>7}  "
              f"{str(rf.get('wall_pct')):>6}{str(rf.get('l2_hit_pct')):>7}{str(rf.get('occ')):>6}"
              f"{str(rf.get('mem_busy')):>6}{str(rf.get('vgpr')):>6}{str(rf.get('lds')):>6}{str(rf.get('scratch')):>5}  {rf.get('roofline','')}")

def banked():
    """Compounded perf BANKED from committed, durable wins — the reward-hack scoreboard.
    Only wins that were git-committed AND passed the serve_harness durability gate
    count (the ledger records losses too, but they never bank). Per kernel, successive
    optimizations COMPOUND: product of (1+Δ)."""
    comp = collections.defaultdict(lambda: [1.0, 0])   # (arch,kernel) -> [product, count]
    for r in load():
        if r.get("verdict") == "WIN" and r.get("committed") and r.get("durable") != "FAIL":
            comp[(r["arch"], r["kernel"])][0] *= (1 + (r.get("delta_pct") or 0) / 100.0)
            comp[(r["arch"], r["kernel"])][1] += 1
    if not comp:
        print("  no committed wins banked yet — the counter moves only on a committed, durable win")
        return
    per_arch = collections.defaultdict(lambda: [1.0, 0])
    for (arch, k), (p, n) in sorted(comp.items(), key=lambda x: -x[1][0]):
        print(f"  {arch:<8} {k:<42} {n} win(s) -> +{(p-1)*100:6.2f}% compounded")
        per_arch[arch][0] *= p
        per_arch[arch][1] += n
    print("  " + "-" * 68)
    total = 0
    for arch, (p, n) in sorted(per_arch.items()):
        print(f"  {arch:<8} TOTAL: {n} wins banked, +{(p-1)*100:.2f}% compounded across its kernels")
        total += n
    print(f"  ══ {total} wins banked across the fleet ══")

def summary():
    tally = collections.defaultdict(lambda: collections.Counter())
    for r in load():
        tally[r.get("arch", "?")][r.get("verdict", "?")] += 1
        if r.get("VOID"):
            tally[r.get("arch", "?")]["VOID"] += 1
    for arch, c in sorted(tally.items()):
        print(f"  {arch:<8} WIN={c['WIN']} LOSS={c['LOSS']} NOISE={c['NOISE']} VOID={c['VOID']}")

def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    cmd = a[0]
    if cmd == "wins": wins()
    elif cmd == "banked": banked()
    elif cmd == "summary": summary()
    elif cmd in ("best", "history", "kernel") and len(a) >= 3: globals()[cmd](a[1], a[2])
    else: print(__doc__)

if __name__ == "__main__":
    main()

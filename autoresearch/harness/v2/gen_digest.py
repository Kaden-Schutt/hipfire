import json, sys, os, re
# args: <exhaustion.json> <bod.json> <cand_wall> <K> <folded_list>  -> tried-levers digest text
exh = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
bod = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {"rows": []}
cand_wall = float(sys.argv[3]); K = int(sys.argv[4])
folded = set(l.strip() for l in open(sys.argv[5])) if os.path.exists(sys.argv[5]) else set()
cands = [(r["kernel"], r.get("wall_pct", 0)) for r in bod.get("rows", []) if r.get("wall_pct", 0) >= cand_wall and r["kernel"] not in folded]
# tried levers + verdicts per kernel from the progress log
tried = {}
if os.path.exists("/tmp/loop_progress.log"):
    for l in open("/tmp/loop_progress.log"):
        m = re.search(r"kernel=(\S+) lever=(\S+) verdict=(\w+)", l)
        if m: tried.setdefault(m.group(1), []).append((m.group(2), m.group(3)))
lines = ["COVERAGE DIGEST -- pick 4 ACTIVE candidates this round, preferring those with the FEWEST attempts; NEVER re-try a lever already marked DEAD/INCONCLUSIVE on its kernel; SKIP every [EXHAUSTED] kernel:"]
# active first (fewest dead), then exhausted
cands.sort(key=lambda x: (exh.get(x[0], 0) >= K, exh.get(x[0], 0), -x[1]))
for k, w in cands:
    n = exh.get(k, 0); status = "EXHAUSTED" if n >= K else "ACTIVE"
    tl = ", ".join(f"{lv}->{vd}" for lv, vd in tried.get(k, [])[-6:]) or "(none tried)"
    lines.append(f"  [{status}] {k} ({w:.1f}% wall, {n}/{K} dead) tried: {tl}")
print("\n".join(lines))

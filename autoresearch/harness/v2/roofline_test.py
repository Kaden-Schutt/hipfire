import json, sys
d = json.load(open(sys.argv[1])); cur = float(sys.argv[2]); rows = d.get("rows", [])
tot_wall = sum(r.get("wall_pct", 0) for r in rows) or 1
speedup = 0.0
print("per-kernel headroom (binding util -> max speedup, capped 3x):")
for i, r in enumerate(rows):
    w = r.get("wall_pct", 0) / tot_wall
    mb = (r.get("mem_busy") or 0) / 100.0
    occ = (r.get("occ") or 0) / 100.0
    util = max(mb, occ, 0.05)
    headroom = min(0.90 / util, 3.0) if util > 0 else 1
    speedup += w * headroom
    if i < 8:
        print(f"  {r['kernel'][:30]:<30} wall={round(r.get('wall_pct',0),1):>4}% mem_busy={round(mb*100):>3}% occ={round(occ*100):>3}% -> {headroom:.2f}x")
ceiling = cur * speedup
print(f"\nROOFLINE-CEILING est: {ceiling:.0f} tok/s  (current {cur:.0f} = {100*cur/ceiling:.0f}% of ceiling)")

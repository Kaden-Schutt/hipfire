import json, sys, os
# args: <exhaustion.json> <bod.json> <cand_wall> <K> <folded_list>
# exit 0 = all candidate kernels exhausted (global stop); exit 1 = keep going.
exh = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
bod = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {"rows": []}
cand_wall = float(sys.argv[3]); K = int(sys.argv[4])
folded = set(l.strip() for l in open(sys.argv[5])) if os.path.exists(sys.argv[5]) else set()
cands = [r["kernel"] for r in bod.get("rows", []) if r.get("wall_pct", 0) >= cand_wall and r["kernel"] not in folded]
allx = bool(cands) and all(exh.get(k, 0) >= K for k in cands)
sys.exit(0 if allx else 1)

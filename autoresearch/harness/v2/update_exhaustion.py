import json, sys, glob, os
# args: <exhaustion.json> <round> <main_repo>  -- update per-kernel consecutive-dead counters
exh_path, rnd, main = sys.argv[1], sys.argv[2], sys.argv[3]
exh = json.load(open(exh_path)) if os.path.exists(exh_path) else {}
prefix = f"R{rnd}c"
byk = {}   # kernel -> [verdicts this round]
for f in glob.glob(main + "/autoresearch/ledger/swarm_gfx1201_*.jsonl"):
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        if str(d.get("label", "")).startswith(prefix):
            byk.setdefault(d.get("kernel"), []).append(d.get("verdict"))
for k, verds in byk.items():
    if not k: continue
    if "WIN" in verds:
        exh[k] = 0                                   # a win resets the kernel's exhaustion
    else:
        fails = sum(1 for v in verds if v in ("DEAD", "INCONCLUSIVE"))
        if fails: exh[k] = exh.get(k, 0) + fails      # +1 per failed adaptive-resolved attempt
json.dump(exh, open(exh_path, "w"))
if byk:
    print("  [exhaustion] round %s -> %s" % (rnd, ", ".join(f"{k.split('_')[-1]}={exh.get(k,0)}" for k in byk if k)))

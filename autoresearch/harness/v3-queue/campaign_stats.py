import sys, glob, json, re, os
# args: <arch> <min_round>  -> campaign verdict tally (excludes date-labels + stale rows)
arch = sys.argv[1]; min_round = int(sys.argv[2])
main = os.path.expanduser("~/hipfire")
n = w = dead = void = inc = cohf = 0
best = []
for f in glob.glob(main + f"/autoresearch/ledger/swarm_{arch}_*.jsonl"):
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        m = re.match(r"R(\d+)c", str(d.get("label", "")))
        if not m: continue
        rnd = int(m.group(1))
        if rnd < min_round or rnd >= 100000: continue   # campaign rounds only (drop date-labels)
        v = d.get("verdict"); n += 1
        if v == "WIN": w += 1; best.append((d.get("delta_pct"), d.get("kernel", "")[:22]))
        elif v == "DEAD": dead += 1
        elif v == "VOID": void += 1
        elif v == "INCONCLUSIVE": inc += 1
        elif v == "COHERENCE_FAIL": cohf += 1
bs = " ".join(f"+{dp}%{k}" for dp, k in sorted(best, reverse=True)[:3])
print(f"certs={n} WIN={w} DEAD={dead} VOID={void} INC={inc} COHF={cohf} {bs}")

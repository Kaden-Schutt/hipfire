import json, sys, glob, os
# args: <exhaustion.json> <round> <main_repo> [arch=gfx1201]  -- update per-kernel consecutive-dead counters
# arch selects the ledger glob so a per-arch loop only counts its OWN verdicts (BODs/ledgers don't cross-talk).
exh_path, rnd, main = sys.argv[1], sys.argv[2], sys.argv[3]
arch = sys.argv[4] if len(sys.argv) > 4 else "gfx1201"
exh = json.load(open(exh_path)) if os.path.exists(exh_path) else {}
prefix = f"R{rnd}c"
byk = {}   # kernel -> [verdicts this round]
for f in glob.glob(main + f"/autoresearch/ledger/swarm_{arch}_*.jsonl"):
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        if str(d.get("label", "")).startswith(prefix):
            byk.setdefault(d.get("kernel"), []).append(d.get("verdict"))
for k, verds in byk.items():
    if not k: continue
    # Unified exhaustion (matches exhaustion.py): a WIN resets; DEAD/COHERENCE_FAIL count toward the
    # dead streak (a kernel only made fast by breaking coherence IS exhausted for the loop — the v2
    # gate emits COHERENCE_FAIL, which the OLD rule never counted -> looped forever). INCONCLUSIVE
    # does NOT count (real-but-small -> re-measure). PARITY_FAIL/BUILD_FAIL are codex-fixable, not
    # dead. Per-round cap at +1, so K = consecutive ROUNDS, not attempts.
    if "WIN" in verds:
        exh[k] = 0
    elif any(v in ("DEAD", "COHERENCE_FAIL", "LOSS", "NOISE") for v in verds):
        exh[k] = exh.get(k, 0) + 1
json.dump(exh, open(exh_path, "w"))
if byk:
    print("  [exhaustion] round %s -> %s" % (rnd, ", ".join(f"{k.split('_')[-1]}={exh.get(k,0)}" for k in byk if k)))

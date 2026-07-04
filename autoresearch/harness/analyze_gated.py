import json, glob
rows = []
for f in glob.glob("/home/kaden/hipfire/autoresearch/ledger/swarm_gfx1201_*.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except: pass
print("=== high-consistency (f>=0.88) rows that did NOT bank as WIN -> why gated? ===")
n = 0
for r in sorted(rows, key=lambda r: -(r.get("mwu_dominance") or 0)):
    f = r.get("mwu_dominance") or 0; v = r.get("verdict")
    if f >= 0.88 and v != "WIN":
        bs = r.get("base_sclk", 0) or 0; vs = r.get("var_sclk", 0) or 0
        clk = abs(bs - vs) / max(bs, vs, 1) * 100 if bs and vs else 0
        d = r.get("delta_pct") or 0
        coh = f"{r.get('base_coh')}/{r.get('var_coh')}"
        print(f"  {v:<6} d=+{d:.2f}% f={f:.3f} coh={coh} sclk={bs}/{vs}(dlt{clk:.1f}%) conf={r.get('confirmed')} {r.get('kernel','')[:32]} ({r.get('label','')})")
        n += 1
if n == 0: print("  (none — all f>=0.88 rows banked as WIN; nothing wrongly gated)")
print(f"\n=== the confirm-branch gap: marginal rows (0.75<=f<0.90, delta>0.3) that FELL to NOISE ===")
m = 0
for r in rows:
    f = r.get("mwu_dominance") or 0; d = r.get("delta_pct") or 0
    if r.get("verdict") == "NOISE" and 0.75 <= f < 0.90 and d > 0.3:
        print(f"  d=+{d:.2f}% f={f:.3f} conf={r.get('confirmed')} {r.get('kernel','')[:34]} ({r.get('label','')})")
        m += 1
print(f"  -> {m} marginal-big rows that didn't confirm (candidates for MORE rounds, not discard)")

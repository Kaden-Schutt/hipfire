import json, glob, collections, statistics as st
FOLDED = set("gemv_hfq4g256_residual_scaled fused_rmsnorm_mq_rotate gemv_hfq4g256_residual gemv_hfq4g256_moe_down_k8_indexed_batched_expanded moe_topk_renorm_k8 gated_delta_net_q8_fast gemv_hfq4g256_moe_gate_up_indexed".split())
rows = []
for f in glob.glob("/home/kaden/hipfire/autoresearch/ledger/swarm_gfx1201_*.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except: pass
print(f"total ledger rows: {len(rows)}")
vc = collections.Counter(r.get("verdict") for r in rows)
print("verdict counts:", dict(vc))

# rows that are NOT banked WINs but are POSITIVE — the potentially-compoundable pool
# (delta>0, verdict!=WIN). Cross with mwu_dominance (f = consistency).
def f_of(r): return r.get("mwu_dominance") or 0.5
subfloor = [r for r in rows if r.get("verdict")!="WIN" and (r.get("delta_pct") or 0) > 0]
print(f"\nnon-WIN rows with delta>0: {len(subfloor)}")
# bucket by consistency
hi = [r for r in subfloor if f_of(r) >= 0.85]           # consistent small positives (real-ish)
mid= [r for r in subfloor if 0.65 <= f_of(r) < 0.85]
lo = [r for r in subfloor if f_of(r) < 0.65]            # noise (delta>0 by chance)
print(f"  f>=0.85 (consistent, LIKELY REAL small wins): {len(hi)}")
print(f"  0.65-0.85 (suggestive):                        {len(mid)}")
print(f"  f<0.65 (indistinguishable from noise):         {len(lo)}")

# the discarded compoundable pool: best-per-kernel among f>=0.85, delta in (0.05, 0.30), NOT folded
print("\n=== DISCARDED-BUT-CONSISTENT small positives (f>=0.85, 0.05<delta<0.30, unfolded), best per kernel ===")
best = {}
for r in rows:
    k = r.get("kernel","")
    if k in FOLDED: continue
    d = r.get("delta_pct") or 0
    if r.get("verdict")=="WIN": continue
    if not (0.05 < d < 0.30 and f_of(r) >= 0.85): continue
    if k not in best or d > best[k][0]: best[k] = (d, f_of(r), r.get("label",""))
tot = 0.0
for k,(d,f,lbl) in sorted(best.items(), key=lambda x:-x[1][0]):
    print(f"  +{d:.2f}%  f={f:.3f}  {k[:40]:<40} ({lbl})"); tot += d
print(f"  -> {len(best)} kernels, naive sum {tot:.2f}%  (rough compound ceiling IF each is real)")

# also: kernels beaten to noise already (many attempts, best delta tiny) = candidates for 'exhausted'
print("\n=== per-unfolded-kernel: #attempts and best delta (exhaustion view) ===")
perk = collections.defaultdict(list)
for r in rows:
    if r.get("kernel","") in FOLDED: continue
    perk[r.get("kernel","")].append((r.get("delta_pct") or 0, f_of(r)))
for k in sorted(perk, key=lambda k:-len(perk[k]))[:12]:
    ds = perk[k]; bd = max(d for d,_ in ds); bf = max(f for d,f in ds if d==bd)
    print(f"  {k[:40]:<40} attempts={len(ds):<3} best=+{bd:.2f}% (f={bf:.2f})")

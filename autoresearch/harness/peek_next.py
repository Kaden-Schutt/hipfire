import json, glob, os
MANIFEST = "/tmp/baseline_manifest.txt"
FOLDED = set("gemv_hfq4g256_residual_scaled fused_rmsnorm_mq_rotate gemv_hfq4g256_residual gemv_hfq4g256_moe_down_k8_indexed_batched_expanded moe_topk_renorm_k8 gated_delta_net_q8_fast gemv_hfq4g256_moe_gate_up_indexed".split())
last_epoch = 0.0; last_round = 0
for line in open(MANIFEST):
    if "rolled_at_epoch=" in line: last_epoch = float(line.split("rolled_at_epoch=")[1].split()[0])
    if "rolled_at_round=" in line: last_round = int(line.split("rolled_at_round=")[1].split()[0])
win = {}
for f in glob.glob("/home/kaden/hipfire/autoresearch/ledger/swarm_gfx1201_*.jsonl"):
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        if d.get("verdict") == "WIN": win[d["label"]] = (d["kernel"], d.get("delta_pct") or 0)
# (A) what a rollover-NOW would fold: fresh (.hip mtime > last rollover), best-per-kernel
best = {}
for wd in ["/tmp/wins", "/tmp/wins_seed"]:
    for hp in glob.glob(wd + "/*.hip"):
        lbl = os.path.basename(hp)[:-4]
        if lbl not in win: continue
        if os.path.getmtime(hp) <= last_epoch: continue
        k, dp = win[lbl]
        if k not in best or dp > best[k][1]: best[k] = (lbl, dp)
tot = sum(v[1] for v in best.values())
print(f"last rollover: round {last_round}")
print(f"(A) rollover-NOW would fold: {len(best)} kernels, naive_sum={tot:.2f}%  (pre-filter needs >=5%)")
for k,(lbl,dp) in sorted(best.items(), key=lambda x:-x[1][1]): print(f"      +{dp:.2f}%  {k[:38]} ({lbl})")
if not best: print("      -> NOTHING fresh to fold (rounds since v1 were winless) => rollover-now is a NO-OP")
# (B) unfolded WIN-kernels that HAVE a saved .hip (any age) — catchable if we relax the freshness filter
avail = {}
for wd in ["/tmp/wins", "/tmp/wins_seed"]:
    for hp in glob.glob(wd + "/*.hip"):
        lbl = os.path.basename(hp)[:-4]
        if lbl in win:
            k, dp = win[lbl]
            if k not in FOLDED and (k not in avail or dp > avail[k][1]): avail[k] = (lbl, dp)
print(f"\n(B) unfolded WIN-kernels with a saved .hip (any age): {len(avail)}")
for k,(lbl,dp) in sorted(avail.items(), key=lambda x:-x[1][1]): print(f"      +{dp:.2f}%  {k[:38]} ({lbl})")
# (C) WIN verdicts with NO saved .hip (the lost-win gap)
saved = set()
for wd in ["/tmp/wins","/tmp/wins_seed"]:
    for hp in glob.glob(wd+"/*.hip"): saved.add(os.path.basename(hp)[:-4])
lost = [(lbl,win[lbl]) for lbl in win if lbl not in saved]
print(f"\n(C) WIN verdicts whose .hip was NOT saved (unrecoverable this run): {len(lost)}")
for lbl,(k,dp) in sorted(lost, key=lambda x:-x[1][1])[:8]: print(f"      +{dp:.2f}%  {k[:38]} ({lbl})")

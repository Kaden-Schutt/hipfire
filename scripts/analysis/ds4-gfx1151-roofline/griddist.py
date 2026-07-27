import csv, collections
p="/home/kaden/g4prof/arA_kernel_trace.csv"; N=100; SLOTS=1280
grids=collections.defaultdict(collections.Counter)
dur=collections.Counter(); cnt=collections.Counter()
for r in csv.DictReader(open(p)):
    k=r.get("Kernel_Name"); s=r.get("Start_Timestamp")
    if not k or not s: continue
    cnt[k]+=1; dur[k]+=int(r["End_Timestamp"])-int(s)
    gx,gy,gz=int(r["Grid_Size_X"]),int(r["Grid_Size_Y"]),int(r["Grid_Size_Z"])
    wx,wy,wz=int(r["Workgroup_Size_X"]),int(r["Workgroup_Size_Y"]),int(r["Workgroup_Size_Z"])
    wg=max(1,(gx//max(wx,1))*(gy//max(wy,1))*(gz//max(wz,1)))
    waves=wg*max(1,-(-(wx*wy*wz)//32))
    grids[k][waves]+=1
big={"gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151","gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
     "gemv_mfp4g32_e8_soa_grouped_gfx1151","gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"}
rows=[(dur[k]/1e6/N,k) for k in dur if cnt[k]>=1000 and k not in big]
rows.sort(reverse=True)
print(f"{'ms/step':>7} {'distinct':>8} {'wave counts (waves x share)':<46} {'wtd fills':>9}")
print("-"*92)
starved=0.0
for ms,k in rows:
    g=grids[k]; tot=sum(g.values())
    top=sorted(g.items(), key=lambda x:-x[1])[:3]
    desc=" ".join(f"{w}x{c*100//tot}%" for w,c in top)
    wf=sum(w*c for w,c in g.items())/tot/SLOTS
    if wf<1.0: starved+=ms
    print(f"{ms:7.3f} {len(g):8d} {desc:<46} {wf:9.3f}  {k[:34]}")
print("-"*92)
print(f"weighted-fill < 1.0 total: {starved:.3f} ms/step of {sum(r[0] for r in rows):.3f}")

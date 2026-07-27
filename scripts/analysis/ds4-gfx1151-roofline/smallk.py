import csv, collections
p="/home/kaden/g4prof/arA_kernel_trace.csv"; N=100; SLOTS=1280
tot=collections.Counter(); cnt=collections.Counter(); meta={}
for r in csv.DictReader(open(p)):
    k=r.get("Kernel_Name"); s=r.get("Start_Timestamp")
    if not k or not s: continue
    tot[k]+=int(r["End_Timestamp"])-int(s); cnt[k]+=1
    if k not in meta:
        gx=int(r["Grid_Size_X"]); gy=int(r["Grid_Size_Y"]); gz=int(r["Grid_Size_Z"])
        wx=int(r["Workgroup_Size_X"]); wy=int(r["Workgroup_Size_Y"]); wz=int(r["Workgroup_Size_Z"])
        wg=max(1,(gx//max(wx,1))*(gy//max(wy,1))*(gz//max(wz,1)))
        thr=wx*wy*wz
        meta[k]=(wg, thr, wg*max(1,-(-thr//32)), int(r["VGPR_Count"]), int(r["LDS_Block_Size"]))
big={"gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151","gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
     "gemv_mfp4g32_e8_soa_grouped_gfx1151","gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"}
rows=[]
for k in tot:
    if cnt[k]<1000: continue
    ms=tot[k]/1e6/N; c=cnt[k]/N; wg,thr,waves,vgpr,lds=meta[k]
    rows.append((ms,c,ms*1000/c,wg,waves,waves/SLOTS,vgpr,k))
rows.sort(reverse=True)
small=[r for r in rows if r[7] not in big]
print(f"{'ms/step':>7} {'calls':>7} {'us/call':>8} {'WGs':>7} {'waves':>7} {'fills':>7} {'VGPR':>5}  kernel")
print("-"*104)
for ms,c,us,wg,waves,fills,vgpr,k in small:
    print(f"{ms:7.3f} {c:7.1f} {us:8.2f} {wg:7d} {waves:7d} {fills:7.3f} {vgpr:5d}  {k[:44]}")
print("-"*104)
tms=sum(r[0] for r in small); tc=sum(r[1] for r in small)
print(f"{tms:7.3f} {tc:7.1f} {tms*1000/tc:8.2f}  <- 26 small kernels")
und=[r for r in small if r[5]<1.0]
print(f"\nunder 1.0 occupancy fill: {len(und)}/{len(small)} kernels, {sum(r[0] for r in und):.3f} ms/step ({sum(r[0] for r in und)/tms:.0%} of the bucket), {sum(r[1] for r in und):.0f} launches/step")

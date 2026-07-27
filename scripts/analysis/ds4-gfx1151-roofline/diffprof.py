import csv, collections, sys
def load(p):
    tot=collections.Counter(); cnt=collections.Counter()
    with open(p) as f:
        for r in csv.DictReader(f):
            k=r.get("Kernel_Name") or r.get("Name") or r.get("kernel_name")
            s=r.get("Start_Timestamp") or r.get("start_ns"); e=r.get("End_Timestamp") or r.get("end_ns")
            if not k or not s: continue
            tot[k]+=int(e)-int(s); cnt[k]+=1
    return tot,cnt
A,Ac=load(sys.argv[1]); B,Bc=load(sys.argv[2])
N=100
keys=set(A)|set(B)
rows=[]
for k in keys:
    d=(B[k]-A[k])/1e6/N   # ms per iteration
    rows.append((d,k,A[k]/1e6/N,B[k]/1e6/N,Ac[k],Bc[k]))
rows.sort(reverse=True)
print(f"{'dMS/iter':>9} {'A ms':>8} {'B ms':>8} {'Acalls':>7} {'Bcalls':>7}  kernel")
print("-"*100)
tot=0
for d,k,a,b,ac,bc in rows:
    tot+=d
    if abs(d)<0.25: continue
    print(f"{d:>9.3f} {a:>8.3f} {b:>8.3f} {ac:>7} {bc:>7}  {k[:58]}")
print("-"*100)
print(f"{tot:>9.3f}  TOTAL GPU delta per iteration (ms)")
print(f"  arm A total {sum(A.values())/1e6/N:.3f} ms/iter | arm B total {sum(B.values())/1e6/N:.3f} ms/iter")

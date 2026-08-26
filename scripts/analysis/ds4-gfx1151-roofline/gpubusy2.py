import csv, collections, sys
p=sys.argv[1]; N=100
tot=collections.Counter(); cnt=collections.Counter()
with open(p) as f:
    for r in csv.DictReader(f):
        k=r.get("Kernel_Name") or r.get("Name")
        s=r.get("Start_Timestamp"); e=r.get("End_Timestamp")
        if not k or not s: continue
        tot[k]+=int(e)-int(s); cnt[k]+=1
# The prefix prefill ran ONCE (one 640-token chunk); decode ran N=100 times.
# So decode-path kernels have counts that scale with N (>=1000 here); the
# prefill path (the _wmma/_batched GEMM kernels) tops out at ~511 total.
dec=[(tot[k]/1e6/N, cnt[k]/N, k) for k in tot if cnt[k]>=1000]
pre=sum(tot[k] for k in tot if cnt[k]<1000)/1e6
dec.sort(reverse=True)
busy=sum(d[0] for d in dec)
wall=35.60; bytes_tok=6.043
print(f"AR decode GPU-busy : {busy:7.3f} ms/step   ({sum(d[1] for d in dec):.0f} launches/step)")
print(f"wall               : {wall:7.3f} ms/step")
print(f"NON-KERNEL gap     : {wall-busy:7.3f} ms/step  ({(wall-busy)/wall:.1%} of the token)")
print(f"prefill (once)     : {pre:7.1f} ms  [excluded]")
print(f"\nBW during GPU-busy : {bytes_tok/busy*1e3:7.1f} GB/s")
print(f"BW wall-clock      : {bytes_tok/wall*1e3:7.1f} GB/s")
print(f"\nif gap -> 0        : {1000/busy:7.2f} tok/s  ({1000/busy/(1000/wall)-1:+.1%})")
for ceil in (200,207,220,256):
    print(f"if BW -> {ceil} GB/s : {ceil/bytes_tok:7.2f} tok/s  (kernels would need {bytes_tok/ceil*1e3:.1f} ms vs {busy:.1f} now)")
print(f"\ntop decode kernels (ms/step):")
for ms,ps,k in dec[:10]:
    print(f"  {ms:7.3f}  {ps:6.1f}/step  {k[:58]}")

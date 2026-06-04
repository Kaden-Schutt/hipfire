# Copyright (c) 2026 Kaden Schutt
# Convert the v3 PyTorch *rotated* per-256-block Hessian npz into an
# UN-rotated (H_unrot = R^T H_rot R) flat binary keyed by dotted hfq name.
# Output format (HFHS-UNROT v1):
#   magic "HUNR" (4) | version u32=1 | n_tensors u32
#   per tensor: name_len u16 | name utf8 | n_groups u32 | (n_groups*256*256) f64 LE
import numpy as np, struct, sys

NPZ=sys.argv[1]; OUT=sys.argv[2]

def gen_signs(seed,n=256):
    s=seed; out=[]
    for _ in range(n):
        s=(s*1103515245+12345)&0x7FFFFFFF
        out.append(1.0 if ((s>>16)&1)==1 else -1.0)
    return np.array(out,dtype=np.float64)
S1=gen_signs(42); S2=gen_signs(1042)
def build_R():
    N=256; X=np.eye(N,dtype=np.float64)*S1[:,None]
    stride=1
    while stride<N:
        Xn=X.copy(); i=0
        while i<N:
            for j in range(stride):
                a=X[i+j].copy(); b=X[i+j+stride].copy()
                Xn[i+j]=a+b; Xn[i+j+stride]=a-b
            i+=stride*2
        X=Xn; stride<<=1
    return X*(0.0625*S2[:,None])
R=build_R()
assert np.max(np.abs(R.T@R-np.eye(256)))<1e-12

def demangle(k):
    return k.replace("__dot__",".")

z=np.load(NPZ)
keys=list(z.keys())
recs=[]
for k in keys:
    H=z[k].astype(np.float64)            # (G,256,256) rotated
    name=demangle(k)
    if not name.endswith(".weight"):
        name=name+".weight" if not name.endswith("weight") else name
    # un-rotate every block: H_unrot[g] = R^T @ H[g] @ R
    Hu=np.einsum("ai,gab,bj->gij", R, H, R, optimize=True)  # R^T H R per block
    Hu=0.5*(Hu+np.transpose(Hu,(0,2,1)))  # symmetrize
    recs.append((name,Hu.astype(np.float64)))
    print(f"{name}: G={Hu.shape[0]} diagmean={np.mean(np.diagonal(Hu,axis1=1,axis2=2)):.3f}", flush=True)

with open(OUT,"wb") as f:
    f.write(b"HUNR"); f.write(struct.pack("<I",1)); f.write(struct.pack("<I",len(recs)))
    for name,Hu in recs:
        nb=name.encode(); f.write(struct.pack("<H",len(nb))); f.write(nb)
        f.write(struct.pack("<I",Hu.shape[0]))
        f.write(Hu.tobytes(order="C"))
print("wrote",OUT,"tensors",len(recs))

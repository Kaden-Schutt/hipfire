#!/usr/bin/env python3
# stdlib-only KLD comparator for two DS4PPL01 logit dumps (no numpy).
#   usage: kld_stdlib.py <full.logits> <pruned.logits>
# A = full (reference P), B = pruned (Q).
import struct, sys, math
from array import array

def open_dump(p):
    f = open(p, "rb")
    assert f.read(8) == b"DS4PPL01", f"{p}: bad magic"
    vocab = struct.unpack("<I", f.read(4))[0]
    n = struct.unpack("<I", f.read(4))[0]
    return f, vocab, n

A, B = sys.argv[1], sys.argv[2]
fa, va, na = open_dump(A)
fb, vb, nb = open_dump(B)
assert va == vb and na == nb, f"mismatch vocab {va}/{vb} n {na}/{nb}"
V = va
exp, log, fsum = math.exp, math.log, math.fsum

skl_pq = skl_qp = 0.0
top1 = 0
for i in range(na):
    pa, ta = struct.unpack("<II", fa.read(8))
    xP = array("f"); xP.frombytes(fa.read(V * 4))
    pb, tb = struct.unpack("<II", fb.read(8))
    xQ = array("f"); xQ.frombytes(fb.read(V * 4))
    assert pa == pb and ta == tb, f"misalign at {i}: pos {pa}/{pb} tgt {ta}/{tb}"

    mP = max(xP); mQ = max(xQ)          # C-level
    eP = [exp(v - mP) for v in xP]
    eQ = [exp(v - mQ) for v in xQ]
    sP = fsum(eP); sQ = fsum(eQ)
    lseP = mP + log(sP); lseQ = mQ + log(sQ)
    # KL(P||Q) = (lseQ-lseP) + sum_i P_i (xP_i - xQ_i);  P_i = eP_i / sP
    accp = fsum(ep * (xp - xq) for ep, xp, xq in zip(eP, xP, xQ))
    accq = fsum(eq * (xq - xp) for eq, xp, xq in zip(eQ, xP, xQ))
    skl_pq += (lseQ - lseP) + accp / sP
    skl_qp += (lseP - lseQ) + accq / sQ
    if xP.index(mP) == xQ.index(mQ):
        top1 += 1
    if (i + 1) % 128 == 0:
        sys.stderr.write(f"  {i+1}/{na} KL(f||p)={skl_pq/(i+1):.4f}\n")

print(f"positions:               {na}")
print(f"mean KL(full || pruned): {skl_pq/na:.6f} nats")
print(f"mean KL(pruned || full): {skl_qp/na:.6f} nats")
print(f"top-1 argmax agreement:  {top1}/{na} = {100*top1/na:.2f}%")

#!/usr/bin/env python3
"""Analyze the ablation oracle: is diag(H) near-optimal as the channel selector?

Reads /tmp/ablation_oracle.tsv (rank, channel, diag_energy, ablation_kld) where
ablation_kld is the TRUE per-channel importance (KLD damage of voiding exactly that
one residual channel, all others exact). Reports:
  - Spearman rho between diag-rank and ablation-KLD (1.0 = diag is a perfect ranker)
  - monotonicity (rank inversions) of true importance along the diag order
  - tail-gain bound: how much an oracle could improve over diag at the tail
No scipy dependency (Spearman computed directly).
"""

import sys

rows = []
with open("/tmp/ablation_oracle.tsv") as f:
    next(f)  # header
    for line in f:
        p = line.split("\t")
        if len(p) < 4 or not p[3].strip():
            continue
        rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3])))

rows.sort(key=lambda r: r[0])  # by diag rank ascending (most important first)
n = len(rows)
print(f"samples: {n}")
print(f"{'diag_rank':>9} {'chan':>5} {'diag_energy':>12} {'ablation_kld':>13}")
for r in rows:
    print(f"{r[0]:>9} {r[1]:>5} {r[2]:>12.4g} {r[3]:>13.6f}")


def spearman(a, b):
    # rank-transform each, then Pearson on ranks
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        rk = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va * vb else 0.0


diag_energy = [r[2] for r in rows]
abl_kld = [r[3] for r in rows]
diag_rank = [r[0] for r in rows]

rho_e = spearman(diag_energy, abl_kld)
# diag-rank is ascending importance; correlate -rank with kld so +1 = perfect
rho_r = spearman([-x for x in diag_rank], abl_kld)
print(f"\nSpearman(diag_energy, ablation_KLD)   = {rho_e:+.4f}")
print(f"Spearman(diag_order,  ablation_KLD)   = {rho_r:+.4f}  (+1 => diag is a perfect ranker)")

# monotonicity along diag order: count inversions (later-rank channel with HIGHER kld)
inv = 0
pairs = 0
for i in range(n):
    for j in range(i + 1, n):
        pairs += 1
        if abl_kld[j] > abl_kld[i]:  # j is less important per diag but hurts more
            inv += 1
print(f"rank inversions: {inv}/{pairs} ({100 * inv / pairs:.1f}%)  (0% => diag never mis-orders a sampled pair)")

# tail-gain bound: among the sampled TAIL (diag_rank in bottom third), is the
# truly-least-important channel the diag-bottom one? If a higher-rank channel has
# LOWER ablation KLD than the diag-bottom, an oracle would void it instead.
tail = [r for r in rows if r[0] >= 0.66 * max(diag_rank)]
if tail:
    tail.sort(key=lambda r: r[3])  # by true importance ascending
    print(f"\ntail (diag_rank >= {0.66 * max(diag_rank):.0f}), sorted by TRUE importance (least first):")
    for r in tail:
        print(f"   diag_rank={r[0]:>4} chan={r[1]:>4} ablation_kld={r[3]:.6f}")
    diagbottom = max(rows, key=lambda r: r[0])
    truebottom = min(tail, key=lambda r: r[3])
    print(f"\n diag would void first: rank {diagbottom[0]} (kld {diagbottom[3]:.6f})")
    print(f" oracle would void first: rank {truebottom[0]} (kld {truebottom[3]:.6f})")
    print(f" tail-gain (diag - oracle) = {diagbottom[3] - truebottom[3]:+.6f} KLD per channel")

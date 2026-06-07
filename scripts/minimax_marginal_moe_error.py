import struct, numpy as np

def load(p):
    b = open(p, "rb").read()
    assert b[:8] == b"HFHS\0\0\0\0", p
    nl, npos, h, _ = struct.unpack_from("<IIII", b, 8)
    return np.frombuffer(b, dtype=np.float32, offset=24).reshape(nl, npos, h)

import sys
pm2_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sens-mq2.hfhs"
pa2_path = sys.argv[2] if len(sys.argv) > 2 else "/workspace/sens-mq2-pa.hfhs"
pm4 = load("/workspace/sens-mq4.hfhs"); pa4 = load("/workspace/sens-mq4-pa.hfhs")
pm2 = load(pm2_path); pa2 = load(pa2_path)
print(f"# comparing {pm2_path} vs mq4 reference")
nl = pm4.shape[0]

def cos(x, y):
    x = x.ravel().astype(np.float64); y = y.ravel().astype(np.float64)
    return float(x @ y / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12))

def rms(x):
    return float(np.sqrt((x.astype(np.float64) ** 2).mean()))

print("  L   in_cos  moe_cos  moe_mag4  moe_err  err/mag")
rows = []
for L in range(nl):
    moe4 = pm4[L] - pa4[L]; moe2 = pm2[L] - pa2[L]
    ic = cos(pa2[L], pa4[L]); mc = cos(moe2, moe4)
    mag = rms(moe4); err = rms(moe2 - moe4); ratio = err / (mag + 1e-9)
    rows.append((L, ic, mc, mag, err, ratio))
    print(f"{L:>3} {ic:>8.4f} {mc:>8.4f} {mag:>9.4f} {err:>8.4f} {ratio:>8.3f}")

print("=== top-15 by lowest MoE-output cosine (most-diverged expert block) ===")
for r in sorted(rows, key=lambda r: r[2])[:15]:
    print(f"  L{r[0]:>2} moe_cos={r[2]:.4f} mag={r[3]:.3f} err/mag={r[5]:.3f}")

print("=== top-15 by err/mag (relative quant error, contribution-weighted) ===")
for r in sorted(rows, key=lambda r: -r[5])[:15]:
    print(f"  L{r[0]:>2} err/mag={r[5]:.3f} moe_cos={r[2]:.4f} mag={r[3]:.3f}")

# Suggest a promotion set: layers whose MoE block is both high-magnitude and
# high relative error (the ones 2-bit hurts most in absolute terms).
thr = sorted([r[5] for r in rows])[int(nl * 0.55)]  # promote ~upper 45% by err/mag
promote = sorted([r[0] for r in rows if r[5] >= thr])
print(f"=== suggested promote set (err/mag >= {thr:.3f}, {len(promote)} layers) ===")
print(" ", promote)

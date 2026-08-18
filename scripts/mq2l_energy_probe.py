"""Replicate quantize_mq2g256_lloyd exactly and measure reconstruction energy.

Mirrors crates/hipfire-quantize/src/main.rs:
  gen_fwht_signs (LCG, seeds 42/1042)  -> :845
  cpu_fwht_256   (signs1 -> H256 -> 1/16 -> signs2) -> :873
  quantize_mq2g256_lloyd (percentile init, 8 Lloyd iters, sort+remap, fp16 cb) -> :3644
"""
import numpy as np

def gen_fwht_signs(seed, n):
    s = seed; out = np.empty(n, np.float32)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0x7fffffff
        out[i] = 1.0 if ((s >> 16) & 1) == 1 else -1.0
    return out

def cpu_fwht_256(x, s1, s2):
    x = x * s1
    stride = 1
    while stride < 256:
        i = 0
        while i < 256:
            a = x[i:i+stride].copy(); b = x[i+stride:i+2*stride].copy()
            x[i:i+stride] = a + b; x[i+stride:i+2*stride] = a - b
            i += stride*2
        stride <<= 1
    return x * (0.0625 * s2)

def lloyd_block(g, max_iter=8):
    """Returns (centroids_fp16, indices). Exact port of the encoder."""
    srt = np.sort(g)
    pct = lambda f: srt[min(int(round(f*255.0)), 255)]
    cb = np.array([pct(.125), pct(.375), pct(.625), pct(.875)], np.float32)
    idx = np.zeros(256, np.uint8)
    if srt[255] - srt[0] > 0:
        prev = None
        for it in range(max_iter):
            d = np.abs(g[:, None] - cb[None, :])
            best = np.argmin(d, axis=1).astype(np.uint8)   # ties -> lowest k, same as Rust
            idx = best
            if it > 0 and prev is not None and np.array_equal(prev, best):
                break
            prev = best.copy()
            for k in range(4):
                m = best == k
                if m.any():
                    cb[k] = np.float32(g[m].astype(np.float64).mean())
    order = np.argsort(cb, kind='stable')
    sorted_cb = cb[order]
    inv = np.empty(4, np.uint8)
    for new_i, old in enumerate(order):
        inv[old] = new_i
    idx = inv[idx]
    return sorted_cb.astype(np.float16).astype(np.float32), idx

def run(x, label, max_iter=8):
    s1 = gen_fwht_signs(42, 256); s2 = gen_fwht_signs(1042, 256)
    nb = len(x)//256
    xr = np.empty_like(x); xh = np.empty_like(x)
    for b in range(nb):
        g = cpu_fwht_256(x[b*256:(b+1)*256].astype(np.float32).copy(), s1, s2)
        cb, idx = lloyd_block(g, max_iter)
        xr[b*256:(b+1)*256] = g
        xh[b*256:(b+1)*256] = cb[idx]
    e_in, e_out = float(xr @ xr), float(xh @ xh)
    dot = float(xr @ xh)
    alpha = dot / e_out                     # least-squares optimal gain
    nrmse = float(np.linalg.norm(xr-xh) / np.linalg.norm(xr))
    xh_c = alpha * xh
    nrmse_c = float(np.linalg.norm(xr-xh_c) / np.linalg.norm(xr))
    print(f"{label:34s} iters={max_iter}")
    print(f"   energy retained  ||x̂||²/||x||²  = {e_out/e_in:.4f}   "
          f"DEFICIT = {100*(1-e_out/e_in):5.2f}%")
    print(f"   LS-optimal gain  α = <x,x̂>/<x̂,x̂> = {alpha:.4f}   "
          f"(gain error {100*(alpha-1):+.2f}%)")
    print(f"   NRMSE  raw = {nrmse:.4f}   after α-correction = {nrmse_c:.4f}   "
          f"({100*(1-nrmse_c/nrmse):+.2f}%)")
    print()

rng = np.random.default_rng(0)
N = 256*4000
run(rng.standard_normal(N).astype(np.float32), "Gaussian (FWHT-rotated ~ CLT)")
run(rng.standard_normal(N).astype(np.float32) * rng.gamma(2.0, 0.5, N).astype(np.float32),
    "Heavy-tailed (Gaussian x Gamma)")
run(rng.standard_normal(N).astype(np.float32), "Gaussian", max_iter=16)

print("Theory: Lloyd-Max 2-bit (4-level) on unit Gaussian has MSE = 0.1175 sigma^2")
print("     -> conditional-mean reconstruction retains 1 - 0.1175 = 0.8825 of the energy")
print("     -> predicted deficit = 11.75%")

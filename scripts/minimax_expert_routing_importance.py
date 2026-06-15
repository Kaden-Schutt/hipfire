import gguf, numpy as np

R = gguf.GGUFReader("/workspace/imatrix_unsloth_minimax.gguf")
T = {t.name: t for t in R.tensors}

NL, NE = 62, 256

def get(name):
    t = T.get(name)
    return None if t is None else np.array(t.data, dtype=np.float32)

# Per-layer routing counts + per-expert activation energy (gate input = MoE input).
counts = np.zeros((NL, NE), dtype=np.float64)
energy = np.zeros((NL, NE), dtype=np.float64)   # mean per-routing activation energy (gate)
denergy = np.zeros((NL, NE), dtype=np.float64)  # down-input energy
for n in range(NL):
    c = get(f"blk.{n}.ffn_gate_exps.weight.counts")
    s = get(f"blk.{n}.ffn_gate_exps.weight.in_sum2")   # [K, NE] flat, expert-major e*K+j
    d = get(f"blk.{n}.ffn_down_exps.weight.in_sum2")
    if c is None or s is None:
        continue
    c = c.reshape(-1)[:NE]
    counts[n] = c
    K = s.size // NE
    s2 = s.reshape(NE, K)            # in_sum2[e][j]
    energy[n] = s2.sum(axis=1) / np.maximum(c, 1)   # mean |x|^2 per routing
    if d is not None:
        Kd = d.size // NE
        denergy[n] = d.reshape(NE, Kd).sum(axis=1) / np.maximum(c, 1)

tot = counts.sum(axis=1)  # total routings per layer (= n_tok * top_k, ~const)
print(f"routings/layer: min={tot.min():.0f} max={tot.max():.0f} (= n_tok*top_k)")
print()

# --- Routing concentration: top-K coverage per layer ---
print("=== ROUTING CONCENTRATION (fraction of routings handled by top-K experts) ===")
print(f"{'layer':>5} {'top8':>6} {'top16':>6} {'top32':>6} {'top64':>6} {'maxfrac':>8} {'gini':>6}")
def gini(x):
    x = np.sort(x); n = len(x); cum = np.cumsum(x)
    return 0.0 if cum[-1] == 0 else (n + 1 - 2*np.sum(cum)/cum[-1]) / n
cov = {8: [], 16: [], 32: [], 64: []}
for n in range(NL):
    c = counts[n]; s = np.sort(c)[::-1]; t = c.sum()
    if t == 0: continue
    row = {k: s[:k].sum()/t for k in cov}
    for k in cov: cov[k].append(row[k])
    if n % 8 == 0 or n in (1, NL-1):
        print(f"{n:>5} {row[8]:>6.2f} {row[16]:>6.2f} {row[32]:>6.2f} {row[64]:>6.2f} {s[0]/t:>8.3f} {gini(c):>6.3f}")
print(f"{'MEAN':>5} {np.mean(cov[8]):>6.2f} {np.mean(cov[16]):>6.2f} {np.mean(cov[32]):>6.2f} {np.mean(cov[64]):>6.2f}")
print()

# --- Are the same expert INDICES hot across layers? ---
print("=== CROSS-LAYER HOTNESS (do hot experts repeat across layers?) ===")
topsets = [set(np.argsort(counts[n])[::-1][:32]) for n in range(NL) if counts[n].sum() > 0]
freq = np.zeros(NE)
for ts in topsets:
    for e in ts: freq[e] += 1
hot_global = np.argsort(freq)[::-1][:16]
print(f"experts in top-32 of the most layers: {[(int(e), int(freq[e])) for e in hot_global[:10]]}")
print(f"  (count = # of {len(topsets)} layers where that expert is top-32) — high => global hot expert")
print(f"  experts top-32 in >50% of layers: {int((freq > len(topsets)*0.5).sum())} / 256")
print()

# --- Importance = counts (selection freq) weighted by activation magnitude ---
imp = counts * np.sqrt(np.maximum(energy, 0))   # ~ counts * typical |x|
print("=== PROMOTION LEVERAGE: promote top-K experts/layer (by count) to higher precision ===")
print(f"{'K/layer':>8} {'rout_cov':>9} {'imp_cov':>8} {'size mq6':>9} {'size mq4':>9}")
# base sizes (per the produced files): mq2-lloyd=67.5GB experts; mq6~2.8x, mq4~1.8x of mq2-lloyd per expert
GB_MQ2 = 67.5
for K in (8, 16, 32, 48, 64):
    rc, ic = [], []
    for n in range(NL):
        c = counts[n]; t = c.sum()
        if t == 0: continue
        order = np.argsort(c)[::-1][:K]
        rc.append(c[order].sum()/t)
        it = imp[n].sum()
        ic.append(imp[n][order].sum()/it if it > 0 else 0)
    frac = K / NE
    # promoted experts cost (mult-1)x extra; mq6~2.8x, mq4~1.8x the mq2-lloyd per-expert bytes
    sz6 = GB_MQ2 + GB_MQ2 * frac * (2.8 - 1.0)
    sz4 = GB_MQ2 + GB_MQ2 * frac * (1.8 - 1.0)
    print(f"{K:>8} {np.mean(rc):>9.2f} {np.mean(ic):>8.2f} {sz6:>7.1f}GB {sz4:>7.1f}GB")
print()
print("rout_cov = fraction of token->expert routings that hit a PROMOTED (high-precision) expert")
print("imp_cov  = fraction of activation-weighted importance covered by the promoted set")

# --- gate/up vs down: which projection carries more energy (where to spend bits) ---
print()
ge = energy[counts > 0].mean(); de = denergy[counts > 0].mean()
print(f"=== mean per-routing energy: gate/up-input={ge:.3f}  down-input={de:.3f}  (ratio {de/ge:.2f}) ===")

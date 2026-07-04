#!/usr/bin/env python3
"""Verify the 5 main weight permutations are BIJECTIVE / function-preserving.

For each permutation type, apply the permutation + its required propagation to a
sub-computation (using random inputs, fp64), and check the output is unchanged
(max-abs-diff ~ 0). A permutation is "free" iff its synthetic forward is invariant.

Types (see docs/roughquant/README.md / the 5-permutation taxonomy):
  1. hidden-dim (general inter-linear)       — y = W2·act(W1·x), permute hidden
  2. MLP neurons (SwiGLU)                     — gate/up rows + down cols
  3. attention heads (GQA)                    — Q/K/V head-blocks + O input-blocks
  4. per-head dims                            — Q/K consistent, V/O consistent;
                                                 BROKEN by RoPE (demonstrated)
  5. residual stream (global)                 — norm γ + reader cols + writer rows

No GPU. Pure numpy on random tensors at the real arch dims (d=1024, mlp=3584,
n_q=16, n_kv=2, head_dim=128 for the GQA value path; RoPE on q/k head_dim).
"""

import numpy as np

rng = np.random.default_rng(0)
TOL = 1e-9  # fp64; a free permutation should be ~machine-zero


def ok(name, a, b):
    d = float(np.max(np.abs(a - b)))
    print(f"  {'PASS' if d < TOL else 'FAIL'}  {name:48s} max|Δ|={d:.2e}")
    return d < TOL


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


print("=== #1 hidden-dim (general two-linear stack) ===")
d_in, d_hid, d_out = 64, 96, 48
W1 = rng.standard_normal((d_hid, d_in))
W2 = rng.standard_normal((d_out, d_hid))
x = rng.standard_normal(d_in)
y = W2 @ silu(W1 @ x)
p = rng.permutation(d_hid)
y2 = (W2[:, p]) @ silu((W1[p]) @ x)  # permute hidden: W1 rows, W2 cols
ok("permute hidden dim (W1 rows + W2 cols)", y, y2)
# round-trip bijectivity of the permutation itself
pinv = np.argsort(p)
ok("permutation round-trip π∘π⁻¹ = id", W1, W1[p][pinv])

print("\n=== #2 MLP neurons (SwiGLU: gate/up rows + down cols) ===")
d, m = 1024, 3584
gate = rng.standard_normal((m, d))
up = rng.standard_normal((m, d))
down = rng.standard_normal((d, m))
x = rng.standard_normal(d)
y = down @ (silu(gate @ x) * (up @ x))
p = rng.permutation(m)
y2 = (down[:, p]) @ (silu((gate[p]) @ x) * ((up[p]) @ x))
ok("permute MLP intermediate neurons", y, y2)

print("\n=== #5 residual stream (global) ===")
d = 1024
h = rng.standard_normal(d)  # residual vector
g_in = rng.standard_normal(d)  # input_layernorm gamma
Wr = rng.standard_normal((512, d))  # a reader (e.g. q_proj), reads norm(h)
Ww = rng.standard_normal((d, 777))  # a writer (e.g. o_proj), writes into h
a = rng.standard_normal(777)  # writer input
read = Wr @ rmsnorm(h, g_in)
h_next = h + Ww @ a  # residual add
p = rng.permutation(d)
# propagate π: permute h, γ, reader input-cols, writer output-rows
read_p = (Wr[:, p]) @ rmsnorm(h[p], g_in[p])
ok("reader output invariant under residual perm", read, read_p)
h_next_p = h[p] + (Ww[p]) @ a  # writer rows permuted -> writes to h[p]
ok("residual-add lands in permuted slots", h_next[p], h_next_p)

print("\n=== #3 attention heads (GQA 16q/2kv, head_dim 128) — permute heads ===")
n_q, n_kv, hd = 16, 2, 128
grp = n_q // n_kv  # 8 query heads per kv head
T = 5
Q = rng.standard_normal((T, n_q, hd))
K = rng.standard_normal((T, n_kv, hd))
V = rng.standard_normal((T, n_kv, hd))
Wo = rng.standard_normal((1024, n_q * hd))  # o_proj over concatenated q-head outputs


def attn(Q, K, V, Wo):
    out = np.zeros((T, n_q * hd))
    for qh in range(n_q):
        kv = qh // grp
        s = Q[:, qh] @ K[:, kv].T / np.sqrt(hd)  # [T,T]
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        out[:, qh * hd : (qh + 1) * hd] = a @ V[:, kv]
    return out @ Wo.T


y = attn(Q, K, V, Wo)
# permute KV heads + carry their query-head groups together (GQA-respecting)
pkv = rng.permutation(n_kv)
pq = np.concatenate([np.arange(k * grp, (k + 1) * grp) for k in pkv])  # query heads follow their kv head
Wo_cols = np.concatenate([np.arange(qh * hd, (qh + 1) * hd) for qh in pq])
y3 = attn(Q[:, pq], K[:, pkv], V[:, pkv], Wo[:, Wo_cols])
ok("permute attn heads (GQA-respecting) + o_proj", y, y3)

print("\n=== #4 per-head dims — Q/K/V/O consistent; RoPE breaks it ===")
hd = 128
q = rng.standard_normal(hd)
k = rng.standard_normal(hd)
pd = rng.permutation(hd)
# (a) NO RoPE: dot product invariant under same dim-permutation of q and k
ok("Q·K invariant under per-head dim perm (no RoPE)", np.array(q @ k), np.array(q[pd] @ k[pd]))


# (b) RoPE: rotate dim pairs (i, i+hd/2) by angle pos*theta_i, THEN permute -> NOT invariant
def rope(v, pos, theta=10000.0):
    h2 = hd // 2
    out = v.copy()
    for i in range(h2):
        ang = pos * theta ** (-2.0 * i / hd)
        c, s = np.cos(ang), np.sin(ang)
        out[i], out[i + h2] = v[i] * c - v[i + h2] * s, v[i] * s + v[i + h2] * c
    return out


# RoPE encodes RELATIVE position: at equal pos the rotation cancels (q·k), which
# would mask the effect. Use DIFFERENT q/k positions to expose the constraint.
pq, pk = 5, 2
qk_rope = rope(q, pq) @ rope(k, pk)
qk_rope_perm = rope(q[pd], pq) @ rope(k[pd], pk)  # permute dims then position-RoPE
ok("Q·K invariant under per-head dim perm (WITH RoPE, pos_q≠pos_k)", np.array(qk_rope), np.array(qk_rope_perm))
print("  ^ EXPECTED FAIL: RoPE pins dims to frequencies; per-head-dim perm is NOT")
print("    free under RoPE. Only RoPE-pair-preserving perms (swap (i,i+hd/2) pairs")
print("    as units, or permute frequency-pairs) are bijective. #4 is constrained.")

print("\nDone. PASS = free/bijective permutation; the #4-with-RoPE FAIL is the constraint.")

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Numpy reference of Gemma4 layer 0 (sliding) from the tiny f32 safetensors.
Compares to the HF oracle AND the hipfire dump; runs hypothesis toggles to find
which op hipfire gets wrong (the variant that matches hipfire == the bug)."""
import sys, struct, numpy as np
from safetensors.numpy import load_file

D = sys.argv[1] if len(sys.argv) > 1 else "/home/kaden/gemma4-tiny-nw"
W = load_file(f"{D}/model.safetensors")
P = "model.language_model."
def w(n): return W[P + n].astype(np.float64)

def rd(p):
    f = open(p, "rb"); f.read(8); nl, npos, h, _ = struct.unpack("<IIII", f.read(16))
    return np.frombuffer(f.read(), dtype=np.float32).reshape(nl, npos, h).astype(np.float64)
hf = rd(f"{D}/oracle_hidden.hfhs")[0]
hip = rd(f"{D}/hipfire.hfhs")[0]

f = open(f"{D}/tokens.hfkldr", "rb"); f.read(8); hdr = f.read(24)
nctx = struct.unpack("<I", hdr[4:8])[0]
toks = list(struct.unpack(f"<{nctx}I", f.read(nctx*4)))

H = 512; nh = 4; nkv = 2; hd = 256; eps = 1e-6; theta = 10000.0
def rms(x, weight=None, axis=-1):
    o = x / np.sqrt((x**2).mean(axis, keepdims=True) + eps)
    return o * weight if weight is not None else o
def gelu(x):
    return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*x**3)))

def rope(x, half_split=True):  # x [npos, heads, hd]
    npos, heads, d = x.shape
    pos = np.arange(npos)[:, None]
    i = np.arange(d//2)[None, :]
    freq = theta ** (-(2*i)/d)
    ang = pos*freq  # [npos, d/2]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if half_split:
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return np.concatenate([x1*cos - x2*sin, x2*cos + x1*sin], -1)
    else:  # interleaved
        x1, x2 = x[..., 0::2], x[..., 1::2]
        r1, r2 = x1*cos - x2*sin, x2*cos + x1*sin
        out = np.empty_like(x); out[..., 0::2] = r1; out[..., 1::2] = r2; return out

def q8_0(x):  # quantize last-dim in 32-elem blocks (Q8_0): int8 * (max|x|/127)
    d = x.shape[-1]; o = x.copy().reshape(*x.shape[:-1], d//32, 32)
    sc = np.abs(o).max(-1, keepdims=True) / 127.0
    sc = np.where(sc == 0, 1.0, sc)
    o = np.round(o / sc) * sc
    return o.reshape(x.shape)

def layer0(qk_norm=True, v_norm=True, do_rope=True, half=True, scale=1.0, embed_scale=True, q8_kv=False):
    emb = w("embed_tokens.weight")[toks]
    if embed_scale: emb = emb * np.sqrt(H)
    L = "layers.0."
    n1 = rms(emb, w(L+"input_layernorm.weight"))
    q = (n1 @ w(L+"self_attn.q_proj.weight").T).reshape(nctx, nh, hd)
    k = (n1 @ w(L+"self_attn.k_proj.weight").T).reshape(nctx, nkv, hd)
    v = (n1 @ w(L+"self_attn.v_proj.weight").T).reshape(nctx, nkv, hd)
    if qk_norm:
        q = rms(q, w(L+"self_attn.q_norm.weight"))
        k = rms(k, w(L+"self_attn.k_norm.weight"))
    if v_norm: v = rms(v)
    if do_rope:
        q = rope(q, half); k = rope(k, half)
    if q8_kv:
        k = q8_0(k); v = q8_0(v)
    g = nh//nkv
    kk = np.repeat(k, g, axis=1); vv = np.repeat(v, g, axis=1)  # [npos, nh, hd]
    out = np.zeros((nctx, nh, hd))
    for h in range(nh):
        s = (q[:, h, :] @ kk[:, h, :].T) * scale  # [npos, npos]
        mask = np.triu(np.ones((nctx, nctx)), 1).astype(bool)
        s[mask] = -1e30
        s = s - s.max(-1, keepdims=True); e = np.exp(s); a = e/e.sum(-1, keepdims=True)
        out[:, h, :] = a @ vv[:, h, :]
    o = out.reshape(nctx, nh*hd) @ w(L+"self_attn.o_proj.weight").T
    o = rms(o, w(L+"post_attention_layernorm.weight"))
    h1 = emb + o
    res = h1
    n2 = rms(h1, w(L+"pre_feedforward_layernorm.weight"))
    act = gelu(n2 @ w(L+"mlp.gate_proj.weight").T) * (n2 @ w(L+"mlp.up_proj.weight").T)
    dn = rms(act @ w(L+"mlp.down_proj.weight").T, w(L+"post_feedforward_layernorm.weight"))
    return res + dn

def cos(a, b):
    return float((a*b).sum(-1).mean() / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).mean())

print(f"tokens={toks}")
variants = {
    "BASELINE (my understanding)": dict(),
    "no qk_norm": dict(qk_norm=False),
    "no v_norm": dict(v_norm=False),
    "interleaved rope": dict(half=False),
    "no rope": dict(do_rope=False),
    "scale=1/sqrt(hd)": dict(scale=1.0/np.sqrt(hd)),
    "no embed_scale": dict(embed_scale=False),
    "Q8 KV": dict(q8_kv=True),
    "Q8 KV + no qk_norm": dict(q8_kv=True, qk_norm=False),
}
print(f"{'variant':32} {'cos_vs_HF':>10} {'cos_vs_hipfire':>14}")
for name, kw in variants.items():
    out = layer0(**kw)
    print(f"{name:32} {cos(out, hf):>10.4f} {cos(out, hip):>14.4f}")

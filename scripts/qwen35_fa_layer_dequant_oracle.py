#!/usr/bin/env python3
"""Replay one Qwen3.5 full-attention layer with HFQ-dequantized MQ4 weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    d = a - b
    mse = float(np.mean(d * d))
    rmse = math.sqrt(mse)
    ref = math.sqrt(float(np.mean(b * b)))
    den = math.sqrt(float(np.dot(a, a))) * math.sqrt(float(np.dot(b, b)))
    return {
        "cosine": float(np.dot(a, b) / den) if den else float("nan"),
        "rel_rmse": float(rmse / ref) if ref else float("nan"),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def silu_t(x):
    return x * torch.sigmoid(x)


def rmsnorm_np(x, w, eps, offset=True):
    scale = (1.0 + w) if offset else w
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * scale


def mq_rotate_rows_np(x, probe):
    return np.stack([probe.mq_rotate_np(row) for row in np.asarray(x, dtype=np.float32)], axis=0)


def mq4_matmat(hfq, name, x, probe, device):
    info, data = hfq.tensor_bytes(name)
    if info["quant_type"] != 13:
        raise ValueError(f"{name} qt={info['quant_type']} expected MQ4")
    w_rot = probe.dequant_hfq4_rotated(data, info["shape"])
    x_rot = mq_rotate_rows_np(x, probe)
    with torch.inference_mode():
        wt = torch.from_numpy(w_rot).to(device=device, dtype=torch.float32)
        xt = torch.from_numpy(x_rot).to(device=device, dtype=torch.float32)
        y = xt @ wt.t()
    del wt, xt
    return y


def tensor_vec(hfq, name, layer0):
    info, data = hfq.tensor_bytes(name)
    if info["quant_type"] == 1:
        return layer0.f16_vec(data)
    if info["quant_type"] == 2:
        return layer0.f32_vec(data)
    raise ValueError(f"unsupported vec qt={info['quant_type']} for {name}")


def rope_cos_sin(seq, rot_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    pos = np.arange(seq, dtype=np.float32)
    freqs = np.outer(pos, inv).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return torch.from_numpy(np.cos(emb).astype(np.float32)), torch.from_numpy(np.sin(emb).astype(np.float32))


def rotate_half(x):
    a, b = torch.chunk(x, 2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(x, cos, sin):
    rot_dim = cos.shape[-1]
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    c = cos[None, :, None, :].to(x.device)
    s = sin[None, :, None, :].to(x.device)
    yr = xr * c + rotate_half(xr) * s
    return torch.cat([yr, xp], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args()

    probe = load_module("scripts/qwen35_hfq_projection_probe.py", "hfq_probe")
    layer0 = load_module("scripts/qwen35_layer0_dequant_oracle.py", "layer0_oracle")
    hfq = probe.Hfq(args.model)
    meta = json.loads(Path(args.hidden_meta).read_text())
    layers = [int(x) for x in meta["layers"]]
    hidden = np.memmap(meta["hidden_path"], dtype="<f4", mode="r", shape=(len(layers), meta["seq_len"], meta["dim"]))
    layer = args.layer
    if layer - 1 not in layers or layer not in layers:
        raise SystemExit("hidden dump must include previous and target layer")
    x = np.asarray(hidden[layers.index(layer - 1)]).astype(np.float32)
    hip_y = np.asarray(hidden[layers.index(layer)]).astype(np.float32)

    seq, dim = x.shape
    eps = 1e-6
    n_heads = 16
    n_kv = 4
    n_groups = n_heads // n_kv
    hd = 256
    rot_dim = 64
    theta = 10_000_000.0
    p = f"model.language_model.layers.{layer}"

    in_norm_w = tensor_vec(hfq, f"{p}.input_layernorm.weight", layer0)
    xn = rmsnorm_np(x, in_norm_w, eps, offset=True).astype(np.float32)
    q_full = mq4_matmat(hfq, f"{p}.self_attn.q_proj.weight", xn, probe, args.device).reshape(seq, n_heads, hd * 2)
    q, gate = torch.chunk(q_full, 2, dim=-1)
    gate = gate.reshape(seq, n_heads * hd)
    k = mq4_matmat(hfq, f"{p}.self_attn.k_proj.weight", xn, probe, args.device).reshape(seq, n_kv, hd)
    v = mq4_matmat(hfq, f"{p}.self_attn.v_proj.weight", xn, probe, args.device).reshape(seq, n_kv, hd)

    qnw = torch.from_numpy(1.0 + tensor_vec(hfq, f"{p}.self_attn.q_norm.weight", layer0)).to(args.device)
    knw = torch.from_numpy(1.0 + tensor_vec(hfq, f"{p}.self_attn.k_norm.weight", layer0)).to(args.device)
    q = q.to(args.device)
    k = k.to(args.device)
    v = v.to(args.device)
    q = q * torch.rsqrt((q * q).mean(dim=-1, keepdim=True) + eps) * qnw
    k = k * torch.rsqrt((k * k).mean(dim=-1, keepdim=True) + eps) * knw

    cos, sin = rope_cos_sin(seq, rot_dim, theta)
    q = apply_rope(q[None, ...], cos, sin)[0].transpose(0, 1)  # heads, seq, hd
    k = apply_rope(k[None, ...], cos, sin)[0].transpose(0, 1)
    v = v.transpose(0, 1)
    k = k.repeat_interleave(n_groups, dim=0)
    v = v.repeat_interleave(n_groups, dim=0)

    scores = torch.matmul(q, k.transpose(1, 2)) * (1.0 / math.sqrt(hd))
    mask = torch.triu(torch.ones(seq, seq, device=args.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask[None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v).transpose(0, 1).reshape(seq, dim)
    out = out * torch.sigmoid(gate.to(args.device))
    o = mq4_matmat(hfq, f"{p}.self_attn.o_proj.weight", out.detach().cpu().numpy().astype(np.float32), probe, args.device)
    x1 = torch.from_numpy(x).to(args.device) + o

    post_w = tensor_vec(hfq, f"{p}.post_attention_layernorm.weight", layer0)
    ffn_in = rmsnorm_np(x1.detach().cpu().numpy().astype(np.float32), post_w, eps, offset=True).astype(np.float32)
    gate_ffn = mq4_matmat(hfq, f"{p}.mlp.gate_proj.weight", ffn_in, probe, args.device)
    up = mq4_matmat(hfq, f"{p}.mlp.up_proj.weight", ffn_in, probe, args.device)
    ffn_hidden = (silu_t(gate_ffn) * up).detach().cpu().numpy().astype(np.float32)
    down = mq4_matmat(hfq, f"{p}.mlp.down_proj.weight", ffn_hidden, probe, args.device)
    y = (x1 + down).detach().cpu().numpy().astype(np.float32)

    report = {
        "schema": "hipfire.qwen35.fa_layer_dequant_oracle.v0",
        "layer": layer,
        "hidden_meta": args.hidden_meta,
        "model": args.model,
        "metrics": {
            "all": metrics(y, hip_y),
            "last": metrics(y[-1], hip_y[-1]),
        },
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    for k2, v2 in report["metrics"].items():
        print(k2, f"rel={v2['rel_rmse']:.6e}", f"max={v2['max_abs']:.6e}", f"cos={v2['cosine']:.9f}")


if __name__ == "__main__":
    main()

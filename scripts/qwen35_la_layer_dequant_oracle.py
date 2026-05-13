#!/usr/bin/env python3
"""Replay one Qwen3.5 linear-attention layer with HFQ-dequantized MQ4 weights."""
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
    return {"cosine": float(np.dot(a, b) / den) if den else float("nan"), "rel_rmse": float(rmse / ref) if ref else float("nan"), "rmse": rmse, "mae": float(np.mean(np.abs(d))), "max_abs": float(np.max(np.abs(d)))}


def silu_np(x):
    return x / (1.0 + np.exp(-x))


def silu_t(x):
    return x * torch.sigmoid(x)


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


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
    if info["quant_type"] == 13:
        n = int(np.prod(info["shape"]))
        return layer0.dequant_mq4_original_flat(data, n, load_module("scripts/qwen35_hfq_projection_probe.py", "probe_again"))
    raise ValueError(f"unsupported vec qt={info['quant_type']} for {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
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
    k_heads = 16
    v_heads = 32
    hd = 128
    k_dim = k_heads * hd
    v_dim = v_heads * hd
    p = f"model.language_model.layers.{layer}"

    in_norm_w = tensor_vec(hfq, f"{p}.input_layernorm.weight", layer0)
    xn = rmsnorm_np(x, in_norm_w, eps, offset=True).astype(np.float32)
    qkv = mq4_matmat(hfq, f"{p}.linear_attn.in_proj_qkv.weight", xn, probe, args.device).detach().cpu().numpy().astype(np.float32)
    z = mq4_matmat(hfq, f"{p}.linear_attn.in_proj_z.weight", xn, probe, args.device).detach().cpu().numpy().astype(np.float32)
    b = mq4_matmat(hfq, f"{p}.linear_attn.in_proj_b.weight", xn, probe, args.device).detach().cpu().numpy().astype(np.float32)
    a = mq4_matmat(hfq, f"{p}.linear_attn.in_proj_a.weight", xn, probe, args.device).detach().cpu().numpy().astype(np.float32)
    beta = 1.0 / (1.0 + np.exp(-b))
    a_log = tensor_vec(hfq, f"{p}.linear_attn.A_log", layer0)
    dt_bias = tensor_vec(hfq, f"{p}.linear_attn.dt_bias", layer0)
    gate = -np.exp(a_log.astype(np.float32))[None, :] * softplus_np((a + dt_bias[None, :]).astype(np.float32))
    alpha = np.exp(gate).astype(np.float32)

    qkv_dim = 2 * k_dim + v_dim
    conv_w = layer0.dequant_mq4_original_flat(hfq.tensor_bytes(f"{p}.linear_attn.conv1d.weight")[1], qkv_dim * 4, probe).reshape(qkv_dim, 4)
    conv = np.empty_like(qkv)
    state = np.zeros((qkv_dim, 3), dtype=np.float32)
    for t in range(seq):
        cur = qkv[t]
        y = conv_w[:, 3] * cur + conv_w[:, 2] * state[:, 0] + conv_w[:, 1] * state[:, 1] + conv_w[:, 0] * state[:, 2]
        conv[t] = silu_np(y).astype(np.float32)
        state[:, 2] = state[:, 1]
        state[:, 1] = state[:, 0]
        state[:, 0] = cur

    q = conv[:, :k_dim].reshape(seq, k_heads, hd)
    k = conv[:, k_dim:2*k_dim].reshape(seq, k_heads, hd)
    v = conv[:, 2*k_dim:].reshape(seq, v_heads, hd)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q = q * np.float32(1.0 / math.sqrt(hd))
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = np.repeat(q, v_heads // k_heads, axis=1)
    k = np.repeat(k, v_heads // k_heads, axis=1)

    s = np.zeros((v_heads, hd, hd), dtype=np.float32)
    attn_out = np.empty((seq, v_heads, hd), dtype=np.float32)
    for t in range(seq):
        for h in range(v_heads):
            kv = s[h] @ k[t, h]
            delta = (v[t, h] - alpha[t, h] * kv) * beta[t, h]
            s[h] = alpha[t, h] * s[h] + np.outer(delta, k[t, h]).astype(np.float32)
            attn_out[t, h] = s[h] @ q[t, h]

    norm_w = tensor_vec(hfq, f"{p}.linear_attn.norm.weight", layer0)
    rms = 1.0 / np.sqrt(np.mean(attn_out * attn_out, axis=-1, keepdims=True) + eps)
    normed = (attn_out * rms * norm_w[None, None, :] * silu_np(z.reshape(seq, v_heads, hd))).reshape(seq, v_dim).astype(np.float32)
    o = mq4_matmat(hfq, f"{p}.linear_attn.out_proj.weight", normed, probe, args.device)
    x1 = torch.from_numpy(x).to(args.device) + o

    post_w = tensor_vec(hfq, f"{p}.post_attention_layernorm.weight", layer0)
    ffn_in = rmsnorm_np(x1.detach().cpu().numpy().astype(np.float32), post_w, eps, offset=True).astype(np.float32)
    gate_ffn = mq4_matmat(hfq, f"{p}.mlp.gate_proj.weight", ffn_in, probe, args.device)
    up = mq4_matmat(hfq, f"{p}.mlp.up_proj.weight", ffn_in, probe, args.device)
    ffn_hidden = (silu_t(gate_ffn) * up).detach().cpu().numpy().astype(np.float32)
    down = mq4_matmat(hfq, f"{p}.mlp.down_proj.weight", ffn_hidden, probe, args.device)
    y = (x1 + down).detach().cpu().numpy().astype(np.float32)

    report = {"schema": "hipfire.qwen35.la_layer_dequant_oracle.v0", "layer": layer, "hidden_meta": args.hidden_meta, "model": args.model, "metrics": {"all": metrics(y, hip_y), "last": metrics(y[-1], hip_y[-1])}}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    for k2, v2 in report["metrics"].items():
        print(k2, f"rel={v2['rel_rmse']:.6e}", f"max={v2['max_abs']:.6e}", f"cos={v2['cosine']:.9f}")


if __name__ == "__main__":
    main()

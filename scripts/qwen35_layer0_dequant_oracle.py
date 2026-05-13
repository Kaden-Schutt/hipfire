#!/usr/bin/env python3
"""Replay Qwen3.5 layer 0 in PyTorch with HFQ-dequantized MQ4 weights.

This is a narrow engine-vs-quant oracle: it starts from hipfire's dumped
pre-layer-0 hidden vector and should match hipfire post-layer-0 output if the
runtime math is correct for the first DeltaNet block.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch


def load_probe_module():
    spec = importlib.util.spec_from_file_location("hfq_probe", "scripts/qwen35_hfq_projection_probe.py")
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


def f16_vec(data):
    return np.frombuffer(data, dtype="<f2").astype(np.float32)


def f32_vec(data):
    return np.frombuffer(data, dtype="<f4").astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsnorm(x, w, eps, offset=True):
    scale = (1.0 + w) if offset else w
    return x * (1.0 / np.sqrt(np.mean(x * x) + eps)) * scale


def inverse_fwht_group(v, signs1, signs2):
    g = v.astype(np.float32, copy=True)
    g *= signs2
    stride = 1
    while stride < 256:
        for j in range(0, 256, stride * 2):
            a = g[j:j+stride].copy()
            b = g[j+stride:j+2*stride].copy()
            g[j:j+stride] = a + b
            g[j+stride:j+2*stride] = a - b
        stride <<= 1
    g *= np.float32(0.0625)
    g *= signs1
    return g


def dequant_mq4_original_flat(data, n, mod):
    group_size = 256
    bytes_per_group = 136
    if len(data) % bytes_per_group:
        raise ValueError(f"bad mq4 byte count {len(data)}")
    out = np.empty((len(data) // bytes_per_group) * group_size, dtype=np.float32)
    pos = 0
    dst = 0
    for _ in range(len(data) // bytes_per_group):
        scale, zero = struct.unpack_from("<ff", data, pos)
        pos += 8
        packed = np.frombuffer(data[pos:pos+128], dtype=np.uint8)
        pos += 128
        vals = np.empty(256, dtype=np.float32)
        vals[0::2] = scale * (packed & 0xF).astype(np.float32) + zero
        vals[1::2] = scale * (packed >> 4).astype(np.float32) + zero
        out[dst:dst+256] = inverse_fwht_group(vals, mod.SIGNS1, mod.SIGNS2)
        dst += 256
    return out[:n]


def tensor_vec(hfq, name, n=None, raw=False, mod=None):
    info, data = hfq.tensor_bytes(name)
    qt = info["quant_type"]
    if qt == 1:
        v = f16_vec(data)
    elif qt == 2:
        v = f32_vec(data)
    elif qt == 13 and raw:
        if mod is None or n is None:
            raise ValueError("raw mq4 needs mod+n")
        v = dequant_mq4_original_flat(data, n, mod)
    else:
        raise ValueError(f"unsupported vec qt={qt} for {name}")
    return v


def mq4_matvec(hfq, name, x, mod, device):
    info, data = hfq.tensor_bytes(name)
    if info["quant_type"] != 13:
        raise ValueError(f"{name} qt={info['quant_type']} expected MQ4")
    shape = info["shape"]
    w_rot = mod.dequant_hfq4_rotated(data, shape)
    x_rot = mod.mq_rotate_np(x)
    with torch.inference_mode():
        y = torch.mv(
            torch.from_numpy(w_rot).to(device=device, dtype=torch.float32),
            torch.from_numpy(x_rot.astype(np.float32)).to(device=device),
        )
    return y.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-meta", required=True)
    ap.add_argument("--hidden-meta", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args()

    mod = load_probe_module()
    probe = json.loads(Path(args.probe_meta).read_text())
    hidden_meta = json.loads(Path(args.hidden_meta).read_text())
    hfq = mod.Hfq(probe["model_path"])
    eps = float(probe["norm_eps"])
    dim = int(probe["dim"])
    k_heads = 16
    v_heads = 32
    hd = 128
    k_dim = k_heads * hd
    v_dim = v_heads * hd
    qkv_dim = 2 * k_dim + v_dim

    paths = {k: Path(v) for k, v in probe["paths"].items()}
    x = np.memmap(paths["x"], dtype="<f4", mode="r", shape=(dim,)).astype(np.float32)
    hip_x_norm = np.memmap(paths["x_norm"], dtype="<f4", mode="r", shape=(dim,)).astype(np.float32)
    hidden_layers = [int(x) for x in hidden_meta["layers"]]
    hidden = np.memmap(hidden_meta["hidden_path"], dtype="<f4", mode="r", shape=(len(hidden_layers), hidden_meta["seq_len"], dim))
    hip_l0 = np.asarray(hidden[hidden_layers.index(0), 0]).astype(np.float32)

    p = "model.language_model.layers.0"
    in_norm = tensor_vec(hfq, f"{p}.input_layernorm.weight")
    x_norm = rmsnorm(x, in_norm, eps, offset=True).astype(np.float32)

    qkv = mq4_matvec(hfq, f"{p}.linear_attn.in_proj_qkv.weight", x_norm, mod, args.device)
    z = mq4_matvec(hfq, f"{p}.linear_attn.in_proj_z.weight", x_norm, mod, args.device)
    b = mq4_matvec(hfq, f"{p}.linear_attn.in_proj_b.weight", x_norm, mod, args.device)
    a = mq4_matvec(hfq, f"{p}.linear_attn.in_proj_a.weight", x_norm, mod, args.device)
    beta = 1.0 / (1.0 + np.exp(-b))
    a_log = tensor_vec(hfq, f"{p}.linear_attn.A_log")
    dt_bias = tensor_vec(hfq, f"{p}.linear_attn.dt_bias")
    gate = -np.exp(a_log.astype(np.float32)) * np.log1p(np.exp((a + dt_bias).astype(np.float32)))
    alpha = np.exp(gate).astype(np.float32)

    conv_w = tensor_vec(hfq, f"{p}.linear_attn.conv1d.weight", n=qkv_dim * 4, raw=True, mod=mod).reshape(qkv_dim, 4)
    conv = conv_w[:, 3] * qkv  # first token, zero conv state
    conv = silu(conv).astype(np.float32)
    q = conv[:k_dim].reshape(k_heads, hd)
    k = conv[k_dim:2*k_dim].reshape(k_heads, hd)
    v = conv[2*k_dim:].reshape(v_heads, hd)
    q = q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + 1e-6)
    q = q * np.float32(1.0 / math.sqrt(hd))
    k = k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + 1e-6)
    q = np.repeat(q, v_heads // k_heads, axis=0)
    k = np.repeat(k, v_heads // k_heads, axis=0)

    dot = np.sum(q * k, axis=1, keepdims=True)
    attn_out = (dot * (v * beta[:, None])).astype(np.float32)
    norm_w = tensor_vec(hfq, f"{p}.linear_attn.norm.weight")
    rms = 1.0 / np.sqrt(np.mean(attn_out * attn_out, axis=1, keepdims=True) + eps)
    normed = (attn_out * rms * norm_w[None, :] * silu(z.reshape(v_heads, hd))).reshape(v_dim).astype(np.float32)
    o = mq4_matvec(hfq, f"{p}.linear_attn.out_proj.weight", normed, mod, args.device)
    x1 = (x + o).astype(np.float32)

    post_norm_w = tensor_vec(hfq, f"{p}.post_attention_layernorm.weight")
    ffn_in = rmsnorm(x1, post_norm_w, eps, offset=True).astype(np.float32)
    gate_ffn = mq4_matvec(hfq, f"{p}.mlp.gate_proj.weight", ffn_in, mod, args.device)
    up = mq4_matvec(hfq, f"{p}.mlp.up_proj.weight", ffn_in, mod, args.device)
    ffn_hidden = (silu(gate_ffn) * up).astype(np.float32)
    down = mq4_matvec(hfq, f"{p}.mlp.down_proj.weight", ffn_hidden, mod, args.device)
    x2 = (x1 + down).astype(np.float32)

    report = {
        "schema": "hipfire.qwen35.layer0_dequant_oracle.v0",
        "probe_meta": args.probe_meta,
        "hidden_meta": args.hidden_meta,
        "metrics": {
            "x_norm_vs_hipfire": metrics(x_norm, hip_x_norm),
            "layer0_dequant_vs_hipfire": metrics(x2, hip_l0),
        },
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    for k, v in report["metrics"].items():
        print(k, f"rel={v['rel_rmse']:.6e}", f"max={v['max_abs']:.6e}", f"cos={v['cosine']:.9f}")


if __name__ == "__main__":
    main()

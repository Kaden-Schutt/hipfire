#!/usr/bin/env python3
"""PyTorch/HFQ oracle for the narrow Qwen3.5 layer-0 MQ4 projection probe."""
from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-meta", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out")
    p.add_argument("--max-tensor", default="qkv", choices=["qkv", "all"])
    return p.parse_args()


def cosine(a, b):
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = math.sqrt(float(np.dot(af, af))) * math.sqrt(float(np.dot(bf, bf)))
    return float(np.dot(af, bf) / denom) if denom else float("nan")


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    mse = float(np.mean(d * d))
    rmse = math.sqrt(mse)
    ref = math.sqrt(float(np.mean(b * b)))
    return {
        "cosine": cosine(a, b),
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": float(rmse / ref) if ref else float("nan"),
        "mae": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def find_json_end(buf, offset, end):
    brace = 0
    in_string = False
    esc = False
    for pos in range(offset, end):
        b = buf[pos]
        if esc:
            esc = False
            continue
        if b == 0x5C and in_string:
            esc = True
            continue
        if b == 0x22:
            in_string = not in_string
            continue
        if not in_string:
            if b == 0x7B:
                brace += 1
            elif b == 0x7D:
                brace -= 1
                if brace == 0:
                    return pos + 1
    raise ValueError("metadata JSON end not found")


class Hfq:
    def __init__(self, path):
        self.path = Path(path)
        self.f = self.path.open("rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.version, self.arch_id, self.n_tensors, meta_off, data_off = struct.unpack_from("<4sIIIQQ", self.mm, 0)
        if magic != b"HFQM":
            raise ValueError(f"not HFQ: {self.path}")
        json_end = find_json_end(self.mm, meta_off, data_off)
        self.metadata = json.loads(self.mm[meta_off:json_end])
        pos = json_end
        idx_n = struct.unpack_from("<I", self.mm, pos)[0]
        pos += 4
        if idx_n != self.n_tensors:
            raise ValueError(f"tensor count mismatch: {idx_n} != {self.n_tensors}")
        self.tensors = {}
        off = data_off
        for _ in range(idx_n):
            name_len = struct.unpack_from("<H", self.mm, pos)[0]
            pos += 2
            name = bytes(self.mm[pos:pos + name_len]).decode("utf-8")
            pos += name_len
            qt = self.mm[pos]
            pos += 1
            nd = self.mm[pos]
            pos += 1
            shape = list(struct.unpack_from("<" + "I" * nd, self.mm, pos))
            pos += 4 * nd
            group = struct.unpack_from("<I", self.mm, pos)[0]
            pos += 4
            size = struct.unpack_from("<Q", self.mm, pos)[0]
            pos += 8
            self.tensors[name] = {"quant_type": qt, "shape": shape, "group_size": group, "offset": off, "size": size}
            off += size

    def tensor_bytes(self, name):
        info = self.tensors[name]
        return info, self.mm[info["offset"]:info["offset"] + info["size"]]


def gen_fwht_signs(seed, n=256):
    state = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out[i] = 1.0 if ((state >> 16) & 1) == 1 else -1.0
    return out


SIGNS1 = gen_fwht_signs(42, 256)
SIGNS2 = gen_fwht_signs(1042, 256)


def fwht_group(v):
    out = v.astype(np.float32, copy=True)
    stride = 1
    while stride < out.shape[0]:
        for j in range(0, out.shape[0], stride * 2):
            a = out[j:j + stride].copy()
            b = out[j + stride:j + 2 * stride].copy()
            out[j:j + stride] = a + b
            out[j + stride:j + 2 * stride] = a - b
        stride <<= 1
    return out


def mq_rotate_np(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size % 256 != 0:
        raise ValueError("MQ rotation needs K multiple of 256")
    out = np.empty_like(x)
    for g in range(x.size // 256):
        sl = slice(g * 256, (g + 1) * 256)
        out[sl] = fwht_group(x[sl] * SIGNS1) * np.float32(0.0625) * SIGNS2
    return out


def dequant_hfq4_rotated(data, shape):
    m, k = shape
    if k % 256 != 0:
        raise ValueError(f"K {k} is not multiple of 256")
    groups_per_row = k // 256
    expected = m * groups_per_row * 136
    if len(data) != expected:
        raise ValueError(f"bad HFQ4/MQ4 byte size: got {len(data)} expected {expected}")
    out = np.empty((m, k), dtype=np.float32)
    pos = 0
    for row in range(m):
        for g in range(groups_per_row):
            scale, zero = struct.unpack_from("<ff", data, pos)
            pos += 8
            packed = data[pos:pos + 128]
            pos += 128
            base = g * 256
            vals = out[row, base:base + 256]
            # Low nibble then high nibble, matching gemv_hfq4g256.
            arr = np.frombuffer(packed, dtype=np.uint8)
            vals[0::2] = scale * (arr & 0xF).astype(np.float32) + zero
            vals[1::2] = scale * (arr >> 4).astype(np.float32) + zero
    return out


def torch_matvec(weight_np, x_np, device):
    with torch.inference_mode():
        w = torch.from_numpy(weight_np).to(device=device, dtype=torch.float32)
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(device=device)
        y = torch.mv(w, x)
        return y.detach().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    meta_path = Path(args.probe_meta)
    meta = json.loads(meta_path.read_text())
    hfq = Hfq(meta["model_path"])
    paths = {k: Path(v) for k, v in meta["paths"].items()}
    shapes = meta["shapes"]

    def load_vec(name):
        return np.memmap(paths[name], dtype="<f4", mode="r", shape=tuple(shapes[name.replace("_split", "").replace("_fused", "")]))

    x_norm = np.memmap(paths["x_norm"], dtype="<f4", mode="r", shape=tuple(shapes["x_norm"]))
    x_rot_split = np.memmap(paths["x_rot_split"], dtype="<f4", mode="r", shape=tuple(shapes["x_rot"]))
    x_rot_fused = np.memmap(paths["x_rot_fused"], dtype="<f4", mode="r", shape=tuple(shapes["x_rot"]))
    report = {
        "schema": "hipfire.qwen35.hfq_projection_probe.v0",
        "probe_meta": str(meta_path),
        "model_path": meta["model_path"],
        "device": args.device,
        "rotation": {},
        "projections": {},
    }

    x_rot_py = mq_rotate_np(np.asarray(x_norm))
    report["rotation"]["split_vs_python"] = metrics(np.asarray(x_rot_split), x_rot_py)
    report["rotation"]["fused_vs_python"] = metrics(np.asarray(x_rot_fused), x_rot_py)
    report["rotation"]["fused_vs_split"] = metrics(np.asarray(x_rot_fused), np.asarray(x_rot_split))

    items = [
        ("qkv", "wqkv", "qkv_split", "qkv_fused"),
        ("z", "wz", "z_split", "z_fused"),
        ("beta", "w_beta", "beta_split", "beta_fused"),
        ("alpha", "w_alpha", "alpha_split", "alpha_fused"),
    ]
    if args.max_tensor == "qkv":
        items = items[:1]
    for label, tensor_key, split_key, fused_key in items:
        tensor_name = meta["tensor_names"][tensor_key]
        info, data = hfq.tensor_bytes(tensor_name)
        if info["quant_type"] != 13:
            raise ValueError(f"{tensor_name} quant_type={info['quant_type']} expected MQ4 13")
        w_rot = dequant_hfq4_rotated(data, info["shape"])
        y_py = torch_matvec(w_rot, x_rot_py, args.device)
        y_split = np.memmap(paths[split_key], dtype="<f4", mode="r", shape=tuple(shapes[label]))
        y_fused = np.memmap(paths[fused_key], dtype="<f4", mode="r", shape=tuple(shapes[label]))
        report["projections"][label] = {
            "tensor": tensor_name,
            "shape": info["shape"],
            "split_vs_python": metrics(np.asarray(y_split), y_py),
            "fused_vs_python": metrics(np.asarray(y_fused), y_py),
            "fused_vs_split": metrics(np.asarray(y_fused), np.asarray(y_split)),
        }

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print("rotation")
    for k, v in report["rotation"].items():
        print(f"  {k}: rel={v['rel_rmse']:.6e} max={v['max_abs']:.6e} cos={v['cosine']:.9f}")
    print("projections")
    for k, r in report["projections"].items():
        print(f"  {k}")
        for kk in ["split_vs_python", "fused_vs_python", "fused_vs_split"]:
            v = r[kk]
            print(f"    {kk}: rel={v['rel_rmse']:.6e} max={v['max_abs']:.6e} cos={v['cosine']:.9f}")


if __name__ == "__main__":
    main()

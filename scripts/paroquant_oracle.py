#!/usr/bin/env python3
"""PyTorch oracle for hipfire PARO4G128 payloads.

This checks the producer-consumer contract we actually need for ParoQuant:
HFQ qtype 28 must preserve the native Paro/AWQ tensors, and the runtime kernel
must decode them as:

  rotate(x, pairs, theta, channel_scales) -> AWQ W4 dequant matmul

The script compares a source Paro safetensors module against the matching HFQ
record and evaluates both through a small PyTorch decode path. It is intentionally
CPU-friendly so it can run anywhere the importer runs.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

import paroquant_import as pqi


GROUP_SIZE = 128
BITS = 4
PACK = 32 // BITS
KROT = 8
PARO_QUANT_TYPE = 28
AWQ_REORDER = (0, 2, 4, 6, 1, 3, 5, 7)
AWQ_INV_REORDER = tuple(AWQ_REORDER.index(i) for i in range(PACK))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_object_end(raw: bytes) -> int:
    depth = 0
    in_string = False
    escape = False
    for i, b in enumerate(raw):
        c = chr(b)
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("could not find end of HFQ metadata JSON")


def normalize_module_name(name: str) -> str:
    for suffix in (".weight", ".qweight", ".qzeros", ".scales", ".pairs", ".theta", ".channel_scales"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_hfq_records(path: Path) -> dict[str, dict]:
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != b"HFQM":
        raise ValueError(f"not an HFQ file: {path}")
    n_tensors = struct.unpack_from("<I", data, 12)[0]
    metadata_offset = struct.unpack_from("<Q", data, 16)[0]
    data_offset = struct.unpack_from("<Q", data, 24)[0]
    metadata_region = data[metadata_offset:data_offset]
    pos = metadata_offset + json_object_end(metadata_region)
    idx_n = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    if idx_n != n_tensors:
        raise ValueError(f"HFQ index count {idx_n} does not match header count {n_tensors}")

    records: dict[str, dict] = {}
    payload_offset = data_offset
    for _ in range(n_tensors):
        name_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2
        name = data[pos : pos + name_len].decode("utf-8")
        pos += name_len
        quant_type = data[pos]
        pos += 1
        n_dims = data[pos]
        pos += 1
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from("<I", data, pos)[0])
            pos += 4
        group_size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        data_size = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        payload = data[payload_offset : payload_offset + data_size]
        if len(payload) != data_size:
            raise ValueError(f"HFQ record {name} payload is truncated")
        records[name] = {
            "name": name,
            "quant_type": quant_type,
            "shape": shape,
            "group_size": group_size,
            "data_size": data_size,
            "payload": payload,
        }
        payload_offset += data_size
    if payload_offset != len(data):
        raise ValueError(f"HFQ data end {payload_offset} does not match file size {len(data)}")
    return records


def read_hfq_paro_tensors(path: Path, base: str) -> dict[str, torch.Tensor]:
    records = read_hfq_records(path)
    name = f"{base}.weight"
    if name not in records:
        candidates = [k for k, v in records.items() if v["quant_type"] == PARO_QUANT_TYPE]
        raise KeyError(f"{name} not found in HFQ; first PARO records: {candidates[:8]}")
    record = records[name]
    if record["quant_type"] != PARO_QUANT_TYPE:
        raise ValueError(f"{name} has quant_type={record['quant_type']}, expected {PARO_QUANT_TYPE}")
    if record["group_size"] != GROUP_SIZE:
        raise ValueError(f"{name} has group_size={record['group_size']}, expected {GROUP_SIZE}")
    m, k = [int(x) for x in record["shape"]]
    if k % GROUP_SIZE != 0 or m % PACK != 0:
        raise ValueError(f"{name} shape {record['shape']} is not PARO4G128-compatible")

    groups = k // GROUP_SIZE
    m_pack = m // PACK
    payload = record["payload"]
    off = 0

    def take(dtype, shape):
        nonlocal off
        count = int(np.prod(shape))
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy().reshape(shape)
        off += arr.nbytes
        return arr

    tensors = {
        "qweight": torch.from_numpy(take("<i4", (k, m_pack))).to(torch.int32),
        "qzeros": torch.from_numpy(take("<i4", (groups, m_pack))).to(torch.int32),
        "scales": torch.from_numpy(take("<f2", (groups, m))).to(torch.float16),
        "pairs": torch.from_numpy(take("<i2", (KROT, k))).to(torch.int16),
        "theta": torch.from_numpy(take("<f2", (KROT, k // 2))).to(torch.float16),
        "channel_scales": torch.from_numpy(take("<f2", (k,))).to(torch.float16),
    }
    if off != len(payload):
        raise ValueError(f"{name} parsed {off} bytes but payload has {len(payload)} bytes")
    return tensors


def load_source_paro_tensors(index: dict[str, Path], base: str) -> dict[str, torch.Tensor]:
    qweight = pqi.load_tensor(index, f"{base}.qweight").to(torch.int32).contiguous()
    qzeros = pqi.load_tensor(index, f"{base}.qzeros").to(torch.int32).contiguous()
    scales = pqi.load_tensor(index, f"{base}.scales").to(torch.float16).contiguous()
    pairs = pqi.load_tensor(index, f"{base}.pairs").to(torch.int16).contiguous()
    theta = pqi.load_tensor(index, f"{base}.theta").to(torch.float16).contiguous()
    channel_scales = (
        pqi.load_tensor(index, f"{base}.channel_scales").reshape(-1).to(torch.float16).contiguous()
    )
    pqi.validate_module(base, qweight, qzeros, scales, pairs, theta, channel_scales)
    return {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
    }


def tensor_compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_match": False, "source_shape": list(a.shape), "hfq_shape": list(b.shape)}
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    if a.is_floating_point():
        diff = (a.float() - b.float()).abs()
        return {
            "shape_match": True,
            "exact": bool(torch.equal(a, b)),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        }
    neq = a.ne(b)
    return {
        "shape_match": True,
        "exact": bool(not neq.any().item()),
        "mismatches": int(neq.sum().item()),
    }


def unpack_awq_i32(packed: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1], PACK)
    out = torch.empty(logical_shape, dtype=torch.int32)
    words = packed.to(torch.int32)
    for logical in range(PACK):
        slot = AWQ_INV_REORDER[logical]
        out[..., logical] = (words >> (BITS * slot)) & 0xF
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * PACK)


def rotate_activation(x: torch.Tensor, pairs: torch.Tensor, theta: torch.Tensor, channel_scales: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    groups = k // GROUP_SIZE
    out = x.float().clone() * channel_scales.reshape(1, k).float()
    pairs_i64 = pairs.to(torch.int64)
    theta_f32 = theta.float()
    for g in range(groups):
        base = g * GROUP_SIZE
        group = out[:, base : base + GROUP_SIZE].clone()
        for r in range(KROT):
            ij = pairs_i64[r, base : base + GROUP_SIZE].reshape(GROUP_SIZE // 2, 2)
            i = ij[:, 0]
            j = ij[:, 1]
            th = theta_f32[r, g * (GROUP_SIZE // 2) : (g + 1) * (GROUP_SIZE // 2)]
            s = torch.sin(th).reshape(1, -1)
            c = torch.cos(th).reshape(1, -1)
            xi = group[:, i].clone()
            xj = group[:, j].clone()
            group[:, i] = xi * c + xj * s
            group[:, j] = xj * c - xi * s
        out[:, base : base + GROUP_SIZE] = group
    return out


def paro_linear_oracle(x: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    qweight = tensors["qweight"]
    qzeros = tensors["qzeros"]
    scales = tensors["scales"]
    pairs = tensors["pairs"]
    theta = tensors["theta"]
    channel_scales = tensors["channel_scales"]

    k = qweight.shape[0]
    m = qweight.shape[1] * PACK
    groups = k // GROUP_SIZE
    x_rot = rotate_activation(x, pairs, theta, channel_scales)
    q = unpack_awq_i32(qweight).float()
    z = unpack_awq_i32(qzeros).float()
    scales_f = scales.float()
    y = torch.zeros((x.shape[0], m), dtype=torch.float32)
    for g in range(groups):
        base = g * GROUP_SIZE
        w = (q[base : base + GROUP_SIZE, :] - z[g, :].reshape(1, m)) * scales_f[g, :].reshape(1, m)
        y += x_rot[:, base : base + GROUP_SIZE].matmul(w)
    return y


def first_paro_base(index: dict[str, Path]) -> str:
    modules, _ = pqi.discover_paro_modules(index)
    if not modules:
        raise ValueError("source checkpoint has no complete Paro modules")
    return modules[0]["base"]


def run_oracle(args: argparse.Namespace) -> dict:
    pqi.require_deps()
    source_dir = pqi.resolve_model(args.source, local_only=args.local_only)
    index = pqi.build_tensor_index(source_dir)
    base = normalize_module_name(args.module) if args.module else first_paro_base(index)
    source = load_source_paro_tensors(index, base)
    hfq = read_hfq_paro_tensors(Path(args.hfq).expanduser(), base)

    comparisons = {name: tensor_compare(source[name], hfq[name]) for name in source}
    tensor_exact = all(item.get("shape_match") and item.get("exact") for item in comparisons.values())

    k = int(source["qweight"].shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    x = torch.randn(args.samples, k, generator=gen, dtype=torch.float32) * args.input_scale
    source_y = paro_linear_oracle(x, source)
    hfq_y = paro_linear_oracle(x, hfq)
    diff = (source_y - hfq_y).abs()
    output_max_abs = float(diff.max().item()) if diff.numel() else 0.0
    output_mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    finite = bool(torch.isfinite(hfq_y).all().item())

    return {
        "schema": "hipfire.astrea.paro_oracle.v0",
        "captured_at_utc": utc_now(),
        "source": str(source_dir),
        "hfq": str(Path(args.hfq).expanduser()),
        "module": base,
        "layout": {
            "qweight": list(source["qweight"].shape),
            "qzeros": list(source["qzeros"].shape),
            "scales": list(source["scales"].shape),
            "pairs": list(source["pairs"].shape),
            "theta": list(source["theta"].shape),
            "channel_scales": list(source["channel_scales"].shape),
            "awq_reorder": list(AWQ_REORDER),
            "awq_inverse_reorder": list(AWQ_INV_REORDER),
        },
        "tensor_compare": comparisons,
        "tensor_exact": tensor_exact,
        "oracle": {
            "samples": args.samples,
            "seed": args.seed,
            "input_scale": args.input_scale,
            "source_vs_hfq_max_abs": output_max_abs,
            "source_vs_hfq_mean_abs": output_mean_abs,
            "hfq_output_finite": finite,
            "hfq_output_preview": [float(x) for x in hfq_y.reshape(-1)[: min(8, hfq_y.numel())]],
        },
        "pass": bool(tensor_exact and finite and output_max_abs <= args.atol),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PARO4G128 HFQ bytes against a PyTorch Paro oracle.")
    parser.add_argument("--source", required=True, help="ParoQuant safetensors directory or HuggingFace repo id.")
    parser.add_argument("--hfq", required=True, help="HFQ file produced by astrea paro-import.")
    parser.add_argument("--module", help="Paro module base name; defaults to the first complete module.")
    parser.add_argument("--local-only", action="store_true", help="Do not download HuggingFace sources.")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_oracle(args)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

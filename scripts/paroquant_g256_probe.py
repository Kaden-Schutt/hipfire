#!/usr/bin/env python3
"""CPU-side probe for ParoQuant G256 format choices.

This does not write HFQ files and does not use the GPU. It loads ParoQuant's
native G128 safetensors, dequantizes the rotated weight body, then compares:

  1. source PARO4G128 oracle output
  2. PARO4G256-style AWQ regrouping of the same rotated weights
  3. PARO4G256_MQ-style row-major HFQ4-G256 body with the same Paro rotation

The result is a format-loss probe, not a replacement for true G256 ParoQuant
calibration. It answers whether the storage/body choice is obviously doomed
before runtime kernels or full-model export work.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import paroquant_import as pqi
import paroquant_oracle as pqo


ROT_GROUP_SIZE = 128
G256 = 256
PACK = 8
AWQ_INV_REORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pack_awq_lanes(lanes: torch.Tensor) -> torch.Tensor:
    """Pack logical output lanes into AutoAWQ int32 physical slot order."""
    word = torch.zeros(lanes.shape[:-1], dtype=torch.int32)
    lanes_i32 = lanes.to(torch.int32)
    for logical in range(PACK):
        slot = AWQ_INV_REORDER[logical]
        word |= (lanes_i32[..., logical] & 0xF) << (4 * slot)
    return word


def unpack_awq_i32(packed: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1], PACK)
    out = torch.empty(logical_shape, dtype=torch.int32)
    words = packed.to(torch.int32)
    for logical in range(PACK):
        slot = AWQ_INV_REORDER[logical]
        out[..., logical] = (words >> (4 * slot)) & 0xF
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * PACK)


def dequant_awq_body(tensors: dict[str, torch.Tensor], *, group_size: int) -> torch.Tensor:
    q = unpack_awq_i32(tensors["qweight"]).float()
    z = unpack_awq_i32(tensors["qzeros"]).float()
    scales = tensors["scales"].float()
    k, m = q.shape
    groups = k // group_size
    rows = []
    for g in range(groups):
        base = g * group_size
        rows.append((q[base : base + group_size, :] - z[g, :].reshape(1, m)) * scales[g, :].reshape(1, m))
    return torch.cat(rows, dim=0)


def quantize_awq_g256(rotated_w_km: torch.Tensor) -> dict[str, torch.Tensor]:
    k, m = rotated_w_km.shape
    if k % G256 != 0:
        raise ValueError(f"K={k} is not divisible by {G256}")
    if m % PACK != 0:
        raise ValueError(f"M={m} is not divisible by {PACK}")
    groups = k // G256
    m_pack = m // PACK
    qweight = torch.empty((k, m_pack), dtype=torch.int32)
    qzeros = torch.empty((groups, m_pack), dtype=torch.int32)
    scales = torch.empty((groups, m), dtype=torch.float16)

    for g in range(groups):
        block = rotated_w_km[g * G256 : (g + 1) * G256, :].float()
        mn = block.min(dim=0).values
        mx = block.max(dim=0).values
        scale = ((mx - mn) / 15.0).clamp_min(1.0e-12)
        zero = torch.round(-mn / scale).clamp(0, 15).to(torch.int32)
        q = torch.round(block / scale.reshape(1, m) + zero.reshape(1, m)).clamp(0, 15).to(torch.uint8)
        scales[g, :] = scale.to(torch.float16)

        qweight[g * G256 : (g + 1) * G256, :] = pack_awq_lanes(q.reshape(G256, m_pack, PACK))
        qzeros[g, :] = pack_awq_lanes(zero.to(torch.uint8).reshape(m_pack, PACK))

    return {"qweight": qweight, "qzeros": qzeros, "scales": scales}


def quantize_hfq4g256_rows(rotated_w_km: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Quantize/dequantize with hipfire HFQ4-G256 row-major body semantics."""
    rows_mk = rotated_w_km.t().contiguous().float()
    m, k = rows_mk.shape
    if k % G256 != 0:
        raise ValueError(f"K={k} is not divisible by {G256}")
    groups = k // G256
    deq = torch.empty_like(rows_mk)
    for row in range(m):
        for g in range(groups):
            start = g * G256
            block = rows_mk[row, start : start + G256]
            mn = block.min()
            mx = block.max()
            scale = (mx - mn) / 15.0
            if float(scale) <= 0.0:
                scale = torch.tensor(1.0, dtype=torch.float32)
                inv = torch.tensor(0.0, dtype=torch.float32)
            else:
                inv = 1.0 / scale
            q = torch.round((block - mn) * inv).clamp(0, 15)
            deq[row, start : start + G256] = q * scale + mn
    payload_bytes = m * groups * 136
    return deq.t().contiguous(), payload_bytes


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (candidate.float() - reference.float()).reshape(-1)
    ref = reference.float().reshape(-1)
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    rms = torch.sqrt(torch.mean(ref * ref)).item()
    denom = rms if rms > 0.0 else 1.0
    max_abs = diff.abs().max().item() if diff.numel() else 0.0
    mean_abs = diff.abs().mean().item() if diff.numel() else 0.0
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "nrmse": rmse / denom,
    }


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    base = tensor_metrics(reference, candidate)
    ref = reference.float().reshape(reference.shape[0], -1)
    cand = candidate.float().reshape(candidate.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(ref, cand, dim=1)
    base["cosine_mean"] = float(cos.mean().item())
    base["cosine_min"] = float(cos.min().item())
    return base


def source_paro_tensors(index: dict[str, Path], base: str) -> dict[str, torch.Tensor]:
    tensors = pqo.load_source_paro_tensors(index, base)
    tensors["channel_scales"] = tensors["channel_scales"].reshape(-1)
    return tensors


def payload_bytes_native(k: int, m: int, *, quant_group: int, theta_bytes: int = 2) -> dict[str, int]:
    groups = k // quant_group
    m_pack = m // PACK
    qweight = k * m_pack * 4
    qzeros = groups * m_pack * 4
    scales = groups * m * 2
    pairs = pqi.KROT * k * 2
    theta = pqi.KROT * (k // 2) * theta_bytes
    channel_scales = k * 2
    total = qweight + qzeros + scales + pairs + theta + channel_scales
    return {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
        "total": total,
    }


def run_module(index: dict[str, Path], base: str, *, samples: int, seed: int, input_scale: float) -> dict:
    source = source_paro_tensors(index, base)
    k = int(source["qweight"].shape[0])
    m = int(source["qweight"].shape[1]) * PACK
    if k % G256 != 0:
        return {"base": base, "skipped": f"K={k} is not divisible by {G256}"}

    rotated_w = dequant_awq_body(source, group_size=ROT_GROUP_SIZE)
    g256_awq = quantize_awq_g256(rotated_w)
    g256_w = dequant_awq_body(g256_awq, group_size=G256)
    mq_w, mq_body_bytes = quantize_hfq4g256_rows(rotated_w)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(samples, k, generator=gen, dtype=torch.float32) * input_scale
    x_rot = pqo.rotate_activation(x, source["pairs"], source["theta"], source["channel_scales"])

    y_source = x_rot.matmul(rotated_w)
    y_g256 = x_rot.matmul(g256_w)
    y_mq = x_rot.matmul(mq_w)

    source_bytes = payload_bytes_native(k, m, quant_group=ROT_GROUP_SIZE)
    g256_bytes = payload_bytes_native(k, m, quant_group=G256)
    side_bytes = source_bytes["pairs"] + source_bytes["theta"] + source_bytes["channel_scales"]
    mq_total_bytes = mq_body_bytes + side_bytes

    return {
        "base": base,
        "shape": {"m": m, "k": k},
        "payload_bytes": {
            "source_paro4g128": source_bytes["total"],
            "paro4g256_awq": g256_bytes["total"],
            "paro4g256_mq_body_plus_side": mq_total_bytes,
            "mq_body_only": mq_body_bytes,
            "shared_paro_side_metadata": side_bytes,
        },
        "payload_ratio_vs_source": {
            "paro4g256_awq": g256_bytes["total"] / source_bytes["total"],
            "paro4g256_mq_body_plus_side": mq_total_bytes / source_bytes["total"],
        },
        "weight_reconstruction": {
            "paro4g256_awq": tensor_metrics(rotated_w, g256_w),
            "paro4g256_mq_body": tensor_metrics(rotated_w, mq_w),
        },
        "output_vs_source": {
            "paro4g256_awq": output_metrics(y_source, y_g256),
            "paro4g256_mq_body": output_metrics(y_source, y_mq),
        },
    }


def summarize(results: list[dict]) -> dict:
    usable = [r for r in results if "output_vs_source" in r]
    if not usable:
        return {"modules": 0}

    def avg(path: tuple[str, ...]) -> float:
        vals = []
        for item in usable:
            cur = item
            for part in path:
                cur = cur[part]
            vals.append(float(cur))
        return sum(vals) / len(vals)

    def worst(path: tuple[str, ...]) -> float:
        vals = []
        for item in usable:
            cur = item
            for part in path:
                cur = cur[part]
            vals.append(float(cur))
        return max(vals)

    return {
        "modules": len(usable),
        "avg_output_nrmse": {
            "paro4g256_awq": avg(("output_vs_source", "paro4g256_awq", "nrmse")),
            "paro4g256_mq_body": avg(("output_vs_source", "paro4g256_mq_body", "nrmse")),
        },
        "worst_output_nrmse": {
            "paro4g256_awq": worst(("output_vs_source", "paro4g256_awq", "nrmse")),
            "paro4g256_mq_body": worst(("output_vs_source", "paro4g256_mq_body", "nrmse")),
        },
        "avg_weight_nrmse": {
            "paro4g256_awq": avg(("weight_reconstruction", "paro4g256_awq", "nrmse")),
            "paro4g256_mq_body": avg(("weight_reconstruction", "paro4g256_mq_body", "nrmse")),
        },
        "avg_payload_ratio_vs_source": {
            "paro4g256_awq": avg(("payload_ratio_vs_source", "paro4g256_awq")),
            "paro4g256_mq_body_plus_side": avg(("payload_ratio_vs_source", "paro4g256_mq_body_plus_side")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU probe for PARO4G256 and PARO4G256_MQ format loss.")
    parser.add_argument("--model", required=True, help="ParoQuant safetensors directory or Hugging Face repo id")
    parser.add_argument("--local-only", action="store_true", help="Do not download model files")
    parser.add_argument("--module", action="append", help="Specific module base name; may be repeated")
    parser.add_argument("--max-modules", type=int, default=3, help="Number of modules to probe when --module is absent")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.125)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pqi.require_deps()
    source_dir = pqi.resolve_model(args.model, local_only=args.local_only)
    index = pqi.build_tensor_index(source_dir)
    modules, incomplete = pqi.discover_paro_modules(index)
    if incomplete:
        raise ValueError(f"{len(incomplete)} Paro modules are incomplete")
    bases = args.module if args.module else [m["base"] for m in modules[: args.max_modules]]
    results = [run_module(index, base, samples=args.samples, seed=args.seed, input_scale=args.input_scale) for base in bases]
    payload = {
        "schema": "hipfire.astrea.paro_g256_probe.v0",
        "captured_at_utc": utc_now(),
        "source": str(source_dir),
        "model": args.model,
        "modules_available": len(modules),
        "modules_probed": len(results),
        "samples": args.samples,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "caveat": "Regroups/dequantizes existing G128 Paro weights; not a true G256 ParoQuant calibration run.",
        "summary": summarize(results),
        "results": results,
    }
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

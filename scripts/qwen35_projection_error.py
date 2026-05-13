#!/usr/bin/env python3
"""Fast PyTorch projection-error probe for Qwen3.5 HFQ candidates.

Uses a hipfire hidden dump as the activation source, then compares HFQ-dequantized
projection outputs against BF16 safetensors outputs for tensors whose inputs are
available from the layer boundary: linear_attn in_proj_{qkv,z,a,b} and
self_attn {q,k,v}_proj.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


def load_attr_module():
    spec = importlib.util.spec_from_file_location("hfq_attr", "scripts/qwen35_torch_hfq_attribution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def parse_layers(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return [x for x in available if x > 0]
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    have = set(available)
    bad = [x for x in out if x not in have or x == 0]
    if bad:
        raise ValueError(f"layers not available or unsupported: {bad}")
    return sorted(set(out))


def state_key_to_hfq_name(state_key: str) -> str:
    if state_key.startswith("model."):
        return "model.language_model." + state_key[len("model."):]
    return state_key


def hfq_name_to_state_key(hfq_name: str) -> str:
    if hfq_name.startswith("model.language_model."):
        return "model." + hfq_name[len("model.language_model."):]
    return hfq_name


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    bf = b.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    d = af - bf
    mse = float(np.mean(d * d))
    rmse = math.sqrt(mse)
    ref = math.sqrt(float(np.mean(bf * bf)))
    denom = math.sqrt(float(np.dot(af, af))) * math.sqrt(float(np.dot(bf, bf)))
    return {
        "cosine": float(np.dot(af, bf) / denom) if denom else float("nan"),
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": float(rmse / ref) if ref else float("nan"),
        "mae": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
    }


def read_hf_tensor(hf_dir: Path, state_key: str) -> torch.Tensor:
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        filename = weight_map[state_key]
    else:
        filename = "model.safetensors"
    with safe_open(str(hf_dir / filename), framework="pt", device="cpu") as f:
        return f.get_tensor(state_key)


def hfq_tensor_np(hfq: Any, name: str, attr: Any) -> np.ndarray:
    info, data = hfq.tensor_bytes(name)
    return attr.dequant_tensor(info, data)


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, offset: bool = True) -> torch.Tensor:
    scale = weight + 1.0 if offset else weight
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * scale


def candidate_projection_names(hfq: Any, layer: int) -> list[str]:
    base = f"model.language_model.layers.{layer}"
    la = [
        f"{base}.linear_attn.in_proj_qkv.weight",
        f"{base}.linear_attn.in_proj_z.weight",
        f"{base}.linear_attn.in_proj_a.weight",
        f"{base}.linear_attn.in_proj_b.weight",
    ]
    fa = [
        f"{base}.self_attn.q_proj.weight",
        f"{base}.self_attn.k_proj.weight",
        f"{base}.self_attn.v_proj.weight",
    ]
    return [name for name in la + fa if name in hfq.tensors]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-meta", required=True)
    ap.add_argument("--hf-model", required=True, help="Local HF safetensors snapshot")
    ap.add_argument("--hfq-model", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--max-print", type=int, default=40)
    args = ap.parse_args()

    attr = load_attr_module()
    hfq = attr.Hfq(args.hfq_model)
    hf_dir = Path(args.hf_model)
    meta = json.loads(Path(args.hidden_meta).read_text())
    dump_layers = [int(x) for x in meta["layers"]]
    layers = parse_layers(args.layers, dump_layers)
    dim = int(meta["dim"])
    seq_len = int(meta["seq_len"])
    hidden = np.memmap(meta["hidden_path"], dtype="<f4", mode="r", shape=(len(dump_layers), seq_len, dim))

    rows = []
    skipped = []
    for layer in layers:
        prev = layer - 1
        if prev not in dump_layers:
            skipped.append({"layer": layer, "reason": "previous_hidden_missing"})
            continue
        x_np = np.asarray(hidden[dump_layers.index(prev)]).astype(np.float32)
        x = torch.from_numpy(x_np).to(args.device, dtype=torch.float32)
        norm_name = f"model.language_model.layers.{layer}.input_layernorm.weight"
        if norm_name in hfq.tensors:
            norm_np = hfq_tensor_np(hfq, norm_name, attr)
        else:
            norm_np = read_hf_tensor(hf_dir, norm_name).float().numpy()
        norm_w = torch.from_numpy(norm_np).to(args.device, dtype=torch.float32)
        x_norm = rmsnorm_torch(x, norm_w, offset=True)
        del x, norm_w

        for hfq_name in candidate_projection_names(hfq, layer):
            state_key = hfq_name_to_state_key(hfq_name)
            try:
                q_np = hfq_tensor_np(hfq, hfq_name, attr)
                q_w = torch.from_numpy(q_np).to(args.device, dtype=torch.float32)
                bf_w = read_hf_tensor(hf_dir, hfq_name).to(args.device, dtype=torch.float32)
                with torch.inference_mode():
                    q_y = x_norm @ q_w.t()
                    bf_y = x_norm @ bf_w.t()
                info = hfq.tensors[hfq_name]
                rows.append({
                    "layer": layer,
                    "state_key": state_key,
                    "hfq_name": hfq_name,
                    "quant_type": int(info["quant_type"]),
                    "shape": list(info["shape"]),
                    "all": metrics(q_y, bf_y),
                    "last": metrics(q_y[-1], bf_y[-1]),
                })
                del q_np, q_w, bf_w, q_y, bf_y
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as e:
                skipped.append({"layer": layer, "state_key": state_key, "reason": "exception", "error": str(e)})
        del x_norm
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    rows_sorted = sorted(rows, key=lambda r: r["last"]["rel_rmse"], reverse=True)
    result = {
        "schema": "hipfire.qwen35.projection_error.v0",
        "hidden_meta": args.hidden_meta,
        "hf_model": args.hf_model,
        "hfq_model": args.hfq_model,
        "seq_len": seq_len,
        "layers": layers,
        "rows": rows_sorted,
        "skipped": skipped,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))

    print("rank layer quant rel_last rel_all  max_last tensor")
    print("---- ----- ----- -------- ------- -------- ----------------")
    for i, row in enumerate(rows_sorted[: args.max_print], 1):
        print(f"{i:4d} {row['layer']:5d} {row['quant_type']:5d} {row['last']['rel_rmse']:8.5f} {row['all']['rel_rmse']:7.5f} {row['last']['max_abs']:8.5f} {row['state_key']}")
    if skipped:
        print(f"skipped={len(skipped)}")


if __name__ == "__main__":
    main()

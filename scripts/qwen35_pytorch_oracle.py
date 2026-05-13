#!/usr/bin/env python3
"""Compare a hipfire Qwen3.5 hidden dump against a local PyTorch/HF oracle."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hipfire-meta", required=True)
    p.add_argument("--hf-model", required=True, help="HF repo id or local path, e.g. Qwen/Qwen3.5-9B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--out", help="Optional JSON metrics output path")
    p.add_argument("--max-print", type=int, default=40)
    p.add_argument("--allow-remote", action="store_true", help="Allow HF to fetch missing files instead of local_files_only")
    return p.parse_args()


def dtype_from_name(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64, copy=False).reshape(-1)
    bf = b.astype(np.float64, copy=False).reshape(-1)
    denom = math.sqrt(float(np.dot(af, af))) * math.sqrt(float(np.dot(bf, bf)))
    return float(np.dot(af, bf) / denom) if denom else float("nan")


def metrics_for(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    ref_norm = math.sqrt(float(np.mean(b.astype(np.float64, copy=False) ** 2)))
    return {
        "cosine": cosine(a, b),
        "mse": mse,
        "rmse": rmse,
        "rel_rmse": float(rmse / ref_norm) if ref_norm else float("nan"),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def tensor_from_hook_output(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def main() -> None:
    args = parse_args()
    meta_path = Path(args.hipfire_meta)
    meta = json.loads(meta_path.read_text())
    layers = [int(x) for x in meta["layers"]]
    tokens = [int(x) for x in meta["tokens"]]
    seq_len = int(meta["seq_len"])
    dim = int(meta["dim"])
    hidden_path = Path(meta["hidden_path"])
    logits_path = Path(meta["logits_path"])

    hip = np.memmap(hidden_path, dtype="<f4", mode="r", shape=(len(layers), seq_len, dim))
    hip_logits = np.memmap(logits_path, dtype="<f4", mode="r", shape=(int(meta["vocab_size"]),))
    final_norm_last_path = meta.get("final_norm_last_path")
    hip_final_norm_last = None
    if final_norm_last_path:
        hip_final_norm_last = np.memmap(Path(final_norm_last_path), dtype="<f4", mode="r", shape=(dim,))

    torch_dtype = dtype_from_name(args.dtype)
    print(f"loading oracle: {args.hf_model} dtype={args.dtype} device={args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=not args.allow_remote,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(args.device)

    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers") or not hasattr(text_model, "norm"):
        raise SystemExit("oracle model does not expose model.layers/model.norm hooks")
    if max(layers) >= len(text_model.layers):
        raise SystemExit(f"oracle has {len(text_model.layers)} layers, cannot compare layer {max(layers)}")

    layer_refs: dict[int, np.ndarray] = {}
    final_norm_ref: dict[str, np.ndarray] = {}
    handles = []

    def make_layer_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = tensor_from_hook_output(output)
            layer_refs[layer_idx] = tensor[0].detach().float().cpu().numpy()
        return hook

    for layer_idx in sorted(set(layers)):
        handles.append(text_model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx)))

    def norm_hook(_module, _inputs, output):
        tensor = tensor_from_hook_output(output)
        final_norm_ref["last"] = tensor[0, -1].detach().float().cpu().numpy()

    handles.append(text_model.norm.register_forward_hook(norm_hook))

    input_ids = torch.tensor([tokens], dtype=torch.long, device=args.device)
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [layer_idx for layer_idx in layers if layer_idx not in layer_refs]
    if missing:
        raise SystemExit(f"missing hook captures for layers: {missing}")
    if "last" not in final_norm_ref:
        raise SystemExit("missing final norm hook capture")

    ref_logits = out.logits[0, -1].detach().float().cpu().numpy()
    layer_types = list(getattr(model.config, "layer_types", []) or [])
    rows = []
    prev_rel = None
    for dump_idx, layer_idx in enumerate(layers):
        ref = layer_refs[layer_idx]
        if ref.shape != (seq_len, dim):
            raise SystemExit(f"shape mismatch layer {layer_idx}: ref {ref.shape} hip {(seq_len, dim)}")
        all_m = metrics_for(np.asarray(hip[dump_idx]), ref)
        last_m = metrics_for(np.asarray(hip[dump_idx, -1]), ref[-1])
        jump = None if prev_rel in (None, 0.0) else last_m["rel_rmse"] / prev_rel
        prev_rel = last_m["rel_rmse"]
        rows.append({
            "layer": layer_idx,
            "layer_type": layer_types[layer_idx] if layer_idx < len(layer_types) else None,
            "all": all_m,
            "last": last_m,
            "last_rel_rmse_jump": jump,
            "ref_source": "decoder_layer_forward_hook_post_block_pre_final_norm",
        })

    logits_m = metrics_for(np.asarray(hip_logits), ref_logits)
    final_norm_m = None
    boundary_pre_norm_vs_hf_final_norm = None
    if hip_final_norm_last is not None:
        final_norm_m = metrics_for(np.asarray(hip_final_norm_last), final_norm_ref["last"])
    if (int(meta.get("n_layers", -1)) - 1) in layers:
        dump_idx = layers.index(int(meta["n_layers"]) - 1)
        boundary_pre_norm_vs_hf_final_norm = metrics_for(np.asarray(hip[dump_idx, -1]), final_norm_ref["last"])

    result = {
        "schema": "hipfire.qwen35.pytorch_oracle.v1",
        "hf_model": args.hf_model,
        "hipfire_meta": str(meta_path),
        "tokens": tokens,
        "seq_len": seq_len,
        "dim": dim,
        "kv_mode": meta.get("kv_mode"),
        "dn_state_quant": meta.get("dn_state_quant"),
        "hidden_ref_source": "decoder_layer_forward_hooks_post_block_pre_final_norm",
        "layers": rows,
        "final_norm_last": final_norm_m,
        "boundary_pre_norm_vs_hf_final_norm": boundary_pre_norm_vs_hf_final_norm,
        "logits_last": logits_m,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))

    print("layer type              cos_last  rel_rmse_last  jump   max_abs_last  cos_all   rel_rmse_all")
    print("----- ----------------- --------- -------------- ------ ------------ -------- -------------")
    for row in rows[: args.max_print]:
        lm = row["last"]
        am = row["all"]
        jump = row["last_rel_rmse_jump"]
        print(
            f"{row['layer']:5d} {(row['layer_type'] or '')[:17]:17s} "
            f"{lm['cosine']:9.6f} {lm['rel_rmse']:14.6f} "
            f"{jump if jump is not None else float('nan'):6.2f} "
            f"{lm['max_abs']:12.6f} {am['cosine']:8.6f} {am['rel_rmse']:13.6f}"
        )
    if final_norm_m is not None:
        print("final_norm_last", json.dumps(final_norm_m, sort_keys=True))
    if boundary_pre_norm_vs_hf_final_norm is not None:
        print("boundary_pre_norm_vs_hf_final_norm", json.dumps(boundary_pre_norm_vs_hf_final_norm, sort_keys=True))
    print("logits_last", json.dumps(logits_m, sort_keys=True))


if __name__ == "__main__":
    main()

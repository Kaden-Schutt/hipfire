#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Dump residual CONTENT (not just L2) at selected layers/positions.

Sibling of residual_pos_traj.py. Streams one Block at a time (~4.7 GiB peak),
captures HC residual slices as f32 for cosine comparison against the parent.

Dumps:
  - layers: L0, L2, L10, L20, L30, L38, L42 (plus embed = layer input -1)
  - positions: 0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023
  - shape per capture: [n_pos, hc_mult=4, dim=4096] f32
  - also full-row norms for cross-check with residual_pos_traj

Output: NPZ + JSON sidecar under --out-dir.
"""
from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kernel_shim as _ks

sys.modules["kernel"] = _ks
import torch
import numpy as np
from model import Block, ModelArgs, Transformer
from weight_loader import _load_tensor, build_tensor_index, dequant_fp8_block128

DEFAULT_MODEL = "/mnt/scratch/models/DeepSeek-V4-Flash-0731"
DEFAULT_TOKENS = (
    "/mnt/scratch/quantization/deepseek-v4-flash-0731-parent-baseline/tokens.bin"
)
DEFAULT_LAYERS = [0, 2, 10, 20, 30, 38, 42]
DEFAULT_POSITIONS = [0, 1, 64, 200, 400, 448, 512, 600, 800, 1000, 1023]


def log(m=""):
    print(m, flush=True)


def load_tokens(path, n):
    raw = Path(path).read_bytes()
    ids = struct.unpack("<" + "i" * n, raw[: n * 4])
    return torch.tensor(ids, dtype=torch.long).view(1, n)


def make_args(config_path, max_seq_len):
    cfg = json.loads(Path(config_path).read_text())
    fields = ModelArgs.__dataclass_fields__
    kwargs = {}
    for k, v in cfg.items():
        if k not in fields:
            continue
        kwargs[k] = (
            tuple(v) if k in ("compress_ratios", "dspark_target_layer_ids") else v
        )
    a = ModelArgs(**kwargs)
    a.max_batch_size = 1
    a.max_seq_len = max_seq_len
    return a


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_embed_only(model, model_dir, device):
    index = build_tensor_index(Path(model_dir))
    w = _load_tensor(index, "embed.weight", device="cpu")
    with torch.no_grad():
        model.embed.weight.copy_(w.to(model.embed.weight.dtype))
    model.to(device)
    return ["embed.weight"]


def load_block_weights(block, model_dir, layer_id, device, verbose=True):
    index = build_tensor_index(Path(model_dir))
    prefix = f"layers.{layer_id}."
    sd = block.state_dict()
    assigned = {}
    loaded = []
    for key in sd.keys():
        src = prefix + key
        if src not in index:
            continue
        tensor = _load_tensor(index, src, device="cpu")
        if src.endswith("wo_a.weight") and tensor.dtype == torch.float8_e4m3fn:
            scale = _load_tensor(
                index, src.replace(".weight", ".scale"), device="cpu"
            )
            tensor = dequant_fp8_block128(tensor, scale).to(torch.bfloat16)
            if verbose:
                log(f"  dequant wo_a -> bf16 {tuple(tensor.shape)}")
        elif tensor.dtype == torch.int8 and ".ffn.experts." in src:
            tensor = tensor.view(torch.float4_e2m1fn_x2)
        elif tensor.dtype == torch.int64 and src.endswith("tid2eid"):
            tensor = tensor.to(torch.int32)
        dst = sd[key]
        if tuple(tensor.shape) != tuple(dst.shape):
            if not (
                tensor.dtype == torch.float4_e2m1fn_x2
                and tuple(tensor.shape) == tuple(dst.shape)
            ):
                raise RuntimeError(
                    f"shape mismatch {src}: ckpt {tuple(tensor.shape)} "
                    f"vs model {tuple(dst.shape)}"
                )
        if tensor.dtype != dst.dtype and dst.dtype != torch.float4_e2m1fn_x2:
            tensor = tensor.to(dst.dtype)
        assigned[key] = tensor
        loaded.append(src)
    incompat = block.load_state_dict(assigned, strict=False)
    if verbose:
        log(
            f"  L{layer_id} assigned {len(assigned)} tensors "
            f"missing={len(incompat.missing_keys)}"
        )

    for name, module in block.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        scale_key = f"{prefix}{name}.scale" if name else f"{prefix}scale"
        if scale_key not in index:
            continue
        if w.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
            continue
        scale = _load_tensor(index, scale_key, device="cpu")
        if scale.dtype != torch.float8_e8m0fnu and scale.element_size() == 1:
            scale = scale.view(torch.float8_e8m0fnu)
        w.scale = scale
        if hasattr(module, "scale"):
            try:
                module.scale = torch.nn.Parameter(scale, requires_grad=False)
            except Exception:
                module.scale = scale
        loaded.append(scale_key)

    block.to(device)
    for name, module in block.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        if hasattr(w, "scale") and torch.is_tensor(w.scale):
            w.scale = w.scale.to(device)
        if hasattr(module, "scale") and torch.is_tensor(module.scale):
            if isinstance(module.scale, torch.nn.Parameter):
                module.scale.data = module.scale.data.to(device)
            else:
                module.scale = module.scale.to(device)
    return loaded


def capture_positions(h, positions):
    """h: [1, S, hc, dim] -> f32 numpy [n_pos, hc, dim]"""
    x = h.detach().float()[0].cpu().numpy()  # [S, hc, dim]
    return np.stack([x[p] for p in positions], axis=0).astype(np.float32)


def row_l2_all(h):
    x = h.detach().float()[0].reshape(h.size(1), -1)
    return x.norm(dim=-1).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def run(args):
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    log(f"torch={torch.__version__} device={device}")
    if device.type == "cuda":
        log(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GiB"
        )

    model_dir = Path(args.model)
    cfgp = model_dir / "inference" / "config.json"
    if not cfgp.exists():
        cfgp = HERE / "config.json"
    margs = make_args(cfgp, max(args.seq, 256))
    n_total = margs.n_layers

    if args.layers.strip() == "all":
        layers_keep = list(range(n_total))
    else:
        layers_keep = [int(x) for x in args.layers.split(",")]
    positions = [int(x) for x in args.positions.split(",")]
    positions = [p for p in positions if 0 <= p < args.seq]
    last_needed = max(layers_keep) if layers_keep else -1
    log(f"seq={args.seq} layers={layers_keep} positions={positions}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_n = margs.n_layers
    margs.n_layers = 0
    margs.n_mtp_layers = 0
    margs.dspark_block_size = 0
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cpu"):
        shell = Transformer(margs)
    margs.n_layers = full_n
    load_embed_only(shell, model_dir, device)
    shell.eval()

    tokens = load_tokens(args.tokens, args.seq).to(device)
    h = shell.embed(tokens).unsqueeze(2).repeat(1, 1, shell.hc_mult, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    log(f"h0 shape={tuple(h.shape)} global_L2={float(h.float().norm()):.6f}")

    npz_arrays = {}
    meta = {
        "seq": args.seq,
        "device": str(device),
        "torch": torch.__version__,
        "layers": layers_keep,
        "positions": positions,
        "hc_mult": int(h.size(2)),
        "dim": int(h.size(3)),
        "dtype": "f32",
        "layout": "[n_pos, hc_mult, dim]",
        "tokens_path": args.tokens,
        "model": args.model,
        "notes": [
            "HC residual content after each captured layer (and embed as layer=-1)",
            "Arithmetic domain: block runs in bf16 weights/acts; dump cast to f32",
            "Streamed one Block at a time; model.py imported verbatim",
        ],
    }

    emb = capture_positions(h, positions)
    npz_arrays["layer_-1_embed"] = emb
    npz_arrays["layer_-1_embed_row_l2"] = row_l2_all(h)
    log(f"captured embed content shape={emb.shape}")

    del shell.head, shell.norm, shell.hc_head_fn, shell.hc_head_base, shell.hc_head_scale
    free_cuda()

    want = set(layers_keep)
    t_wall0 = time.time()
    for i in range(last_needed + 1):
        t0 = time.time()
        with torch.device("cpu"):
            block = Block(i, margs)
        load_block_weights(block, model_dir, i, device, verbose=True)
        block.eval()
        t_load = time.time() - t0

        t1 = time.time()
        if device.type == "cuda":
            torch.set_default_device("cuda")
        try:
            h = block(h, 0, tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            torch.set_default_device("cpu")
        t_fwd = time.time() - t1
        g = float(h.float().norm())
        ratio = (
            int(margs.compress_ratios[i]) if i < len(margs.compress_ratios) else -1
        )
        vram = (
            torch.cuda.memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        )
        log(
            f"L{i:02d} ratio={ratio} load={t_load:.1f}s fwd={t_fwd:.2f}s "
            f"global_L2={g:.4f} vram={vram:.2f}GiB"
        )

        if i in want:
            content = capture_positions(h, positions)
            npz_arrays[f"layer_{i}"] = content
            npz_arrays[f"layer_{i}_row_l2"] = row_l2_all(h)
            log(f"  captured L{i} content shape={content.shape}")

        del block
        free_cuda()

    meta["wall_s"] = time.time() - t_wall0
    meta["h_final_global_l2"] = float(h.float().norm())

    npz_path = out_dir / "residual_content_ref.npz"
    meta_path = out_dir / "residual_content_ref.json"
    np.savez_compressed(npz_path, **npz_arrays)
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"wrote {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB)")
    log(f"wrote {meta_path}")
    log(f"wall={meta['wall_s']:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument(
        "--layers",
        default=",".join(str(x) for x in DEFAULT_LAYERS),
        help="comma list or 'all'",
    )
    ap.add_argument(
        "--positions",
        default=",".join(str(x) for x in DEFAULT_POSITIONS),
    )
    ap.add_argument(
        "--out-dir",
        default=str(HERE / "artifacts" / "residual_content"),
    )
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
